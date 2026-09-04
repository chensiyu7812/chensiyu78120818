#!/usr/bin/env python3
"""Judge the complete PM-v1.5 action sweep with two model families.

The input sweep must carry the PM-v1.5 full-scope gate and the same attested
automated semantic review required here. The legacy PM-v2.2 compatibility-
pilot branch is intentionally rejected on this fast track.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.api import (
    chat_request_payload,
    make_client,
    request_payload_has_schema,
)
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key as make_physical_call_key,
    reported_prompt_token_error,
)
from metacom_pm.artifacts import create_artifact_attestation, require_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from metacom_pm.pm_v2_audit import audit_pilot_label_value_feasibility
from metacom_pm.pm_v2_judging import (
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    build_response_messages,
    build_risk_messages,
    composite_spec_from_config,
    composite_weights_hash,
    judge_one,
    labeling_settings_from_config,
    prompt_contract_hash,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
    validate_action_applicable_risk_signal,
    validate_judge_table,
)
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)
from metacom_pm.pm_v1_5_shortcut_audit import (
    require_step0_shortcut_audit_pass,
)
from metacom_pm.v1_5_automated_semantic_review import (
    require_automated_semantic_review_pass,
)
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.v1_5_actual_corpus_review import (
    require_actual_corpus_semantic_review_pass,
)
from metacom_pm.v1_5_judge_isolation import require_judge_role_isolation
from metacom_pm.internal_holdout import seal_internal_label_bundle
from metacom_pm.text import conservative_token_bound, estimate_tokens

ROOT = Path(__file__).resolve().parents[2]


def raw_key(row):
    return str(row["state_id"]), str(row["action_id"]), str(row["judge_family"])


def raw_row_succeeded(row) -> bool:
    return bool(row.get("schema_success")) and row.get("status") == "SUCCESS"


def preflight_development_judge_clients(
    endpoints: Mapping[str, Any], *, client_factory=make_client
) -> dict[str, Any]:
    """Resolve every family credential, then build every client before reserve."""

    normalized = {str(family): endpoint for family, endpoint in endpoints.items()}
    if len(normalized) != len(endpoints) or not normalized:
        raise RuntimeError("development judge endpoints have invalid family keys")
    for endpoint in normalized.values():
        endpoint.api_key
    clients: dict[str, Any] = {}
    try:
        for family, endpoint in normalized.items():
            clients[family] = client_factory(endpoint)
    except Exception:
        for client in clients.values():
            client.close()
        raise
    return clients


def persist_or_validate_judge_dry_run(
    *,
    ledger_path: str | Path,
    estimate_path: str | Path,
    call_plan_path: str | Path,
    cost_estimate: Mapping[str, Any],
    budget_gate: Mapping[str, Any],
    call_plan: list[dict[str, Any]],
) -> str:
    """Never mutate the accepted full plan after a physical attempt exists."""

    ledger_path = Path(ledger_path)
    estimate_path = Path(estimate_path)
    call_plan_path = Path(call_plan_path)
    expected_estimate = {**dict(cost_estimate), "budget_gate": dict(budget_gate)}
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not call_plan_path.is_file():
            raise RuntimeError(
                "spent development-judge ledger freezes the accepted dry-run, "
                "but its estimate or full call plan is missing"
            )
        if read_json(estimate_path) != expected_estimate or list(
            iter_jsonl(call_plan_path)
        ) != call_plan:
            raise RuntimeError(
                "spent development-judge ledger freezes the original exact "
                "estimate and full call plan; current drift is rejected"
            )
        return "VALIDATED_EXISTING"
    write_json(estimate_path, expected_estimate)
    write_jsonl(call_plan_path, call_plan)
    return "WRITTEN"


def require_exact_saved_judge_dry_run(
    *,
    estimate_path: str | Path,
    call_plan_path: str | Path,
    cost_estimate: Mapping[str, Any],
    budget_gate: Mapping[str, Any],
    call_plan: list[dict[str, Any]],
) -> None:
    estimate_path = Path(estimate_path)
    call_plan_path = Path(call_plan_path)
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("judge API run requires a matching saved --dry-run")
    expected_estimate = {**dict(cost_estimate), "budget_gate": dict(budget_gate)}
    if read_json(estimate_path) != expected_estimate:
        raise RuntimeError("saved PM-v2 judge dry-run exact estimate is stale")
    if list(iter_jsonl(call_plan_path)) != call_plan:
        raise RuntimeError("saved PM-v2 judge full call plan is stale")


def _attested_path(
    attestation: Mapping[str, Any], section: str, logical_name: str
) -> Path:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, Mapping) or not record.get("path"):
        raise RuntimeError(
            f"action sweep attestation lacks {section}.{logical_name}"
        )
    return Path(str(record["path"])).resolve()


def require_action_sweep_source_chain(
    *,
    outcomes_path: str | Path,
    summary_path: str | Path,
    manifest_path: str | Path,
    attestation_path: str | Path,
) -> dict[str, Any]:
    """Content-address every paid sweep artifact before judging it."""

    outcomes_path = Path(outcomes_path).resolve()
    summary_path = Path(summary_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="action_sweep",
        required_output_paths={
            "action_outcomes": outcomes_path,
            "summary": summary_path,
        },
    )
    attestation = read_json(attestation_path)
    if _attested_path(attestation, "inputs", "run_manifest") != manifest_path:
        raise RuntimeError("action sweep attestation manifest path mismatch")
    for name in ("raw_calls", "physical_attempt_ledger"):
        _attested_path(attestation, "outputs", name)
    manifest = read_json(manifest_path)
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("stage") != "action_sweep"
        or manifest.get("manifest_sha256")
        != sha256_text(canonical_json(manifest_payload))
    ):
        raise RuntimeError("action sweep manifest is stale or self-inconsistent")
    summary = read_json(summary_path)
    outcomes = list(iter_jsonl(outcomes_path))
    expected = int(summary.get("expected_outcomes") or -1)
    if (
        summary.get("status") != "COMPLETE"
        or int(summary.get("completed_outcomes") or -1) != expected
        or len(outcomes) != expected
        or len(
            {
                (str(row.get("card_id")), str(row.get("action_id")))
                for row in outcomes
            }
        )
        != expected
    ):
        raise RuntimeError("action sweep source matrix is not exact COMPLETE")
    manifest_bindings = manifest.get("contract_bindings") or {}
    if (
        summary.get("contract_bindings") != manifest_bindings
        or (attestation.get("parameters") or {}).get("contract_bindings")
        != manifest_bindings
        or int((attestation.get("expected") or {}).get("outcomes") or -1)
        != expected
    ):
        raise RuntimeError("action sweep contract bindings are inconsistent")
    return {
        "status": "PASS",
        "attestation_sha256": verification["attestation_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "contract_bindings": manifest_bindings,
        "expected_outcomes": expected,
    }


def require_full_sweep_development_binding(
    sweep_source_chain: Mapping[str, Any],
    development_pilot_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject full judging when the paid sweep bypassed the current gate."""

    observed = (sweep_source_chain.get("contract_bindings") or {}).get(
        "development_pilot_gate"
    )
    expected = dict(development_pilot_gate)
    if (
        observed != expected
        or str((observed or {}).get("binding_sha256") or "")
        != str(expected.get("binding_sha256") or "")
        or len(str(expected.get("binding_sha256") or "")) != 64
    ):
        raise RuntimeError(
            "full action sweep is not bound to the current complete "
            "development pilot/human gate"
        )
    return expected


def require_v1_5_full_sweep_binding(
    sweep_source_chain: Mapping[str, Any],
    *,
    automated_review_report_sha256: str,
    automated_review_attestation_sha256: str,
    actual_corpus_review_report_sha256: str,
    actual_corpus_review_attestation_sha256: str,
    step0_shortcut_audit_report_sha256: str,
    step0_shortcut_audit_attestation_sha256: str,
) -> dict[str, Any]:
    """Require an honestly full V1.5 matrix bound to the current review."""

    bindings = sweep_source_chain.get("contract_bindings") or {}
    observed = bindings.get("v1_5_full_sweep_gate") or {}
    expected = {
        "protocol": "pm-v1.5-full-sweep-gate-v2",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "automated_review_attestation_sha256": (
            automated_review_attestation_sha256
        ),
        "automated_review_report_sha256": automated_review_report_sha256,
        "actual_corpus_review_attestation_sha256": (
            actual_corpus_review_attestation_sha256
        ),
        "actual_corpus_review_report_sha256": actual_corpus_review_report_sha256,
        "step0_shortcut_audit_attestation_sha256": (
            step0_shortcut_audit_attestation_sha256
        ),
        "step0_shortcut_audit_report_sha256": (
            step0_shortcut_audit_report_sha256
        ),
    }
    if bindings.get("scope") != "full" or observed != expected:
        raise RuntimeError(
            "action sweep is not the exact PM-v1.5 full matrix bound to the "
            "current automated semantic review"
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl")
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--compatibility-pilot", action="store_true")
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "pilot_plan.json",
    )
    parser.add_argument("--sweep-manifest", type=Path)
    parser.add_argument("--sweep-summary", type=Path)
    parser.add_argument("--sweep-attestation", type=Path)
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--compatibility-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_development_pilot_judging"
        / "summary.json",
    )
    parser.add_argument(
        "--compatibility-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_development_pilot_judging"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_semantic_sanity"
            / "semantic_sanity_report.json"
        ),
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_semantic_sanity"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_automated_semantic_review" / "gate_report.json",
        help=(
            "PM-v1.5 replacement for the human semantic-sanity and pilot "
            "spot-check gates: output of "
            "scripts/v1_5_run_automated_semantic_review.py, must show "
            "status=PASS."
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_automated_semantic_review"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        help=(
            "Exact paid nine-case pilot bound by the automated semantic review; "
            "required for PM-v1.5 judging."
        ),
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-report",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v1_5_actual_corpus_semantic_review" / "gate_report.json"
        ),
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_actual_corpus_semantic_review"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--step0-shortcut-audit-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "step0_shortcut_audit.json"
        ),
    )
    parser.add_argument(
        "--step0-shortcut-audit-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--development-data-attestation",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--pilot-human-spot-check-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "pilot_human_spot_check_report.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument("--max-api-calls", type=int, default=50000)
    parser.add_argument("--max-estimated-usd", type=float, default=50.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid API runs prohibit --overwrite; use a new output directory"
        )
    if args.compatibility_pilot:
        raise RuntimeError(
            "PM-v1.5 does not run the PM-v2.2 compatibility-pilot judging "
            "branch; judge the complete V1.5 sweep instead"
        )

    compatibility_pilot = bool(args.compatibility_pilot)
    outcomes_path = args.outcomes or (
        ROOT
        / "outputs"
        / ("pm_v1_5_development_pilot" if compatibility_pilot else "pm_v1_5_sweep")
        / "action_outcomes.jsonl"
    )
    out_dir = args.out_dir or (
        ROOT
        / "outputs"
        / (
            "pm_v1_5_development_pilot_judging"
            if compatibility_pilot
            else "pm_v1_5_judging"
        )
    )
    sweep_manifest_path = args.sweep_manifest or outcomes_path.parent / "run_manifest.json"
    sweep_summary_path = args.sweep_summary or outcomes_path.parent / "summary.json"
    sweep_attestation_path = (
        args.sweep_attestation
        or outcomes_path.parent / "artifact_attestation.json"
    )
    sweep_source_chain = require_action_sweep_source_chain(
        outcomes_path=outcomes_path,
        summary_path=sweep_summary_path,
        manifest_path=sweep_manifest_path,
        attestation_path=sweep_attestation_path,
    )

    states = load_states(args.states)
    state_by_card = {state.card_id: state for state in states}
    evaluator_index = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    evaluator_by_state = evaluator_index.by_state
    outcomes = [ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)]
    unknown_cards = sorted({row.card_id for row in outcomes} - set(state_by_card))
    if unknown_cards:
        raise RuntimeError(f"outcomes contain unknown PM-v2 cards: {unknown_cards[:10]}")
    if len({(row.card_id, row.action_id) for row in outcomes}) != len(outcomes):
        raise RuntimeError("duplicate state-action outcomes")
    config = load_config(args.config)
    pm_v2_config = load_config(args.pm_v2_config)
    require_paid_run_release(
        pm_v2_config,
        config_path=args.pm_v2_config,
        stage="development_action_judging",
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    api_cost_planning = dict(pm_v2_config["api_cost_planning"])
    if set(api_cost_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning must contain exactly the frozen keys")
    input_token_safety_factor = float(
        api_cost_planning["input_token_safety_factor"]
    )
    if input_token_safety_factor < 1.0 or not bool(
        api_cost_planning["fail_on_reported_input_overrun"]
    ):
        raise ValueError("PM-v2 requires conservative fail-closed API cost planning")
    composite_spec = composite_spec_from_config(pm_v2_config)
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v2_config)
    judging_config = dict(pm_v2_config["development_judging"])
    pilot_config = dict(judging_config["compatibility_pilot"])
    endpoint_names = [str(value) for value in judging_config["judge_endpoints"]]
    judge_role_isolation = require_judge_role_isolation(
        config,
        pm_v2_config,
        development_endpoint_names=endpoint_names,
    )
    judge_seed = int(judging_config["seed"])
    response_max_output_tokens = int(
        judging_config["response_max_output_tokens"]
    )
    risk_max_output_tokens = int(judging_config["risk_max_output_tokens"])
    if response_max_output_tokens <= 0 or risk_max_output_tokens <= 0:
        raise ValueError("development judge max-output-token limits must be positive")
    endpoints = [endpoint_from_config(config, name) for name in endpoint_names]
    families = {endpoint.family for endpoint in endpoints}
    if (
        None in families
        or len(families) < labeling["minimum_families"]
        or len(families) != len(endpoints)
    ):
        raise ValueError(
            "PM-v2 requires at least two endpoints from distinct, declared judge families"
        )
    endpoint_descriptors = [
        {
            "name": name,
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
        }
        for name, endpoint in zip(endpoint_names, endpoints)
    ]
    pricing_by_family = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in dict(
            judging_config["pricing_usd_per_mtok"]
        ).items()
    }
    if set(pricing_by_family) != {str(endpoint.family) for endpoint in endpoints}:
        raise RuntimeError(
            "development judge pricing must exactly cover frozen endpoint families"
        )
    if any(
        set(values) != {"input", "output"}
        or any(value < 0.0 for value in values.values())
        for values in pricing_by_family.values()
    ):
        raise ValueError("development judge family pricing is invalid")
    pilot_plan_sha256 = None
    pilot_expected_keys_sha256 = None
    compatibility_attestation_sha256 = None
    development_pilot_gate = None
    v1_5_full_sweep_gate = None
    if compatibility_pilot:
        pilot_plan = read_json(args.pilot_plan)
        pilot_payload = {
            key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
        }
        if (
            pilot_plan.get("status") != "READY"
            or pilot_plan.get("pilot_plan_sha256")
            != sha256_text(canonical_json(pilot_payload))
        ):
            raise RuntimeError("PM-v2 compatibility pilot plan is invalid or stale")
        expected_plan_values = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "states_sha256": sha256_file(args.states),
            "evaluator_contexts_sha256": evaluator_index.source_sha256,
            "evaluator_contexts_map_sha256": evaluator_index.map_sha256,
            "sample_seed": int(pilot_config["sample_seed"]),
            "states_per_regime": int(pilot_config["states_per_regime"]),
            "actions": [str(value) for value in pilot_config["actions"]],
        }
        for key, expected in expected_plan_values.items():
            if pilot_plan.get(key) != expected:
                raise RuntimeError(f"compatibility pilot plan mismatch: {key}")
        expected_outcome_keys = sorted(
            [str(row["card_id"]), str(action)]
            for row in pilot_plan["selected_states"]
            for action in row["actions"]
        )
        actual_outcome_keys = sorted([row.card_id, row.action_id] for row in outcomes)
        if actual_outcome_keys != expected_outcome_keys:
            raise RuntimeError("compatibility pilot outcomes do not exactly match plan")
        if pilot_plan.get("expected_keys_sha256") != sha256_text(
            canonical_json(expected_outcome_keys)
        ):
            raise RuntimeError("compatibility pilot expected-key hash mismatch")
        sweep_bindings = sweep_source_chain["contract_bindings"]
        expected_sweep_bindings = {
            "scope": "compatibility_pilot",
            "pilot_plan_sha256": pilot_plan["pilot_plan_sha256"],
            "pilot_expected_keys_sha256": pilot_plan["expected_keys_sha256"],
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        }
        for key, expected in expected_sweep_bindings.items():
            if sweep_bindings.get(key) != expected:
                raise RuntimeError(f"compatibility sweep manifest mismatch: {key}")
        judge_schema_smoke = require_development_judge_schema_smoke_pass(
            summary_path=args.judge_schema_smoke_summary,
            attestation_path=args.judge_schema_smoke_attestation,
            experiment_config_path=args.config,
            pm_v2_config_path=args.pm_v2_config,
            states_path=args.states,
            backend_path=args.backend,
            evaluator_contexts_path=args.evaluator_contexts,
            pilot_plan_path=args.pilot_plan,
            semantic_sanity_report_path=args.semantic_sanity_report,
            semantic_sanity_attestation_path=args.semantic_sanity_attestation,
        )
        if (
            sweep_bindings.get("judge_schema_smoke") != judge_schema_smoke
            or sweep_bindings.get("deployable_feature_observability")
            != pilot_plan.get("deployable_feature_observability")
        ):
            raise RuntimeError(
                "compatibility action sweep bypassed the current schema-smoke "
                "or deployable-feature observability gate"
            )
        pilot_plan_sha256 = str(pilot_plan["pilot_plan_sha256"])
        pilot_expected_keys_sha256 = str(pilot_plan["expected_keys_sha256"])
    else:
        # V1.5's fast track does not run V2.2's 180-generation/360-judge
        # compatibility pilot. Full judging itself is fail-closed on the first
        # unsuccessful physical call and subsequently enforces the complete
        # two-family matrix/quality gates. The independent automated semantic
        # review remains a required, content-bound input here; external key
        # claims additionally require the frozen forced-swap sensitivity canary.
        automated_review_verification = require_automated_semantic_review_pass(
            args.automated_semantic_review_report,
            args.automated_semantic_review_attestation,
            expected_experiment_config_path=args.config,
            expected_pm_config_path=args.pm_v2_config,
            expected_strategy_bank_path=args.strategy_bank,
            expected_generation_pilot_attestation_path=(
                args.generation_pilot_attestation
            ),
        )
        automated_review_report = automated_review_verification["report"]
        actual_corpus_verification = require_actual_corpus_semantic_review_pass(
            args.actual_corpus_semantic_review_report,
            args.actual_corpus_semantic_review_attestation,
            expected_experiment_config_path=args.config,
            expected_states_path=args.states,
            expected_evaluator_contexts_path=args.evaluator_contexts,
            expected_backend_path=args.backend,
            expected_strategy_bank_path=args.strategy_bank,
            expected_pm_config_path=args.pm_v2_config,
        )
        shortcut_audit_verification = require_step0_shortcut_audit_pass(
            args.step0_shortcut_audit_report,
            args.step0_shortcut_audit_attestation,
            expected_states_path=args.states,
            expected_evaluator_contexts_path=args.evaluator_contexts,
            expected_pm_config_path=args.pm_v2_config,
            expected_generation_attestation_path=args.development_data_attestation,
        )
        semantic_sanity = {
            "protocol": "pm-v1.5-pilot-plus-actual-corpus-and-shortcut-gate-v2",
            "status": "PASS",
            "human_calibration_performed": False,
            "automated_review_report_sha256": sha256_text(
                canonical_json(automated_review_report)
            ),
            "automated_review_attestation_sha256": (
                automated_review_verification["attestation_sha256"]
            ),
            "actual_corpus_review_report_sha256": actual_corpus_verification[
                "report_sha256"
            ],
            "actual_corpus_review_attestation_sha256": actual_corpus_verification[
                "attestation_sha256"
            ],
            "step0_shortcut_audit_report_sha256": shortcut_audit_verification[
                "report_sha256"
            ],
            "step0_shortcut_audit_attestation_sha256": (
                shortcut_audit_verification["attestation_sha256"]
            ),
        }
        v1_5_full_sweep_gate = require_v1_5_full_sweep_binding(
            sweep_source_chain,
            automated_review_report_sha256=semantic_sanity[
                "automated_review_report_sha256"
            ],
            automated_review_attestation_sha256=semantic_sanity[
                "automated_review_attestation_sha256"
            ],
            actual_corpus_review_report_sha256=semantic_sanity[
                "actual_corpus_review_report_sha256"
            ],
            actual_corpus_review_attestation_sha256=semantic_sanity[
                "actual_corpus_review_attestation_sha256"
            ],
            step0_shortcut_audit_report_sha256=semantic_sanity[
                "step0_shortcut_audit_report_sha256"
            ],
            step0_shortcut_audit_attestation_sha256=semantic_sanity[
                "step0_shortcut_audit_attestation_sha256"
            ],
        )
        compatibility_attestation_sha256 = None
        runtime_state_lineage = require_pmv2_runtime_state_lineage(
            args.runtime, states
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "action_labels.jsonl"
    train_calibration_labels_path = (
        out_dir / "action_labels_train_calibration.jsonl"
    )
    internal_test_labels_path = out_dir / "action_labels_internal_test.jsonl"
    raw_path = out_dir / "judge_results.jsonl"
    ledger_path = out_dir / "judge_call_ledger.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    cost_estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    attestation_path = out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 development judging",
    )
    if args.overwrite:
        paths = (
            labels_path,
            train_calibration_labels_path,
            internal_test_labels_path,
            raw_path,
            out_dir / "summary.json",
            manifest_path,
            attestation_path,
        )
        if args.dry_run:
            paths = (*paths, cost_estimate_path, call_plan_path)
        for path in paths:
            if path.exists():
                path.unlink()
    stage = (
        "pm_v2_development_judge_compatibility"
        if compatibility_pilot
        else "pm_v2_action_judging"
    )
    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": stage,
            "experiment_config_sha256": sha256_file(args.config),
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "states_sha256": sha256_file(args.states),
            "outcomes_sha256": sha256_file(outcomes_path),
            "sweep_manifest_sha256": sha256_file(sweep_manifest_path),
            "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
            "evaluator_context_map_sha256": evaluator_index.map_sha256,
            "judge_endpoints": endpoint_descriptors,
            "judge_role_isolation": judge_role_isolation,
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling": labeling,
            "development_judging": judging_config,
            "seed": judge_seed,
            "response_max_output_tokens": response_max_output_tokens,
            "risk_max_output_tokens": risk_max_output_tokens,
            "scope": "compatibility_pilot" if compatibility_pilot else "full",
            "max_outcomes": None,
            "pilot_plan_sha256": pilot_plan_sha256,
            "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
            "compatibility_attestation_sha256": compatibility_attestation_sha256,
            "sweep_source_chain": sweep_source_chain,
            "development_pilot_gate": development_pilot_gate,
            "v1_5_full_sweep_gate": v1_5_full_sweep_gate,
            "judge_retries": 1,
            "pricing_usd_per_mtok": pricing_by_family,
            "api_cost_planning": api_cost_planning,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        },
    )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    raw_rows = list(iter_jsonl(raw_path)) if raw_path.exists() else []
    raw_by_key = {}
    for row in raw_rows:
        key = raw_key(row)
        if key in raw_by_key:
            raise RuntimeError(f"duplicate judge result key: {key}")
        endpoint = endpoint_by_family.get(key[2])
        if endpoint is None or str(row.get("judge_model")) != endpoint.model:
            raise RuntimeError(f"stale judge endpoint provenance: {key}")
        if row.get("prompt_contract_hash") != prompt_contract_hash():
            raise RuntimeError(f"stale judge prompt provenance: {key}")
        if raw_row_succeeded(row):
            ResponseJudgeOutput.model_validate(row.get("response"))
            RiskJudgeOutput.model_validate(row.get("risk"))
            if not row.get("response_request_hash") or not row.get(
                "risk_request_hash"
            ):
                raise RuntimeError(f"successful judge row lacks request hashes: {key}")
        elif not compatibility_pilot:
            raise RuntimeError(
                "full development judging cannot resume from a failed schema row; "
                "rerun the compatibility pilot or use --overwrite"
            )
        raw_by_key[key] = row
    required_keys = {
        (state_by_card[outcome.card_id].state_id, outcome.action_id, str(endpoint.family))
        for outcome in outcomes
        for endpoint in endpoints
    }
    unexpected = sorted(set(raw_by_key) - required_keys)
    if unexpected:
        raise RuntimeError(
            "existing raw judge file contains rows outside this frozen run: "
            + str(unexpected[:10])
        )
    missing_keys = required_keys - set(raw_by_key)
    remaining_calls = len(missing_keys) * 2
    full_calls = len(required_keys) * 2
    cost_rows = []
    execution_by_call_key = {}
    for outcome_index_for_plan, outcome in enumerate(outcomes):
        state = state_by_card[outcome.card_id]
        authorized_context = str(
            evaluator_by_state[state.state_id].get("authorized_user_context") or ""
        )
        if not authorized_context:
            raise RuntimeError(f"state {state.card_id} lacks authorized evaluator context")
        selected_context = "\n".join(
            [f"MEMORY[{item.source.value}]: {item.text}" for item in outcome.memory_view]
            + [
                f"STRATEGY[{card.strategy_label}]: {card.guidance_text}"
                for card in outcome.strategy_view
            ]
        )
        response_messages = build_response_messages(
            state=state,
            authorized_user_context=authorized_context,
            candidate_response=outcome.response,
        )
        risk_messages = build_risk_messages(
            state=state,
            authorized_user_context=authorized_context,
            selected_context=selected_context,
            candidate_response=outcome.response,
        )
        for endpoint_index, endpoint in enumerate(endpoints):
            base_call_seed = (
                judge_seed + outcome_index_for_plan * 10 + endpoint_index
            )
            for judge_type, messages, max_output_tokens, schema, call_seed in (
                (
                    "response",
                    response_messages,
                    response_max_output_tokens,
                    ResponseJudgeOutput,
                    base_call_seed,
                ),
                (
                    "risk",
                    risk_messages,
                    risk_max_output_tokens,
                    RiskJudgeOutput,
                    base_call_seed + 1,
                ),
            ):
                request_payload = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=0.0,
                    max_tokens=max_output_tokens,
                    seed=call_seed,
                    response_schema=schema,
                )
                if not request_payload_has_schema(request_payload):
                    raise RuntimeError(
                        "development judge request payload lacks structured schema"
                    )
                payload_text = canonical_json(request_payload)
                base_input_tokens_est = estimate_tokens(payload_text)
                input_tokens_est = conservative_token_bound(
                    payload_text, safety_factor=input_token_safety_factor
                )
                pricing = pricing_by_family[str(endpoint.family)]
                plan_row = {
                        "state_id": state.state_id,
                        "action_id": outcome.action_id,
                        "judge_family": endpoint.family,
                        "judge_model": endpoint.model,
                        "judge_type": judge_type,
                        "seed": call_seed,
                        "input_tokens_est": input_tokens_est,
                        "base_input_tokens_est": base_input_tokens_est,
                        "max_output_tokens": max_output_tokens,
                        "max_http_attempts": 1,
                        "prompt_hash": sha256_text(canonical_json(messages)),
                        "request_payload_sha256": sha256_text(
                            canonical_json(request_payload)
                        ),
                        "request_payload_includes_schema": (
                            request_payload_has_schema(request_payload)
                        ),
                        "pricing_usd_per_mtok": pricing,
                        "maximum_cost_usd": input_tokens_est / 1_000_000
                        * pricing["input"]
                        + max_output_tokens / 1_000_000 * pricing["output"],
                }
                plan_row["physical_call_key"] = make_physical_call_key(
                    stage=stage,
                    record_ids={
                        "state_id": state.state_id,
                        "action_id": outcome.action_id,
                        "judge_family": str(endpoint.family),
                        "judge_type": judge_type,
                    },
                    prompt_sha256=str(plan_row["prompt_hash"]),
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": 0.0,
                        "max_tokens": int(max_output_tokens),
                        "seed": int(call_seed),
                        "response_schema": schema.__name__,
                        "request_payload_sha256": plan_row[
                            "request_payload_sha256"
                        ],
                        "retries": 1,
                    },
                )
                cost_rows.append(plan_row)
                execution_by_call_key[
                    (
                        state.state_id,
                        outcome.action_id,
                        str(endpoint.family),
                        judge_type,
                    )
                ] = {
                    "messages": messages,
                    "schema": schema,
                    "card_id": state.card_id,
                }
    def call_key(row):
        return (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["judge_family"]),
            str(row["judge_type"]),
        )

    plan_by_call_key = {call_key(row): row for row in cost_rows}
    if len(plan_by_call_key) != len(cost_rows):
        raise RuntimeError("duplicate development judge call-plan key")
    plan_by_physical_key = {
        str(row["physical_call_key"]): row for row in cost_rows
    }
    if len(plan_by_physical_key) != len(cost_rows):
        raise RuntimeError("duplicate development judge physical-call key")
    attempt_ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={key: 1 for key in plan_by_physical_key},
        maximum_total_attempts=int(args.max_api_calls),
    )
    ledger_rows = attempt_ledger.event_rows
    successful_call_rows: dict[tuple[str, str, str, str], dict] = {}
    for ledger_row in ledger_rows:
        physical_key = str(ledger_row["call_key"])
        plan = plan_by_physical_key[physical_key]
        key = call_key(plan)
        expected_record_ids = {
            "state_id": key[0],
            "action_id": key[1],
            "judge_family": key[2],
            "judge_type": key[3],
        }
        if (
            ledger_row.get("record_ids") != expected_record_ids
            or ledger_row.get("prompt_sha256") != plan["prompt_hash"]
        ):
            raise RuntimeError(f"stale judge attempt-ledger provenance: {key}")
        if ledger_row.get("event") == "SUCCEEDED":
            if key in successful_call_rows:
                raise RuntimeError(f"duplicate successful judge HTTP call: {key}")
            result_payload = ledger_row.get("result") or {}
            schema = (
                ResponseJudgeOutput if key[3] == "response" else RiskJudgeOutput
            )
            parsed = schema.model_validate(result_payload.get("parsed"))
            if not ledger_row.get("request_hash"):
                raise RuntimeError(f"successful judge ledger row lacks request hash: {key}")
            usage_error = reported_prompt_token_error(
                ledger_row.get("usage"),
                maximum_prompt_tokens=int(plan["input_tokens_est"]),
                stage="persisted development judge",
                require_positive=True,
            )
            if usage_error is not None:
                raise RuntimeError(
                    f"successful development judge usage is invalid for {key}: "
                    f"{usage_error}"
                )
            successful_call_rows[key] = {
                **plan,
                "parsed": parsed.model_dump(mode="json"),
                "request_hash": str(ledger_row["request_hash"]),
            }
    pending_cost_rows = [
        row
        for row in cost_rows
        if not attempt_ledger.succeeded(str(row["physical_call_key"]))
        and not attempt_ledger.exhausted(str(row["physical_call_key"]))
    ]
    completed_pair_keys = {
        (state_id, action_id, family)
        for state_id, action_id, family in required_keys
        if (state_id, action_id, family, "response") in successful_call_rows
        and (state_id, action_id, family, "risk") in successful_call_rows
    }
    missing_keys = required_keys - completed_pair_keys
    remaining_calls = len(pending_cost_rows)
    historical_attempts = attempt_ledger.started_attempts
    # The accepted estimate is immutable and always describes the complete
    # pre-attempt matrix.  Mutable resume state belongs in summary diagnostics,
    # never in the approval hash or saved call plan.
    maximum_physical_attempts = len(cost_rows)
    input_counts = [int(row["input_tokens_est"]) for row in cost_rows]
    total_input_tokens = sum(input_counts)
    total_output_tokens = sum(
        int(row["max_output_tokens"]) for row in cost_rows
    )
    cost_payload = {
        "stage": stage,
        "full_logical_api_calls": len(cost_rows),
        "historical_physical_http_attempts": 0,
        "planned_new_api_calls": len(cost_rows),
        "maximum_physical_http_attempts": maximum_physical_attempts,
        "expected_judge_pairs": len(required_keys),
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output_tokens,
        "estimated_cost_usd": sum(
            float(row["maximum_cost_usd"]) for row in cost_rows
        ),
        "pricing_usd_per_mtok": pricing_by_family,
        "api_cost_planning": api_cost_planning,
        "judge_role_isolation": judge_role_isolation,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "call_plan_sha256": sha256_text(canonical_json(cost_rows)),
        "ledger_sha256": sha256_text(canonical_json([])),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "scope": "compatibility_pilot" if compatibility_pilot else "full",
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "budget_limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    budget_checks = {
        "api_calls": maximum_physical_attempts <= int(args.max_api_calls),
        "estimated_cost_usd": cost_estimate["estimated_cost_usd"]
        <= float(args.max_estimated_usd),
        "max_input_tokens_per_call": cost_estimate["max_input_tokens_per_call_est"]
        <= int(args.max_input_tokens_per_call),
    }
    budget_gate = {
        "status": "PASS" if all(budget_checks.values()) else "FAIL",
        "checks": budget_checks,
        "limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    summary = {
        "status": "DRY_RUN_COMPLETE" if args.dry_run else "STARTING",
        "outcomes": len(outcomes),
        "judge_endpoints": endpoint_names,
        "judge_families": sorted(str(value) for value in families),
        "full_expected_api_calls": full_calls,
        "completed_judge_pairs": len(completed_pair_keys),
        "remaining_judge_pairs": len(missing_keys),
        "remaining_api_calls": remaining_calls,
        "historical_physical_http_attempts": historical_attempts,
        "planned_new_api_calls": len(pending_cost_rows),
        "prompt_contract_hash": prompt_contract_hash(),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling_gates": labeling,
        "development_judging": judging_config,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "judge_endpoint_descriptors": endpoint_descriptors,
        "judge_role_isolation": judge_role_isolation,
        "scope": "compatibility_pilot" if compatibility_pilot else "full",
        "reportability_status": (
            "COMPATIBILITY_GATE_PENDING" if compatibility_pilot else "REPORTABLE"
        ),
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "resumable": True,
        "cost_estimate": cost_estimate,
        "budget_gate": budget_gate,
    }
    print(summary)
    if args.dry_run or budget_gate["status"] != "PASS":
        persist_or_validate_judge_dry_run(
            ledger_path=ledger_path,
            estimate_path=cost_estimate_path,
            call_plan_path=call_plan_path,
            cost_estimate=cost_estimate,
            budget_gate=budget_gate,
            call_plan=cost_rows,
        )
    if budget_gate["status"] != "PASS":
        raise RuntimeError("PM-v2 judge budget gate failed before API calls")
    if args.dry_run:
        return
    require_exact_saved_judge_dry_run(
        estimate_path=cost_estimate_path,
        call_plan_path=call_plan_path,
        cost_estimate=cost_estimate,
        budget_gate=budget_gate,
        call_plan=cost_rows,
    )
    if args.accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "judge API run requires exact --accept-cost-estimate-sha256 from dry-run"
        )

    allowed_schema_failures = math.floor(
        (1.0 - float(pilot_config["minimum_schema_success_rate"]))
        * len(required_keys)
        + 1e-12
    )
    exhausted_unsuccessful_calls = {
        call_key(plan)
        for plan in cost_rows
        if call_key(plan) not in successful_call_rows
        and attempt_ledger.exhausted(str(plan["physical_call_key"]))
    }
    failed_pair_keys = {key[:3] for key in exhausted_unsuccessful_calls}
    schema_failures_seen = len(failed_pair_keys)
    pilot_futility_reason = None
    if compatibility_pilot and schema_failures_seen > allowed_schema_failures:
        pilot_futility_reason = (
            "schema success threshold is mathematically unreachable from existing "
            "failed pilot rows"
        )
    if not compatibility_pilot and failed_pair_keys:
        raise RuntimeError(
            "full development judging contains a spent unsuccessful physical call; "
            "the one-attempt protocol forbids reissuing it"
        )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    clients = (
        preflight_development_judge_clients(endpoint_by_family)
        if pending_cost_rows and pilot_futility_reason is None
        else {}
    )
    try:
        for plan in pending_cost_rows:
            if pilot_futility_reason is not None:
                break
            key = call_key(plan)
            if key in successful_call_rows:
                continue
            execution = execution_by_call_key[key]
            endpoint = endpoint_by_family[key[2]]
            record_ids = {
                "state_id": key[0],
                "action_id": key[1],
                "judge_family": key[2],
                "judge_type": key[3],
            }
            reservation = attempt_ledger.reserve(
                str(plan["physical_call_key"]),
                record_ids=record_ids,
                prompt_sha256=str(plan["prompt_hash"]),
            )
            try:
                result, parsed = clients[key[2]].chat(
                    execution["messages"],
                    temperature=0.0,
                    max_tokens=int(plan["max_output_tokens"]),
                    seed=int(plan["seed"]),
                    response_schema=execution["schema"],
                    retries=1,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                attempt_ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=None,
                    usage=None,
                    error=error,
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(plan)),
                        "prompt_contract_hash": prompt_contract_hash(),
                    },
                )
            else:
                assert parsed is not None
                parsed_payload = parsed.model_dump(mode="json")
                usage_error = reported_prompt_token_error(
                    result.usage,
                    maximum_prompt_tokens=int(plan["input_tokens_est"]),
                    stage=f"development {key[3]} judge",
                    require_positive=bool(
                        api_cost_planning["fail_on_reported_input_overrun"]
                    ),
                )
                if usage_error is not None:
                    attempt_ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=usage_error,
                        result={"parsed": parsed_payload},
                        metadata={
                            "plan_sha256": sha256_text(canonical_json(plan)),
                            "prompt_contract_hash": prompt_contract_hash(),
                        },
                    )
                else:
                    attempt_ledger.finish(
                        reservation,
                        succeeded=True,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=None,
                        result={"parsed": parsed_payload},
                        metadata={
                            "plan_sha256": sha256_text(canonical_json(plan)),
                            "prompt_contract_hash": prompt_contract_hash(),
                        },
                    )
                    successful_call_rows[key] = {
                        **plan,
                        "parsed": parsed_payload,
                        "request_hash": result.request_hash,
                    }
            if attempt_ledger.succeeded(str(plan["physical_call_key"])):
                pass
            elif compatibility_pilot:
                failed_pair_keys.add(key[:3])
                schema_failures_seen = len(failed_pair_keys)
                if schema_failures_seen > allowed_schema_failures:
                    pilot_futility_reason = (
                        "schema success threshold became mathematically unreachable; "
                        "remaining compatibility-pilot API calls were not attempted"
                    )
            else:
                raise RuntimeError(
                    "development judge HTTP call failed after its ledger row was saved: "
                    + str(key)
                )
    finally:
        for client in clients.values():
            client.close()
    ledger_rows = attempt_ledger.event_rows

    raw_by_key = {}
    state_card = {state.state_id: state.card_id for state in states}
    for pair_key in sorted(required_keys):
        state_id, action_id, family = pair_key
        response_row = successful_call_rows.get(
            (state_id, action_id, family, "response")
        )
        risk_row = successful_call_rows.get((state_id, action_id, family, "risk"))
        if response_row is None or risk_row is None:
            continue
        pair_row = {
            "status": "SUCCESS",
            "schema_success": True,
            "state_id": state_id,
            "card_id": state_card[state_id],
            "action_id": action_id,
            "judge_family": family,
            "judge_model": response_row["judge_model"],
            "prompt_contract_hash": prompt_contract_hash(),
            "response": response_row["parsed"],
            "risk": risk_row["parsed"],
            "response_request_hash": response_row["request_hash"],
            "risk_request_hash": risk_row["request_hash"],
        }
        raw_by_key[pair_key] = pair_row
    write_jsonl(raw_path, [raw_by_key[key] for key in sorted(raw_by_key)])
    missing_after = sorted(required_keys - set(raw_by_key))
    if missing_after and pilot_futility_reason is None:
        report = {**summary, "status": "INCOMPLETE", "missing_keys": missing_after[:50]}
        write_json(out_dir / "summary.json", report)
        raise RuntimeError(f"judge run incomplete: {len(missing_after)} missing pairs")

    canonical_raw_rows = [
        {
            "judge_family": row["judge_family"],
            "action_id": row["action_id"],
            "response": row["response"],
            "risk": row["risk"],
        }
        for row in raw_by_key.values()
    ]
    raw_family_global_gate = validate_raw_judge_family_health(
        canonical_raw_rows,
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling[
            "composite_support_exact_match_rate"
        ],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        reject_constant_response_dimensions=labeling[
            "reject_constant_response_dimensions"
        ],
        reject_constant_risk_dimensions=labeling[
            "reject_constant_risk_dimensions"
        ],
        composite_spec=composite_spec,
        raise_on_failure=not compatibility_pilot,
    )
    raw_family_action_gate = validate_raw_judge_family_subgroup_health(
        canonical_raw_rows,
        subgroup_key="action_id",
        expected_subgroups=sorted({row.action_id for row in outcomes}),
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling[
            "composite_support_exact_match_rate"
        ],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        reject_constant_response_dimensions=labeling[
            "reject_constant_response_dimensions"
        ],
        reject_constant_risk_dimensions=labeling[
            "reject_constant_risk_dimensions"
        ],
        composite_spec=composite_spec,
        raise_on_failure=not compatibility_pilot,
    )
    action_applicable_risk_gate = validate_action_applicable_risk_signal(
        canonical_raw_rows,
        expected_actions=sorted({row.action_id for row in outcomes}),
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        minimum_signal_rate=labeling[
            "minimum_action_applicable_risk_signal_rate"
        ],
        minimum_distinct_values=labeling[
            "minimum_action_applicable_risk_distinct_values"
        ],
        raise_on_failure=not compatibility_pilot,
    )
    raw_family_quality_gate = {
        "status": (
            "PASS"
            if raw_family_global_gate.get("status") == "PASS"
            and raw_family_action_gate.get("status") == "PASS"
            and (
                compatibility_pilot
                or action_applicable_risk_gate.get("status") == "PASS"
            )
            else "FAIL"
        ),
        "global": raw_family_global_gate,
        "family_by_action": raw_family_action_gate,
        "action_applicable_risk_signal": {
            **action_applicable_risk_gate,
            "enforced": not compatibility_pilot,
        },
    }

    labels = []
    labels_path.write_text("", encoding="utf-8")
    train_calibration_labels_path.write_text("", encoding="utf-8")
    internal_test_labels_path.write_text("", encoding="utf-8")
    prompt_equivalence_class_sizes: dict[tuple[str, str], int] = {}
    for outcome in outcomes:
        state = state_by_card[outcome.card_id]
        key = (state.state_id, str(outcome.prompt_equivalence_id))
        prompt_equivalence_class_sizes[key] = (
            prompt_equivalence_class_sizes.get(key, 0) + 1
        )
    for outcome in outcomes:
        state = state_by_card[outcome.card_id]
        results = []
        for endpoint in endpoints:
            row = raw_by_key.get(
                (state.state_id, outcome.action_id, str(endpoint.family))
            )
            if row is None:
                results = []
                break
            if not raw_row_succeeded(row):
                results = []
                break
            if str(row["judge_model"]) != endpoint.model:
                raise RuntimeError(
                    f"judge model changed for {state.state_id}/{outcome.action_id}/"
                    f"{endpoint.family}: {row['judge_model']} != {endpoint.model}"
                )
            results.append(
                JudgeResult(
                    family=str(row["judge_family"]),
                    model=str(row["judge_model"]),
                    response=ResponseJudgeOutput.model_validate(row["response"]),
                    risk=RiskJudgeOutput.model_validate(row["risk"]),
                    response_request_hash=str(row["response_request_hash"]),
                    risk_request_hash=str(row["risk_request_hash"]),
                )
            )
        if not results:
            continue
        label = build_action_label(
            state=state,
            action_id=outcome.action_id,
            observed_input_tokens=outcome.cost.total_input_tokens,
            retrieval_calls=outcome.cost.retrieval_calls,
            results=results,
            composite_spec=composite_spec,
            minimum_families=labeling["minimum_families"],
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            provenance={
                "outcome_request_hash": outcome.request_hash,
                "outcome_prompt_hash": outcome.prompt_hash,
                "requested_action_id": outcome.requested_action_id,
                "realized_action_id": outcome.realized_action_id,
                "prompt_equivalence_id": outcome.prompt_equivalence_id,
                "label_lineage_id": outcome.label_lineage_id,
                "prompt_equivalence_class_size": (
                    prompt_equivalence_class_sizes[
                        (state.state_id, str(outcome.prompt_equivalence_id))
                    ]
                ),
            },
        )
        labels.append(label)
        append_jsonl(labels_path, label.model_dump(mode="json"))
        split_path = (
            internal_test_labels_path
            if state.split.value == "internal_test"
            else train_calibration_labels_path
        )
        append_jsonl(split_path, label.model_dump(mode="json"))
    pilot_reliable_threshold = float(pilot_config["minimum_reliable_label_rate"])
    if labels:
        quality_gate = validate_judge_table(
            labels,
            minimum_families=labeling["minimum_families"],
            minimum_reliable_rate=(
                pilot_reliable_threshold
                if compatibility_pilot
                else labeling["minimum_reliable_rate"]
            ),
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            minimum_low_mad_coverage_per_dimension=(
                float(pilot_config["minimum_low_mad_coverage_per_dimension"])
                if compatibility_pilot
                else labeling["minimum_low_mad_coverage_per_dimension"]
            ),
            minimum_low_mad_coverage_per_action_dimension=(
                float(
                    pilot_config[
                        "minimum_low_mad_coverage_per_action_dimension"
                    ]
                )
                if compatibility_pilot
                else labeling["minimum_low_mad_coverage_per_action_dimension"]
            ),
            duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
            maximum_absolute_dimension_correlation=labeling[
                "maximum_absolute_dimension_correlation"
            ],
            composite_support_exact_match_rate=labeling[
                "composite_support_exact_match_rate"
            ],
            maximum_absolute_composite_support_correlation=labeling[
                "maximum_absolute_composite_support_correlation"
            ],
            reject_constant_response_dimensions=labeling[
                "reject_constant_response_dimensions"
            ],
            reject_constant_risk_dimensions=labeling[
                "reject_constant_risk_dimensions"
            ],
            composite_spec=composite_spec,
            raise_on_failure=not compatibility_pilot,
        )
    else:
        quality_gate = {
            "status": "FAIL",
            "n": 0,
            "reason": "no complete two-family label rows",
        }
    successful_pairs = sum(raw_row_succeeded(row) for row in raw_by_key.values())
    schema_success_rate = successful_pairs / len(required_keys) if required_keys else 0.0
    reliable_rate = (
        sum(label.label_reliable for label in labels) / len(labels) if labels else 0.0
    )
    compatibility_thresholds = {
        "minimum_schema_success_rate": float(
            pilot_config["minimum_schema_success_rate"]
        ),
        "minimum_reliable_label_rate": pilot_reliable_threshold,
        "minimum_low_mad_coverage_per_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_action_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_action_dimension"]
        ),
    }
    compatibility_checks = {
        "exact_raw_matrix": set(raw_by_key) == required_keys,
        "schema_success_rate": schema_success_rate
        >= compatibility_thresholds["minimum_schema_success_rate"],
        "complete_two_family_labels": len(labels) == len(outcomes),
        "dimension_quality_gate": quality_gate.get("status") == "PASS",
        "raw_family_dimension_quality_gate": (
            raw_family_quality_gate.get("status") == "PASS"
        ),
    }
    compatibility_gate = {
        "status": "PASS" if all(compatibility_checks.values()) else "NONREPORTABLE",
        "checks": compatibility_checks,
        "thresholds": compatibility_thresholds,
        "schema_success_rate": schema_success_rate,
        "reliable_label_rate": reliable_rate,
        "joint_reliable_rate_is_diagnostic_only": True,
        "futility_triggered": pilot_futility_reason is not None,
        "futility_reason": pilot_futility_reason,
        "allowed_schema_failures": allowed_schema_failures,
    }
    label_value_feasibility = None
    if compatibility_pilot:
        selected_state_by_id = {
            state_by_card[outcome.card_id].state_id: state_by_card[outcome.card_id]
            for outcome in outcomes
        }
        label_value_feasibility = audit_pilot_label_value_feasibility(
            [selected_state_by_id[key] for key in sorted(selected_state_by_id)],
            labels,
            evaluator_contexts=evaluator_index,
            composite_spec=composite_spec,
            pilot_actions=[str(value) for value in pilot_config["actions"]],
            thresholds=dict(pilot_config["label_value_feasibility"]),
            risk_weight=float(pm_v2_config["selection"]["risk_weight"]),
            cost_weight=float(pm_v2_config["selection"]["cost_weight"]),
        )
        compatibility_gate["checks"]["label_value_feasibility"] = (
            label_value_feasibility["status"] == "PASS"
        )
        compatibility_gate["status"] = (
            "PASS"
            if all(compatibility_gate["checks"].values())
            else "NONREPORTABLE"
        )
    final_status = (
        compatibility_gate["status"] if compatibility_pilot else "COMPLETE"
    )
    final_missing_api_calls = sum(
        call_key(row) not in successful_call_rows for row in cost_rows
    )
    sealed_internal_bundle_path = out_dir / "sealed_internal_bundle_manifest.json"
    sealed_internal_bundle = seal_internal_label_bundle(
        sealed_internal_bundle_path,
        internal_labels_path=internal_test_labels_path,
    )
    report = {
        **summary,
        "status": final_status,
        "reportability_status": (
            "COMPATIBILITY_GATE_ONLY" if compatibility_pilot else "REPORTABLE"
        ),
        "completed_judge_pairs": len(raw_by_key),
        "remaining_judge_pairs": len(required_keys - set(raw_by_key)),
        "remaining_api_calls": final_missing_api_calls,
        "label_rows": len(labels),
        "reliable_rows": sum(label.label_reliable for label in labels),
        "schema_success_pairs": successful_pairs,
        "schema_success_rate": schema_success_rate,
        "quality_gate": quality_gate,
        "raw_family_quality_gate": raw_family_quality_gate,
        "compatibility_gate": compatibility_gate if compatibility_pilot else None,
        "label_value_feasibility": label_value_feasibility,
        "labels_path": str(labels_path),
        "train_calibration_labels_path": str(train_calibration_labels_path),
        "internal_test_labels_path": str(internal_test_labels_path),
        "sealed_internal_bundle_path": str(sealed_internal_bundle_path),
        "sealed_internal_bundle": sealed_internal_bundle,
        "raw_path": str(raw_path),
        "ledger_path": str(ledger_path),
        "physical_http_attempts": attempt_ledger.started_attempts,
        "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, report)
    attestation_inputs = {
        "experiment_config": args.config,
        "pm_v2_config": args.pm_v2_config,
        "states": args.states,
        "outcomes": outcomes_path,
        "evaluator_contexts": args.evaluator_contexts,
        "sweep_manifest": sweep_manifest_path,
        "sweep_summary": sweep_summary_path,
        "sweep_attestation": sweep_attestation_path,
        "cost_estimate": cost_estimate_path,
        "call_plan": call_plan_path,
    }
    if compatibility_pilot:
        attestation_inputs["pilot_plan"] = args.pilot_plan
    else:
        attestation_inputs["automated_semantic_review"] = (
            args.automated_semantic_review_report
        )
        attestation_inputs["automated_semantic_review_attestation"] = (
            args.automated_semantic_review_attestation
        )
        attestation_inputs["actual_corpus_semantic_review"] = (
            args.actual_corpus_semantic_review_report
        )
        attestation_inputs["actual_corpus_semantic_review_attestation"] = (
            args.actual_corpus_semantic_review_attestation
        )
    create_artifact_attestation(
        attestation_path,
        stage=stage,
        inputs=attestation_inputs,
        outputs={
            "summary": (summary_path, False),
            "labels": (labels_path, True),
            "sealed_internal_bundle": (sealed_internal_bundle_path, False),
            "train_calibration_labels": (train_calibration_labels_path, True),
            "internal_test_labels": (internal_test_labels_path, True),
            "raw_results": (raw_path, True),
            "call_ledger": (ledger_path, True),
        },
        parameters={
            "status": final_status,
            "scope": "compatibility_pilot" if compatibility_pilot else "full",
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling_gates": labeling,
            "development_judging": judging_config,
            "judge_endpoint_descriptors": endpoint_descriptors,
            "judge_role_isolation": judge_role_isolation,
            "pilot_plan_sha256": pilot_plan_sha256,
            "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
            "compatibility_attestation_sha256": compatibility_attestation_sha256,
            "compatibility_gate": (
                compatibility_gate if compatibility_pilot else None
            ),
            "label_value_feasibility": label_value_feasibility,
            "raw_family_quality_gate": raw_family_quality_gate,
            "accepted_cost_estimate_sha256": cost_estimate[
                "cost_estimate_sha256"
            ],
            "judge_retries": 1,
            "api_cost_planning": api_cost_planning,
            "pricing_usd_per_mtok": pricing_by_family,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "physical_http_attempts": attempt_ledger.started_attempts,
            "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        },
        expected={
            "outcomes": len(outcomes),
            "judge_role_isolation_status": judge_role_isolation["status"],
            "judge_pairs": len(required_keys),
            "full_logical_api_calls": len(cost_rows),
            "new_physical_http_attempts": (
                attempt_ledger.started_attempts - historical_attempts
            ),
            "physical_http_attempts": attempt_ledger.started_attempts,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
