#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.attempt_ledger import forbid_overwrite_of_spent_attempts
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import parse_action_id
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_audit import audit_deployable_feature_observability
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)


ROOT = Path(__file__).resolve().parents[1]


def _budget_gate(
    estimate: dict,
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict:
    checks = {
        "physical_api_attempts": int(estimate["maximum_physical_api_attempts"])
        <= int(max_api_calls),
        "estimated_cost_usd": float(estimate["estimated_cost_usd"])
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": int(estimate["max_input_tokens"])
        <= int(max_input_tokens_per_call),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_physical_api_attempts": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _persist_or_validate_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> str:
    """Freeze the exact accepted estimate/plan once an HTTP attempt exists."""

    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError(
                "spent action-sweep ledger freezes the accepted dry-run, but its "
                "estimate or full call plan is missing"
            )
        if read_json(estimate_path) != current or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError(
                "spent action-sweep ledger freezes the original exact estimate "
                "and full call plan; current dry-run drift is rejected"
            )
        return "VALIDATED_EXISTING"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, current)
    write_jsonl(plan_path, rows)
    return "WRITTEN"


def _require_saved_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> None:
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same output "
            "directory"
        )
    saved = read_json(estimate_path)
    if saved != current:
        raise RuntimeError(
            "saved action-sweep dry-run does not match current inputs/configuration; "
            "run --dry-run again"
        )
    if list(iter_jsonl(plan_path)) != rows:
        raise RuntimeError("saved action-sweep full call plan is stale")
    if saved.get("budget_gate", {}).get("status") != "PASS":
        raise RuntimeError("saved action-sweep dry-run did not pass its budget gate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed action sweep: exact dry-run and accepted cost hash before API use."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        help=(
            "Bind a PM-v2 sweep to its preregistered generator and retrieval "
            "settings. Required for data/pm_v2 runtime inputs."
        ),
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--endpoint",
        "--generator-endpoint",
        dest="endpoint",
        default=None,
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "data" / "synthetic" / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "synthetic" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "synthetic_sweep"
    )
    parser.add_argument("--max-cards", type=int)
    parser.add_argument(
        "--actions",
        help="Optional comma-separated pilot action IDs; full confirmatory runs omit this.",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        help="Frozen balanced PM-v2 compatibility pilot plan from script 31.",
    )
    parser.add_argument(
        "--completed-pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "pilot_plan.json",
        help="Pilot plan whose completed action/judge chain gates a full PM-v2 sweep.",
    )
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v2_development_pilot" / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_pilot_judging"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_pilot_judging"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_pilot_human_spot_check"
            / "pilot_human_spot_check_report.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_pilot_human_spot_check"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_judge_schema_smoke"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_development_judge_schema_smoke"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pm-v2-states",
        type=Path,
        help=(
            "PM-v2 states bound by the human semantic-sanity attestation; defaults "
            "to pm_v2_states.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        help=(
            "Evaluator-only PM-v2 contexts bound by the semantic-sanity "
            "attestation; defaults to evaluator_contexts.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--request-retries",
        type=int,
        help=(
            "Maximum physical attempts per logical generation call. PM-v2 binds "
            "this to development_sweep.request_retries=1."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop the sweep after the first failed logical generation call.",
    )
    parser.add_argument("--strategy-top-k", type=int)
    parser.add_argument(
        "--memory-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap memory items.",
    )
    parser.add_argument(
        "--strategy-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap strategy cards.",
    )
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=6000,
        help="Maximum physical HTTP attempts, including retries.",
    )
    parser.add_argument("--max-estimated-usd", type=float, default=10.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if args.max_api_calls <= 0 or args.max_estimated_usd < 0:
        raise ValueError("budget limits must be positive")

    experiment_config = load_config(args.config)
    pm_v2_root = (ROOT / "data" / "pm_v2").resolve()
    runtime_has_pm_v2_provenance = any(
        bool((row.get("provenance") or {}).get("pm_v2_state_id"))
        for row in iter_jsonl(args.runtime)
    )
    runtime_is_pm_v2 = (
        pm_v2_root in args.runtime.resolve().parents
        or runtime_has_pm_v2_provenance
    )
    if runtime_is_pm_v2:
        forbid_overwrite_of_spent_attempts(
            args.out_dir / "physical_attempt_ledger.jsonl",
            overwrite=args.overwrite,
            stage="PM-v2 action sweep",
        )
        if args.pilot_plan is None and (
            args.max_cards is not None or bool(args.actions)
        ):
            raise RuntimeError(
                "PM-v2 filtered sweeps require the exact frozen --pilot-plan; "
                "ad-hoc --actions/--max-cards cannot bypass the full-sweep "
                "development pilot gate"
            )
    if runtime_is_pm_v2 and args.pm_v2_config is None:
        raise RuntimeError(
            "PM-v2 action sweeps must pass --pm-v2-config so generator and "
            "retrieval settings are immutable and included in the cost hash"
        )
    if args.pilot_plan is not None and args.pm_v2_config is None:
        raise RuntimeError("--pilot-plan requires --pm-v2-config")
    if (args.pm_v2_states is not None or args.evaluator_contexts is not None) and (
        args.pm_v2_config is None
    ):
        raise RuntimeError(
            "--pm-v2-states/--evaluator-contexts require --pm-v2-config"
        )
    if args.pilot_plan is not None and (args.max_cards is not None or args.actions):
        raise RuntimeError("--pilot-plan cannot be combined with ad-hoc pilot filters")
    pilot_plan = read_json(args.pilot_plan) if args.pilot_plan is not None else None

    contract_bindings: dict = {}
    pm_v2_states_path = None
    evaluator_contexts_path = None
    deployable_feature_observability = None
    judge_schema_smoke = None
    evidence_filter_config = None
    memory_helpfulness_model = None
    evidence_filter_model_binding = None
    supporter_generation_contract = None
    if args.pm_v2_config is not None:
        pm_config = load_config(args.pm_v2_config)
        if pm_config.get("version") != "pm-v2.2":
            raise RuntimeError("action sweep requires PM-v2.2 config")
        supporter_generation_contract = SupporterGenerationContract.from_config(
            pm_config
        )
        sweep_config = dict(pm_config["development_sweep"])
        if set(sweep_config) != {
            "pricing_usd_per_mtok",
            "seed",
            "request_retries",
            "fail_fast",
        }:
            raise RuntimeError(
                "development_sweep must not duplicate the PM-v2.2 supporter "
                "generation treatment"
            )
        retrieval_config = dict(pm_config["retrieval"])
        evidence_filter_config = EvidenceFilterConfig.from_mapping(
            pm_config["evidence_filter"]
        )
        if not evidence_filter_config.enabled:
            raise RuntimeError(
                "PM-v2 main development sweep requires the frozen Evidence Filter"
            )
        filter_training = dict(pm_config["evidence_filter_training"])
        if filter_training.get("require_before_development_sweep") is not True:
            raise RuntimeError(
                "PM-v2 requires the supervised Evidence Filter before development sweep"
            )
        if runtime_is_pm_v2 or args.pilot_plan is not None:
            (
                memory_helpfulness_model,
                evidence_filter_model_binding,
            ) = require_evidence_filter_artifacts(
                checkpoint_path=args.evidence_filter_checkpoint,
                report_path=args.evidence_filter_report,
                attestation_path=args.evidence_filter_attestation,
                pm_v2_config_path=args.pm_v2_config,
            )
        else:
            evidence_filter_model_binding = {
                "mode": "lexical_diagnostic_only",
                "formal_pm_v2_sweep": False,
                "external_usefulness_status": "UNPROVEN",
            }
        sweep_pricing = dict(sweep_config["pricing_usd_per_mtok"])
        if set(sweep_pricing) != {"input", "output"}:
            raise RuntimeError(
                "development_sweep pricing must contain exactly input/output"
            )
        frozen_input_price = float(sweep_pricing["input"])
        frozen_output_price = float(sweep_pricing["output"])
        if frozen_input_price <= 0.0 or frozen_output_price <= 0.0:
            raise RuntimeError("PM-v2 sweep budget-accounting prices must be positive")
        semantic_sanity = None
        runtime_state_lineage = None
        if runtime_is_pm_v2 or pilot_plan is not None:
            pm_v2_states_path = args.pm_v2_states or (
                args.runtime.parent / "pm_v2_states.jsonl"
            )
            evaluator_contexts_path = args.evaluator_contexts or (
                args.runtime.parent / "evaluator_contexts.jsonl"
            )
            audited_states = load_states(pm_v2_states_path)
            runtime_state_lineage = require_pmv2_runtime_state_lineage(
                args.runtime, audited_states
            )
            semantic_sanity = require_semantic_sanity_pass(
                report_path=args.semantic_sanity_report,
                attestation_path=args.semantic_sanity_attestation,
                config=pm_config,
                config_path=args.pm_v2_config,
                states_path=pm_v2_states_path,
                backend_path=args.backend,
                evaluator_contexts_path=evaluator_contexts_path,
            )
            if pilot_plan is not None:
                evaluator_index = load_evaluator_context_index(
                    evaluator_contexts_path,
                    states=audited_states,
                    require_exact=True,
                )
                deployable_feature_observability = (
                    audit_deployable_feature_observability(
                        audited_states,
                        evaluator_contexts=evaluator_index,
                        settings=dict(
                            pm_config["development_judging"][
                                "compatibility_pilot"
                            ]["deployable_feature_observability"]
                        ),
                    )
                )
                if (
                    deployable_feature_observability.get("status") != "PASS"
                    or pilot_plan.get("deployable_feature_observability")
                    != deployable_feature_observability
                ):
                    raise RuntimeError(
                        "PM-v2 pilot deployable-feature observability gate "
                        "is absent, stale, or failed"
                    )
                judge_schema_smoke = (
                    require_development_judge_schema_smoke_pass(
                        summary_path=args.judge_schema_smoke_summary,
                        attestation_path=args.judge_schema_smoke_attestation,
                        experiment_config_path=args.config,
                        pm_v2_config_path=args.pm_v2_config,
                        states_path=pm_v2_states_path,
                        backend_path=args.backend,
                        evaluator_contexts_path=evaluator_contexts_path,
                        pilot_plan_path=args.pilot_plan,
                        semantic_sanity_report_path=args.semantic_sanity_report,
                        semantic_sanity_attestation_path=(
                            args.semantic_sanity_attestation
                        ),
                    )
                )
        api_cost_config = dict(pm_config["api_cost_planning"])
        if set(api_cost_config) != {
            "input_token_safety_factor",
            "fail_on_reported_input_overrun",
        }:
            raise RuntimeError("api_cost_planning keys do not match PM-v2")
        input_token_safety_factor = float(
            api_cost_config["input_token_safety_factor"]
        )
        fail_on_reported_input_overrun = bool(
            api_cost_config["fail_on_reported_input_overrun"]
        )
        if input_token_safety_factor < 1.0 or not fail_on_reported_input_overrun:
            raise RuntimeError(
                "PM-v2 requires input token safety factor >=1 and fail-on-overrun"
            )

        frozen_values = {
            "endpoint": supporter_generation_contract.generator_endpoint,
            "temperature": supporter_generation_contract.temperature,
            "max_output_tokens": (
                supporter_generation_contract.max_output_tokens
            ),
            "seed": int(sweep_config["seed"]),
            "request_retries": int(sweep_config["request_retries"]),
            "fail_fast": bool(sweep_config["fail_fast"]),
            "strategy_top_k": int(retrieval_config["strategy_top_k"]),
            "memory_min_score": float(retrieval_config["memory_min_score"]),
            "strategy_min_score": float(retrieval_config["strategy_min_score"]),
        }
        requested_values = {
            "endpoint": args.endpoint,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "request_retries": args.request_retries,
            "fail_fast": args.fail_fast,
            "strategy_top_k": args.strategy_top_k,
            "memory_min_score": args.memory_min_score,
            "strategy_min_score": args.strategy_min_score,
        }
        for name, requested in requested_values.items():
            if requested is not None and requested != frozen_values[name]:
                raise RuntimeError(
                    f"{name} override differs from the PM-v2 config; edit and "
                    "review the preregistration before generating any labels"
                )
        endpoint_name = frozen_values["endpoint"]
        temperature = frozen_values["temperature"]
        max_output_tokens = frozen_values["max_output_tokens"]
        seed = frozen_values["seed"]
        request_retries = frozen_values["request_retries"]
        if request_retries != 1:
            raise RuntimeError(
                "PM-v2 development_sweep.request_retries must equal 1 so the "
                "dry-run is a physical-attempt cap"
            )
        fail_fast = frozen_values["fail_fast"]
        if not fail_fast:
            raise RuntimeError(
                "PM-v2 development_sweep.fail_fast must be true so an exact "
                "action-matrix failure cannot spend the remaining budget"
            )
        strategy_top_k = frozen_values["strategy_top_k"]
        memory_min_score = frozen_values["memory_min_score"]
        strategy_min_score = frozen_values["strategy_min_score"]
        if (
            args.input_usd_per_mtok is not None
            and float(args.input_usd_per_mtok) != frozen_input_price
        ):
            raise RuntimeError("input pricing override differs from PM-v2 YAML")
        if (
            args.output_usd_per_mtok is not None
            and float(args.output_usd_per_mtok) != frozen_output_price
        ):
            raise RuntimeError("output pricing override differs from PM-v2 YAML")
        args.input_usd_per_mtok = frozen_input_price
        args.output_usd_per_mtok = frozen_output_price
        contract_bindings = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "pm_v2_version": str(pm_config["version"]),
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "development_sweep": sweep_config,
            "retrieval": retrieval_config,
            "evidence_filter": evidence_filter_config.payload(),
            "evidence_filter_config_sha256": evidence_filter_config.digest(),
            "evidence_filter_model": evidence_filter_model_binding,
            **(
                {
                    "semantic_sanity": semantic_sanity,
                    "runtime_state_lineage": runtime_state_lineage,
                }
                if semantic_sanity is not None
                else {}
            ),
            **(
                {
                    "deployable_feature_observability": (
                        deployable_feature_observability
                    ),
                    "judge_schema_smoke": judge_schema_smoke,
                }
                if pilot_plan is not None
                else {}
            ),
            "api_cost_planning": api_cost_config,
            "scope": (
                "compatibility_pilot"
                if pilot_plan is not None
                else "pilot"
                if args.max_cards is not None or args.actions
                else "full"
            ),
        }
    else:
        if args.input_usd_per_mtok is None or args.output_usd_per_mtok is None:
            raise RuntimeError(
                "non-PM-v2 sweeps require explicit input/output pricing"
            )
        endpoint_name = args.endpoint or "generator"
        temperature = 0.0 if args.temperature is None else float(args.temperature)
        max_output_tokens = (
            300 if args.max_output_tokens is None else int(args.max_output_tokens)
        )
        seed = 4311 if args.seed is None else int(args.seed)
        request_retries = (
            3 if args.request_retries is None else int(args.request_retries)
        )
        fail_fast = bool(args.fail_fast) if args.fail_fast is not None else False
        strategy_top_k = 3 if args.strategy_top_k is None else int(args.strategy_top_k)
        memory_min_score = args.memory_min_score
        strategy_min_score = args.strategy_min_score
        evidence_filter_config = None
        memory_helpfulness_model = None
        input_token_safety_factor = 1.0
        fail_on_reported_input_overrun = False

    endpoint = endpoint_from_config(experiment_config, endpoint_name)
    action_filter = None
    card_filter = None
    if pilot_plan is not None:
        assert supporter_generation_contract is not None
        if pilot_plan.get("status") != "READY":
            raise RuntimeError("PM-v2 compatibility pilot plan is not READY")
        if pilot_plan.get("protocol") != (
            "pm_v2_development_compatibility_pilot_v2_treatment_bound"
        ):
            raise RuntimeError("PM-v2 compatibility pilot protocol is stale")
        if (
            pilot_plan.get("supporter_generation_treatment")
            != supporter_generation_contract.payload()
            or pilot_plan.get("supporter_generation_treatment_sha256")
            != supporter_generation_contract.digest()
        ):
            raise RuntimeError(
                "PM-v2 pilot plan supporter-generation treatment mismatch"
            )
        expected_hashes = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "runtime_sha256": sha256_file(args.runtime),
            "backend_sha256": sha256_file(args.backend),
            "strategy_bank_sha256": sha256_file(args.strategy_bank),
        }
        mismatches = {
            name: {"expected": expected, "observed": pilot_plan.get(name)}
            for name, expected in expected_hashes.items()
            if pilot_plan.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"PM-v2 pilot plan lineage mismatch: {mismatches}")
        if pilot_plan.get("semantic_sanity") != contract_bindings.get(
            "semantic_sanity"
        ):
            raise RuntimeError(
                "PM-v2 pilot plan semantic-sanity attestation mismatch"
            )
        if pilot_plan.get("runtime_state_lineage") != contract_bindings.get(
            "runtime_state_lineage"
        ):
            raise RuntimeError("PM-v2 pilot plan runtime/state lineage mismatch")
        pilot_payload = {
            key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
        }
        if pilot_plan.get("pilot_plan_sha256") != sha256_text(
            canonical_json(pilot_payload)
        ):
            raise RuntimeError("PM-v2 pilot plan self-hash mismatch")
        action_filter = {str(value) for value in pilot_plan["actions"]}
        card_filter = {
            str(row["card_id"]) for row in pilot_plan["selected_states"]
        }
        contract_bindings["pilot_plan_sha256"] = str(
            pilot_plan["pilot_plan_sha256"]
        )
        contract_bindings["pilot_expected_keys_sha256"] = str(
            pilot_plan["expected_keys_sha256"]
        )
    elif args.actions:
        action_filter = {value.strip() for value in args.actions.split(",") if value.strip()}
        if not action_filter:
            raise ValueError("--actions did not contain any action IDs")
        for action_id in action_filter:
            parse_action_id(action_id)
    elif runtime_is_pm_v2 and args.max_cards is None:
        assert pm_v2_states_path is not None
        assert evaluator_contexts_path is not None
        if semantic_sanity is None or runtime_state_lineage is None:
            raise RuntimeError(
                "full PM-v2 sweep lacks semantic/runtime lineage required by the "
                "development pilot gate"
            )
        pilot_gate = require_development_pilot_gate(
            experiment_config_path=args.config,
            pm_v2_config_path=args.pm_v2_config,
            states_path=pm_v2_states_path,
            runtime_path=args.runtime,
            backend_path=args.backend,
            evaluator_contexts_path=evaluator_contexts_path,
            strategy_bank_path=args.strategy_bank,
            semantic_sanity=semantic_sanity,
            semantic_sanity_report_path=args.semantic_sanity_report,
            semantic_sanity_attestation_path=args.semantic_sanity_attestation,
            runtime_state_lineage=runtime_state_lineage,
            pilot_plan_path=args.completed_pilot_plan,
            pilot_sweep_summary_path=args.pilot_sweep_summary,
            pilot_sweep_attestation_path=args.pilot_sweep_attestation,
            judge_compatibility_summary_path=args.judge_compatibility_summary,
            judge_compatibility_attestation_path=(
                args.judge_compatibility_attestation
            ),
            pilot_human_spot_check_report_path=(
                args.pilot_human_spot_check_report
            ),
            pilot_human_spot_check_attestation_path=(
                args.pilot_human_spot_check_attestation
            ),
            judge_schema_smoke_summary_path=args.judge_schema_smoke_summary,
            judge_schema_smoke_attestation_path=(
                args.judge_schema_smoke_attestation
            ),
        )
        contract_bindings["development_pilot_gate"] = pilot_gate
    estimate, rows = plan_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        contract_bindings=contract_bindings,
    )
    gate = _budget_gate(
        estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    result = {**estimate, "budget_gate": gate}

    if args.dry_run:
        _persist_or_validate_dry_run(args.out_dir, result, rows)
        print(result)
        if gate["status"] != "PASS":
            raise RuntimeError("action-sweep dry-run failed the frozen budget gate")
        return

    if gate["status"] != "PASS":
        raise RuntimeError("action-sweep API run blocked by budget gate")
    _require_saved_dry_run(args.out_dir, result, rows)
    expected_hash = str(result["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match current action-sweep plan"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        args.out_dir / "action_outcomes.jsonl",
        args.out_dir / "raw_api_calls.jsonl",
        args.out_dir / "summary.json",
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        overwrite=args.overwrite,
        max_physical_api_attempts=args.max_api_calls,
        contract_bindings={
            **contract_bindings,
            "accepted_cost_estimate_sha256": expected_hash,
            "pricing": result["pricing"],
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
