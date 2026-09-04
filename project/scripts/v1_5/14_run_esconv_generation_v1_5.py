#!/usr/bin/env python3
"""Generate PM-v1.5 supporter responses for the frozen ESConv external test.

Action-first, not condition-first: every ESConv state has exactly two legal
actions (M0+R0, M0+RS -- memory is structurally unavailable for this
single-session corpus), so the real generation space is
``len(states) * 2`` logical (card_id, action_id) outcomes. The four policy
conditions (learned_pm, transparent_rule_same_step0, always_r0, always_rs)
are a downstream *mapping* over these same two real per-state outcomes,
computed by a later evaluation step that reads this run's
``action_outcomes.jsonl`` plus ``policy_choices.jsonl`` -- never a reason to
generate more than two replies per state here.

This script is deliberately a thin wrapper around the already-hardened,
already-tested ``plan_action_sweep``/``run_action_sweep`` (the same
functions the internal 7,488-action sweep uses), restricted to the ESConv
card set via ``action_filter={"M0+R0", "M0+RS"}``, rather than a new
generation loop. That reuse gets the existing prompt-equivalence
deduplication (a zero-strategy-hit RS prompt identical to R0's is paid for
once, not twice -- reported via ``prompt_alias_rate``/
``prompt_equivalence_classes``, never silently backfilled with invented
evidence), the same query-builder/retrieval/action-execution/prompt-compiler
code, and the same physical-attempt-ledger/bounded-retry machinery, for
free.

Never reads ``audit_only.jsonl`` (evaluator-only gold_response/
gold_strategy) -- this script only ever touches ``runtime_states.jsonl``
(observable state) and ``memory_backend.jsonl`` (structurally empty for
ESConv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.esconv_v1_5 import ESCONV_V1_5_ALLOWED_ACTIONS
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.freeze import require_study_freeze
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
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.response_mechanism_contract import (
    build_response_mechanism_contract,
    require_matching_response_mechanism_contract,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep


ROOT = Path(__file__).resolve().parents[2]
ESCONV_GENERATION_STAGE = "external_esconv_generation"
# Frozen, conservative list pricing (see scripts/v1_5_create_freeze.py's
# generation_contract["generator_pricing_usd_per_mtok"], shared with EvoEmo).
FROZEN_GENERATOR_PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}


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


def _persist_or_validate_dry_run(out_dir: Path, current: dict, rows: list[dict]) -> str:
    """Freeze the exact accepted estimate/plan once an HTTP attempt exists."""

    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError(
                "spent ESConv-generation ledger freezes the accepted dry-run, "
                "but its estimate or full call plan is missing"
            )
        if read_json(estimate_path) != current or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError(
                "spent ESConv-generation ledger freezes the original exact "
                "estimate and full call plan; current dry-run drift is rejected"
            )
        return "VALIDATED_EXISTING"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, current)
    write_jsonl(plan_path, rows)
    return "WRITTEN"


def _require_saved_dry_run(out_dir: Path, current: dict, rows: list[dict]) -> None:
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same "
            "output directory"
        )
    saved = read_json(estimate_path)
    if saved != current:
        raise RuntimeError(
            "saved ESConv-generation dry-run does not match current "
            "inputs/configuration; run --dry-run again"
        )
    if list(iter_jsonl(plan_path)) != rows:
        raise RuntimeError("saved ESConv-generation full call plan is stale")
    if saved.get("budget_gate", {}).get("status") != "PASS":
        raise RuntimeError(
            "saved ESConv-generation dry-run did not pass its budget gate"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--esconv-runtime-states",
        type=Path,
        default=ROOT / "data" / "esconv_test_v1_5" / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--esconv-memory-backend",
        type=Path,
        default=ROOT / "data" / "esconv_test_v1_5" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--esconv-policy-choices",
        type=Path,
        default=ROOT / "outputs" / "esconv_v1_5_preflight" / "policy_choices.jsonl",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_study_freeze.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "esconv_generation_v1_5"
    )
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--max-input-tokens-per-call", type=int, required=True)
    parser.add_argument("--input-usd-per-mtok", type=float)
    parser.add_argument("--output-usd-per-mtok", type=float)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid ESConv generation runs prohibit --overwrite; use a new "
            "output directory"
        )
    if args.input_usd_per_mtok is not None and (
        float(args.input_usd_per_mtok)
        != FROZEN_GENERATOR_PRICING_USD_PER_MTOK["input"]
    ):
        raise RuntimeError("--input-usd-per-mtok must equal the frozen 0.15")
    if args.output_usd_per_mtok is not None and (
        float(args.output_usd_per_mtok)
        != FROZEN_GENERATOR_PRICING_USD_PER_MTOK["output"]
    ):
        raise RuntimeError("--output-usd-per-mtok must equal the frozen 0.60")

    experiment_config = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    if pm_v1_5_config.get("version") != "pm-v1.5":
        raise RuntimeError("ESConv generation requires a pm-v1.5 config")

    require_paid_run_release(
        pm_v1_5_config,
        config_path=args.pm_v1_5_config,
        stage=ESCONV_GENERATION_STAGE,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )

    legal_actions = tuple(
        pm_v1_5_config["esconv_external_evaluation"]["legal_actions"]
    )
    if legal_actions != ESCONV_V1_5_ALLOWED_ACTIONS:
        raise RuntimeError(
            "ESConv legal actions differ from the frozen adapter contract"
        )
    action_filter = set(legal_actions)

    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v1_5_config
    )
    generator_endpoint = endpoint_from_config(
        experiment_config, supporter_generation_contract.generator_endpoint
    )
    generator_endpoint_sha256 = sha256_text(
        canonical_json(
            {
                "model": generator_endpoint.model,
                "family": generator_endpoint.family,
                "base_url": generator_endpoint.base_url,
            }
        )
    )

    # PM-v1.5: Evidence Filter is out of scope everywhere (sweep, EvoEmo,
    # ESConv) -- PM is a pure pre-retrieval router.
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v1_5_config["evidence_filter"], "enabled": False}
    )
    # Reuse the exact same development/external retrieval settings as the
    # internal sweep and EvoEmo -- required so response_mechanism_contract
    # is byte-identical across all three real generation stages, not merely
    # equal by coincidence.
    external = dict(pm_v1_5_config["external_evaluation"])
    strategy_top_k = int(external["strategy_top_k"])
    memory_min_score = external.get("memory_min_score")
    strategy_min_score = external.get("strategy_min_score")

    response_mechanism_contract = build_response_mechanism_contract(
        project_root=ROOT,
        supporter_generation_contract=supporter_generation_contract,
        generator_endpoint_sha256=generator_endpoint_sha256,
        strategy_bank_sha256=sha256_file(args.strategy_bank),
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        strategy_top_k=strategy_top_k,
        evidence_filter_enabled=bool(evidence_filter_config.enabled),
    )

    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[
            args.pm_v1_5_config,
            args.esconv_runtime_states,
            args.esconv_memory_backend,
            args.strategy_bank,
            args.esconv_policy_choices,
        ],
    )
    freeze_data = read_json(args.freeze)
    notes = freeze_data.get("notes") or {}
    esconv_contract = notes.get("esconv_generation_contract") or {}
    if not esconv_contract:
        raise RuntimeError("study freeze lacks esconv_generation_contract")
    if list(esconv_contract.get("legal_actions") or []) != list(legal_actions):
        raise RuntimeError("study freeze ESConv legal actions are absent or stale")
    if esconv_contract.get("states_sha256") != sha256_file(
        args.esconv_runtime_states
    ):
        raise RuntimeError("study freeze ESConv states are absent or stale")
    if esconv_contract.get("policy_choices_sha256") != sha256_file(
        args.esconv_policy_choices
    ):
        raise RuntimeError(
            "study freeze ESConv policy choices are absent or stale"
        )
    live_card_ids = sorted(
        {str(row["card_id"]) for row in iter_jsonl(args.esconv_runtime_states)}
    )
    live_expected_action_keys = sorted(
        (card_id, action_id)
        for card_id in live_card_ids
        for action_id in legal_actions
    )
    if (
        esconv_contract.get("expected_state_count") != len(live_card_ids)
        or esconv_contract.get("expected_logical_action_outcomes")
        != len(live_expected_action_keys)
        or esconv_contract.get("expected_action_keys_sha256")
        != sha256_text(canonical_json(live_expected_action_keys))
    ):
        raise RuntimeError(
            "study freeze ESConv action-first plan is absent or stale"
        )
    require_matching_response_mechanism_contract(
        expected=esconv_contract.get("response_mechanism_contract") or {},
        actual=response_mechanism_contract,
        context="study freeze vs live ESConv external generation",
    )

    contract_bindings = {
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "pm_v1_5_version": str(pm_v1_5_config["version"]),
        "response_mechanism_contract": response_mechanism_contract,
        "esconv_generation_contract_sha256": sha256_text(
            canonical_json(esconv_contract)
        ),
        "legal_actions": list(legal_actions),
        "evidence_filter": evidence_filter_config.payload(),
        "evidence_filter_config_sha256": evidence_filter_config.digest(),
        "retrieval": {
            "strategy_top_k": strategy_top_k,
            "memory_min_score": memory_min_score,
            "strategy_min_score": strategy_min_score,
        },
        "scope": "esconv_action_first_full",
    }

    api_cost_config = dict(pm_v1_5_config["api_cost_planning"])
    input_token_safety_factor = float(
        api_cost_config["input_token_safety_factor"]
    )
    fail_on_reported_input_overrun = bool(
        api_cost_config["fail_on_reported_input_overrun"]
    )
    if input_token_safety_factor < 1.0 or not fail_on_reported_input_overrun:
        raise RuntimeError(
            "PM-v1.5 requires input token safety factor >=1 and fail-on-overrun"
        )
    input_usd_per_mtok = FROZEN_GENERATOR_PRICING_USD_PER_MTOK["input"]
    output_usd_per_mtok = FROZEN_GENERATOR_PRICING_USD_PER_MTOK["output"]
    # Deterministic, project-wide default (matches development_sweep.seed);
    # temperature=0 generation makes this a reproducibility/audit binding,
    # not a source of real output variation.
    seed = 4311

    plan_kwargs = dict(
        endpoint=generator_endpoint,
        card_filter=None,
        action_filter=action_filter,
        temperature=supporter_generation_contract.temperature,
        max_tokens=supporter_generation_contract.max_output_tokens,
        seed=seed,
        request_retries=1,
        fail_fast=True,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        supporter_generation_contract=supporter_generation_contract,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        contract_bindings=contract_bindings,
    )
    estimate, rows = plan_action_sweep(
        args.esconv_runtime_states,
        args.esconv_memory_backend,
        args.strategy_bank,
        **plan_kwargs,
    )
    if int(estimate["logical_api_calls"]) != len(live_expected_action_keys):
        raise RuntimeError(
            "ESConv action-first plan does not equal len(states) * "
            "len(legal_actions) -- action_filter or state set drifted"
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
            raise RuntimeError(
                "ESConv-generation dry-run failed the frozen budget gate"
            )
        return

    if gate["status"] != "PASS":
        raise RuntimeError("ESConv-generation API run blocked by budget gate")
    _require_saved_dry_run(args.out_dir, result, rows)
    expected_hash = str(result["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match the current "
            "ESConv-generation plan"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_action_sweep(
        args.esconv_runtime_states,
        args.esconv_memory_backend,
        args.strategy_bank,
        args.out_dir / "action_outcomes.jsonl",
        args.out_dir / "raw_api_calls.jsonl",
        args.out_dir / "summary.json",
        overwrite=False,
        study_freeze_sha256=verification["freeze_sha256"],
        **plan_kwargs,
    )
    print(summary)


if __name__ == "__main__":
    main()
