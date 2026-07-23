#!/usr/bin/env python3
"""Generate PM-v1.5 supporter responses for the ESConv-auxiliary training states.

This is the same action-first mechanism as
``14_run_esconv_generation_v1_5.py`` (the frozen external ESConv test
generation) -- every auxiliary state has exactly the same two legal actions
(M0+R0, M0+RS; memory is structurally unavailable for this single-session
corpus) -- applied instead to the 719 bank-disjoint auxiliary states built by
``12b_build_esconv_auxiliary_v1_5.py`` (train=318, calibration=170,
internal_test=231, one split per invocation via ``--split``).

Deliberately not coupled to ``require_study_freeze``/``esconv_generation_
contract``: the auxiliary track is training-support data, not part of the
frozen external-evaluation study. It still reuses the identical
``SupporterGenerationContract``/retrieval settings as the frozen ESConv-test
generation (``external_evaluation`` config block) so the resulting
``response_mechanism_contract`` is byte-identical across both real ESConv
generation stages -- required so a PM trained partly on this data sees the
exact same generation mechanism it will be evaluated against externally.

Never reads ``audit_only.jsonl`` (evaluator-only gold_response/
gold_strategy) -- only ``runtime_states.jsonl`` (observable state) and
``memory_backend.jsonl`` (structurally empty for ESConv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.esconv_v1_5 import ESCONV_V1_5_ALLOWED_ACTIONS
from metacom_pm.evidence_filter import EvidenceFilterConfig
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
from metacom_pm.paid_run_release import (
    require_paid_run_release,
    resolve_first_unconsumed_output_directory,
)
from metacom_pm.response_mechanism_contract import build_response_mechanism_contract
from metacom_pm.sweep import plan_action_sweep, run_action_sweep


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "calibration", "internal_test")
# Frozen, conservative list pricing -- identical to the ESConv-test generator
# (scripts/v1_5/14_run_esconv_generation_v1_5.py), shared with EvoEmo.
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
                "spent ESConv-auxiliary-generation ledger freezes the accepted "
                "dry-run, but its estimate or full call plan is missing"
            )
        if read_json(estimate_path) != current or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError(
                "spent ESConv-auxiliary-generation ledger freezes the original "
                "exact estimate and full call plan; current dry-run drift is "
                "rejected"
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
            "saved ESConv-auxiliary-generation dry-run does not match current "
            "inputs/configuration; run --dry-run again"
        )
    if list(iter_jsonl(plan_path)) != rows:
        raise RuntimeError("saved ESConv-auxiliary-generation full call plan is stale")
    if saved.get("budget_gate", {}).get("status") != "PASS":
        raise RuntimeError(
            "saved ESConv-auxiliary-generation dry-run did not pass its budget gate"
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
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument(
        "--scope",
        required=True,
        choices=("pilot", "full"),
        help=(
            "Folded into the stage name and output directory so a pilot-scale "
            "and full-scale run for the same split can never collide on the "
            "same default directory."
        ),
    )
    parser.add_argument(
        "--auxiliary-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
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
            "paid ESConv-auxiliary generation runs prohibit --overwrite; use a "
            "new output directory"
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
        raise RuntimeError("ESConv-auxiliary generation requires a pm-v1.5 config")

    split = str(args.split)
    scope = str(args.scope)
    stage = f"esconv_auxiliary_generation_{scope}_{split}"
    require_paid_run_release(
        pm_v1_5_config,
        config_path=args.pm_v1_5_config,
        stage=stage,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    out_dir = resolve_first_unconsumed_output_directory(
        args.out_root / f"esconv_auxiliary_generation_v1_5_{scope}_{split}",
        config=pm_v1_5_config,
        config_path=args.pm_v1_5_config,
    )

    legal_actions = tuple(
        pm_v1_5_config["esconv_external_evaluation"]["legal_actions"]
    )
    if legal_actions != ESCONV_V1_5_ALLOWED_ACTIONS:
        raise RuntimeError(
            "ESConv-auxiliary legal actions differ from the frozen adapter contract"
        )
    action_filter = set(legal_actions)

    split_dir = args.auxiliary_dir / split
    runtime_states_path = split_dir / "runtime_states.jsonl"
    memory_backend_path = split_dir / "memory_backend.jsonl"
    if not runtime_states_path.is_file() or not memory_backend_path.is_file():
        raise RuntimeError(
            f"ESConv-auxiliary split {split!r} is missing runtime_states.jsonl "
            "or memory_backend.jsonl; build it with "
            "scripts/v1_5/12b_build_esconv_auxiliary_v1_5.py first"
        )

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

    # PM-v1.5: Evidence Filter is out of scope everywhere -- PM is a pure
    # pre-retrieval router. Identical to the ESConv-test/internal-sweep forks.
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v1_5_config["evidence_filter"], "enabled": False}
    )
    # Same external retrieval settings as the frozen ESConv-test generation, so
    # response_mechanism_contract is byte-identical across both real ESConv
    # generation stages.
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

    live_card_ids = sorted(
        {str(row["card_id"]) for row in iter_jsonl(runtime_states_path)}
    )
    live_expected_action_keys = sorted(
        (card_id, action_id)
        for card_id in live_card_ids
        for action_id in legal_actions
    )

    # Binds the execution-behavior code that will actually run the batch
    # (transport-retry/circuit-breaker logic lives in sweep.py, not in this
    # script) into the identity, matching the shared_code_manifest lesson
    # from v1_5_run_automated_semantic_review.py: without this, a future
    # runtime bugfix to that shared code would silently apply under an
    # already-approved cost_estimate_sha256 instead of minting a fresh one.
    shared_code_paths = {
        "runner": Path(__file__).resolve(),
        "sweep": ROOT / "src" / "metacom_pm" / "sweep.py",
        "api": ROOT / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
    }
    shared_code_manifest = {
        name: {
            "relative_path": str(path.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(shared_code_paths.items())
    }

    # Transport-only bounded retry: every physical HTTP attempt is ledgered;
    # 5xx/timeout/429 failures get bounded retries with backoff up to
    # request_retries; content-level terminal errors (schema, completion-gate
    # rejection, reported-token-overrun) are never blindly retried. fail_fast
    # is False so one isolated failure never aborts the whole 1,438-call
    # batch; a same-class circuit breaker still stops a systemic outage.
    transport_retry_policy = "bounded_transport"
    transport_backoff_seconds = (10.0, 30.0, 60.0)
    consecutive_same_class_circuit_breaker = 5

    contract_bindings = {
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "pm_v1_5_version": str(pm_v1_5_config["version"]),
        "response_mechanism_contract": response_mechanism_contract,
        "legal_actions": list(legal_actions),
        "evidence_filter": evidence_filter_config.payload(),
        "evidence_filter_config_sha256": evidence_filter_config.digest(),
        "retrieval": {
            "strategy_top_k": strategy_top_k,
            "memory_min_score": memory_min_score,
            "strategy_min_score": strategy_min_score,
        },
        "scope": f"esconv_auxiliary_action_first_{split}",
        "auxiliary_runtime_states_sha256": sha256_file(runtime_states_path),
        "auxiliary_memory_backend_sha256": sha256_file(memory_backend_path),
        "shared_code_manifest": shared_code_manifest,
        "shared_code_manifest_sha256": sha256_text(
            canonical_json(shared_code_manifest)
        ),
        "transport_retry_policy": transport_retry_policy,
        "transport_backoff_seconds": list(transport_backoff_seconds),
        "consecutive_same_class_circuit_breaker": (
            consecutive_same_class_circuit_breaker
        ),
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
    # Deterministic, project-wide default (matches development_sweep.seed and
    # the ESConv-test generator); temperature=0 generation makes this a
    # reproducibility/audit binding, not a source of real output variation.
    seed = 4311

    plan_kwargs = dict(
        endpoint=generator_endpoint,
        card_filter=None,
        action_filter=action_filter,
        temperature=supporter_generation_contract.temperature,
        max_tokens=supporter_generation_contract.max_output_tokens,
        seed=seed,
        # Bounded per-call physical-attempt budget for transient transport
        # failures (500/503/timeouts/429); a single request_retries=1 budget
        # is not viable for a real 1,438-call batch on an endpoint with a
        # documented history of rate limits and transient 5xx/timeouts.
        # fail_fast=False so one isolated failure never aborts the whole
        # batch (transport_retry_policy/circuit breaker are added below,
        # only for run_action_sweep -- plan_action_sweep does not accept
        # them, so they are bound into contract_bindings instead).
        request_retries=4,
        fail_fast=False,
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
        runtime_states_path,
        memory_backend_path,
        args.strategy_bank,
        **plan_kwargs,
    )
    if int(estimate["logical_api_calls"]) != len(live_expected_action_keys):
        raise RuntimeError(
            "ESConv-auxiliary action-first plan does not equal len(states) * "
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
        _persist_or_validate_dry_run(out_dir, result, rows)
        print(result)
        if gate["status"] != "PASS":
            raise RuntimeError(
                "ESConv-auxiliary-generation dry-run failed the frozen budget gate"
            )
        return

    if gate["status"] != "PASS":
        raise RuntimeError(
            "ESConv-auxiliary-generation API run blocked by budget gate"
        )
    _require_saved_dry_run(out_dir, result, rows)
    expected_hash = str(result["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match the current "
            "ESConv-auxiliary-generation plan"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # run_action_sweep, unlike plan_action_sweep, does not accept pricing
    # kwargs (it executes generation; cost accounting was already frozen in
    # the accepted dry-run estimate above).
    run_kwargs = {
        key: value
        for key, value in plan_kwargs.items()
        if key not in ("input_usd_per_mtok", "output_usd_per_mtok")
    }
    run_kwargs.update(
        transport_retry_policy=transport_retry_policy,
        transport_backoff_seconds=transport_backoff_seconds,
        consecutive_same_class_circuit_breaker=(
            consecutive_same_class_circuit_breaker
        ),
    )
    summary = run_action_sweep(
        runtime_states_path,
        memory_backend_path,
        args.strategy_bank,
        out_dir / "action_outcomes.jsonl",
        out_dir / "raw_api_calls.jsonl",
        out_dir / "summary.json",
        overwrite=False,
        study_freeze_sha256=None,
        **run_kwargs,
    )
    print(summary)


if __name__ == "__main__":
    main()
