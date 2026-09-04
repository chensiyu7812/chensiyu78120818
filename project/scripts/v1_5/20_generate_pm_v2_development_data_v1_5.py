#!/usr/bin/env python3
"""Generate PM-v1.5 synthetic development data on the frozen fast track.

The V1.5 branch requires both the one-call generation-compatibility pilot and
an attested, multi-family automated semantic review before a paid full run.
It intentionally does not claim independent human validation and must not be
reported as equivalent to the PM-v2.2 human-review protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.api import (
    StructuredOutputValidationError,
    chat_request_payload,
    require_reported_usage,
)
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.contracts import StrategyCard
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import PMV2Split, ResourceNeedRegime
from metacom_pm.pm_v2_data import (
    GENERATION_MAX_OUTPUT_TOKENS,
    GENERATION_TEMPERATURE,
    GenerationDraftCompilationError,
    GeneratedBundleDraft,
    GeneratedUserBundle,
    audit_cross_split_near_duplicates,
    bind_bundle_to_generation_run,
    generate_user_bundle,
    generation_contract_hash,
    generation_messages,
    load_bundles,
    load_states,
    require_bundle_generation_binding,
    validate_bundle,
    validate_successful_generation_trace,
    write_development_dataset,
)
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    SEMANTIC_FAMILY_COHORTS_BY_SPLIT,
    TRAIN_SEMANTIC_FAMILIES,
    build_generation_compatibility_contract,
    generation_family_schedule,
    require_generation_compatibility_attestation,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.v1_5_automated_semantic_review import (
    require_automated_semantic_review_pass,
)

ROOT = Path(__file__).resolve().parents[2]

GENERATION_REQUEST_RETRIES = 1
GENERATION_COST_PROTOCOL = "pm_v2_generation_cost_v2_persistent_attempt_ledger"


class ReportedInputTokenOverrun(RuntimeError):
    pass


def _finish_generation_attempt(
    ledger: PersistentAttemptLedger,
    reservation,
    *,
    bundle: GeneratedUserBundle | None,
    succeeded: bool,
    error: str | None,
) -> None:
    """Persist terminal provider provenance for both success and failure."""

    ledger.finish(
        reservation,
        succeeded=succeeded,
        request_hash=(
            str(bundle.provenance.get("request_hash") or "") or None
            if bundle is not None
            else None
        ),
        usage=(
            dict(bundle.provenance.get("reported_usage") or {})
            if bundle is not None
            else None
        ),
        error=error,
        result=(
            {"bundle": bundle.model_dump(mode="json")}
            if bundle is not None
            else None
        ),
    )


def _require_bundle_reported_usage(bundle: GeneratedUserBundle) -> dict[str, int]:
    usage = require_reported_usage(
        bundle.provenance.get("reported_usage"),
        stage="PM-v2 synthetic generation",
    )
    bundle.provenance["reported_usage"] = usage
    return usage


def _recover_successful_bundles_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    all_user_attempts: dict[str, dict[str, Any]],
    existing: dict[str, GeneratedUserBundle],
    work_path: Path,
    generation_binding: dict[str, Any],
    family_by_user: dict[str, list[str]],
) -> int:
    """Materialize paid successes after a crash between ledger and work append."""

    recovered = 0
    for user_id, row in all_user_attempts.items():
        if user_id in existing:
            continue
        successful = [
            attempt
            for attempt in row["attempts"]
            if ledger.succeeded(str(attempt["call_key"]))
        ]
        if not successful:
            continue
        if len(successful) != 1:
            raise RuntimeError(
                f"user {user_id} has multiple successful physical generation calls"
            )
        call_key = str(successful[0]["call_key"])
        terminal = ledger.terminal_row(call_key)
        result = (terminal or {}).get("result") or {}
        bundle_payload = result.get("bundle")
        if not isinstance(bundle_payload, dict):
            raise RuntimeError(
                f"successful generation attempt lacks recoverable bundle: {user_id}"
            )
        bundle = GeneratedUserBundle.model_validate(bundle_payload)
        if bundle.user_id != user_id:
            raise RuntimeError("recovered generation bundle user_id mismatch")
        require_bundle_generation_binding(bundle, generation_binding)
        strict_bundle_check(bundle, family_by_user[user_id])
        if str(bundle.provenance.get("physical_call_key") or "") != call_key:
            raise RuntimeError("recovered generation bundle call-key mismatch")
        append_jsonl(work_path, bundle.model_dump(mode="json"))
        existing[user_id] = bundle
        recovered += 1
    return recovered


def _budget_gate(
    estimate: dict[str, Any],
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict[str, Any]:
    checks = {
        "maximum_api_calls": int(
            estimate["maximum_physical_api_attempts_including_history"]
        )
        <= max_api_calls,
        "pending_users_have_attempt_capacity": not bool(
            estimate["blocked_pending_users"]
        ),
        "maximum_estimated_cost_usd": float(
            estimate["maximum_estimated_cost_usd"]
        )
        <= max_estimated_usd,
        "max_input_tokens_per_call": int(estimate["max_input_tokens"])
        <= max_input_tokens_per_call,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _strict_existing_bundles(path: Path) -> dict[str, GeneratedUserBundle]:
    result: dict[str, GeneratedUserBundle] = {}
    if not path.exists():
        return result
    for bundle in load_bundles(path):
        if bundle.user_id in result:
            raise RuntimeError(
                f"duplicate resumable generation bundle for user {bundle.user_id}"
            )
        result[bundle.user_id] = bundle
    return result


def _require_saved_dry_run(
    out_dir: Path,
    current: dict[str, Any],
) -> None:
    estimate_path = out_dir / "generation_cost_estimate.json"
    plan_path = out_dir / "generation_call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same output "
            "directory"
        )
    saved = read_json(estimate_path)
    if saved.get("cost_estimate_sha256") != current.get("cost_estimate_sha256"):
        raise RuntimeError(
            "saved generation dry-run does not match the current generator, seed, "
            "configuration, prompt, code, resume state, pricing, or budget limits; "
            "run --dry-run again"
        )
    if saved.get("call_plan_sha256") != current.get("call_plan_sha256"):
        raise RuntimeError("saved generation call-plan hash is stale")


def read_seed_dialogues(path: Path) -> list[str]:
    seeds: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, str):
                text = row
            else:
                text = row.get("dialogue_text") or row.get("text") or row.get("content")
                if not text and isinstance(row.get("dialogue"), list):
                    text = "\n".join(
                        f"{turn.get('role','unknown')}: {turn.get('content','')}"
                        for turn in row["dialogue"]
                    )
            if text and str(text).strip():
                seeds.append(str(text).strip())
    if not seeds:
        raise ValueError("seed dialogue file contains no usable text")
    return seeds


def strict_bundle_check(bundle: GeneratedUserBundle, allowed_families: list[str]) -> None:
    validate_bundle(bundle)
    expected_regimes = set(ResourceNeedRegime)
    observed_regimes = {case.regime for case in bundle.cases}
    if observed_regimes != expected_regimes or len(bundle.cases) != len(expected_regimes):
        raise ValueError(
            "bundle must contain exactly one case per PM-v2 regime; "
            f"missing={sorted(x.value for x in expected_regimes-observed_regimes)}, "
            f"extra={len(bundle.cases)-len(observed_regimes)}"
        )
    expected_families = set(allowed_families)
    observed_families = {case.semantic_family for case in bundle.cases}
    invalid_families = sorted(observed_families - expected_families)
    missing_families = sorted(expected_families - observed_families)
    if invalid_families or missing_families:
        raise ValueError(
            "bundle semantic families must exactly cover its frozen assignment: "
            f"missing={missing_families}, extra={invalid_families}"
        )
    for case in bundle.cases:
        missing_sources = [
            name
            for name, memories in (
                ("MP", case.profile_memories),
                ("MS", case.summary_memories),
                ("ME", case.event_memories),
            )
            if not memories
        ]
        if missing_sources:
            raise ValueError(
                f"case {case.case_id} is missing sources {missing_sources}; "
                "all development states must expose the complete 16-action space"
            )
        source_sizes = {
            "MP": len(case.profile_memories),
            "MS": len(case.summary_memories),
            "ME": len(case.event_memories),
        }
        if set(source_sizes.values()) != {2}:
            raise ValueError(
                f"case {case.case_id} has label-revealing source inventory sizes: "
                f"{source_sizes}; every source must expose exactly two items"
            )
    validate_successful_generation_trace(bundle)


def enforce_full_state_design(
    report: dict,
    *,
    train_users: int,
    calibration_users: int,
    internal_test_users: int,
) -> dict:
    """Fail closed on the YAML-frozen full-state and 14/5/5 family design."""

    cases_per_user = len(ResourceNeedRegime)
    expected_split_state_counts = {
        PMV2Split.TRAIN.value: train_users * cases_per_user,
        PMV2Split.CALIBRATION.value: calibration_users * cases_per_user,
        PMV2Split.INTERNAL_TEST.value: internal_test_users * cases_per_user,
    }
    if report["split_counts"] != expected_split_state_counts:
        raise RuntimeError(
            "generated PM-v2 state counts do not match the frozen full design: "
            f"expected={expected_split_state_counts}, observed={report['split_counts']}"
        )
    expected_total_states = sum(expected_split_state_counts.values())
    split_manifest_report = report["split_manifest"]
    if (
        report["n_states"] != expected_total_states
        or split_manifest_report["total_states"] != expected_total_states
        or split_manifest_report["unique_normalized_current_user_texts"]
        != expected_total_states
        or split_manifest_report["normalized_current_user_text_unique_rate"] != 1.0
    ):
        raise RuntimeError(
            "generated PM-v2 current-user texts are not globally unique over the "
            "frozen full state design"
        )
    expected_family_union_counts = {
        PMV2Split.TRAIN.value: 14,
        PMV2Split.CALIBRATION.value: 5,
        PMV2Split.INTERNAL_TEST.value: 5,
    }
    observed_family_union_counts = {
        split: int(row["observed_count"])
        for split, row in report["semantic_family_coverage"]["splits"].items()
    }
    if observed_family_union_counts != expected_family_union_counts:
        raise RuntimeError(
            "generated PM-v2 semantic-family union counts do not match 14/5/5: "
            f"{observed_family_union_counts}"
        )
    result = {
        "status": "PASS",
        "cases_per_user": cases_per_user,
        "expected_split_state_counts": expected_split_state_counts,
        "expected_total_states": expected_total_states,
        "unique_normalized_current_user_texts": expected_total_states,
        "normalized_current_user_text_unique_rate": 1.0,
        "semantic_family_union_counts": observed_family_union_counts,
    }
    report["full_state_design"] = result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed PM-v2 synthetic generation: save an exact prompt plan "
            "and accepted conservative cost bound before API use."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--seed-dialogues",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "train_seed_dialogues_v1_5.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "pm_v1_5")
    parser.add_argument("--train-users", type=int)
    parser.add_argument("--calibration-users", type=int)
    parser.add_argument("--internal-test-users", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-generation-attempts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        help="Optional exact-match assertion against frozen data-generation pricing.",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        help="Optional exact-match assertion against frozen data-generation pricing.",
    )
    parser.add_argument("--max-api-calls", type=int, default=200)
    parser.add_argument("--max-estimated-usd", type=float, default=25.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_generation_compatibility_pilot"
            / "artifact_attestation.json"
        ),
        help="Required PASS source-grounded one-call pilot for full --run only.",
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_automated_semantic_review" / "gate_report.json",
        help=(
            "PM-v1.5 replacement for the human V8 review: output of "
            "scripts/v1_5_run_automated_semantic_review.py, must show "
            "status=PASS for --run."
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
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid PM-v1.5 data-generation runs prohibit --overwrite")

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if (
        args.max_api_calls <= 0
        or args.max_estimated_usd < 0
        or args.max_input_tokens_per_call <= 0
    ):
        raise ValueError("budget limits must be positive (USD may be zero)")

    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") != "pm-v1.5":
        raise ValueError("PM-v1.5 data generation requires a pm-v1.5 config")
    generation_cfg = pm_config["data_generation"]
    generation_pricing = dict(generation_cfg["pricing_usd_per_mtok"])
    if set(generation_pricing) != {"input", "output"}:
        raise ValueError("data_generation pricing must contain input/output")
    frozen_input_price = float(generation_pricing["input"])
    frozen_output_price = float(generation_pricing["output"])
    if (frozen_input_price, frozen_output_price) != (0.15, 0.60):
        raise ValueError("PM-v2 gpt-4o-mini generation pricing must be 0.15/0.60")
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
    api_cost_cfg = dict(pm_config["api_cost_planning"])
    if set(api_cost_cfg) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning keys do not match the PM-v2 contract")
    input_token_safety_factor = float(api_cost_cfg["input_token_safety_factor"])
    fail_on_reported_input_overrun = bool(
        api_cost_cfg["fail_on_reported_input_overrun"]
    )
    if input_token_safety_factor < 1.0:
        raise ValueError("input_token_safety_factor must be at least 1")
    if not fail_on_reported_input_overrun:
        raise ValueError("PM-v2 requires fail_on_reported_input_overrun=true")
    retrieval_cfg = pm_config["retrieval"]
    external_cfg = pm_config["external_evaluation"]
    strategy_top_k = int(retrieval_cfg["strategy_top_k"])
    if int(external_cfg["strategy_top_k"]) != strategy_top_k:
        raise ValueError(
            "development/external strategy_top_k mismatch in PM-v2 config"
        )
    strategy_estimated_tokens = int(external_cfg["strategy_action_tokens"])
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(args.strategy_bank)
    ]
    if not strategy_cards:
        raise ValueError("strategy bank is empty")
    strategy_catalog_count = len(strategy_cards)
    strategy_bank_sha256 = sha256_file(args.strategy_bank)
    frozen_generation_values = {
        "endpoint": str(generation_cfg["generator_endpoint"]),
        "train_users": int(generation_cfg["train_users"]),
        "calibration_users": int(generation_cfg["calibration_users"]),
        "internal_test_users": int(generation_cfg["internal_test_users"]),
        "max_generation_attempts": int(generation_cfg["max_generation_attempts"]),
        "pricing_usd_per_mtok": {
            "input": frozen_input_price,
            "output": frozen_output_price,
        },
        "seed": int(generation_cfg["base_seed"]),
    }
    requested_generation_values = {
        "endpoint": args.endpoint,
        "train_users": args.train_users,
        "calibration_users": args.calibration_users,
        "internal_test_users": args.internal_test_users,
        "max_generation_attempts": args.max_generation_attempts,
        "pricing_usd_per_mtok": {
            "input": args.input_usd_per_mtok,
            "output": args.output_usd_per_mtok,
        },
        "seed": args.seed,
    }
    for name, requested in requested_generation_values.items():
        if requested is not None and requested != frozen_generation_values[name]:
            raise RuntimeError(
                f"{name} override differs from the PM-v2 config; data-generation "
                "protocol changes must be reviewed in YAML before dry-run"
            )
    args.endpoint = frozen_generation_values["endpoint"]
    args.train_users = frozen_generation_values["train_users"]
    args.calibration_users = frozen_generation_values["calibration_users"]
    args.internal_test_users = frozen_generation_values["internal_test_users"]
    args.max_generation_attempts = frozen_generation_values[
        "max_generation_attempts"
    ]
    args.seed = frozen_generation_values["seed"]
    required_regimes = {str(value) for value in generation_cfg["required_regimes"]}
    if required_regimes != {regime.value for regime in ResourceNeedRegime}:
        raise ValueError("PM-v2 config required_regimes does not match code contract")
    if int(generation_cfg["cases_per_user"]) != len(ResourceNeedRegime):
        raise ValueError("PM-v2 config cases_per_user must equal the regime count")
    if min(args.train_users, args.calibration_users, args.internal_test_users) < 0:
        raise ValueError("user counts must be non-negative")
    if args.max_users is not None and args.max_users < 0:
        raise ValueError("max-users must be non-negative")
    if args.run and args.max_users is not None:
        raise RuntimeError(
            "API generation forbids --max-users: a partial synthetic bundle cannot "
            "satisfy the frozen train/calibration/internal-test design"
        )
    if args.max_generation_attempts < 1:
        raise ValueError("max-generation-attempts must be positive")

    seeds = read_seed_dialogues(args.seed_dialogues)
    experiment_config = load_config(args.config)
    endpoint = endpoint_from_config(experiment_config, args.endpoint)
    split_specs = [
        (PMV2Split.TRAIN, args.train_users, TRAIN_SEMANTIC_FAMILIES),
        (
            PMV2Split.CALIBRATION,
            args.calibration_users,
            CALIBRATION_SEMANTIC_FAMILIES,
        ),
        (
            PMV2Split.INTERNAL_TEST,
            args.internal_test_users,
            INTERNAL_TEST_SEMANTIC_FAMILIES,
        ),
    ]
    split_by_user: dict[str, PMV2Split] = {}
    family_by_user: dict[str, list[str]] = {}
    planned_users: list[str] = []
    global_index = 0
    for split, count, family_pool in split_specs:
        cohorts = SEMANTIC_FAMILY_COHORTS_BY_SPLIT[split.value]
        family_schedule = generation_family_schedule(split.value, count)
        cohort_union = {family for cohort in cohorts for family in cohort}
        if cohort_union != set(family_pool):
            raise RuntimeError(
                f"semantic-family cohorts do not exactly cover {split.value}: "
                f"expected={sorted(family_pool)}, observed={sorted(cohort_union)}"
            )
        if any(len(cohort) != 3 or len(set(cohort)) != 3 for cohort in cohorts):
            raise RuntimeError(
                f"semantic-family cohorts for {split.value} must be unique triads"
            )
        for local_index, scheduled_families in enumerate(family_schedule):
            if args.max_users is not None and global_index >= args.max_users:
                break
            user_id = f"pmv2_{split.value}_u{local_index + 1:03d}"
            families = list(scheduled_families)
            split_by_user[user_id] = split
            family_by_user[user_id] = families
            planned_users.append(user_id)
            global_index += 1

    code_paths = [
        Path(__file__).resolve(),
        ROOT / "src" / "metacom_pm" / "pm_v2_data.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_contracts.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_generation_pilot.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_generation_review_v8.py",
        ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        ROOT / "src" / "metacom_pm" / "api.py",
        ROOT / "src" / "metacom_pm" / "config.py",
        ROOT / "src" / "metacom_pm" / "io.py",
        ROOT / "src" / "metacom_pm" / "text.py",
    ]
    code_manifest = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
    }
    endpoint_descriptor = {
        "endpoint_name": args.endpoint,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "api_key_env": endpoint.api_key_env,
        "timeout_seconds": endpoint.timeout_seconds,
        "temperature": GENERATION_TEMPERATURE,
        "max_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
        # One HTTP request per high-level attempt makes the accepted maximum
        # enforceable by this script instead of hiding client-internal retries.
        "request_retries": GENERATION_REQUEST_RETRIES,
        "max_generation_attempts": args.max_generation_attempts,
    }
    user_plan = [
        {
            "user_id": user_id,
            "split": split_by_user[user_id].value,
            "semantic_families": family_by_user[user_id],
            "seed_dialogue_index": index % len(seeds),
            "seed_dialogue_sha256": sha256_text(seeds[index % len(seeds)]),
        }
        for index, user_id in enumerate(planned_users)
    ]
    data_plan = {
        "requested_user_counts": {
            "train": args.train_users,
            "calibration": args.calibration_users,
            "internal_test": args.internal_test_users,
        },
        "max_users": args.max_users,
        "users": user_plan,
    }
    full_user_count = (
        int(args.train_users)
        + int(args.calibration_users)
        + int(args.internal_test_users)
    )
    generation_compatibility_contract = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=endpoint,
        base_generation_seed=int(args.seed),
        full_user_count=full_user_count,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
    )
    generation_binding = {
        "protocol": "pm_v2_generation_resume_binding_v2_attempt_ledger",
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "api_cost_planning": {
            "input_token_safety_factor": input_token_safety_factor,
            "fail_on_reported_input_overrun": fail_on_reported_input_overrun,
        },
        "experiment_config_sha256": sha256_file(args.config),
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "seed_dialogues_sha256": sha256_file(args.seed_dialogues),
        "strategy_catalog": {
            "path": str(args.strategy_bank.resolve()),
            "sha256": strategy_bank_sha256,
            "count": strategy_catalog_count,
            "estimated_action_tokens": strategy_estimated_tokens,
            "top_k": strategy_top_k,
        },
        "base_generation_seed": args.seed,
        "generator": endpoint_descriptor,
        "generator_config_sha256": sha256_text(canonical_json(endpoint_descriptor)),
        "prompt_contract_sha256": generation_contract_hash(),
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
        "data_plan_sha256": sha256_text(canonical_json(data_plan)),
        "generation_compatibility_contract_version": (
            generation_compatibility_contract["version"]
        ),
        "generation_compatibility_contract_sha256": (
            generation_compatibility_contract["contract_sha256"]
        ),
    }
    generation_binding_sha256 = sha256_text(canonical_json(generation_binding))

    work_path = args.out_dir / "_generated_bundles_work.jsonl"
    error_path = args.out_dir / "_generation_errors.jsonl"
    attempt_ledger_path = args.out_dir / "_generation_physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        attempt_ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 synthetic bundle generation",
    )
    if args.overwrite:
        for path in (work_path, error_path, attempt_ledger_path):
            if path.exists():
                path.unlink()
    existing = _strict_existing_bundles(work_path)
    unexpected_existing = sorted(set(existing) - set(planned_users))
    if unexpected_existing:
        raise RuntimeError(
            "resumable work file contains users outside the active immutable data "
            f"plan: {unexpected_existing}; use a new output directory or --overwrite"
        )
    all_user_attempts: dict[str, dict[str, Any]] = {}
    for index, user_id in enumerate(planned_users):
        seed_dialogue = seeds[index % len(seeds)]
        messages = generation_messages(
            seed_dialogue=seed_dialogue,
            user_id=user_id,
            semantic_families=family_by_user[user_id],
            regimes=list(ResourceNeedRegime),
        )
        attempts = []
        for attempt in range(args.max_generation_attempts):
            attempt_seed = args.seed + index * 100 + attempt
            # Include the strict JSON schema and all request controls in the
            # provider-neutral input estimate.  Estimating messages alone would
            # understate structured-output generation prompts.
            request_contract = chat_request_payload(
                endpoint,
                messages,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
                seed=attempt_seed,
                response_schema=GeneratedBundleDraft,
            )
            request_contract_sha256 = sha256_text(canonical_json(request_contract))
            request_contract_json = canonical_json(request_contract)
            prompt_sha256 = sha256_text(canonical_json(messages))
            record_ids = {
                "user_id": user_id,
                "generation_seed": attempt_seed,
                "attempt_index": attempt + 1,
            }
            call_key = physical_call_key(
                stage="pm_v2_synthetic_bundle_generation",
                record_ids=record_ids,
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters={
                    "temperature": GENERATION_TEMPERATURE,
                    "max_tokens": GENERATION_MAX_OUTPUT_TOKENS,
                    "seed": attempt_seed,
                    "response_schema": GeneratedBundleDraft.__name__,
                    "request_contract_sha256": request_contract_sha256,
                },
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "request_contract_sha256": request_contract_sha256,
                    "call_key": call_key,
                    "raw_estimated_input_tokens": estimate_tokens(
                        request_contract_json
                    ),
                    "estimated_input_tokens": conservative_token_bound(
                        request_contract_json,
                        safety_factor=input_token_safety_factor,
                    ),
                }
            )
        all_user_attempts[user_id] = {
            "user_id": user_id,
            "split": split_by_user[user_id].value,
            "semantic_families": family_by_user[user_id],
            "seed_dialogue_index": index % len(seeds),
            "seed_dialogue_sha256": sha256_text(seed_dialogue),
            "prompt_sha256": sha256_text(canonical_json(messages)),
            "maximum_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
            "maximum_attempts": args.max_generation_attempts,
            "attempts": attempts,
        }

    all_call_keys = [
        str(attempt["call_key"])
        for row in all_user_attempts.values()
        for attempt in row["attempts"]
    ]
    if len(all_call_keys) != len(set(all_call_keys)):
        raise RuntimeError("synthetic generation produced duplicate physical call keys")
    ledger = PersistentAttemptLedger(
        attempt_ledger_path,
        stage="pm_v2_synthetic_bundle_generation",
        expected_calls={call_key: 1 for call_key in all_call_keys},
        maximum_total_attempts=max(len(all_call_keys), 1),
    )
    recovered_successful_bundles = _recover_successful_bundles_from_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        existing=existing,
        work_path=work_path,
        generation_binding=generation_binding,
        family_by_user=family_by_user,
    )
    for user_id, bundle in existing.items():
        require_bundle_generation_binding(bundle, generation_binding)
        strict_bundle_check(bundle, family_by_user[user_id])
        physical_call_key_value = str(
            bundle.provenance.get("physical_call_key") or ""
        )
        valid_call_keys = {
            str(row["call_key"])
            for row in all_user_attempts[user_id]["attempts"]
        }
        if (
            physical_call_key_value not in valid_call_keys
            or not ledger.succeeded(physical_call_key_value)
        ):
            raise RuntimeError(
                f"resumable bundle lacks a successful physical-attempt binding: {user_id}"
            )

    pending_users = [user_id for user_id in planned_users if user_id not in existing]
    call_plan: list[dict[str, Any]] = []
    blocked_pending_users: list[dict[str, Any]] = []
    for user_id in pending_users:
        row = all_user_attempts[user_id]
        successful = [
            attempt
            for attempt in row["attempts"]
            if ledger.succeeded(str(attempt["call_key"]))
        ]
        remaining_attempts = [
            attempt
            for attempt in row["attempts"]
            if not ledger.exhausted(str(attempt["call_key"]))
        ]
        if successful or not remaining_attempts:
            blocked_pending_users.append(
                {
                    "user_id": user_id,
                    "reason": (
                        "successful_attempt_missing_bundle"
                        if successful
                        else "maximum_generation_attempts_exhausted"
                    ),
                    "historical_attempts": sum(
                        ledger.attempts_for(str(attempt["call_key"]))
                        for attempt in row["attempts"]
                    ),
                }
            )
            continue
        call_plan.append(
            {
                **{key: value for key, value in row.items() if key != "attempts"},
                "estimated_input_tokens": remaining_attempts[0][
                    "estimated_input_tokens"
                ],
                "maximum_estimated_input_tokens": max(
                    int(attempt["estimated_input_tokens"])
                    for attempt in remaining_attempts
                ),
                "remaining_attempts": len(remaining_attempts),
                "attempts": remaining_attempts,
            }
        )

    input_tokens = [int(row["estimated_input_tokens"]) for row in call_plan]
    expected_input_tokens = sum(input_tokens)
    potential_input_tokens = [
        int(attempt["estimated_input_tokens"])
        for row in call_plan
        for attempt in row["attempts"]
    ]
    maximum_input_tokens = sum(potential_input_tokens)
    expected_output_tokens = len(call_plan) * GENERATION_MAX_OUTPUT_TOKENS
    maximum_output_tokens = (
        len(potential_input_tokens) * GENERATION_MAX_OUTPUT_TOKENS
    )
    expected_cost = (
        expected_input_tokens / 1_000_000 * args.input_usd_per_mtok
        + expected_output_tokens / 1_000_000 * args.output_usd_per_mtok
    )
    maximum_cost = (
        maximum_input_tokens / 1_000_000 * args.input_usd_per_mtok
        + maximum_output_tokens / 1_000_000 * args.output_usd_per_mtok
    )
    completed_bundle_hashes = {
        user_id: sha256_text(canonical_json(existing[user_id].model_dump(mode="json")))
        for user_id in sorted(existing)
    }
    estimate: dict[str, Any] = {
        "protocol": GENERATION_COST_PROTOCOL,
        "input_token_safety_factor": input_token_safety_factor,
        "fail_on_reported_input_overrun": fail_on_reported_input_overrun,
        "generation_run_binding_sha256": generation_binding_sha256,
        "generation_run_binding": generation_binding,
        "completed_bundle_hashes": completed_bundle_hashes,
        "completed_bundle_manifest_sha256": sha256_text(
            canonical_json(completed_bundle_hashes)
        ),
        "planned_users": len(planned_users),
        "completed_users": len(existing),
        "pending_users": len(pending_users),
        "blocked_pending_users": blocked_pending_users,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "physical_attempt_ledger_sha256": (
            sha256_file(attempt_ledger_path)
            if attempt_ledger_path.exists()
            else sha256_text("")
        ),
        "historical_physical_api_attempts": ledger.started_attempts,
        "recovered_successful_bundles_from_ledger": recovered_successful_bundles,
        # Exact one-attempt success-path count plus an accepted hard upper bound.
        "expected_api_calls": len(call_plan),
        "maximum_api_calls": len(potential_input_tokens),
        "maximum_new_physical_api_attempts": len(potential_input_tokens),
        "maximum_physical_api_attempts_including_history": (
            ledger.started_attempts + len(potential_input_tokens)
        ),
        "expected_total_input_tokens": expected_input_tokens,
        "maximum_total_input_tokens": maximum_input_tokens,
        "expected_maximum_output_tokens": expected_output_tokens,
        "maximum_total_output_tokens": maximum_output_tokens,
        "mean_input_tokens": (
            float(expected_input_tokens / len(input_tokens)) if input_tokens else 0.0
        ),
        "max_input_tokens": max(potential_input_tokens, default=0),
        "pricing": {
            "input_usd_per_mtok": args.input_usd_per_mtok,
            "output_usd_per_mtok": args.output_usd_per_mtok,
        },
        "expected_estimated_cost_usd": expected_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
    }
    estimate["budget_gate"] = _budget_gate(
        estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    estimate["cost_estimate_sha256"] = sha256_text(canonical_json(estimate))

    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "generation_cost_estimate.json", estimate)
        write_jsonl(args.out_dir / "generation_call_plan.jsonl", call_plan)
        print({**estimate, "status": "DRY_RUN"})
        if estimate["budget_gate"]["status"] != "PASS":
            raise RuntimeError("generation dry-run failed the frozen budget gate")
        return

    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("generation API run blocked by the budget gate")
    _require_saved_dry_run(args.out_dir, estimate)
    expected_hash = str(estimate["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match the current generation plan"
        )

    generation_pilot_verification = require_generation_compatibility_attestation(
        args.generation_pilot_attestation,
        expected_contract=generation_compatibility_contract,
    )
    automated_review_verification = require_automated_semantic_review_pass(
        args.automated_semantic_review_report,
        args.automated_semantic_review_attestation,
    )
    automated_review_report = automated_review_verification["report"]
    generation_pilot_semantic_verification = {
        "protocol": "pm-v1.5-generation-semantic-review-v8-replaced-by-automated-review",
        "status": "PASS_VIA_AUTOMATED_MULTI_FAMILY_REVIEW",
        "human_calibration_performed": False,
        "automated_review_report_sha256": sha256_text(
            canonical_json(automated_review_report)
        ),
        "automated_review_attestation_sha256": automated_review_verification[
            "attestation_sha256"
        ],
        "automated_review_judge_families": automated_review_report.get("judge_families"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve credentials before reserving the first paid attempt.  Every
    # reservation after this point is fsynced before its HTTP request.
    if call_plan:
        _ = endpoint.api_key
    plan_by_user = {str(row["user_id"]): row for row in call_plan}
    historical_api_calls = ledger.started_attempts
    api_calls_used = 0
    for index, user_id in enumerate(planned_users):
        if user_id in existing:
            continue
        user_call_plan = plan_by_user.get(user_id)
        if user_call_plan is None:
            raise RuntimeError(
                f"pending synthetic user has no remaining authorized attempt: {user_id}"
            )
        last_error = None
        for attempt in user_call_plan["attempts"]:
            if ledger.started_attempts >= args.max_api_calls:
                raise RuntimeError("generation API call cap exhausted before completion")
            record_ids = {
                "user_id": user_id,
                "generation_seed": int(attempt["seed"]),
                "attempt_index": int(attempt["attempt"]),
            }
            reservation = ledger.reserve(
                str(attempt["call_key"]),
                record_ids=record_ids,
                prompt_sha256=str(user_call_plan["prompt_sha256"]),
            )
            api_calls_used += 1
            bundle = None
            try:
                bundle = generate_user_bundle(
                    endpoint=endpoint,
                    seed_dialogue=seeds[index % len(seeds)],
                    user_id=user_id,
                    semantic_families=family_by_user[user_id],
                    regimes=list(ResourceNeedRegime),
                    seed=int(attempt["seed"]),
                    request_retries=GENERATION_REQUEST_RETRIES,
                )
                reported_usage = _require_bundle_reported_usage(bundle)
                reported_prompt_tokens = reported_usage["prompt_tokens"]
                planned_input_bound = int(attempt["estimated_input_tokens"])
                if (
                    fail_on_reported_input_overrun
                    and reported_prompt_tokens > planned_input_bound
                ):
                    raise ReportedInputTokenOverrun(
                        "reported prompt_tokens exceed the frozen conservative "
                        f"bound: reported={reported_prompt_tokens}, "
                        f"bound={planned_input_bound}, user_id={user_id}, "
                        f"attempt={attempt['attempt']}"
                    )
                bundle.provenance.update(
                    {
                        "physical_call_key": reservation.call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                    }
                )
                bind_bundle_to_generation_run(bundle, generation_binding)
                strict_bundle_check(bundle, family_by_user[user_id])
            except StructuredOutputValidationError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                usage = require_reported_usage(
                    exc.call.usage,
                    stage="failed PM-v2 full generation structured response",
                )
                failure_result = {
                    "provider_response": exc.call.raw_response,
                    "parsed_payload": exc.parsed_payload,
                    "validation_errors": exc.validation_errors,
                }
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=exc.call.request_hash,
                    usage=usage,
                    error=last_error,
                    result=failure_result,
                )
                append_jsonl(
                    error_path,
                    {
                        "user_id": user_id,
                        "attempt": int(attempt["attempt"]),
                        "generation_seed": int(attempt["seed"]),
                        "physical_call_key": reservation.call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                        "error": last_error,
                        "provider_response_preserved": True,
                        "validation_errors": exc.validation_errors,
                        "generation_run_binding_sha256": generation_binding_sha256,
                        "accepted_cost_estimate_sha256": expected_hash,
                    },
                )
                continue
            except GenerationDraftCompilationError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                usage = require_reported_usage(
                    exc.call.usage,
                    stage="failed PM-v2 full generation draft compilation",
                )
                failure_result = {
                    "provider_response": exc.call.raw_response,
                    "parsed_payload": exc.parsed_payload,
                    "compilation_error": exc.compilation_error,
                }
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=exc.call.request_hash,
                    usage=usage,
                    error=last_error,
                    result=failure_result,
                )
                append_jsonl(
                    error_path,
                    {
                        "user_id": user_id,
                        "attempt": int(attempt["attempt"]),
                        "generation_seed": int(attempt["seed"]),
                        "physical_call_key": reservation.call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                        "error": last_error,
                        "provider_response_preserved": True,
                        "compilation_error": exc.compilation_error,
                        "generation_run_binding_sha256": generation_binding_sha256,
                        "accepted_cost_estimate_sha256": expected_hash,
                    },
                )
                continue
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _finish_generation_attempt(
                    ledger,
                    reservation,
                    bundle=bundle,
                    succeeded=False,
                    error=last_error,
                )
                append_jsonl(
                    error_path,
                    {
                        "user_id": user_id,
                        "attempt": int(attempt["attempt"]),
                        "generation_seed": int(attempt["seed"]),
                        "physical_call_key": reservation.call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                        "error": last_error,
                        "generation_run_binding_sha256": generation_binding_sha256,
                        "accepted_cost_estimate_sha256": expected_hash,
                    },
                )
                if isinstance(exc, ReportedInputTokenOverrun):
                    raise RuntimeError(last_error) from exc
                continue
            _finish_generation_attempt(
                ledger,
                reservation,
                bundle=bundle,
                succeeded=True,
                error=None,
            )
            append_jsonl(work_path, bundle.model_dump(mode="json"))
            existing[user_id] = bundle
            last_error = None
            break
        if last_error is not None:
            raise RuntimeError(
                f"failed to generate valid bundle for {user_id} after "
                f"its remaining frozen attempts: {last_error}"
            )
    bundles = [existing[user_id] for user_id in planned_users]
    report = write_development_dataset(
        bundles=bundles,
        split_by_user=split_by_user,
        out_dir=args.out_dir,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        strategy_top_k=strategy_top_k,
        strategy_bank_sha256=strategy_bank_sha256,
        expected_semantic_families_by_split={
            PMV2Split.TRAIN: TRAIN_SEMANTIC_FAMILIES,
            PMV2Split.CALIBRATION: CALIBRATION_SEMANTIC_FAMILIES,
            PMV2Split.INTERNAL_TEST: INTERNAL_TEST_SEMANTIC_FAMILIES,
        },
    )
    enforce_full_state_design(
        report,
        train_users=args.train_users,
        calibration_users=args.calibration_users,
        internal_test_users=args.internal_test_users,
    )
    generated_states = load_states(args.out_dir / "pm_v2_states.jsonl")
    states_by_split = {
        split: [state for state in generated_states if state.split == split]
        for split in (
            PMV2Split.TRAIN,
            PMV2Split.CALIBRATION,
            PMV2Split.INTERNAL_TEST,
        )
    }
    near_duplicate_cfg = pm_config["splits"]["near_duplicate_audit"]
    if near_duplicate_cfg.get("method") != (
        "fixed_hash_word_1_2_and_char_3_5_cosine"
    ):
        raise ValueError("unsupported PM-v2 near-duplicate audit method")
    near_duplicate_audit = audit_cross_split_near_duplicates(
        states_by_split,
        maximum_word_hash_cosine=float(
            near_duplicate_cfg["maximum_word_hash_cosine"]
        ),
        maximum_char_hash_cosine=float(
            near_duplicate_cfg["maximum_char_hash_cosine"]
        ),
        report_top_pairs=int(near_duplicate_cfg["report_top_pairs"]),
    )
    report.update(
        {
            "work_path": str(work_path),
            "error_path": str(error_path),
            "pm_v2_config": str(args.pm_v2_config),
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "seed_dialogues": str(args.seed_dialogues),
            "seed_dialogues_sha256": sha256_file(args.seed_dialogues),
            "generation_run_binding": generation_binding,
            "generation_run_binding_sha256": generation_binding_sha256,
            "accepted_cost_estimate_sha256": expected_hash,
            "generation_api_calls_used": api_calls_used,
            "generation_historical_api_calls": historical_api_calls,
            "generation_total_physical_api_attempts": ledger.started_attempts,
            "generation_attempt_ledger_path": str(attempt_ledger_path),
            "generation_attempt_ledger_sha256": sha256_file(attempt_ledger_path),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "generation_compatibility_pilot": generation_pilot_verification,
            "generation_pilot_semantic_review": (
                generation_pilot_semantic_verification
            ),
            "generation_expected_api_calls": estimate["expected_api_calls"],
            "generation_maximum_api_calls": estimate["maximum_api_calls"],
            "cross_split_near_duplicate_audit": near_duplicate_audit,
            "all_states_have_16_actions": all(
                len(state.allowed_actions) == 16 for state in generated_states
            ),
        }
    )
    if not report["all_states_have_16_actions"]:
        raise RuntimeError("generated PM-v2 dataset does not expose all 16 actions")
    write_json(args.out_dir / "pm_v2_data_report.json", report)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage="pm_v1_5_development_data",
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v2_config,
            "seed_dialogues": args.seed_dialogues,
            "strategy_bank": args.strategy_bank,
            "generation_pilot_attestation": args.generation_pilot_attestation,
            "automated_semantic_review": args.automated_semantic_review_report,
            "automated_semantic_review_attestation": (
                args.automated_semantic_review_attestation
            ),
            "cost_estimate": args.out_dir / "generation_cost_estimate.json",
            "call_plan": args.out_dir / "generation_call_plan.jsonl",
        },
        outputs={
            "states": (args.out_dir / "pm_v2_states.jsonl", True),
            "runtime": (args.out_dir / "runtime_states.jsonl", True),
            "backend": (args.out_dir / "memory_backend.jsonl", True),
            "evaluator_contexts": (
                args.out_dir / "evaluator_contexts.jsonl",
                True,
            ),
            "data_report": (args.out_dir / "pm_v2_data_report.json", False),
            "physical_attempt_ledger": (attempt_ledger_path, True),
        },
        parameters={
            "track": "pm-v1.5",
            "generation_run_binding_sha256": generation_binding_sha256,
            "accepted_cost_estimate_sha256": expected_hash,
            "human_semantic_review_performed": False,
        },
        expected={
            "states": len(generated_states),
            "users": len(planned_users),
            "physical_attempts": ledger.started_attempts,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
