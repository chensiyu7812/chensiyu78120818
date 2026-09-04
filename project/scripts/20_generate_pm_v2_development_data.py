#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.api import (
    CallResult,
    StructuredOutputValidationError,
    chat_request_payload,
    make_client,
    require_reported_usage,
)
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
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
    GENERATION_CASE_FIELDS,
    GENERATION_TEMPERATURE,
    SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
    SURFACE_GENERATION_MAX_REPAIRS,
    GeneratedSurfaceOnlyCaseDraft,
    GeneratedUserBundle,
    audit_cross_split_near_duplicates,
    compile_surface_only_user_bundle,
    bind_bundle_to_generation_run,
    generation_case_family_assignments,
    generation_case_messages,
    lint_generation_surface_case,
    load_bundles,
    load_states,
    require_bundle_generation_binding,
    surface_generation_contract_hash,
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
from metacom_pm.pm_v2_generation_review_v8 import (
    require_generation_semantic_review_v8,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens

ROOT = Path(__file__).resolve().parents[1]

GENERATION_REQUEST_RETRIES = 1
GENERATION_COST_PROTOCOL = (
    "pm_v2_generation_cost_v3_casewise_surface_bounded_repair_ledger"
)
GENERATION_STAGE = "pm_v2_synthetic_surface_generation"


class ReportedInputTokenOverrun(RuntimeError):
    pass


def _load_successful_surface_attempt(
    *,
    ledger: PersistentAttemptLedger,
    case_plan: dict[str, Any],
) -> tuple[dict[str, Any], GeneratedSurfaceOnlyCaseDraft, CallResult] | None:
    successful = [
        attempt
        for attempt in case_plan["attempts"]
        if ledger.succeeded(str(attempt["call_key"]))
    ]
    if len(successful) > 1:
        raise RuntimeError(
            "casewise generation has multiple successful calls for "
            f"{case_plan['user_id']}/{case_plan['case_field']}"
        )
    if not successful:
        return None
    attempt = successful[0]
    terminal = ledger.terminal_row(str(attempt["call_key"])) or {}
    result = terminal.get("result") or {}
    surface_payload = result.get("surface")
    provider_response = result.get("provider_response")
    if not isinstance(surface_payload, dict) or not isinstance(
        provider_response, dict
    ):
        raise RuntimeError("successful surface attempt lacks recoverable trace")
    usage = require_reported_usage(
        terminal.get("usage"), stage="recovered PM-v2 surface generation"
    )
    request_hash = str(terminal.get("request_hash") or "")
    if not request_hash:
        raise RuntimeError("successful surface attempt lacks request hash")
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(surface_payload)
    call = CallResult(
        text=canonical_json(surface_payload),
        raw_response=provider_response,
        usage=usage,
        latency_ms=0.0,
        request_hash=request_hash,
    )
    return attempt, surface, call


def _compile_casewise_user_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    user_plan: dict[str, Any],
    seed_dialogue: str,
    endpoint,
    generation_binding: dict[str, Any],
) -> GeneratedUserBundle | None:
    surfaces: dict[str, GeneratedSurfaceOnlyCaseDraft] = {}
    calls: dict[str, CallResult] = {}
    messages: dict[str, list[dict[str, str]]] = {}
    attempt_kinds: dict[str, str] = {}
    accepted_call_keys: dict[str, str] = {}
    for case_plan in user_plan["cases"]:
        loaded = _load_successful_surface_attempt(
            ledger=ledger, case_plan=case_plan
        )
        if loaded is None:
            return None
        attempt, surface, call = loaded
        case_field = str(case_plan["case_field"])
        surfaces[case_field] = surface
        calls[case_field] = call
        messages[case_field] = attempt["messages"]
        attempt_kinds[case_field] = str(attempt["attempt_kind"])
        accepted_call_keys[case_field] = str(attempt["call_key"])
    bundle = compile_surface_only_user_bundle(
        surfaces=surfaces,
        accepted_calls=calls,
        accepted_messages=messages,
        accepted_attempt_kinds=attempt_kinds,
        seed_dialogue=seed_dialogue,
        user_id=str(user_plan["user_id"]),
        semantic_families=[str(value) for value in user_plan["semantic_families"]],
        regimes=list(ResourceNeedRegime),
        generator_model=endpoint.model,
        generator_family=endpoint.family,
    )
    bundle.provenance.update(
        {
            "accepted_surface_call_keys": accepted_call_keys,
            "all_physical_attempt_usage": _ledger_usage_for_keys(
                ledger,
                [
                    str(attempt["call_key"])
                    for case_plan in user_plan["cases"]
                    for attempt in case_plan["attempts"]
                ],
            ),
        }
    )
    bind_bundle_to_generation_run(bundle, generation_binding)
    return bundle


def _ledger_usage_for_keys(
    ledger: PersistentAttemptLedger, call_keys: list[str]
) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for call_key in call_keys:
        terminal = ledger.terminal_row(call_key)
        if terminal is None or terminal.get("usage") is None:
            continue
        usage = require_reported_usage(
            terminal.get("usage"), stage="completed PM-v2 surface call"
        )
        for key in totals:
            totals[key] += usage[key]
    return totals


def _recover_casewise_bundles_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    all_user_attempts: dict[str, dict[str, Any]],
    existing: dict[str, GeneratedUserBundle],
    work_path: Path,
    generation_binding: dict[str, Any],
    family_by_user: dict[str, list[str]],
    seeds: list[str],
    endpoint,
) -> int:
    recovered = 0
    for user_id, user_plan in all_user_attempts.items():
        if user_id in existing:
            continue
        bundle = _compile_casewise_user_from_ledger(
            ledger=ledger,
            user_plan=user_plan,
            seed_dialogue=seeds[int(user_plan["seed_dialogue_index"])],
            endpoint=endpoint,
            generation_binding=generation_binding,
        )
        if bundle is None:
            continue
        strict_bundle_check(bundle, family_by_user[user_id])
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
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument("--endpoint")
    parser.add_argument("--seed-dialogues", type=Path, required=True)
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "pm_v2")
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
    parser.add_argument("--max-api-calls", type=int, default=1000)
    parser.add_argument("--max-estimated-usd", type=float, default=25.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_generation_compatibility_pilot_surface_v8_2"
            / "artifact_attestation.json"
        ),
        help="Required PASS casewise surface-only pilot for full --run only.",
    )
    parser.add_argument(
        "--generation-pilot-semantic-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v2_generation_pilot_semantic_review_v8_validation"
            / "artifact_attestation.json"
        ),
        help=(
            "Required PASS two-independent-annotator 27-case V8 validation "
            "review for full --run only; a 9-case smoke attestation is rejected."
        ),
    )
    args = parser.parse_args()

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
    if pm_config.get("version") != "pm-v2.2":
        raise ValueError("unsupported PM-v2 config version")
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
    if args.max_generation_attempts != 1 + SURFACE_GENERATION_MAX_REPAIRS:
        raise ValueError(
            "max-generation-attempts must equal initial plus the single frozen repair"
        )

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
        "max_output_tokens": SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
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
        "protocol": "pm_v2_generation_resume_binding_v3_casewise_surface_ledger",
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
        "prompt_contract_sha256": surface_generation_contract_hash(),
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
    for user_index, user_id in enumerate(planned_users):
        seed_dialogue_index = user_index % len(seeds)
        seed_dialogue = seeds[seed_dialogue_index]
        families = family_by_user[user_id]
        assignments = generation_case_family_assignments(
            families, list(ResourceNeedRegime)
        )
        cases: list[dict[str, Any]] = []
        for case_index, (case_field, regime) in enumerate(
            GENERATION_CASE_FIELDS
        ):
            semantic_family = assignments[case_field]
            forbidden_families = [
                family for family in families if family != semantic_family
            ]
            attempts: list[dict[str, Any]] = []
            for attempt_index in range(args.max_generation_attempts):
                attempt_kind = "initial" if attempt_index == 0 else "repair"
                messages = generation_case_messages(
                    seed_dialogue=seed_dialogue,
                    user_id=user_id,
                    case_field=case_field,
                    regime=regime,
                    semantic_family=semantic_family,
                    forbidden_families=forbidden_families,
                    repair=attempt_kind == "repair",
                )
                attempt_seed = (
                    int(args.seed)
                    + user_index * 1000
                    + case_index * 10
                    + attempt_index
                )
                request_contract = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                    seed=attempt_seed,
                    response_schema=GeneratedSurfaceOnlyCaseDraft,
                )
                request_contract_json = canonical_json(request_contract)
                request_contract_sha256 = sha256_text(request_contract_json)
                prompt_sha256 = sha256_text(canonical_json(messages))
                record_ids = {
                    "user_id": user_id,
                    "case_field": case_field,
                    "attempt_kind": attempt_kind,
                    "generation_seed": attempt_seed,
                }
                call_key = physical_call_key(
                    stage=GENERATION_STAGE,
                    record_ids=record_ids,
                    prompt_sha256=prompt_sha256,
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": GENERATION_TEMPERATURE,
                        "max_tokens": SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                        "seed": attempt_seed,
                        "response_schema": GeneratedSurfaceOnlyCaseDraft.__name__,
                        "request_contract_sha256": request_contract_sha256,
                    },
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "attempt_kind": attempt_kind,
                        "seed": attempt_seed,
                        "messages": messages,
                        "prompt_sha256": prompt_sha256,
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
            cases.append(
                {
                    "user_id": user_id,
                    "case_field": case_field,
                    "regime": regime.value,
                    "semantic_family": semantic_family,
                    "attempts": attempts,
                }
            )
        all_user_attempts[user_id] = {
            "user_id": user_id,
            "split": split_by_user[user_id].value,
            "semantic_families": families,
            "seed_dialogue_index": seed_dialogue_index,
            "seed_dialogue_sha256": sha256_text(seed_dialogue),
            "maximum_output_tokens_per_call": (
                SURFACE_GENERATION_MAX_OUTPUT_TOKENS
            ),
            "maximum_attempts_per_case": args.max_generation_attempts,
            "cases": cases,
        }

    all_call_keys = [
        str(attempt["call_key"])
        for row in all_user_attempts.values()
        for case_plan in row["cases"]
        for attempt in case_plan["attempts"]
    ]
    if len(all_call_keys) != len(set(all_call_keys)):
        raise RuntimeError("synthetic generation produced duplicate physical call keys")
    ledger = PersistentAttemptLedger(
        attempt_ledger_path,
        stage=GENERATION_STAGE,
        expected_calls={call_key: 1 for call_key in all_call_keys},
        maximum_total_attempts=max(len(all_call_keys), 1),
    )
    recovered_successful_bundles = _recover_casewise_bundles_from_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        existing=existing,
        work_path=work_path,
        generation_binding=generation_binding,
        family_by_user=family_by_user,
        seeds=seeds,
        endpoint=endpoint,
    )
    for user_id, bundle in existing.items():
        require_bundle_generation_binding(bundle, generation_binding)
        strict_bundle_check(bundle, family_by_user[user_id])
        accepted_call_keys = bundle.provenance.get(
            "accepted_surface_call_keys"
        )
        valid_call_keys = {
            str(attempt["call_key"])
            for case_plan in all_user_attempts[user_id]["cases"]
            for attempt in case_plan["attempts"]
        }
        if (
            not isinstance(accepted_call_keys, dict)
            or set(accepted_call_keys)
            != {field for field, _ in GENERATION_CASE_FIELDS}
            or any(
                str(value) not in valid_call_keys
                or not ledger.succeeded(str(value))
                for value in accepted_call_keys.values()
            )
        ):
            raise RuntimeError(
                f"resumable bundle lacks nine successful surface bindings: {user_id}"
            )

    pending_users = [user_id for user_id in planned_users if user_id not in existing]
    call_plan: list[dict[str, Any]] = []
    blocked_pending_users: list[dict[str, Any]] = []
    for user_id in pending_users:
        user_plan = all_user_attempts[user_id]
        for case_plan in user_plan["cases"]:
            successful = [
                attempt
                for attempt in case_plan["attempts"]
                if ledger.succeeded(str(attempt["call_key"]))
            ]
            if successful:
                continue
            remaining_attempts = [
                attempt
                for attempt in case_plan["attempts"]
                if not ledger.exhausted(str(attempt["call_key"]))
            ]
            if not remaining_attempts:
                blocked_pending_users.append(
                    {
                        "user_id": user_id,
                        "case_field": case_plan["case_field"],
                        "reason": "maximum_case_attempts_exhausted",
                        "historical_attempts": sum(
                            ledger.attempts_for(str(attempt["call_key"]))
                            for attempt in case_plan["attempts"]
                        ),
                    }
                )
                continue
            public_attempts = [
                {
                    key: value
                    for key, value in attempt.items()
                    if key != "messages"
                }
                for attempt in remaining_attempts
            ]
            call_plan.append(
                {
                    "user_id": user_id,
                    "split": user_plan["split"],
                    "semantic_families": user_plan["semantic_families"],
                    "seed_dialogue_index": user_plan["seed_dialogue_index"],
                    "seed_dialogue_sha256": user_plan["seed_dialogue_sha256"],
                    "case_field": case_plan["case_field"],
                    "regime": case_plan["regime"],
                    "semantic_family": case_plan["semantic_family"],
                    "estimated_input_tokens": public_attempts[0][
                        "estimated_input_tokens"
                    ],
                    "maximum_estimated_input_tokens": max(
                        int(attempt["estimated_input_tokens"])
                        for attempt in public_attempts
                    ),
                    "remaining_attempts": len(public_attempts),
                    "attempts": public_attempts,
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
    expected_output_tokens = (
        len(call_plan) * SURFACE_GENERATION_MAX_OUTPUT_TOKENS
    )
    maximum_output_tokens = (
        len(potential_input_tokens) * SURFACE_GENERATION_MAX_OUTPUT_TOKENS
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
    generation_pilot_semantic_verification = (
        require_generation_semantic_review_v8(
            args.generation_pilot_semantic_attestation,
            require_validation=True,
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve credentials before reserving the first paid attempt.  Every
    # reservation after this point is fsynced before its HTTP request.
    if call_plan:
        _ = endpoint.api_key
    authorized_call_keys = {
        str(attempt["call_key"])
        for row in call_plan
        for attempt in row["attempts"]
    }
    historical_api_calls = ledger.started_attempts
    api_calls_used = 0
    client = make_client(endpoint) if call_plan else None
    try:
        for user_id in planned_users:
            if user_id in existing:
                continue
            user_plan = all_user_attempts[user_id]
            prior_current_user_texts: list[str] = []
            for case_plan in user_plan["cases"]:
                loaded = _load_successful_surface_attempt(
                    ledger=ledger, case_plan=case_plan
                )
                if loaded is None:
                    last_error: str | None = None
                    for attempt in case_plan["attempts"]:
                        call_key = str(attempt["call_key"])
                        if call_key not in authorized_call_keys:
                            continue
                        if ledger.started_attempts >= args.max_api_calls:
                            raise RuntimeError(
                                "generation API call cap exhausted before completion"
                            )
                        reservation = ledger.reserve(
                            call_key,
                            record_ids={
                                "user_id": user_id,
                                "case_field": case_plan["case_field"],
                                "attempt_kind": attempt["attempt_kind"],
                                "generation_seed": int(attempt["seed"]),
                            },
                            prompt_sha256=str(attempt["prompt_sha256"]),
                        )
                        api_calls_used += 1
                        try:
                            assert client is not None
                            call, surface = client.chat(
                                attempt["messages"],
                                temperature=GENERATION_TEMPERATURE,
                                max_tokens=SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                                seed=int(attempt["seed"]),
                                response_schema=GeneratedSurfaceOnlyCaseDraft,
                                retries=GENERATION_REQUEST_RETRIES,
                            )
                            assert surface is not None
                            usage = require_reported_usage(
                                call.usage,
                                stage="PM-v2 full surface generation",
                            )
                            if usage["prompt_tokens"] > int(
                                attempt["estimated_input_tokens"]
                            ):
                                result = {
                                    "surface": surface.model_dump(mode="json"),
                                    "provider_response": call.raw_response,
                                }
                                last_error = (
                                    "reported prompt_tokens exceed the frozen "
                                    "conservative bound"
                                )
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=call.request_hash,
                                    usage=usage,
                                    error=last_error,
                                    result=result,
                                )
                                raise ReportedInputTokenOverrun(last_error)
                            lint = lint_generation_surface_case(
                                case_field=str(case_plan["case_field"]),
                                regime=ResourceNeedRegime(
                                    str(case_plan["regime"])
                                ),
                                family=str(case_plan["semantic_family"]),
                                forbidden_families=[
                                    family
                                    for family in user_plan["semantic_families"]
                                    if family != case_plan["semantic_family"]
                                ],
                                surface=surface,
                                prior_current_user_texts=(
                                    prior_current_user_texts
                                ),
                            )
                            result = {
                                "surface": surface.model_dump(mode="json"),
                                "provider_response": call.raw_response,
                                "lint": lint,
                                "attempt_kind": attempt["attempt_kind"],
                            }
                            if lint["status"] != "PASS":
                                last_error = "surface lint failed: " + canonical_json(
                                    lint["errors"]
                                )
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=call.request_hash,
                                    usage=usage,
                                    error=last_error,
                                    result=result,
                                )
                                append_jsonl(
                                    error_path,
                                    {
                                        "user_id": user_id,
                                        "case_field": case_plan["case_field"],
                                        "attempt_kind": attempt["attempt_kind"],
                                        "generation_seed": int(attempt["seed"]),
                                        "physical_call_key": call_key,
                                        "error": last_error,
                                        "provider_response_preserved": True,
                                        "accepted_cost_estimate_sha256": expected_hash,
                                    },
                                )
                                continue
                            ledger.finish(
                                reservation,
                                succeeded=True,
                                request_hash=call.request_hash,
                                usage=usage,
                                error=None,
                                result=result,
                            )
                            loaded = (attempt, surface, call)
                            last_error = None
                            break
                        except StructuredOutputValidationError as exc:
                            usage = require_reported_usage(
                                exc.call.usage,
                                stage="failed PM-v2 surface schema",
                            )
                            last_error = f"{type(exc).__name__}: {exc}"
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
                                    "case_field": case_plan["case_field"],
                                    "attempt_kind": attempt["attempt_kind"],
                                    "generation_seed": int(attempt["seed"]),
                                    "physical_call_key": call_key,
                                    "error": last_error,
                                    "provider_response_preserved": True,
                                    "validation_errors": exc.validation_errors,
                                    "accepted_cost_estimate_sha256": expected_hash,
                                },
                            )
                            continue
                        except ReportedInputTokenOverrun:
                            raise
                        except Exception as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            ledger.finish(
                                reservation,
                                succeeded=False,
                                request_hash=None,
                                usage=None,
                                error=last_error,
                            )
                            raise RuntimeError(last_error) from exc
                    if loaded is None:
                        raise RuntimeError(
                            "surface generation exhausted initial+repair for "
                            f"{user_id}/{case_plan['case_field']}: {last_error}"
                        )
                _, surface, _ = loaded
                prior_current_user_texts.append(surface.current_user_text)
            bundle = _compile_casewise_user_from_ledger(
                ledger=ledger,
                user_plan=user_plan,
                seed_dialogue=seeds[int(user_plan["seed_dialogue_index"])],
                endpoint=endpoint,
                generation_binding=generation_binding,
            )
            if bundle is None:
                raise RuntimeError(f"user {user_id} lacks nine accepted surfaces")
            strict_bundle_check(bundle, family_by_user[user_id])
            append_jsonl(work_path, bundle.model_dump(mode="json"))
            existing[user_id] = bundle
    finally:
        if client is not None:
            client.close()
    bundles = [existing[user_id] for user_id in planned_users]
    report = write_development_dataset(
        bundles=bundles,
        split_by_user=split_by_user,
        out_dir=args.out_dir,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        strategy_top_k=strategy_top_k,
        strategy_bank_sha256=strategy_bank_sha256,
        strategy_cards=strategy_cards,
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
    generation_repair_cases = [
        {"user_id": bundle.user_id, "case_field": case_field}
        for bundle in bundles
        for case_field, attempt_kind in (
            bundle.provenance.get("provider_surface_attempt_kinds") or {}
        ).items()
        if attempt_kind == "repair"
    ]
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
            "generation_surface_protocol": "surface_only_casewise",
            "generation_repair_case_count": len(generation_repair_cases),
            "generation_repair_case_rate": (
                len(generation_repair_cases) / len(generated_states)
                if generated_states
                else 0.0
            ),
            "generation_repair_cases": generation_repair_cases,
            "cross_split_near_duplicate_audit": near_duplicate_audit,
            "all_states_have_16_actions": all(
                len(state.allowed_actions) == 16 for state in generated_states
            ),
        }
    )
    if not report["all_states_have_16_actions"]:
        raise RuntimeError("generated PM-v2 dataset does not expose all 16 actions")
    write_json(args.out_dir / "pm_v2_data_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
