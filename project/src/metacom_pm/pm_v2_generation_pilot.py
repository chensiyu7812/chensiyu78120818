from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Sequence

from .api import Endpoint, chat_request_payload, require_reported_usage
from .artifacts import require_artifact_attestation
from .attempt_ledger import PersistentAttemptLedger, physical_call_key
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from .pm_v2_contracts import ResourceNeedRegime
from .pm_v2_data import (
    GENERATION_MAX_OUTPUT_TOKENS,
    GENERATION_TEMPERATURE,
    GeneratedBundleDraft,
    GeneratedUserBundle,
    generation_contract_hash,
    generation_messages,
    validate_bundle,
    validate_successful_generation_trace,
)
from .text import conservative_token_bound, estimate_tokens


GENERATION_PILOT_STAGE = (
    "pm_v2_synthetic_generation_deterministic_evidence_pilot"
)
GENERATION_PILOT_CONTRACT_VERSION = (
    "pm-v2-generation-compatibility-pilot-v7-deterministic-evidence-rationale-"
    "neutral-metadata-single-attempt"
)
GENERATION_PILOT_USER_ID = "pmv2_generation_compatibility_pilot"
GENERATION_PILOT_SEED_OFFSET = 9_000_000
GENERATION_PILOT_MAX_ATTEMPTS = 1
GENERATION_PILOT_REPLAY_PROTOCOL = (
    "pm-v2-paid-schema-response-offline-deterministic-compiler-replay-v1"
)

# Keep semantic-family splits explicit.  More importantly, give each generated
# user one of the curated cohorts below instead of three adjacent labels.  The
# former pilot combined relocation loneliness, friendship distance, and work
# burnout, then required those naturally interacting topics to be mutually
# irrelevant distractors.  That was an internally contradictory data design.
TRAIN_SEMANTIC_FAMILIES = (
    "workload_burnout",
    "friendship_distance",
    "family_expectations",
    "career_change",
    "caregiving_stress",
    "social_anxiety",
    "health_routine_stress",
    "conflict_repair",
    "belonging_and_isolation",
    "decision_paralysis",
    "life_stage_transition",
    "motivation_loss",
    "parenting_pressure",
    "uncertain_future",
)
CALIBRATION_SEMANTIC_FAMILIES = (
    "relocation_loneliness",
    "academic_pressure",
    "trust_rebuilding",
    "self_confidence",
    "sleep_disruption",
)
INTERNAL_TEST_SEMANTIC_FAMILIES = (
    "workplace_conflict",
    "grief_adjustment",
    "relationship_uncertainty",
    "identity_transition",
    "financial_uncertainty",
)
SEMANTIC_FAMILIES = [
    *TRAIN_SEMANTIC_FAMILIES,
    *CALIBRATION_SEMANTIC_FAMILIES,
    *INTERNAL_TEST_SEMANTIC_FAMILIES,
]

# Cohorts are deliberately non-adjacent in meaning.  Each family appears in at
# least one cohort for its split, while related pairs such as relocation /
# friendship-distance and family-expectations / parenting are never forced to
# act as one another's off-topic negative in the same generated user.
SEMANTIC_FAMILY_COHORTS_BY_SPLIT: dict[str, tuple[tuple[str, str, str], ...]] = {
    "train": (
        ("workload_burnout", "family_expectations", "health_routine_stress"),
        ("friendship_distance", "career_change", "decision_paralysis"),
        ("caregiving_stress", "social_anxiety", "motivation_loss"),
        ("conflict_repair", "life_stage_transition", "workload_burnout"),
        ("belonging_and_isolation", "career_change", "parenting_pressure"),
        ("uncertain_future", "friendship_distance", "health_routine_stress"),
        ("family_expectations", "motivation_loss", "belonging_and_isolation"),
        ("caregiving_stress", "decision_paralysis", "conflict_repair"),
        ("social_anxiety", "life_stage_transition", "parenting_pressure"),
        ("uncertain_future", "workload_burnout", "caregiving_stress"),
    ),
    "calibration": (
        ("relocation_loneliness", "academic_pressure", "trust_rebuilding"),
        ("relocation_loneliness", "self_confidence", "sleep_disruption"),
        ("academic_pressure", "trust_rebuilding", "sleep_disruption"),
        ("self_confidence", "academic_pressure", "relocation_loneliness"),
    ),
    "internal_test": (
        ("workplace_conflict", "grief_adjustment", "relationship_uncertainty"),
        ("workplace_conflict", "identity_transition", "financial_uncertainty"),
        ("grief_adjustment", "identity_transition", "financial_uncertainty"),
        ("relationship_uncertainty", "financial_uncertainty", "grief_adjustment"),
    ),
}

# The compatibility pilot spans three maximally distinct topics.  It is not a
# training row and therefore may draw one family from more than one split.
GENERATION_PILOT_FAMILIES = (
    "relocation_loneliness",
    "academic_pressure",
    "workplace_conflict",
)


def generation_family_schedule(
    split: str, user_count: int
) -> list[tuple[str, str, str]]:
    """Return a frozen cohort schedule counterbalanced across regime positions.

    The nine regime slots repeat family positions 0/1/2.  If a family always
    occupied one position, the model could learn a topic-to-policy shortcut.
    We therefore rotate each curated orthogonal cohort to maximize previously
    unseen (family, position) cells, then minimize cell imbalance.  No labels or
    generated text participate in this deterministic pre-API schedule.
    """

    if user_count < 0:
        raise ValueError("user_count must be non-negative")
    if split not in SEMANTIC_FAMILY_COHORTS_BY_SPLIT:
        raise ValueError(f"unknown semantic-family split: {split}")
    cohorts = SEMANTIC_FAMILY_COHORTS_BY_SPLIT[split]
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    schedule: list[tuple[str, str, str]] = []
    for user_index in range(user_count):
        base = cohorts[user_index % len(cohorts)]
        candidates: list[
            tuple[int, int, int, int, tuple[str, str, str]]
        ] = []
        for rotation in range(3):
            cohort = base[rotation:] + base[:rotation]
            new_cells = sum(
                counts[family][position] == 0
                for position, family in enumerate(cohort)
            )
            current_load = sum(
                counts[family][position]
                for position, family in enumerate(cohort)
            )
            post_square_load = sum(
                (counts[family][position] + 1) ** 2
                for position, family in enumerate(cohort)
            )
            candidates.append(
                (
                    -new_cells,
                    current_load,
                    post_square_load,
                    rotation,
                    cohort,
                )
            )
        *_, selected = min(candidates)
        schedule.append(selected)
        for position, family in enumerate(selected):
            counts[family][position] += 1
    return schedule


def read_generation_seed_dialogues(path: str | Path) -> list[str]:
    seeds: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, str):
                text = row
            else:
                text = row.get("dialogue_text") or row.get("text") or row.get(
                    "content"
                )
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


def shared_generation_code_manifest(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    relative_paths = (
        "src/metacom_pm/api.py",
        "src/metacom_pm/attempt_ledger.py",
        "src/metacom_pm/config.py",
        "src/metacom_pm/io.py",
        "src/metacom_pm/pm_v2_contracts.py",
        "src/metacom_pm/pm_v2_data.py",
        "src/metacom_pm/pm_v2_generation_pilot.py",
        "src/metacom_pm/text.py",
    )
    return {relative: sha256_file(root / relative) for relative in relative_paths}


def _held_out_seed_index(seeds: Sequence[str], *, full_user_count: int) -> int:
    if full_user_count < 1:
        raise ValueError("full_user_count must be positive")
    if len(seeds) <= full_user_count:
        raise RuntimeError(
            "generation compatibility pilot requires a train seed outside the "
            "full generation success-path seed indices"
        )
    used_hashes = {
        sha256_text(str(seed)) for seed in seeds[: int(full_user_count)]
    }
    for index in range(len(seeds) - 1, int(full_user_count) - 1, -1):
        if sha256_text(str(seeds[index])) not in used_hashes:
            return index
    raise RuntimeError(
        "no content-distinct held-out train seed is available for the generation pilot"
    )


def build_generation_compatibility_contract(
    *,
    project_root: str | Path,
    experiment_config_path: str | Path,
    pm_v2_config_path: str | Path,
    seed_dialogues_path: str | Path,
    endpoint: Endpoint,
    base_generation_seed: int,
    full_user_count: int,
    input_token_safety_factor: float,
    fail_on_reported_input_overrun: bool,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> dict[str, Any]:
    seeds = read_generation_seed_dialogues(seed_dialogues_path)
    held_out_index = _held_out_seed_index(seeds, full_user_count=full_user_count)
    held_out_seed = seeds[held_out_index]
    generation_seeds = [
        int(base_generation_seed) + GENERATION_PILOT_SEED_OFFSET + index
        for index in range(GENERATION_PILOT_MAX_ATTEMPTS)
    ]
    messages = generation_messages(
        seed_dialogue=held_out_seed,
        user_id=GENERATION_PILOT_USER_ID,
        semantic_families=GENERATION_PILOT_FAMILIES,
        regimes=list(ResourceNeedRegime),
    )
    request_payload_sha256s = [
        sha256_text(
            canonical_json(
                chat_request_payload(
                    endpoint,
                    messages,
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
                    seed=generation_seed,
                    response_schema=GeneratedBundleDraft,
                )
            )
        )
        for generation_seed in generation_seeds
    ]
    code_manifest = shared_generation_code_manifest(project_root)
    payload = {
        "version": GENERATION_PILOT_CONTRACT_VERSION,
        "stage": GENERATION_PILOT_STAGE,
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "seed_dialogues_path": str(Path(seed_dialogues_path).resolve()),
        "seed_dialogues_sha256": sha256_file(seed_dialogues_path),
        "full_user_count": int(full_user_count),
        "held_out_seed_policy": "last_content_distinct_index_after_full_success_path",
        "held_out_seed_index": held_out_index,
        "held_out_seed_sha256": sha256_text(held_out_seed),
        "pilot_user_id": GENERATION_PILOT_USER_ID,
        "semantic_families": list(GENERATION_PILOT_FAMILIES),
        "required_regimes": [regime.value for regime in ResourceNeedRegime],
        "generation_seed": generation_seeds[0],
        "generation_attempt_seeds": generation_seeds,
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "api_key_env": endpoint.api_key_env,
            "timeout_seconds": endpoint.timeout_seconds,
        },
        "generation_controls": {
            "temperature": GENERATION_TEMPERATURE,
            "max_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
            "request_retries": 1,
            "maximum_physical_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
            "stop_after_first_success": True,
        },
        "api_cost_planning": {
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": bool(
                fail_on_reported_input_overrun
            ),
        },
        "generator_pricing_usd_per_mtok": {
            "input": float(input_usd_per_mtok),
            "output": float(output_usd_per_mtok),
        },
        "prompt_contract_sha256": generation_contract_hash(),
        "prompt_sha256": sha256_text(canonical_json(messages)),
        "schema_sha256": sha256_text(
            canonical_json(GeneratedBundleDraft.model_json_schema())
        ),
        "request_payload_sha256": request_payload_sha256s[0],
        "request_payload_sha256s": request_payload_sha256s,
        "shared_code_manifest": code_manifest,
        "shared_code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def build_generation_compatibility_plan(
    contract: dict[str, Any],
    *,
    endpoint: Endpoint,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    seeds = read_generation_seed_dialogues_from_contract(contract)
    held_out_seed = seeds[int(contract["held_out_seed_index"])]
    messages = generation_messages(
        seed_dialogue=held_out_seed,
        user_id=str(contract["pilot_user_id"]),
        semantic_families=[str(value) for value in contract["semantic_families"]],
        regimes=list(ResourceNeedRegime),
    )
    controls = contract["generation_controls"]
    prompt_sha256 = sha256_text(canonical_json(messages))
    attempt_seeds = [int(value) for value in contract["generation_attempt_seeds"]]
    if len(attempt_seeds) != int(controls["maximum_physical_attempts"]):
        raise RuntimeError("generation pilot attempt seeds differ from its cap")
    rows: list[dict[str, Any]] = []
    for attempt_index, generation_seed in enumerate(attempt_seeds, 1):
        request_payload = chat_request_payload(
            endpoint,
            messages,
            temperature=float(controls["temperature"]),
            max_tokens=int(controls["max_output_tokens"]),
            seed=generation_seed,
            response_schema=GeneratedBundleDraft,
        )
        request_json = canonical_json(request_payload)
        request_payload_sha256 = sha256_text(request_json)
        record_ids = {
            "user_id": str(contract["pilot_user_id"]),
            "generation_seed": generation_seed,
            "attempt_index": attempt_index,
        }
        call_key = physical_call_key(
            stage=GENERATION_PILOT_STAGE,
            record_ids=record_ids,
            prompt_sha256=prompt_sha256,
            endpoint=endpoint,
            request_parameters={
                "temperature": float(controls["temperature"]),
                "max_tokens": int(controls["max_output_tokens"]),
                "seed": generation_seed,
                "response_schema": GeneratedBundleDraft.__name__,
                "request_payload_sha256": request_payload_sha256,
            },
        )
        rows.append(
            {
                "user_id": str(contract["pilot_user_id"]),
                "generation_seed": generation_seed,
                "attempt_index": attempt_index,
                "physical_call_key": call_key,
                "prompt_sha256": prompt_sha256,
                "request_payload_sha256": request_payload_sha256,
                "raw_estimated_input_tokens": estimate_tokens(request_json),
                "input_token_upper_bound": conservative_token_bound(
                    request_json,
                    safety_factor=float(
                        contract["api_cost_planning"][
                            "input_token_safety_factor"
                        ]
                    ),
                ),
                "maximum_output_tokens": int(controls["max_output_tokens"]),
                "maximum_physical_attempts": 1,
            }
        )
    if (
        prompt_sha256 != contract["prompt_sha256"]
        or [row["request_payload_sha256"] for row in rows]
        != contract["request_payload_sha256s"]
    ):
        raise RuntimeError("generation pilot plan differs from its lineage contract")
    return rows[0], rows, messages


def read_generation_seed_dialogues_from_contract(contract: dict[str, Any]) -> list[str]:
    path = contract.get("seed_dialogues_path")
    if not path:
        raise RuntimeError("generation compatibility contract lacks seed_dialogues_path")
    seeds = read_generation_seed_dialogues(path)
    if sha256_file(path) != contract["seed_dialogues_sha256"]:
        raise RuntimeError("generation compatibility seed file changed")
    return seeds


def validate_generation_pilot_bundle(
    bundle: GeneratedUserBundle, contract: dict[str, Any]
) -> dict[str, Any]:
    bundle_validation = validate_bundle(bundle)
    trace_error = None
    try:
        trace_validation = validate_successful_generation_trace(bundle)
    except RuntimeError as exc:
        trace_error = str(exc)
        trace_validation = {
            "status": "FAIL",
            "provider_draft_sha256": None,
            "provider_response_sha256": None,
            "deterministic_draft_lint_status": "FAIL",
        }
    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    observed_regimes = {case.regime.value for case in bundle.cases}
    expected_families = {str(value) for value in contract["semantic_families"]}
    observed_families = {case.semantic_family for case in bundle.cases}
    all_sources_present = all(
        case.profile_memories and case.summary_memories and case.event_memories
        for case in bundle.cases
    )
    balanced_source_inventory = all(
        len(case.profile_memories)
        == len(case.summary_memories)
        == len(case.event_memories)
        == 2
        for case in bundle.cases
    )
    source_form_correct = all(
        all(item.text.casefold().startswith("the user ") for item in case.profile_memories)
        and all(
            item.text.casefold().startswith(
                "across multiple prior sessions, the user "
            )
            for item in case.summary_memories
        )
        and all(
            item.text.casefold().startswith(("in ", "during ", "at "))
            for item in case.event_memories
        )
        for case in bundle.cases
    )
    strictly_prior_dialogue = all(
        case.recent_dialogue
        and case.recent_dialogue[-1].role == "assistant"
        and all(
            turn.content.strip().casefold()
            != case.current_user_text.strip().casefold()
            for turn in case.recent_dialogue
        )
        for case in bundle.cases
    )
    memory_ages = [
        case.session_index - item.created_session
        for case in bundle.cases
        for pool in (
            case.profile_memories,
            case.summary_memories,
            case.event_memories,
        )
        for item in pool
    ]
    checks = {
        "user_id": bundle.user_id == contract["pilot_user_id"],
        "exact_nine_regimes": (
            len(bundle.cases) == len(ResourceNeedRegime)
            and observed_regimes == expected_regimes
        ),
        "exact_semantic_families": observed_families == expected_families,
        "all_memory_sources_present": all_sources_present,
        "balanced_two_items_per_source": balanced_source_inventory,
        "source_specific_memory_form": source_form_correct,
        "dialogue_is_strictly_before_current": strictly_prior_dialogue,
        "varied_memory_ages": len(set(memory_ages)) >= 6,
        "deterministic_draft_lint": (
            trace_validation["deterministic_draft_lint_status"] == "PASS"
        ),
        "provider_oracle_evidence_not_used": (
            trace_validation.get("provider_oracle_evidence_used") is False
        ),
        "provider_oracle_rationale_not_used": (
            trace_validation.get("provider_oracle_rationale_used") is False
        ),
        "deterministic_bundle_recompiled": (
            trace_validation.get("deterministic_bundle_recompiled") is True
        ),
        "neutral_case_ids": all(
            not any(regime.value in case.case_id for regime in ResourceNeedRegime)
            for case in bundle.cases
        ),
        "all_session_positions_present": (
            {case.session_index for case in bundle.cases}
            == {20 + 3 * position for position in range(len(ResourceNeedRegime))}
        ),
        "evidence_blueprint_bound": bool(
            trace_validation.get("evidence_blueprint_sha256")
        ),
        "reconstructable_provider_trace": bool(
            trace_validation["provider_draft_sha256"]
            and trace_validation["provider_response_sha256"]
        ),
        "generation_contract_binding": (
            bundle.provenance.get("generation_compatibility_contract_sha256")
            == contract["contract_sha256"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_regimes": sorted(observed_regimes),
        "observed_semantic_families": sorted(observed_families),
        "memory_age_sessions": bundle_validation["memory_age_sessions"],
        "successful_generation_trace": trace_validation,
        "successful_generation_trace_error": trace_error,
        "scope": (
            "deterministic structure, surface lint/fallback, evaluator rationale, "
            "and evidence blueprints only; independent human semantic review is required "
            "before full generation"
        ),
    }


def require_generation_compatibility_attestation(
    attestation_path: str | Path,
    *,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    attestation_path = Path(attestation_path).resolve()
    if not attestation_path.is_file():
        raise RuntimeError(
            "full PM-v2 generation requires a PASS generation compatibility "
            f"pilot attestation: {attestation_path}"
        )
    attestation = read_json(attestation_path)
    parameters = attestation.get("parameters") or {}
    compatibility_mode = parameters.get("compatibility_mode")
    pilot_dir = attestation_path.parent
    bundle_path = pilot_dir / "pilot_bundle.json"
    summary_path = pilot_dir / "summary.json"
    if compatibility_mode == GENERATION_PILOT_REPLAY_PROTOCOL:
        replay_path = pilot_dir / "compiler_replay_provenance.json"
        verification = require_artifact_attestation(
            attestation_path,
            required_stage=GENERATION_PILOT_STAGE,
            required_output_paths={
                "pilot_bundle": bundle_path,
                "summary": summary_path,
                "compiler_replay_provenance": replay_path,
            },
        )
        recorded_contract = parameters.get("compatibility_contract")
        if (
            recorded_contract != expected_contract
            or parameters.get("compatibility_contract_sha256")
            != expected_contract["contract_sha256"]
        ):
            raise RuntimeError(
                "generation replay lineage differs from the current "
                "endpoint/config/prompt/schema/shared-code contract"
            )
        replay = read_json(replay_path)
        summary = read_json(summary_path)
        required_replay_checks = {
            "status": replay.get("status") == "PASS",
            "protocol": replay.get("protocol") == GENERATION_PILOT_REPLAY_PROTOCOL,
            "current_contract": replay.get("compatibility_contract_sha256")
            == expected_contract["contract_sha256"],
            "same_provider_schema": replay.get("provider_schema_sha256")
            == expected_contract["schema_sha256"],
            "one_source_physical_attempt": replay.get(
                "source_physical_attempts"
            )
            == 1,
            "zero_new_physical_attempts": replay.get(
                "new_physical_api_attempts"
            )
            == 0,
            "source_schema_valid": replay.get(
                "source_payload_validates_current_provider_schema"
            )
            is True,
            "raw_response_reconstructs": replay.get(
                "raw_response_reconstructs_provider_draft"
            )
            is True,
            "provider_oracle_ignored": replay.get(
                "provider_oracle_evidence_used"
            )
            is False,
        }
        if not all(required_replay_checks.values()):
            raise RuntimeError(
                "generation compiler replay failed: "
                + canonical_json(required_replay_checks)
            )
        if (
            summary.get("status") != "PASS"
            or summary.get("compatibility_mode") != GENERATION_PILOT_REPLAY_PROTOCOL
            or summary.get("compatibility_contract_sha256")
            != expected_contract["contract_sha256"]
        ):
            raise RuntimeError("generation compiler replay summary is not PASS")
        bundle = GeneratedUserBundle.model_validate(read_json(bundle_path))
        bundle_report = validate_generation_pilot_bundle(bundle, expected_contract)
        if bundle_report["status"] != "PASS":
            raise RuntimeError(
                "generation compiler replay bundle failed validation: "
                f"{bundle_report}"
            )
        if summary.get("bundle_validation") != bundle_report:
            raise RuntimeError("generation compiler replay summary is stale")
        return {
            **verification,
            "status": "PASS",
            "compatibility_mode": GENERATION_PILOT_REPLAY_PROTOCOL,
            "compatibility_contract_sha256": expected_contract[
                "contract_sha256"
            ],
            "new_physical_api_attempts": 0,
            "source_physical_attempts": 1,
            "bundle_report": bundle_report,
        }

    ledger_path = pilot_dir / "physical_attempt_ledger.jsonl"
    estimate_path = pilot_dir / "cost_estimate.json"
    plan_path = pilot_dir / "call_plan.jsonl"
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=GENERATION_PILOT_STAGE,
        required_output_paths={
            "pilot_bundle": bundle_path,
            "physical_attempt_ledger": ledger_path,
            "summary": summary_path,
        },
    )
    recorded_contract = parameters.get("compatibility_contract")
    if (
        recorded_contract != expected_contract
        or parameters.get("compatibility_contract_sha256")
        != expected_contract["contract_sha256"]
    ):
        raise RuntimeError(
            "generation compatibility pilot lineage differs from the current "
            "endpoint/config/prompt/schema/shared-code contract"
        )
    estimate = read_json(estimate_path)
    estimate_payload = {
        key: value
        for key, value in estimate.items()
        if key not in {"cost_estimate_sha256", "budget_gate"}
    }
    accepted_cost_hash = sha256_text(canonical_json(estimate_payload))
    plan_rows = list(iter_jsonl(plan_path))
    if (
        estimate.get("cost_estimate_sha256") != accepted_cost_hash
        or parameters.get("accepted_cost_estimate_sha256") != accepted_cost_hash
        or (estimate.get("budget_gate") or {}).get("status") != "PASS"
        or estimate.get("call_plan_sha256")
        != sha256_text(canonical_json(plan_rows))
        or len(plan_rows)
        != int(expected_contract["generation_controls"]["maximum_physical_attempts"])
    ):
        raise RuntimeError(
            "generation compatibility pilot cost plan/accepted hash is stale"
        )
    summary = read_json(summary_path)
    if (
        summary.get("status") != "PASS"
        or summary.get("compatibility_contract_sha256")
        != expected_contract["contract_sha256"]
    ):
        raise RuntimeError("generation compatibility pilot summary is not PASS")
    bundle = GeneratedUserBundle.model_validate(read_json(bundle_path))
    bundle_report = validate_generation_pilot_bundle(bundle, expected_contract)
    if bundle_report["status"] != "PASS":
        raise RuntimeError(
            f"generation compatibility pilot bundle failed validation: {bundle_report}"
        )
    plan_row = summary.get("successful_call_plan") or {}
    if plan_row not in plan_rows:
        raise RuntimeError("generation compatibility pilot summary plan is stale")
    call_key = str(plan_row.get("physical_call_key") or "")
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={
            str(row["physical_call_key"]): 1 for row in plan_rows
        },
        maximum_total_attempts=len(plan_rows),
    )
    success_index = plan_rows.index(plan_row)
    expected_started_keys = {
        str(row["physical_call_key"])
        for row in plan_rows[: success_index + 1]
    }
    successful_keys = {
        str(row["physical_call_key"])
        for row in plan_rows
        if ledger.succeeded(str(row["physical_call_key"]))
    }
    if (
        successful_keys != {call_key}
        or ledger.started_call_keys != expected_started_keys
        or ledger.started_attempts != success_index + 1
        or int(summary.get("physical_attempts") or 0) != ledger.started_attempts
        or summary.get("bundle_validation") != bundle_report
    ):
        raise RuntimeError(
            "generation compatibility pilot ledger does not stop at first success"
        )
    terminal = ledger.terminal_row(call_key)
    if terminal is None or not terminal.get("request_hash"):
        raise RuntimeError("generation compatibility pilot success lacks request hash")
    require_reported_usage(
        terminal.get("usage"), stage="persisted generation compatibility pilot"
    )
    for started_row in plan_rows[: success_index + 1]:
        started_key = str(started_row["physical_call_key"])
        terminal_row = ledger.terminal_row(started_key)
        if terminal_row is None:
            raise RuntimeError("generation pilot attempt lacks a terminal event")
        result = terminal_row.get("result") or {}
        if isinstance(result, dict) and (
            "provider_response" in result or "bundle" in result
        ):
            require_reported_usage(
                terminal_row.get("usage"),
                stage="persisted completed generation-pilot attempt",
            )
    result_bundle = (terminal.get("result") or {}).get("bundle")
    if result_bundle != bundle.model_dump(mode="json"):
        raise RuntimeError(
            "generation compatibility pilot ledger result differs from artifact"
        )
    return {
        **verification,
        "status": "PASS",
        "compatibility_contract_sha256": expected_contract["contract_sha256"],
        "accepted_cost_estimate_sha256": accepted_cost_hash,
        "cost_estimate_sha256": sha256_file(estimate_path),
        "call_plan_sha256": sha256_file(plan_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "bundle_report": bundle_report,
    }
