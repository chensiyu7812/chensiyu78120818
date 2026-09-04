"""Automated multi-family replacement for the V8 human semantic-review gate.

`scripts/20_generate_pm_v2_development_data.py` normally requires a PASSed
two-independent-human-annotator review of a 27-case validation set
(`require_generation_semantic_review_v8`) before the real 52-user synthetic
generation is allowed to run. The user opted for PM-v1.5's disclosed, no-
human-calibration track instead, so the v1.5 fork of that script
(`scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py`) does not call that
human gate. This module is what stands in its place: instead of an
unconditional bypass, it runs the SAME 12-question rubric the human reviewers
would have used, through 2-3 independent LLM judge families, and only reports
PASS if every family affirms every field on every real case AND a strict
majority correctly flags the specific corrupted field in each deliberately
mislabeled positive control. If the panel cannot tell a corrupted case from a
real one, the gate fails closed rather than reporting a false PASS.

This is still not independent human validation. Any paper reporting on
results produced under this gate must disclose that explicitly.
"""

from __future__ import annotations

import json
import copy
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .api import Endpoint, endpoint_transport, make_client, require_reported_usage
from .artifacts import require_artifact_attestation
from .config import endpoint_from_config, load_config
from .contracts import MemorySource
from .io import canonical_json, read_json, sha256_file, sha256_text
from .pm_v2_contracts import StrictModel
from .pm_v2_generation_review_v8 import (
    RATING_FIELDS,
    REVIEW_QUESTIONS_EN,
    V8ReviewCase,
)

AUTOMATED_REVIEW_PROTOCOL = (
    "pm-v1.5-automated-semantic-review-v4-native-gemini-"
    "deterministic27-plus-paid9"
)
AUTOMATED_CONTROL_PROTOCOL = "pm-v1.5-pilot-controls-v2"


def build_judge_endpoint_descriptors(
    experiment_config: Mapping[str, Any], endpoint_names: Sequence[str]
) -> list[dict[str, str]]:
    descriptors = []
    for name in endpoint_names:
        endpoint = endpoint_from_config(experiment_config, str(name))
        if not endpoint.family:
            raise RuntimeError(f"semantic-review endpoint {name} lacks family")
        descriptors.append(
            {
                "alias": str(name),
                "family": str(endpoint.family),
                "model": str(endpoint.model),
                "base_url": str(endpoint.base_url),
                "transport": endpoint_transport(endpoint),
            }
        )
    if len(descriptors) < 2 or len({row["family"] for row in descriptors}) != len(
        descriptors
    ):
        raise RuntimeError(
            "semantic-review panel requires distinct declared judge families"
        )
    return descriptors

# Frozen replacements for the legacy V8 review cards. The original V8 IDs all
# came from dialogue sources that are now among the 52 formal development seed
# sources, so retaining them would defeat instance-level seed/Bank isolation.
# These cards preserve the same eleven semantic roles, are present in the
# 11,590-card source-disjoint V1.5 Bank, and come from eleven distinct retained
# source dialogues. They are review stimuli only; they are never PM features.
V1_5_REVIEW_STRATEGY_CARD_IDS = {
    "gentle_question": "strat_3117fcf093c71e45f41b",
    "open_restatement": "strat_5e7c70020f3cff635dfb",
    "stress_reflection": "strat_a0c6982aa523734322d0",
    "exam_reflection": "strat_c11fdbd724094b26fab8",
    "one_problem_suggestion": "strat_bfae933e5a8b5bd35388",
    "social_connection_suggestion": "strat_8ed3e61eeff96909f66e",
    "low_pressure_connection": "strat_7496a93d31d3325ec3da",
    "intrusive_long_plan": "strat_f2e1526602e265075feb",
    "video_call_self_disclosure": "strat_e5381557588a3941ddc1",
    "support_affirmation": "strat_4b223b52154bcbe29f4c",
    "job_information": "strat_d54feb99bfd36cfd0702",
}


class AutomatedSemanticReviewOutput(StrictModel):
    semantic_family_match: Literal[0, 1]
    regime_match: Literal[0, 1]
    memory_sources_marginal_value_match: Literal[0, 1]
    memory_item_utility_match: Literal[0, 1]
    source_type_match: Literal[0, 1]
    dialogue_temporal_order_match: Literal[0, 1]
    context_grounding_match: Literal[0, 1]
    memory_age_design_match: Literal[0, 1]
    strategy_resource_need_match: Literal[0, 1]
    strategy_item_utility_match: Literal[0, 1]
    advice_readiness_match: Literal[0, 1]
    surface_naturalness_match: Literal[0, 1]
    notes: str


def _require_attested_input_hash(
    attestation: Mapping[str, Any], logical_name: str, expected_path: str | Path
) -> None:
    record = (attestation.get("inputs") or {}).get(logical_name)
    if not isinstance(record, dict) or record.get("sha256") != sha256_file(
        expected_path
    ):
        raise RuntimeError(
            f"automated semantic-review attestation does not bind current {logical_name}"
        )


def require_automated_semantic_review_pass(
    report_path: str | Path,
    attestation_path: str | Path,
    *,
    expected_experiment_config_path: str | Path,
    expected_pm_config_path: str | Path,
    expected_strategy_bank_path: str | Path,
    expected_generation_pilot_attestation_path: str | Path,
) -> dict[str, Any]:
    if expected_generation_pilot_attestation_path is None:
        raise RuntimeError(
            "automated semantic review requires the exact paid pilot attestation"
        )
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="pm_v1_5_automated_semantic_review",
        required_output_paths={"gate_report": report_path},
    )
    report = read_json(report_path)
    attestation = read_json(attestation_path)
    _require_attested_input_hash(
        attestation, "experiment_config", expected_experiment_config_path
    )
    _require_attested_input_hash(
        attestation, "pm_v1_5_config", expected_pm_config_path
    )
    _require_attested_input_hash(
        attestation, "strategy_bank", expected_strategy_bank_path
    )
    _require_attested_input_hash(
        attestation,
        "generation_pilot_attestation",
        expected_generation_pilot_attestation_path,
    )
    expected_pilot_attestation = read_json(
        expected_generation_pilot_attestation_path
    )
    expected_pilot_contract = (
        expected_pilot_attestation.get("parameters") or {}
    ).get("compatibility_contract")
    if not isinstance(expected_pilot_contract, dict):
        raise RuntimeError("expected generation pilot lacks compatibility contract")
    required_outputs = {
        "real_case_judgments",
        "control_judgments",
        "controls",
        "gate_report",
        "physical_attempt_ledger",
    }
    output_records = attestation.get("outputs") or {}
    control_manifest = report.get("control_manifest") or []
    controls_record = output_records.get("controls") or {}
    controls_path = Path(str(controls_record.get("path") or ""))
    pm_config = load_config(expected_pm_config_path)
    expected_endpoint_names = list(
        (pm_config.get("automated_semantic_review") or {}).get(
            "judge_endpoints"
        )
        or []
    )
    expected_descriptors = build_judge_endpoint_descriptors(
        load_config(expected_experiment_config_path), expected_endpoint_names
    )
    if (
        report.get("protocol") != AUTOMATED_REVIEW_PROTOCOL
        or report.get("status") != "PASS"
        or report.get("human_calibration_performed") is not False
        or report.get("control_protocol") != AUTOMATED_CONTROL_PROTOCOL
        or list(report.get("required_control_fields") or []) != list(RATING_FIELDS)
        or int(report.get("controls_per_field") or 0) != 2
        or int(report.get("n_controls") or 0) != 2 * len(RATING_FIELDS)
        or report.get("control_field_counts")
        != {field: 2 for field in RATING_FIELDS}
        or list(report.get("control_contract_errors") or [])
        or len(str(report.get("control_matrix_sha256") or "")) != 64
        or len(report.get("control_catches") or []) != 2 * len(RATING_FIELDS)
        or list(report.get("control_misses") or [])
        or not required_outputs <= set(output_records)
        or len(control_manifest) != 2 * len(RATING_FIELDS)
        or sha256_text(canonical_json(control_manifest))
        != report.get("control_matrix_sha256")
        or not controls_path.is_file()
        or read_json(controls_path) != control_manifest
        or report.get("judge_endpoint_descriptors") != expected_descriptors
        or report.get("judge_families") != expected_endpoint_names
        or int(report.get("n_real_cases") or 0) != 36
        or int(report.get("n_paid_pilot_cases") or 0) != 9
        or (report.get("paid_pilot_audit") or {}).get(
            "pilot_attestation_sha256"
        )
        != sha256_file(expected_generation_pilot_attestation_path)
        or (report.get("paid_pilot_audit") or {}).get(
            "pilot_contract_sha256"
        )
        != expected_pilot_contract.get("contract_sha256")
        or (attestation.get("parameters") or {}).get(
            "judge_endpoint_descriptors"
        )
        != expected_descriptors
        or (attestation.get("parameters") or {}).get(
            "paid_pilot_attestation_sha256"
        )
        != sha256_file(expected_generation_pilot_attestation_path)
    ):
        raise RuntimeError("PM-v1.5 automated semantic-review gate did not PASS")
    return {
        "report": report,
        "attestation_sha256": verification["attestation_sha256"],
    }

# Deliberately wrong values rotated in to build positive-control cases. Each
# entry corrupts exactly one field so a competent reviewer should flag that
# field (and ideally only that field) as unsupported.
def _render_case_text(case: V8ReviewCase) -> str:
    lines = [
        f"Candidate semantic family: {case.semantic_family}",
        f"Candidate regime: {case.regime}",
        f"Candidate materially-useful memory sources: {[s.value for s in case.materially_useful_memory_sources]}",
        f"Current session index: {case.session_index}",
        "",
        "Dialogue before current turn:",
    ]
    for turn in case.dialogue_before_current:
        lines.append(f"  {turn.role}: {turn.content}")
    lines += [
        f"Current user message: {case.current_user_text}",
        f"Session summary: {case.session_summary}",
        f"Authorized user context: {case.authorized_user_context}",
        "",
        "Profile memories (MP):",
    ]
    for item in case.profile_memories:
        lines.append(
            f"  - [source={item.source.value}, utility={item.utility}, "
            f"created_session={item.created_session}, age={item.age_sessions}, "
            f"stale={item.stale}] {item.text}"
        )
    lines.append("Summary memories (MS):")
    for item in case.summary_memories:
        lines.append(
            f"  - [source={item.source.value}, utility={item.utility}, "
            f"created_session={item.created_session}, age={item.age_sessions}, "
            f"stale={item.stale}] {item.text}"
        )
    lines.append("Event memories (ME):")
    for item in case.event_memories:
        lines.append(
            f"  - [source={item.source.value}, utility={item.utility}, "
            f"created_session={item.created_session}, age={item.age_sessions}, "
            f"stale={item.stale}] {item.text}"
        )
    lines += [
        "",
        f"Strategy target: use_strategy_rag={case.strategy_target.use_strategy_rag}, "
        f"advice_readiness={case.strategy_target.advice_readiness}",
        "Strategy evidence:",
    ]
    for item in case.strategy_evidence:
        lines.append(f"  - [{item.utility}] {item.strategy_type}: {item.guidance_text}")
    return "\n".join(lines)


_JUDGE_SYSTEM = """You are an evaluator-only auditor checking whether a synthetic
training case's labels are internally coherent with its own text. You do not
choose a PM action and you must not reward more memory or more strategy
evidence. For each of the 12 fields below, answer 1 if the label is clearly
supported by the case text, 0 if it is not supported, contradicted, or you are
not confident. Every 0 must include a one-sentence reason in `notes` naming
the field. Return only JSON with exactly these keys: the 12 rating fields
(each "0" or "1"), plus "notes" (a single string covering all zero fields, or
empty string if none)."""


def judge_messages(
    case_text: str,
    *,
    rating_questions: Mapping[str, str] = REVIEW_QUESTIONS_EN,
) -> list[dict[str, str]]:
    if set(rating_questions) != set(RATING_FIELDS):
        raise RuntimeError("judge questions must exactly cover the 12-field rubric")
    questions = "\n".join(
        f"- {field}: {rating_questions[field]}" for field in RATING_FIELDS
    )
    payload = {"case": case_text, "rating_questions": questions}
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _different(values: Sequence[Any], current: Any) -> Any:
    for value in values:
        if value != current:
            return copy.deepcopy(value)
    raise RuntimeError("control construction requires a different donor value")


def _flip_utility(value: str) -> str:
    return "irrelevant" if value == "helpful" else "helpful"


def _corrupt_pilot_case(
    case: V8ReviewCase,
    *,
    field: str,
    donors: Sequence[V8ReviewCase],
) -> tuple[V8ReviewCase, dict[str, Any]]:
    corrupted = copy.deepcopy(case)
    override: dict[str, Any]
    if field == "semantic_family_match":
        value = _different([row.semantic_family for row in donors], case.semantic_family)
        corrupted.semantic_family = value
        override = {"semantic_family": value}
    elif field == "regime_match":
        value = _different([row.regime for row in donors], case.regime)
        corrupted.regime = value
        override = {"regime": value}
    elif field == "memory_sources_marginal_value_match":
        value = _different(
            [row.materially_useful_memory_sources for row in donors],
            case.materially_useful_memory_sources,
        )
        corrupted.materially_useful_memory_sources = value
        override = {"materially_useful_memory_sources": [row.value for row in value]}
    elif field == "memory_item_utility_match":
        item = corrupted.profile_memories[0]
        item.utility = _flip_utility(item.utility)
        override = {"memory_id": item.memory_id, "utility": item.utility}
    elif field == "source_type_match":
        item = corrupted.profile_memories[0]
        item.source = MemorySource.MS
        override = {"memory_id": item.memory_id, "source": item.source.value}
    elif field == "dialogue_temporal_order_match":
        turn = next(
            row for row in corrupted.dialogue_before_current if row.role == "user"
        )
        turn.content = corrupted.current_user_text
        override = {"prior_user_turn": "copied_current_user_message"}
    elif field == "context_grounding_match":
        value = next(
            row.session_summary
            for row in donors
            if row.semantic_family != case.semantic_family
            and row.session_summary != case.session_summary
        )
        corrupted.session_summary = value
        override = {"session_summary": value}
    elif field == "memory_age_design_match":
        item = corrupted.profile_memories[0]
        item.age_sessions += 1
        override = {"memory_id": item.memory_id, "age_sessions": item.age_sessions}
    elif field == "strategy_resource_need_match":
        corrupted.strategy_target.use_strategy_rag = not bool(
            corrupted.strategy_target.use_strategy_rag
        )
        override = {"use_strategy_rag": corrupted.strategy_target.use_strategy_rag}
    elif field == "strategy_item_utility_match":
        item = corrupted.strategy_evidence[0]
        item.utility = _flip_utility(item.utility)
        override = {"strategy_card_id": item.card_id, "utility": item.utility}
    elif field == "advice_readiness_match":
        current = corrupted.strategy_target.advice_readiness
        value = "structured_plan" if current != "structured_plan" else "listen_only"
        corrupted.strategy_target.advice_readiness = value
        override = {"advice_readiness": value}
    elif field == "surface_naturalness_match":
        corrupted.current_user_text = (
            "As the generated user for this resource-condition example, "
            + corrupted.current_user_text
        )
        override = {"current_user_text": corrupted.current_user_text}
    else:
        raise RuntimeError(f"unsupported pilot control field: {field}")
    return corrupted, override


def build_positive_controls(
    cases: Sequence[V8ReviewCase],
    *,
    seed: int,
    required_fields: Sequence[str],
    controls_per_field: int,
) -> list[dict[str, Any]]:
    """Build the exact frozen 12-field pilot control matrix."""

    if list(required_fields) != list(RATING_FIELDS) or int(controls_per_field) != 2:
        raise RuntimeError("pilot control contract must be exactly 2 x 12 fields")
    if not cases:
        raise RuntimeError("pilot controls require review cases")
    controls: list[dict[str, Any]] = []
    for field_index, field in enumerate(required_fields):
        candidates = list(cases)
        random.Random(int(seed) + field_index * 1009).shuffle(candidates)
        for replica in range(int(controls_per_field)):
            case = candidates[replica % len(candidates)]
            corrupted, override = _corrupt_pilot_case(
                case, field=field, donors=cases
            )
            controls.append(
                {
                    "item_id": f"{case.item_id}__control_{field}_{replica + 1}",
                    "case_item_id": case.item_id,
                    "corrupted_field": field,
                    "rating_field": field,
                    "override": override,
                    "case_text": _render_case_text(corrupted),
                }
            )
    return controls


def judge_one(
    endpoint: Endpoint, case_text: str, *, stage: str, client: Any | None = None
) -> dict[str, Any]:
    owned_client = client is None
    active_client = client or make_client(endpoint)
    messages = judge_messages(case_text)
    try:
        result, parsed = active_client.chat(
            messages,
            temperature=0.0,
            max_tokens=500,
            seed=13,
            response_schema=AutomatedSemanticReviewOutput,
            retries=1,
        )
        if parsed is None:
            raise RuntimeError("automated semantic review returned no parsed object")
        usage = require_reported_usage(result.usage, stage=stage)
        payload = parsed.model_dump(mode="json")
        return {
            "ratings": {field: int(payload[field]) for field in RATING_FIELDS},
            "notes": payload["notes"],
            "raw_text": result.text,
            "usage": usage,
            "request_hash": result.request_hash,
        }
    finally:
        if owned_client:
            active_client.close()


def build_control_manifest(
    controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": str(row["item_id"]),
            "case_item_id": str(row["case_item_id"]),
            "corrupted_field": str(row["corrupted_field"]),
            "rating_field": str(row["rating_field"]),
            "override": row.get("override") or {},
            "case_text_sha256": sha256_text(str(row["case_text"])),
        }
        for row in controls
    ]


def aggregate_gate(
    *,
    real_case_results: dict[str, dict[str, dict[str, Any]]],
    control_results: dict[str, dict[str, dict[str, Any]]],
    controls: Sequence[dict[str, Any]],
    judge_family_names: Sequence[str],
    control_protocol: str,
    required_control_fields: Sequence[str],
    controls_per_field: int,
    protocol: str = AUTOMATED_REVIEW_PROTOCOL,
) -> dict[str, Any]:
    """`real_case_results[item_id][family] = judge_one(...)` and similarly for
    `control_results[control_item_id][family]`."""

    required_fields = list(required_control_fields)
    field_counts = Counter(str(row.get("rating_field") or "") for row in controls)
    expected_count = len(required_fields) * int(controls_per_field)
    control_contract_errors: list[str] = []
    if required_fields != list(RATING_FIELDS):
        control_contract_errors.append("required fields do not equal the 12-field rubric")
    if int(controls_per_field) != 2:
        control_contract_errors.append("controls_per_field must equal 2")
    if len(controls) != expected_count:
        control_contract_errors.append(
            f"expected {expected_count} controls, found {len(controls)}"
        )
    if len({str(row.get("item_id") or "") for row in controls}) != len(controls):
        control_contract_errors.append("control item IDs are not unique")
    for field in required_fields:
        if field_counts.get(field, 0) != int(controls_per_field):
            control_contract_errors.append(
                f"{field} has {field_counts.get(field, 0)} controls"
            )
    if set(field_counts) != set(required_fields):
        control_contract_errors.append("control fields differ from the required set")
    if any(
        row.get("corrupted_field") != row.get("rating_field") for row in controls
    ):
        control_contract_errors.append("control corrupted/rating field mismatch")

    control_manifest = build_control_manifest(controls)
    control_matrix_sha256 = sha256_text(canonical_json(control_manifest))

    real_failures = []
    for item_id, by_family in real_case_results.items():
        for family in judge_family_names:
            ratings = by_family[family]["ratings"]
            for field in RATING_FIELDS:
                if ratings.get(field) != 1:
                    real_failures.append(
                        {"item_id": item_id, "family": family, "field": field}
                    )

    control_catches = []
    control_misses = []
    for control in controls:
        item_id = control["item_id"]
        rating_field = control["rating_field"]
        by_family = control_results.get(item_id, {})
        caught_by = [
            family
            for family in judge_family_names
            if by_family.get(family, {}).get("ratings", {}).get(rating_field) == 0
        ]
        # Require a majority of families to catch each injected error. This is
        # deliberately stricter than "at least one" since we require ALL
        # families to affirm every field on real cases; a majority bar for
        # catching a deliberately planted error is the matching standard for
        # trusting that same panel's affirmations on real cases.
        if len(caught_by) * 2 > len(judge_family_names):
            control_catches.append({"item_id": item_id, "caught_by": caught_by})
        else:
            control_misses.append(
                {"item_id": item_id, "field": rating_field, "caught_by": caught_by}
            )

    status = (
        "PASS"
        if not control_contract_errors and not real_failures and not control_misses
        else "FAIL"
    )
    return {
        "protocol": str(protocol),
        "status": status,
        "human_calibration_performed": False,
        "judge_families": list(judge_family_names),
        "control_protocol": str(control_protocol),
        "required_control_fields": required_fields,
        "controls_per_field": int(controls_per_field),
        "control_field_counts": dict(sorted(field_counts.items())),
        "control_contract_errors": control_contract_errors,
        "control_matrix_sha256": control_matrix_sha256,
        "control_manifest": control_manifest,
        "n_real_cases": len(real_case_results),
        "n_controls": len(controls),
        "real_case_failures": real_failures,
        "control_catches": control_catches,
        "control_misses": control_misses,
        "gate_reason": (
            "all families affirmed all fields on all real cases, and all "
            "positive-control corruptions were caught by a strict majority"
            if status == "PASS"
            else "see control_contract_errors / real_case_failures / control_misses"
        ),
    }
