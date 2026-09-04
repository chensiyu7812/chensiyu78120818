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
import random
from pathlib import Path
from typing import Any, Literal, Sequence

from .api import Endpoint, make_client, require_reported_usage
from .artifacts import require_artifact_attestation
from .io import canonical_json, read_json, sha256_text
from .pm_v2_contracts import StrictModel
from .pm_v2_generation_review_v8 import (
    RATING_FIELDS,
    REVIEW_QUESTIONS_EN,
    V8ReviewCase,
)

AUTOMATED_REVIEW_PROTOCOL = "pm-v1.5-automated-semantic-review-v1"


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


def require_automated_semantic_review_pass(
    report_path: str | Path, attestation_path: str | Path
) -> dict[str, Any]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="pm_v1_5_automated_semantic_review",
        required_output_paths={"gate_report": report_path},
    )
    report = read_json(report_path)
    if (
        report.get("protocol") != AUTOMATED_REVIEW_PROTOCOL
        or report.get("status") != "PASS"
        or report.get("human_calibration_performed") is not False
    ):
        raise RuntimeError("PM-v1.5 automated semantic-review gate did not PASS")
    return {
        "report": report,
        "attestation_sha256": verification["attestation_sha256"],
    }

# Deliberately wrong values rotated in to build positive-control cases. Each
# entry corrupts exactly one field so a competent reviewer should flag that
# field (and ideally only that field) as unsupported.
_CORRUPTIBLE_FIELDS = {
    "regime": lambda true_value: next(
        v
        for v in (
            "context_only",
            "profile_useful",
            "summary_useful",
            "event_useful",
            "multi_source_useful",
            "memory_harmful",
            "strategy_helpful",
            "advice_harmful",
            "ambiguous",
        )
        if v != true_value
    ),
    "semantic_family": lambda true_value: f"unrelated_topic_swap::{true_value}",
}


def _render_case_text(case: V8ReviewCase, *, override: dict[str, Any] | None = None) -> str:
    override = override or {}
    regime = override.get("regime", case.regime)
    semantic_family = override.get("semantic_family", case.semantic_family)
    lines = [
        f"Candidate semantic family: {semantic_family}",
        f"Candidate regime: {regime}",
        f"Candidate materially-useful memory sources: {[s.value for s in case.materially_useful_memory_sources]}",
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
            f"  - [{item.utility}, age={item.age_sessions}, stale={item.stale}] {item.text}"
        )
    lines.append("Summary memories (MS):")
    for item in case.summary_memories:
        lines.append(
            f"  - [{item.utility}, age={item.age_sessions}, stale={item.stale}] {item.text}"
        )
    lines.append("Event memories (ME):")
    for item in case.event_memories:
        lines.append(
            f"  - [{item.utility}, age={item.age_sessions}, stale={item.stale}] {item.text}"
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


def judge_messages(case_text: str) -> list[dict[str, str]]:
    questions = "\n".join(f"- {field}: {REVIEW_QUESTIONS_EN[field]}" for field in RATING_FIELDS)
    payload = {"case": case_text, "rating_questions": questions}
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_positive_controls(
    cases: Sequence[V8ReviewCase], *, seed: int, n_controls: int
) -> list[dict[str, Any]]:
    """Build corrupted (case, override, corrupted_field) triples for testing."""

    rng = random.Random(seed)
    corruptible_fields = list(_CORRUPTIBLE_FIELDS)
    controls = []
    sample = rng.sample(list(cases), k=min(n_controls, len(cases)))
    for index, case in enumerate(sample):
        field = corruptible_fields[index % len(corruptible_fields)]
        true_value = getattr(case, field)
        corrupted_value = _CORRUPTIBLE_FIELDS[field](true_value)
        controls.append(
            {
                "item_id": f"{case.item_id}__control_{field}",
                "case_item_id": case.item_id,
                "corrupted_field": field,
                "rating_field": (
                    "regime_match" if field == "regime" else "semantic_family_match"
                ),
                "override": {field: corrupted_value},
                "case_text": _render_case_text(case, override={field: corrupted_value}),
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


def aggregate_gate(
    *,
    real_case_results: dict[str, dict[str, dict[str, Any]]],
    control_results: dict[str, dict[str, dict[str, Any]]],
    controls: Sequence[dict[str, Any]],
    judge_family_names: Sequence[str],
) -> dict[str, Any]:
    """`real_case_results[item_id][family] = judge_one(...)` and similarly for
    `control_results[control_item_id][family]`."""

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
        if not real_failures and not control_misses
        else "FAIL"
    )
    return {
        "protocol": AUTOMATED_REVIEW_PROTOCOL,
        "status": status,
        "human_calibration_performed": False,
        "judge_families": list(judge_family_names),
        "n_real_cases": len(real_case_results),
        "n_controls": len(controls),
        "real_case_failures": real_failures,
        "control_catches": control_catches,
        "control_misses": control_misses,
        "gate_reason": (
            "all families affirmed all fields on all real cases, and all "
            "positive-control corruptions were caught by a strict majority"
            if status == "PASS"
            else "see real_case_failures / control_misses"
        ),
    }
