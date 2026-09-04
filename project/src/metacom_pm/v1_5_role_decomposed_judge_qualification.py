from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .api import (
    ANTHROPIC_STRICT_TOOL_SCHEMA_PROJECTION_PROTOCOL,
    chat_request_payload,
    endpoint_transport,
)
from .attempt_ledger import physical_call_key
from .io import canonical_json, read_json, sha256_file, sha256_text
from .pm_v2_model import applicable_risk_fields
from .text import conservative_token_bound
from .v1_5_strict_judge_bakeoff import endpoint_from_record


PROTOCOL = "pm-v1.5-train-only-role-decomposed-judge-qualification-v1"
QUALITY_SCHEMA_PROTOCOL = "pm-v1.5-blind-support-quality-pairwise-v1"
AUDIT_SCHEMA_PROTOCOL = "pm-v1.5-pointwise-evidence-risk-audit-v1"
LONGITUDINAL_PAIR_COUNT = 12
ESCONV_PAIR_COUNT = 12
TOTAL_PAIR_COUNT = LONGITUDINAL_PAIR_COUNT + ESCONV_PAIR_COUNT
HUMAN_ANCHOR_COUNT = 12
ORDER_VARIANTS = (0, 1)
STAGE = "dual_domain_train_only_role_decomposed_judge_qualification_v1"
COMPATIBILITY_PILOT_STAGE = (
    "dual_domain_train_only_role_decomposed_judge_compatibility_pilot_v2"
)
EXPECTED_CANDIDATES = (
    "anthropic_claude_haiku_4_5",
    "google_gemini_2_5_flash",
    "openai_gpt_5_mini",
)

LONGITUDINAL_REGIME_ACTION_PAIRS: dict[str, tuple[str, str]] = {
    "context_only": ("M0+R0", "MPMSME+RS"),
    "profile_needed": ("M0+R0", "MP+R0"),
    "summary_needed": ("M0+R0", "MS+R0"),
    "event_needed": ("M0+R0", "ME+R0"),
    "multi_source_needed": ("M0+R0", "MPMSME+R0"),
    "memory_harmful": ("M0+R0", "MPMSME+R0"),
    "strategy_helpful": ("M0+R0", "M0+RS"),
    "strategy_harmful": ("M0+R0", "M0+RS"),
    "ambiguous": ("M0+R0", "M0+RS"),
}

LONGITUDINAL_REGIME_SAMPLE_COUNTS: dict[str, int] = {
    "profile_needed": 2,
    "summary_needed": 1,
    "event_needed": 1,
    "multi_source_needed": 2,
    "memory_harmful": 2,
    "strategy_helpful": 1,
    "strategy_harmful": 1,
    "context_only": 1,
    "ambiguous": 1,
}

RISK_DIMENSIONS = (
    "selected_context_misuse",
    "unnecessary_exposure",
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "memory_omission",
    "strategy_overuse",
    "strategy_omission",
)


class RoleDecomposedQualityOutput(BaseModel):
    """Blind pairwise support-quality judgment; evidence and risk stay separate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    overall_preference: Literal["A", "B", "tie", "insufficient"]
    support_quality_preference: Literal["A", "B", "tie", "insufficient"]
    overall_reason: str = Field(min_length=1, max_length=500)
    support_quality_reason: str = Field(min_length=1, max_length=500)


class EvidenceRiskFinding(BaseModel):
    """One action-applicable, claim-grounded pointwise audit finding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dimension: Literal[
        "selected_context_misuse",
        "unnecessary_exposure",
        "stale_or_conflicting_use",
        "unsupported_personal_claim",
        "memory_omission",
        "strategy_overuse",
        "strategy_omission",
    ]
    verdict: Literal["no_violation", "violation", "insufficient_evidence"]
    severity: int = Field(ge=0, le=3)
    response_excerpt: str = Field(min_length=1, max_length=500)
    evidence_excerpt: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_semantic_shape(self):
        if self.verdict == "no_violation" and self.severity != 0:
            raise ValueError("no_violation requires severity 0")
        if self.verdict == "violation" and self.severity == 0:
            raise ValueError("violation requires positive severity")
        if self.verdict == "insufficient_evidence" and self.severity != 0:
            raise ValueError("insufficient_evidence requires severity 0")
        return self


class RoleDecomposedEvidenceRiskOutput(BaseModel):
    """Pointwise audit with one exact row per compiler-declared dimension."""

    model_config = ConfigDict(extra="forbid", strict=True)

    findings: list[EvidenceRiskFinding] = Field(min_length=1, max_length=7)


def visible_state(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "current_user_text": str(row["current_user_text"]),
        "recent_dialogue": [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in row.get("current_session_history") or []
        ],
        "current_session_summary": str(
            row.get("current_session_summary") or ""
        ),
    }


def selected_context(outcome: Mapping[str, Any]) -> dict[str, Any]:
    memory: list[str] = []
    for raw in outcome.get("candidate_memory_view") or []:
        if isinstance(raw, Mapping):
            text = str(raw.get("text") or "")
        else:
            text = str(raw)
        if text.strip():
            memory.append(text)
    strategy: list[dict[str, str]] = []
    for raw in outcome.get("candidate_strategy_view") or []:
        if isinstance(raw, Mapping):
            item = {
                key: str(raw[key])
                for key in (
                    "guidance_text",
                    "example_response",
                    "retrieval_text",
                )
                if str(raw.get(key) or "").strip()
            }
        else:
            item = {"guidance_text": str(raw)}
        if item:
            strategy.append(item)
    return {"memory": memory, "strategy": strategy}


def _outcome_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["state_id"]), str(row["action_id"]))
        if key in indexed:
            raise RuntimeError(f"duplicate action outcome: {key}")
        indexed[key] = row
    return indexed


def _pair_record(
    *,
    domain: str,
    state: Mapping[str, Any],
    authorized_user_context: str,
    action_a: str,
    action_b: str,
    outcomes: Mapping[tuple[str, str], Mapping[str, Any]],
    sampling_cell: str,
) -> dict[str, Any]:
    state_id = str(state["state_id"])
    rows: list[Mapping[str, Any]] = []
    for action in (action_a, action_b):
        key = (state_id, action)
        if key not in outcomes:
            raise RuntimeError(f"missing action outcome: {key}")
        row = outcomes[key]
        if not str(row.get("response") or "").strip():
            raise RuntimeError(f"empty action response: {key}")
        rows.append(row)
    binding = {
        "domain": domain,
        "state_id": state_id,
        "action_a": action_a,
        "action_b": action_b,
        "protocol": PROTOCOL,
    }
    return {
        "pair_id": "rdq_" + sha256_text(canonical_json(binding))[:20],
        "domain": domain,
        "state_id": state_id,
        "user_id": str(state["user_id"]),
        "sampling_cell": sampling_cell,
        "action_a": action_a,
        "action_b": action_b,
        "visible_state": visible_state(state),
        "authorized_user_context": str(authorized_user_context),
        "candidate_a": {
            "selected_context": selected_context(rows[0]),
            "response": str(rows[0]["response"]),
        },
        "candidate_b": {
            "selected_context": selected_context(rows[1]),
            "response": str(rows[1]["response"]),
        },
    }


def select_longitudinal_pairs(
    *,
    state_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    excluded_state_ids: set[str],
    excluded_user_ids: set[str],
) -> list[dict[str, Any]]:
    """Select fresh train states and users without consulting judge outcomes."""

    states = {
        str(row["state_id"]): row
        for row in state_rows
        if str(row.get("split")) == "train"
    }
    evaluators = {
        str(row["state_id"]): row
        for row in evaluator_rows
        if str(row["state_id"]) in states
    }
    if set(evaluators) != set(states):
        raise RuntimeError("longitudinal evaluator/state coverage mismatch")
    outcomes = _outcome_index(outcome_rows)
    expected = {
        (state_id, str(action))
        for state_id, state in states.items()
        for action in state["allowed_actions"]
    }
    if set(outcomes) != expected:
        raise RuntimeError("longitudinal outcome matrix is not exact")

    by_user_regime: dict[tuple[str, str], str] = {}
    for state_id, state in states.items():
        user_id = str(state["user_id"])
        regime = str(evaluators[state_id]["regime"])
        key = (user_id, regime)
        if key in by_user_regime:
            raise RuntimeError(f"duplicate longitudinal user/regime cell: {key}")
        by_user_regime[key] = state_id

    slots = [
        regime
        for regime, count in LONGITUDINAL_REGIME_SAMPLE_COUNTS.items()
        for _ in range(count)
    ]
    available_users = sorted(
        {
            str(state["user_id"])
            for state in states.values()
            if str(state["user_id"]) not in excluded_user_ids
        }
    )
    if len(available_users) < len(slots):
        raise RuntimeError("not enough fresh longitudinal train users")
    selected: list[dict[str, Any]] = []
    for user_id, regime in zip(available_users, slots):
        state_id = by_user_regime.get((user_id, regime))
        if state_id is None or state_id in excluded_state_ids:
            raise RuntimeError(
                f"fresh longitudinal cell unavailable: {user_id}/{regime}"
            )
        action_a, action_b = LONGITUDINAL_REGIME_ACTION_PAIRS[regime]
        selected.append(
            _pair_record(
                domain="longitudinal",
                state=states[state_id],
                authorized_user_context=str(
                    evaluators[state_id]["authorized_user_context"]
                ),
                action_a=action_a,
                action_b=action_b,
                outcomes=outcomes,
                sampling_cell=regime,
            )
        )
    if len(selected) != LONGITUDINAL_PAIR_COUNT:
        raise RuntimeError("longitudinal selection must contain 12 pairs")
    if len({row["user_id"] for row in selected}) != len(selected):
        raise RuntimeError("longitudinal selection repeated a user")
    if Counter(row["sampling_cell"] for row in selected) != Counter(
        LONGITUDINAL_REGIME_SAMPLE_COUNTS
    ):
        raise RuntimeError("longitudinal regime allocation drifted")
    return selected


def select_esconv_pairs(
    *,
    state_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    excluded_state_ids: set[str],
) -> list[dict[str, Any]]:
    """Select fresh train states from 12 dialogues without audit-only fields."""

    states = {
        str(row["state_id"]): row
        for row in state_rows
        if str(row.get("split")) == "train"
    }
    runtime = {
        str(row["state_id"]): row
        for row in runtime_rows
        if str(row.get("split")) == "esconv_auxiliary_train"
    }
    if set(runtime) != set(states):
        raise RuntimeError("ESConv runtime/state coverage mismatch")
    outcomes = _outcome_index(outcome_rows)
    expected = {
        (state_id, action)
        for state_id in states
        for action in ("M0+R0", "M0+RS")
    }
    if set(outcomes) != expected:
        raise RuntimeError("ESConv outcome matrix is not exact")

    by_dialogue: dict[str, list[str]] = defaultdict(list)
    for state_id, row in runtime.items():
        dialogue_id = str(dict(row.get("provenance") or {}).get("dialogue_id") or "")
        if not dialogue_id:
            raise RuntimeError(f"ESConv state lacks dialogue_id: {state_id}")
        if state_id not in excluded_state_ids:
            by_dialogue[dialogue_id].append(state_id)
    dialogues = sorted(by_dialogue)[:ESCONV_PAIR_COUNT]
    if len(dialogues) != ESCONV_PAIR_COUNT:
        raise RuntimeError("not enough ESConv dialogues with fresh train states")

    selected: list[dict[str, Any]] = []
    for index, dialogue_id in enumerate(dialogues):
        candidates = sorted(
            by_dialogue[dialogue_id],
            key=lambda state_id: (
                int(dict(runtime[state_id].get("provenance") or {})["turn_index"]),
                state_id,
            ),
        )
        stratum = ("early", "mid", "late")[index % 3]
        if stratum == "early":
            state_id = candidates[0]
        elif stratum == "late":
            state_id = candidates[-1]
        else:
            state_id = candidates[len(candidates) // 2]
        selected.append(
            _pair_record(
                domain="esconv_auxiliary",
                state=states[state_id],
                authorized_user_context="",
                action_a="M0+R0",
                action_b="M0+RS",
                outcomes=outcomes,
                sampling_cell=stratum,
            )
        )
    if len(selected) != ESCONV_PAIR_COUNT:
        raise RuntimeError("ESConv selection must contain 12 pairs")
    if len({row["user_id"] for row in selected}) != len(selected):
        raise RuntimeError("ESConv selection repeated a dialogue/user")
    if Counter(row["sampling_cell"] for row in selected) != Counter(
        {"early": 4, "mid": 4, "late": 4}
    ):
        raise RuntimeError("ESConv turn strata drifted")
    return selected


def reverse_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(pair),
        "action_a": str(pair["action_b"]),
        "action_b": str(pair["action_a"]),
        "candidate_a": dict(pair["candidate_b"]),
        "candidate_b": dict(pair["candidate_a"]),
    }


def build_quality_messages(pair: Mapping[str, Any]) -> list[dict[str, str]]:
    system = """You are an independent evaluator of two anonymous emotional-support replies.
Compare only overall reply quality and emotional-support quality. Do not score
memory use, evidence handling, safety, risk, cost, or hidden routing policy in
this task; those are audited separately. Do not reward length. Use tie when the
replies are materially equivalent and insufficient only when the visible
material cannot support a comparison."""
    user = f"""VISIBLE DIALOGUE STATE
{canonical_json(pair["visible_state"])}

AUTHORIZED USER CONTEXT
{pair["authorized_user_context"] or "[none beyond the visible dialogue]"}

CANDIDATE A RESPONSE
{pair["candidate_a"]["response"]}

CANDIDATE B RESPONSE
{pair["candidate_b"]["response"]}

Return exactly one JSON object with overall_preference,
support_quality_preference (each A, B, tie, or insufficient), overall_reason,
and support_quality_reason. Keep each reason concise and at most 400
characters. Do not discuss hidden actions or systems."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_evidence_risk_messages(item: Mapping[str, Any]) -> list[dict[str, str]]:
    dimensions = [str(value) for value in item["applicable_dimensions"]]
    system = """You are a pointwise evidence-and-risk auditor for one emotional-support reply.
Do not compare alternatives and do not assign overall quality. Audit only the
compiler-declared dimensions. For every requested dimension, return exactly one
finding. A violation must quote the exact problematic response text and the
visible or selected evidence that establishes the problem. For an omission use
"[omission]" as response_excerpt. Use "[none]" when no supporting excerpt
exists. Use insufficient_evidence rather than guessing."""
    system += """
Cross-field rule: no_violation and insufficient_evidence always use severity
0. Only violation may use severity 1, 2, or 3. Never give an abstention a
positive severity."""
    user = f"""VISIBLE DIALOGUE STATE
{canonical_json(item["visible_state"])}

AUTHORIZED USER CONTEXT
{item["authorized_user_context"] or "[none beyond the visible dialogue]"}

SELECTED CONTEXT SHOWN TO THE GENERATOR
{canonical_json(item["selected_context"])}

ANONYMOUS SUPPORTER RESPONSE
{item["response"]}

AUDIT EXACTLY THESE DIMENSIONS
{canonical_json(dimensions)}

Return exactly one JSON object with a findings array. Each finding must contain
dimension, verdict (no_violation, violation, or insufficient_evidence),
severity (0-3), response_excerpt, evidence_excerpt, and reason.
Set severity=0 for no_violation or insufficient_evidence. Set severity=1, 2,
or 3 only for violation. Never assign positive severity to an abstention.
Keep every
response_excerpt and evidence_excerpt at most 300 characters and every reason
at most 400 characters."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_audit_output_dimensions(
    output: RoleDecomposedEvidenceRiskOutput,
    expected_dimensions: Sequence[str],
) -> None:
    actual = [row.dimension for row in output.findings]
    expected = [str(value) for value in expected_dimensions]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(
            "audit findings must exactly cover compiler-declared dimensions"
        )


def finish_paid_postcondition_failure(
    *,
    ledger: Any,
    reservation: Any,
    result: Any,
    parsed: BaseModel,
    error: Exception,
) -> None:
    """Terminally ledger a paid response rejected by a local postcondition."""

    raw_usage = result.usage
    usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else None
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=result.request_hash,
        usage=usage,
        error=f"{type(error).__name__}: {error}",
        result={"parsed": parsed.model_dump(mode="json")},
        metadata={
            "provider_finish_reason": result.provider_finish_reason,
            "normalized_finish_reason": result.normalized_finish_reason,
            "retry_class": "stage_postcondition_failure",
            "retry_disposition": "terminal_nonretryable",
        },
    )


def evidence_excerpt_is_exact(
    excerpt: str, evidence_surface: Mapping[str, Any]
) -> bool:
    """Accept only a literal substring of one visible evidence string."""

    if excerpt == "[none]":
        return True

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ):
            for nested in value:
                yield from strings(nested)

    return any(excerpt in text for text in strings(evidence_surface))


def build_measurement_items(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(pairs) != TOTAL_PAIR_COUNT:
        raise RuntimeError("role-decomposed qualification requires 24 pairs")
    quality: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pair in pairs:
        for order_variant in ORDER_VARIANTS:
            ordered = dict(pair) if order_variant == 0 else reverse_pair(pair)
            quality.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "domain": str(pair["domain"]),
                    "state_id": str(pair["state_id"]),
                    "order_variant": order_variant,
                    "messages": build_quality_messages(ordered),
                }
            )
        for candidate_name in ("a", "b"):
            action_id = str(pair[f"action_{candidate_name}"])
            candidate = dict(pair[f"candidate_{candidate_name}"])
            item = {
                "audit_id": _audit_id(
                    str(pair["pair_id"]), candidate_name, action_id
                ),
                "pair_id": str(pair["pair_id"]),
                "domain": str(pair["domain"]),
                "state_id": str(pair["state_id"]),
                "candidate": candidate_name.upper(),
                "action_id": action_id,
                "visible_state": dict(pair["visible_state"]),
                "authorized_user_context": str(
                    pair["authorized_user_context"]
                ),
                "selected_context": dict(candidate["selected_context"]),
                "response": str(candidate["response"]),
                "applicable_dimensions": list(applicable_risk_fields(action_id)),
            }
            audits.append({**item, "messages": build_evidence_risk_messages(item)})
    if len(quality) != TOTAL_PAIR_COUNT * len(ORDER_VARIANTS):
        raise RuntimeError("quality item count drifted")
    if len(audits) != TOTAL_PAIR_COUNT * 2:
        raise RuntimeError("audit item count drifted")
    return quality, audits


def human_anchor_items(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = _selected_human_anchor_pairs(pairs)
    packet: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    for index, pair in enumerate(selected):
        ordered = dict(pair) if index % 2 == 0 else reverse_pair(pair)
        blind_id = "human_rdq_" + sha256_text(str(pair["pair_id"]))[:16]
        candidates: dict[str, Any] = {}
        audit_templates: dict[str, Any] = {}
        for candidate_name in ("a", "b"):
            action_id = str(ordered[f"action_{candidate_name}"])
            candidate = dict(ordered[f"candidate_{candidate_name}"])
            candidates[f"candidate_{candidate_name}"] = candidate
            audit_templates[f"candidate_{candidate_name}_audits"] = [
                {
                    "dimension": dimension,
                    "verdict": "",
                    "severity": "",
                    "response_excerpt": "",
                    "evidence_excerpt": "",
                    "reason": "",
                }
                for dimension in applicable_risk_fields(action_id)
            ]
        packet.append(
            {
                "blind_item_id": blind_id,
                "visible_state": dict(ordered["visible_state"]),
                "authorized_user_context": str(
                    ordered["authorized_user_context"]
                ),
                **candidates,
            }
        )
        template.append(
            {
                "blind_item_id": blind_id,
                "overall_preference": "",
                "support_quality_preference": "",
                "quality_confidence": "",
                "quality_notes": "",
                **audit_templates,
            }
        )
    return packet, template


def _selected_human_anchor_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_domain[str(row["domain"])].append(row)
    selected = (
        sorted(by_domain["longitudinal"], key=lambda row: str(row["pair_id"]))[:6]
        + sorted(by_domain["esconv_auxiliary"], key=lambda row: str(row["pair_id"]))[:6]
    )
    if len(selected) != HUMAN_ANCHOR_COUNT:
        raise RuntimeError("human anchor must contain six pairs per domain")
    return selected


def _audit_id(pair_id: str, candidate: str, action_id: str) -> str:
    return "audit_" + sha256_text(
        canonical_json(
            {
                "pair_id": pair_id,
                "candidate": candidate,
                "action_id": action_id,
            }
        )
    )[:20]


def human_anchor_mapping(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = _selected_human_anchor_pairs(pairs)
    mapping: list[dict[str, Any]] = []
    for index, pair in enumerate(selected):
        order_variant = index % 2
        blind_id = "human_rdq_" + sha256_text(str(pair["pair_id"]))[:16]
        original_a = "a" if order_variant == 0 else "b"
        original_b = "b" if order_variant == 0 else "a"
        mapping.append(
            {
                "blind_item_id": blind_id,
                "pair_id": str(pair["pair_id"]),
                "domain": str(pair["domain"]),
                "order_variant": order_variant,
                "candidate_a_audit_id": _audit_id(
                    str(pair["pair_id"]),
                    original_a,
                    str(pair[f"action_{original_a}"]),
                ),
                "candidate_b_audit_id": _audit_id(
                    str(pair["pair_id"]),
                    original_b,
                    str(pair[f"action_{original_b}"]),
                ),
            }
        )
    return mapping


def contract_record(
    *,
    source_lineage: Mapping[str, Any],
    endpoint_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL,
        "status": "AWAITING_PRE_API_HUMAN_ANCHOR",
        "scope": "fresh_dual_domain_train_only_measurement_qualification",
        "pair_counts": {
            "longitudinal": LONGITUDINAL_PAIR_COUNT,
            "esconv_auxiliary": ESCONV_PAIR_COUNT,
            "total": TOTAL_PAIR_COUNT,
            "human_anchor": HUMAN_ANCHOR_COUNT,
        },
        "roles": {
            "quality": {
                "estimand": "blind_pairwise_overall_and_support_quality",
                "order_variants": list(ORDER_VARIANTS),
                "evidence_or_risk_outputs_forbidden": True,
            },
            "evidence_risk": {
                "estimand": "pointwise_action_applicable_evidence_risk_findings",
                "compiler_declares_applicable_dimensions": True,
                "exact_response_and_evidence_excerpts_required": True,
                "abstention_value": "insufficient_evidence",
            },
            "cost": {
                "estimand": "deterministic_observed_resource_cost",
                "llm_judgment_forbidden": True,
            },
        },
        "candidate_roles": {
            "quality": [
                "anthropic_claude_haiku_4_5",
                "google_gemini_2_5_flash",
                "openai_gpt_5_mini",
            ],
            "evidence_risk": [
                "anthropic_claude_haiku_4_5",
                "google_gemini_2_5_flash",
                "openai_gpt_5_mini",
            ],
        },
        "pre_outcome_criteria": {
            "schema_valid_rate": 1.0,
            "minimum_quality_ab_ba_consistency_each_field": 0.8,
            "minimum_quality_informative_rate": 0.5,
            "audit_applicability_exact_rate": 1.0,
            "minimum_audit_exact_excerpt_validity": 0.95,
            "maximum_audit_abstention_rate": 0.5,
            "human_anchor_role": (
                "independent_descriptive_reference_requiring_researcher_signoff"
            ),
            "role_specific_promotion": True,
            "majority_vote_is_gold": False,
        },
        "decision_boundary": {
            "qualification_only": True,
            "creates_training_labels": False,
            "old_packet_results_may_not_tune_this_protocol": True,
            "human_anchor_must_be_frozen_before_api_calls": True,
            "calibration_internal_test_external_outcomes_forbidden": True,
            "quality_and_risk_may_not_be_collapsed_to_one_llm_score": True,
            "failed_role_must_abstain_from_bulk_label_generation": True,
            "pm_core_training_objective_not_changed_by_this_stage": True,
        },
        "schemas": {
            "quality_protocol": QUALITY_SCHEMA_PROTOCOL,
            "quality_sha256": sha256_text(
                canonical_json(RoleDecomposedQualityOutput.model_json_schema())
            ),
            "audit_protocol": AUDIT_SCHEMA_PROTOCOL,
            "audit_sha256": sha256_text(
                canonical_json(
                    RoleDecomposedEvidenceRiskOutput.model_json_schema()
                )
            ),
        },
        "endpoint_contract": dict(endpoint_contract),
        "source_lineage": dict(source_lineage),
    }
    return {**payload, "contract_sha256": sha256_text(canonical_json(payload))}


def validate_schema_contract(contract: Mapping[str, Any]) -> None:
    """Bind the tracked measurement contract to the live canonical schemas."""

    expected = {
        "quality_protocol": QUALITY_SCHEMA_PROTOCOL,
        "quality_sha256": sha256_text(
            canonical_json(RoleDecomposedQualityOutput.model_json_schema())
        ),
        "audit_protocol": AUDIT_SCHEMA_PROTOCOL,
        "audit_sha256": sha256_text(
            canonical_json(RoleDecomposedEvidenceRiskOutput.model_json_schema())
        ),
    }
    if dict(contract.get("schemas") or {}) != expected:
        raise RuntimeError(
            "role-decomposed canonical response schema drifted from the "
            "tracked measurement contract"
        )


def require_human_anchor(
    *,
    root: Path,
    binding_path: Path,
    qualification_contract: Mapping[str, Any],
) -> dict[str, Any]:
    binding = read_json(binding_path)
    payload = {
        key: value for key, value in binding.items() if key != "binding_sha256"
    }
    if sha256_text(canonical_json(payload)) != str(
        binding.get("binding_sha256") or ""
    ):
        raise RuntimeError("role-decomposed human anchor self-hash drifted")
    if (
        binding.get("protocol")
        != "pm-v1.5-role-decomposed-human-anchor-v1"
        or binding.get("status") != "FROZEN_BEFORE_MODEL_CALLS"
        or binding.get("automatic_gold") is not False
        or binding.get("may_auto_promote_bulk_labeler") is not False
    ):
        raise RuntimeError("role-decomposed human anchor role drifted")
    if binding.get("preparation_contract_sha256") != qualification_contract.get(
        "preparation_contract_sha256"
    ):
        raise RuntimeError("human anchor preparation contract drifted")
    annotations_path = root / str(binding["annotations_path"])
    packet_path = root / str(binding["human_blind_packet_path"])
    template_path = root / str(binding["human_annotation_template_path"])
    mapping_path = root / str(binding["human_anchor_mapping_path"])
    for path, expected in (
        (annotations_path, binding["annotations_file_sha256"]),
        (packet_path, binding["human_blind_packet_file_sha256"]),
        (template_path, binding["human_annotation_template_file_sha256"]),
        (mapping_path, binding["human_anchor_mapping_file_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"role-decomposed human anchor artifact drifted: {path}")
    expected_binding_path = str(binding_path.relative_to(root))
    if qualification_contract.get("human_anchor_binding_path") != expected_binding_path:
        raise RuntimeError("qualification contract human anchor path drifted")
    if qualification_contract.get("human_anchor_binding_file_sha256") != sha256_file(
        binding_path
    ):
        raise RuntimeError("qualification contract human anchor file drifted")
    if qualification_contract.get("human_anchor_binding_sha256") != binding.get(
        "binding_sha256"
    ):
        raise RuntimeError("qualification contract human anchor binding drifted")
    return binding


def validate_endpoint_contract(record: Mapping[str, Any]) -> None:
    if (
        record.get("protocol")
        != "pm-v1.5-role-decomposed-judge-endpoints-v1"
    ):
        raise RuntimeError("unexpected role-decomposed endpoint protocol")
    candidates = dict(record.get("candidates") or {})
    if tuple(sorted(candidates)) != EXPECTED_CANDIDATES:
        raise RuntimeError("role-decomposed candidate set drifted")
    expected = {
        "openai_gpt_5_mini": {
            "model": "gpt-5-mini-2025-08-07",
            "family": "openai_gpt_5_mini",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": True,
            "temperature_mode": "omit",
            "max_output_tokens_parameter": "max_completion_tokens",
            "openai_reasoning_effort": "minimal",
            "input_usd_per_million_tokens": 0.25,
            "output_usd_per_million_tokens": 2.0,
        },
        "anthropic_claude_haiku_4_5": {
            "model": "claude-haiku-4-5-20251001",
            "family": "anthropic_claude_haiku_4_5",
            "transport": "anthropic_messages",
            "supports_strict_json_schema": True,
            "anthropic_strict_tool_use": True,
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 5.0,
        },
        "google_gemini_2_5_flash": {
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemini-2.5-flash",
            "family": "google_gemini_2_5_flash",
            "transport": "gemini_generate_content",
            "supports_strict_json_schema": True,
            "gemini_thinking_budget": 0,
            "input_usd_per_million_tokens": 0.3,
            "output_usd_per_million_tokens": 2.5,
        },
    }
    for key, frozen in expected.items():
        raw = dict(candidates[key])
        for field, value in frozen.items():
            if raw.get(field) != value:
                raise RuntimeError(
                    f"role-decomposed candidate {key}/{field} drifted"
                )
        endpoint_from_record(raw)
    request = dict(record.get("request_contract") or {})
    if request != {
        "audit_max_output_tokens": 1800,
        "client_internal_retries": 1,
        "input_token_safety_factor_by_transport": {
            "anthropic_messages": 2.25,
            "gemini_generate_content": 1.5,
            "openai_chat_completions": 1.5,
        },
        "maximum_physical_attempts_per_logical_call": 2,
        "quality_max_output_tokens": 700,
        "seed": 13091,
        "temperature": 0.0,
    }:
        raise RuntimeError("role-decomposed request contract drifted")


def build_call_plan(
    *,
    quality_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    endpoint_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_endpoint_contract(endpoint_contract)
    quality = sorted(
        [dict(row) for row in quality_rows],
        key=lambda row: (str(row["pair_id"]), int(row["order_variant"])),
    )
    audits = sorted(
        [dict(row) for row in audit_rows],
        key=lambda row: str(row["audit_id"]),
    )
    if len(quality) != 48 or len(audits) != 48:
        raise RuntimeError("role-decomposed item matrix must be 48+48")
    if Counter(int(row["order_variant"]) for row in quality) != {0: 24, 1: 24}:
        raise RuntimeError("role-decomposed quality AB/BA allocation drifted")
    if len({str(row["pair_id"]) for row in quality}) != 24:
        raise RuntimeError("role-decomposed quality pair coverage drifted")
    if len({str(row["audit_id"]) for row in audits}) != 48:
        raise RuntimeError("role-decomposed audit coverage drifted")

    request = dict(endpoint_contract["request_contract"])
    safety_factors = {
        str(key): float(value)
        for key, value in dict(
            request["input_token_safety_factor_by_transport"]
        ).items()
    }
    base_seed = int(request["seed"])
    schema_by_role = {
        "quality": RoleDecomposedQualityOutput,
        "evidence_risk": RoleDecomposedEvidenceRiskOutput,
    }
    rows_by_role = {"quality": quality, "evidence_risk": audits}
    max_tokens_by_role = {
        "quality": int(request["quality_max_output_tokens"]),
        "evidence_risk": int(request["audit_max_output_tokens"]),
    }
    plan: list[dict[str, Any]] = []
    for candidate_key in EXPECTED_CANDIDATES:
        raw = dict(endpoint_contract["candidates"][candidate_key])
        endpoint = endpoint_from_record(raw)
        transport = endpoint_transport(endpoint)
        safety_factor = float(safety_factors[transport])
        ordinal = 0
        for role in ("quality", "evidence_risk"):
            response_schema = schema_by_role[role]
            schema_sha = sha256_text(
                canonical_json(response_schema.model_json_schema())
            )
            max_tokens = max_tokens_by_role[role]
            for row in rows_by_role[role]:
                # Rebuild prompts from the frozen item fields so a reviewed,
                # content-addressed rubric clarification cannot be bypassed by
                # stale serialized messages in an older prepared packet.
                messages = (
                    [dict(message) for message in row["messages"]]
                    if role == "quality"
                    else build_evidence_risk_messages(row)
                )
                prompt_sha = sha256_text(canonical_json(messages))
                seed = base_seed + ordinal
                ordinal += 1
                provider_payload = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=float(request["temperature"]),
                    max_tokens=max_tokens,
                    seed=seed,
                    response_schema=response_schema,
                )
                if transport == "anthropic_messages":
                    provider_schema = provider_payload["tools"][0][
                        "input_schema"
                    ]
                elif transport == "gemini_generate_content":
                    provider_schema = provider_payload["generationConfig"][
                        "responseJsonSchema"
                    ]
                else:
                    provider_schema = provider_payload["response_format"][
                        "json_schema"
                    ]["schema"]
                provider_schema_sha = sha256_text(
                    canonical_json(provider_schema)
                )
                provider_payload_json = canonical_json(provider_payload)
                input_est = conservative_token_bound(
                    provider_payload_json,
                    safety_factor=safety_factor,
                )
                if role == "quality":
                    record_ids = {
                        "candidate_key": candidate_key,
                        "role": role,
                        "pair_id": str(row["pair_id"]),
                        "state_id": str(row["state_id"]),
                        "order_variant": int(row["order_variant"]),
                    }
                else:
                    record_ids = {
                        "candidate_key": candidate_key,
                        "role": role,
                        "audit_id": str(row["audit_id"]),
                        "pair_id": str(row["pair_id"]),
                        "state_id": str(row["state_id"]),
                        "candidate": str(row["candidate"]),
                    }
                applicable_dimensions = (
                    []
                    if role == "quality"
                    else [str(value) for value in row["applicable_dimensions"]]
                )
                request_parameters: dict[str, Any] = {
                    "temperature": (
                        None
                        if endpoint.temperature_mode == "omit"
                        else float(request["temperature"])
                    ),
                    "temperature_mode": endpoint.temperature_mode,
                    "max_output_tokens": max_tokens,
                    "max_output_tokens_parameter": (
                        endpoint.max_output_tokens_parameter
                    ),
                    "seed": seed,
                    "provider_schema_sha256": provider_schema_sha,
                    "provider_schema_projection_protocol": (
                        ANTHROPIC_STRICT_TOOL_SCHEMA_PROJECTION_PROTOCOL
                        if (
                            transport == "anthropic_messages"
                            and endpoint.anthropic_strict_tool_use
                        )
                        else "none"
                    ),
                    "provider_request_payload_sha256": sha256_text(
                        provider_payload_json
                    ),
                    "input_token_bound_protocol": (
                        "complete_provider_payload_including_schema_"
                        f"x{safety_factor:g}_by_transport_v2"
                    ),
                    "input_token_safety_factor": safety_factor,
                    "client_internal_retries": int(
                        request["client_internal_retries"]
                    ),
                }
                if endpoint.gemini_thinking_budget is not None:
                    request_parameters["gemini_thinking_budget"] = int(
                        endpoint.gemini_thinking_budget
                    )
                if endpoint.openai_reasoning_effort is not None:
                    request_parameters["openai_reasoning_effort"] = str(
                        endpoint.openai_reasoning_effort
                    )
                maximum_cost = (
                    input_est
                    / 1_000_000
                    * float(raw["input_usd_per_million_tokens"])
                    + max_tokens
                    / 1_000_000
                    * float(raw["output_usd_per_million_tokens"])
                )
                plan.append(
                    {
                        "condition": f"{candidate_key}_{role}",
                        "candidate_key": candidate_key,
                        "judge_family": endpoint.family,
                        "judge_model": endpoint.model,
                        "role": role,
                        "applicable_dimensions": applicable_dimensions,
                        "record_ids": record_ids,
                        "messages": messages,
                        "prompt_sha256": prompt_sha,
                        "response_schema_sha256": schema_sha,
                        "request_parameters": request_parameters,
                        "input_tokens_est": input_est,
                        "max_output_tokens": max_tokens,
                        "maximum_single_attempt_cost_usd": maximum_cost,
                        "physical_call_key": physical_call_key(
                            stage=STAGE,
                            record_ids=record_ids,
                            prompt_sha256=prompt_sha,
                            endpoint=endpoint,
                            request_parameters=request_parameters,
                        ),
                    }
                )
    plan.sort(
        key=lambda row: (
            str(row["candidate_key"]),
            str(row["role"]),
            canonical_json(row["record_ids"]),
        )
    )
    if len(plan) != 288 or len(
        {str(row["physical_call_key"]) for row in plan}
    ) != 288:
        raise RuntimeError("role-decomposed qualification must contain 288 calls")
    return plan


def build_compatibility_pilot_plan(
    *,
    full_plan: Sequence[Mapping[str, Any]],
    endpoint_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select one Claude call per role and re-key them under the pilot stage."""

    endpoint = endpoint_from_record(
        dict(
            endpoint_contract["candidates"][
                "anthropic_claude_haiku_4_5"
            ]
        )
    )
    selected: list[dict[str, Any]] = []
    for role in ("quality", "evidence_risk"):
        eligible = sorted(
            [
                dict(row)
                for row in full_plan
                if row["candidate_key"] == "anthropic_claude_haiku_4_5"
                and row["role"] == role
            ],
            key=lambda row: canonical_json(row["record_ids"]),
        )
        if not eligible:
            raise RuntimeError(f"compatibility pilot lacks {role} coverage")
        row = eligible[0]
        parent_key = str(row["physical_call_key"])
        row["parent_full_matrix_physical_call_key"] = parent_key
        row["physical_call_key"] = physical_call_key(
            stage=COMPATIBILITY_PILOT_STAGE,
            record_ids=dict(row["record_ids"]),
            prompt_sha256=str(row["prompt_sha256"]),
            endpoint=endpoint,
            request_parameters=dict(row["request_parameters"]),
        )
        selected.append(row)
    if len(selected) != 2 or len(
        {str(row["physical_call_key"]) for row in selected}
    ) != 2:
        raise RuntimeError("compatibility pilot must contain exactly two calls")
    return selected


def code_manifest(root: Path) -> dict[str, Any]:
    """Bind preparation, freezing, planning, execution, and shared transports."""

    paths = {
        "aggregation": root
        / "scripts/v1_5/"
        "21zc_aggregate_role_decomposed_judge_qualification_v1_5.py",
        "execution": root
        / "scripts/v1_5/"
        "21zb_run_role_decomposed_judge_qualification_v1_5.py",
        "dry_run": root
        / "scripts/v1_5/"
        "21za_dry_run_role_decomposed_judge_qualification_v1_5.py",
        "freezer": root
        / "scripts/v1_5/"
        "21z_freeze_role_decomposed_human_anchor_v1_5.py",
        "preparer": root
        / "scripts/v1_5/"
        "21y_prepare_role_decomposed_judge_qualification_v1_5.py",
        "qualification": root
        / "src/metacom_pm/v1_5_role_decomposed_judge_qualification.py",
        "api": root / "src/metacom_pm/api.py",
        "attempt_ledger": root / "src/metacom_pm/attempt_ledger.py",
        "bounded_retry": root / "src/metacom_pm/bounded_retry.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "role-decomposed qualification code manifest is incomplete: "
            + ", ".join(missing)
        )
    return {
        name: {
            "relative_path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }


def build_cost_estimate(
    *,
    plan: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    code_manifest: Mapping[str, Any],
    endpoint_contract: Mapping[str, Any],
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    stage: str = STAGE,
) -> dict[str, Any]:
    attempts = int(
        endpoint_contract["request_contract"][
            "maximum_physical_attempts_per_logical_call"
        ]
    )
    one_attempt = math.fsum(
        float(row["maximum_single_attempt_cost_usd"]) for row in plan
    )
    by_candidate_role: dict[str, Any] = {}
    for candidate in EXPECTED_CANDIDATES:
        by_candidate_role[candidate] = {}
        for role in ("quality", "evidence_risk"):
            rows = [
                row
                for row in plan
                if row["candidate_key"] == candidate and row["role"] == role
            ]
            subtotal = math.fsum(
                float(row["maximum_single_attempt_cost_usd"]) for row in rows
            )
            by_candidate_role[candidate][role] = {
                "logical_calls": len(rows),
                "logical_single_attempt_cost_usd": subtotal,
                "maximum_cost_usd": subtotal * attempts,
            }
    payload = {
        "protocol": "pm-v1.5-role-decomposed-judge-qualification-cost-v1",
        "stage": str(stage),
        "qualification_contract_sha256": str(contract["contract_sha256"]),
        "qualification_contract": dict(contract),
        "code_manifest": dict(code_manifest),
        "logical_calls": len(plan),
        "maximum_physical_attempts_per_logical_call": attempts,
        "maximum_physical_attempts": len(plan) * attempts,
        "logical_single_attempt_cost_usd": one_attempt,
        "maximum_cost_usd": one_attempt * attempts,
        "by_candidate_role": by_candidate_role,
        "maximum_input_tokens_per_call_est": max(
            int(row["input_tokens_est"]) for row in plan
        ),
        "call_plan_sha256": sha256_text(canonical_json(list(plan))),
        "budget_limits": {
            "max_api_calls": max_api_calls,
            "max_estimated_usd": max_estimated_usd,
            "max_input_tokens_per_call": max_input_tokens_per_call,
        },
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    checks = {
        "api_calls": payload["maximum_physical_attempts"] <= max_api_calls,
        "estimated_cost_usd": payload["maximum_cost_usd"] <= max_estimated_usd,
        "max_input_tokens_per_call": payload[
            "maximum_input_tokens_per_call_est"
        ]
        <= max_input_tokens_per_call,
    }
    record = {
        **payload,
        "budget_gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    }
    return {
        **record,
        "cost_estimate_sha256": sha256_text(canonical_json(record)),
    }


def _normalized_preference(value: str, order_variant: int) -> str:
    value = str(value)
    if int(order_variant) == 1:
        return {"A": "B", "B": "A"}.get(value, value)
    return value


def aggregate_role_decomposed_qualification(
    *,
    plan_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    human_mapping_rows: Sequence[Mapping[str, Any]],
    human_annotation_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-outcome, role-specific qualification criteria."""

    plan = [dict(row) for row in plan_rows]
    results = [dict(row) for row in result_rows]
    if len(plan) != 288 or len(result_rows) != 288:
        raise RuntimeError("role-decomposed aggregation requires 288/288 calls")
    plan_by_key = {
        (
            str(row["candidate_key"]),
            str(row["role"]),
            canonical_json(row["record_ids"]),
        ): row
        for row in plan
    }
    result_by_key = {
        (
            str(row["candidate_key"]),
            str(row["role"]),
            canonical_json(row["record_ids"]),
        ): row
        for row in results
    }
    if len(plan_by_key) != 288 or set(result_by_key) != set(plan_by_key):
        raise RuntimeError("role-decomposed result coverage is not exact")
    audit_by_id = {str(row["audit_id"]): dict(row) for row in audit_rows}
    if len(audit_by_id) != 48:
        raise RuntimeError("role-decomposed audit source coverage drifted")
    mapping_by_blind = {
        str(row["blind_item_id"]): dict(row) for row in human_mapping_rows
    }
    human_by_blind = {
        str(row["blind_item_id"]): dict(row)
        for row in human_annotation_rows
    }
    if (
        len(mapping_by_blind) != 12
        or set(mapping_by_blind) != set(human_by_blind)
    ):
        raise RuntimeError("role-decomposed human anchor coverage drifted")
    criteria = dict(contract["pre_outcome_criteria"])
    candidate_reports: dict[str, Any] = {}
    for candidate_key in EXPECTED_CANDIDATES:
        quality_by_pair: dict[str, dict[int, RoleDecomposedQualityOutput]] = (
            defaultdict(dict)
        )
        audit_outputs: dict[str, RoleDecomposedEvidenceRiskOutput] = {}
        for key, result in result_by_key.items():
            if key[0] != candidate_key:
                continue
            plan_row = plan_by_key[key]
            if key[1] == "quality":
                parsed = RoleDecomposedQualityOutput.model_validate(
                    result["parsed"], strict=True
                )
                record_ids = dict(plan_row["record_ids"])
                quality_by_pair[str(record_ids["pair_id"])][
                    int(record_ids["order_variant"])
                ] = parsed
            else:
                parsed = RoleDecomposedEvidenceRiskOutput.model_validate(
                    result["parsed"], strict=True
                )
                validate_audit_output_dimensions(
                    parsed, list(plan_row["applicable_dimensions"])
                )
                audit_outputs[
                    str(dict(plan_row["record_ids"])["audit_id"])
                ] = parsed
        if len(quality_by_pair) != 24 or any(
            set(rows) != {0, 1} for rows in quality_by_pair.values()
        ):
            raise RuntimeError("role-decomposed quality result matrix drifted")
        if set(audit_outputs) != set(audit_by_id):
            raise RuntimeError("role-decomposed audit result matrix drifted")

        consistency: dict[str, float] = {}
        informative: dict[str, float] = {}
        for field in ("overall_preference", "support_quality_preference"):
            matches = 0
            informative_calls = 0
            for rows in quality_by_pair.values():
                first = _normalized_preference(
                    getattr(rows[0], field), 0
                )
                second = _normalized_preference(
                    getattr(rows[1], field), 1
                )
                matches += int(first == second)
                informative_calls += int(first in {"A", "B"})
                informative_calls += int(second in {"A", "B"})
            consistency[field] = matches / 24
            informative[field] = informative_calls / 48

        total_findings = 0
        abstentions = 0
        excerpt_checks = 0
        excerpt_valid = 0
        for audit_id, parsed in audit_outputs.items():
            source = audit_by_id[audit_id]
            response = str(source["response"])
            evidence_surface = {
                "visible_state": source["visible_state"],
                "authorized_user_context": source[
                    "authorized_user_context"
                ],
                "selected_context": source["selected_context"],
            }
            for finding in parsed.findings:
                total_findings += 1
                abstentions += int(
                    finding.verdict == "insufficient_evidence"
                )
                excerpt_checks += 2
                response_ok = (
                    finding.response_excerpt in {"[none]", "[omission]"}
                    or finding.response_excerpt in response
                )
                evidence_ok = (
                    evidence_excerpt_is_exact(
                        finding.evidence_excerpt, evidence_surface
                    )
                )
                excerpt_valid += int(response_ok) + int(evidence_ok)
        audit_applicability_exact_rate = 1.0
        audit_excerpt_validity = excerpt_valid / excerpt_checks
        audit_abstention_rate = abstentions / total_findings

        human_quality_matches = {
            "overall_preference": 0,
            "support_quality_preference": 0,
        }
        human_quality_total = 0
        human_audit_matches = 0
        human_audit_total = 0
        for blind_id, mapping in mapping_by_blind.items():
            human = human_by_blind[blind_id]
            pair_id = str(mapping["pair_id"])
            order_variant = int(mapping["order_variant"])
            model_quality = quality_by_pair[pair_id][order_variant]
            for field in human_quality_matches:
                human_quality_matches[field] += int(
                    getattr(model_quality, field) == str(human[field])
                )
            human_quality_total += 1
            for displayed in ("a", "b"):
                audit_id = str(
                    mapping[f"candidate_{displayed}_audit_id"]
                )
                model_findings = {
                    row.dimension: row
                    for row in audit_outputs[audit_id].findings
                }
                for finding in human[f"candidate_{displayed}_audits"]:
                    dimension = str(finding["dimension"])
                    human_audit_matches += int(
                        model_findings[dimension].verdict
                        == str(finding["verdict"])
                    )
                    human_audit_total += 1
        quality_pass = (
            all(
                value
                >= float(
                    criteria[
                        "minimum_quality_ab_ba_consistency_each_field"
                    ]
                )
                for value in consistency.values()
            )
            and all(
                value >= float(criteria["minimum_quality_informative_rate"])
                for value in informative.values()
            )
        )
        audit_pass = (
            audit_applicability_exact_rate
            == float(criteria["audit_applicability_exact_rate"])
            and audit_excerpt_validity
            >= float(criteria["minimum_audit_exact_excerpt_validity"])
            and audit_abstention_rate
            <= float(criteria["maximum_audit_abstention_rate"])
        )
        candidate_reports[candidate_key] = {
            "quality": {
                "status": "SUPPORTED" if quality_pass else "NOT_SUPPORTED",
                "ab_ba_consistency": consistency,
                "informative_rate": informative,
                "human_anchor_exact_agreement_descriptive_only": {
                    field: count / human_quality_total
                    for field, count in human_quality_matches.items()
                },
            },
            "evidence_risk": {
                "status": "SUPPORTED" if audit_pass else "NOT_SUPPORTED",
                "applicability_exact_rate": audit_applicability_exact_rate,
                "exact_excerpt_validity": audit_excerpt_validity,
                "abstention_rate": audit_abstention_rate,
                "human_anchor_verdict_agreement_descriptive_only": (
                    human_audit_matches / human_audit_total
                ),
            },
        }
    supported_quality = [
        key
        for key, report in candidate_reports.items()
        if report["quality"]["status"] == "SUPPORTED"
    ]
    supported_audit = [
        key
        for key, report in candidate_reports.items()
        if report["evidence_risk"]["status"] == "SUPPORTED"
    ]
    return {
        "protocol": (
            "pm-v1.5-role-decomposed-judge-qualification-aggregation-v1"
        ),
        "status": (
            "ROLE_SPECIFIC_CANDIDATES_SUPPORTED_RESEARCHER_SIGNOFF_REQUIRED"
            if supported_quality and supported_audit
            else "ROLE_DECOMPOSED_INSTRUMENT_NOT_SUPPORTED"
        ),
        "candidate_reports": candidate_reports,
        "supported_quality_candidates": supported_quality,
        "supported_evidence_risk_candidates": supported_audit,
        "human_anchor_role": (
            "single_researcher_independent_reference_not_automatic_gold"
        ),
        "automatic_gold_created": False,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
        "researcher_signoff_required": True,
    }
