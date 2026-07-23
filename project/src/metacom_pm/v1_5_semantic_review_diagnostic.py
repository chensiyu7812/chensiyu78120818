"""Diagnostic-only decomposition of the failed PM-v1.5 V4 semantic review.

This module deliberately does not modify or replace the formal V4 gate.  It
reuses the already-observed V4 calibration cases to test a narrower causal
hypothesis: can a judge evaluate one candidate claim when that claim is kept
separate from the only evidence it is allowed to use?

There is no endpoint resolution, paid-run release, cost estimate, run identity,
or HTTP client in this module.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import Field

from .api import normalize_provider_finish_reason
from .io import canonical_json, sha256_text
from .text import estimate_tokens
from .pm_v2_contracts import StrictModel
from .pm_v2_generation_review_v8 import RATING_FIELDS, V8ReviewCase
from .v1_5_automated_semantic_review import (
    _corrupt_pilot_case,
    _render_case_text,
)


DIAGNOSTIC_PROTOCOL = (
    "pm-v1.5-balanced-proxy-measurement-diagnostic-v4.2-first-paper-scoped"
)
DIAGNOSTIC_STATUS = "DIAGNOSTIC_PACKET_ONLY_NO_API_NOT_A_GATE"
DIAGNOSTIC_RUN_PROTOCOL = (
    "pm-v1.5-balanced-proxy-measurement-runner-v4.2-first-paper-scoped"
)
DIAGNOSTIC_RUN_STAGE = "development_v4_root_cause_single_field_diagnostic"
DIAGNOSTIC_RUN_STATUS = "DIAGNOSTIC_COMPLETE_NOT_A_FORMAL_GATE"

ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE = (
    "ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE"
)
PROMPT_VARIANTS = (ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE,)
DIAGNOSTIC_POLARITIES = ("positive", "negative")

DIAGNOSTIC_FIELDS = (
    "source_type_match",
    "dialogue_temporal_order_match",
    "context_grounding_match",
    "memory_age_design_match",
    "advice_readiness_match",
    "surface_naturalness_match",
)

DETERMINISTIC_DIAGNOSTIC_FIELDS = (
    "source_type_match",
    "dialogue_temporal_order_match",
    "memory_age_design_match",
    "surface_naturalness_match",
)
SEMANTIC_DIAGNOSTIC_FIELDS = tuple(
    field for field in DIAGNOSTIC_FIELDS if field not in DETERMINISTIC_DIAGNOSTIC_FIELDS
)


class FieldReviewSpecification(StrictModel):
    field: str
    evaluation_mode: Literal[
        "deterministic_code",
        "semantic_single_claim",
        "hybrid_code_then_semantic",
        "derived_from_validated_primitives",
        "semantic_aggregate_after_primitives",
    ]
    claim: str = Field(min_length=1)
    allowed_evidence: tuple[str, ...]
    prohibited_evidence: tuple[str, ...]
    dependencies: tuple[str, ...]
    definition: str = Field(min_length=1)
    guaranteed_control: str = Field(min_length=1)


FIELD_REVIEW_SPECIFICATIONS: dict[str, FieldReviewSpecification] = {
    "semantic_family_match": FieldReviewSpecification(
        field="semantic_family_match",
        evaluation_mode="semantic_single_claim",
        claim="The proposed semantic family describes the visible dialogue and summary.",
        allowed_evidence=("current_user_text", "history", "session_summary"),
        prohibited_evidence=("regime", "utility", "coverage_rationale"),
        dependencies=(),
        definition=(
            "Compare the proposed family itself with the main topic expressed by "
            "the current turn, prior turns, and summary; do not merely check that "
            "those three texts are mutually coherent."
        ),
        guaranteed_control=(
            "Replace the proposed family with a predefined orthogonal family whose "
            "definition conflicts with every visible text section."
        ),
    ),
    "regime_match": FieldReviewSpecification(
        field="regime_match",
        evaluation_mode="derived_from_validated_primitives",
        claim="The resource regime equals the mapping from validated primitive claims.",
        allowed_evidence=(
            "validated_memory_source_value",
            "validated_strategy_resource_value",
            "validated_advice_readiness",
            "validated_risk",
        ),
        prohibited_evidence=("candidate_regime", "coverage_rationale"),
        dependencies=(
            "memory_sources_marginal_value_match",
            "strategy_resource_need_match",
            "advice_readiness_match",
        ),
        definition=(
            "Regime is a deterministic development label derived after primitive "
            "resource-value and readiness claims pass; it is not a free-form LLM judgment."
        ),
        guaranteed_control=(
            "Change one validated primitive and require the deterministic regime "
            "mapping to change; do not rotate an arbitrary donor regime."
        ),
    ),
    "memory_sources_marginal_value_match": FieldReviewSpecification(
        field="memory_sources_marginal_value_match",
        evaluation_mode="semantic_aggregate_after_primitives",
        claim="The proposed useful-source set matches nonredundant source-level value.",
        allowed_evidence=(
            "visible_context",
            "memory_item_text",
            "memory_source",
            "stale_conflict_sensitivity",
            "validated_item_utility",
        ),
        prohibited_evidence=("regime", "coverage_rationale", "unvalidated_utility"),
        dependencies=("memory_item_utility_match", "source_type_match"),
        definition=(
            "A source belongs in the proposed set only when at least one validated "
            "item adds material, nonredundant value and the source is not net harmful."
        ),
        guaranteed_control=(
            "Remove the sole source with a clearly helpful item or add a source "
            "whose items are all clearly irrelevant or harmful."
        ),
    ),
    "memory_item_utility_match": FieldReviewSpecification(
        field="memory_item_utility_match",
        evaluation_mode="semantic_single_claim",
        claim="One specified memory item's proposed utility fits the visible state.",
        allowed_evidence=(
            "visible_context",
            "one_memory_item_text",
            "source_stale_conflict_sensitivity",
        ),
        prohibited_evidence=(
            "regime",
            "source_level_useful_set",
            "other_item_utility",
            "coverage_rationale",
        ),
        dependencies=(),
        definition=(
            "Helpful means the item adds material current value; irrelevant means "
            "no material current value; harmful means using it risks contradiction, "
            "intrusion, staleness, or inappropriate steering."
        ),
        guaranteed_control=(
            "Label an orthogonal item helpful or label an item that directly "
            "conflicts with the user's current correction helpful."
        ),
    ),
    "source_type_match": FieldReviewSpecification(
        field="source_type_match",
        evaluation_mode="deterministic_code",
        claim="One specified memory item's proposed MP/MS/ME source matches its frozen record metadata.",
        allowed_evidence=("memory_id", "registered_source"),
        prohibited_evidence=("regime", "utility", "display_section"),
        dependencies=(),
        definition=(
            "Source type is a property of the frozen memory record and construction "
            "contract. Do not ask an LLM to infer an exclusive source taxonomy from "
            "one potentially ambiguous natural-language sentence."
        ),
        guaranteed_control=(
            "Assign a stable preference to MS/ME, a repeated pattern to MP/ME, "
            "or a concrete event to MP/MS."
        ),
    ),
    "dialogue_temporal_order_match": FieldReviewSpecification(
        field="dialogue_temporal_order_match",
        evaluation_mode="deterministic_code",
        claim="History is prior to current, role-valid, and free of copied-current text.",
        allowed_evidence=("indexed_history", "current_user_text"),
        prohibited_evidence=("regime", "utility", "summary_rationale"),
        dependencies=(),
        definition=(
            "Every history turn must precede current, roles must follow the frozen "
            "dialogue contract, and no prior user turn may duplicate current."
        ),
        guaranteed_control=(
            "Copy current into a prior user turn, insert a future turn, or break the role sequence."
        ),
    ),
    "context_grounding_match": FieldReviewSpecification(
        field="context_grounding_match",
        evaluation_mode="hybrid_code_then_semantic",
        claim="One displayed summary or authorized-context claim is evidence-grounded.",
        allowed_evidence=(
            "history",
            "current_user_text",
            "authorized_user_context",
            "claim_provenance",
        ),
        prohibited_evidence=("coverage_rationale", "regime", "utility"),
        dependencies=(),
        definition=(
            "First require a valid provenance pointer; then determine whether the "
            "pointed-to evidence entails the candidate claim."
        ),
        guaranteed_control=(
            "Replace the summary with an orthogonal donor summary while retaining "
            "the original provenance, or remove the only supporting evidence."
        ),
    ),
    "memory_age_design_match": FieldReviewSpecification(
        field="memory_age_design_match",
        evaluation_mode="deterministic_code",
        claim="One memory's proposed age is causal and arithmetically correct.",
        allowed_evidence=("current_session", "created_session", "proposed_age"),
        prohibited_evidence=("regime", "utility", "memory_topic"),
        dependencies=(),
        definition=(
            "Age must equal current_session minus created_session, must be nonnegative, "
            "and created_session cannot be in the future."
        ),
        guaranteed_control="Add one to age, use a negative age, or place creation in the future.",
    ),
    "strategy_resource_need_match": FieldReviewSpecification(
        field="strategy_resource_need_match",
        evaluation_mode="semantic_aggregate_after_primitives",
        claim="The proposed RS marginal-value decision fits the visible state and actual cards.",
        allowed_evidence=(
            "visible_context",
            "base_generator_capability",
            "actual_strategy_card_text",
            "validated_strategy_item_utility",
        ),
        prohibited_evidence=("regime", "coverage_rationale", "unvalidated_readiness"),
        dependencies=("strategy_item_utility_match", "advice_readiness_match"),
        definition=(
            "Use RS only when actual retrieved evidence adds material value beyond "
            "the base generator; skip when it is redundant, irrelevant, or risky."
        ),
        guaranteed_control=(
            "Propose use when every actual card is clearly irrelevant/overdirective, "
            "or propose skip when a directly fitting card adds otherwise unavailable value."
        ),
    ),
    "strategy_item_utility_match": FieldReviewSpecification(
        field="strategy_item_utility_match",
        evaluation_mode="semantic_single_claim",
        claim="One specified Strategy card's proposed utility fits the visible dialogue.",
        allowed_evidence=("visible_context", "one_strategy_card_text", "visible_boundary"),
        prohibited_evidence=("regime", "strategy_resource_target", "other_card_utility"),
        dependencies=(),
        definition=(
            "Judge the specified card, not whether every retrieved card should be used; "
            "irrelevant and harmful candidate cards may legitimately exist in inventory."
        ),
        guaranteed_control=(
            "Mark a rigid immediate plan helpful under an explicit listen-only boundary, "
            "or mark a directly requested low-pressure card irrelevant."
        ),
    ),
    "advice_readiness_match": FieldReviewSpecification(
        field="advice_readiness_match",
        evaluation_mode="semantic_single_claim",
        claim="The proposed directiveness level is supported by the visible dialogue.",
        allowed_evidence=("current_user_text", "history", "session_summary"),
        prohibited_evidence=("regime", "strategy_resource_target", "coverage_rationale"),
        dependencies=(),
        definition=(
            "These labels describe desired directiveness, not whether Strategy RAG "
            "is enabled. listen_only means the user asks to be heard and declines "
            "suggestions for this turn. explore_first means the user wants help "
            "naming, understanding, or untangling the situation before choosing an "
            "action; the literal words 'ready' or 'explore' are not required. "
            "light_suggestion means the user asks for or welcomes one gentle, "
            "low-pressure idea without a detailed plan. structured_plan means the "
            "user explicitly asks for concrete ordered steps or implementation help. "
            "ambiguous means there is no reliable directiveness signal."
        ),
        guaranteed_control=(
            "Mark an explicit request to be heard as structured planning, or an "
            "explicit request for one gentle idea as listen-only."
        ),
    ),
    "surface_naturalness_match": FieldReviewSpecification(
        field="surface_naturalness_match",
        evaluation_mode="deterministic_code",
        claim="The visible surface contains none of the frozen explicit data-generation or label-leakage markers.",
        allowed_evidence=("current_user_text", "history", "session_summary"),
        prohibited_evidence=("regime_metadata", "utility_metadata"),
        dependencies=(),
        definition=(
            "This diagnostic checks explicit author-facing leakage markers only. "
            "Broader conversational naturalness remains a disclosed qualitative limitation."
        ),
        guaranteed_control=(
            "Insert explicit generated-user/resource-condition wording or break natural role flow."
        ),
    ),
}


# Real evidence from actual-468 (development_actual_corpus_semantic_review):
# an unbounded schema let one real advice_readiness_match call generate 4515
# characters of real content and hit finish_reason=length mid-answer at
# max_tokens=900 -- not a random flake, a genuine unbounded-output-shape gap.
# These bounds are a frozen output contract, not a semantic change to what
# counts as supported/not_supported: a single atomic claim's evidence rarely
# needs more than a handful of distinct quotes, and StrictModel/Pydantic
# validation (not silent truncation) is what enforces this either way.
MAX_EVIDENCE_ITEMS = 4
MAX_EVIDENCE_QUOTE_CHARS = 320
MAX_REASON_CHARS = 600


class SingleFieldDiagnosticOutput(StrictModel):
    verdict: Literal["supported", "not_supported"]
    evidence_keys: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    evidence_quotes: list[
        Annotated[str, Field(max_length=MAX_EVIDENCE_QUOTE_CHARS)]
    ] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)


_COMMON_SYSTEM = (
    "Evaluate exactly one atomic candidate claim against the exhaustive evidence "
    "provided for that claim. The candidate may be true or false. Return exactly "
    "one JSON object with verdict, evidence_keys, evidence_quotes, and reason. "
    "evidence_keys and evidence_quotes must be aligned arrays; every key must name "
    "a top-level key in exhaustive_evidence and every quote must occur in that "
    "evidence value. Cite every distinct evidence section needed for a relational "
    "judgment. Use supported only when the exhaustive evidence entails every "
    "required part of the claim. Use not_supported whenever the evidence conflicts "
    "with the claim OR does not establish every required part. Do not distinguish "
    "contradiction from missing support in the verdict. Numerical closeness, topical "
    "overlap, or a plausible guess is not support. The candidate claim, metadata, "
    "and omitted labels are not evidence. "
    f"Cite at most {MAX_EVIDENCE_ITEMS} evidence sections: use the fewest quotes "
    "that establish the claim, never every available section. Each evidence_quotes "
    f"entry must be the shortest exact substring that proves its point, at most "
    f"{MAX_EVIDENCE_QUOTE_CHARS} characters -- never quote an entire field verbatim. "
    f"reason must be at most {MAX_REASON_CHARS} characters."
)


def maximum_legal_single_field_diagnostic_output_tokens() -> int:
    """Computed (not eyeballed) worst-case token estimate for a maximally-
    sized, schema-legal SingleFieldDiagnosticOutput -- every list at its
    MAX_EVIDENCE_ITEMS cap, every quote at MAX_EVIDENCE_QUOTE_CHARS, reason
    at MAX_REASON_CHARS. response_max_tokens for this scope should be
    derived from this, with real margin, rather than raised by guesswork
    after each real truncation."""

    longest_verdict = max(("supported", "not_supported"), key=len)
    worst_case = SingleFieldDiagnosticOutput(
        verdict=longest_verdict,
        evidence_keys=["x" * 32] * MAX_EVIDENCE_ITEMS,
        evidence_quotes=["x" * MAX_EVIDENCE_QUOTE_CHARS] * MAX_EVIDENCE_ITEMS,
        reason="x" * MAX_REASON_CHARS,
    )
    return estimate_tokens(canonical_json(worst_case.model_dump(mode="json")))


OFFLINE_LENGTH_BOUND_RECOVERY_PROTOCOL = (
    "pm-v1.5-actual-468-offline-length-bound-recovery-v1"
)


class RecoveredSingleFieldDiagnosticOutput(StrictModel):
    """Offline-recovery-only counterpart to SingleFieldDiagnosticOutput.

    Never used to generate a live provider JSON schema and never sent as a
    request contract -- SingleFieldDiagnosticOutput's max_length constraints
    on reason/evidence_quotes are load-bearing for two live concerns this
    model must never touch: the provider request schema, and
    maximum_legal_single_field_diagnostic_output_tokens()'s worst-case token
    preflight. Every other constraint (verdict enum, <=4 evidence items,
    non-empty strings, forbid extra fields) is kept identical, so this can
    only recover a response that was already a complete, well-formed answer
    rejected purely for exceeding an arbitrary character count.
    """

    verdict: Literal["supported", "not_supported"]
    evidence_keys: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    evidence_quotes: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    reason: str = Field(min_length=1)


def recover_length_bound_failure(
    *,
    raw_provider_response: Mapping[str, Any],
    parsed_payload: Mapping[str, Any],
) -> RecoveredSingleFieldDiagnosticOutput:
    """Locally re-validate an already-received response, ignoring only length.

    Fails closed (raises ValueError) unless the provider's own finish reason
    was a normal completion -- this never recovers a genuinely truncated
    response -- and the payload still satisfies every other constraint.
    """

    _, normalized_finish_reason = normalize_provider_finish_reason(raw_provider_response)
    if normalized_finish_reason != "complete":
        raise ValueError(
            "refusing to recover a response whose provider finish reason was "
            f"not a normal completion: {normalized_finish_reason!r}"
        )
    return RecoveredSingleFieldDiagnosticOutput.model_validate(parsed_payload)


def validate_field_review_specifications() -> None:
    if list(FIELD_REVIEW_SPECIFICATIONS) != list(RATING_FIELDS):
        raise RuntimeError("field review specifications must follow the 12-field rubric")
    known = set(FIELD_REVIEW_SPECIFICATIONS)
    for field, spec in FIELD_REVIEW_SPECIFICATIONS.items():
        if spec.field != field:
            raise RuntimeError(f"field specification key mismatch: {field}")
        if field in spec.dependencies or not set(spec.dependencies) <= known:
            raise RuntimeError(f"invalid dependencies for {field}")
        if set(spec.allowed_evidence) & set(spec.prohibited_evidence):
            raise RuntimeError(f"allowed/prohibited evidence overlap for {field}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(field: str) -> None:
        if field in visiting:
            raise RuntimeError("field review dependency graph contains a cycle")
        if field in visited:
            return
        visiting.add(field)
        for dependency in FIELD_REVIEW_SPECIFICATIONS[field].dependencies:
            visit(dependency)
        visiting.remove(field)
        visited.add(field)

    for field in FIELD_REVIEW_SPECIFICATIONS:
        visit(field)
    if FIELD_REVIEW_SPECIFICATIONS["regime_match"].evaluation_mode != (
        "derived_from_validated_primitives"
    ):
        raise RuntimeError("regime must be derived from validated primitives")


validate_field_review_specifications()


def _visible_context(case: V8ReviewCase) -> dict[str, Any]:
    return {
        "history": [
            {"role": turn.role, "content": turn.content}
            for turn in case.dialogue_before_current
        ],
        "current_user_text": case.current_user_text,
        "session_summary": case.session_summary,
    }


def _all_memories(case: V8ReviewCase) -> list[Any]:
    return [
        *case.profile_memories,
        *case.summary_memories,
        *case.event_memories,
    ]


def _memory_by_id(case: V8ReviewCase, memory_id: str) -> Any:
    matches = [item for item in _all_memories(case) if item.memory_id == memory_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one diagnostic memory item: {memory_id}")
    return matches[0]


def _claim_and_evidence(
    case: V8ReviewCase,
    *,
    field: str,
    override: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    visible = _visible_context(case)
    if field == "source_type_match":
        memory_id = str(override["memory_id"])
        item = _memory_by_id(case, memory_id)
        return (
            {"memory_id": memory_id, "proposed_source": str(override["source"])},
            {"registered_source": item.source.value},
        )
    if field == "dialogue_temporal_order_match":
        return (
            {"history_is_strictly_prior_role_valid_and_uncopied": True},
            {
                "indexed_history": [
                    {"index": index, "role": turn.role, "content": turn.content}
                    for index, turn in enumerate(case.dialogue_before_current)
                ],
                "current_user_text": case.current_user_text,
            },
        )
    if field == "context_grounding_match":
        return (
            {
                "session_summary": case.session_summary,
                "proposed_grounding": "grounded",
            },
            {
                "history": visible["history"],
                "current_user_text": case.current_user_text,
                "authorized_user_context": case.authorized_user_context,
                "claim_provenance": [
                    {
                        # ``row.claim`` is intentionally excluded.  In the V4
                        # source artifact it restates the intended answer and
                        # would therefore turn provenance into label leakage.
                        "target_field": row.target_field,
                        "provenance_type": row.provenance_type,
                        "provenance_source": row.provenance_source,
                        "provenance_session": row.provenance_session,
                        "evidence_quote": row.evidence_quote,
                    }
                    for row in case.context_provenance
                ],
            },
        )
    if field == "memory_age_design_match":
        memory_id = str(override["memory_id"])
        item = _memory_by_id(case, memory_id)
        return (
            {"memory_id": memory_id, "proposed_age": int(override["age_sessions"])},
            {
                "memory_id": memory_id,
                "current_session": int(case.session_index),
                "created_session": int(item.created_session),
            },
        )
    if field == "advice_readiness_match":
        return (
            {"proposed_advice_readiness": str(override["advice_readiness"])},
            visible,
        )
    if field == "surface_naturalness_match":
        return (
            {"surface_is_natural_and_free_of_generation_leakage": True},
            visible,
        )
    raise RuntimeError(f"field is not in the minimal V4 diagnostic cohort: {field}")


def _positive_override(
    case: V8ReviewCase,
    *,
    field: str,
    negative_override: Mapping[str, Any],
) -> dict[str, Any]:
    if field == "source_type_match":
        memory_id = str(negative_override["memory_id"])
        return {
            "memory_id": memory_id,
            "source": _memory_by_id(case, memory_id).source.value,
        }
    if field == "memory_age_design_match":
        memory_id = str(negative_override["memory_id"])
        return {
            "memory_id": memory_id,
            "age_sessions": int(_memory_by_id(case, memory_id).age_sessions),
        }
    if field == "advice_readiness_match":
        return {"advice_readiness": case.strategy_target.advice_readiness}
    return {}


def _minimum_evidence_citations(field: str) -> int:
    # Grounding compares a proposed summary with the dialogue evidence and its
    # provenance. The other atomic claims can be decided from one section.
    return 2 if field == "context_grounding_match" else 1


def single_field_diagnostic_messages(
    *,
    field: str,
    candidate_claim: Mapping[str, Any],
    allowed_evidence: Mapping[str, Any],
    variant: str = ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE,
) -> list[dict[str, str]]:
    if field not in DIAGNOSTIC_FIELDS:
        raise RuntimeError(f"field is not diagnostic-enabled: {field}")
    if variant not in PROMPT_VARIANTS:
        raise RuntimeError(f"unknown diagnostic prompt variant: {variant}")
    payload: dict[str, Any] = {
        "field": field,
        "field_definition": FIELD_REVIEW_SPECIFICATIONS[field].definition,
        "candidate_claim": dict(candidate_claim),
        "exhaustive_evidence": dict(allowed_evidence),
        "valid_evidence_keys": list(allowed_evidence),
        "minimum_distinct_evidence_citations": _minimum_evidence_citations(field),
        "decision_rule": (
            "supported only when the exhaustive evidence entails every required "
            "part of the candidate; not_supported when any required part conflicts "
            "with the evidence or is not established by the listed evidence"
        ),
    }
    messages = [
        {"role": "system", "content": _COMMON_SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ]
    serialized = canonical_json(messages)
    for forbidden in (
        "expected_verdict",
        "source_control_id",
        "corrupted_field",
        "coverage_rationale",
        "polarity",
    ):
        if forbidden in serialized:
            raise RuntimeError(f"diagnostic prompt leaked {forbidden}")
    return messages


def expected_verdict_for_polarity(polarity: str) -> str:
    if polarity == "positive":
        return "supported"
    if polarity == "negative":
        return "not_supported"
    raise RuntimeError(f"unknown diagnostic polarity: {polarity}")


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def evaluate_deterministic_diagnostic_item(item: Mapping[str, Any]) -> dict[str, Any]:
    field = str(item.get("field") or "")
    if field not in DETERMINISTIC_DIAGNOSTIC_FIELDS:
        raise RuntimeError("semantic item cannot be evaluated as deterministic code")
    evidence = item.get("allowed_evidence")
    claim = item.get("candidate_claim")
    if not isinstance(evidence, Mapping) or not isinstance(claim, Mapping):
        raise RuntimeError("deterministic diagnostic item lacks claim/evidence")
    if field == "source_type_match":
        registered_source = str(evidence.get("registered_source") or "")
        proposed_source = str(claim.get("proposed_source") or "")
        code_result = bool(registered_source) and proposed_source == registered_source
        details = {
            "registered_source": registered_source,
            "proposed_source": proposed_source,
            "source_registration_match": code_result,
            "truth_source": "frozen_memory_record_metadata",
        }
    elif field == "dialogue_temporal_order_match":
        history = evidence.get("indexed_history")
        if not isinstance(history, list):
            raise RuntimeError("temporal diagnostic lacks indexed history")
        roles_valid = all(
            isinstance(row, Mapping)
            and int(row.get("index", -1)) == index
            and row.get("role") == ("user" if index % 2 == 0 else "assistant")
            for index, row in enumerate(history)
        )
        ends_with_assistant = bool(history) and history[-1].get("role") == "assistant"
        current = _normalized_text(evidence.get("current_user_text", ""))
        current_not_copied = bool(current) and all(
            _normalized_text(row.get("content", "")) != current
            for row in history
            if isinstance(row, Mapping)
        )
        code_result = roles_valid and ends_with_assistant and current_not_copied
        details = {
            "roles_alternate_from_user": roles_valid,
            "history_ends_with_assistant": ends_with_assistant,
            "current_text_absent_from_history": current_not_copied,
        }
    elif field == "memory_age_design_match":
        current_session = int(evidence["current_session"])
        created_session = int(evidence["created_session"])
        proposed_age = int(claim["proposed_age"])
        nonnegative_causal_order = current_session >= created_session
        arithmetic_match = proposed_age == current_session - created_session
        code_result = nonnegative_causal_order and arithmetic_match
        details = {
            "nonnegative_causal_order": nonnegative_causal_order,
            "arithmetic_match": arithmetic_match,
            "computed_age": current_session - created_session,
        }
    else:
        visible_text = " ".join(
            _normalized_text(value)
            for value in (
                evidence.get("current_user_text", ""),
                evidence.get("history", ""),
                evidence.get("session_summary", ""),
            )
        )
        prohibited_markers = (
            "as the generated user",
            "resource-condition example",
            "data-generation example",
            "regime label",
            "utility label",
        )
        observed_markers = [
            marker for marker in prohibited_markers if marker in visible_text
        ]
        code_result = not observed_markers
        details = {
            "explicit_generation_leakage_markers": observed_markers,
            "marker_contract": list(prohibited_markers),
            "truth_source": "explicit_surface_marker_lint",
        }
    expected = bool(item.get("expected_code_result"))
    return {
        "diagnostic_id": str(item.get("diagnostic_id") or ""),
        "field": field,
        "polarity": str(item.get("polarity") or ""),
        "code_result": code_result,
        "expected_code_result": expected,
        "correct": code_result is expected,
        "details": details,
    }


def assess_single_field_diagnostic_output(
    *,
    item: Mapping[str, Any],
    output: SingleFieldDiagnosticOutput,
) -> dict[str, Any]:
    """Score verdict and citation integrity as separate diagnostic dimensions.

    Citation defects are observations about judge contract adherence, not
    transport failures. The diagnostic runner records them and continues the
    frozen matrix; formal gates may still call the strict validator below.
    """

    allowed_evidence = item.get("allowed_evidence")
    if not isinstance(allowed_evidence, Mapping):
        raise RuntimeError("diagnostic item lacks allowed evidence")
    citation_errors: list[str] = []
    if len(output.evidence_keys) != len(output.evidence_quotes):
        citation_errors.append("evidence_keys_and_quotes_not_aligned")
    if len(set(output.evidence_keys)) != len(output.evidence_keys):
        citation_errors.append("evidence_keys_not_distinct")
    minimum = int(item.get("minimum_evidence_citations") or 1)
    if len(output.evidence_keys) < minimum:
        citation_errors.append("too_few_evidence_sections")
    for evidence_key, evidence_quote in zip(
        output.evidence_keys, output.evidence_quotes
    ):
        if evidence_key not in allowed_evidence:
            citation_errors.append(f"unknown_evidence_key:{evidence_key}")
            continue
        quote = _normalized_text(evidence_quote)
        evidence_text = _normalized_text(canonical_json(allowed_evidence[evidence_key]))
        if not quote or quote not in evidence_text:
            citation_errors.append(f"quote_absent_from_cited_evidence:{evidence_key}")
    expected = str(item.get("expected_verdict") or "")
    if expected not in {"supported", "not_supported"}:
        raise RuntimeError("semantic diagnostic item lacks a frozen expected verdict")
    verdict_correct = output.verdict == expected
    citation_valid = not citation_errors
    return {
        "polarity": str(item.get("polarity") or ""),
        "verdict": output.verdict,
        "expected_verdict": expected,
        "verdict_correct": verdict_correct,
        "citation_valid": citation_valid,
        "correct": verdict_correct and citation_valid,
        "false_positive": expected == "not_supported" and output.verdict == "supported",
        "false_negative": expected == "supported" and output.verdict != "supported",
        "evidence_attribution_valid": citation_valid,
        "citation_errors": citation_errors,
        "evidence_citation_count": len(output.evidence_keys),
    }


def validate_single_field_diagnostic_output(
    *,
    item: Mapping[str, Any],
    output: SingleFieldDiagnosticOutput,
) -> dict[str, Any]:
    """Strict wrapper for formal callers and unit-level contract checks."""

    assessment = assess_single_field_diagnostic_output(item=item, output=output)
    errors = list(assessment["citation_errors"])
    if errors:
        first = errors[0]
        messages = {
            "evidence_keys_and_quotes_not_aligned": (
                "diagnostic evidence keys and quotes are not aligned"
            ),
            "evidence_keys_not_distinct": (
                "diagnostic evidence keys must be distinct"
            ),
            "too_few_evidence_sections": (
                "diagnostic output cites too few evidence sections"
            ),
        }
        if first.startswith("unknown_evidence_key:"):
            raise RuntimeError(
                "diagnostic output cites a prohibited or unknown evidence key"
            )
        if first.startswith("quote_absent_from_cited_evidence:"):
            raise RuntimeError(
                "diagnostic evidence quote is absent from the cited evidence"
            )
        raise RuntimeError(messages.get(first, f"diagnostic citation failure: {first}"))
    return assessment


def build_v4_single_field_diagnostic_items(
    cases: Sequence[V8ReviewCase],
    controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build paired positive/negative items from one observed V4 control per field.

    The original full-case rendering must exactly match the already-observed V4
    control before it can be transformed.  This prevents a later diagnostic
    code change from silently substituting a different calibration example.
    """

    case_index = {case.item_id: case for case in cases}
    if len(case_index) != len(cases):
        raise RuntimeError("diagnostic cases have duplicate item IDs")
    items: list[dict[str, Any]] = []
    for field in DIAGNOSTIC_FIELDS:
        matches = [
            dict(control)
            for control in controls
            if control.get("corrupted_field") == field
        ]
        if len(matches) != 2:
            raise RuntimeError(f"expected exactly two V4 controls for {field}")
        control = matches[0]
        case_id = str(control["case_item_id"])
        if case_id not in case_index:
            raise RuntimeError(f"diagnostic control references unknown case: {case_id}")
        base_case = case_index[case_id]
        corrupted_case, override = _corrupt_pilot_case(
            base_case,
            field=field,
            donors=cases,
        )
        if override != control.get("override"):
            raise RuntimeError(f"diagnostic control override drift for {field}")
        rendered = _render_case_text(corrupted_case)
        if rendered != str(control.get("case_text") or ""):
            raise RuntimeError(f"diagnostic control rendering drift for {field}")
        positive_override = _positive_override(
            base_case, field=field, negative_override=override
        )
        for polarity, selected_case, selected_override in (
            ("positive", base_case, positive_override),
            ("negative", corrupted_case, override),
        ):
            candidate_claim, allowed_evidence = _claim_and_evidence(
                selected_case,
                field=field,
                override=selected_override,
            )
            if field == "source_type_match":
                memory_id = str(selected_override["memory_id"])
                allowed_evidence["registered_source"] = _memory_by_id(
                    base_case, memory_id
                ).source.value
            evaluation_mode = (
                "deterministic_code"
                if field in DETERMINISTIC_DIAGNOSTIC_FIELDS
                else "semantic_atomic_claim"
            )
            item: dict[str, Any] = {
                "diagnostic_id": f"v4_calibration__{field}__{polarity}",
                "source_case_id": case_id,
                "source_control_id": str(control["item_id"]),
                "source_control_case_text_sha256": sha256_text(rendered),
                "case_text_sha256": sha256_text(_render_case_text(selected_case)),
                "field": field,
                "polarity": polarity,
                "evaluation_mode": evaluation_mode,
                "candidate_claim": candidate_claim,
                "allowed_evidence": allowed_evidence,
                "prohibited_evidence": list(
                    FIELD_REVIEW_SPECIFICATIONS[field].prohibited_evidence
                ),
                "minimum_evidence_citations": _minimum_evidence_citations(field),
                "diagnostic_only": True,
            }
            if field in DETERMINISTIC_DIAGNOSTIC_FIELDS:
                item["expected_code_result"] = polarity == "positive"
                item["deterministic_assessment"] = (
                    evaluate_deterministic_diagnostic_item(item)
                )
            else:
                item["expected_verdict"] = expected_verdict_for_polarity(polarity)
                item["messages"] = single_field_diagnostic_messages(
                    field=field,
                    candidate_claim=candidate_claim,
                    allowed_evidence=allowed_evidence,
                )
            items.append(item)
    return items


def build_v4_root_cause_diagnostic_packet(
    cases: Sequence[V8ReviewCase],
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    items = build_v4_single_field_diagnostic_items(cases, controls)
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "status": DIAGNOSTIC_STATUS,
        "formal_gate": False,
        "api_calls_made": 0,
        "uses_existing_v4_calibration_data": True,
        "creates_new_heldout": False,
        "creates_paid_run_identity": False,
        "changes_v4_gate_result": False,
        "source_v4_result": "CONSUMED_FAILED_CLOSED",
        "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
        "claim_boundary": (
            "Synthetic-state proxy-utility measurement only; not real-user need "
            "recognition, welfare, clinical efficacy, or real-world behavior change."
        ),
        "diagnostic_polarities": list(DIAGNOSTIC_POLARITIES),
        "semantic_protocol": ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE,
        "semantic_fields": list(SEMANTIC_DIAGNOSTIC_FIELDS),
        "deterministic_fields": list(DETERMINISTIC_DIAGNOSTIC_FIELDS),
        "prepared_prompt_count": sum(
            item["evaluation_mode"] == "semantic_atomic_claim" for item in items
        ),
        "deterministic_check_count": sum(
            item["evaluation_mode"] == "deterministic_code" for item in items
        ),
        "field_review_specifications": {
            field: spec.model_dump(mode="json")
            for field, spec in FIELD_REVIEW_SPECIFICATIONS.items()
        },
        "items": items,
    }


def validate_v4_diagnostic_source_artifacts(
    *,
    packet: Mapping[str, Any],
    observed_controls: Sequence[Mapping[str, Any]],
    observed_control_judgments: Mapping[str, Any],
    observed_gate_report: Mapping[str, Any],
    endpoint_names: Sequence[str],
) -> dict[str, Any]:
    """Bind the diagnostic to the already-spent V4 calibration artifacts.

    The diagnostic is allowed to reuse observed material, but it must neither
    substitute a newly generated control nor reinterpret the failed V4 result.
    """

    if (
        packet.get("protocol") != DIAGNOSTIC_PROTOCOL
        or packet.get("status") != DIAGNOSTIC_STATUS
        or packet.get("formal_gate") is not False
        or packet.get("uses_existing_v4_calibration_data") is not True
        or packet.get("changes_v4_gate_result") is not False
    ):
        raise RuntimeError("diagnostic packet contract drift")
    if list(packet.get("diagnostic_fields") or []) != list(DIAGNOSTIC_FIELDS):
        raise RuntimeError("diagnostic field cohort drift")
    if (
        list(packet.get("diagnostic_polarities") or []) != list(DIAGNOSTIC_POLARITIES)
        or packet.get("semantic_protocol")
        != ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE
        or list(packet.get("semantic_fields") or [])
        != list(SEMANTIC_DIAGNOSTIC_FIELDS)
        or list(packet.get("deterministic_fields") or [])
        != list(DETERMINISTIC_DIAGNOSTIC_FIELDS)
    ):
        raise RuntimeError("diagnostic balanced measurement contract drift")
    items = packet.get("items")
    if not isinstance(items, list) or len(items) != (
        len(DIAGNOSTIC_FIELDS) * len(DIAGNOSTIC_POLARITIES)
    ):
        raise RuntimeError("diagnostic packet item count drift")
    if (
        observed_gate_report.get("status") != "FAIL"
        or int(observed_gate_report.get("n_controls") or 0) != 24
        or len(observed_gate_report.get("control_misses") or []) != 24
        or len(observed_gate_report.get("control_catches") or []) != 0
    ):
        raise RuntimeError("observed V4 source is not the failed 24-control run")
    if list(observed_gate_report.get("judge_families") or []) != list(endpoint_names):
        raise RuntimeError("observed V4 judge panel differs from diagnostic panel")

    controls_by_id = {
        str(row.get("item_id") or ""): row for row in observed_controls
    }
    if len(controls_by_id) != len(observed_controls):
        raise RuntimeError("observed V4 controls contain duplicate item IDs")
    baseline_rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_matrix: set[tuple[str, str]] = set()
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise RuntimeError("diagnostic item must be an object")
        item = dict(raw_item)
        diagnostic_id = str(item.get("diagnostic_id") or "")
        field = str(item.get("field") or "")
        polarity = str(item.get("polarity") or "")
        control_id = str(item.get("source_control_id") or "")
        if not diagnostic_id or diagnostic_id in seen_ids:
            raise RuntimeError("diagnostic item IDs must be non-empty and unique")
        seen_ids.add(diagnostic_id)
        if field not in DIAGNOSTIC_FIELDS:
            raise RuntimeError("diagnostic item has an unapproved field")
        matrix_key = (field, polarity)
        if polarity not in DIAGNOSTIC_POLARITIES or matrix_key in seen_matrix:
            raise RuntimeError("diagnostic polarity matrix is duplicated or invalid")
        seen_matrix.add(matrix_key)
        control = controls_by_id.get(control_id)
        if not isinstance(control, Mapping):
            raise RuntimeError(f"observed V4 controls lack {control_id}")
        if (
            control.get("corrupted_field") != field
            or control.get("case_text_sha256")
            != item.get("source_control_case_text_sha256")
        ):
            raise RuntimeError(f"observed V4 control binding drift for {control_id}")
        if field in SEMANTIC_DIAGNOSTIC_FIELDS:
            expected_messages = single_field_diagnostic_messages(
                field=field,
                candidate_claim=dict(item.get("candidate_claim") or {}),
                allowed_evidence=dict(item.get("allowed_evidence") or {}),
            )
            if item.get("messages") != expected_messages:
                raise RuntimeError("diagnostic prompt content drift")
            if item.get("expected_verdict") != expected_verdict_for_polarity(polarity):
                raise RuntimeError("diagnostic semantic truth drift")
        else:
            assessment = evaluate_deterministic_diagnostic_item(item)
            if item.get("deterministic_assessment") != assessment:
                raise RuntimeError("diagnostic deterministic truth drift")
            deterministic_rows.append(assessment)
        if field == "context_grounding_match" and any(
            "claim" in row
            for row in (item.get("allowed_evidence") or {}).get(
                "claim_provenance", []
            )
        ):
            raise RuntimeError("context-grounding evidence leaks provenance claims")

        if polarity == "negative":
            judgments = observed_control_judgments.get(control_id)
            if not isinstance(judgments, Mapping):
                raise RuntimeError(f"observed V4 judgments lack {control_id}")
            for endpoint_name in endpoint_names:
                judgment = judgments.get(endpoint_name)
                if not isinstance(judgment, Mapping):
                    raise RuntimeError("observed V4 diagnostic judgment is incomplete")
                ratings = judgment.get("ratings")
                if not isinstance(ratings, Mapping) or ratings.get(field) not in (0, 1):
                    raise RuntimeError("observed V4 target rating is invalid")
                baseline_rows.append(
                    {
                        "diagnostic_id": diagnostic_id,
                        "source_control_id": control_id,
                        "field": field,
                        "endpoint_name": str(endpoint_name),
                        "p0_target_rating": int(ratings[field]),
                        "p0_control_caught": int(ratings[field]) == 0,
                    }
                )
    if seen_matrix != {
        (field, polarity)
        for field in DIAGNOSTIC_FIELDS
        for polarity in DIAGNOSTIC_POLARITIES
    }:
        raise RuntimeError("diagnostic balanced field matrix is incomplete")
    return {
        "p0_negative_control_rows": baseline_rows,
        "deterministic_rows": deterministic_rows,
        "source_binding_verified": True,
    }


def aggregate_v4_single_field_diagnostic(
    *,
    deterministic_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    endpoint_names: Sequence[str],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Summarize balanced code/semantic calibration without creating a gate."""

    expected_keys = {
        (field, polarity, endpoint)
        for field in SEMANTIC_DIAGNOSTIC_FIELDS
        for polarity in DIAGNOSTIC_POLARITIES
        for endpoint in endpoint_names
    }
    observed_keys = [
        (
            str(row.get("field") or ""),
            str(row.get("polarity") or ""),
            str(row.get("endpoint_name") or ""),
        )
        for row in result_rows
    ]
    observed_key_set = set(observed_keys)
    if len(observed_keys) != len(observed_key_set) or not observed_key_set.issubset(
        expected_keys
    ):
        raise RuntimeError("diagnostic result matrix is duplicated or contains extras")
    matrix_complete = observed_key_set == expected_keys
    if require_complete and not matrix_complete:
        raise RuntimeError("diagnostic result matrix is incomplete")
    expected_deterministic = {
        (field, polarity)
        for field in DETERMINISTIC_DIAGNOSTIC_FIELDS
        for polarity in DIAGNOSTIC_POLARITIES
    }
    observed_deterministic = {
        (str(row.get("field") or ""), str(row.get("polarity") or ""))
        for row in deterministic_rows
    }
    if observed_deterministic != expected_deterministic:
        raise RuntimeError("diagnostic deterministic matrix is incomplete")

    field_strata: list[dict[str, Any]] = []
    endpoint_summaries: list[dict[str, Any]] = []
    for endpoint in endpoint_names:
        endpoint_rows = [
            row for row in result_rows if row.get("endpoint_name") == endpoint
        ]
        for field in SEMANTIC_DIAGNOSTIC_FIELDS:
            rows = [row for row in endpoint_rows if row.get("field") == field]
            positive = next(
                (row for row in rows if row.get("polarity") == "positive"), None
            )
            negative = next(
                (row for row in rows if row.get("polarity") == "negative"), None
            )
            sensitivity = (
                float(positive.get("verdict") == "supported")
                if positive is not None
                else None
            )
            specificity = (
                float(negative.get("verdict") == "not_supported")
                if negative is not None
                else None
            )
            positive_citation_valid = (
                bool(positive.get("citation_valid"))
                if positive is not None
                else None
            )
            negative_citation_valid = (
                bool(negative.get("citation_valid"))
                if negative is not None
                else None
            )
            complete_pair = positive is not None and negative is not None
            field_strata.append(
                {
                    "endpoint_name": str(endpoint),
                    "field": field,
                    "n_observed": len(rows),
                    "complete_pair": complete_pair,
                    "positive_correct": (
                        bool(positive.get("verdict_correct"))
                        if positive is not None
                        else None
                    ),
                    "negative_correct": (
                        bool(negative.get("verdict_correct"))
                        if negative is not None
                        else None
                    ),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "balanced_accuracy": (
                        (sensitivity + specificity) / 2.0
                        if complete_pair
                        else None
                    ),
                    "citation_integrity_rate": (
                        float(positive_citation_valid)
                        + float(negative_citation_valid)
                    )
                    / 2.0
                    if complete_pair
                    else None,
                    "joint_validated_accuracy": (
                        float(bool(positive.get("correct")))
                        + float(bool(negative.get("correct")))
                    )
                    / 2.0
                    if complete_pair
                    else None,
                }
            )
        positive_rows = [
            row for row in endpoint_rows if row.get("polarity") == "positive"
        ]
        negative_rows = [
            row for row in endpoint_rows if row.get("polarity") == "negative"
        ]
        sensitivity = (
            sum(row.get("verdict") == "supported" for row in positive_rows)
            / len(positive_rows)
            if positive_rows
            else None
        )
        specificity = (
            sum(row.get("verdict") == "not_supported" for row in negative_rows)
            / len(negative_rows)
            if negative_rows
            else None
        )
        endpoint_complete = len(endpoint_rows) == (
            len(SEMANTIC_DIAGNOSTIC_FIELDS) * len(DIAGNOSTIC_POLARITIES)
        )
        endpoint_summaries.append(
            {
                "endpoint_name": str(endpoint),
                "n": len(endpoint_rows),
                "n_expected": len(SEMANTIC_DIAGNOSTIC_FIELDS)
                * len(DIAGNOSTIC_POLARITIES),
                "complete": endpoint_complete,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": (
                    (sensitivity + specificity) / 2.0
                    if sensitivity is not None and specificity is not None
                    else None
                ),
                "verdict_accuracy": (
                    sum(bool(row.get("verdict_correct")) for row in endpoint_rows)
                    / len(endpoint_rows)
                    if endpoint_rows
                    else None
                ),
                "citation_integrity_rate": (
                    sum(bool(row.get("citation_valid")) for row in endpoint_rows)
                    / len(endpoint_rows)
                    if endpoint_rows
                    else None
                ),
                "joint_validated_accuracy": (
                    sum(bool(row.get("correct")) for row in endpoint_rows)
                    / len(endpoint_rows)
                    if endpoint_rows
                    else None
                ),
                "not_supported_count": sum(
                    row.get("verdict") == "not_supported" for row in endpoint_rows
                ),
            }
        )
    deterministic_correct = sum(bool(row.get("correct")) for row in deterministic_rows)
    semantic_joint_correct = sum(bool(row.get("correct")) for row in result_rows)
    semantic_verdict_correct = sum(
        bool(row.get("verdict_correct")) for row in result_rows
    )
    citation_valid = sum(bool(row.get("citation_valid")) for row in result_rows)
    semantic_support_classifier_ready = (
        matrix_complete
        and len(result_rows) > 0
        and semantic_verdict_correct == len(result_rows)
    )
    citation_audit_ready = (
        matrix_complete
        and len(result_rows) > 0
        and citation_valid == len(result_rows)
    )
    return {
        "protocol": DIAGNOSTIC_RUN_PROTOCOL,
        "status": (
            DIAGNOSTIC_RUN_STATUS
            if matrix_complete
            else "INCONCLUSIVE_PROVIDER_AVAILABILITY_NOT_A_FORMAL_GATE"
        ),
        "formal_gate": False,
        "authorizes_v5": False,
        "authorizes_training": False,
        "authorizes_formal_generation": False,
        "changes_v4_gate_result": False,
        "uses_calibration_data_only": True,
        "claim_boundary": (
            "Synthetic-state proxy-utility measurement only; no real-user need "
            "recognition or real-world outcome claim."
        ),
        "n_fields": len(DIAGNOSTIC_FIELDS),
        "n_semantic_fields": len(SEMANTIC_DIAGNOSTIC_FIELDS),
        "n_deterministic_fields": len(DETERMINISTIC_DIAGNOSTIC_FIELDS),
        "n_endpoints": len(endpoint_names),
        "n_result_rows": len(result_rows),
        "n_expected_result_rows": len(expected_keys),
        "n_missing_result_rows": len(expected_keys - observed_key_set),
        "result_matrix_complete": matrix_complete,
        "missing_result_keys": [
            {"field": field, "polarity": polarity, "endpoint_name": endpoint}
            for field, polarity, endpoint in sorted(expected_keys - observed_key_set)
        ],
        "n_deterministic_rows": len(deterministic_rows),
        "measurement_instrument_ready": (
            semantic_support_classifier_ready
            and deterministic_correct == len(deterministic_rows)
        ),
        "semantic_support_classifier_ready": semantic_support_classifier_ready,
        "citation_audit_ready": citation_audit_ready,
        "citation_integrity_is_outcome_gate": False,
        "deterministic_accuracy": deterministic_correct / len(deterministic_rows),
        "semantic_verdict_accuracy": (
            semantic_verdict_correct / len(result_rows) if result_rows else None
        ),
        "citation_integrity_rate": (
            citation_valid / len(result_rows) if result_rows else None
        ),
        "semantic_joint_validated_accuracy": (
            semantic_joint_correct / len(result_rows) if result_rows else None
        ),
        "interpretation": (
            "Code evaluates frozen source metadata, arithmetic, dialogue structure, "
            "and explicit generation-leakage markers. Two independent judge families "
            "evaluate only whether context grounding and advice readiness are supported "
            "by exhaustive visible evidence. Citation adherence is reported separately "
            "and does not redefine semantic correctness. This is calibration evidence "
            "for a synthetic-state proxy measurement instrument, not held-out accuracy "
            "or permission to claim real-user understanding or benefit."
        ),
        "field_strata": field_strata,
        "endpoint_summaries": endpoint_summaries,
    }
