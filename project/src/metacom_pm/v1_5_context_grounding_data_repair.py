"""Repair-content generation for the 25 DATA_DEFECT states found in
data/pm_v1_5_contracts/context_grounding_defect_classification_v1.jsonl.

Two isolated repair modes, matching the classification's repair_mode field:

- FIELD_ONLY_REPAIR (19 states): current_user_text and current_session_
  history are internally consistent. authorized_user_context is always
  regenerated; session_summary is ALSO regenerated for the 2 states in
  FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED (their summary independently
  asserts an ungrounded fact, e.g. state_1d327ed7...'s "nine days ago"),
  and is otherwise left byte-identical to the original -- it must not be
  silently rewritten by whatever the model happens to echo back, since an
  unnecessary summary change would (a) invalidate that row's advice_
  readiness_match carry-forward for no reason and (b) go through the
  visible-semantic-state embedding, which authorized_user_context never
  does (authorized_user_context is evaluator-only and is not a field of
  PMV2State at all -- confirmed against case_to_state/prepare_visible_
  semantic_state in pm_v2_data.py / pm_v1_5_semantic.py).

- VISIBLE_SURFACE_REPAIR (6 states): the visible dialogue itself names two
  different people/relations across turns (or an unresolved two-move
  narrative). Only the specific conflicting history turn(s) are
  regenerated -- current_user_text, regime, semantic_family, turn count,
  and role order are all frozen -- then authorized_user_context/
  session_summary are regenerated from the corrected dialogue.

VISIBLE_SURFACE_REPAIR_TURN_INDICES below is a hand-verified, disclosed
specification of exactly which turn indices are inconsistent with the
current_user_text anchor for each of the 6 states -- re-derived by reading
every turn's text directly (not just the turn the judges happened to
complain about), including pronoun cascades a shallower read would miss
(e.g. state_d0eee004b44a562c57a0e6c6 needs turns 0, 2, 3, 4, 5 -- not just
turn 0 -- because "her"/"her memory" recurs in four separate turns after
the initial grandmother/grandfather mismatch).

Whenever current_user_text, current_session_history, or current_session_
summary changes, PMV2State's text_embedding and provenance["semantic_
observation"] are stale (both are a pure function of exactly those three
fields -- see prepare_visible_semantic_state in pm_v1_5_semantic.py) and
must be recomputed via recompute_visible_semantic_fields(), never left as
whatever was frozen before the repair. This affects 8 of the 25 states: the
6 VISIBLE_SURFACE_REPAIR states plus the 2 FIELD_ONLY_REPAIR states in
FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED. The other 17 FIELD_ONLY_REPAIR
states change nothing that feeds the embedding, so their text_embedding/
provenance are correctly left untouched.

Deliberately NOT recomputed here: step0_observation (built by build_step0_
observation in pm_v1_5_step0.py), which additionally needs the case's
memory inventory from the matching pm_v2_bundles.jsonl row and the
strategy catalog -- a real, disclosed gap, not silently skipped. Recompute
it before ever treating a repaired state as fully consistent.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping, Sequence

from pydantic import Field

from .io import canonical_json, sha256_text
from .pm_v2_contracts import StrictModel
from .text import estimate_tokens
from .v1_5_context_grounding_repair import ContextGroundingClassificationRecord

# Of the 19 FIELD_ONLY_REPAIR states, only these 2 have a session_summary
# that independently asserts an ungrounded fact and must be regenerated;
# the other 17's summary is either already empty (N/A) or already
# accurate, and must be left byte-identical -- never silently overwritten
# by whatever the repair call happens to echo back.
FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED = frozenset(
    {
        "state_1d327ed7b58f3b13298b968c",  # summary independently asserts an ungrounded "nine days ago"
        "state_df5f52854a6eaae0e0a229bb",  # summary independently asserts an ungrounded "long-term partner"
    }
)

# Every state/evaluator_context field NOT in these sets must be
# byte-identical before and after repair -- checked explicitly, field by
# field, rather than trusting the repair call to only touch what it was
# asked to. context_payload_sha256 is a hash DERIVED from every other
# evaluator_context field (evaluator_context_payload_sha256 in
# pm_v2_data.py) -- it is expected to change when authorized_user_context
# changes, but is separately verified to be the correct recomputed hash,
# never just allowed to be anything. text_embedding/provenance are allowed
# to change ONLY when current_user_text/history/summary actually changed
# (checked independently below, not gated purely on repair_mode, since 2
# of the 19 FIELD_ONLY_REPAIR states also touch summary).
STATE_FIELDS_ALLOWED_TO_CHANGE = frozenset(
    {"current_session_summary", "text_embedding", "provenance"}
)
STATE_FIELDS_ALLOWED_TO_CHANGE_VISIBLE_SURFACE = frozenset(
    {
        "current_session_summary",
        "current_session_history",
        "text_embedding",
        "provenance",
    }
)
EVALUATOR_CONTEXT_FIELDS_ALLOWED_TO_CHANGE = frozenset(
    {"authorized_user_context", "context_payload_sha256"}
)

REPAIR_GENERATION_PROTOCOL = "pm-v1.5-context-grounding-data-repair-v1"

MAX_CONTEXT_FIELD_CHARS = 250

# Hand-verified per-state specification: which current_session_history
# turn indices (0-based) are inconsistent with the current_user_text
# anchor and must be regenerated. Every other turn, and current_user_text
# itself, is frozen. Re-derive by reading every turn if this ever needs to
# change -- a shallow read (fixing only the turn a judge complained about)
# will under-count pronoun cascades, as it did on the first pass here.
VISIBLE_SURFACE_REPAIR_TURN_INDICES: dict[str, tuple[int, ...]] = {
    "state_44550214bf7c9fa22284a731": (0, 1),  # manager -> coworker (turn 1 replies "your manager")
    "state_481fdb962d3774c3e0d720bb": (0,),  # grandfather -> grandmother
    "state_7d1e36f110bdacbd2a0545f2": (0,),  # brother -> friend
    "state_9e0c7dd3b2e6081857a7d19f": (0, 3),  # grandfather -> grandmother; "him" -> "her"
    "state_d0eee004b44a562c57a0e6c6": (0, 2, 3, 4, 5),  # grandmother -> grandfather; "her"(x4) -> "his"
    "state_ba1e526156d1cf3810fa8e98": (0,),  # remove "since I moved here" -- conflicts with current turn's prospective move
}


class FieldOnlyRepairOutput(StrictModel):
    authorized_user_context: str = Field(min_length=1, max_length=MAX_CONTEXT_FIELD_CHARS)
    session_summary: str = Field(min_length=0, max_length=MAX_CONTEXT_FIELD_CHARS)


# Real corpus max observed history-turn length is 180 chars (checked
# directly against every turn in data/pm_v1_5_formal_v8_18_duplicate_
# repair_candidate); 400 gives real margin without being unbounded --
# an unbounded per-item string here would repeat the exact
# output_token_limit crash class this project already root-caused once
# for SingleFieldDiagnosticOutput.
MAX_REPAIRED_TURN_CONTENT_CHARS = 400


class VisibleSurfaceRepairOutput(StrictModel):
    repaired_turn_contents: list[
        Annotated[str, Field(max_length=MAX_REPAIRED_TURN_CONTENT_CHARS)]
    ] = Field(min_length=1, max_length=8)
    authorized_user_context: str = Field(min_length=1, max_length=MAX_CONTEXT_FIELD_CHARS)
    session_summary: str = Field(min_length=0, max_length=MAX_CONTEXT_FIELD_CHARS)


def maximum_legal_field_only_repair_output_tokens() -> int:
    """Computed (not eyeballed) worst-case token estimate for a maximally-
    sized, schema-legal FieldOnlyRepairOutput."""

    worst_case = FieldOnlyRepairOutput(
        authorized_user_context="x" * MAX_CONTEXT_FIELD_CHARS,
        session_summary="x" * MAX_CONTEXT_FIELD_CHARS,
    )
    return estimate_tokens(canonical_json(worst_case.model_dump(mode="json")))


def maximum_legal_visible_surface_repair_output_tokens() -> int:
    """Computed (not eyeballed) worst-case token estimate for a maximally-
    sized, schema-legal VisibleSurfaceRepairOutput (all 8 turns at the cap)."""

    worst_case = VisibleSurfaceRepairOutput(
        repaired_turn_contents=["x" * MAX_REPAIRED_TURN_CONTENT_CHARS] * 8,
        authorized_user_context="x" * MAX_CONTEXT_FIELD_CHARS,
        session_summary="x" * MAX_CONTEXT_FIELD_CHARS,
    )
    return estimate_tokens(canonical_json(worst_case.model_dump(mode="json")))


_COMMON_REPAIR_SYSTEM = (
    "You repair one field of a frozen synthetic support-conversation record. "
    "The current_user_text and every history turn not explicitly listed as "
    "repairable are FROZEN and must never be referenced as needing a change "
    "-- you are only asked to produce replacement text for the specific "
    "field(s) named in this task. "
    "authorized_user_context and session_summary may summarize ONLY facts "
    "that are explicitly stated or unambiguously implied by the visible "
    "dialogue given to you -- never invent an identity attribute (gender, "
    "marital status, age), a relationship duration, or a third party's "
    "belief that is not actually present in the dialogue. "
    "Follow the task-specific session_summary instruction exactly: some "
    "records require a non-empty grounded replacement because summary "
    "presence is a frozen corpus-design property, while other records require "
    "an exact echo that the caller will ignore. Never invent content merely "
    "to make a summary non-empty."
)


def build_field_only_repair_messages(
    *,
    current_user_text: str,
    history: Sequence[Mapping[str, Any]],
    frozen_session_summary: str,
    repair_summary: bool,
    defect_note: str,
) -> list[dict[str, str]]:
    if repair_summary:
        summary_instruction = (
            "Produce authorized_user_context and a NON-EMPTY session_summary, each "
            "grounded only in frozen_current_user_text and frozen_history "
            "above. The original record's summary-presence is frozen, so an "
            "empty session_summary is invalid; a concise grounded summary may "
            "overlap authorized_user_context. Do not reproduce the defect "
            "described in defect_note."
        )
    else:
        summary_instruction = (
            "Produce ONLY authorized_user_context, grounded only in "
            "frozen_current_user_text and frozen_history above. "
            "frozen_session_summary given below is already correct and is "
            "not part of this repair -- the response schema still requires "
            "a session_summary value, so return frozen_session_summary "
            "back unchanged (it will not be used regardless -- the caller "
            "always keeps the original byte-for-byte for this state)."
        )
    payload = {
        "task": "field_only_repair",
        "defect_note": defect_note,
        "frozen_current_user_text": current_user_text,
        "frozen_history": list(history),
        "frozen_session_summary": frozen_session_summary,
        "instruction": summary_instruction,
    }
    return [
        {"role": "system", "content": _COMMON_REPAIR_SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ]


def build_visible_surface_repair_messages(
    *,
    current_user_text: str,
    history: Sequence[Mapping[str, Any]],
    turn_indices_to_repair: Sequence[int],
    defect_note: str,
) -> list[dict[str, str]]:
    indexed_history = [
        {"index": index, "role": turn["role"], "content": turn["content"]}
        for index, turn in enumerate(history)
    ]
    payload = {
        "task": "visible_surface_repair",
        "defect_note": defect_note,
        "frozen_current_user_text": current_user_text,
        "indexed_history": indexed_history,
        "turn_indices_to_repair": list(turn_indices_to_repair),
        "instruction": (
            "Every history turn NOT listed in turn_indices_to_repair is "
            "frozen and must stay conceptually identical (do not restate "
            "it). Produce replacement content ONLY for the turns listed in "
            "turn_indices_to_repair, in that exact order, each rewritten to "
            "be consistent with frozen_current_user_text's topic/referent "
            "and with every frozen turn around it -- preserve each turn's "
            "role and natural conversational flow. Then produce "
            "authorized_user_context and session_summary grounded in the "
            "corrected full conversation (frozen_current_user_text plus "
            "every history turn, repaired or not). Return a concise NON-EMPTY "
            "session_summary; for records whose original summary is structurally "
            "absent the caller deterministically discards it."
        ),
    }
    return [
        {"role": "system", "content": _COMMON_REPAIR_SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ]


def repair_call_plan_row(
    record: ContextGroundingClassificationRecord,
) -> dict[str, Any]:
    """Build one (unpriced) call-plan row for a single DATA_DEFECT record."""

    if record.repair_mode == "FIELD_ONLY_REPAIR":
        repair_summary = record.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
        messages = build_field_only_repair_messages(
            current_user_text=record.current_user_text,
            history=record.history,
            frozen_session_summary=record.session_summary,
            repair_summary=repair_summary,
            defect_note=record.rationale,
        )
        response_schema_name = FieldOnlyRepairOutput.__name__
    elif record.repair_mode == "VISIBLE_SURFACE_REPAIR":
        turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[record.state_id]
        messages = build_visible_surface_repair_messages(
            current_user_text=record.current_user_text,
            history=record.history,
            turn_indices_to_repair=turn_indices,
            defect_note=record.rationale,
        )
        response_schema_name = VisibleSurfaceRepairOutput.__name__
    else:
        raise RuntimeError(
            f"unsupported repair_mode for {record.state_id}: {record.repair_mode}"
        )
    payload_text = canonical_json(messages)
    return {
        "state_id": record.state_id,
        "repair_mode": record.repair_mode,
        "defect_type": record.defect_type,
        "response_schema": response_schema_name,
        "messages": messages,
        "raw_estimated_input_tokens": estimate_tokens(payload_text),
        "repair_summary": (
            record.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
            if record.repair_mode == "FIELD_ONLY_REPAIR"
            else True  # VISIBLE_SURFACE_REPAIR always re-derives summary from the corrected dialogue
        ),
    }


def _evaluator_context_payload_sha256(evaluator_context: Mapping[str, Any]) -> str:
    from .pm_v2_data import EVALUATOR_CONTEXT_FIELDS

    payload = {
        key: evaluator_context[key]
        for key in sorted(EVALUATOR_CONTEXT_FIELDS - {"context_payload_sha256"})
    }
    return sha256_text(canonical_json(payload))


def apply_field_only_repair(
    *,
    state: Mapping[str, Any],
    evaluator_context: Mapping[str, Any],
    repair: FieldOnlyRepairOutput,
    repair_summary: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """repair_summary=False (17 of 19 states) deliberately IGNORES
    repair.session_summary and keeps the original current_session_summary
    byte-identical -- never trusts the model to echo the frozen value back
    verbatim. Only the 2 states in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
    pass repair_summary=True.
    """

    repaired_state = dict(state)
    if repair_summary:
        repaired_state["current_session_summary"] = repair.session_summary
    repaired_context = dict(evaluator_context)
    repaired_context["authorized_user_context"] = repair.authorized_user_context
    repaired_context["context_payload_sha256"] = _evaluator_context_payload_sha256(
        repaired_context
    )
    return repaired_state, repaired_context


def apply_visible_surface_repair(
    *,
    state: Mapping[str, Any],
    evaluator_context: Mapping[str, Any],
    turn_indices: Sequence[int],
    repair: VisibleSurfaceRepairOutput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Whether session_summary is present or absent for a given (user,
    case_field) cell is a frozen, counterbalanced design property enforced
    by write_development_dataset's summarize_observable_state_support
    (real check, confirmed by running the real recompilation pipeline
    against a placeholder-repaired copy of the actual v8_18 bundles: it
    hard-fails with "summary support drifted" the moment a repair turns an
    originally-empty summary non-empty). 3 of the 6 VISIBLE_SURFACE_REPAIR
    states have an originally EMPTY summary (state_44550214..., state_
    7d1e36f1..., state_ba1e5261...) -- for those, repair.session_summary is
    deliberately ignored and the summary is forced back to empty,
    regardless of what the model returned, mirroring the same
    never-trust-the-model-to-echo-correctly philosophy already applied to
    apply_field_only_repair's 17 non-summary states.
    """

    if len(repair.repaired_turn_contents) != len(turn_indices):
        raise RuntimeError(
            "repaired_turn_contents length does not match turn_indices_to_repair"
        )
    new_history = [dict(turn) for turn in state["current_session_history"]]
    for index, content in zip(turn_indices, repair.repaired_turn_contents):
        new_history[index] = {**new_history[index], "content": content}
    repaired_state = dict(state)
    repaired_state["current_session_history"] = new_history
    original_summary_present = bool(str(state.get("current_session_summary") or "").strip())
    repaired_state["current_session_summary"] = (
        repair.session_summary if original_summary_present else ""
    )
    repaired_context = dict(evaluator_context)
    repaired_context["authorized_user_context"] = repair.authorized_user_context
    repaired_context["context_payload_sha256"] = _evaluator_context_payload_sha256(
        repaired_context
    )
    return repaired_state, repaired_context


def validate_repair_allowlist_diff(
    *,
    original_state: Mapping[str, Any],
    original_evaluator_context: Mapping[str, Any],
    repaired_state: Mapping[str, Any],
    repaired_evaluator_context: Mapping[str, Any],
    repair_mode: str,
    turn_indices: Sequence[int] | None = None,
) -> None:
    """Fail closed (raise) if any field outside the allowed set changed.

    This is the zero-API post-hoc check: it never trusts that a repair call
    "only touched what it was asked to" -- every state/evaluator_context
    field is compared explicitly, field by field, against its pre-repair
    value. turn_indices is REQUIRED for VISIBLE_SURFACE_REPAIR: every index
    in it must have genuinely changed (a "repair" that left the flagged
    turn untouched is not a repair), and every index not in it must be
    byte-identical to the original.
    """

    if repair_mode not in ("FIELD_ONLY_REPAIR", "VISIBLE_SURFACE_REPAIR"):
        raise RuntimeError(f"unsupported repair_mode: {repair_mode}")
    if repair_mode == "VISIBLE_SURFACE_REPAIR" and turn_indices is None:
        raise RuntimeError(
            "VISIBLE_SURFACE_REPAIR validation requires turn_indices"
        )

    if str(original_state["current_user_text"]) != str(
        repaired_state["current_user_text"]
    ):
        raise RuntimeError(
            "repair changed current_user_text, which must always be frozen"
        )

    # Whether session_summary is present or absent for this (user,
    # case_field) cell is a frozen, counterbalanced corpus-design property
    # (observable_state_design / summarize_observable_state_support in
    # pm_v2_data.py) -- confirmed as a REAL hard-fail check by running the
    # actual recompilation pipeline against the real bundles. A repair may
    # change summary CONTENT but must never flip empty<->non-empty.
    original_summary_present = bool(
        str(original_state.get("current_session_summary") or "").strip()
    )
    repaired_summary_present = bool(
        str(repaired_state.get("current_session_summary") or "").strip()
    )
    if original_summary_present != repaired_summary_present:
        raise RuntimeError(
            "repair changed whether session_summary is present/absent -- "
            "this is a frozen, counterbalanced corpus-design property and "
            "must never drift, only the summary's content may change"
        )

    allowed_state_fields = (
        STATE_FIELDS_ALLOWED_TO_CHANGE_VISIBLE_SURFACE
        if repair_mode == "VISIBLE_SURFACE_REPAIR"
        else STATE_FIELDS_ALLOWED_TO_CHANGE
    )
    for key in set(original_state) | set(repaired_state):
        if key in allowed_state_fields:
            continue
        if original_state.get(key) != repaired_state.get(key):
            raise RuntimeError(f"repair changed a frozen state field: {key!r}")

    for key in set(original_evaluator_context) | set(repaired_evaluator_context):
        if key in EVALUATOR_CONTEXT_FIELDS_ALLOWED_TO_CHANGE:
            continue
        if original_evaluator_context.get(key) != repaired_evaluator_context.get(key):
            raise RuntimeError(
                f"repair changed a frozen evaluator_context field: {key!r}"
            )

    expected_hash = _evaluator_context_payload_sha256(repaired_evaluator_context)
    if repaired_evaluator_context.get("context_payload_sha256") != expected_hash:
        raise RuntimeError(
            "repaired evaluator_context's context_payload_sha256 does not "
            "match its own recomputed hash"
        )

    # FIELD_ONLY_REPAIR already fails closed on any current_session_history
    # change via the generic per-field loop above (history is not in
    # STATE_FIELDS_ALLOWED_TO_CHANGE for that mode). Only VISIBLE_SURFACE_
    # REPAIR needs the additional structural/content checks below, since
    # history IS allowed to change there -- but only at the specified turn
    # indices, and only the content, never role or ordering.
    if repair_mode == "VISIBLE_SURFACE_REPAIR":
        original_history = list(original_state["current_session_history"])
        repaired_history = list(repaired_state["current_session_history"])
        if len(original_history) != len(repaired_history):
            raise RuntimeError("repair changed the number of history turns")
        turn_indices_set = set(turn_indices or ())
        for index, (old_turn, new_turn) in enumerate(
            zip(original_history, repaired_history)
        ):
            if str(old_turn["role"]) != str(new_turn["role"]):
                raise RuntimeError(
                    f"repair changed the role of history turn {index}"
                )
            content_changed = str(old_turn["content"]) != str(new_turn["content"])
            if index in turn_indices_set and not content_changed:
                raise RuntimeError(
                    f"repair left turn {index} unchanged, but it was flagged "
                    "for repair -- the defect was not actually fixed"
                )
            if index not in turn_indices_set and content_changed:
                raise RuntimeError(
                    f"repair changed turn {index}, which was not flagged for "
                    "repair and must stay byte-identical"
                )

    # Embedding-freshness invariant: text_embedding and provenance are a
    # pure function of exactly current_user_text + current_session_history
    # + current_session_summary (prepare_visible_semantic_state in
    # pm_v1_5_semantic.py). If any of those three changed, the derived
    # fields MUST also have changed (recompute_visible_semantic_fields must
    # have been applied -- a repair that changes visible text but forgets
    # to refresh the embedding is exactly the staleness bug this check
    # exists to catch); if none changed, the derived fields must be
    # byte-identical (nothing legitimately explains a change).
    visible_text_changed = (
        str(original_state["current_user_text"])
        != str(repaired_state["current_user_text"])
        or original_state["current_session_history"]
        != repaired_state["current_session_history"]
        or str(original_state.get("current_session_summary") or "")
        != str(repaired_state.get("current_session_summary") or "")
    )
    embedding_changed = original_state.get("text_embedding") != repaired_state.get(
        "text_embedding"
    )
    provenance_changed = original_state.get("provenance") != repaired_state.get(
        "provenance"
    )
    if visible_text_changed and not (embedding_changed and provenance_changed):
        raise RuntimeError(
            "current_user_text/history/summary changed but text_embedding/"
            "provenance were not refreshed -- call "
            "recompute_visible_semantic_fields before treating this repair "
            "as complete (a stale embedding is not caught by PMV2State's "
            "own schema validation)"
        )
    if not visible_text_changed and (embedding_changed or provenance_changed):
        raise RuntimeError(
            "text_embedding/provenance changed but nothing that feeds them "
            "(current_user_text/history/summary) actually changed"
        )


def visible_text_changed(state_a: Mapping[str, Any], state_b: Mapping[str, Any]) -> bool:
    return (
        str(state_a["current_user_text"]) != str(state_b["current_user_text"])
        or state_a["current_session_history"] != state_b["current_session_history"]
        or str(state_a.get("current_session_summary") or "")
        != str(state_b.get("current_session_summary") or "")
    )


def recompute_visible_semantic_fields(
    *, state: Mapping[str, Any], encoder: Any
) -> dict[str, Any]:
    """Refresh text_embedding and provenance["semantic_observation"] from the
    state's OWN current current_user_text/current_session_history/
    current_session_summary.

    Uses encode_visible_state (pm_v1_5_semantic.py) -- the same narrow,
    single-state function the project's own preflight smoke test
    (scripts/v1_5/19_preflight_semantic_runtime_v1_5.py) calls, NOT the
    whole-corpus write_development_dataset pipeline. That pipeline
    truncates and rewrites every output file from scratch for all 52
    users/468 states at once and enforces cross-user/cross-split
    invariants that require the whole corpus -- there is no way to run it
    for "just 25 states" without regenerating and revalidating everything.
    encode_visible_state has no such requirement: it only needs the frozen
    encoder and this one state's three visible-text fields.

    Deliberately does NOT recompute step0_observation (needs the case's
    memory inventory from the matching pm_v2_bundles.jsonl row and the
    strategy catalog, via case_to_state/build_step0_observation) -- a real,
    disclosed gap, not silently glossed over.
    """

    from .pm_v1_5_semantic import encode_visible_state

    embedding, semantic_observation = encode_visible_state(
        encoder,
        current_user_text=str(state["current_user_text"]),
        current_session_history=state["current_session_history"],
        current_session_summary=str(state.get("current_session_summary") or ""),
    )
    repaired_state = dict(state)
    repaired_state["text_embedding"] = embedding
    repaired_state["provenance"] = {
        **dict(state["provenance"]),
        "semantic_observation": semantic_observation,
    }
    return repaired_state
