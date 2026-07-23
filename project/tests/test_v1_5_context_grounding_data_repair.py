from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pydantic import ValidationError

from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    SemanticEncoderBinding,
)
from metacom_pm.v1_5_context_grounding_data_repair import (
    FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED,
    MAX_REPAIRED_TURN_CONTENT_CHARS,
    FieldOnlyRepairOutput,
    VISIBLE_SURFACE_REPAIR_TURN_INDICES,
    VisibleSurfaceRepairOutput,
    apply_field_only_repair,
    apply_visible_surface_repair,
    build_field_only_repair_messages,
    build_visible_surface_repair_messages,
    maximum_legal_field_only_repair_output_tokens,
    maximum_legal_visible_surface_repair_output_tokens,
    recompute_visible_semantic_fields,
    repair_call_plan_row,
    validate_repair_allowlist_diff,
)
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    data_defect_state_ids,
    field_only_repair_state_ids,
    load_context_grounding_defect_classification,
    visible_surface_repair_state_ids,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"


class _FakeEncoder:
    """Lightweight stand-in matching SemanticTextEncoder's duck-typed
    interface (spec/binding/encode) -- same pattern as _CanaryEncoder in
    test_pm_v1_5_semantic_runtime.py. No real transformer model is loaded;
    tests only need a deterministic, distinct vector per distinct text."""

    spec = FrozenSemanticEncoderSpec(
        model_id="fixture/canary",
        revision="1" * 40,
        snapshot_tree_sha256="2" * 64,
        max_length=64,
        output_dimension=16,
    )
    binding = SemanticEncoderBinding(
        spec_sha256=spec.digest(),
        snapshot_tree_sha256="2" * 64,
        snapshot_file_count=1,
        implementation="transformers-auto-model-cls-float32",
    )

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            row = rng.random(16) + 0.1
            rows.append(row / np.linalg.norm(row))
        return np.vstack(rows)


def _load_real_state_and_context(state_id: str) -> tuple[dict, dict]:
    states_path = DATA_DIR / "pm_v2_states.jsonl"
    contexts_path = DATA_DIR / "evaluator_contexts.jsonl"
    if not states_path.is_file() or not contexts_path.is_file():
        pytest.skip(
            "requires the private local V8.18 development-data artifact; "
            "the artifact is intentionally excluded from Git"
        )
    state = None
    with open(states_path) as f:
        for line in f:
            row = json.loads(line)
            if row["state_id"] == state_id:
                state = row
                break
    context = None
    with open(contexts_path) as f:
        for line in f:
            row = json.loads(line)
            if row["state_id"] == state_id:
                context = row
                break
    assert state is not None and context is not None
    return state, context


@pytest.fixture(scope="module")
def classification_records():
    return load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)


def test_visible_surface_turn_indices_exactly_match_the_classification(classification_records) -> None:
    expected = visible_surface_repair_state_ids(classification_records)
    assert set(VISIBLE_SURFACE_REPAIR_TURN_INDICES) == expected
    assert len(expected) == 6


def test_field_only_and_visible_surface_partition_the_25_defects(classification_records) -> None:
    field_only = field_only_repair_state_ids(classification_records)
    surface = visible_surface_repair_state_ids(classification_records)
    assert len(field_only) == 19
    assert len(surface) == 6
    assert field_only | surface == data_defect_state_ids(classification_records)


def test_repair_call_plan_row_builds_for_every_data_defect_record(classification_records) -> None:
    defects = [r for r in classification_records if r.classification == "DATA_DEFECT"]
    assert len(defects) == 25
    for record in defects:
        row = repair_call_plan_row(record)
        assert row["state_id"] == record.state_id
        assert row["raw_estimated_input_tokens"] > 0
        if record.repair_mode == "FIELD_ONLY_REPAIR":
            assert row["response_schema"] == "FieldOnlyRepairOutput"
        else:
            assert row["response_schema"] == "VisibleSurfaceRepairOutput"


def test_field_only_repair_summary_mask_matches_classification_exactly(classification_records) -> None:
    field_only = [
        r for r in classification_records if r.repair_mode == "FIELD_ONLY_REPAIR"
    ]
    assert len(field_only) == 19
    assert FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED == {
        "state_1d327ed7b58f3b13298b968c",
        "state_df5f52854a6eaae0e0a229bb",
    }
    for record in field_only:
        row = repair_call_plan_row(record)
        expected = record.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
        assert row["repair_summary"] == expected


def test_field_only_repair_messages_never_mention_turn_indices() -> None:
    messages = build_field_only_repair_messages(
        current_user_text="text",
        history=[{"role": "user", "content": "hi"}],
        frozen_session_summary="",
        repair_summary=False,
        defect_note="note",
    )
    serialized = json.dumps(messages)
    assert "turn_indices_to_repair" not in serialized


def test_apply_field_only_repair_with_repair_summary_false_never_touches_summary() -> None:
    # state_032737f9... is one of the 17 states that only needs
    # authorized_user_context fixed -- its (empty) summary must stay
    # byte-identical even if the model's output disagrees.
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    repair = FieldOnlyRepairOutput(
        authorized_user_context="The user is navigating adult friendships.",
        session_summary="a summary the model produced but that must be IGNORED",
    )
    repaired_state, repaired_context = apply_field_only_repair(
        state=state, evaluator_context=context, repair=repair, repair_summary=False
    )
    assert repaired_state["current_session_summary"] == state["current_session_summary"]
    validate_repair_allowlist_diff(
        original_state=state,
        original_evaluator_context=context,
        repaired_state=repaired_state,
        repaired_evaluator_context=repaired_context,
        repair_mode="FIELD_ONLY_REPAIR",
    )
    assert repaired_context["authorized_user_context"] == repair.authorized_user_context
    assert repaired_state["current_session_history"] == state["current_session_history"]
    # Nothing that feeds the embedding changed -- text_embedding/provenance
    # must stay byte-identical too (not just "allowed to change").
    assert repaired_state["text_embedding"] == state["text_embedding"]
    assert repaired_state["provenance"] == state["provenance"]


def test_apply_field_only_repair_with_repair_summary_true_requires_fresh_embedding() -> None:
    # state_1d327ed7... is one of the 2 states where summary is also
    # genuinely repaired -- this changes what feeds the embedding, so a
    # repair that doesn't refresh text_embedding/provenance must fail.
    state, context = _load_real_state_and_context("state_1d327ed7b58f3b13298b968c")
    repair = FieldOnlyRepairOutput(
        authorized_user_context="The user is struggling after a breakup.",
        session_summary="User is struggling with unmotivation after a breakup.",
    )
    repaired_state, repaired_context = apply_field_only_repair(
        state=state, evaluator_context=context, repair=repair, repair_summary=True
    )
    assert repaired_state["current_session_summary"] == repair.session_summary
    with pytest.raises(RuntimeError, match="were not refreshed"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=repaired_context,
            repair_mode="FIELD_ONLY_REPAIR",
        )
    fresh_state = recompute_visible_semantic_fields(
        state=repaired_state, encoder=_FakeEncoder()
    )
    assert fresh_state["text_embedding"] != state["text_embedding"]
    validate_repair_allowlist_diff(
        original_state=state,
        original_evaluator_context=context,
        repaired_state=fresh_state,
        repaired_evaluator_context=repaired_context,
        repair_mode="FIELD_ONLY_REPAIR",
    )


def test_apply_field_only_repair_rejects_if_history_also_changed() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    repair = FieldOnlyRepairOutput(authorized_user_context="fixed.", session_summary="")
    repaired_state, repaired_context = apply_field_only_repair(
        state=state, evaluator_context=context, repair=repair, repair_summary=False
    )
    tampered_history = list(repaired_state["current_session_history"])
    tampered_history[0] = {**tampered_history[0], "content": "a sneaky rewrite"}
    repaired_state = {**repaired_state, "current_session_history": tampered_history}
    with pytest.raises(RuntimeError, match="frozen state field: 'current_session_history'"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=repaired_context,
            repair_mode="FIELD_ONLY_REPAIR",
        )


def test_apply_visible_surface_repair_replaces_only_specified_turns() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    assert turn_indices == (0,)
    repair = VisibleSurfaceRepairOutput(
        repaired_turn_contents=["I've been struggling with the loss of my grandmother. It's been really tough on my family and me."],
        authorized_user_context="The user has recently lost their grandmother.",
        session_summary="User expressed feelings of grief due to the loss of their grandmother.",
    )
    repaired_state, repaired_context = apply_visible_surface_repair(
        state=state, evaluator_context=context, turn_indices=turn_indices, repair=repair
    )
    fresh_state = recompute_visible_semantic_fields(state=repaired_state, encoder=_FakeEncoder())
    validate_repair_allowlist_diff(
        original_state=state,
        original_evaluator_context=context,
        repaired_state=fresh_state,
        repaired_evaluator_context=repaired_context,
        repair_mode="VISIBLE_SURFACE_REPAIR",
        turn_indices=turn_indices,
    )
    assert fresh_state["current_session_history"][0]["content"] == repair.repaired_turn_contents[0]
    # The untouched turn must be byte-identical.
    assert (
        fresh_state["current_session_history"][1]
        == state["current_session_history"][1]
    )
    assert fresh_state["text_embedding"] != state["text_embedding"]


def test_validate_rejects_visible_surface_repair_without_a_refreshed_embedding() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    repair = VisibleSurfaceRepairOutput(
        repaired_turn_contents=["I've been struggling with the loss of my grandmother."],
        authorized_user_context="The user has recently lost their grandmother.",
        session_summary="User expressed feelings of grief due to the loss of their grandmother.",
    )
    repaired_state, repaired_context = apply_visible_surface_repair(
        state=state, evaluator_context=context, turn_indices=turn_indices, repair=repair
    )
    # Deliberately skip recompute_visible_semantic_fields -- the embedding
    # is still describing the OLD (grandfather) text.
    with pytest.raises(RuntimeError, match="were not refreshed"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=repaired_context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
            turn_indices=turn_indices,
        )


def test_validate_rejects_a_flagged_turn_left_unchanged() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    # "Repair" that echoes the original (unfixed) turn content back.
    original_turn_0_content = state["current_session_history"][0]["content"]
    repair = VisibleSurfaceRepairOutput(
        repaired_turn_contents=[original_turn_0_content],
        authorized_user_context="The user has recently lost their grandmother.",
        session_summary="User expressed feelings of grief due to the loss of their grandmother.",
    )
    repaired_state, repaired_context = apply_visible_surface_repair(
        state=state, evaluator_context=context, turn_indices=turn_indices, repair=repair
    )
    with pytest.raises(RuntimeError, match="left turn 0 unchanged"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=repaired_context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
            turn_indices=turn_indices,
        )


def test_validate_rejects_an_unflagged_turn_that_changed() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    assert turn_indices == (0,)
    repair = VisibleSurfaceRepairOutput(
        repaired_turn_contents=["I've been struggling with the loss of my grandmother."],
        authorized_user_context="The user has recently lost their grandmother.",
        session_summary="User expressed feelings of grief due to the loss of their grandmother.",
    )
    repaired_state, repaired_context = apply_visible_surface_repair(
        state=state, evaluator_context=context, turn_indices=turn_indices, repair=repair
    )
    fresh_state = recompute_visible_semantic_fields(state=repaired_state, encoder=_FakeEncoder())
    # Sneak an extra change into a turn that was never flagged.
    tampered_history = list(fresh_state["current_session_history"])
    tampered_history[1] = {**tampered_history[1], "content": "an unrelated sneaky edit"}
    tampered_state = {**fresh_state, "current_session_history": tampered_history}
    with pytest.raises(RuntimeError, match="turn 1, which was not flagged"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=tampered_state,
            repaired_evaluator_context=repaired_context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
            turn_indices=turn_indices,
        )


def test_validate_visible_surface_repair_requires_turn_indices() -> None:
    state, context = _load_real_state_and_context("state_481fdb962d3774c3e0d720bb")
    with pytest.raises(RuntimeError, match="requires turn_indices"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=state,
            repaired_evaluator_context=context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
        )


def test_validate_rejects_a_changed_current_user_text() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    tampered_state = {**state, "current_user_text": "a completely different turn"}
    with pytest.raises(RuntimeError, match="current_user_text"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=tampered_state,
            repaired_evaluator_context=context,
            repair_mode="FIELD_ONLY_REPAIR",
        )


def test_validate_rejects_a_changed_semantic_family() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    tampered_state = {**state, "semantic_family": "some_other_family"}
    with pytest.raises(RuntimeError, match="frozen state field"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=tampered_state,
            repaired_evaluator_context=context,
            repair_mode="FIELD_ONLY_REPAIR",
        )


def test_validate_rejects_a_changed_regime() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    tampered_context = {**context, "regime": "some_other_regime"}
    with pytest.raises(RuntimeError, match="frozen evaluator_context field"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=state,
            repaired_evaluator_context=tampered_context,
            repair_mode="FIELD_ONLY_REPAIR",
        )


def test_validate_rejects_a_stale_context_payload_sha256() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    repair = FieldOnlyRepairOutput(authorized_user_context="a new real context.", session_summary="")
    repaired_state, repaired_context = apply_field_only_repair(
        state=state, evaluator_context=context, repair=repair, repair_summary=False
    )
    stale_context = {**repaired_context, "context_payload_sha256": context["context_payload_sha256"]}
    with pytest.raises(RuntimeError, match="does not match its own recomputed hash"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=stale_context,
            repair_mode="FIELD_ONLY_REPAIR",
        )


def test_validate_rejects_a_turn_count_change_for_visible_surface_repair() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    repaired_state = {
        **state,
        "current_session_history": list(state["current_session_history"]) + [
            {"role": "user", "content": "an extra turn"}
        ],
    }
    with pytest.raises(RuntimeError, match="number of history turns"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
            turn_indices=turn_indices,
        )


def test_validate_rejects_a_role_change_for_visible_surface_repair() -> None:
    state_id = "state_481fdb962d3774c3e0d720bb"
    state, context = _load_real_state_and_context(state_id)
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    new_history = [dict(t) for t in state["current_session_history"]]
    new_history[0]["role"] = "assistant"
    repaired_state = {**state, "current_session_history": new_history}
    with pytest.raises(RuntimeError, match="changed the role"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=repaired_state,
            repaired_evaluator_context=context,
            repair_mode="VISIBLE_SURFACE_REPAIR",
            turn_indices=turn_indices,
        )


def test_visible_surface_output_rejects_a_turn_over_the_length_cap() -> None:
    with pytest.raises(ValidationError):
        VisibleSurfaceRepairOutput(
            repaired_turn_contents=["x" * (MAX_REPAIRED_TURN_CONTENT_CHARS + 1)],
            authorized_user_context="fine.",
            session_summary="",
        )


def test_worst_case_token_preflight_functions_are_real_and_positive() -> None:
    field_only = maximum_legal_field_only_repair_output_tokens()
    visible_surface = maximum_legal_visible_surface_repair_output_tokens()
    assert field_only > 0
    assert visible_surface > field_only  # 8 bounded turns dominate the payload
    # Corpus's real observed max history-turn length is 180 chars; the 400-char
    # cap must clear it with real margin -- if this ever regresses, the cap
    # itself (not just this test) needs revisiting.
    assert MAX_REPAIRED_TURN_CONTENT_CHARS > 180


def test_apply_visible_surface_repair_forces_empty_summary_to_stay_empty() -> None:
    # state_44550214... has an originally EMPTY session_summary. Real
    # end-to-end recompilation against the actual v8_18 bundles caught this
    # exact bug: write_development_dataset's summarize_observable_state_
    # support hard-fails ("summary support drifted") if a repair flips an
    # originally-empty summary non-empty, because whether a (user,
    # case_field) cell has a summary at all is a frozen, counterbalanced
    # corpus-design property. The model's session_summary output must be
    # ignored here, never trusted, exactly like the FIELD_ONLY_REPAIR mask.
    state_id = "state_44550214bf7c9fa22284a731"
    state, context = _load_real_state_and_context(state_id)
    assert state["current_session_summary"] == ""
    turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[state_id]
    repair = VisibleSurfaceRepairOutput(
        repaired_turn_contents=[f"repaired turn {i}." for i in turn_indices],
        authorized_user_context="The user is navigating a workplace conflict with a coworker.",
        session_summary="a summary the model produced but that must be IGNORED",
    )
    repaired_state, _ = apply_visible_surface_repair(
        state=state, evaluator_context=context, turn_indices=turn_indices, repair=repair
    )
    assert repaired_state["current_session_summary"] == ""


def test_validate_rejects_a_repair_that_flips_summary_presence() -> None:
    state, context = _load_real_state_and_context("state_032737f91e3c332227f04942")
    assert state["current_session_summary"] == ""
    tampered_state = {**state, "current_session_summary": "a summary that should not exist"}
    with pytest.raises(RuntimeError, match="session_summary is present/absent"):
        validate_repair_allowlist_diff(
            original_state=state,
            original_evaluator_context=context,
            repaired_state=tampered_state,
            repaired_evaluator_context=context,
            repair_mode="FIELD_ONLY_REPAIR",
        )
