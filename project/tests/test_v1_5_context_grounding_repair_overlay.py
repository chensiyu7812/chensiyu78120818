"""Tests for the repair-overlay + canonical local recompilation mechanism.

test_full_recompilation_against_real_v8_18_bundles is the important one: it
runs the REAL write_development_dataset pipeline (case_to_state, global
family/duplicate/required-hit/observable-state-support validation) against
the actual 52-user/468-case v8_18 bundles with the 25 DATA_DEFECT cases
patched (placeholder text -- the real repair-generation calls have not been
made yet). This is what first caught a real bug: an early version of
apply_visible_surface_repair always wrote a non-empty session_summary,
which flipped 3 states' summary-presence and tripped
summarize_observable_state_support's real hard-fail check
("... summary support drifted"). Fixed in v1_5_context_grounding_data_
repair.py and mirrored in this overlay module; this test locks the fix in.

The "leaves 443 untouched states identical" tests deliberately compare only
the TEXT fields listed (current_user_text/history/summary/semantic_family/
split/user_id/surface_form_id), not the full state object -- this test run
uses a lightweight fixture encoder (not the real frozen BGE snapshot used
for v8_18), so text_embedding/inventory.query_similarity_mean/
step0_observation legitimately differ from the original for ALL 468 states
regardless of which 25 were repaired. A real byte-for-byte-including-
embedding certification requires running with the actual frozen encoder
and is a separate, later step (real repair content also does not exist yet
-- these tests use placeholder text).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import iter_jsonl, sha256_file
from metacom_pm.pm_v1_5_semantic import FrozenSemanticEncoderSpec, SemanticEncoderBinding
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    TRAIN_SEMANTIC_FAMILIES,
)
from metacom_pm.v1_5_context_grounding_data_repair import (
    FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED,
    VISIBLE_SURFACE_REPAIR_TURN_INDICES,
)
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    load_context_grounding_defect_classification,
)
from metacom_pm.v1_5_context_grounding_repair_overlay import (
    RepairOverlayRecord,
    apply_repair_overlay_to_bundles,
    build_repair_overlay_record,
    find_bundle_location_for_state,
    recompile_repaired_development_dataset,
    split_by_user_from_existing_states,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"
STRATEGY_BANK_PATH = ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"


class _FakeEncoder:
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


def _build_placeholder_overlays(bundles, classification_sha256):
    records = load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)
    defects = [r for r in records if r.classification == "DATA_DEFECT"]
    overlays = []
    for record in defects:
        if record.repair_mode == "FIELD_ONLY_REPAIR":
            overlays.append(
                build_repair_overlay_record(
                    record=record,
                    bundles=bundles,
                    classification_sha256=classification_sha256,
                    authorized_user_context="placeholder repaired context for testing only.",
                    session_summary=(
                        "placeholder repaired summary for testing only."
                        if record.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
                        else None
                    ),
                    recent_dialogue_patch={},
                )
            )
        else:
            turn_indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[record.state_id]
            patch = {i: f"Placeholder repaired turn {i} for testing only." for i in turn_indices}
            overlays.append(
                build_repair_overlay_record(
                    record=record,
                    bundles=bundles,
                    classification_sha256=classification_sha256,
                    authorized_user_context="placeholder repaired context for testing only.",
                    session_summary="Placeholder repaired summary for testing only.",
                    recent_dialogue_patch=patch,
                )
            )
    return overlays, defects


@pytest.fixture(scope="module")
def real_bundles():
    bundles_path = DATA_DIR / "pm_v2_bundles.jsonl"
    if not bundles_path.is_file():
        pytest.skip(
            "requires the private local V8.18 development-data artifact; "
            "the artifact is intentionally excluded from Git"
        )
    return load_bundles(str(bundles_path))


@pytest.fixture(scope="module")
def real_classification_sha256():
    return sha256_file(DEFAULT_CLASSIFICATION_PATH)


@pytest.fixture(scope="module")
def real_classification_records():
    return load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)


def _apply(bundles, overlays, classification_records, classification_sha256):
    return apply_repair_overlay_to_bundles(
        bundles=bundles,
        overlays=overlays,
        classification_records=classification_records,
        expected_classification_sha256=classification_sha256,
    )


def test_find_bundle_location_maps_every_defect_state(real_bundles, real_classification_sha256) -> None:
    records = load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)
    defects = [r for r in records if r.classification == "DATA_DEFECT"]
    assert len(defects) == 25
    for record in defects:
        case_id = find_bundle_location_for_state(
            bundles=real_bundles,
            user_id=record.user_id,
            current_user_text=record.current_user_text,
        )
        assert case_id.startswith("case_")


def test_apply_overlay_changes_exactly_25_cases_and_leaves_443_unchanged(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, defects = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    assert len(overlays) == 25
    patched_bundles = _apply(
        real_bundles, overlays, real_classification_records, real_classification_sha256
    )

    orig_by_case_id = {c.case_id: c for b in real_bundles for c in b.cases}
    changed_case_ids = {o.case_id for o in overlays}
    n_changed = 0
    n_unchanged = 0
    for bundle in patched_bundles:
        for case in bundle.cases:
            original = orig_by_case_id[case.case_id]
            if case.case_id in changed_case_ids:
                n_changed += 1
                assert case.model_dump(mode="json") != original.model_dump(mode="json")
            else:
                n_unchanged += 1
                # These 443 ARE compared byte-for-byte here (this is the
                # bundle-level GeneratedStateCase, not the recompiled
                # PMV2State -- no encoder involved at this layer, so full
                # equality is meaningful, unlike the post-recompilation
                # comparison below).
                assert case.model_dump(mode="json") == original.model_dump(mode="json")
    assert n_changed == 25
    assert n_unchanged == 443


def test_repair_overlay_record_survives_a_real_json_round_trip() -> None:
    """Locks in the fix for a real bug: model_dump(mode="json") always
    stringifies recent_dialogue_patch's integer keys (JSON has no integer
    keys), and RepairOverlayRecord's inherited strict=True config used to
    refuse to coerce them back to int on reload -- meaning every persisted
    repair_overlays.jsonl was unloadable by any later consumer."""

    import json

    original = RepairOverlayRecord(
        state_id="s1",
        user_id="u1",
        case_id="c1",
        repair_mode="VISIBLE_SURFACE_REPAIR",
        original_bundle_case_sha256="a" * 64,
        classification_sha256="b" * 64,
        authorized_user_context="hello world context",
        session_summary=None,
        recent_dialogue_patch={0: "turn zero", 3: "turn three"},
    )
    on_disk = json.dumps(original.model_dump(mode="json"))
    reloaded = RepairOverlayRecord.model_validate(json.loads(on_disk))
    assert reloaded == original
    assert set(reloaded.recent_dialogue_patch) == {0, 3}


def test_repair_overlay_record_rejects_colliding_non_canonical_patch_keys() -> None:
    with pytest.raises(ValueError, match="collide after conversion"):
        RepairOverlayRecord(
            state_id="s1",
            user_id="u1",
            case_id="c1",
            repair_mode="VISIBLE_SURFACE_REPAIR",
            original_bundle_case_sha256="a" * 64,
            classification_sha256="b" * 64,
            authorized_user_context="hello world context",
            session_summary=None,
            recent_dialogue_patch={"1": "a", "01": "b"},
        )


def test_apply_overlay_rejects_a_stale_original_case_hash(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    tampered[0] = tampered[0].model_copy(update={"original_bundle_case_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="stale or mismatched overlay"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_duplicate_state_id(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    duplicated = list(overlays) + [overlays[0]]
    with pytest.raises(RuntimeError, match="duplicate state_id"):
        _apply(real_bundles, duplicated, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_a_missing_defect_state(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    subset = list(overlays)[:-1]  # 24 of 25 -- a real missing-coverage bug
    with pytest.raises(RuntimeError, match="do not exactly match"):
        _apply(real_bundles, subset, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_a_state_not_in_the_frozen_defect_set(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    # Build one legitimate overlay for a real INSTRUMENT_AMBIGUITY state --
    # i.e. one that should NOT be repaired at all -- and confirm it is
    # refused rather than silently accepted as an "extra" repair.
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    ambiguity_record = next(
        r for r in real_classification_records if r.classification == "INSTRUMENT_AMBIGUITY"
    )
    extra = build_repair_overlay_record(
        record=ambiguity_record,
        bundles=real_bundles,
        classification_sha256=real_classification_sha256,
        authorized_user_context="should never be applied.",
        session_summary=None,
        recent_dialogue_patch={},
    )
    with pytest.raises(RuntimeError, match="do not exactly match"):
        _apply(
            real_bundles,
            list(overlays) + [extra],
            real_classification_records,
            real_classification_sha256,
        )


def test_apply_overlay_rejects_a_stale_classification_sha256(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    tampered[0] = tampered[0].model_copy(update={"classification_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="different classification_sha256"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_a_field_only_overlay_with_a_dialogue_patch(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    for index, overlay in enumerate(tampered):
        if overlay.repair_mode == "FIELD_ONLY_REPAIR":
            tampered[index] = overlay.model_copy(
                update={"recent_dialogue_patch": {0: "an unauthorized dialogue edit"}}
            )
            break
    with pytest.raises(RuntimeError, match="must not carry a recent_dialogue_patch"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_unauthorized_field_only_summary_rewrite(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    for index, overlay in enumerate(tampered):
        if (
            overlay.repair_mode == "FIELD_ONLY_REPAIR"
            and overlay.state_id not in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
        ):
            tampered[index] = overlay.model_copy(
                update={"session_summary": "an unauthorized summary rewrite"}
            )
            break
    with pytest.raises(RuntimeError, match="leave session_summary byte-identical"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_requires_frozen_field_only_summary_repair(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    for index, overlay in enumerate(tampered):
        if overlay.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED:
            tampered[index] = overlay.model_copy(update={"session_summary": None})
            break
    with pytest.raises(RuntimeError, match="required session_summary repair"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_rejects_an_off_spec_turn_index(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    tampered = list(overlays)
    for index, overlay in enumerate(tampered):
        if overlay.repair_mode == "VISIBLE_SURFACE_REPAIR":
            # Patch an extra, off-spec turn index beyond the frozen set.
            bad_patch = {**overlay.recent_dialogue_patch, 7: "an off-spec turn edit"}
            tampered[index] = overlay.model_copy(update={"recent_dialogue_patch": bad_patch})
            break
    with pytest.raises(RuntimeError, match="frozen spec requires exactly"):
        _apply(real_bundles, tampered, real_classification_records, real_classification_sha256)


def test_apply_overlay_preserves_summary_presence_for_originally_empty_states(
    real_bundles, real_classification_sha256, real_classification_records
) -> None:
    # state_44550214... has an originally-empty session_summary; even though
    # the overlay supplies a non-empty placeholder, it must be forced back
    # to empty -- this is the real bug the full recompilation test caught.
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    patched_bundles = _apply(
        real_bundles, overlays, real_classification_records, real_classification_sha256
    )
    by_case_id = {c.case_id: c for b in patched_bundles for c in b.cases}
    empty_summary_state_ids = {
        "state_44550214bf7c9fa22284a731",
        "state_7d1e36f110bdacbd2a0545f2",
        "state_ba1e526156d1cf3810fa8e98",
    }
    records = {
        r.state_id: r
        for r in load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)
    }
    for state_id in empty_summary_state_ids:
        record = records[state_id]
        case_id = find_bundle_location_for_state(
            bundles=real_bundles, user_id=record.user_id, current_user_text=record.current_user_text
        )
        assert by_case_id[case_id].session_summary == ""


@pytest.fixture(scope="module")
def full_recompilation_report(
    real_bundles, real_classification_sha256, real_classification_records, tmp_path_factory
):
    overlays, _ = _build_placeholder_overlays(real_bundles, real_classification_sha256)
    patched_bundles = _apply(
        real_bundles, overlays, real_classification_records, real_classification_sha256
    )
    split_by_user = split_by_user_from_existing_states(str(DATA_DIR / "pm_v2_states.jsonl"))
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(STRATEGY_BANK_PATH)
    ]
    out_dir = tmp_path_factory.mktemp("v8_19_recompile_test")
    report = recompile_repaired_development_dataset(
        bundles=patched_bundles,
        split_by_user=split_by_user,
        out_dir=out_dir,
        strategy_catalog_count=len(strategy_cards),
        strategy_estimated_tokens=64,
        strategy_top_k=3,
        strategy_bank_sha256=sha256_file(STRATEGY_BANK_PATH),
        strategy_cards=strategy_cards,
        memory_min_score=0.0,
        strategy_min_score=0.0,
        expected_semantic_families_by_split={
            PMV2Split.TRAIN: TRAIN_SEMANTIC_FAMILIES,
            PMV2Split.CALIBRATION: CALIBRATION_SEMANTIC_FAMILIES,
            PMV2Split.INTERNAL_TEST: INTERNAL_TEST_SEMANTIC_FAMILIES,
        },
        semantic_encoder=_FakeEncoder(),
        enforce_required_hit_preflight=False,
    )
    return report, out_dir


def test_full_recompilation_against_real_v8_18_bundles_completes(full_recompilation_report) -> None:
    report, _ = full_recompilation_report
    assert report["status"] == "COMPLETE"
    assert report["n_users"] == 52
    assert report["n_states"] == 468


def test_full_recompilation_leaves_443_untouched_states_textually_identical(
    full_recompilation_report,
) -> None:
    report, out_dir = full_recompilation_report
    records = load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)
    defect_ids = {r.state_id for r in records if r.classification == "DATA_DEFECT"}

    old_states = {row["state_id"]: row for row in iter_jsonl(DATA_DIR / "pm_v2_states.jsonl")}
    new_states = {row["state_id"]: row for row in iter_jsonl(Path(out_dir) / "pm_v2_states.jsonl")}
    assert set(old_states) == set(new_states)

    text_fields = [
        "current_user_text",
        "current_session_history",
        "current_session_summary",
        "semantic_family",
        "split",
        "user_id",
        "surface_form_id",
    ]
    n_untouched_identical = 0
    n_touched_changed = 0
    for state_id, old in old_states.items():
        new = new_states[state_id]
        identical = all(old[field] == new[field] for field in text_fields)
        if state_id in defect_ids:
            if not identical:
                n_touched_changed += 1
        else:
            assert identical, f"untouched state {state_id} drifted"
            n_untouched_identical += 1
    assert n_untouched_identical == 443
    # The 6 VISIBLE_SURFACE_REPAIR states touch history and (when present)
    # summary; exactly 2 FIELD_ONLY_REPAIR states additionally carry a frozen,
    # independently-required summary repair. The other 17 field-only states
    # change evaluator context only.
    assert n_touched_changed == 8


def test_full_recompilation_updates_authorized_user_context_for_all_25_defects(
    full_recompilation_report,
) -> None:
    report, out_dir = full_recompilation_report
    records = load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)
    defect_ids = {r.state_id for r in records if r.classification == "DATA_DEFECT"}

    old_ec = {row["state_id"]: row for row in iter_jsonl(DATA_DIR / "evaluator_contexts.jsonl")}
    new_ec = {
        row["state_id"]: row for row in iter_jsonl(Path(out_dir) / "evaluator_contexts.jsonl")
    }
    n_untouched_identical = 0
    n_touched_changed = 0
    for state_id, old in old_ec.items():
        new = new_ec[state_id]
        same = old["authorized_user_context"] == new["authorized_user_context"]
        if state_id in defect_ids:
            assert not same, f"defect state {state_id} authorized_user_context unchanged"
            n_touched_changed += 1
        else:
            assert same, f"untouched state {state_id} authorized_user_context drifted"
            n_untouched_identical += 1
    assert n_untouched_identical == 443
    assert n_touched_changed == 25
