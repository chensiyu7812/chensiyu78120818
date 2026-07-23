from __future__ import annotations

import pytest

from metacom_pm.io import write_jsonl
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    EXPECTED_DATA_DEFECT_COUNT,
    EXPECTED_FIELD_ONLY_REPAIR_COUNT,
    EXPECTED_INSTRUMENT_AMBIGUITY_COUNT,
    EXPECTED_TOTAL_RECORDS,
    EXPECTED_VISIBLE_SURFACE_REPAIR_COUNT,
    FROZEN_CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_SHA256,
    data_defect_state_ids,
    field_only_repair_state_ids,
    load_context_grounding_defect_classification,
    visible_surface_repair_state_ids,
)

CLASSIFICATION_PATH = DEFAULT_CLASSIFICATION_PATH


def _minimal_record(**overrides):
    base = {
        "state_id": "state_test",
        "field": "context_grounding_match",
        "bucket": "REJECT",
        "split": "train",
        "user_id": "pmv2_train_u001",
        "regime": "context_only",
        "current_user_text": "text",
        "history": [],
        "session_summary": "",
        "authorized_user_context": "",
        "judges": {},
        "classification": "DATA_DEFECT",
        "defect_type": "garbage",
        "repair_mode": "FIELD_ONLY_REPAIR",
        "rationale": "test rationale",
    }
    base.update(overrides)
    return base


def test_real_frozen_classification_file_matches_expected_sha256_and_counts() -> None:
    assert CLASSIFICATION_PATH.is_file()
    records = load_context_grounding_defect_classification(CLASSIFICATION_PATH)
    assert len(records) == EXPECTED_TOTAL_RECORDS == 91
    n_defect = sum(1 for r in records if r.classification == "DATA_DEFECT")
    n_ambiguity = sum(1 for r in records if r.classification == "INSTRUMENT_AMBIGUITY")
    assert n_defect == EXPECTED_DATA_DEFECT_COUNT == 25
    assert n_ambiguity == EXPECTED_INSTRUMENT_AMBIGUITY_COUNT == 66
    field_only = field_only_repair_state_ids(records)
    surface = visible_surface_repair_state_ids(records)
    assert len(field_only) == EXPECTED_FIELD_ONLY_REPAIR_COUNT == 19
    assert len(surface) == EXPECTED_VISIBLE_SURFACE_REPAIR_COUNT == 6
    assert field_only.isdisjoint(surface)
    assert data_defect_state_ids(records) == field_only | surface


def test_specific_states_are_forced_to_their_required_classification() -> None:
    records = load_context_grounding_defect_classification(CLASSIFICATION_PATH)
    by_id = {r.state_id: r for r in records}
    # Forced per explicit instruction: "girlfriend" does not entail "male",
    # and no evidence establishes a long-term relationship.
    assert by_id["state_8ce4e8be2e8bb4ed6b097b7b"].classification == "DATA_DEFECT"
    assert by_id["state_8ce4e8be2e8bb4ed6b097b7b"].repair_mode == "FIELD_ONLY_REPAIR"
    # Real referent mismatch between history (manager) and current/context
    # (coworker) -- not an empty-summary artifact.
    assert by_id["state_44550214bf7c9fa22284a731"].classification == "DATA_DEFECT"
    assert by_id["state_44550214bf7c9fa22284a731"].repair_mode == "VISIBLE_SURFACE_REPAIR"
    # Cross-state text leakage: "six-and-a-half-year" belongs to a different
    # state's history.
    assert by_id["state_eb4da35de8c37834663cc7e1"].defect_type == "unsupported_fact"


def test_garbage_authorized_context_states_are_flagged_regardless_of_verdict() -> None:
    records = load_context_grounding_defect_classification(CLASSIFICATION_PATH)
    by_id = {r.state_id: r for r in records}
    # These three passed both judges despite a garbage context -- must still
    # be DATA_DEFECT per the "regardless of verdict" rule.
    for sid in (
        "state_d49035f92c3e658eabbb65b4",
        "state_08e30ce683cf9f4c39de1285",
        "state_062b3dd61638973f95199d76",
    ):
        assert by_id[sid].bucket == "PASS"
        assert by_id[sid].classification == "DATA_DEFECT"
        assert by_id[sid].defect_type == "garbage"


def test_loader_rejects_a_tampered_classification_file(tmp_path) -> None:
    tampered = tmp_path / "tampered.jsonl"
    write_jsonl(tampered, [_minimal_record()])
    with pytest.raises(RuntimeError, match="frozen sha256"):
        load_context_grounding_defect_classification(tampered)


def test_loader_rejects_wrong_record_count(tmp_path, monkeypatch) -> None:
    import metacom_pm.v1_5_context_grounding_repair as module

    path = tmp_path / "classification.jsonl"
    write_jsonl(path, [_minimal_record()])
    from metacom_pm.io import sha256_file

    monkeypatch.setattr(
        module,
        "FROZEN_CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_SHA256",
        sha256_file(path),
    )
    with pytest.raises(RuntimeError, match="expected 91"):
        load_context_grounding_defect_classification(path)


def test_loader_rejects_instrument_ambiguity_with_a_data_repair_mode(tmp_path, monkeypatch) -> None:
    import metacom_pm.v1_5_context_grounding_repair as module
    from metacom_pm.io import sha256_file

    records = [
        _minimal_record(
            state_id=f"state_{i:03d}",
            classification="INSTRUMENT_AMBIGUITY",
            defect_type="claim_phrasing_self_referential",
            repair_mode="FIELD_ONLY_REPAIR",  # invalid: must be AUDIT_CONTRACT_FIX_ONLY
        )
        for i in range(module.EXPECTED_TOTAL_RECORDS)
    ]
    path = tmp_path / "classification.jsonl"
    write_jsonl(path, records)
    monkeypatch.setattr(
        module,
        "FROZEN_CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_SHA256",
        sha256_file(path),
    )
    with pytest.raises(RuntimeError, match="AUDIT_CONTRACT_FIX_ONLY"):
        load_context_grounding_defect_classification(path)
