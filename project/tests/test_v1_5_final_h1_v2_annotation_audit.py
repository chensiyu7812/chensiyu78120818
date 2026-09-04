from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/pm_v1_5_final_h1_v2_annotation_audit_v1/audit_report.json"


def test_h1_v2_primary_annotations_are_preserved_but_training_is_blocked() -> None:
    audit = read_json(AUDIT)
    assert audit["status"] == "HOLD_DO_NOT_TRAIN"
    assert audit["primary_contract"]["rows"] == 256
    assert audit["primary_contract"]["schema_error_count"] == 0
    assert "meaningful_state_signature_cross_split_duplicates" in audit["failures"]
    assert "row_level_secondary_overlap_missing_iaa_not_computable" in audit["failures"]


def test_h1_v2_data_quality_failures_are_measured() -> None:
    audit = read_json(AUDIT)
    duplicates = audit["meaningful_state_duplicate_audit"]
    assert duplicates["cross_split_duplicate_groups"] == 40
    assert duplicates["cross_split_affected_states"] == 80
    assert (
        audit["human_label_distribution"]["MS"]["overall"]
        ["majority_constant_accuracy"]
        > 0.73
    )
    assert (
        audit["single_cue_shortcut_audit"]["cue_only_balanced_accuracy"]
        > 0.88
    )
    assert audit["candidate_subtype_distribution"]["MP"]["MP_PREFERENCE"] == {
        "off": 72
    }
    assert (
        audit["RS_strategy_family_shortcut_audit"]
        ["family_only_oracle_balanced_accuracy"]
        > 0.84
    )
