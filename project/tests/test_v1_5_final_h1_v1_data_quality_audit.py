from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/pm_v1_5_final_h1_v1_data_quality_audit/audit.json"


def test_h1_v1_is_blocked_before_training() -> None:
    audit = read_json(AUDIT)
    assert audit["status"] == "INVALID_FOR_TRAINING_REBUILD_H1_PACKET"
    assert audit["decision"]["annotations_usable_as_final_gold"] is False
    assert audit["decision"]["training_authorized"] is False


def test_h1_v1_duplicate_and_constant_shortcuts_are_preserved() -> None:
    audit = read_json(AUDIT)
    duplicates = audit["duplicate_audit"]
    assert duplicates["exact_duplicate_groups"] == 37
    assert duplicates["cross_split_exact_duplicate_groups"] == 36
    assert audit["component_summary"]["MP"]["majority_constant_accuracy"] > 0.97
    assert audit["component_summary"]["RS"]["majority_constant_accuracy"] > 0.94
