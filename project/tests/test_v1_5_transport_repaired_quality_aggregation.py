from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_quality_review_is_complete_stable_and_non_degenerate() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_transport_repaired_quality_aggregation_v1/"
            "quality_aggregation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "QUALITY_LABELS_VALID_PENDING_COMPONENT_ON_WIN_RISK_REVIEW"
    )
    assert all(report["checks"].values())
    assert report["review_grain"] == {
        "presentations": 308,
        "independent_contrasts": 256,
        "reliability_repeats": 52,
        "components": {"ME": 64, "MP": 64, "MS": 64, "RS": 64},
    }
    assert report["reliability"]["semantic_direction_agreement"] == 1.0
    assert report["reliability"]["decisive_criterion_agreement"] == 1.0
    assert report["blind_position_profile"]["primary_only"] == {
        "A": 100,
        "B": 107,
        "tie": 49,
    }
    for component, verdicts in report[
        "quality_verdicts_by_component"
    ].items():
        assert component in {"RS", "MP", "MS", "ME"}
        assert verdicts["treatment"] >= 8
        assert verdicts["control"] + verdicts["tie"] >= 8


def test_pre_risk_labels_have_one_row_per_independent_contrast() -> None:
    path = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_quality_aggregation_v1/"
        "quality_effect_labels_pre_risk.jsonl"
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 256
    assert len({row["contrast_slot_id"] for row in rows}) == 256
    assert len({row["state_id"] for row in rows}) == 256
    assert sum(
        row["quality_label_pre_risk"]
        == "ON_QUALITY_WIN_PENDING_MATERIAL_RISK"
        for row in rows
    ) == 109
