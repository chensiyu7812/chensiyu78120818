from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1"
)
PROTOCOL = "pm-v1.5-transport-repaired-four-component-final-labels-v1"


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_final_effect_labels_are_complete_and_training_ready() -> None:
    report = json.loads(
        (OUT_DIR / "label_report.json").read_text(encoding="utf-8")
    )
    assert report["protocol"] == PROTOCOL
    assert report["status"] == (
        "READY_FOR_FOUR_COMPONENT_GROUPED_OOF_TRAINING"
    )
    assert all(report["checks"].values())
    assert report["row_count"] == 256
    assert report["label_counts"] == {
        "on": 106,
        "off": 150,
        "unknown": 0,
    }
    assert report["risk_review_counts"] == {"no": 106, "yes": 3}
    assert report["risk_event_counts"] == {
        "stale_or_conflicting_evidence_use": 2,
        "unsupported_personal_claim": 2,
    }
    assert report["component_training_ready"] == {
        "RS": True,
        "MP": True,
        "MS": True,
        "ME": True,
    }
    assert report["development_binary_labels_by_component"] == {
        "ME": {"0": 37, "1": 11},
        "MP": {"0": 18, "1": 30},
        "MS": {"0": 30, "1": 18},
        "RS": {"0": 26, "1": 22},
    }
    assert report["internal_test_excluded_from_model_fitting"] is True


def test_material_risk_blocks_candidate_without_causal_overclaim() -> None:
    labels = _rows(OUT_DIR / "component_effect_labels.jsonl")
    assert len(labels) == 256
    assert len({row["contrast_slot_id"] for row in labels}) == 256
    assert Counter(row["component"] for row in labels) == {
        "RS": 64,
        "MP": 64,
        "MS": 64,
        "ME": 64,
    }
    assert all(row["incremental_prompt_tokens"] > 0 for row in labels)

    risk_rows = [
        row
        for row in labels
        if row["component_effect_label"]
        == "OFF_QUALITY_WIN_BUT_MATERIAL_RISK"
    ]
    assert {
        row["risk_review_item_id"]: (
            row["component"],
            row["split"],
            tuple(row["material_risk_categories"]),
        )
        for row in risk_rows
    } == {
        "transport_risk_1dcc48271276240f6167": (
            "MS",
            "internal_test",
            ("stale_or_conflicting_evidence_use",),
        ),
        "transport_risk_1023ad8706c6ffdba55e": (
            "ME",
            "internal_test",
            ("unsupported_personal_claim",),
        ),
        "transport_risk_eb621b59138874822ca2": (
            "MP",
            "train",
            (
                "stale_or_conflicting_evidence_use",
                "unsupported_personal_claim",
            ),
        ),
    }
    assert all(row["target_y"] == 0 for row in risk_rows)
    assert all(
        row[
            "risk_is_candidate_action_admissibility_not_"
            "component_causal_attribution"
        ]
        is True
        for row in risk_rows
    )


def test_validated_risk_evidence_is_preserved() -> None:
    rows = _rows(OUT_DIR / "validated_risk_annotations.jsonl")
    assert len(rows) == 109
    assert len({row["review_item_id"] for row in rows}) == 109
    assert all(
        row["evidence_grounding_validation"] == "PASS" for row in rows
    )
    assert Counter(row["any_material_risk"] for row in rows) == {
        "no": 106,
        "yes": 3,
    }
