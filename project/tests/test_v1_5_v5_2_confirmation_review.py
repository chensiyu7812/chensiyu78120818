from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/pm_v1_5_v5_2_confirmation_review_v1_candidate"

pytestmark = pytest.mark.skipif(
    not (REVIEW / "review_manifest.json").is_file(),
    reason="V5.2 private human-review artifact bundle is not included in a clean checkout",
)


def jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (REVIEW / name).read_text().splitlines()
        if line.strip()
    ]


def test_confirmation_review_shape_and_exact_deduplication() -> None:
    manifest = json.loads((REVIEW / "review_manifest.json").read_text())
    assert manifest["status"] == (
        "READY_FOR_SINGLE_V5_2_CONFIRMATION_PRIMARY_AND_INDEPENDENT_REVIEW"
    )
    assert manifest["underlying"] == {"quality_pairs": 256, "risk_arm_responses": 512}
    assert manifest["primary"] == {"quality": 255, "risk": 423}
    assert manifest["exact_public_surface_deduplication"] == {
        "quality_rows_propagated": 1,
        "risk_rows_propagated": 89,
        "underlying_statistical_rows_retained": True,
    }
    quality_key = jsonl("private_quality_key.jsonl")
    risk_key = jsonl("private_risk_key.jsonl")
    primary_quality = jsonl("primary_quality_packet.jsonl")
    primary_risk = jsonl("primary_risk_packet.jsonl")
    assert len(quality_key) == 256
    assert len(risk_key) == 512
    assert {row["manual_representative_id"] for row in quality_key} == {
        row["blind_item_id"] for row in primary_quality
    }
    assert {row["manual_representative_id"] for row in risk_key} == {
        row["risk_item_id"] for row in primary_risk
    }


def test_confirmation_overlap_is_group_hashed_and_about_twenty_percent() -> None:
    overlap = json.loads((REVIEW / "private_overlap_groups.json").read_text())
    quality_key = jsonl("private_quality_key.jsonl")
    risk_key = jsonl("private_risk_key.jsonl")
    overlap_quality = jsonl("overlap_quality_packet.jsonl")
    overlap_risk = jsonl("overlap_risk_packet.jsonl")
    groups = set(overlap["groups"])
    assert overlap["group_count"] == 13
    assert len(groups) == 13
    assert sum(row["counterfactual_group_id"] in groups for row in quality_key) == 52
    assert sum(row["counterfactual_group_id"] in groups for row in risk_key) == 104
    selected_quality_reps = {
        row["manual_representative_id"]
        for row in quality_key
        if row["counterfactual_group_id"] in groups
    }
    selected_risk_reps = {
        row["manual_representative_id"]
        for row in risk_key
        if row["counterfactual_group_id"] in groups
    }
    assert selected_quality_reps == {row["blind_item_id"] for row in overlap_quality}
    assert selected_risk_reps == {row["risk_item_id"] for row in overlap_risk}


def test_quality_is_blind_and_risk_authorization_has_no_opaque_ids() -> None:
    for row in jsonl("primary_quality_packet.jsonl"):
        assert set(row) == {
            "protocol",
            "blind_item_id",
            "visible_conversation",
            "response_a",
            "response_b",
        }
    for row in jsonl("primary_risk_packet.jsonl"):
        authorization = row["authorized_evidence_and_instruction"]
        assert "mem_" not in authorization
        assert "strategy_v" not in authorization
        assert "card_" not in authorization


def test_review_html_files_embed_the_expected_panel_roles() -> None:
    expected = {
        "human_quality_primary.html": "primary",
        "human_risk_primary.html": "primary",
        "human_quality_overlap.html": "independent_overlap",
        "human_risk_overlap.html": "independent_overlap",
    }
    for name, role in expected.items():
        text = (REVIEW / name).read_text()
        assert f'"panel_role":"{role}"' in text
        assert "localStorage" in text
        assert "导出JSONL" in text
