from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET_DIR = (
    ROOT
    / "outputs/pm_v1_5_transport_repaired_on_win_risk_review_v1_candidate"
)
SOURCE_PACKET_DIR = (
    ROOT / "outputs/pm_v1_5_transport_repaired_four_component_blind_v1"
)
PROTOCOL = (
    "pm-v1.5-transport-repaired-component-on-win-minimal-risk-review-v1"
)
RISK_CATEGORIES = {
    "explicit_boundary_violation",
    "unsupported_personal_claim",
    "stale_or_conflicting_evidence_use",
    "excessive_directiveness",
}


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_final_on_win_risk_packet_is_complete_and_blinded() -> None:
    manifest = json.loads(
        (PACKET_DIR / "risk_review_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    items = _rows(PACKET_DIR / "human_risk_packet.jsonl")
    private = _rows(PACKET_DIR / "private_risk_key.jsonl")
    templates = _rows(
        PACKET_DIR / "human_risk_annotation_template.jsonl"
    )

    assert manifest["protocol"] == PROTOCOL
    assert manifest["status"] == (
        "READY_FOR_FINAL_MINIMAL_ON_WIN_RISK_REVIEW"
    )
    assert manifest["review_item_count"] == 109
    assert manifest["questions_per_item"] == 1
    assert manifest["quality_is_not_rejudged"] is True
    assert set(manifest["risk_categories"]) == RISK_CATEGORIES
    assert len(items) == len(private) == len(templates) == 109

    item_ids = {row["review_item_id"] for row in items}
    private_ids = {row["review_item_id"] for row in private}
    template_ids = {row["review_item_id"] for row in templates}
    assert len(item_ids) == 109
    assert item_ids == private_ids == template_ids
    assert Counter(row["component"] for row in private) == {
        "RS": 29,
        "MP": 39,
        "MS": 24,
        "ME": 17,
    }

    forbidden_public_keys = {
        "component",
        "contrast_slot_id",
        "source_blind_item_id",
        "state_id",
        "user_id",
        "split",
        "control_action",
        "treatment_action",
        "selected_strategy_card_id",
        "selected_strategy_core_submove_id",
        "memory_id",
        "source",
    }
    expected_public_keys = {
        "protocol",
        "review_item_id",
        "current_session_summary",
        "recent_dialogue",
        "current_user_text",
        "authorized_auxiliary_evidence",
        "candidate_response",
    }
    for item in items:
        assert set(item) == expected_public_keys
        assert not forbidden_public_keys.intersection(item)
        assert item["protocol"] == PROTOCOL
        for evidence in item["authorized_auxiliary_evidence"]:
            assert set(evidence) == {
                "evidence_id",
                "relative_age_sessions",
                "content",
            }
            assert evidence["relative_age_sessions"] >= 0

    for row in templates:
        assert row == {
            "protocol": PROTOCOL,
            "review_item_id": row["review_item_id"],
            "any_material_risk": None,
            "selected_categories": [],
            "evidence_by_category": {},
            "risk_notes": "",
            "annotator_id": "",
        }


def test_public_candidate_is_exact_component_on_quality_winner() -> None:
    items = {
        row["review_item_id"]: row
        for row in _rows(PACKET_DIR / "human_risk_packet.jsonl")
    }
    private = {
        row["review_item_id"]: row
        for row in _rows(PACKET_DIR / "private_risk_key.jsonl")
    }
    source_packets = {
        row["blind_item_id"]: row
        for row in _rows(SOURCE_PACKET_DIR / "human_blind_packet.jsonl")
    }
    source_keys = {
        row["blind_item_id"]: row
        for row in _rows(SOURCE_PACKET_DIR / "private_blind_key.jsonl")
    }

    for review_id, key in private.items():
        item = items[review_id]
        source_id = key["source_blind_item_id"]
        source_packet = source_packets[source_id]
        source_key = source_keys[source_id]
        assert source_key["is_reliability_repeat"] is False
        treatment_field = (
            "response_a"
            if source_key["a_role"] == "treatment"
            else "response_b"
        )
        assert item["candidate_response"] == source_packet[treatment_field]
        assert _sha(item["candidate_response"]) == key[
            "candidate_response_sha256"
        ]
        public_evidence = item["authorized_auxiliary_evidence"]
        private_evidence = key["selected_memory_evidence"]
        assert len(public_evidence) == len(private_evidence)
        for public, hidden in zip(
            public_evidence,
            private_evidence,
            strict=True,
        ):
            assert public["evidence_id"] == hidden["evidence_id"]
            assert public["relative_age_sessions"] == hidden[
                "relative_age_sessions"
            ]
            assert _sha(public["content"]) == hidden["content_sha256"]


def test_html_contains_no_private_machine_identifiers() -> None:
    page = (PACKET_DIR / "human_risk_review.html").read_text(
        encoding="utf-8"
    )
    private = _rows(PACKET_DIR / "private_risk_key.jsonl")
    assert "这不是第二轮质量评审" in page
    assert "interaction-and-grounding risk proxy" in page
    for row in private:
        for key in (
            "contrast_slot_id",
            "source_blind_item_id",
            "state_id",
            "user_id",
            "control_action",
            "treatment_action",
            "selected_strategy_card_id",
            "selected_strategy_core_submove_id",
        ):
            value = row.get(key)
            if value:
                assert str(value) not in page
        for evidence in row["selected_memory_evidence"]:
            assert evidence["memory_id"] not in page
