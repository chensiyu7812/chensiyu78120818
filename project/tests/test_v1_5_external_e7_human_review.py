from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/pm_v1_5_v5_2_external_e7_human_review_v1_candidate"


def jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (REVIEW / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_external_e7_is_outcome_blind_to_e6_and_has_frozen_clusters() -> None:
    manifest = json.loads((REVIEW / "review_manifest.json").read_text())
    assert manifest["status"] == (
        "READY_FOR_FINAL_EXTERNAL_E7_PRIMARY_AND_INDEPENDENT_REVIEW"
    )
    assert manifest["api_calls"] == 0
    assert manifest["selected_clusters"] == {
        "quality": {
            "esconv_dialogues": 64,
            "evoemo_users": 12,
            "evoemo_core_states": 48,
            "evoemo_raw_states": 24,
        },
        "risk": {
            "esconv_dialogues": 32,
            "evoemo_users": 12,
            "evoemo_core_states": 24,
            "evoemo_raw_states": 12,
        },
    }
    assert manifest["attestations"] == {
        "e6_judge_outcomes_read": False,
        "sampling_used_llm_judge_result": False,
        "sampling_used_quality_or_risk_outcome": False,
        "pm_route_generator_or_resource_changed": False,
        "quality_public_packet_blind_to_policy_component_resource_domain": True,
        "risk_and_quality_constructs_separate": True,
        "es_memeval_qa_excluded_from_e7": True,
    }
    assert not any("e6" in key.lower() for key in manifest["input_sha256"])


def test_external_e7_panel_shape_alias_collapse_and_overlap() -> None:
    manifest = json.loads((REVIEW / "review_manifest.json").read_text())
    quality = jsonl("primary_quality_packet.jsonl")
    risk = jsonl("primary_risk_packet.jsonl")
    overlap_quality = jsonl("overlap_quality_packet.jsonl")
    overlap_risk = jsonl("overlap_risk_packet.jsonl")
    ties = jsonl("automatic_exact_identity_ties.jsonl")
    assert manifest["quality"]["plan_level_comparisons_after_alias_collapse"] == 340
    assert len(ties) == manifest["quality"]["automatic_exact_identity_ties"] == 84
    assert len(quality) == manifest["quality"]["primary_public_surfaces"] == 256
    assert len(risk) == manifest["risk"]["primary_public_surfaces"] == 186
    assert len(overlap_quality) == manifest["independent_overlap"]["quality"] == 51
    assert len(overlap_risk) == manifest["independent_overlap"]["risk"] == 37
    assert abs(len(overlap_quality) / len(quality) - 0.2) < 0.01
    assert abs(len(overlap_risk) / len(risk) - 0.2) < 0.01
    assert {row["blind_item_id"] for row in overlap_quality} <= {
        row["blind_item_id"] for row in quality
    }
    assert {row["risk_item_id"] for row in overlap_risk} <= {
        row["risk_item_id"] for row in risk
    }
    assert manifest["quality"]["per_comparator_including_automatic_ties"] == {
        "esconv:core:always_off": 64,
        "esconv:core:cost_matched_fixed": 64,
        "esconv:core:fixed_high_eligible": 64,
        "esconv:core:transparent_rule": 64,
        "evoemo:core:always_off": 48,
        "evoemo:core:cost_matched_fixed": 48,
        "evoemo:core:fixed_high_eligible": 48,
        "evoemo:core:transparent_rule": 48,
        "evoemo:raw:all_raw_sessions_plus_frozen_strategy": 24,
        "evoemo:raw:raw_session_top4_plus_frozen_strategy": 24,
    }
    assert manifest["risk"]["per_policy_or_condition_underlying"] == {
        "esconv:core:always_off": 32,
        "esconv:core:cost_matched_fixed": 32,
        "esconv:core:fixed_high_eligible": 32,
        "esconv:core:learned_pm": 32,
        "esconv:core:transparent_rule": 32,
        "evoemo:core:always_off": 24,
        "evoemo:core:cost_matched_fixed": 24,
        "evoemo:core:fixed_high_eligible": 24,
        "evoemo:core:learned_pm": 24,
        "evoemo:core:transparent_rule": 24,
        "evoemo:raw:all_raw_sessions_plus_frozen_strategy": 12,
        "evoemo:raw:raw_session_top4_plus_frozen_strategy": 12,
    }


def test_external_e7_public_surfaces_hide_private_policy_and_ids() -> None:
    for row in jsonl("primary_quality_packet.jsonl"):
        assert set(row) == {
            "protocol",
            "blind_item_id",
            "visible_conversation",
            "response_a",
            "response_b",
        }
        blob = json.dumps(row, ensure_ascii=False)
        assert "learned_pm" not in blob
        assert "fixed_high" not in blob
        assert "evoemo" not in blob.lower()
        assert "esconv" not in blob.lower()
    for row in jsonl("primary_risk_packet.jsonl"):
        assert set(row) == {
            "protocol",
            "risk_item_id",
            "visible_conversation",
            "authorized_evidence_and_instruction",
            "candidate_response",
        }
        evidence = row["authorized_evidence_and_instruction"]
        assert "mem_" not in evidence
        assert "strategy_v" not in evidence
        assert "card_" not in evidence


def test_external_e7_html_has_separate_primary_and_overlap_storage() -> None:
    expected = {
        "human_quality_primary.html": "primary",
        "human_risk_primary.html": "primary",
        "human_quality_overlap.html": "independent_overlap",
        "human_risk_overlap.html": "independent_overlap",
    }
    for name, role in expected.items():
        text = (REVIEW / name).read_text(encoding="utf-8")
        assert f'"panel_role":"{role}"' in text
        assert "localStorage" in text
        assert "导出JSONL" in text


def test_external_e7_sample_is_one_seed_per_state_and_all_evo_users_covered() -> None:
    sample = json.loads(
        (REVIEW / "sample_identity_preoutcome.json").read_text(encoding="utf-8")
    )
    assert set(sample["selected_seed_by_state"].values()) <= {"seed-a", "seed-b"}
    assert len(sample["esconv_quality_states"]) == 64
    assert len(sample["evoemo_quality_states_by_user"]) == 12
    assert all(len(states) == 4 for states in sample["evoemo_quality_states_by_user"].values())
    assert all(
        len(states) == 2
        for states in sample["evoemo_raw_quality_states_by_user"].values()
    )
