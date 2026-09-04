from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_final_h1_candidate_gold_v1_candidate"
V2_OUT = ROOT / "outputs/pm_v1_5_final_h1_v2_candidate"
V2_GATE = ROOT / "outputs/pm_v1_5_final_h1_v2_pre_human_static_gate_v1"


def test_final_h1_packet_is_complete_and_private_intent_blind() -> None:
    packet = read_json(OUT / "human_review_packet.json")
    assert len(packet["items"]) == 256
    assert packet["candidate_count"] == 813
    assert packet["absent_count"] == 211
    forbidden = {
        "state_id",
        "split",
        "logic_family",
        "topic_family",
        "construction_mode",
        "intended_action",
        "candidate_id",
        "candidate_text_sha256",
        "model_features",
        "ranked_card_ids",
    }
    for item in packet["items"]:
        assert set(item) == {
            "blind_state_id",
            "visible_dialogue",
            "current_user_text",
            "components",
        }
        assert len(item["components"]) == 4
        assert not (forbidden & set(item))
        for component in item["components"]:
            assert not (forbidden & set(component))
            if component["candidate_present"]:
                if component["component"] == "RS":
                    assert component["candidate_age_sessions"] is None
                else:
                    assert component["candidate_age_sessions"] > 0


def test_final_h1_overlap_is_result_blind_fixed_64() -> None:
    overlap = read_json(OUT / "independent_overlap_packet.json")
    manifest = read_json(OUT / "freeze_manifest.json")
    assert len(overlap["items"]) == 64
    assert manifest["overlap_split_counts"] == {
        "FIT": 32,
        "FRESH_CONFIRMATION": 16,
        "SEALED_INTERNAL_TEST": 16,
    }
    assert manifest["construction_intent_visible_to_reviewer"] is False
    assert manifest["response_or_outcome_read"] is False
    assert manifest["external_lockbox_read"] is False


def test_h1_v2_static_gate_precedes_human_packet() -> None:
    gate = read_json(V2_GATE / "static_gate_report.json")
    manifest = read_json(V2_OUT / "freeze_manifest.json")
    assert gate["status"] == "PASS"
    assert gate["failures"] == []
    assert gate["human_labels_read"] == 0
    assert gate["response_or_outcome_read"] is False
    assert gate["construction_proxy_used_as_h1_gold"] is False
    assert gate["rs_candidate_family_realization"]["mismatches"] == []
    assert manifest["status"] == "READY_FOR_FINAL_H1_V2_REVIEW"
    assert manifest["static_gate_sha256"]


def test_h1_v2_packet_uses_two_gate_contract_without_private_fields() -> None:
    packet = read_json(V2_OUT / "human_review_packet.json")
    overlap = read_json(V2_OUT / "independent_overlap_packet.json")
    manifest = read_json(V2_OUT / "freeze_manifest.json")
    assert len(packet["items"]) == 256
    assert packet["candidate_count"] == 905
    assert packet["absent_count"] == 119
    assert len(overlap["items"]) == 64
    assert overlap["candidate_count"] == 221
    assert packet["actual_paired_response_benefit_is_h2_not_h1"] is True
    assert "applicability_safe=yes" in packet["decision_rule"]
    assert "incremental_over_r0=yes" in packet["decision_rule"]
    assert manifest["construction_intent_is_gold"] is False
    assert manifest["actual_response_benefit_used_as_h1_gold"] is False
    forbidden = {
        "state_id",
        "split",
        "logic_family",
        "topic_family",
        "construction_mode",
        "intended_action",
        "candidate_id",
        "candidate_text_sha256",
        "model_features",
        "ranked_card_ids",
    }
    for item in packet["items"]:
        assert not (forbidden & set(item))
        for component in item["components"]:
            assert not (forbidden & set(component))
