import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v3_h_eligibility_packet_is_blind_and_balanced() -> None:
    out = ROOT / "outputs/pm_v1_5_v3_h_eligibility_review_candidate"
    manifest = json.loads((out / "freeze_manifest.json").read_text(encoding="utf-8"))
    packet = json.loads((out / "human_review_packet.json").read_text(encoding="utf-8"))
    overlap = json.loads((out / "independent_overlap_packet.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY_FOR_H_ELIGIBILITY"
    assert manifest["items"] == 128
    assert manifest["per_component"] == {"MP": 32, "MS": 32, "ME": 32, "RS": 32}
    assert manifest["overlap_items"] == 32
    assert manifest["overlap_per_component"] == {"MP": 8, "MS": 8, "ME": 8, "RS": 8}
    assert len(packet["items"]) == 128
    assert len(overlap["items"]) == 32
    forbidden = {"split", "logic_family", "private_eligibility_intent", "benefit_enrichment"}
    assert all(not (forbidden & set(item)) for item in packet["items"])
    assert all(item["candidate_text"] for item in packet["items"])
