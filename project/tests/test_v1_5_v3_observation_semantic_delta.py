from collections import Counter
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_v3_observation_semantic_repair_delta_v2_candidate"


def test_semantic_repair_is_bounded_and_carries_forward_unchanged_labels() -> None:
    packet = read_json(OUT / "human_review_packet.json")
    manifest = read_json(OUT / "freeze_manifest.json")
    carry = list(iter_jsonl(OUT / "carry_forward_reference.jsonl"))
    assert len(packet["items"]) == 16
    assert Counter(row["component"] for row in packet["items"]) == Counter(
        {"RS": 11, "MP": 5}
    )
    assert len(carry) == 80
    assert manifest["full_re_review_required"] is False
    assert manifest["api_calls"] == 0
    assert manifest["responses_generated"] == 0


def test_repaired_full_packet_keeps_exact_rank1_and_no_private_fields() -> None:
    packet = read_json(
        ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v2/human_review_packet.json"
    )
    assert len(packet["items"]) == 96
    forbidden = {"track", "private_factor_plan", "private_composed_eligibility"}
    assert all(not (forbidden & set(row)) for row in packet["items"])
