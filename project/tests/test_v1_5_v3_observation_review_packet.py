from collections import Counter
from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v1"


def test_observation_packet_has_one_complete_primary_and_fixed_overlap() -> None:
    primary = read_json(OUT / "human_review_packet.json")
    overlap = read_json(OUT / "independent_overlap_packet.json")
    assert len(primary["items"]) == 96
    assert len(overlap["items"]) == 24
    assert Counter(row["component"] for row in primary["items"]) == Counter(
        {"MP": 24, "MS": 24, "ME": 24, "RS": 24}
    )
    assert Counter(row["component"] for row in overlap["items"]) == Counter(
        {"MP": 6, "MS": 6, "ME": 6, "RS": 6}
    )
    assert primary["is_step1_worth_opening_gold"] is False


def test_review_items_do_not_expose_private_design_fields() -> None:
    primary = read_json(OUT / "human_review_packet.json")
    forbidden = {
        "track",
        "topic_family",
        "history_scale",
        "prefix_family",
        "length_direction",
        "private_factor_plan",
        "private_composed_eligibility",
    }
    assert all(not (forbidden & set(row)) for row in primary["items"])
