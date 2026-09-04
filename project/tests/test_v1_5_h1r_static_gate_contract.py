import json
from collections import Counter

from metacom_pm.v1_5_final_dataset_blueprint import build_h1r_blueprint


def test_h1r_decision_surface_topic_and_scale_are_not_bit_shortcuts() -> None:
    rows = build_h1r_blueprint()
    for component in ("MP", "MS", "ME", "RS"):
        for topic in {row["topic_family"] for row in rows}:
            assert Counter(
                row["private_construction_intent"]["intended_bits"][component]
                for row in rows
                if row["topic_family"] == topic
            ) == Counter({False: 3, True: 3})
        for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
            assert Counter(
                (
                    row["history_shape"],
                    row["private_construction_intent"]["intended_bits"][component],
                )
                for row in rows
                if row["split"] == split
            ) == Counter(
                {
                    ("SMALL", False): 8,
                    ("SMALL", True): 8,
                    ("EVO_LIKE_LARGE", False): 8,
                    ("EVO_LIKE_LARGE", True): 8,
                }
            )


def test_h1r_candidate_rows_never_contain_h1_gold() -> None:
    path = "outputs/pm_v1_5_final_h1r_exact_rank1_v1/candidate_rows_private.jsonl"
    try:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    except FileNotFoundError:
        return
    assert len(rows) == 96
    assert all(row["h1_gold"] is None for row in rows)
    assert all(row["construction_intent_present"] is False for row in rows)
    assert all(row["response_or_outcome_read"] is False for row in rows)
