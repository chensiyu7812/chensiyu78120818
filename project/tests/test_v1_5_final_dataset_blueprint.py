from collections import Counter, defaultdict

from metacom_pm.v1_5_final_dataset_blueprint import (
    COMPONENTS,
    FIT_LOGIC_FAMILIES,
    audit_primary_blueprint,
    build_primary_blueprint,
)


def test_final_primary_blueprint_is_balanced_and_nonconfounded() -> None:
    rows = build_primary_blueprint()
    report = audit_primary_blueprint(rows)
    assert report["status"] == "PASS"
    assert report["primary_rows"] == 256
    assert report["unique_users"] == 256

    fit = [row for row in rows if row["split"] == "FIT"]
    assert set(
        Counter(
            row["private_construction_intent"]["intended_action"] for row in fit
        ).values()
    ) == {8}
    by_family = defaultdict(list)
    for row in fit:
        by_family[row["logic_family"]].append(row)
    assert set(by_family) == set(FIT_LOGIC_FAMILIES)
    for family_rows in by_family.values():
        for component in COMPONENTS:
            assert Counter(
                row["private_construction_intent"]["intended_bits"][component]
                for row in family_rows
            ) == Counter({False: 4, True: 4})
        peers = Counter(
            peer["flipped_component"]
            for row in family_rows
            for peer in row["private_counterfactual_peers"]
        )
        assert peers == Counter({component: 4 for component in COMPONENTS})


def test_blueprint_intent_is_explicitly_private_not_gold_or_feature() -> None:
    for row in build_primary_blueprint():
        assert row["construction_intent_is_gold"] is False
        assert row["construction_intent_is_model_input"] is False
        assert row["actual_rank1_candidates"] is None
        assert row["h1_gold"] is None
