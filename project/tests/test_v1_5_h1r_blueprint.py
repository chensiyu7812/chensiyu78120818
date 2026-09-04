from collections import Counter

from metacom_pm.v1_5_final_dataset_blueprint import (
    H1R_RS_FAMILIES,
    audit_h1r_blueprint,
    build_h1r_blueprint,
)


def test_h1r_blueprint_has_exact_frozen_size_and_action_coverage() -> None:
    rows = build_h1r_blueprint()
    report = audit_h1r_blueprint(rows)
    assert report["status"] == "PASS"
    assert report["states"] == 96
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        split_rows = [row for row in rows if row["split"] == split]
        actions = Counter(
            row["private_construction_intent"]["intended_action"]
            for row in split_rows
        )
        assert len(split_rows) == 32
        assert len(actions) == 16
        assert set(actions.values()) == {2}


def test_h1r_breaks_mp_me_and_rs_surface_shortcuts() -> None:
    rows = build_h1r_blueprint()
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        split_rows = [row for row in rows if row["split"] == split]
        mp = Counter(
            (
                row["private_construction_intent"]["component_plans"]["MP"]["candidate_subtype_target"],
                row["private_construction_intent"]["intended_bits"]["MP"],
            )
            for row in split_rows
        )
        assert all(mp[(subtype, bit)] >= 4 for subtype in ("MP_PREFERENCE", "MP_PROFILE") for bit in (False, True))
        me = Counter(
            (
                row["private_construction_intent"]["component_plans"]["ME"]["past_help_invitation_visible"],
                row["private_construction_intent"]["intended_bits"]["ME"],
            )
            for row in split_rows
        )
        assert set(me.values()) == {8}
        rs = Counter(
            (
                row["private_construction_intent"]["component_plans"]["RS"]["strategy_family_target"],
                row["private_construction_intent"]["intended_bits"]["RS"],
            )
            for row in split_rows
        )
        assert all(rs[(family, bit)] >= 4 for family in H1R_RS_FAMILIES for bit in (False, True))


def test_h1r_does_not_expose_construction_gold_or_external_content() -> None:
    for row in build_h1r_blueprint():
        assert row["construction_intent_is_gold"] is False
        assert row["construction_intent_is_model_input"] is False
        assert row["external_content_or_outcome_read"] is False
        assert row["h1_gold"] is None


def test_h1r_topic_cannot_predict_any_component_bit() -> None:
    rows = build_h1r_blueprint()
    for topic in {row["topic_family"] for row in rows}:
        topic_rows = [row for row in rows if row["topic_family"] == topic]
        assert len(topic_rows) == 6
        for component in ("MP", "MS", "ME", "RS"):
            assert Counter(
                row["private_construction_intent"]["intended_bits"][component]
                for row in topic_rows
            ) == Counter({False: 3, True: 3})
