from collections import Counter

from metacom_pm.v1_5_v3_effect_blueprint import (
    COMPONENTS,
    EFFECT_SPLITS,
    SOURCE_CATALOG_TARGETS,
    audit_blueprint,
    build_blueprint,
)


def test_blueprint_passes_frozen_static_audit() -> None:
    rows = build_blueprint()
    report = audit_blueprint(rows)
    assert report["status"] == "PASS", report["failures"]
    assert report["rows"] == 640
    assert report["unique_users"] == 640
    assert report["eligibility_rows"] == 128
    assert report["effect_rows"] == 512


def test_effect_rows_are_eligible_but_never_gold() -> None:
    rows = [row for row in build_blueprint() if row["track"] == "COMPONENT_EFFECT"]
    assert all(row["intended_eligibility_for_construction_only"] == "ELIGIBLE" for row in rows)
    assert all(row["construction_intent_is_gold"] is False for row in rows)
    assert all(row["worth_opening_gold"] is None for row in rows)
    assert all(row["exact_rank1_candidate"] is None for row in rows)


def test_eligibility_audit_is_balanced_and_never_generates_responses() -> None:
    rows = [row for row in build_blueprint() if row["track"] == "ELIGIBILITY_AUDIT"]
    for component in COMPONENTS:
        subset = [row for row in rows if row["target_component"] == component]
        assert Counter(row["private_eligibility_intent"] for row in subset) == Counter(
            {"ELIGIBLE": 16, "INELIGIBLE": 16}
        )
        assert all(row["effect_treatments"] is None for row in subset)


def test_me_invalid_subtypes_match_the_candidate_the_compiler_must_rediscover() -> None:
    rows = [
        row
        for row in build_blueprint()
        if row["track"] == "ELIGIBILITY_AUDIT"
        and row["target_component"] == "ME"
        and row["private_eligibility_intent"] == "INELIGIBLE"
    ]
    by_family = {row["logic_family"]: row["candidate_subtype_target"] for row in rows}
    assert by_family["ME_CONTEXT_EVENT_NO_OUTCOME"] == "ME_CONTEXT_EVENT"
    assert by_family["ME_UNRESOLVED_EVENT"] == "ME_UNRESOLVED_EVENT"
    assert by_family["ME_VALID_OUTCOME_WRONG_CURRENT_GOAL"] == "ME_REUSABLE_OUTCOME"


def test_effect_nuisance_shape_is_balanced_in_every_component_split() -> None:
    rows = [row for row in build_blueprint() if row["track"] == "COMPONENT_EFFECT"]
    for component in COMPONENTS:
        for split in EFFECT_SPLITS:
            subset = [
                row
                for row in rows
                if row["target_component"] == component and row["split"] == split
            ]
            assert len({row["logic_family"] for row in subset}) == 8
            counts = Counter(row["private_benefit_enrichment"] for row in subset)
            assert counts["HIGH"] == counts["LOW_OR_NEUTRAL"]
            for label in counts:
                topic_counts = Counter(
                    row["topic_family"]
                    for row in subset
                    if row["private_benefit_enrichment"] == label
                )
                assert len(set(topic_counts.values())) == 1
            assert {
                row["source_catalog_size_target"] for row in subset
            } == set(SOURCE_CATALOG_TARGETS[component].values())


def test_counterfactual_pairs_do_not_cross_split_or_component() -> None:
    groups: dict[str, list[dict]] = {}
    for row in build_blueprint():
        groups.setdefault(row["counterfactual_group_id"], []).append(row)
    assert all(len(group) == 2 for group in groups.values())
    assert all(len({row["split"] for row in group}) == 1 for group in groups.values())
    assert all(len({row["target_component"] for row in group}) == 1 for group in groups.values())
    assert all(len({row["user_id"] for row in group}) == 2 for group in groups.values())
