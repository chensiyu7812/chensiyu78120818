from collections import Counter

from metacom_pm.v1_5_v3_observation_orthogonal import (
    COMPONENTS,
    FACTORS,
    LEARNED_FACTORS,
    audit_blueprint,
    build_blueprint,
)


def test_orthogonal_blueprint_passes_static_structure_gate() -> None:
    rows = build_blueprint()
    report = audit_blueprint(rows)
    assert report["status"] == "PASS", report["failures"]
    assert report["rows"] == 96
    assert report["factor_fit_rows"] == 64
    assert report["eligibility_confirmation_rows"] == 32
    assert report["unique_decision_surfaces"] == 96
    assert report["pairwise_phi_absolute_max"] == 0.0


def test_factor_fit_has_exact_cells_and_rs_owner_is_not_fabricated() -> None:
    rows = [row for row in build_blueprint() if row["track"] == "FACTOR_FIT"]
    for component in COMPONENTS:
        subset = [row for row in rows if row["target_component"] == component]
        assert len(subset) == 16
        for factor in FACTORS:
            values = Counter(
                row["private_factor_plan"][factor]["private_target"] for row in subset
            )
            if factor in LEARNED_FACTORS[component]:
                assert values == Counter({False: 8, True: 8})
            else:
                assert component == "RS"
                assert values == Counter({True: 16})


def test_confirmation_is_balanced_but_never_used_as_training_gold() -> None:
    rows = [
        row for row in build_blueprint() if row["track"] == "ELIGIBILITY_CONFIRMATION"
    ]
    for component in COMPONENTS:
        subset = [row for row in rows if row["target_component"] == component]
        assert Counter(row["private_composed_eligibility"] for row in subset) == Counter(
            {True: 4, False: 4}
        )
    assert all(row["construction_intent_is_gold"] is False for row in rows)
    assert all(row["step1_worth_opening_gold"] is None for row in rows)


def test_no_response_or_external_information_is_consumed() -> None:
    rows = build_blueprint()
    assert all(row["response_generated"] is False for row in rows)
    assert all(row["external_text_or_outcome_read"] is False for row in rows)
    assert all(row["actual_rank1_candidate"] is None for row in rows)
