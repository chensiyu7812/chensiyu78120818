import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_v3_h_eligibility_reference_v3"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v3_eligibility_reference_is_balanced_frozen_and_not_step1_gold() -> None:
    report = json.loads((OUT / "freeze_report.json").read_text(encoding="utf-8"))
    rows = _jsonl(OUT / "eligibility_reference.jsonl")

    assert report["status"] == "PASS_ELIGIBILITY_REFERENCE_FROZEN_OBSERVATION_QUALIFICATION_NEXT"
    assert report["rows"] == len(rows) == 128
    assert report["failures"] == []
    assert report["posthoc_construction_qa_match"] == "128/128"
    assert report["construction_intent_used_as_gold"] is False
    assert report["eligible_for_observation_development"] is True
    assert report["is_step1_worth_opening_gold"] is False
    assert report["identified_human_gold"] is False
    assert report["python_executable"].endswith("/.venv-pm-v1-5/bin/python")
    assert report["python_version"] == "3.13.2"

    assert report["lineage_counts"] == {
        "v1_exact_surface": 104,
        "v2_exact_dual_review": 16,
        "v2_dual_review_adjudicated": 4,
        "v3_monotone_profile_repair_adjudicated": 4,
    }
    assert len(report["explicit_adjudications"]) == 8
    assert len({row["state_id"] for row in rows}) == 128
    for component in ("MP", "MS", "ME", "RS"):
        labels = Counter(
            row["derived_eligibility"] for row in rows if row["component"] == component
        )
        assert labels == {"eligible": 16, "ineligible": 16}


def test_delta_agreement_and_gate_adjudications_remain_separate() -> None:
    report = json.loads((OUT / "freeze_report.json").read_text(encoding="utf-8"))
    rows = _jsonl(OUT / "eligibility_reference.jsonl")

    assert report["delta_final_decision_agreement"]["raw_agreement"] == 1.0
    increment = report["delta_pre_adjudication_agreement"][
        "specific_nonredundant_increment"
    ]
    assert increment["agreement_count"] == 20
    assert increment["n"] == 24
    assert increment["raw_agreement"] == 20 / 24

    wrong_owner = [
        row
        for row in rows
        if row["annotation_basis"]
        == "V2_DUAL_REVIEW_WITH_FIELD_INDEPENDENCE_ADJUDICATION"
    ]
    assert len(wrong_owner) == 4
    assert all(row["owner_time_valid"] == "no" for row in wrong_owner)
    assert all(row["goal_function_fit"] == "no" for row in wrong_owner)
    assert all(row["specific_nonredundant_increment"] == "yes" for row in wrong_owner)
    assert all(row["derived_eligibility"] == "ineligible" for row in wrong_owner)

    current_request = [
        row
        for row in rows
        if row["annotation_basis"]
        == "V3_LOW_BURDEN_PROFILE_REPAIR_PLUS_REDUNDANCY_ADJUDICATION"
    ]
    assert len(current_request) == 4
    assert all(row["owner_time_valid"] == "yes" for row in current_request)
    assert all(row["goal_function_fit"] == "yes" for row in current_request)
    assert all(row["boundary_burden_fit"] == "yes" for row in current_request)
    assert all(row["specific_nonredundant_increment"] == "no" for row in current_request)
    assert all(row["derived_eligibility"] == "ineligible" for row in current_request)
