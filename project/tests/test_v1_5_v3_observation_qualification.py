import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_transparent_observation_v1_failure_is_frozen_at_the_correct_layer() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_qualification_v1/report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "FAIL_TRANSPARENT_OBSERVATION_REPAIR_REQUIRED"
    assert report["role"] == "OBSERVATION_QUALIFICATION_NOT_STEP1_GOLD_NOT_STEP2_EVALUATION"
    assert report["step1_training_authorized"] is False
    assert report["bge_used"] is False
    assert report["response_or_outcome_read"] is False
    assert report["external_lockbox_read"] is False
    assert report["python_executable"].endswith("/.venv-pm-v1-5/bin/python")
    assert report["python_version"] == "3.13.2"

    factors = report["global_factor_metrics"]
    assert factors["owner_time_entity_valid"]["status"] == "PASS"
    assert factors["specific_increment"]["status"] == "PASS"
    assert factors["goal_function_fit"]["status"] == "FAIL"
    assert factors["boundary_burden_compatible"]["status"] == "FAIL"
    assert report["composite_eligibility_diagnostic"]["balanced_accuracy"] == 0.6484375
    assert all(
        item["status"] == "FAIL"
        for item in report["minimal_counterfactual_direction"].values()
    )
