import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_observation_bakeoff_rejects_all_candidates_and_blocks_step2() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_factor_bakeoff_v1/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "FAIL_NO_OBSERVATION_CANDIDATE_QUALIFIED"
    assert report["selected_candidate"] is None
    assert report["final_model_fit"] is False
    assert report["step1_training_authorized"] is False
    assert report["step2_qualification_authorized"] is False
    assert report["response_or_outcome_read"] is False
    assert report["external_lockbox_read"] is False
    assert report["python_executable"].endswith("/.venv-pm-v1-5/bin/python")
    assert report["python_version"] == "3.13.2"
    assert report["data_quality"]["unique_states"] == 128
    assert report["data_quality"]["nuisance_probe_pass"] is False
    assert all(
        candidate["status"] == "FAIL_ONE_OR_MORE_GATES"
        for candidate in report["results"].values()
    )


def test_bge_has_local_signal_but_not_required_generalization() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_factor_bakeoff_v1/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    bge = report["results"]["bge_small_hybrid_v1"]["factors"]
    assert bge["owner_time_entity_valid"]["status"] == "PASS"
    assert bge["goal_function_fit"]["balanced_accuracy"] >= 0.79
    assert bge["goal_function_fit"]["minimal_counterfactual_direction"]["rate"] < 0.75
    assert bge["boundary_burden_compatible"]["leave_semantic_family_out"][
        "balanced_accuracy"
    ] < 0.60
    assert bge["specific_increment"]["leave_semantic_family_out"][
        "balanced_accuracy"
    ] < 0.60
