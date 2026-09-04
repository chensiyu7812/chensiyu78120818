import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_observation_reference_is_complete_balanced_and_not_step1_gold() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_human_reference_v2/reference_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS_REFERENCE_FROZEN"
    assert report["rows"] == 96
    assert report["carry_forward_rows"] == 80
    assert report["delta_rows"] == 16
    assert report["factor_fit_exact_8_8_pass"] is True
    assert report["eligibility_confirmation_exact_4_4_pass"] is True
    assert report["uncertain_rows"] == 0
    assert report["step1_worth_opening_labels_created"] == 0
    assert report["response_or_outcome_read"] is False
    assert report["external_lockbox_read"] is False


def test_primary_and_nli_candidates_failed_without_consuming_confirmation() -> None:
    primary = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_orthogonal_bakeoff_v2/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    nli = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_nli_recovery_v1/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    assert primary["selected_candidate"] is None
    assert primary["one_shot_confirmation"] is None
    assert primary["step2_qualification_authorized"] is False
    assert nli["fit_pass"] is False
    assert nli["one_shot_confirmation"] is None
    assert nli["confirmation_read_before_fit_decision"] is False
    assert nli["step2_qualification_authorized"] is False

