from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from metacom_pm.config import load_config
from metacom_pm.v1_5_judge_isolation import (
    final_judge_endpoint_names,
    judge_role_isolation_report,
    require_judge_role_isolation,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    return (
        load_config(ROOT / "configs" / "experiment.yaml"),
        load_config(ROOT / "configs" / "pm_v1_5.yaml"),
    )


def test_current_v1_5_judge_roles_are_hard_isolated() -> None:
    experiment, pm_config = _configs()
    report = require_judge_role_isolation(experiment, pm_config)
    assert report["status"] == "PASS"
    assert {row["name"] for row in report["development_endpoints"]} == {
        "training_judge_gemini_flash_lite",
        "training_judge_deepseek_flash",
    }
    gemini = next(
        row
        for row in report["development_endpoints"]
        if row["family"] == "google_gemini"
    )
    assert gemini["transport"] == "gemini_generate_content"
    assert gemini["base_url"].endswith("/v1beta")
    assert set(final_judge_endpoint_names(pm_config)) == {
        "final_judge",
        "final_judge_claude",
    }


@pytest.mark.parametrize(
    ("development_names", "expected_error"),
    [
        (
            ["training_judge_deepseek_flash", "final_judge"],
            "endpoint aliases overlap",
        ),
        (
            ["training_judge_deepseek_flash", "training_judge_openai_mini"],
            "declared model families overlap",
        ),
    ],
)
def test_final_judge_leakage_fails_closed(
    development_names: list[str], expected_error: str
) -> None:
    experiment, pm_config = _configs()
    report = judge_role_isolation_report(
        experiment,
        pm_config,
        development_endpoint_names=development_names,
    )
    assert report["status"] == "FAIL"
    assert any(expected_error in error for error in report["errors"])
    with pytest.raises(RuntimeError, match="judge-role isolation failed"):
        require_judge_role_isolation(
            experiment,
            pm_config,
            development_endpoint_names=development_names,
        )


def test_same_final_model_under_a_new_alias_is_rejected() -> None:
    experiment, pm_config = _configs()
    experiment = deepcopy(experiment)
    experiment["endpoints"]["development_alias_of_final"] = {
        **experiment["endpoints"]["final_judge"],
        "family": "invented_alias_family",
    }
    report = judge_role_isolation_report(
        experiment,
        pm_config,
        development_endpoint_names=[
            "training_judge_deepseek_flash",
            "development_alias_of_final",
        ],
    )
    assert report["status"] == "FAIL"
    assert report["overlap"]["model_identifiers"] == ["gpt-4o"]
    assert report["overlap"]["resolved_routes"]
