"""Fail-closed guard tests for
scripts/v1_5_recover_esconv_auxiliary_judging_rationale_length.py.

These test only the early guards (manifest/ledger/config binding checks, and
the exact-shape checks on each recoverable failure) that run before the
label-aggregation machinery (which needs real states/outcomes/composite
config). The real, full end-to-end recovery path -- including label
aggregation and the quality gates -- was independently verified against the
real esconv_auxiliary_judging_v1_5_full_train output directory before this
script was used for real; see recovery_report.json/recovered_summary.json
under that directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.io import sha256_file, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "scripts" / "v1_5_recover_esconv_auxiliary_judging_rationale_length.py"
)

_SUCCEEDED_KEY = "a" * 64
_FAILED_KEY = "b" * 64

_VALID_VALIDATION_ERRORS = [
    {
        "type": "string_too_long",
        "loc": ["rationale"],
        "msg": "String should have at most 800 characters",
        "ctx": {"max_length": 800},
    }
]

_COMPLETE_PROVIDER_RESPONSE = {
    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
}

_TRUNCATED_PROVIDER_RESPONSE = {
    "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
}

_PARSED_PAYLOAD = {
    "emotional_support": 4,
    "personalization": 4,
    "memory_appropriateness": 4,
    "factual_grounding": 4,
    "temporal_consistency": 4,
    "non_intrusiveness": 4,
    "rationale": "x" * 900,
}


def _load_recovery_module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_recover_esconv_auxiliary_judging_rationale_length", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def recovery_module():
    return _load_recovery_module()


def _call_plan_row(call_key: str) -> dict:
    return {
        "state_id": "state_0",
        "action_id": "M0+R0",
        "judge_family": "deepseek_official",
        "judge_type": "response",
        "physical_call_key": call_key,
    }


def _write_ledger(
    ledger_path: Path,
    *,
    stage: str,
    call_plan: list[dict],
    validation_errors=_VALID_VALIDATION_ERRORS,
    provider_response=_COMPLETE_PROVIDER_RESPONSE,
    parsed_payload=_PARSED_PAYLOAD,
    failed_retry_class: str = "structured_output_validation_error",
) -> None:
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={str(row["physical_call_key"]): 10 for row in call_plan},
        maximum_total_attempts=len(call_plan) * 10,
    )
    succeeded_reservation = ledger.reserve(
        _SUCCEEDED_KEY, record_ids={}, prompt_sha256="s" * 64
    )
    ledger.finish(
        succeeded_reservation,
        succeeded=True,
        request_hash="req-1",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        error=None,
        result={"parsed": parsed_payload},
    )
    failed_reservation = ledger.reserve(
        _FAILED_KEY, record_ids={}, prompt_sha256="t" * 64
    )
    ledger.finish(
        failed_reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="StructuredOutputValidationError: too long",
        result={
            "parsed_payload": parsed_payload,
            "provider_response": provider_response,
            "validation_errors": validation_errors,
            "response_schema": "ResponseJudgeOutput",
        },
        metadata={"retry_class": failed_retry_class},
    )


def _base_fixture(tmp_path: Path, **ledger_kwargs) -> dict:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    call_plan = [_call_plan_row(_SUCCEEDED_KEY), _call_plan_row(_FAILED_KEY)]
    write_jsonl(out_dir / "call_plan.jsonl", call_plan)
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    _write_ledger(
        ledger_path,
        stage="esconv_auxiliary_judging_full_train",
        call_plan=call_plan,
        **ledger_kwargs,
    )
    real_ledger_sha256 = sha256_file(ledger_path)

    config_path = tmp_path / "experiment.yaml"
    pm_v1_5_config_path = tmp_path / "pm_v1_5.yaml"
    config_path.write_text("endpoints: {}\n", encoding="utf-8")
    pm_v1_5_config_path.write_text("version: pm-v1.5\n", encoding="utf-8")

    write_json(
        out_dir / "run_manifest.json",
        {
            "stage": "esconv_auxiliary_judging_full_train",
            "pm_v1_5_config_sha256": sha256_file(pm_v1_5_config_path),
            "experiment_config_sha256": sha256_file(config_path),
        },
    )

    manifest_path = tmp_path / "paid_run_release.json"
    write_json(
        manifest_path,
        {
            "stage_consumptions": {
                "esconv_auxiliary_judging_full_train": {
                    "output_directory": str(out_dir),
                    "status": "CONSUMED_INCOMPLETE_NONREPORTABLE",
                    "physical_attempt_ledger_sha256": real_ledger_sha256,
                }
            }
        },
    )
    return {
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "pm_v1_5_config_path": pm_v1_5_config_path,
    }


def _run_main(module, fixture: dict, **overrides) -> None:
    sys.argv = [
        "v1_5_recover_esconv_auxiliary_judging_rationale_length.py",
        "--out-dir",
        str(fixture["out_dir"]),
        "--paid-run-release",
        str(fixture["manifest_path"]),
        "--config",
        str(fixture["config_path"]),
        "--pm-v1-5-config",
        str(fixture["pm_v1_5_config_path"]),
        "--stage-consumption-key",
        "esconv_auxiliary_judging_full_train",
        "--expected-succeeded-calls",
        str(overrides.get("expected_succeeded_calls", 1)),
        "--expected-recoverable-failures",
        str(overrides.get("expected_recoverable_failures", 1)),
    ]
    module.main()


def test_refuses_when_stage_status_is_not_consumed_incomplete_nonreportable(
    tmp_path, recovery_module
) -> None:
    fixture = _base_fixture(tmp_path)
    manifest = __import__("json").loads(fixture["manifest_path"].read_text())
    manifest["stage_consumptions"]["esconv_auxiliary_judging_full_train"]["status"] = (
        "CONSUMED_PASS"
    )
    write_json(fixture["manifest_path"], manifest)
    with pytest.raises(RuntimeError, match="CONSUMED_INCOMPLETE_NONREPORTABLE"):
        _run_main(recovery_module, fixture)


def test_refuses_when_out_dir_does_not_match_manifest(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    fixture["out_dir"] = wrong_dir
    with pytest.raises(RuntimeError, match="output_directory"):
        _run_main(recovery_module, fixture)


def test_refuses_when_ledger_sha256_does_not_match_manifest(
    tmp_path, recovery_module
) -> None:
    fixture = _base_fixture(tmp_path)
    (fixture["out_dir"] / "physical_attempt_ledger.jsonl").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="sha256"):
        _run_main(recovery_module, fixture)


def test_refuses_when_config_does_not_match_run_manifest(
    tmp_path, recovery_module
) -> None:
    fixture = _base_fixture(tmp_path)
    fixture["pm_v1_5_config_path"].write_text("version: pm-v1.5\nextra: true\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pm_v1_5.yaml"):
        _run_main(recovery_module, fixture)


def test_refuses_when_succeeded_count_does_not_match(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="succeeded logical calls"):
        _run_main(recovery_module, fixture, expected_succeeded_calls=2)


def test_refuses_when_failed_count_does_not_match(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="failed logical calls"):
        _run_main(recovery_module, fixture, expected_recoverable_failures=0)


def test_refuses_when_failure_is_not_structured_output_validation_error(
    tmp_path, recovery_module
) -> None:
    fixture = _base_fixture(tmp_path, failed_retry_class="http_5xx")
    with pytest.raises(RuntimeError, match="not structured_output_validation_error"):
        _run_main(recovery_module, fixture)


def test_refuses_when_validation_errors_shape_is_unexpected(
    tmp_path, recovery_module
) -> None:
    fixture = _base_fixture(
        tmp_path,
        validation_errors=[
            {
                "type": "string_too_long",
                "loc": ["rationale"],
                "msg": "String should have at most 800 characters",
                "ctx": {"max_length": 800},
            },
            {
                "type": "greater_than_equal",
                "loc": ["emotional_support"],
                "msg": "bad score",
                "ctx": {},
            },
        ],
    )
    with pytest.raises(RuntimeError, match="unexpected validation_errors shape"):
        _run_main(recovery_module, fixture)


def test_refuses_when_finish_reason_is_not_complete(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(
        tmp_path, provider_response=_TRUNCATED_PROVIDER_RESPONSE
    )
    with pytest.raises(RuntimeError, match="complete .stop/STOP. finish reason"):
        _run_main(recovery_module, fixture)


def test_recognizes_gemini_native_complete_finish_reason(
    recovery_module,
) -> None:
    provider_finish_reason = recovery_module.require_complete_provider_response(
        {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": []}}
            ],
        }
    )
    assert provider_finish_reason == "STOP"
