"""Fail-closed guard tests for scripts/v1_5_recover_actual_468_length_bound_failures.py.

These test only the early guards (manifest/ledger/gate_report binding checks)
that run before the heavy semantic-packet reconstruction (which needs the
project's full ML dependency set). The real, full end-to-end recovery path
-- including packet reconstruction, prompt_sha256 binding, the 24 real
recovered judgments, and the aggregated gate -- was independently verified
against the real actual-468 output directory before this script was written;
see recovery_report.json / recovered_gate_report.json under
outputs/pm_v1_5_actual_corpus_semantic_review_v8_18_deepseek_official_v2_candidate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.io import write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "v1_5_recover_actual_468_length_bound_failures.py"


def _load_recovery_module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_recover_actual_468_length_bound_failures", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def recovery_module():
    return _load_recovery_module()


def _write_ledger(path: Path, real_sha_target: Path) -> str:
    write_jsonl(path, [])
    from metacom_pm.io import sha256_file

    return sha256_file(real_sha_target if real_sha_target.exists() else path)


def _base_fixture(tmp_path: Path) -> dict:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    write_jsonl(ledger_path, [])
    from metacom_pm.io import sha256_file

    real_ledger_sha256 = sha256_file(ledger_path)

    write_json(
        out_dir / "gate_report.json",
        {"status": "INCOMPLETE_NO_GATE_DECISION", "incomplete_calls": []},
    )
    write_jsonl(out_dir / "call_plan.jsonl", [])

    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "stage_consumptions": {
                "development_actual_corpus_semantic_review": {
                    "output_directory": str(out_dir),
                    "status": "CONSUMED_INCOMPLETE",
                    "physical_attempt_ledger_sha256": real_ledger_sha256,
                },
                "development_data_generation": {
                    "status": "CONSUMED_PASS",
                    "output_directory": str(tmp_path / "states"),
                },
            }
        },
    )
    return {
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "real_ledger_sha256": real_ledger_sha256,
    }


def _run_main(module, out_dir: Path, manifest_path: Path) -> None:
    sys.argv = [
        "v1_5_recover_actual_468_length_bound_failures.py",
        "--out-dir",
        str(out_dir),
        "--paid-run-release",
        str(manifest_path),
    ]
    module.main()


def test_refuses_when_out_dir_does_not_match_manifest(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    with pytest.raises(RuntimeError, match="output_directory"):
        _run_main(recovery_module, wrong_dir, fixture["manifest_path"])


def test_refuses_when_stage_status_is_not_consumed_incomplete(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    manifest = __import__("json").loads(fixture["manifest_path"].read_text())
    manifest["stage_consumptions"]["development_actual_corpus_semantic_review"]["status"] = (
        "CONSUMED_PASS"
    )
    write_json(fixture["manifest_path"], manifest)
    with pytest.raises(RuntimeError, match="CONSUMED_INCOMPLETE"):
        _run_main(recovery_module, fixture["out_dir"], fixture["manifest_path"])


def test_refuses_when_ledger_sha256_does_not_match_manifest(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    (fixture["out_dir"] / "physical_attempt_ledger.jsonl").write_text('{"tampered": true}\n')
    with pytest.raises(RuntimeError, match="sha256"):
        _run_main(recovery_module, fixture["out_dir"], fixture["manifest_path"])


def test_refuses_when_data_generation_stage_is_not_consumed_pass(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    manifest = __import__("json").loads(fixture["manifest_path"].read_text())
    manifest["stage_consumptions"]["development_data_generation"]["status"] = "CONSUMED_CRASH"
    write_json(fixture["manifest_path"], manifest)
    with pytest.raises(RuntimeError, match="CONSUMED_PASS"):
        _run_main(recovery_module, fixture["out_dir"], fixture["manifest_path"])


def test_refuses_when_gate_report_status_is_not_incomplete(tmp_path, recovery_module) -> None:
    fixture = _base_fixture(tmp_path)
    write_json(
        fixture["out_dir"] / "gate_report.json",
        {"status": "PASS", "incomplete_calls": []},
    )
    with pytest.raises(RuntimeError, match="INCOMPLETE_NO_GATE_DECISION"):
        _run_main(recovery_module, fixture["out_dir"], fixture["manifest_path"])
