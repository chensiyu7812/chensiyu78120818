from __future__ import annotations

from pathlib import Path

from metacom_pm.v1_5_v5_3_runner_wiring_audit import (
    REQUIRED_CALLS,
    audit_formal_runner_wiring,
    script_call_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repo_does_not_mistake_static_symbol_coverage_for_runtime_readiness() -> None:
    report = audit_formal_runner_wiring(project_root=ROOT)
    assert report["formal_runner_ready"] is False
    assert report["api_calls"] == 0
    if report["complete_candidate_entrypoints"]:
        assert report["status"] == (
            "STATIC_REQUIRED_CALLS_PRESENT_IN_ONE_ENTRYPOINT_NEEDS_RUNTIME_AUDIT"
        )
    else:
        assert report["status"] == "FORMAL_RUNNER_ENTRYPOINT_NOT_YET_IMPLEMENTED"


def test_static_coverage_requires_import_and_call(tmp_path: Path) -> None:
    lines = []
    calls = []
    for _label, (module, symbol) in REQUIRED_CALLS.items():
        lines.append(f"from {module} import {symbol}")
        calls.append(f"    {symbol}()")
    script = tmp_path / "formal.py"
    script.write_text(
        "\n".join(lines + ["", "def run():", *calls, ""]),
        encoding="utf-8",
    )
    assert script_call_coverage(script) == set(REQUIRED_CALLS)
