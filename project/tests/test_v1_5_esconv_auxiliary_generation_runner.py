"""End-to-end, no-API wiring tests for
scripts/v1_5/13b_run_esconv_auxiliary_generation_v1_5.py.

Mirrors tests/test_v1_5_esconv_generation_runner.py's pattern for the frozen
ESConv-test generator, but this script is deliberately not coupled to
require_study_freeze/esconv_generation_contract -- the auxiliary track is
training-support data, not the frozen external-evaluation study.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

from metacom_pm.contracts import (
    DialogueTurn,
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    SourceCatalog,
)
from metacom_pm.esconv_v1_5 import ESCONV_V1_5_ALLOWED_ACTIONS
from metacom_pm.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "outputs" / "v1_5_test_fixtures"


@pytest.fixture()
def workdir() -> Path:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="esconv_auxiliary_runner_", dir=FIXTURE_ROOT
    ) as directory:
        yield Path(directory)


def _load_module(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(aux_dir: Path, split: str, *, n_states: int) -> None:
    split_dir = aux_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows = []
    backend_rows = []
    for index in range(n_states):
        state = RuntimeState(
            state_id=f"state_esconv_aux_{split}_{index}",
            card_id=f"card_esconv_aux_{split}_{index}",
            user_id=f"esconv_dialogue_{split}_{index}",
            split="esconv_auxiliary_train",
            semantic_family="esconv_auxiliary_strategy_routing",
            current_user_text=f"I've been feeling overwhelmed lately, turn {index}.",
            current_session_history=[
                DialogueTurn(role="user", content="Work has been a lot recently."),
                DialogueTurn(
                    role="assistant", content="That does sound like a lot to carry."
                ),
            ],
            current_session_summary="",
            session_index=1,
            inventory={
                source: SourceCatalog(available=False, count=0)
                for source in MemorySource
            },
            allowed_actions=list(ESCONV_V1_5_ALLOWED_ACTIONS),
        )
        backend = MemoryBackendRecord(card_id=state.card_id, items=[])
        runtime_rows.append(state.model_dump(mode="json"))
        backend_rows.append(backend.model_dump(mode="json"))
    write_jsonl(split_dir / "runtime_states.jsonl", runtime_rows)
    write_jsonl(split_dir / "memory_backend.jsonl", backend_rows)


def _run_dry_run(aux_dir: Path, split: str, out_root: Path) -> dict:
    module = _load_module(
        "scripts/v1_5/13b_run_esconv_auxiliary_generation_v1_5.py",
        f"v1_5_esconv_auxiliary_generation_runner_test_{id(aux_dir)}_{split}",
    )
    argv = [
        "13b_run_esconv_auxiliary_generation_v1_5.py",
        "--dry-run",
        "--split",
        split,
        "--scope",
        "full",
        "--auxiliary-dir",
        str(aux_dir),
        "--out-root",
        str(out_root),
        "--max-api-calls",
        "1000",
        "--max-estimated-usd",
        "5.0",
        "--max-input-tokens-per-call",
        "12000",
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous
    out_dir = out_root / f"esconv_auxiliary_generation_v1_5_full_{split}"
    return json.loads((out_dir / "cost_estimate.json").read_text(encoding="utf-8"))


def test_dry_run_is_action_first_not_condition_first(workdir):
    _write_fixture(workdir, "train", n_states=4)
    estimate = _run_dry_run(workdir, "train", workdir / "outputs")
    # 4 states x 2 legal actions = 8, never 4 x 4 = 16.
    assert estimate["logical_api_calls"] == 8
    assert estimate["action_filter"] == ["M0+R0", "M0+RS"]
    assert estimate["budget_gate"]["status"] == "PASS"
    assert estimate["contract_bindings"]["scope"] == "esconv_auxiliary_action_first_train"


def test_dry_run_shape_is_stable_across_invocations(workdir):
    # retrieval_attempts records a real measured latency_ms per call, so the
    # full cost_estimate_sha256 is not byte-identical across separate
    # invocations (same pre-existing property as the shared plan_action_sweep
    # used by scripts/v1_5/06 and 14) -- check the substantive, non-timing
    # fields instead.
    _write_fixture(workdir, "calibration", n_states=3)
    first = _run_dry_run(workdir, "calibration", workdir / "outputs_a")
    second = _run_dry_run(workdir, "calibration", workdir / "outputs_b")
    for key in (
        "logical_api_calls",
        "action_filter",
        "estimated_cost_usd",
        "maximum_physical_api_attempts",
        "contract_bindings",
    ):
        assert first[key] == second[key], key


def test_missing_split_directory_fails_closed(workdir):
    with pytest.raises(RuntimeError, match="missing runtime_states"):
        _run_dry_run(workdir, "internal_test", workdir / "outputs")


def test_output_directory_guard_is_wired_in_before_any_expensive_work(workdir):
    """Proves main() calls resolve_first_unconsumed_output_directory with
    the real, scope-qualified out_dir, and does so before plan_action_
    sweep's expensive retrieval work -- not just that the underlying guard
    function itself is correct (see
    test_output_directory_previously_consumed_is_permanently_protected and
    test_resolve_first_unconsumed_output_directory_falls_back_to_a_retry_
    sibling in tests/test_v1_5_latest_protocol_repairs.py for that)."""

    _write_fixture(workdir, "train", n_states=4)
    module = _load_module(
        "scripts/v1_5/13b_run_esconv_auxiliary_generation_v1_5.py",
        f"v1_5_esconv_auxiliary_generation_guard_wiring_test_{id(workdir)}",
    )
    calls: list[Path] = []

    def fake_resolver(out_dir, *, config, config_path):
        calls.append(Path(out_dir))
        raise RuntimeError("guard invoked -- stopping before any real work")

    module.resolve_first_unconsumed_output_directory = fake_resolver
    argv = [
        "13b_run_esconv_auxiliary_generation_v1_5.py",
        "--dry-run",
        "--split",
        "train",
        "--scope",
        "full",
        "--auxiliary-dir",
        str(workdir),
        "--out-root",
        str(workdir / "outputs"),
        "--max-api-calls",
        "1000",
        "--max-estimated-usd",
        "5.0",
        "--max-input-tokens-per-call",
        "12000",
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        with pytest.raises(RuntimeError, match="guard invoked"):
            module.main()
    finally:
        sys.argv = previous
    assert len(calls) == 1
    assert calls[0].name == "esconv_auxiliary_generation_v1_5_full_train"


def test_script_never_imports_audit_only():
    source = (
        ROOT / "scripts" / "v1_5" / "13b_run_esconv_auxiliary_generation_v1_5.py"
    ).read_text(encoding="utf-8")
    _, _, code_after_docstring = source.partition('"""')
    _, _, code = code_after_docstring.partition('"""')
    assert "audit_only" not in code
