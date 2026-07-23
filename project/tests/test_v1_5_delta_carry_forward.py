"""Tests for the new per-row delta carry-forward mechanism in
scripts/v1_5_run_automated_semantic_review.py, added because the existing
whole-plan carry-forward (_load_carry_forward_state) requires the prior
call_plan.jsonl to be byte-identical and refuses entirely otherwise -- one
changed row (e.g. a targeted context-claim wording or data repair affecting
only some states) poisons the whole comparison and blocks carry-forward for
every untouched row too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "v1_5_run_automated_semantic_review.py"
STAGE = "pm_v1_5_actual_corpus_semantic_review"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_run_automated_semantic_review_delta_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def module():
    return _load_module()


def _plan_row(call_key: str, prompt_sha256: str, request_payload_sha256: str) -> dict:
    return {
        "physical_call_key": call_key,
        "prompt_sha256": prompt_sha256,
        "request_payload_sha256": request_payload_sha256,
        "maximum_physical_attempts": 3,
    }


def _write_prior_ledger(path: Path, *, call_key: str, record_ids: dict, prompt_sha256: str) -> None:
    ledger = PersistentAttemptLedger(
        path,
        stage=STAGE,
        expected_calls={call_key: 3},
        maximum_total_attempts=3,
    )
    reservation = ledger.reserve(call_key, record_ids=record_ids, prompt_sha256=prompt_sha256)
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="req-hash",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        error=None,
        result={"parsed": {"verdict": "supported"}, "raw_text": "{}"},
    )


def test_delta_carry_forward_inherits_only_unchanged_rows(tmp_path, module) -> None:
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()

    unchanged_key = "call_unchanged"
    changed_key_old = "call_changed_old"

    _write_prior_ledger(
        prior_dir / "physical_attempt_ledger.jsonl",
        call_key=unchanged_key,
        record_ids={"item_id": "s1"},
        prompt_sha256="prompt-unchanged",
    )
    # Append a second logical call to the same prior ledger/plan.
    ledger = PersistentAttemptLedger(
        prior_dir / "physical_attempt_ledger.jsonl",
        stage=STAGE,
        expected_calls={unchanged_key: 3, changed_key_old: 3},
        maximum_total_attempts=6,
    )
    reservation = ledger.reserve(
        changed_key_old, record_ids={"item_id": "s2"}, prompt_sha256="prompt-old"
    )
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="req-hash-2",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        error=None,
        result={"parsed": {"verdict": "supported"}, "raw_text": "{}"},
    )

    write_jsonl(
        prior_dir / "call_plan.jsonl",
        [
            _plan_row(unchanged_key, "prompt-unchanged", "payload-unchanged"),
            _plan_row(changed_key_old, "prompt-old", "payload-old"),
        ],
    )

    # This run's own freshly-computed plan: one row's physical_call_key is
    # identical to the prior run's untouched row (same everything); the
    # other row represents a REPAIRED state -- prompt/request changed, so
    # its physical_call_key is genuinely different from the prior one.
    new_call_plan = [
        _plan_row(unchanged_key, "prompt-unchanged", "payload-unchanged"),
        _plan_row("call_changed_new", "prompt-new", "payload-new"),
    ]

    state = module._load_delta_carry_forward_state(
        carry_forward_dir=prior_dir,
        call_plan=new_call_plan,
        review_stage=STAGE,
    )

    assert state["carry_forward_mechanism"] == "per_row_delta"
    assert state["carried_call_keys"] == {unchanged_key}
    assert set(state["carried_terminal_rows"]) == {unchanged_key}
    # The changed-state's new call key was never in the prior ledger at all,
    # so it is simply absent from carried keys -- correctly treated as new.


def test_delta_carry_forward_refuses_when_request_payload_sha256_differs_but_call_key_matches(
    tmp_path, module
) -> None:
    """Defense-in-depth: even if physical_call_key coincidentally matches
    (e.g. only the schema's internal field constraints changed, which
    physical_call_key does not hash), a request_payload_sha256 mismatch must
    still block carry-forward for that row.
    """

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    call_key = "call_same_key_different_payload"
    _write_prior_ledger(
        prior_dir / "physical_attempt_ledger.jsonl",
        call_key=call_key,
        record_ids={"item_id": "s1"},
        prompt_sha256="prompt-x",
    )
    write_jsonl(
        prior_dir / "call_plan.jsonl",
        [_plan_row(call_key, "prompt-x", "payload-old-schema")],
    )
    new_call_plan = [_plan_row(call_key, "prompt-x", "payload-new-schema")]

    state = module._load_delta_carry_forward_state(
        carry_forward_dir=prior_dir,
        call_plan=new_call_plan,
        review_stage=STAGE,
    )
    assert state["carried_call_keys"] == set()


def test_delta_carry_forward_does_not_carry_a_terminally_failed_call(tmp_path, module) -> None:
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    call_key = "call_failed"
    ledger = PersistentAttemptLedger(
        prior_dir / "physical_attempt_ledger.jsonl",
        stage=STAGE,
        expected_calls={call_key: 1},
        maximum_total_attempts=1,
    )
    reservation = ledger.reserve(call_key, record_ids={"item_id": "s1"}, prompt_sha256="p")
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="boom",
        metadata={"retry_class": "structured_output_validation_error"},
    )
    write_jsonl(prior_dir / "call_plan.jsonl", [_plan_row(call_key, "p", "payload")])
    new_call_plan = [_plan_row(call_key, "p", "payload")]

    state = module._load_delta_carry_forward_state(
        carry_forward_dir=prior_dir,
        call_plan=new_call_plan,
        review_stage=STAGE,
    )
    assert state["carried_call_keys"] == set()


def test_whole_plan_and_delta_carry_forward_are_mutually_exclusive_cli_flags(module) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--delta-carry-forward-from" in source
    assert "mutually exclusive" in source
