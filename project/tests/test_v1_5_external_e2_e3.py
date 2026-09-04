from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json, sha256_file
from metacom_pm.v1_5_external_memory_adapter import (
    qa_messages,
    render_evoemo_session_document,
)


ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3 = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"


def _rows(path: Path) -> list[dict]:
    return [dict(row) for row in iter_jsonl(path)]


def test_external_e4_freeze_has_exact_nonfallback_call_surface():
    plan_dir = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1"
    seal = read_json(plan_dir / "execution_seal.json")
    cost = read_json(plan_dir / "cost_estimate.json")
    calls = _rows(plan_dir / "physical_call_plan_private.jsonl")
    assert seal["status"] == "SEALED_READY_FOR_E4_PAID_RELEASE"
    assert cost["logical_calls"] == len(calls) == 4218
    assert cost["call_counts"] == {
        "response_core": 1576,
        "response_raw": 552,
        "qa": 2090,
    }
    assert len({row["call_id"] for row in calls}) == len(calls)
    assert len({row["physical_call_key"] for row in calls}) == len(calls)
    assert all(row["maximum_physical_attempts"] == 3 for row in calls)
    assert sha256_file(plan_dir / "physical_call_plan_private.jsonl") == seal[
        "physical_call_plan_sha256"
    ]


def test_external_e4_runner_uses_active_paid_guard_and_no_action_fallback():
    runner = (
        ROOT / "scripts/v1_5/28e_run_v5_2_external_e4_generation_v1_5.py"
    ).read_text(encoding="utf-8")
    assert "require_paid_run_release" in runner
    assert 'parser.add_argument("--run"' in runner
    assert '"fallback_used": False' in runner
    assert "realized_action_id" not in runner
    assert "compose_locked_response" in runner
    assert "locked_response_guard_errors" in runner


def test_external_e4_qa_messages_remain_gold_free():
    qa_path = (
        ROOT
        / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
        / "qa_final_call_plan_private.jsonl"
    )
    forbidden = {"answer", "answers", "evidence", "capability", "summaries"}
    for row in _rows(qa_path):
        assert row["answer_evidence_capability_visible_to_generation"] is False
        serialized = json.dumps(row["messages"], ensure_ascii=False)
        # Exact evaluator field names must not be serialized into generation.
        assert not any(f'"{key}"' in serialized for key in forbidden)


def test_external_e5_metrics_are_frozen_preoutcome_and_match_official_set_f1():
    script = ROOT / "scripts/v1_5/28f_score_v5_2_external_e5_v1_5.py"
    spec = importlib.util.spec_from_file_location("external_e5", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Official ES-MemEval counts unique overlap but keeps token-list
    # denominators, rather than SQuAD's multiset intersection.
    assert module.official_token_f1("a a b", "a b b") == 2 / 3
    # Reproduce the released implementation literally: after removing '#',
    # the leading whitespace prevents str.strip("Answer:") from consuming the
    # apparent prefix.  Do not silently replace this with removeprefix().
    assert module.official_prediction_surface("### Answer: hello **") == "Answer: hello"
    freeze = read_json(
        ROOT
        / "outputs/pm_v1_5_v5_2_external_e5_scoring_v1"
        / "metric_freeze.json"
    )
    assert freeze["status"] == "E5_METRICS_FROZEN_BEFORE_E4_OUTCOMES"
    assert freeze["e4_outcomes_read"] is False
    assert freeze["implementation_sha256"] == sha256_file(script)


def test_external_adapter_uses_dialogue_only_and_qa_has_no_gold() -> None:
    session = {
        "id": "s1",
        "timestamp": "2025-01-01",
        "summary": "must not appear",
        "observation": [{"content": "must not appear either"}],
        "dialogue": [
            {"role": "seeker", "content": "  I need help. "},
            {"role": "supporter", "content": " What happened? "},
        ],
    }
    document = render_evoemo_session_document(session, human_name="Ari")
    assert document == {
        "session_id": "s1",
        "date": "2025-01-01",
        "text": "Ari: I need help.\nSupporter: What happened?",
    }
    messages = qa_messages(question="What happened?", memory_fragments=[])
    blob = json.dumps(messages)
    assert "What happened?" in blob
    assert "must not appear" not in blob
    assert '"answer"' not in blob
    assert '"evidence"' not in blob


def test_e2_logical_plan_is_gold_separated_and_complete() -> None:
    report = read_json(E2 / "plan_manifest.json")
    assert report["status"] == "PASS_READY_FOR_E3_RETRIEVAL"
    assert all(report["checks"].values())
    assert report["response"]["core_calls_after_state_action_policy_alias_dedup"] == 1576
    assert report["response"]["secondary_raw_calls_after_e3"] == 552
    assert report["qa"]["logical_calls"] == 2090
    assert report["qa"]["retrieval_or_context_fit_pending_e3"] == 1672
    calls = _rows(E2 / "qa_logical_call_plan_private.jsonl")
    forbidden = {"answer", "answers", "evidence", "capability", "summaries"}
    assert not any(forbidden & set(row) for row in calls)
    assert len({row["call_id"] for row in calls}) == 2090


def test_e3_seals_exact_prompts_without_gold_and_preserves_owner() -> None:
    report = read_json(E3 / "retrieval_report.json")
    assert report["status"] == "PASS_READY_FOR_E4_PAID_GENERATION"
    assert all(report["checks"].values())
    calls = _rows(E3 / "qa_final_call_plan_private.jsonl")
    assert len(calls) == 2090
    assert all(row["messages"] and row["messages_sha256"] for row in calls)
    assert all(row["input_token_upper_bound"] <= 20000 for row in calls)
    assert not any("answer" in row or "evidence" in row or "capability" in row for row in calls)
    learned = [row for row in calls if row["condition"] == "typed_memory_learned_pm"]
    assert len(learned) == 418
    assert set(component for row in learned for component in row["selected_typed_components"]) <= {
        "MP",
        "MS",
        "ME",
    }
    assert report["qa"]["official_session_retrieval"]["comparable_questions"] == 341
