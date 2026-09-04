from __future__ import annotations

from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.contracts import MemoryBackendRecord, MemoryItem, MemorySource
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import write_jsonl
from metacom_pm.pm_v1_6_preflight import build_preflight
from metacom_pm.pm_v2_data import evaluator_context_payload_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_inputs(tmp_path, tiny_state, tiny_strategy, *, mp_text: str):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    evaluator = tmp_path / "evaluator.jsonl"
    strategy = tmp_path / "strategy.jsonl"

    mp = MemoryItem(
        memory_id="mem_aaaaaaaaaaaaaaaa",
        source=MemorySource.MP,
        created_session=1,
        text=mp_text,
    )
    me = MemoryItem(
        memory_id="mem_bbbbbbbbbbbbbbbb",
        source=MemorySource.ME,
        created_session=2,
        text="Botany orchids cacti.",
    )
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id,
                items=[mp, me],
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategy, [tiny_strategy.model_dump(mode="json")])
    context = {
        "evaluator_context_id": "eval_0123456789abcdef",
        "state_id": tiny_state.state_id,
        "card_id": tiny_state.card_id,
        "regime": "profile_needed",
        "needed_memory_sources": ["MP"],
        "authorized_user_context": "The user feels unsettled and prefers gentle questions.",
        "coverage_rationale": "The stable preference is needed for personalized support.",
        "memory_annotations": [
            {
                "memory_id": mp.memory_id,
                "source": "MP",
                "created_session": 1,
                "stale": False,
                "conflicts_with_current_state": False,
                "private_sensitivity": "ordinary",
                "item_utility": "helpful",
            },
            {
                "memory_id": me.memory_id,
                "source": "ME",
                "created_session": 2,
                "stale": False,
                "conflicts_with_current_state": False,
                "private_sensitivity": "ordinary",
                "item_utility": "irrelevant",
            },
        ],
    }
    context["context_payload_sha256"] = evaluator_context_payload_sha256(context)
    write_jsonl(evaluator, [context])
    return runtime, backend, evaluator, strategy


def _supporter() -> SupporterGenerationContract:
    return SupporterGenerationContract.from_config(
        load_config(PROJECT_ROOT / "configs" / "pm_v1_6.yaml")
    )


def test_preflight_passes_required_hit_and_preserves_zero_hit_alias(
    tmp_path, tiny_state, tiny_strategy
) -> None:
    runtime, backend, evaluator, strategy = _write_inputs(
        tmp_path,
        tiny_state,
        tiny_strategy,
        mp_text="The user feels unsettled and prefers gentle questions.",
    )
    summary, rows = build_preflight(
        runtime_path=runtime,
        backend_path=backend,
        evaluator_contexts_path=evaluator,
        strategy_bank_path=strategy,
        supporter_contract=_supporter(),
        memory_min_score=0.0,
        strategy_min_score=0.0,
        strategy_top_k=1,
    )
    assert summary["status"] == "PASS"
    assert summary["required_hit_failures"] == 0
    assert summary["requested_realized_mismatch_rows"] > 0
    assert summary["prompt_alias_rows"] > 0

    by_action = {row["requested_action_id"]: row for row in rows}
    assert by_action["MP+R0"]["realized_action_id"] == "MP+R0"
    assert by_action["ME+R0"]["realized_action_id"] == "M0+R0"
    assert by_action["ME+R0"]["retrieval_attempts"][0]["call_count"] == 1
    assert by_action["ME+R0"]["retrieval_attempts"][0]["hit_count"] == 0
    assert by_action["ME+R0"]["shared_quality_label_weight"] < 1.0


def test_preflight_fails_required_source_before_outcomes(
    tmp_path, tiny_state, tiny_strategy
) -> None:
    runtime, backend, evaluator, strategy = _write_inputs(
        tmp_path,
        tiny_state,
        tiny_strategy,
        mp_text="Botany orchids cacti.",
    )
    summary, _ = build_preflight(
        runtime_path=runtime,
        backend_path=backend,
        evaluator_contexts_path=evaluator,
        strategy_bank_path=strategy,
        supporter_contract=_supporter(),
        memory_min_score=0.0,
        strategy_min_score=0.0,
        strategy_top_k=1,
    )
    assert summary["status"] == "FAIL"
    assert summary["required_hit_failures"] == 1
    state_gate = summary["required_hit_states"][0]
    assert state_gate["checks"]["required_source_MP_has_hit"] is False
