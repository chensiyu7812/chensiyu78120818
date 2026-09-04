from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import Endpoint
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import load_config
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_judge_schema_smoke import (
    _select_smoke_state,
    judge_schema_smoke_settings,
    pending_judge_schema_smoke_calls,
    persist_or_validate_schema_smoke_dry_run,
    preflight_judge_schema_smoke_clients,
    require_exact_pilot_deployable_feature_observability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_schema_smoke_is_frozen_to_four_calls() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    settings = judge_schema_smoke_settings(config)
    assert settings["required_before_pilot_action_api"] is True
    assert settings["expected_physical_calls"] == 4
    assert settings["candidate_response"]
    assert settings["selected_context"] == "[none]"


def test_schema_smoke_state_selection_is_deterministic_and_train_only() -> None:
    class State:
        def __init__(self, state_id: str):
            self.state_id = state_id
            self.card_id = "card_" + state_id
            self.split = PMV2Split.TRAIN

    states = {name: State(name) for name in ("a", "b", "c")}
    plan = {
        "selected_states": [
            {
                "state_id": name,
                "card_id": state.card_id,
                "split": "train",
                "regime": "context_only",
            }
            for name, state in states.items()
        ]
    }
    first, first_row = _select_smoke_state(plan, states, seed=8053)
    second, second_row = _select_smoke_state(plan, states, seed=8053)
    assert first.state_id == second.state_id
    assert first_row == second_row
    assert first.split is PMV2Split.TRAIN
    plan["selected_states"][0]["split"] = "calibration"
    with pytest.raises(RuntimeError, match="entirely train-only"):
        _select_smoke_state(plan, states, seed=8053)


def test_all_endpoint_keys_are_resolved_before_any_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = Endpoint("https://first.invalid", "first", "SMOKE_FIRST", family="a")
    second = Endpoint("https://second.invalid", "second", "SMOKE_SECOND", family="b")
    execution = {
        "a-response": {"endpoint": first},
        "a-risk": {"endpoint": first},
        "b-response": {"endpoint": second},
        "b-risk": {"endpoint": second},
    }
    monkeypatch.setenv("SMOKE_FIRST", "present")
    monkeypatch.delenv("SMOKE_SECOND", raising=False)
    factory_calls = []

    def factory(endpoint):
        factory_calls.append(endpoint.family)
        return object()

    with pytest.raises(RuntimeError, match="SMOKE_SECOND"):
        preflight_judge_schema_smoke_clients(
            execution, client_factory=factory
        )
    assert factory_calls == []


def test_both_clients_are_built_only_after_both_keys_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = Endpoint("https://first.invalid", "first", "SMOKE_FIRST", family="a")
    second = Endpoint("https://second.invalid", "second", "SMOKE_SECOND", family="b")
    execution = {
        "a-response": {"endpoint": first},
        "a-risk": {"endpoint": first},
        "b-response": {"endpoint": second},
        "b-risk": {"endpoint": second},
    }
    monkeypatch.setenv("SMOKE_FIRST", "present")
    monkeypatch.setenv("SMOKE_SECOND", "present")

    class Client:
        def close(self):
            pass

    calls = []

    def factory(endpoint):
        calls.append(endpoint.family)
        return Client()

    clients = preflight_judge_schema_smoke_clients(
        execution, client_factory=factory
    )
    assert calls == ["a", "b"]
    assert set(clients) == {"a", "b"}


def test_spent_third_call_failure_blocks_unreached_fourth_call(
    tmp_path: Path,
) -> None:
    call_plan = [
        {"physical_call_key": f"call-{index}"} for index in range(1, 5)
    ]
    ledger = PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage="pm_v2_development_judge_schema_smoke",
        expected_calls={f"call-{index}": 1 for index in range(1, 5)},
        maximum_total_attempts=4,
    )
    for index in (1, 2):
        reservation = ledger.reserve(
            f"call-{index}", record_ids={"index": index}, prompt_sha256="p"
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=f"request-{index}",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            error=None,
            result={"parsed": {}},
        )
    failed = ledger.reserve(
        "call-3", record_ids={"index": 3}, prompt_sha256="p"
    )
    ledger.finish(
        failed,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected",
    )
    pending, reason = pending_judge_schema_smoke_calls(ledger, call_plan)
    assert pending == []
    assert "call-3" in str(reason)
    assert ledger.attempts_for("call-4") == 0


def test_nonempty_ledger_makes_dry_run_validate_only(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    estimate_path = tmp_path / "estimate.json"
    plan_path = tmp_path / "plan.jsonl"
    ledger_path.write_text('{"spent":true}\n', encoding="utf-8")
    estimate = {"cost_estimate_sha256": "a" * 64}
    gate = {"status": "PASS", "checks": {"exact": True}}
    plan = [{"physical_call_key": "call-1"}]
    from metacom_pm.io import write_json, write_jsonl, sha256_file

    write_json(estimate_path, {**estimate, "budget_gate": gate})
    write_jsonl(plan_path, plan)
    before = (sha256_file(estimate_path), sha256_file(plan_path))
    status = persist_or_validate_schema_smoke_dry_run(
        ledger_path=ledger_path,
        estimate_path=estimate_path,
        call_plan_path=plan_path,
        cost_estimate=estimate,
        budget_gate=gate,
        call_plan=plan,
    )
    assert status == "VALIDATED_EXISTING"
    assert before == (sha256_file(estimate_path), sha256_file(plan_path))
    with pytest.raises(RuntimeError, match="refusing to rewrite"):
        persist_or_validate_schema_smoke_dry_run(
            ledger_path=ledger_path,
            estimate_path=estimate_path,
            call_plan_path=plan_path,
            cost_estimate={"cost_estimate_sha256": "b" * 64},
            budget_gate=gate,
            call_plan=plan,
        )
    assert before == (sha256_file(estimate_path), sha256_file(plan_path))


def test_schema_smoke_recomputes_observability_and_rejects_forged_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import metacom_pm.pm_v2_judge_schema_smoke as module

    recomputed = {
        "status": "PASS",
        "checks": {"regime_macro_f1": True},
        "metrics": {"regime_macro_f1": 0.42},
    }
    monkeypatch.setattr(
        module,
        "audit_deployable_feature_observability",
        lambda states, evaluator_contexts, settings: recomputed,
    )
    plan = {"deployable_feature_observability": dict(recomputed)}
    assert require_exact_pilot_deployable_feature_observability(
        plan=plan,
        states=[],
        evaluator_contexts=object(),
        settings={"frozen": True},
    ) == recomputed
    plan["deployable_feature_observability"] = {
        **recomputed,
        "metrics": {"regime_macro_f1": 0.99},
    }
    with pytest.raises(RuntimeError, match="forged"):
        require_exact_pilot_deployable_feature_observability(
            plan=plan,
            states=[],
            evaluator_contexts=object(),
            settings={"frozen": True},
        )
