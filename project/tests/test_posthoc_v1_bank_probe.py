from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest

from metacom_pm.api import CallResult, Endpoint
from metacom_pm.contracts import StrategyCard
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_jsonl,
)
import metacom_pm.posthoc_v1_bank_probe as probe_module
from metacom_pm.posthoc_v1_bank_probe import (
    CONDITIONAL_ESTIMAND,
    EXCLUDED_ESTIMANDS,
    NONCANONICAL_DEBUG_LABEL,
    persist_posthoc_v1_bank_dry_run,
    plan_posthoc_v1_bank_probe,
    run_posthoc_v1_bank_probe,
)
from metacom_pm.retrieval_v1_canonical import StrategyRetrieverV1Canonical, context_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def test_private_diagnostic_output_directory_is_gitignored() -> None:
    candidate = (
        "project/outputs/"
        "posthoc_v1_bank_mechanism_diagnostic_fixture/private.jsonl"
    )
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", candidate],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_formal_cli_exposes_no_noncanonical_sha_or_condition_override() -> None:
    script = PROJECT_ROOT / "scripts" / "34_posthoc_v1_bank_mechanism_diagnostic.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--expected-turns-sha256" not in completed.stdout
    assert "--expected-legacy-bank-sha256" not in completed.stdout
    assert "--expected-full-bank-sha256" not in completed.stdout
    assert "--source-condition" not in completed.stdout
    assert "common replay generator treatment" in completed.stdout


def _endpoint() -> Endpoint:
    return Endpoint(
        base_url="https://fixture.invalid/v1",
        model="fixture-generator",
        api_key_env="FIXTURE_API_KEY",
        family="fixture-family",
        timeout_seconds=9.0,
    )


def _card(index: int, retrieval_text: str) -> StrategyCard:
    return StrategyCard(
        strategy_id=f"strat_{index:012x}",
        strategy_label=f"strategy-{index}",
        retrieval_text=retrieval_text,
        guidance_text=f"Use strategy {index} carefully.",
        example_response=f"Example response {index}.",
        source_dialogue_id=f"dialogue-{index}",
        source_turn_index=index,
    )


def _fixture_inputs(tmp_path: Path, *, users: int = 2, turns_per_user: int = 2):
    legacy_cards = [
        _card(1, "legacy stress support"),
        _card(2, "legacy work empathy"),
        _card(3, "legacy anxiety reflection"),
        _card(4, "unrelated legacy"),
    ]
    full_cards = [
        _card(11, "work stress anxiety overwhelmed"),
        _card(12, "stress empathy work"),
        _card(13, "anxiety support stress"),
        _card(14, "unrelated weather"),
    ]
    legacy_path = tmp_path / "legacy.jsonl"
    full_path = tmp_path / "full.jsonl"
    write_jsonl(legacy_path, [card.model_dump(mode="json") for card in legacy_cards])
    write_jsonl(full_path, [card.model_dump(mode="json") for card in full_cards])

    full_retriever = StrategyRetrieverV1Canonical(full_cards)
    rows = []
    for user_index in range(1, users + 1):
        for turn_index in range(1, turns_per_user + 1):
            context = [
                {"role": "supporter", "content": "I am here with you."},
                {
                    "role": "seeker",
                    "content": f"Earlier context for user {user_index}.",
                },
            ]
            seeker = f"I feel overwhelmed by work stress and anxiety {turn_index}."
            query = context_query(seeker, context, "")
            selected = full_retriever.retrieve(query)
            rows.append(
                {
                    "action_id": "M0+RS",
                    "card_id": f"card_{user_index:012x}",
                    "condition": "pm",
                    "context_before_turn": context,
                    "context_sha256": sha256_text(canonical_json(context)),
                    "cost": {},
                    "exogenous_state_id": (
                        f"state_{(1000 + user_index * 10 + turn_index):012x}"
                    ),
                    "input_tokens": 1,
                    "interaction_mode": "fixed",
                    "latency_ms": 1.0,
                    "output_tokens": 1,
                    "pm_decision_report": {},
                    "pm_ood_report": {},
                    "protocol": "selective",
                    "seed": 101,
                    "seeker_message": seeker,
                    "selected_memory": [],
                    "selected_strategy": [
                        card.model_dump(mode="json") for card in selected
                    ],
                    "simulator_id": "seeker_main",
                    "state_id": f"state_{(2000 + user_index * 10 + turn_index):012x}",
                    "supporter_message": "old response",
                    "topic_index": 1,
                    "track_id": f"track_{user_index:012x}",
                    "trajectory_comparability": {},
                    "turn_index": turn_index,
                    "user_id": f"p{user_index}",
                }
            )
    turns_path = tmp_path / "turns.jsonl"
    write_jsonl(turns_path, rows)
    hashes = {
        "turns": sha256_file(turns_path),
        "legacy": sha256_file(legacy_path),
        "full": sha256_file(full_path),
    }
    return turns_path, legacy_path, full_path, hashes


def _plan(tmp_path: Path, *, sample_size: int = 3):
    turns, legacy, full, hashes = _fixture_inputs(tmp_path)
    estimate, samples, plan = plan_posthoc_v1_bank_probe(
        turns,
        legacy,
        full,
        endpoint=_endpoint(),
        input_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=0.6,
        input_token_safety_factor=1.25,
        sample_size=sample_size,
        sample_seed=17,
        expected_turns_sha256=hashes["turns"],
        expected_legacy_bank_sha256=hashes["legacy"],
        expected_full_bank_sha256=hashes["full"],
    )
    return estimate, samples, plan


def test_dry_run_is_deterministic_balanced_and_changes_only_strategy_section(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        probe_module,
        "make_client",
        lambda endpoint: pytest.fail("dry-run constructed an API client"),
    )
    estimate, samples, plan = _plan(tmp_path)
    assert estimate["diagnostic_label"] == NONCANONICAL_DEBUG_LABEL
    assert estimate["canonical_frozen_v1_inputs"] is False
    assert estimate["input_classification"] == "NONCANONICAL_DEBUG_ONLY"
    assert estimate["conditional_estimand"] == CONDITIONAL_ESTIMAND
    assert estimate["excluded_estimands"] == list(EXCLUDED_ESTIMANDS)
    assert "frozen V1 PM+RS action" in estimate["conditional_estimand"]
    assert any("routing" in value for value in estimate["excluded_estimands"])
    assert any("Total effect" in value for value in estimate["excluded_estimands"])
    assert estimate["confirmatory"] is False
    assert estimate["v2_training_gate"] is False
    assert estimate["sample_size"] == 3
    assert estimate["planned_logical_calls"] == 6
    assert estimate["maximum_physical_api_attempts"] == 6
    assert estimate["expected_call_formula"] == "2 * sample_size"
    assert estimate["maximum_estimated_usd"] > 0
    assert "replay_generator_treatment" in estimate
    assert "generator_endpoint" not in estimate
    assert "not claimed" in estimate["replay_generator_treatment_note"]
    assert sorted(estimate["selected_per_user"].values()) == [1, 2]
    assert all(sample["paired_invariance"]["only_strategy_evidence_section_differs"] for sample in samples)

    by_pair: dict[str, list[dict]] = {}
    for row in plan:
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert len(by_pair) == 3
    for rows in by_pair.values():
        assert {row["bank_condition"] for row in rows} == {
            "legacy_156",
            "full_12429",
        }
        assert len({row["non_strategy_messages_sha256"] for row in rows}) == 1
        assert len({row["selected_memory_sha256"] for row in rows}) == 1
        assert len({row["generator_seed"] for row in rows}) == 1
        assert {row["within_pair_call_order"] for row in rows} == {1, 2}

    out = tmp_path / "out"
    assert persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan) == "WRITTEN"
    assert persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan) == "VALIDATED_EXISTING"
    assert len(list(iter_jsonl(out / "private_blind_review_plan.jsonl"))) == 6
    review_schema = read_json(out / "review_annotation_schema.json")
    assert review_schema["diagnostic_label"] == NONCANONICAL_DEBUG_LABEL
    assert review_schema["canonical_frozen_v1_inputs"] is False
    assert review_schema["conditional_estimand"] == CONDITIONAL_ESTIMAND
    assert review_schema["excluded_estimands"] == list(EXCLUDED_ESTIMANDS)
    assert len((out / "reviewer_a_template.csv").read_text().splitlines()) == 4
    assert len((out / "reviewer_b_template.csv").read_text().splitlines()) == 4


class _FakeClient:
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        self.calls = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        index = len(self.calls)
        reason = self.reasons.pop(0)
        provider_reason = "stop" if reason == "complete" else "length"
        return (
            CallResult(
                text=f" response {index} ",
                raw_response={
                    "choices": [
                        {
                            "message": {"content": f" response {index} "},
                            "finish_reason": provider_reason,
                        }
                    ]
                },
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                latency_ms=1.0,
                request_hash=f"request-{index}",
                provider_finish_reason=provider_reason,
                normalized_finish_reason=reason,
            ),
            None,
        )

    def close(self):
        self.closed = True


def test_truncated_call_is_raw_logged_and_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=1)
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)
    fake = _FakeClient(["length"])
    monkeypatch.setattr(probe_module, "make_client", lambda endpoint: fake)
    with pytest.raises(RuntimeError, match="complete-only finish gate"):
        run_posthoc_v1_bank_probe(
            out,
            estimate,
            samples,
            plan,
            endpoint=_endpoint(),
            accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            max_api_calls=2,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=100_000,
        )
    assert fake.closed
    assert len(fake.calls) == 1
    assert fake.calls[0]["max_tokens"] == 300
    assert fake.calls[0]["retries"] == 1
    assert [row["event"] for row in iter_jsonl(out / "physical_attempt_ledger.jsonl")] == [
        "STARTED",
        "FAILED",
    ]
    raw = list(iter_jsonl(out / "raw_api_calls.jsonl"))
    assert raw[0]["normalized_finish_reason"] == "length"
    assert raw[0]["completion_truncated"] is True
    assert not (out / "paired_generations.jsonl").exists()


def test_reported_prompt_token_overrun_is_raw_logged_and_fails_closed(
    tmp_path: Path,
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=1)
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)
    prompt_tokens = int(plan[0]["conservative_input_tokens"]) + 1

    class PromptOverrunClient(_FakeClient):
        def chat(self, messages, **kwargs):
            call, parsed = super().chat(messages, **kwargs)
            call.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 5,
                "total_tokens": prompt_tokens + 5,
            }
            call.raw_response["usage"] = dict(call.usage)
            return call, parsed

    fake = PromptOverrunClient(["complete"])
    with pytest.raises(RuntimeError, match="reported prompt_tokens exceed"):
        run_posthoc_v1_bank_probe(
            out,
            estimate,
            samples,
            plan,
            endpoint=_endpoint(),
            accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            max_api_calls=2,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=100_000,
            client_factory=lambda endpoint: fake,
        )
    assert fake.closed
    assert [row["event"] for row in iter_jsonl(out / "physical_attempt_ledger.jsonl")] == [
        "STARTED",
        "FAILED",
    ]
    terminal = list(iter_jsonl(out / "physical_attempt_ledger.jsonl"))[-1]
    assert terminal["usage"]["prompt_tokens"] == prompt_tokens
    assert terminal["result"]["maximum_prompt_tokens"] == plan[0][
        "conservative_input_tokens"
    ]
    raw = list(iter_jsonl(out / "raw_api_calls.jsonl"))
    assert "reported prompt_tokens exceed" in raw[0]["error"]
    assert raw[0]["raw_response"]["usage"]["prompt_tokens"] == prompt_tokens
    assert not (out / "paired_generations.jsonl").exists()


def test_run_rejects_unaccepted_cost_hash_before_client(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=1)
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)
    monkeypatch.setattr(
        probe_module,
        "make_client",
        lambda endpoint: pytest.fail("unaccepted run constructed an API client"),
    )
    with pytest.raises(RuntimeError, match="must exactly match"):
        run_posthoc_v1_bank_probe(
            out,
            estimate,
            samples,
            plan,
            endpoint=_endpoint(),
            accepted_cost_estimate_sha256="0" * 64,
            max_api_calls=2,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=100_000,
        )
    assert not (out / "physical_attempt_ledger.jsonl").exists()


def test_client_initialization_failure_does_not_reserve_physical_attempt(
    tmp_path: Path,
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=1)
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)

    def fail_before_http(endpoint):
        raise RuntimeError("fixture credential initialization failure")

    with pytest.raises(RuntimeError, match="credential initialization failure"):
        run_posthoc_v1_bank_probe(
            out,
            estimate,
            samples,
            plan,
            endpoint=_endpoint(),
            accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            max_api_calls=2,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=100_000,
            client_factory=fail_before_http,
        )
    assert not (out / "physical_attempt_ledger.jsonl").exists()


def test_complete_run_builds_reverse_order_blind_review_package(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=2)
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)
    fake = _FakeClient(["complete"] * 4)
    monkeypatch.setattr(probe_module, "make_client", lambda endpoint: fake)
    summary = run_posthoc_v1_bank_probe(
        out,
        estimate,
        samples,
        plan,
        endpoint=_endpoint(),
        accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
        max_api_calls=4,
        max_estimated_usd=10.0,
        max_input_tokens_per_call=100_000,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["successful_calls"] == 4
    assert summary["completion_truncated_count"] == 0
    assert summary["observed_budget_gate"]["status"] == "PASS"
    assert summary["observed_budget_gate"]["within_accepted_cost_estimate"] is True
    assert summary["observed_budget_gate"]["within_cli_max_estimated_usd"] is True
    assert summary["observed_usd"] <= estimate["maximum_estimated_usd"]
    assert all(call["retries"] == 1 for call in fake.calls)
    mappings = list(iter_jsonl(out / "private_blind_review_plan.jsonl"))
    by_pair: dict[str, list[dict]] = {}
    for row in mappings:
        by_pair.setdefault(row["pair_id"], []).append(row)
    for rows in by_pair.values():
        a = next(row for row in rows if row["reviewer_id"] == "reviewer_a")
        b = next(row for row in rows if row["reviewer_id"] == "reviewer_b")
        assert a["response_A_condition"] == b["response_B_condition"]
        assert a["response_B_condition"] == b["response_A_condition"]
    public_items = list(iter_jsonl(out / "dual_order_review_items.jsonl"))
    assert len(public_items) == 4
    assert all(
        "legacy_156" not in canonical_json(item)
        and "full_12429" not in canonical_json(item)
        and "response_A_condition" not in item
        and "response_B_condition" not in item
        for item in public_items
    )
    assert len((out / "reviewer_a.csv").read_text().splitlines()) == 3
    assert len((out / "reviewer_b.csv").read_text().splitlines()) == 3


def test_final_observed_cost_must_fit_accepted_and_cli_budgets(
    tmp_path: Path,
) -> None:
    estimate, samples, plan = _plan(tmp_path, sample_size=1)
    estimate = dict(estimate)
    estimate["maximum_estimated_usd"] = 1e-9
    estimate_payload = {
        key: value
        for key, value in estimate.items()
        if key != "cost_estimate_sha256"
    }
    estimate["cost_estimate_sha256"] = sha256_text(
        canonical_json(estimate_payload)
    )
    out = tmp_path / "out"
    persist_posthoc_v1_bank_dry_run(out, estimate, samples, plan)
    fake = _FakeClient(["complete", "complete"])
    with pytest.raises(RuntimeError, match="exceeded an accepted budget ceiling"):
        run_posthoc_v1_bank_probe(
            out,
            estimate,
            samples,
            plan,
            endpoint=_endpoint(),
            accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            max_api_calls=2,
            max_estimated_usd=5e-9,
            max_input_tokens_per_call=100_000,
            client_factory=lambda endpoint: fake,
        )
    gate = read_json(out / "observed_budget_gate_failure.json")
    assert gate["status"] == "FAIL"
    assert gate["within_accepted_cost_estimate"] is False
    assert gate["within_cli_max_estimated_usd"] is False
    assert not (out / "paired_generations.jsonl").exists()
