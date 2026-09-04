from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import CallResult, Endpoint, RetryableProviderError
from metacom_pm.bounded_retry import TERMINAL_DISPOSITION
from metacom_pm.config import load_config
import metacom_pm.evoemo as evoemo_module
from metacom_pm.evoemo import (
    build_fixed_seeker_tracks_v22,
    evaluator_context,
    load_evoemo,
    persist_fixed_seeker_tracks_v22_dry_run,
    plan_fixed_seeker_tracks_v22,
    seeker_system_prompt,
)
from metacom_pm.fixed_seeker_contract import (
    FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256,
    FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256_V3,
    FIXED_SEEKER_SURFACE_SELECTION_PROTOCOL,
    FixedSeekerGenerationContract,
)
from metacom_pm.io import iter_jsonl, read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVOEMO_PATH = PROJECT_ROOT / "data" / "external" / "evo_emo.json"


def _contract_mapping(**overrides):
    value = {
        "version": "pm-v2.2-fixed-seeker-generation-v1",
        "seeker_endpoint": "seeker",
        "system_prompt_id": "evoemo-fixed-seeker-v1",
        "system_prompt_template_sha256": (
            FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256
        ),
        "response_instruction_token_limit": 60,
        "temperature": 0.2,
        "max_output_tokens": 300,
        "output_normalization": "normalize_space_v1",
        "finish_reason_protocol": "pm-v2-finish-reason-v1",
        "accepted_normalized_finish_reasons": ["complete"],
        "maximum_physical_attempts_per_logical_call": 1,
        "seed_protocol": "base-seed-plus-turn-index-v1",
        "elicitation_scaffold_protocol": (
            "deterministic-generic-open-loop-v1"
        ),
    }
    value.update(overrides)
    return value


def _contract() -> FixedSeekerGenerationContract:
    return FixedSeekerGenerationContract.from_mapping(_contract_mapping())


def _v2_contract_mapping(**overrides):
    value = _contract_mapping(
        version="pm-v2.2-fixed-seeker-generation-v2",
        maximum_physical_attempts_per_logical_call=3,
    )
    value.update(overrides)
    return value


def _v2_contract(**overrides) -> FixedSeekerGenerationContract:
    return FixedSeekerGenerationContract.from_mapping(
        _v2_contract_mapping(**overrides)
    )


def _v3_contract_mapping(**overrides):
    value = {
        "version": "pm-v2.2-fixed-seeker-generation-v3-bounded-surface",
        "seeker_endpoint": "seeker",
        "system_prompt_id": "evoemo-fixed-seeker-bounded-surface-v1",
        "system_prompt_template_sha256": (
            FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256_V3
        ),
        "response_instruction_word_limit": 60,
        "surface_selection_protocol": FIXED_SEEKER_SURFACE_SELECTION_PROTOCOL,
        "temperature": 0.2,
        "max_output_tokens": 300,
        "output_normalization": "normalize_space_v1",
        "finish_reason_protocol": "pm-v2-finish-reason-v1",
        "accepted_normalized_finish_reasons": ["complete", "length"],
        "maximum_physical_attempts_per_logical_call": 3,
        "seed_protocol": "base-seed-plus-turn-index-v1",
        "elicitation_scaffold_protocol": (
            "deterministic-generic-open-loop-v1"
        ),
    }
    value.update(overrides)
    return value


def _v3_contract(**overrides) -> FixedSeekerGenerationContract:
    return FixedSeekerGenerationContract.from_mapping(
        _v3_contract_mapping(**overrides)
    )


def _cost_planning() -> dict:
    return {
        "protocol": "pm-v2.2-fixed-seeker-cost-planning-v1",
        "pricing_usd_per_mtok": {"input": 0.15, "output": 0.60},
        "input_token_safety_factor": 1.50,
        "fail_on_reported_input_overrun": True,
    }


def _budget_kwargs() -> dict:
    return {
        "max_api_calls": 10,
        "max_estimated_usd": 10.0,
        "max_input_tokens_per_call": 20_000,
    }


def _endpoint(*, model: str = "fixture-seeker") -> Endpoint:
    return Endpoint(
        base_url="https://fixture.invalid/v1",
        model=model,
        api_key_env="FIXTURE_KEY",
        family="fixture-family",
        timeout_seconds=17.0,
    )


class _FakeClient:
    def __init__(
        self,
        *,
        normalized_finish_reason: str = "complete",
        provider_finish_reason: str | None = "stop",
        prompt_tokens: int = 20,
        completion_tokens: int = 5,
    ) -> None:
        self.normalized_finish_reason = normalized_finish_reason
        self.provider_finish_reason = provider_finish_reason
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.calls: list[dict] = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        index = len(self.calls)
        return (
            CallResult(
                text=f"  seeker response {index}  ",
                raw_response={
                    "choices": [
                        {"finish_reason": self.provider_finish_reason}
                    ],
                    "usage": {
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                        "total_tokens": (
                            self.prompt_tokens + self.completion_tokens
                        ),
                    },
                },
                usage={
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
                latency_ms=1.0,
                request_hash=f"request-{index}",
                provider_finish_reason=self.provider_finish_reason,
                normalized_finish_reason=self.normalized_finish_reason,
            ),
            None,
        )

    def close(self):
        self.closed = True


def _plan(out_dir: Path, *, max_turns: int = 2):
    estimate, rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=max_turns,
        seeds=[101],
        max_scenarios=1,
    )
    persist_fixed_seeker_tracks_v22_dry_run(out_dir, estimate, rows)
    return estimate, rows


def test_fixed_seeker_contract_separates_instruction_from_api_cap() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    contract = FixedSeekerGenerationContract.from_mapping(
        config["fixed_seeker_generation_treatment"]
    )
    assert contract == _contract()
    assert contract.response_instruction_token_limit == 60
    assert contract.max_output_tokens == 300
    assert "at most 60 tokens" in contract.render_system_prompt({"topic": "x"})
    user = load_evoemo(EVOEMO_PATH)[0]
    topic = user["subsequent_topics"][0]
    assert contract.render_system_prompt(
        evaluator_context(user, topic)
    ) == seeker_system_prompt(user, topic)

    with pytest.raises(ValueError, match="max_output_tokens must equal 300"):
        FixedSeekerGenerationContract.from_mapping(
            _contract_mapping(max_output_tokens=60)
        )
    with pytest.raises(ValueError, match="accept only complete"):
        FixedSeekerGenerationContract.from_mapping(
            _contract_mapping(
                accepted_normalized_finish_reasons=["complete", "length"]
            )
        )


def test_v3_fixed_seeker_selects_a_complete_bounded_surface() -> None:
    contract = _v3_contract()
    assert "at most\n60 whitespace-delimited words" in contract.render_system_prompt(
        {"topic": "x"}
    )
    first = " ".join(["First"] + ["grounded"] * 38) + "."
    second = " ".join(["Second"] + ["detail"] * 30) + "."
    surface, error = contract.select_surface(
        f"  {first}   {second}  ",
        normalized_finish_reason="length",
        provider_finish_reason="length",
    )
    assert error is None
    assert surface is not None
    assert surface.text == first
    assert surface.selected_word_count == 39
    assert surface.original_word_count == 70
    assert surface.sentence_count == 1
    assert surface.prefix_selected is True
    assert surface.text.endswith(".")


def test_v3_fixed_seeker_refuses_mid_sentence_truncation() -> None:
    contract = _v3_contract()
    surface, error = contract.select_surface(
        " ".join(["unfinished"] * 61),
        normalized_finish_reason="length",
        provider_finish_reason="length",
    )
    assert surface is None
    assert "no complete sentence prefix" in str(error)


def test_v3_fixed_seeker_keeps_a_short_complete_response_byte_stable() -> None:
    contract = _v3_contract()
    surface, error = contract.select_surface(
        "  I feel uncertain, but talking about it helps.  ",
        normalized_finish_reason="complete",
        provider_finish_reason="stop",
    )
    assert error is None
    assert surface is not None
    assert surface.text == "I feel uncertain, but talking about it helps."
    assert surface.prefix_selected is False


def test_v3_dry_run_has_a_separate_stage_and_identity(tmp_path: Path) -> None:
    legacy, legacy_rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=_v2_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    bounded, bounded_rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=_v3_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    assert legacy["stage"] == "evoemo_fixed_seeker_tracks_v22"
    assert bounded["stage"] == "evoemo_fixed_seeker_tracks_v23_bounded_surface"
    assert legacy["dry_run_acceptance_sha256"] != bounded[
        "dry_run_acceptance_sha256"
    ]
    assert legacy_rows[0]["logical_call_key"] != bounded_rows[0][
        "logical_call_key"
    ]


def test_fixed_seeker_dry_run_is_client_free_and_has_exact_budget(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        evoemo_module,
        "make_client",
        lambda endpoint: pytest.fail("dry-run created an API client"),
    )
    estimate, rows = _plan(tmp_path)
    assert estimate["scenario_count"] == 1
    assert estimate["maximum_physical_api_attempts"] == 2
    assert estimate["maximum_output_tokens_per_call"] == 300
    assert estimate["maximum_total_output_tokens"] == 600
    assert estimate["budget_gate"]["status"] == "PASS"
    assert estimate["fixed_seeker_cost_planning"][
        "pricing_usd_per_mtok"
    ] == {"input": 0.15, "output": 0.60}
    assert estimate["fixed_seeker_cost_planning"][
        "input_token_safety_factor"
    ] == 1.5
    assert len(rows) == 2
    assert len({row["logical_call_key"] for row in rows}) == 2
    assert all(row["maximum_physical_attempts"] == 1 for row in rows)
    assert all(row["maximum_input_tokens"] > 0 for row in rows)
    assert rows[1]["history_completion_token_cap"] == 300
    assert rows[1]["maximum_input_tokens"] > rows[0]["maximum_input_tokens"]
    assert all(row["maximum_cost_usd"] > 0 for row in rows)
    assert all(
        row["fixed_seeker_generation_contract_sha256"]
        == estimate["fixed_seeker_generation_contract_sha256"]
        for row in rows
    )

    changed_endpoint_estimate, changed_rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(model="changed-model"),
        contract=_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=2,
        seeds=[101],
        max_scenarios=1,
    )
    assert changed_endpoint_estimate["dry_run_acceptance_sha256"] != estimate[
        "dry_run_acceptance_sha256"
    ]
    assert changed_rows[0]["logical_call_key"] != rows[0]["logical_call_key"]


def test_fixed_seeker_run_rejects_wrong_hash_before_client(
    monkeypatch, tmp_path: Path
) -> None:
    _plan(tmp_path, max_turns=1)
    monkeypatch.setattr(
        evoemo_module,
        "make_client",
        lambda endpoint: pytest.fail("invalid acceptance created a client"),
    )
    with pytest.raises(RuntimeError, match="must exactly equal"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            tmp_path,
            seeker_endpoint=_endpoint(),
            contract=_contract(),
            cost_planning=_cost_planning(),
            simulator_id="seeker_main",
            accepted_dry_run_sha256="wrong",
            **_budget_kwargs(),
            max_turns=1,
            seeds=[101],
            max_scenarios=1,
        )
    assert not (tmp_path / "physical_attempt_ledger.jsonl").exists()


def test_fixed_seeker_failed_budget_cannot_authorize_client(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        max_api_calls=1,
        max_estimated_usd=10.0,
        max_input_tokens_per_call=20_000,
        max_turns=2,
        seeds=[101],
        max_scenarios=1,
    )
    assert estimate["budget_gate"]["status"] == "FAIL"
    persist_fixed_seeker_tracks_v22_dry_run(tmp_path, estimate, rows)
    monkeypatch.setattr(
        evoemo_module,
        "make_client",
        lambda endpoint: pytest.fail("failed budget created an API client"),
    )
    with pytest.raises(RuntimeError, match="budget gate did not PASS"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            tmp_path,
            seeker_endpoint=_endpoint(),
            contract=_contract(),
            cost_planning=_cost_planning(),
            simulator_id="seeker_main",
            accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
            max_api_calls=1,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=20_000,
            max_turns=2,
            seeds=[101],
            max_scenarios=1,
        )


def test_fixed_seeker_cost_contract_rejects_zero_price(tmp_path: Path) -> None:
    bad = _cost_planning()
    bad["pricing_usd_per_mtok"] = {"input": 0.0, "output": 0.60}
    with pytest.raises(ValueError, match="positive 0.15/0.60"):
        plan_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            seeker_endpoint=_endpoint(),
            contract=_contract(),
            cost_planning=bad,
            simulator_id="seeker_main",
            **_budget_kwargs(),
            max_turns=1,
            seeds=[101],
            max_scenarios=1,
        )


def test_fixed_seeker_complete_calls_use_300_and_bind_all_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, rows = _plan(tmp_path)
    fake = _FakeClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    summary = build_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        tmp_path,
        seeker_endpoint=_endpoint(),
        contract=_contract(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
        **_budget_kwargs(),
        max_turns=2,
        seeds=[101],
        max_scenarios=1,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["completion_truncated_count"] == 0
    assert summary["normalized_finish_reason_counts"] == {
        "complete": 2,
        "length": 0,
        "tool_call": 0,
        "content_filter": 0,
        "unknown": 0,
    }
    assert summary["planned_budget_gate"]["status"] == "PASS"
    assert summary["observed_budget_gate"]["status"] == "PASS"
    assert summary["observed_budget_gate"]["reported_usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
    }
    assert fake.closed
    assert len(fake.calls) == 2
    assert all(call["max_tokens"] == 300 for call in fake.calls)
    assert all(call["temperature"] == 0.2 for call in fake.calls)
    assert all(call["retries"] == 1 for call in fake.calls)

    track = list(iter_jsonl(tmp_path / "fixed_seeker_tracks.jsonl"))[0]
    digest = estimate["fixed_seeker_generation_contract_sha256"]
    assert track["fixed_seeker_generation_contract_sha256"] == digest
    assert track["fixed_seeker_generation_contract"] == estimate[
        "fixed_seeker_generation_contract"
    ]
    assert [item["logical_call_key"] for item in track["turn_provenance"]] == [
        row["logical_call_key"] for row in rows
    ]
    assert all(
        item["normalized_finish_reason"] == "complete"
        for item in track["turn_provenance"]
    )

    raw_rows = list(iter_jsonl(tmp_path / "raw_seeker_calls.jsonl"))
    assert [row["provider_finish_reason"] for row in raw_rows] == ["stop", "stop"]
    assert [row["normalized_finish_reason"] for row in raw_rows] == [
        "complete",
        "complete",
    ]
    assert all(row["completion_truncated"] is False for row in raw_rows)
    manifest = read_json(tmp_path / "run_manifest.json")
    attestation = read_json(tmp_path / "artifact_attestation.json")
    for artifact in (summary, manifest, attestation["parameters"]):
        assert artifact["fixed_seeker_generation_contract_sha256"] == digest
        assert artifact["fixed_seeker_generation_contract"] == estimate[
            "fixed_seeker_generation_contract"
        ]
        assert artifact["fixed_seeker_cost_planning_sha256"] == estimate[
            "fixed_seeker_cost_planning_sha256"
        ]
    assert attestation["expected"]["completion_truncated_count"] == 0
    assert attestation["expected"]["planned_budget_gate_status"] == "PASS"
    assert attestation["expected"]["observed_budget_gate_status"] == "PASS"


def test_v3_accepts_only_the_complete_bounded_prefix_of_a_length_response(
    monkeypatch, tmp_path: Path
) -> None:
    contract = _v3_contract()
    estimate, rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    persist_fixed_seeker_tracks_v22_dry_run(tmp_path, estimate, rows)
    first = " ".join(["I"] + ["feel"] * 38) + "."
    second = " ".join(["More"] + ["detail"] * 30) + "."

    class _LengthClient(_FakeClient):
        def chat(self, messages, **kwargs):
            result, parsed = super().chat(messages, **kwargs)
            result.text = f"{first} {second}"
            result.provider_finish_reason = "length"
            result.normalized_finish_reason = "length"
            return result, parsed

    fake = _LengthClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    summary = build_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        tmp_path,
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["stage"] == "evoemo_fixed_seeker_tracks_v23_bounded_surface"
    assert summary["provider_length_finish_count"] == 1
    assert summary["surface_selection"] == {
        "protocol": FIXED_SEEKER_SURFACE_SELECTION_PROTOCOL,
        "metadata_complete": True,
        "prefix_selected_count": 1,
        "maximum_selected_word_count": 39,
        "mid_sentence_truncation_count": 0,
    }
    track = list(iter_jsonl(tmp_path / "fixed_seeker_tracks.jsonl"))[0]
    assert track["seeker_turns"] == [first]
    ledger = list(iter_jsonl(tmp_path / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger] == ["STARTED", "SUCCEEDED"]
    assert ledger[-1]["result"]["surface_selection"]["prefix_selected"] is True
    assert ledger[-1]["result"]["provider_output_sha256"]
    raw = list(iter_jsonl(tmp_path / "raw_seeker_calls.jsonl"))
    assert raw[0]["completion_truncated"] is True
    attestation = read_json(tmp_path / "artifact_attestation.json")
    assert attestation["expected"]["accepted_normalized_finish_reasons"] == [
        "complete",
        "length",
    ]
    assert attestation["expected"]["surface_selection"] == summary[
        "surface_selection"
    ]


def test_fixed_seeker_reported_input_overrun_is_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    estimate, _ = _plan(tmp_path, max_turns=1)
    fake = _FakeClient(prompt_tokens=999_999)
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    with pytest.raises(RuntimeError, match="generation incomplete"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            tmp_path,
            seeker_endpoint=_endpoint(),
            contract=_contract(),
            cost_planning=_cost_planning(),
            simulator_id="seeker_main",
            accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
            **_budget_kwargs(),
            max_turns=1,
            seeds=[101],
            max_scenarios=1,
        )
    ledger = list(iter_jsonl(tmp_path / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger] == ["STARTED", "FAILED"]
    assert "prompt_tokens exceed" in ledger[-1]["error"]
    assert read_json(tmp_path / "summary.json")["observed_budget_gate"][
        "status"
    ] == "FAIL"


@pytest.mark.parametrize(
    ("normalized", "provider"),
    [
        ("length", "length"),
        ("unknown", None),
        ("tool_call", "tool_calls"),
        ("content_filter", "content_filter"),
    ],
)
def test_fixed_seeker_noncomplete_is_terminal_before_track_write(
    monkeypatch,
    tmp_path: Path,
    normalized: str,
    provider: str | None,
) -> None:
    estimate, _ = _plan(tmp_path, max_turns=2)
    fake = _FakeClient(
        normalized_finish_reason=normalized,
        provider_finish_reason=provider,
    )
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    with pytest.raises(RuntimeError, match="generation incomplete"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            tmp_path,
            seeker_endpoint=_endpoint(),
            contract=_contract(),
            cost_planning=_cost_planning(),
            simulator_id="seeker_main",
            accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
            **_budget_kwargs(),
            max_turns=2,
            seeds=[101],
            max_scenarios=1,
        )
    assert len(fake.calls) == 1
    assert not (tmp_path / "fixed_seeker_tracks.jsonl").exists()
    ledger_rows = list(iter_jsonl(tmp_path / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "FAILED"]
    assert ledger_rows[-1]["result"]["normalized_finish_reason"] == normalized
    raw = list(iter_jsonl(tmp_path / "raw_seeker_calls.jsonl"))[0]
    assert raw["provider_finish_reason"] == provider
    assert raw["normalized_finish_reason"] == normalized
    assert raw["completion_truncated"] is (normalized == "length")
    assert raw["error"]
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "INCOMPLETE"
    assert summary["completed_tracks"] == 0


# --- v2 retry contract: real transport-retry tolerance, content gate still
# fails closed on its first attempt, and same-contract carry-forward. ---


class _FlakyThenSucceedsClient:
    """Fails the first ``fail_first_n`` calls with a transport-classified
    RetryableProviderError, then behaves like _FakeClient."""

    def __init__(
        self, *, fail_first_n: int, retry_class: str = "network_timeout"
    ) -> None:
        self.fail_first_n = int(fail_first_n)
        self.retry_class = retry_class
        self.calls: list[dict] = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        index = len(self.calls)
        if index <= self.fail_first_n:
            raise RetryableProviderError(
                f"transient failure {index}",
                last_retry_class=self.retry_class,
                last_status_code=503,
                attempts_tried=index,
            )
        return (
            CallResult(
                text=f"  seeker response {index}  ",
                raw_response={
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                    },
                },
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                latency_ms=1.0,
                request_hash=f"request-{index}",
                provider_finish_reason="stop",
                normalized_finish_reason="complete",
            ),
            None,
        )

    def close(self):
        self.closed = True


def test_fixed_seeker_v1_contract_still_forbids_multiple_attempts() -> None:
    with pytest.raises(ValueError, match="exactly one physical attempt"):
        FixedSeekerGenerationContract.from_mapping(
            _contract_mapping(maximum_physical_attempts_per_logical_call=3)
        )


def test_fixed_seeker_v2_contract_allows_a_real_retry_budget() -> None:
    contract = _v2_contract(maximum_physical_attempts_per_logical_call=6)
    assert contract.maximum_physical_attempts_per_logical_call == 6
    with pytest.raises(ValueError, match="positive"):
        FixedSeekerGenerationContract.from_mapping(
            _v2_contract_mapping(maximum_physical_attempts_per_logical_call=0)
        )


def test_fixed_seeker_v2_transport_failure_is_retried_and_recovers(
    monkeypatch, tmp_path: Path
) -> None:
    contract = _v2_contract()
    estimate, rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    persist_fixed_seeker_tracks_v22_dry_run(tmp_path, estimate, rows)
    fake = _FlakyThenSucceedsClient(fail_first_n=1)
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    summary = build_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        tmp_path,
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
        transport_backoff_seconds=(0.0,),
    )
    assert summary["status"] == "COMPLETE"
    assert len(fake.calls) == 2
    ledger_rows = list(iter_jsonl(tmp_path / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == [
        "STARTED",
        "FAILED",
        "STARTED",
        "SUCCEEDED",
    ]
    assert ledger_rows[1]["metadata"]["retry_class"] == "network_timeout"
    assert ledger_rows[1]["metadata"]["retry_disposition"] == "retryable_transient"


def test_fixed_seeker_v2_content_gate_is_never_retried(
    monkeypatch, tmp_path: Path
) -> None:
    contract = _v2_contract()
    estimate, rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    persist_fixed_seeker_tracks_v22_dry_run(tmp_path, estimate, rows)
    fake = _FakeClient(
        normalized_finish_reason="length", provider_finish_reason="length"
    )
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake)
    with pytest.raises(RuntimeError, match="generation incomplete"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            tmp_path,
            seeker_endpoint=_endpoint(),
            contract=contract,
            cost_planning=_cost_planning(),
            simulator_id="seeker_main",
            accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
            **_budget_kwargs(),
            max_turns=1,
            seeds=[101],
            max_scenarios=1,
            transport_backoff_seconds=(0.0,),
        )
    # A content-gate rejection must never consume the transport-retry
    # budget (3 attempts allowed), even though it succeeded at the HTTP
    # layer -- only one physical attempt is ever made.
    assert len(fake.calls) == 1
    ledger_rows = list(iter_jsonl(tmp_path / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "FAILED"]
    assert ledger_rows[-1]["metadata"]["retry_class"] == (
        "stage_postcondition_failure"
    )
    assert ledger_rows[-1]["metadata"]["retry_disposition"] == TERMINAL_DISPOSITION


def test_fixed_seeker_v2_cost_plan_scales_with_retry_budget() -> None:
    shared_kwargs = dict(
        seeker_endpoint=_endpoint(),
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        max_api_calls=100,
        max_estimated_usd=100.0,
        max_input_tokens_per_call=20_000,
        max_turns=2,
        seeds=[101],
        max_scenarios=1,
    )
    v1_estimate, _ = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH, contract=_contract(), **shared_kwargs
    )
    v2_estimate, v2_rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        contract=_v2_contract(maximum_physical_attempts_per_logical_call=3),
        **shared_kwargs,
    )
    assert v1_estimate["maximum_physical_attempts_per_logical_call"] == 1
    assert v2_estimate["maximum_physical_attempts_per_logical_call"] == 3
    assert (
        v2_estimate["maximum_physical_api_attempts"]
        == 3 * v1_estimate["maximum_physical_api_attempts"]
    )
    assert v2_estimate["maximum_total_output_tokens"] == (
        3 * v1_estimate["maximum_total_output_tokens"]
    )
    assert v2_estimate["maximum_estimated_usd"] == pytest.approx(
        3 * v1_estimate["maximum_estimated_usd"]
    )
    assert all(row["maximum_physical_attempts"] == 3 for row in v2_rows)
    # The per-call input-token bound (used for the per-attempt budget-gate
    # check) is not multiplied, only the aggregate totals are.
    assert (
        v2_estimate["maximum_input_tokens_per_call"]
        == v1_estimate["maximum_input_tokens_per_call"]
    )


def test_fixed_seeker_v2_carry_forward_recovers_prior_successes(
    monkeypatch, tmp_path: Path
) -> None:
    contract = _v2_contract()
    plan_kwargs = dict(
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=2,
        seeds=[101],
        max_scenarios=1,
    )
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    estimate, rows = plan_fixed_seeker_tracks_v22(EVOEMO_PATH, **plan_kwargs)
    persist_fixed_seeker_tracks_v22_dry_run(run1_dir, estimate, rows)

    class _SecondTurnRejectedClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.closed = False

        def chat(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            index = len(self.calls)
            normalized = "complete" if index == 1 else "length"
            provider = "stop" if index == 1 else "length"
            return (
                CallResult(
                    text=f"  seeker response {index}  ",
                    raw_response={
                        "choices": [{"finish_reason": provider}],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 5,
                            "total_tokens": 25,
                        },
                    },
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                    },
                    latency_ms=1.0,
                    request_hash=f"request-{index}",
                    provider_finish_reason=provider,
                    normalized_finish_reason=normalized,
                ),
                None,
            )

        def close(self):
            self.closed = True

    fake1 = _SecondTurnRejectedClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake1)
    with pytest.raises(RuntimeError, match="generation incomplete"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            run1_dir,
            accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
            transport_backoff_seconds=(0.0,),
            **plan_kwargs,
        )
    assert len(fake1.calls) == 2  # turn 1 succeeded, turn 2 content-rejected

    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    persist_fixed_seeker_tracks_v22_dry_run(run2_dir, estimate, rows)
    fake2 = _FakeClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake2)
    summary = build_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        run2_dir,
        accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
        carry_forward_tracks_dir=run1_dir,
        transport_backoff_seconds=(0.0,),
        **plan_kwargs,
    )
    assert summary["status"] == "COMPLETE"
    # Only turn 2 needed a real call; turn 1 was carried forward read-only.
    assert len(fake2.calls) == 1
    ledger_rows = list(iter_jsonl(run2_dir / "physical_attempt_ledger.jsonl"))
    carried_rows = [
        row
        for row in ledger_rows
        if (row.get("metadata") or {}).get("carried_forward")
    ]
    assert len(carried_rows) == 1
    assert carried_rows[0]["metadata"]["carried_forward_source_directory"] == (
        str(run1_dir)
    )
    track = list(iter_jsonl(run2_dir / "fixed_seeker_tracks.jsonl"))[0]
    assert track["seeker_turns"][0] == "seeker response 1"
    assert track["seeker_turns"][1] == "seeker response 1"


def test_fixed_seeker_v2_carry_forward_rejects_mismatched_plan(
    monkeypatch, tmp_path: Path
) -> None:
    contract = _v2_contract()
    plan_kwargs = dict(
        seeker_endpoint=_endpoint(),
        contract=contract,
        cost_planning=_cost_planning(),
        simulator_id="seeker_main",
        **_budget_kwargs(),
        max_turns=1,
        seeds=[101],
        max_scenarios=1,
    )
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    estimate, rows = plan_fixed_seeker_tracks_v22(EVOEMO_PATH, **plan_kwargs)
    persist_fixed_seeker_tracks_v22_dry_run(run1_dir, estimate, rows)
    fake1 = _FakeClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake1)
    build_fixed_seeker_tracks_v22(
        EVOEMO_PATH,
        run1_dir,
        accepted_dry_run_sha256=estimate["dry_run_acceptance_sha256"],
        transport_backoff_seconds=(0.0,),
        **plan_kwargs,
    )

    # A different seed changes every logical_call_key, so the freshly
    # computed plan can never byte-match run1's call_plan.jsonl.
    other_kwargs = {**plan_kwargs, "seeds": [202]}
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    other_estimate, other_rows = plan_fixed_seeker_tracks_v22(
        EVOEMO_PATH, **other_kwargs
    )
    persist_fixed_seeker_tracks_v22_dry_run(run2_dir, other_estimate, other_rows)
    fake2 = _FakeClient()
    monkeypatch.setattr(evoemo_module, "make_client", lambda endpoint: fake2)
    with pytest.raises(RuntimeError, match="differs from this run's own"):
        build_fixed_seeker_tracks_v22(
            EVOEMO_PATH,
            run2_dir,
            accepted_dry_run_sha256=other_estimate["dry_run_acceptance_sha256"],
            carry_forward_tracks_dir=run1_dir,
            transport_backoff_seconds=(0.0,),
            **other_kwargs,
        )


def test_fixed_seeker_reuses_bounded_retry_not_reimplemented() -> None:
    source = (
        PROJECT_ROOT / "src" / "metacom_pm" / "evoemo.py"
    ).read_text(encoding="utf-8")
    assert "from .bounded_retry import" in source
    assert "execute_with_bounded_retry" in source
    # The old hardcoded single-attempt block must be gone, not just disabled.
    assert "already spent its one" not in source
