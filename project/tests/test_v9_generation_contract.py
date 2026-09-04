from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import CallResult
from metacom_pm.config import load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import iter_jsonl, read_json
from metacom_pm.pm_v2_generation_review_v9 import (
    prepare_v9,
    run_v9_paired_generation,
    validate_v9_dry_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V9_SOURCE_ARCHIVE_AVAILABLE = all(
    path.is_file()
    for path in (
        PROJECT_ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v8"
        / "generation_pilot_semantic_review_cases.json",
        PROJECT_ROOT
        / "outputs"
        / "strategy_rag_v1_frozen_candidate"
        / "strategy_rag_manifest.json",
    )
)


@pytest.mark.skipif(
    not V9_SOURCE_ARCHIVE_AVAILABLE,
    reason="historical V8/V9 artifact-vault inputs are absent from public checkout",
)
def test_future_v9_is_treatment_bound_and_rejects_truncation(
    tmp_path, monkeypatch
):
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    contract = SupporterGenerationContract.from_config(
        load_config(pm_v2_config_path)
    )
    out_dir = tmp_path / "v9_treatment_bound"
    prepare_v9(
        v8_cases_path=(
            PROJECT_ROOT
            / "outputs"
            / "pm_v2_generation_pilot_semantic_review_v8"
            / "generation_pilot_semantic_review_cases.json"
        ),
        current_bank_path=(
            PROJECT_ROOT / "data" / "strategy" / "strategy_cards.jsonl"
        ),
        experiment_config_path=PROJECT_ROOT / "configs" / "experiment.yaml",
        pm_v2_config_path=pm_v2_config_path,
        frozen_manifest_path=(
            PROJECT_ROOT
            / "outputs"
            / "strategy_rag_v1_frozen_candidate"
            / "strategy_rag_manifest.json"
        ),
        out_dir=out_dir,
    )
    plan = list(iter_jsonl(out_dir / "paired_generation_call_plan.jsonl"))
    assert len(plan) == 18
    assert all(
        row["supporter_generation_treatment"] == contract.payload()
        and row["supporter_generation_treatment_sha256"] == contract.digest()
        and row["messages"][0]
        == {"role": "system", "content": contract.system_prompt}
        and row["max_tokens"] == contract.max_output_tokens
        for row in plan
    )
    estimate = read_json(out_dir / "paired_generation_cost_estimate.json")
    assert estimate["protocol"] == (
        "pm-v2-v9-paired-generation-cost-v2-treatment-bound"
    )
    assert estimate["supporter_generation_treatment_sha256"] == contract.digest()
    validate_v9_dry_run(
        out_dir=out_dir,
        max_api_calls=18,
        max_estimated_usd=10.0,
        max_input_tokens_per_call=100_000,
    )

    class TruncatedClient:
        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            return (
                CallResult(
                    text="A cut-off response",
                    raw_response={
                        "choices": [
                            {
                                "message": {"content": "A cut-off response"},
                                "finish_reason": "length",
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": contract.max_output_tokens,
                        "total_tokens": 10 + contract.max_output_tokens,
                    },
                    latency_ms=1.0,
                    request_hash="v9-truncated",
                    provider_finish_reason="length",
                    normalized_finish_reason="length",
                ),
                None,
            )

    import metacom_pm.pm_v2_generation_review_v9 as v9_module

    monkeypatch.setattr(v9_module, "OpenAICompatibleClient", TruncatedClient)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only")
    with pytest.raises(RuntimeError, match="stopped fail-fast"):
        run_v9_paired_generation(
            out_dir=out_dir,
            experiment_config_path=PROJECT_ROOT / "configs" / "experiment.yaml",
            accept_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            max_api_calls=18,
            max_estimated_usd=10.0,
            max_input_tokens_per_call=100_000,
        )
    assert not (out_dir / "paired_generations.jsonl").exists()
    ledger = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger] == ["STARTED", "FAILED"]
    raw = list(iter_jsonl(out_dir / "raw_api_calls.jsonl"))
    assert raw[0]["normalized_finish_reason"] == "length"
    assert raw[0]["completion_truncated"] is True
    assert raw[0]["raw_response"]["choices"][0]["finish_reason"] == "length"
