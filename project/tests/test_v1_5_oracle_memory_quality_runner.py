from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import iter_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/v1_5/"
    "21p_run_longitudinal_oracle_memory_quality_v1_5.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "v1_5_oracle_memory_quality_runner_test", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(module):
    prepared = (
        ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_quality_packet_v1"
    )
    tracked = (
        ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_quality_diagnostic_v1.json"
    )
    items, prompts, contract = module._validate_packet(
        prepared_dir=prepared,
        tracked_contract_path=tracked,
    )
    experiment = load_config(ROOT / "configs/experiment.yaml")
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    names, endpoints, prices = module._endpoint_records(
        experiment=experiment,
        pm_config=pm_config,
        judge_endpoint_names=None,
    )
    return prepared, tracked, items, prompts, contract, pm_config, names, endpoints, prices


def test_quality_runner_builds_balanced_two_family_plan() -> None:
    module = _load_runner()
    (
        _,
        _,
        items,
        prompts,
        _,
        pm_config,
        names,
        endpoints,
        prices,
    ) = _inputs(module)
    plan = module.build_execution_plan(
        items=items,
        prompt_rows=prompts,
        endpoint_names=names,
        endpoints=endpoints,
        prices=prices,
        input_token_safety_factor=float(
            pm_config["api_cost_planning"]["input_token_safety_factor"]
        ),
    )
    assert len(plan) == 72
    assert len({row["physical_call_key"] for row in plan}) == 72
    assert {row["judge_family"] for row in plan} == {
        "google_gemini",
        "deepseek_official",
    }
    state_ids = {row["state_id"] for row in plan}
    assert len(state_ids) == 18
    for state_id in state_ids:
        for family in {"google_gemini", "deepseek_official"}:
            rows = [
                row
                for row in plan
                if row["state_id"] == state_id
                and row["judge_family"] == family
            ]
            assert {row["order_variant"] for row in rows} == {0, 1}
            assert len({row["seed"] for row in rows}) == 1


def test_quality_runner_cost_identity_counts_all_physical_attempts() -> None:
    module = _load_runner()
    (
        prepared,
        tracked,
        items,
        prompts,
        contract,
        pm_config,
        names,
        endpoints,
        prices,
    ) = _inputs(module)
    plan = module.build_execution_plan(
        items=items,
        prompt_rows=prompts,
        endpoint_names=names,
        endpoints=endpoints,
        prices=prices,
        input_token_safety_factor=float(
            pm_config["api_cost_planning"]["input_token_safety_factor"]
        ),
    )
    code_manifest = {"fixture": {"relative_path": "fixture", "sha256": "a" * 64}}
    transport = module._transport_contract(
        provider_output_attempts_by_family={
            "google_gemini": 2,
            "deepseek_official": 3,
        },
        code_manifest=code_manifest,
    )
    kwargs = dict(
        plan=plan,
        quality_contract=contract,
        transport_contract=transport,
        code_manifest=code_manifest,
        experiment_config_path=ROOT / "configs/experiment.yaml",
        pm_config_path=ROOT / "configs/pm_v1_5.yaml",
        tracked_contract_path=tracked,
        prepared_dir=prepared,
        endpoint_names=names,
        endpoints=endpoints,
        prices=prices,
        max_api_calls=1000,
        max_estimated_usd=1.0,
        max_input_tokens_per_call=8000,
    )
    first = module.build_cost_estimate(**kwargs)
    second = module.build_cost_estimate(**kwargs)
    assert first == second
    assert first["logical_calls"] == 72
    assert first["maximum_physical_attempts"] == 720
    assert first["maximum_cost_usd"] == pytest.approx(
        first["logical_single_attempt_cost_usd"] * 10
    )
    assert first["budget_gate"]["status"] == "PASS"
    assert first["training_labels_created"] is False
    assert len(first["cost_estimate_sha256"]) == 64


def test_quality_runner_rejects_mutated_prepared_prompt(tmp_path: Path) -> None:
    module = _load_runner()
    source = (
        ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_quality_packet_v1"
    )
    prepared = tmp_path / "packet"
    shutil.copytree(source, prepared)
    rows = [dict(row) for row in iter_jsonl(prepared / "quality_prompt_rows.jsonl")]
    rows[0]["messages"][0]["content"] += "\nMUTATED"
    write_jsonl(prepared / "quality_prompt_rows.jsonl", rows)
    with pytest.raises(RuntimeError, match="prompt hash drifted"):
        module._validate_packet(
            prepared_dir=prepared,
            tracked_contract_path=(
                ROOT
                / "data/pm_v1_5_contracts/"
                "longitudinal_oracle_memory_quality_diagnostic_v1.json"
            ),
        )


def test_quality_runner_rejects_unpriced_third_family() -> None:
    module = _load_runner()
    experiment = load_config(ROOT / "configs/experiment.yaml")
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    with pytest.raises(RuntimeError, match="pricing is not frozen.*qwen"):
        module._endpoint_records(
            experiment=experiment,
            pm_config=pm_config,
            judge_endpoint_names=[
                "training_judge_gemini_flash_lite",
                "training_judge_deepseek_official_flash",
                "training_judge_qwen122",
            ],
        )


def test_quality_runner_circuit_breaker_is_exactly_bounded() -> None:
    module = _load_runner()
    previous_class = None
    previous_count = 0
    for expected_count in range(1, 5):
        previous_class, previous_count = module._advance_failure_streak(
            previous_class=previous_class,
            previous_count=previous_count,
            retry_class="http_5xx",
        )
        assert previous_count == expected_count
    with pytest.raises(RuntimeError, match="circuit breaker"):
        module._advance_failure_streak(
            previous_class=previous_class,
            previous_count=previous_count,
            retry_class="http_5xx",
        )


def test_quality_runner_third_family_majority_cannot_be_training_gold() -> None:
    contract = load_config(
        ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_quality_diagnostic_v1.json"
    )
    assert contract["analysis"]["no_pooled_training_label"] is True
    assert (
        contract["analysis"]["third_family_majority"]
        == "descriptive diagnostic only; never a gold label"
    )
    source = RUNNER_PATH.read_text()
    assert '"training_labels_created": False' in source
    assert '"pm_training_authorized": False' in source
