from __future__ import annotations

import importlib.util
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.io import iter_jsonl, write_json, write_jsonl
from metacom_pm.v1_5_deterministic_sharding import (
    partition_call_plan,
    sharding_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "v1_5"
    / "21c_merge_longitudinal_judging_shards_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_longitudinal_judging_shard_merge", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan_rows() -> list[dict]:
    return [
        {
            "physical_call_key": f"physical-{index:02d}",
            "state_id": f"state-{index:02d}",
            "action_id": "M0+R0",
            "judge_family": "google_gemini",
            "judge_type": "response",
            "prompt_hash": f"{index:064x}",
            "input_tokens_est": 100,
            "max_output_tokens": 50,
            "max_http_attempts": 4,
            "maximum_cost_usd": 0.001,
        }
        for index in range(12)
    ]


def test_complete_shard_ledgers_merge_to_exact_full_carry_forward(tmp_path) -> None:
    module = _module()
    full_rows = _plan_rows()
    full_plan_path = tmp_path / "full_call_plan.jsonl"
    write_jsonl(full_plan_path, full_rows)
    shards = partition_call_plan(full_rows, shard_count=4)
    contract = sharding_contract(
        full_rows,
        shards,
        shard_count=4,
        source_call_plan_path=full_plan_path,
    )
    prep = tmp_path / "prep"
    prep.mkdir()
    contract_path = prep / "sharding_contract.json"
    write_json(contract_path, contract)
    for index, rows in enumerate(shards):
        write_jsonl(prep / f"call_plan_shard_{index:02d}.jsonl", rows)

    shard_dirs = []
    stage = "pm_v2_action_judging_train_calibration"
    for index, rows in enumerate(shards):
        shard_dir = tmp_path / f"shard-{index}"
        shard_dir.mkdir()
        shard_dirs.append(shard_dir)
        write_jsonl(shard_dir / "call_plan.jsonl", full_rows)
        write_json(
            shard_dir / "cost_estimate.json",
            {
                "stage": stage,
                "execution_shard_index": index,
                "execution_sharding_contract": contract,
            },
        )
        write_json(
            shard_dir / "summary.json",
            {
                "status": "SHARD_COMPLETE_NO_AGGREGATE",
                "execution_shard_index": index,
                "execution_sharding_contract": contract,
                "completed_shard_calls": len(rows),
            },
        )
        write_jsonl(
            shard_dir / "shard_call_results.jsonl",
            [{**row, "parsed": {"ok": True}} for row in rows],
        )
        ledger = PersistentAttemptLedger(
            shard_dir / "judge_call_ledger.jsonl",
            stage=stage,
            expected_calls={
                str(row["physical_call_key"]): 4 for row in rows
            },
            maximum_total_attempts=len(rows) * 4,
        )
        for row in rows:
            reservation = ledger.reserve(
                str(row["physical_call_key"]),
                record_ids={
                    "state_id": row["state_id"],
                    "action_id": row["action_id"],
                    "judge_family": row["judge_family"],
                    "judge_type": row["judge_type"],
                },
                prompt_sha256=row["prompt_hash"],
            )
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash="f" * 64,
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                error=None,
                result={"parsed": {"ok": True}},
            )
        create_artifact_attestation(
            shard_dir / "artifact_attestation.json",
            stage=stage,
            inputs={
                "sharding_contract": contract_path,
                "full_call_plan": full_plan_path,
            },
            outputs={
                "summary": (shard_dir / "summary.json", False),
                "shard_call_results": (
                    shard_dir / "shard_call_results.jsonl",
                    True,
                ),
                "call_ledger": (
                    shard_dir / "judge_call_ledger.jsonl",
                    True,
                ),
            },
            parameters={"execution_shard_index": index},
        )

    out_dir = tmp_path / "merged"
    report = module.merge_complete_shard_ledgers(
        full_call_plan_path=full_plan_path,
        sharding_contract_path=contract_path,
        shard_dirs=shard_dirs,
        out_dir=out_dir,
    )
    assert report["status"] == "COMPLETE_ZERO_API_MERGE"
    assert report["logical_calls"] == len(full_rows)
    assert report["exact_key_coverage"] is True
    assert list(iter_jsonl(out_dir / "call_plan.jsonl")) == full_rows
    merged = PersistentAttemptLedger(
        out_dir / "judge_call_ledger.jsonl",
        stage=stage,
        expected_calls={
            str(row["physical_call_key"]): 4 for row in full_rows
        },
        maximum_total_attempts=10**9,
    )
    assert all(
        merged.succeeded(str(row["physical_call_key"])) for row in full_rows
    )


def test_formal_runner_exposes_shard_and_zero_api_aggregate_modes() -> None:
    source = (
        ROOT / "scripts" / "v1_5" / "21_judge_pm_v2_action_sweep_v1_5.py"
    ).read_text(encoding="utf-8")
    assert '"--execution-sharding-contract"' in source
    assert '"--execution-shard-index"' in source
    assert '"--aggregate-only"' in source
    assert '"SHARD_COMPLETE_NO_AGGREGATE"' in source
    assert "training_labels_created" in source
