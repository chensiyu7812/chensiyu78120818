#!/usr/bin/env python3
"""Merge complete longitudinal-judging shard ledgers without any API calls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_deterministic_sharding import (
    load_validated_sharding_contract,
)


MERGE_PROTOCOL = "pm-v1.5-longitudinal-judging-deterministic-shard-merge-v1"


def merge_complete_shard_ledgers(
    *,
    full_call_plan_path: Path,
    sharding_contract_path: Path,
    shard_dirs: list[Path],
    out_dir: Path,
) -> dict:
    """Validate every shard and materialize one exact full-plan carry-forward ledger."""

    full_rows = list(iter_jsonl(full_call_plan_path))
    if not full_rows:
        raise RuntimeError("full call plan is empty")
    contract, shard_plans = load_validated_sharding_contract(
        full_rows=full_rows,
        contract_path=sharding_contract_path,
    )
    if len(shard_dirs) != len(shard_plans):
        raise RuntimeError("one and only one directory is required for every shard")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("shard merge refuses a non-empty output directory")

    events_by_key: dict[str, list[dict]] = {}
    shard_records = []
    stage = None
    for shard_index, (shard_dir, shard_rows) in enumerate(
        zip(shard_dirs, shard_plans, strict=True)
    ):
        shard_dir = shard_dir.resolve()
        call_plan_path = shard_dir / "call_plan.jsonl"
        ledger_path = shard_dir / "judge_call_ledger.jsonl"
        summary_path = shard_dir / "summary.json"
        estimate_path = shard_dir / "cost_estimate.json"
        attestation_path = shard_dir / "artifact_attestation.json"
        if list(iter_jsonl(call_plan_path)) != full_rows:
            raise RuntimeError(f"shard {shard_index} full call plan drifted")
        estimate = read_json(estimate_path)
        summary = read_json(summary_path)
        if int(estimate.get("execution_shard_index", -1)) != shard_index:
            raise RuntimeError(f"shard {shard_index} cost identity index mismatch")
        if int(summary.get("execution_shard_index", -1)) != shard_index:
            raise RuntimeError(f"shard {shard_index} summary index mismatch")
        if (
            (estimate.get("execution_sharding_contract") or {}).get(
                "contract_sha256"
            )
            != contract["contract_sha256"]
            or (summary.get("execution_sharding_contract") or {}).get(
                "contract_sha256"
            )
            != contract["contract_sha256"]
        ):
            raise RuntimeError(f"shard {shard_index} contract binding mismatch")
        if summary.get("status") != "SHARD_COMPLETE_NO_AGGREGATE":
            raise RuntimeError(f"shard {shard_index} is not complete")
        if int(summary.get("completed_shard_calls") or -1) != len(shard_rows):
            raise RuntimeError(f"shard {shard_index} completed-call count mismatch")
        observed_stage = str(estimate.get("stage") or "")
        if not observed_stage or (stage is not None and observed_stage != stage):
            raise RuntimeError("shards do not share one scientific judging stage")
        stage = observed_stage
        require_artifact_attestation(
            attestation_path,
            required_stage=stage,
            required_output_paths={
                "summary": summary_path,
                "shard_call_results": shard_dir / "shard_call_results.jsonl",
                "call_ledger": ledger_path,
            },
        )
        expected_calls = {
            str(row["physical_call_key"]): int(row.get("max_http_attempts") or 1)
            for row in shard_rows
        }
        ledger = PersistentAttemptLedger(
            ledger_path,
            stage=stage,
            expected_calls=expected_calls,
            maximum_total_attempts=10**9,
        )
        if not all(ledger.succeeded(key) for key in expected_calls):
            raise RuntimeError(f"shard {shard_index} ledger is not fully successful")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for event in ledger.event_rows:
            key = str(event.get("call_key") or "")
            if key not in expected_calls:
                raise RuntimeError(f"shard {shard_index} ledger contains extra key")
            grouped[key].append(event)
        if set(grouped) != set(expected_calls):
            raise RuntimeError(f"shard {shard_index} ledger key coverage mismatch")
        for key, events in grouped.items():
            if key in events_by_key:
                raise RuntimeError(f"physical call appears in multiple shards: {key}")
            events_by_key[key] = events
        shard_records.append(
            {
                "shard_index": shard_index,
                "directory": str(shard_dir),
                "summary_sha256": sha256_file(summary_path),
                "ledger_sha256": sha256_file(ledger_path),
                "attestation_sha256": sha256_file(attestation_path),
                "physical_attempts": ledger.started_attempts,
                "logical_calls": len(expected_calls),
            }
        )

    full_keys = [str(row["physical_call_key"]) for row in full_rows]
    if set(events_by_key) != set(full_keys):
        raise RuntimeError("merged shard ledgers do not exactly cover the full call plan")
    merged_events = [event for key in full_keys for event in events_by_key[key]]
    assert stage is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_plan_path = out_dir / "call_plan.jsonl"
    merged_ledger_path = out_dir / "judge_call_ledger.jsonl"
    write_jsonl(merged_plan_path, full_rows)
    write_jsonl(merged_ledger_path, merged_events)
    merged_ledger = PersistentAttemptLedger(
        merged_ledger_path,
        stage=stage,
        expected_calls={
            str(row["physical_call_key"]): int(row.get("max_http_attempts") or 1)
            for row in full_rows
        },
        maximum_total_attempts=10**9,
    )
    if not all(merged_ledger.succeeded(key) for key in full_keys):
        raise RuntimeError("merged full ledger validation failed")
    report = {
        "protocol": MERGE_PROTOCOL,
        "status": "COMPLETE_ZERO_API_MERGE",
        "zero_api": True,
        "stage": stage,
        "sharding_contract_sha256": contract["contract_sha256"],
        "full_call_plan_sha256": sha256_text(canonical_json(full_rows)),
        "logical_calls": len(full_rows),
        "physical_attempts": merged_ledger.started_attempts,
        "exact_key_coverage": True,
        "shards": shard_records,
        "merged_call_plan_path": str(merged_plan_path),
        "merged_ledger_path": str(merged_ledger_path),
        "merged_ledger_sha256": sha256_file(merged_ledger_path),
    }
    report_path = out_dir / "merge_report.json"
    write_json(report_path, report)
    create_artifact_attestation(
        out_dir / "artifact_attestation.json",
        stage="development_action_judging_shard_merge",
        inputs={
            "full_call_plan": full_call_plan_path,
            "sharding_contract": sharding_contract_path,
            **{
                f"shard_{index:02d}_attestation": directory
                / "artifact_attestation.json"
                for index, directory in enumerate(shard_dirs)
            },
        },
        outputs={
            "merge_report": (report_path, False),
            "merged_call_plan": (merged_plan_path, True),
            "merged_call_ledger": (merged_ledger_path, True),
        },
        parameters={
            "protocol": MERGE_PROTOCOL,
            "status": report["status"],
            "sharding_contract_sha256": contract["contract_sha256"],
            "exact_key_coverage": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-call-plan", type=Path, required=True)
    parser.add_argument("--sharding-contract", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        merge_complete_shard_ledgers(
            full_call_plan_path=args.full_call_plan,
            sharding_contract_path=args.sharding_contract,
            shard_dirs=args.shard_dir,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
