#!/usr/bin/env python3
"""Build the strict-schema judge bake-off call plan and budget (zero API)."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_strict_judge_bakeoff import (
    build_call_plan,
    build_cost_estimate,
    strict_bakeoff_carry_forward,
    validate_endpoint_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _persist_json(path: Path, payload: dict) -> None:
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError(f"existing dry-run artifact drifted: {path}")
    write_json(path, payload)


def _persist_rows(path: Path, rows: list[dict]) -> None:
    if path.is_file() and _rows(path) != rows:
        raise RuntimeError(f"existing dry-run artifact drifted: {path}")
    write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "strict_pairwise_judge_bakeoff_v1.json",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_strict_judge_bakeoff_v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--carry-forward-from", type=Path)
    parser.add_argument("--max-api-calls", type=int, default=200)
    parser.add_argument("--max-estimated-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    args = parser.parse_args()

    prepared = read_json(args.prepared_dir / "qualification_contract.json")
    tracked = read_json(args.tracked_contract)
    if prepared != tracked:
        raise RuntimeError("prepared and tracked strict bake-off contracts differ")
    endpoint_contract = read_json(args.endpoint_contract)
    validate_endpoint_contract(endpoint_contract)
    expected_endpoint = {
        **endpoint_contract,
        "file_sha256": sha256_file(args.endpoint_contract),
        "content_sha256": sha256_text(canonical_json(endpoint_contract)),
    }
    if dict(prepared["endpoint_contract"]) != expected_endpoint:
        raise RuntimeError("strict bake-off endpoint contract drifted")
    packet = dict(prepared["packet_files"])
    source_dir = ROOT / str(packet["source_packet_directory"])
    ordered_path = source_dir / "ordered_pairwise_prompts.jsonl"
    if sha256_file(ordered_path) != packet["ordered_pairwise_prompts"]["sha256"]:
        raise RuntimeError("strict bake-off ordered prompt artifact drifted")
    plan = build_call_plan(
        ordered_pairwise_rows=_rows(ordered_path),
        endpoint_contract=endpoint_contract,
    )
    carry_forward = None
    if args.carry_forward_from is not None:
        carry_forward, _ = strict_bakeoff_carry_forward(
            source_dir=args.carry_forward_from,
            plan=plan,
            root=ROOT,
        )
    request = dict(endpoint_contract["request_contract"])
    code_paths = {
        "execution": ROOT
        / "scripts/v1_5/21w_run_strict_judge_bakeoff_v1_5.py",
        "dry_run": Path(__file__).resolve(),
        "preparer": ROOT
        / "scripts/v1_5/21u_prepare_strict_judge_bakeoff_v1_5.py",
        "bakeoff": ROOT
        / "src/metacom_pm/v1_5_strict_judge_bakeoff.py",
        "qualification_prompt": ROOT
        / "src/metacom_pm/v1_5_judge_qualification.py",
        "api": ROOT / "src/metacom_pm/api.py",
        "attempt_ledger": ROOT / "src/metacom_pm/attempt_ledger.py",
        "bounded_retry": ROOT / "src/metacom_pm/bounded_retry.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    estimate = build_cost_estimate(
        plan=plan,
        contract=prepared,
        code_manifest=code_manifest,
        maximum_attempts=int(
            request["maximum_physical_attempts_per_logical_call"]
        ),
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        carry_forward=carry_forward,
    )
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("strict bake-off dry-run budget gate failed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _persist_rows(args.out_dir / "call_plan.jsonl", plan)
    _persist_json(args.out_dir / "cost_estimate.json", estimate)
    summary = {
        "status": "DRY_RUN_COMPLETE_ZERO_API",
        "stage": estimate["stage"],
        "qualification_contract_sha256": estimate[
            "qualification_contract_sha256"
        ],
        "call_plan_sha256": estimate["call_plan_sha256"],
        "cost_estimate_sha256": estimate["cost_estimate_sha256"],
        "logical_calls": estimate["logical_calls"],
        "carried_forward_logical_calls": estimate[
            "carried_forward_logical_calls"
        ],
        "new_logical_calls": estimate["new_logical_calls"],
        "maximum_physical_attempts": estimate[
            "maximum_physical_attempts"
        ],
        "logical_single_attempt_cost_usd": estimate[
            "logical_single_attempt_cost_usd"
        ],
        "maximum_cost_usd": estimate["maximum_cost_usd"],
        "by_candidate": estimate["by_candidate"],
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    _persist_json(args.out_dir / "summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
