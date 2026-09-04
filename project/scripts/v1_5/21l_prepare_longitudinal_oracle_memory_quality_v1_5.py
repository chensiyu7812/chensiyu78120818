from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    canonical_json,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_oracle_memory_quality import (
    build_quality_items,
    build_quality_messages,
    quality_contract_record,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_upper_bound_pilot_v1.json",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_upper_bound_backend_v1.jsonl",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_upper_bound_pilot_v1_execution_candidate/"
        "pilot_outcomes.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pilot_contract = read_json(args.pilot_contract)
    state_rows = [dict(row) for row in iter_jsonl(args.states)]
    backend_rows = [dict(row) for row in iter_jsonl(args.backend)]
    outcome_rows = [dict(row) for row in iter_jsonl(args.outcomes)]
    items = build_quality_items(
        pilot_contract=pilot_contract,
        state_rows=state_rows,
        backend_rows=backend_rows,
        outcome_rows=outcome_rows,
    )
    prompt_rows = [
        {
            "pair_id": str(item["pair_id"]),
            "state_id": str(item["state_id"]),
            "order_variant": int(item["order_variant"]),
            "messages": build_quality_messages(item),
        }
        for item in items
    ]
    source_lineage = {
        "pilot_contract_path": str(args.pilot_contract.relative_to(ROOT)),
        "pilot_contract_file_sha256": sha256_file(args.pilot_contract),
        "pilot_contract_sha256": pilot_contract["contract_sha256"],
        "states_path": str(args.states.relative_to(ROOT)),
        "states_sha256": sha256_file(args.states),
        "backend_path": str(args.backend.relative_to(ROOT)),
        "backend_sha256": sha256_file(args.backend),
        "outcomes_path": str(args.outcomes.relative_to(ROOT)),
        "outcomes_sha256": sha256_file(args.outcomes),
        "preparation_code_sha256": sha256_file(
            ROOT / "src/metacom_pm/v1_5_oracle_memory_quality.py"
        ),
        "preparation_runner_sha256": sha256_file(__file__),
        "items_sha256": sha256_text(canonical_json(items)),
        "prompt_rows_sha256": sha256_text(canonical_json(prompt_rows)),
    }
    contract = quality_contract_record(source_lineage=source_lineage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "quality_items.jsonl", items)
    write_jsonl(args.out_dir / "quality_prompt_rows.jsonl", prompt_rows)
    write_json(args.out_dir / "quality_contract.json", contract)
    summary = {
        "protocol": contract["protocol"],
        "status": "COMPLETE_ZERO_API_PACKET",
        "states": 18,
        "items": len(items),
        "orders_per_state": 2,
        "contract_sha256": contract["contract_sha256"],
        "items_sha256": source_lineage["items_sha256"],
        "prompt_rows_sha256": source_lineage["prompt_rows_sha256"],
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
