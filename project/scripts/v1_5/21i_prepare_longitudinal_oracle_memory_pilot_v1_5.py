from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_data import (
    load_evaluator_context_index,
    load_states,
)
from metacom_pm.sweep import load_backends
from metacom_pm.v1_5_oracle_memory_pilot import (
    build_oracle_memory_pilot_contract,
    materialize_oracle_backend,
    select_oracle_memory_pilot_states,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--exclude-contract", type=Path, required=True)
    parser.add_argument("--contract-out", type=Path, required=True)
    parser.add_argument("--runtime-out", type=Path, required=True)
    parser.add_argument("--backend-out", type=Path, required=True)
    args = parser.parse_args()

    states = load_states(args.states)
    evaluator = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    backends = load_backends(args.backend)
    previous = read_json(args.exclude_contract)
    prior_rows = [dict(row) for row in previous.get("selected_states") or []]
    excluded_state_ids = {str(row["state_id"]) for row in prior_rows}
    prior_user_ids = {str(row["user_id"]) for row in prior_rows}

    selected = select_oracle_memory_pilot_states(
        states,
        evaluator.by_state,
        backends,
        excluded_state_ids=excluded_state_ids,
        prefer_unused_user_ids={state.user_id for state in states} - prior_user_ids,
    )
    selected_card_ids = {str(row["card_id"]) for row in selected}
    runtime_rows = [
        dict(row)
        for row in iter_jsonl(args.runtime)
        if str(row["card_id"]) in selected_card_ids
    ]
    if len(runtime_rows) != len(selected):
        raise RuntimeError(
            f"oracle runtime subset is incomplete: "
            f"{len(runtime_rows)} != {len(selected)}"
        )
    runtime_rows.sort(key=lambda row: str(row["card_id"]))
    backend_rows = materialize_oracle_backend(
        selected_states=selected,
        backend_by_card=backends,
    )
    write_jsonl(args.runtime_out, runtime_rows)
    write_jsonl(args.backend_out, backend_rows)

    contract = build_oracle_memory_pilot_contract(
        selected_states=selected,
        source_lineage={
            "states_path": str(args.states),
            "states_sha256": sha256_file(args.states),
            "runtime_source_path": str(args.runtime),
            "runtime_source_sha256": sha256_file(args.runtime),
            "evaluator_contexts_path": str(args.evaluator_contexts),
            "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
            "backend_path": str(args.backend),
            "backend_sha256": sha256_file(args.backend),
            "excluded_prior_contract_path": str(args.exclude_contract),
            "excluded_prior_contract_sha256": sha256_file(args.exclude_contract),
            "oracle_runtime_path": str(args.runtime_out),
            "oracle_runtime_sha256": sha256_file(args.runtime_out),
            "oracle_backend_path": str(args.backend_out),
            "oracle_backend_sha256": sha256_file(args.backend_out),
        },
    )
    write_json(args.contract_out, contract)
    print(contract)


if __name__ == "__main__":
    main()
