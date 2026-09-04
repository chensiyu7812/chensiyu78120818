from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.v1_5_response_mechanism_pilot import (
    response_mechanism_pilot_contract,
    select_response_mechanism_pilot_states,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    states = load_states(args.states)
    evaluator = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    selected = select_response_mechanism_pilot_states(
        states,
        evaluator.by_state,
    )
    contract = response_mechanism_pilot_contract(
        selected_states=selected,
        source_lineage={
            "states_path": str(args.states),
            "states_sha256": sha256_file(args.states),
            "evaluator_contexts_path": str(args.evaluator_contexts),
            "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        },
    )
    write_json(args.out, contract)
    print(contract)


if __name__ == "__main__":
    main()
