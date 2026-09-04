from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
)
from metacom_pm.v1_5_multisource_decomposition import (
    build_multisource_decomposition_contract,
    validate_existing_multisource_outcomes,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_upper_bound_pilot_v1.json",
    )
    parser.add_argument(
        "--oracle-freeze",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_uptake_freeze_v1.json",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_upper_bound_backend_v1.jsonl",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_upper_bound_runtime_v1.jsonl",
    )
    parser.add_argument(
        "--oracle-outcomes",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_upper_bound_pilot_v1_execution_candidate/"
        "pilot_outcomes.jsonl",
    )
    parser.add_argument(
        "--oracle-artifact-attestation",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_upper_bound_pilot_v1_execution_candidate/"
        "artifact_attestation.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    oracle_contract = read_json(args.oracle_contract)
    oracle_freeze = read_json(args.oracle_freeze)
    backend_rows = [dict(row) for row in iter_jsonl(args.backend)]
    outcome_rows = [dict(row) for row in iter_jsonl(args.oracle_outcomes)]
    verification = require_artifact_attestation(
        args.oracle_artifact_attestation,
        required_stage="longitudinal_oracle_memory_upper_bound_pilot",
        required_output_paths={"pilot_outcomes": args.oracle_outcomes},
    )
    oracle_lineage = dict(oracle_contract.get("source_lineage") or {})
    if (
        oracle_freeze.get("status")
        != "ORACLE_MEMORY_UPTAKE_CAPACITY_SUPPORTED_REPORT_ONLY"
        or oracle_freeze.get("pilot_contract_sha256")
        != oracle_contract.get("contract_sha256")
        or oracle_freeze.get("pilot_outcomes_sha256")
        != sha256_file(args.oracle_outcomes)
        or oracle_lineage.get("oracle_runtime_sha256")
        != sha256_file(args.runtime)
        or oracle_lineage.get("oracle_backend_sha256")
        != sha256_file(args.backend)
    ):
        raise RuntimeError("oracle-memory freeze/lineage is not qualified")
    source_lineage = {
        "oracle_contract_path": str(args.oracle_contract.relative_to(ROOT)),
        "oracle_contract_file_sha256": sha256_file(args.oracle_contract),
        "oracle_contract_sha256": oracle_contract["contract_sha256"],
        "oracle_freeze_path": str(args.oracle_freeze.relative_to(ROOT)),
        "oracle_freeze_sha256": sha256_file(args.oracle_freeze),
        "oracle_backend_path": str(args.backend.relative_to(ROOT)),
        "oracle_backend_sha256": sha256_file(args.backend),
        "oracle_runtime_path": str(args.runtime.relative_to(ROOT)),
        "oracle_runtime_sha256": sha256_file(args.runtime),
        "oracle_outcomes_path": str(args.oracle_outcomes.relative_to(ROOT)),
        "oracle_outcomes_sha256": sha256_file(args.oracle_outcomes),
        "oracle_artifact_attestation_path": str(
            args.oracle_artifact_attestation.relative_to(ROOT)
        ),
        "oracle_artifact_attestation_sha256": verification[
            "attestation_sha256"
        ],
        "preparation_code_sha256": sha256_file(
            ROOT / "src/metacom_pm/v1_5_multisource_decomposition.py"
        ),
        "preparation_runner_sha256": sha256_file(__file__),
    }
    contract = build_multisource_decomposition_contract(
        oracle_contract=oracle_contract,
        backend_rows=backend_rows,
        source_lineage=source_lineage,
    )
    reused = validate_existing_multisource_outcomes(
        contract=contract,
        outcome_rows=outcome_rows,
    )
    if len(reused) != 6:
        raise RuntimeError("multisource decomposition must reuse six outcomes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, contract)
    print(
        {
            "status": "COMPLETE_ZERO_API_CONTRACT",
            "states": len(contract["states"]),
            "planned_new_logical_calls": contract[
                "planned_new_logical_calls"
            ],
            "reused_zero_api_outcomes": len(reused),
            "contract_sha256": contract["contract_sha256"],
            "api_clients_created": 0,
            "training_labels_created": False,
        }
    )


if __name__ == "__main__":
    main()
