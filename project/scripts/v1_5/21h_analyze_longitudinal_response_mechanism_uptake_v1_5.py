from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.config import load_config
from metacom_pm.contracts import ActionOutcome, MemoryBackendRecord
from metacom_pm.io import iter_jsonl, read_json, write_json
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.v1_5_response_mechanism_uptake import analyze_uptake


PILOT_STAGE = "longitudinal_response_mechanism_pilot"
UPTAKE_STAGE = "longitudinal_response_mechanism_uptake_diagnostic"
UPTAKE_MEASUREMENT_CODE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/metacom_pm/v1_5_response_mechanism_uptake.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-config", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--measurement-contract", type=Path, required=True)
    parser.add_argument("--pilot-outcomes", type=Path, required=True)
    parser.add_argument("--pilot-attestation", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require_artifact_attestation(
        args.pilot_attestation,
        required_stage=PILOT_STAGE,
        required_output_paths={"pilot_outcomes": args.pilot_outcomes},
    )
    config = load_config(args.pm_config)
    pilot_contract = read_json(args.pilot_contract)
    measurement_contract = read_json(args.measurement_contract)
    states = load_states(args.states)
    evaluator = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    backend_by_card = {
        str(row["card_id"]): MemoryBackendRecord.model_validate(row)
        for row in iter_jsonl(args.backend)
    }
    outcomes = [
        ActionOutcome.model_validate(row) for row in iter_jsonl(args.pilot_outcomes)
    ]
    spec = semantic_encoder_spec_from_config(config)
    if measurement_contract.get("semantic_encoder_spec_sha256") != spec.digest():
        raise RuntimeError("uptake measurement encoder spec is stale")
    encoder = FrozenTransformerSemanticEncoder.load(spec)
    semantic_runtime = require_semantic_runtime_contract(config, encoder)
    report = analyze_uptake(
        measurement_contract=measurement_contract,
        pilot_contract=pilot_contract,
        states=states,
        evaluator_by_state=evaluator.by_state,
        backend_by_card=backend_by_card,
        outcomes=outcomes,
        encoder=encoder,
    )
    report["semantic_runtime"] = semantic_runtime
    write_json(args.out, report)
    create_artifact_attestation(
        args.out.parent / "uptake_artifact_attestation.json",
        stage=UPTAKE_STAGE,
        inputs={
            "pm_config": args.pm_config,
            "pilot_contract": args.pilot_contract,
            "measurement_contract": args.measurement_contract,
            "uptake_measurement_code": UPTAKE_MEASUREMENT_CODE_PATH,
            "uptake_analyzer": Path(__file__).resolve(),
            "pilot_outcomes": args.pilot_outcomes,
            "pilot_attestation": args.pilot_attestation,
            "states": args.states,
            "evaluator_contexts": args.evaluator_contexts,
            "backend": args.backend,
        },
        outputs={"uptake_report": (args.out, False)},
        parameters={
            "measurement_contract_sha256": report[
                "measurement_contract_sha256"
            ],
            "pilot_contract_sha256": report["pilot_contract_sha256"],
            "semantic_encoder_spec_sha256": report[
                "semantic_encoder_spec_sha256"
            ],
            "api_judges_used": False,
            "training_labels_created": False,
            "pm_training_authorized": False,
        },
        expected={"pairs": 14},
    )
    print(report)


if __name__ == "__main__":
    main()
