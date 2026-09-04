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
from metacom_pm.v1_5_multisource_decomposition import (
    MULTISOURCE_DECOMPOSITION_STAGE,
)
from metacom_pm.v1_5_multisource_uptake import (
    MULTISOURCE_UPTAKE_PROTOCOL,
    analyze_multisource_uptake,
)


ANALYSIS_STAGE = "longitudinal_multisource_uptake_analysis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--existing-outcomes", type=Path, required=True)
    parser.add_argument("--single-source-outcomes", type=Path, required=True)
    parser.add_argument("--execution-attestation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require_artifact_attestation(
        args.execution_attestation,
        required_stage=MULTISOURCE_DECOMPOSITION_STAGE,
        required_output_paths={
            "single_source_outcomes": args.single_source_outcomes
        },
    )
    config = load_config(args.pm_config)
    contract = read_json(args.contract)
    backend_by_card = {
        str(row["card_id"]): MemoryBackendRecord.model_validate(row)
        for row in iter_jsonl(args.backend)
    }
    existing = [
        ActionOutcome.model_validate(row)
        for row in iter_jsonl(args.existing_outcomes)
    ]
    single_source = [
        ActionOutcome.model_validate(row)
        for row in iter_jsonl(args.single_source_outcomes)
    ]
    spec = semantic_encoder_spec_from_config(config)
    encoder = FrozenTransformerSemanticEncoder.load(spec)
    semantic_runtime = require_semantic_runtime_contract(config, encoder)
    report = analyze_multisource_uptake(
        contract=contract,
        backend_by_card=backend_by_card,
        existing_outcomes=existing,
        single_source_outcomes=single_source,
        encoder=encoder,
    )
    if report.get("protocol") != MULTISOURCE_UPTAKE_PROTOCOL:
        raise RuntimeError("multisource analysis protocol drifted")
    report["semantic_runtime"] = semantic_runtime
    write_json(args.out, report)
    create_artifact_attestation(
        args.out.parent / "multisource_uptake_attestation.json",
        stage=ANALYSIS_STAGE,
        inputs={
            "pm_config": args.pm_config,
            "contract": args.contract,
            "analysis_code": (
                Path(__file__).resolve().parents[2]
                / "src/metacom_pm/v1_5_multisource_uptake.py"
            ),
            "analysis_runner": Path(__file__).resolve(),
            "backend": args.backend,
            "existing_outcomes": args.existing_outcomes,
            "single_source_outcomes": args.single_source_outcomes,
            "execution_attestation": args.execution_attestation,
        },
        outputs={"uptake_report": (args.out, False)},
        parameters={
            "contract_sha256": report["contract_sha256"],
            "semantic_encoder_spec_sha256": report[
                "semantic_encoder_spec_sha256"
            ],
            "classification": report["status"],
            "api_judges_used": False,
            "training_labels_created": False,
            "pm_training_authorized": False,
        },
        expected={"states": 3, "source_contrasts": 9},
    )
    print(report)


if __name__ == "__main__":
    main()
