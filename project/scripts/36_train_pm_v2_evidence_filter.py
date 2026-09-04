#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.evidence_filter_model import (
    PMV2EvidenceFilterModel,
    build_memory_filter_examples,
)
from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_semantic_audit import require_semantic_sanity_pass
from metacom_pm.sweep import load_backends


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter",
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    settings = dict(config["evidence_filter_training"])
    expected_keys = {
        "protocol",
        "train_split_only",
        "word_hash_features",
        "char_hash_features",
        "random_seed",
        "calibration_thresholds",
        "calibration_minimum_recall",
        "calibration_maximum_false_positive_rate",
        "internal_minimum_recall",
        "internal_minimum_precision",
        "internal_minimum_specificity",
        "require_before_development_sweep",
    }
    if set(settings) != expected_keys:
        raise RuntimeError("evidence_filter_training keys do not match the contract")
    if (
        settings["protocol"] != "pm-v2-evidence-filter-training-v1"
        or settings["train_split_only"] is not True
        or settings["require_before_development_sweep"] is not True
    ):
        raise RuntimeError("invalid evidence-filter training protocol")

    semantic_sanity = require_semantic_sanity_pass(
        report_path=args.semantic_sanity_report,
        attestation_path=args.semantic_sanity_attestation,
        config=config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
    )

    states = load_states(args.states)
    evaluator = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    backends = load_backends(args.backend)
    if set(backends) != {state.card_id for state in states}:
        raise RuntimeError("evidence-filter state/backend universe mismatch")
    examples_by_split = {}
    for split in (
        PMV2Split.TRAIN,
        PMV2Split.CALIBRATION,
        PMV2Split.INTERNAL_TEST,
    ):
        split_states = [state for state in states if state.split is split]
        examples_by_split[split] = build_memory_filter_examples(
            split_states,
            {card_id: record.items for card_id, record in backends.items()},
            evaluator.by_state,
        )
        if not examples_by_split[split]:
            raise RuntimeError(f"evidence-filter {split.value} examples are empty")

    model = PMV2EvidenceFilterModel(
        word_features=int(settings["word_hash_features"]),
        char_features=int(settings["char_hash_features"]),
        seed=int(settings["random_seed"]),
    ).fit(examples_by_split[PMV2Split.TRAIN])
    calibration = model.calibrate(
        examples_by_split[PMV2Split.CALIBRATION],
        thresholds=[float(value) for value in settings["calibration_thresholds"]],
        minimum_recall=float(settings["calibration_minimum_recall"]),
        maximum_false_positive_rate=float(
            settings["calibration_maximum_false_positive_rate"]
        ),
    )
    internal = model.evaluate_internal(
        examples_by_split[PMV2Split.INTERNAL_TEST],
        minimum_recall=float(settings["internal_minimum_recall"]),
        minimum_precision=float(settings["internal_minimum_precision"]),
        minimum_specificity=float(settings["internal_minimum_specificity"]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "evidence_filter.joblib"
    report_path = args.out_dir / "training_report.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    model.save(checkpoint_path)
    report = {
        "status": "COMPLETE",
        "protocol": settings["protocol"],
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "states_sha256": sha256_file(args.states),
        "backend_sha256": sha256_file(args.backend),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_contract_sha256": model.contract_hash(),
        "training": model.training_report,
        "calibration": calibration,
        "internal": internal,
        "scientific_scope": (
            "train-only synthetic item supervision; calibration threshold selection; "
            "internal held-out gate; external usefulness remains unproven"
        ),
        "semantic_sanity": semantic_sanity,
    }
    write_json(report_path, report)
    create_artifact_attestation(
        attestation_path,
        stage="pm_v2_evidence_filter_training",
        inputs={
            "pm_v2_config": args.pm_v2_config,
            "states": args.states,
            "backend": args.backend,
            "evaluator_contexts": args.evaluator_contexts,
            "semantic_sanity_report": args.semantic_sanity_report,
            "semantic_sanity_attestation": args.semantic_sanity_attestation,
        },
        outputs={
            "checkpoint": (checkpoint_path, False),
            "report": (report_path, False),
        },
        parameters={
            "settings": settings,
            "model_contract_sha256": model.contract_hash(),
            "semantic_sanity": semantic_sanity,
        },
        expected={
            "train_examples": len(examples_by_split[PMV2Split.TRAIN]),
            "calibration_examples": len(examples_by_split[PMV2Split.CALIBRATION]),
            "internal_examples": len(examples_by_split[PMV2Split.INTERNAL_TEST]),
        },
    )
    print(report)


if __name__ == "__main__":
    main()
