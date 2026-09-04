#!/usr/bin/env python3
"""Run and attest the zero-API Step-0 shortcut gate before the full sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.config import load_config
from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v1_5_shortcut_audit import (
    SHORTCUT_AUDIT_PROTOCOL,
    SHORTCUT_AUDIT_STAGE,
    audit_step0_shortcuts,
    require_step0_shortcut_audit_pass,
)
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit formal Step-0 features on the complete 468-state corpus before "
            "any 7,488-action response generation. This command makes no API calls."
        )
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--generation-attestation",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_step0_shortcut_audit",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report_path = args.out_dir / "step0_shortcut_audit.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    if not args.overwrite and (report_path.exists() or attestation_path.exists()):
        raise RuntimeError(
            "shortcut-audit outputs already exist; pass --overwrite only before "
            "the full action sweep has begun"
        )
    require_artifact_attestation(
        args.generation_attestation,
        required_stage="pm_v1_5_development_data",
        required_output_paths={
            "states": args.states,
            "evaluator_contexts": args.evaluator_contexts,
        },
    )
    config = load_config(args.pm_v2_config)
    if config.get("version") != "pm-v1.5":
        raise ValueError("Step-0 shortcut runner requires the PM-v1.5 config")
    shortcut_cfg = dict(config.get("shortcut_audit") or {})
    if (
        shortcut_cfg.get("protocol") != SHORTCUT_AUDIT_PROTOCOL
        or shortcut_cfg.get("fail_on_near_oracle_threshold") is not True
        or shortcut_cfg.get("fail_on_near_oracle_multivariate_probe") is not True
    ):
        raise ValueError("Step-0 shortcut-audit config is missing or stale")

    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    train_states = [state for state in states if state.split is PMV2Split.TRAIN]
    report = audit_step0_shortcuts(
        predictive_states=train_states,
        structural_states=states,
        evaluator_contexts=evaluator_contexts,
        expected_predictive_states=int(
            shortcut_cfg["expected_predictive_train_states"]
        ),
        expected_structural_states=int(
            shortcut_cfg["expected_structural_all_split_states"]
        ),
        maximum_single_threshold_balanced_accuracy=float(
            shortcut_cfg["maximum_single_threshold_balanced_accuracy"]
        ),
        maximum_multivariate_probe_balanced_accuracy=float(
            shortcut_cfg["maximum_multivariate_probe_balanced_accuracy"]
        ),
        centroid_noise_std=float(shortcut_cfg["centroid_noise_std"]),
        shuffle_seed=int(shortcut_cfg["shuffle_seed"]),
        group_cv_folds=int(shortcut_cfg["group_cv_folds"]),
        probe_logistic_c=float(shortcut_cfg["probe_logistic_c"]),
        probe_logistic_max_iter=int(shortcut_cfg["probe_logistic_max_iter"]),
        probe_tree_max_depth=int(shortcut_cfg["probe_tree_max_depth"]),
        probe_tree_min_samples_leaf=int(
            shortcut_cfg["probe_tree_min_samples_leaf"]
        ),
    )
    report["input_hashes"] = {
        "states": sha256_file(args.states),
        "evaluator_contexts": sha256_file(args.evaluator_contexts),
        "pm_v1_5_config": sha256_file(args.pm_v2_config),
        "development_data_attestation": sha256_file(
            args.generation_attestation
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    create_artifact_attestation(
        attestation_path,
        stage=SHORTCUT_AUDIT_STAGE,
        inputs={
            "states": args.states,
            "evaluator_contexts": args.evaluator_contexts,
            "pm_v1_5_config": args.pm_v2_config,
            "development_data_attestation": args.generation_attestation,
        },
        outputs={"shortcut_audit_report": (report_path, False)},
        parameters={
            "protocol": SHORTCUT_AUDIT_PROTOCOL,
            "outcome_labels_read": False,
            "resource_oracle_scope": "train_only",
        },
        expected={
            "predictive_train_states": int(
                shortcut_cfg["expected_predictive_train_states"]
            ),
            "structural_all_split_states": int(
                shortcut_cfg["expected_structural_all_split_states"]
            ),
        },
    )
    verification = require_step0_shortcut_audit_pass(
        report_path,
        attestation_path,
        expected_states_path=args.states,
        expected_evaluator_contexts_path=args.evaluator_contexts,
        expected_pm_config_path=args.pm_v2_config,
        expected_generation_attestation_path=args.generation_attestation,
    )
    print(
        {
            "status": verification["status"],
            "report": str(report_path),
            "maximum_single_threshold_balanced_accuracy": report[
                "maximum_observed_single_threshold_balanced_accuracy"
            ],
            "maximum_multivariate_probe_balanced_accuracy": report[
                "maximum_observed_multivariate_probe_balanced_accuracy"
            ],
        }
    )
    if report["status"] != "PASS":
        raise RuntimeError(
            "Step-0 shortcut audit failed; full action sweep remains blocked"
        )


if __name__ == "__main__":
    main()
