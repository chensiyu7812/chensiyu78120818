#!/usr/bin/env python3
"""Audit whether existing train-only weak labels contain learnable action effects."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.v1_5_factorized_weak_learning import (
    FACTORIZED_SIGNAL_AUDIT_PROTOCOL,
    build_factorized_signal_report,
)
from metacom_pm.v1_5_weak_supervision import require_weak_supervision_contract


ROOT = Path(__file__).resolve().parents[2]


def _train_states(path: Path):
    return [
        state for state in load_states(path) if state.split is PMV2Split.TRAIN
    ]


def _rows_for_states(path: Path, state_ids: set[str]):
    return [
        row for row in iter_jsonl(path) if str(row.get("state_id")) in state_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--weak-contract",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "llm_weak_supervision_v1.json",
    )
    parser.add_argument("--longitudinal-states", type=Path, required=True)
    parser.add_argument("--longitudinal-contexts", type=Path, required=True)
    parser.add_argument("--longitudinal-labels", type=Path, required=True)
    parser.add_argument("--longitudinal-raw-judges", type=Path, required=True)
    parser.add_argument("--auxiliary-states", type=Path, required=True)
    parser.add_argument("--auxiliary-labels", type=Path, required=True)
    parser.add_argument("--auxiliary-raw-judges", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    contract = require_weak_supervision_contract(args.weak_contract)
    families = list(contract["judge_aggregation"]["judge_families"])
    weights = {
        str(key): float(value)
        for key, value in config["quality_composite"]["weights"].items()
    }

    long_states = _train_states(args.longitudinal_states)
    long_ids = {state.state_id for state in long_states}
    long_contexts = {
        str(row["state_id"]): row
        for row in iter_jsonl(args.longitudinal_contexts)
        if str(row.get("state_id")) in long_ids
    }
    if set(long_contexts) != long_ids:
        raise RuntimeError("longitudinal evaluator contexts do not cover train states")
    long_report = build_factorized_signal_report(
        states=long_states,
        weak_labels=_rows_for_states(args.longitudinal_labels, long_ids),
        raw_judge_rows=_rows_for_states(
            args.longitudinal_raw_judges, long_ids
        ),
        response_weights=weights,
        expected_judge_families=families,
        evaluator_contexts=long_contexts,
    )

    auxiliary_states = _train_states(args.auxiliary_states)
    auxiliary_ids = {state.state_id for state in auxiliary_states}
    auxiliary_report = build_factorized_signal_report(
        states=auxiliary_states,
        weak_labels=_rows_for_states(args.auxiliary_labels, auxiliary_ids),
        raw_judge_rows=_rows_for_states(
            args.auxiliary_raw_judges, auxiliary_ids
        ),
        response_weights=weights,
        expected_judge_families=families,
    )

    inputs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            "config": args.config,
            "weak_contract": args.weak_contract,
            "longitudinal_states": args.longitudinal_states,
            "longitudinal_contexts": args.longitudinal_contexts,
            "longitudinal_labels": args.longitudinal_labels,
            "longitudinal_raw_judges": args.longitudinal_raw_judges,
            "auxiliary_states": args.auxiliary_states,
            "auxiliary_labels": args.auxiliary_labels,
            "auxiliary_raw_judges": args.auxiliary_raw_judges,
            "script": Path(__file__).resolve(),
        }.items()
    }
    report = {
        "protocol": FACTORIZED_SIGNAL_AUDIT_PROTOCOL,
        "status": "COMPLETE_ZERO_API_TRAIN_ONLY_DIAGNOSTIC",
        "api_calls_made": 0,
        "internal_or_external_outcomes_opened": False,
        "inputs": inputs,
        "input_binding_sha256": sha256_text(canonical_json(inputs)),
        "longitudinal_synthetic": long_report,
        "esconv_auxiliary": auxiliary_report,
        "interpretation": {
            "direct_16_action_absolute_regression_supported": False,
            "reason": (
                "A 16-way maximum over noisy absolute scores is vulnerable to "
                "winner's-curse pseudo-oracles; the factorized report retains "
                "within-state effects and judge disagreement instead."
            ),
            "next_training_unit": (
                "state-by-component repeated-measure effect, never an individual "
                "factorial contrast treated as an independent example"
            ),
        },
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "factorized_weak_signal_report.json", report)
    print(report["status"])


if __name__ == "__main__":
    main()
