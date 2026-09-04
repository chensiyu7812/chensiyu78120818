#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v1_6_contracts import Step0Observation
from metacom_pm.pm_v1_6_shortcut_audit import (
    ShortcutAuditConfig,
    run_shortcut_audit,
)
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-states", type=Path, required=True)
    parser.add_argument("--train-step0", type=Path, required=True)
    parser.add_argument("--train-evaluator-contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6113)
    parser.add_argument("--high-similarity-threshold", type=float, default=0.45)
    parser.add_argument("--low-similarity-threshold", type=float, default=0.15)
    parser.add_argument("--maximum-probe-only-exact-match", type=float, default=0.85)
    parser.add_argument("--minimum-combined-gain-over-text", type=float, default=0.01)
    parser.add_argument("--minimum-shuffled-probe-drop", type=float, default=0.01)
    parser.add_argument("--minimum-challenge-count", type=int, default=1)
    args = parser.parse_args()

    states = load_states(args.train_states)
    step_rows = [
        Step0Observation.model_validate(row) for row in iter_jsonl(args.train_step0)
    ]
    step0 = {row.state_id: row for row in step_rows}
    if len(step0) != len(step_rows):
        raise RuntimeError("duplicate train Step-0 observation")
    evaluator = load_evaluator_context_index(
        args.train_evaluator_contexts,
        states=states,
        require_exact=True,
    )
    report = run_shortcut_audit(
        states=states,
        step0_by_state=step0,
        evaluator_contexts=evaluator,
        config=ShortcutAuditConfig(
            folds=args.folds,
            seed=args.seed,
            high_similarity_threshold=args.high_similarity_threshold,
            low_similarity_threshold=args.low_similarity_threshold,
            maximum_probe_only_exact_match=args.maximum_probe_only_exact_match,
            minimum_combined_gain_over_text=args.minimum_combined_gain_over_text,
            minimum_shuffled_probe_drop=args.minimum_shuffled_probe_drop,
            minimum_challenge_count=args.minimum_challenge_count,
        ),
    )
    report.update(
        {
            "train_states_sha256": sha256_file(args.train_states),
            "train_step0_sha256": sha256_file(args.train_step0),
            "train_evaluator_contexts_sha256": sha256_file(
                args.train_evaluator_contexts
            ),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    if report["status"] != "PASS":
        raise RuntimeError(
            "PM-v1.6 shortcut audit failed; formal training is prohibited"
        )


if __name__ == "__main__":
    main()
