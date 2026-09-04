from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v9 import (
    annotation_guidelines_text,
    prepare_bank_comparison,
    prepare_frozen_candidate,
    prepare_v9,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real-RAG V9 smoke-review artifacts without API use."
    )
    parser.add_argument(
        "--v8-cases",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v8"
        / "generation_pilot_semantic_review_cases.json",
    )
    parser.add_argument(
        "--legacy-bank",
        type=Path,
        default=ROOT.parents[2]
        / "esconv_experiment_bundle"
        / "policy_manager_35"
        / "data"
        / "strategy"
        / "pilot_strategy_cards_for_synthetic_sweep.jsonl",
    )
    parser.add_argument(
        "--current-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=ROOT / "outputs" / "strategy_rag_v1_bank_comparison",
    )
    parser.add_argument(
        "--frozen-candidate-dir",
        type=Path,
        default=ROOT / "outputs" / "strategy_rag_v1_frozen_candidate",
    )
    parser.add_argument(
        "--v9-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9",
    )
    parser.add_argument(
        "--distribution-shift-note",
        type=Path,
        default=ROOT
        / "outputs"
        / "strategy_rag_v1_audit"
        / "V1_BANK_DISTRIBUTION_SHIFT_NOTE.md",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v2.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (
        args.comparison_dir,
        args.frozen_candidate_dir,
        args.v9_dir,
    ):
        if directory.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {directory}")
    if args.distribution_shift_note.exists():
        raise RuntimeError(
            f"refusing to overwrite distribution-shift note: {args.distribution_shift_note}"
        )
    comparison = prepare_bank_comparison(
        v8_cases_path=args.v8_cases,
        legacy_bank_path=args.legacy_bank,
        current_bank_path=args.current_bank,
        out_dir=args.comparison_dir,
    )
    frozen = prepare_frozen_candidate(
        current_bank_path=args.current_bank,
        split_manifest_path=ROOT
        / "data"
        / "strategy"
        / "esconv_split_manifest.jsonl",
        bank_audit_path=ROOT / "data" / "strategy" / "strategy_bank_audit.json",
        out_dir=args.frozen_candidate_dir,
    )
    v9 = prepare_v9(
        v8_cases_path=args.v8_cases,
        current_bank_path=args.current_bank,
        experiment_config_path=args.experiment_config,
        pm_v2_config_path=args.pm_v2_config,
        frozen_manifest_path=args.frozen_candidate_dir
        / "strategy_rag_manifest.json",
        out_dir=args.v9_dir,
    )
    (args.v9_dir / "annotation_guidelines_ZH.md").write_text(
        annotation_guidelines_text(pair_complete=False),
        encoding="utf-8",
    )
    args.distribution_shift_note.write_text(
        """# V1 Strategy Bank distribution-shift note

## Facts

- training-time bank: 156 cards, logical name `v1_training_legacy_bank`, SHA256 `5b2d58c6be07d3c27e20e9162a3494f47f875d02b94b983f8b05c08ff044bf46`;
- confirmatory/external bank: 12,429 cards, logical name `v1_confirmatory_external_bank`, SHA256 `f64ded1e23b4a79c0e08c6b47355ad76b3880d5cbc0bfa7530f366149b5c70db`;
- both banks are recorded as ESConv-train-derived;
- no material leakage was identified in the completed V1 Strategy RAG audit;
- the resource environment changed between PM training-label construction and confirmatory/external execution.

This note neither invalidates the reported results nor hides the mismatch. It
records a stage-specific evidence-distribution shift.

## Optional future sensitivity checks

1. replay a bounded sample with both banks and the same canonical retriever;
2. compare retrieved strategy-type and score distributions;
3. regenerate a matched PM-label subset under the 12,429-card bank;
4. compare fixed-action and PM response deltas under both banks.

No formal experiment is rerun in this stage.
""",
        encoding="utf-8",
    )
    print(
        {
            "comparison": comparison,
            "frozen_candidate_manifest_sha256": frozen["manifest_sha256"],
            "v9": v9,
            "distribution_shift_note": str(args.distribution_shift_note),
            "api_calls": 0,
        }
    )


if __name__ == "__main__":
    main()
