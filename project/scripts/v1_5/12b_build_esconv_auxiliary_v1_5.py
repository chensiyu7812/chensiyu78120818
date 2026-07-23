#!/usr/bin/env python3
"""Build the ESConv auxiliary train/calibration/internal-test training support.

Uses only the 52 ESConv dialogues already used to seed the 52 synthetic
development users -- bank-disjoint from the Strategy Bank's 823 source
dialogues by construction. Never reads or touches the held-out 169-dialogue
ESConv test split (data/esconv_test_v1_5/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.esconv_v1_5 import build_esconv_v1_5_auxiliary_training_artifacts
from metacom_pm.io import sha256_file
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--esconv", type=Path, default=ROOT / "data" / "external" / "ESConv.json"
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data" / "strategy" / "esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--selected-seed-sources",
        type=Path,
        default=ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data" / "esconv_auxiliary_v1_5"
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    bank_cfg = config.get("strategy_bank_contract") or {}
    aux_cfg = config.get("esconv_auxiliary_training") or {}
    expected_bank = (ROOT / str(bank_cfg.get("relative_path") or "")).resolve()
    expected_seeds = (
        ROOT / str(bank_cfg.get("selected_seed_sources_relative_path") or "")
    ).resolve()
    if (
        args.strategy_bank.resolve() != expected_bank
        or sha256_file(args.strategy_bank) != bank_cfg.get("sha256")
        or args.selected_seed_sources.resolve() != expected_seeds
        or sha256_file(args.selected_seed_sources)
        != bank_cfg.get("selected_seed_sources_sha256")
    ):
        raise RuntimeError("ESConv auxiliary adapter input drifts from the frozen V1.5 bank")
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    require_semantic_runtime_contract(config, encoder)
    report = build_esconv_v1_5_auxiliary_training_artifacts(
        esconv_path=args.esconv,
        split_manifest_path=args.split_manifest,
        strategy_bank_path=args.strategy_bank,
        selected_seed_sources_path=args.selected_seed_sources,
        out_dir=args.out_dir,
        semantic_encoder=encoder,
        strategy_estimated_tokens=int(
            config["external_evaluation"]["strategy_action_tokens"]
        ),
    )
    expected_state_counts = dict(aux_cfg.get("expected_state_counts") or {})
    expected_dialogue_counts = dict(aux_cfg.get("split_dialogue_counts") or {})
    observed_state_counts = {
        split: int(row["state_count"]) for split, row in report["split_reports"].items()
    }
    observed_dialogue_counts = {
        split: int(row["dialogue_count"])
        for split, row in report["split_reports"].items()
    }
    if (
        report.get("protocol") != aux_cfg.get("protocol")
        or report.get("semantic_family") != aux_cfg.get("semantic_family")
        or report.get("seed_dialogue_count") != int(aux_cfg.get("seed_dialogue_count", -1))
        or report.get("sample_selection_rule") != aux_cfg.get("sample_selection_rule")
        or report.get("bank_disjoint") is not True
        or int(aux_cfg.get("bank_disjoint") is True) != 1
        or report.get("strategy_bank_source_dialogue_overlap_count")
        != int(aux_cfg.get("strategy_bank_source_dialogue_overlap_count", -1))
        or observed_dialogue_counts != {k: int(v) for k, v in expected_dialogue_counts.items()}
        or observed_state_counts != {k: int(v) for k, v in expected_state_counts.items()}
        or sum(observed_state_counts.values())
        != int(aux_cfg.get("expected_total_states", -1))
        or report.get("excluded_history_below_two_count")
        != int(aux_cfg.get("excluded_history_below_two_count", -1))
        or report.get("turn_selection_protocol")
        != aux_cfg.get("turn_selection_protocol")
    ):
        raise RuntimeError(
            "ESConv auxiliary adapter output drifts from its frozen V1.5 design: "
            f"observed_state_counts={observed_state_counts}, "
            f"observed_dialogue_counts={observed_dialogue_counts}"
        )
    print(report)


if __name__ == "__main__":
    main()
