#!/usr/bin/env python3
"""Build the frozen single-session ESConv test under the PM-v1.5 contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.esconv_v1_5 import build_esconv_v1_5_test_artifacts
from metacom_pm.io import read_json, sha256_file
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
        "--out-dir", type=Path, default=ROOT / "data" / "esconv_test_v1_5"
    )
    parser.add_argument(
        "--development-data-report",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_data_report.json",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    bank_cfg = config.get("strategy_bank_contract") or {}
    esconv_cfg = config.get("esconv_external_evaluation") or {}
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
        raise RuntimeError("ESConv adapter input drifts from the frozen V1.5 bank")
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    require_semantic_runtime_contract(config, encoder)
    report = build_esconv_v1_5_test_artifacts(
        esconv_path=args.esconv,
        split_manifest_path=args.split_manifest,
        strategy_bank_path=args.strategy_bank,
        selected_seed_sources_path=args.selected_seed_sources,
        out_dir=args.out_dir,
        semantic_encoder=encoder,
        strategy_estimated_tokens=int(
            config["external_evaluation"]["strategy_action_tokens"]
        ),
        development_observable_state_support=(
            read_json(args.development_data_report).get("observable_state_support")
            or {}
        ),
    )
    if (
        report.get("protocol") != esconv_cfg.get("protocol")
        or report.get("turn_selection_protocol")
        != esconv_cfg.get("turn_selection_protocol")
        or report.get("raw_supporter_turns")
        != int(esconv_cfg.get("raw_test_supporter_turns", -1))
        or report.get("excluded_history_below_two_count")
        != int(esconv_cfg.get("excluded_history_below_two_turns", -1))
        or report.get("test_turns")
        != int(esconv_cfg.get("expected_primary_states", -1))
        or report.get("test_dialogues")
        != int((esconv_cfg.get("nonoverlap_dialogue_counts") or {}).get("test", -1))
    ):
        raise RuntimeError("ESConv adapter output drifts from its frozen V1.5 design")
    print(report)


if __name__ == "__main__":
    main()
