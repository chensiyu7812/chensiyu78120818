#!/usr/bin/env python3
"""Prepare the zero-API minimum RS clean-pair pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import write_json, write_jsonl
from metacom_pm.v1_5_mvp_rs_pilot import build_minimum_rs_clean_pair_plan
from metacom_pm.v1_5_strategy_bank import (
    validate_strategy_bank_v2_human_annotations,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=(
            ROOT
            / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
            / "runtime_states.jsonl"
        ),
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
            / "strategy_cards_v2_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--strategy-candidate-dir",
        type=Path,
        default=(
            ROOT / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
        ),
    )
    parser.add_argument(
        "--strategy-lineage",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
            / "strategy_bank_v2_lineage.jsonl"
        ),
    )
    parser.add_argument(
        "--selected-seed-sources",
        type=Path,
        default=ROOT / "data/strategy/pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument("--human-annotations", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1",
    )
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    pm_config = load_config(args.pm_config)
    experiment_config = load_config(args.experiment_config)
    generation_contract = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(
        experiment_config,
        generation_contract.generator_endpoint,
    )

    human_passed = False
    human_review = None
    if args.human_annotations is not None:
        human_review = validate_strategy_bank_v2_human_annotations(
            candidate_dir=args.strategy_candidate_dir,
            annotations_path=args.human_annotations,
        )
        human_passed = human_review["status"].startswith("HUMAN_REVIEW_PASS")

    payload = build_minimum_rs_clean_pair_plan(
        runtime_states_path=str(args.runtime_states),
        strategy_cards_path=str(args.strategy_cards),
        strategy_lineage_path=str(args.strategy_lineage),
        selected_seed_sources_path=str(args.selected_seed_sources),
        generation_contract=generation_contract,
        generator_identity={
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "transport": endpoint.transport,
        },
        seed=args.seed,
        human_bank_review_passed=human_passed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "plan_report.json", payload["report"])
    write_jsonl(args.out_dir / "selected_states.jsonl", payload["selected_states"])
    write_jsonl(args.out_dir / "call_plan.jsonl", payload["call_plan"])
    if human_review is not None:
        write_json(args.out_dir / "human_review_binding.json", human_review)
    print(payload["report"])


if __name__ == "__main__":
    main()
