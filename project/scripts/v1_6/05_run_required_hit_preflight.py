#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.pm_v1_6_preflight import build_preflight

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.pm_config)
    if config.get("version") != "pm-v1.6":
        raise RuntimeError("required-hit preflight requires PM-v1.6")
    supporter = SupporterGenerationContract.from_config(config)
    retrieval = config["retrieval"]
    summary, rows = build_preflight(
        runtime_path=args.runtime,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        strategy_bank_path=args.strategy_bank,
        supporter_contract=supporter,
        memory_min_score=float(retrieval["memory_min_score"]),
        strategy_min_score=float(retrieval["strategy_min_score"]),
        strategy_top_k=int(retrieval["strategy_top_k"]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "action_preflight.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(rows_path, rows)
    summary = {
        **summary,
        "pm_config_sha256": sha256_file(args.pm_config),
        "action_preflight_sha256": sha256_file(rows_path),
    }
    write_json(summary_path, summary)
    if summary["status"] != "PASS":
        raise RuntimeError(
            "PM-v1.6 required-hit preflight failed before any response generation"
        )


if __name__ == "__main__":
    main()
