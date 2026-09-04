#!/usr/bin/env python3
"""Run the zero-API, train-only human-anchor learnability diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.v1_5_support_need_learnability import (
    run_human_anchor_learnability_diagnostic,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--normalized-anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    runtime = require_semantic_runtime_contract(config, encoder)
    report = run_human_anchor_learnability_diagnostic(
        encoder=encoder,
        packet_rows=[
            dict(row)
            for row in iter_jsonl(
                args.packet_dir / "human_blind_packet.jsonl"
            )
        ],
        normalized_anchor_rows=[
            dict(row) for row in iter_jsonl(args.normalized_anchors)
        ],
    )
    report["source_lineage"] = {
        "config_sha256": sha256_file(args.config),
        "packet_contract_sha256": sha256_file(
            args.packet_dir / "qualification_contract.json"
        ),
        "human_blind_packet_file_sha256": sha256_file(
            args.packet_dir / "human_blind_packet.jsonl"
        ),
        "normalized_anchors_sha256": sha256_file(args.normalized_anchors),
        "semantic_runtime_contract_sha256": runtime["contract_sha256"],
    }
    write_json(args.out, report)


if __name__ == "__main__":
    main()
