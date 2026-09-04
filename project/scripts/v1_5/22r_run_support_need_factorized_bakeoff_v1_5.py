#!/usr/bin/env python3
"""Run the zero-API factorized SupportNeed representation bakeoff."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
)
from metacom_pm.v1_5_support_need_bakeoff import (
    FrozenSupportNeedEmbeddingEncoder,
    FrozenSupportNeedEmbeddingSpec,
    FrozenSupportNeedNLIScorer,
    FrozenSupportNeedNLISpec,
    candidate_runtime_attestation,
    embedding_candidate_feature_views,
    nli_candidate_feature_bundle,
    run_factorized_support_need_bakeoff,
    transparent_feature_views,
)


ROOT = Path(__file__).resolve().parents[2]


def _release_model() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bakeoff-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5_support_need_bakeoff_v1.json",
    )
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument(
        "--packet-file",
        type=Path,
        help=(
            "Optional packet JSONL override. Defaults to "
            "<packet-dir>/human_blind_packet.jsonl."
        ),
    )
    parser.add_argument(
        "--packet-contract",
        type=Path,
        help=(
            "Optional packet/anchor contract override. Defaults to "
            "<packet-dir>/qualification_contract.json."
        ),
    )
    parser.add_argument("--normalized-anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument(
        "--expansion-human-annotations-opened",
        action="store_true",
        help=(
            "Bind the report to the expanded train-only fit subset after "
            "human adjudication. The confirmation subset must remain sealed."
        ),
    )
    args = parser.parse_args()

    config = read_json(args.bakeoff_config)
    packet_file = (
        args.packet_file
        if args.packet_file is not None
        else args.packet_dir / "human_blind_packet.jsonl"
    )
    packet_contract = (
        args.packet_contract
        if args.packet_contract is not None
        else args.packet_dir / "qualification_contract.json"
    )
    packet_rows = [
        dict(row) for row in iter_jsonl(packet_file)
    ]
    normalized_anchors = [
        dict(row) for row in iter_jsonl(args.normalized_anchors)
    ]
    feature_views = transparent_feature_views(packet_rows)
    candidate_runtimes = []

    for raw_spec in config["embedding_candidates"]:
        spec = FrozenSupportNeedEmbeddingSpec.model_validate(raw_spec)
        encoder = FrozenSupportNeedEmbeddingEncoder.load(
            spec,
            device=args.device,
            batch_size=args.embedding_batch_size,
        )
        feature_views.update(
            embedding_candidate_feature_views(encoder, packet_rows)
        )
        candidate_runtimes.append(
            candidate_runtime_attestation(encoder.binding)
        )
        del encoder
        _release_model()

    nli_spec = FrozenSupportNeedNLISpec.model_validate(
        config["nli_candidate"]
    )
    scorer = FrozenSupportNeedNLIScorer.load(
        nli_spec,
        device=args.device,
        batch_size=args.nli_batch_size,
    )
    nli_bundle = nli_candidate_feature_bundle(scorer, packet_rows)
    feature_views.update(nli_bundle["feature_views"])
    candidate_runtimes.append(candidate_runtime_attestation(scorer.binding))
    del scorer
    _release_model()

    source_lineage = {
        "bakeoff_config_sha256": sha256_file(args.bakeoff_config),
        "packet_contract_sha256": sha256_file(packet_contract),
        "human_blind_packet_sha256": sha256_file(packet_file),
        "normalized_anchors_sha256": sha256_file(args.normalized_anchors),
    }
    report = run_factorized_support_need_bakeoff(
        packet_rows=packet_rows,
        normalized_anchor_rows=normalized_anchors,
        feature_views=feature_views,
        nli_bundle=nli_bundle,
        candidate_runtimes=candidate_runtimes,
        hybrid_embedding_candidate_id=config[
            "hybrid_embedding_candidate_id"
        ],
        source_lineage=source_lineage,
        expansion_human_annotations_opened=(
            args.expansion_human_annotations_opened
        ),
    )
    write_json(args.out, report)


if __name__ == "__main__":
    main()
