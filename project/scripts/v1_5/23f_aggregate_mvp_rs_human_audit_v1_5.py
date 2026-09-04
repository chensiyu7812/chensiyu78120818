#!/usr/bin/env python3
"""Validate and aggregate completed minimum-RS human audit annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_mvp_human_audit_results import analyze_human_audit


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _json_stream(path: Path) -> list[dict[str, Any]]:
    """Read adjacent JSON objects as well as conventional JSONL."""

    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise ValueError("human annotation stream must contain JSON objects")
        rows.append(dict(value))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_candidate",
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_judging_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_analysis",
    )
    args = parser.parse_args()

    packet_path = args.audit_dir / "human_blind_packet.jsonl"
    key_path = args.audit_dir / "private_blinding_key.jsonl"
    quality_path = args.judge_dir / "resolved_quality_pairs.jsonl"
    risk_path = args.judge_dir / "atomic_risk_events.jsonl"
    annotations = _json_stream(args.annotations)
    result = analyze_human_audit(
        annotations=annotations,
        packet=_rows(packet_path),
        private_key=_rows(key_path),
        llm_quality_rows=_rows(quality_path),
        llm_risk_rows=_rows(risk_path),
    )
    analysis = dict(result["analysis"])
    analysis["lineage"] = {
        "annotations_path": str(args.annotations.resolve()),
        "annotations_sha256": sha256_file(args.annotations),
        "packet_sha256": sha256_file(packet_path),
        "private_blinding_key_sha256": sha256_file(key_path),
        "llm_quality_sha256": sha256_file(quality_path),
        "llm_risk_sha256": sha256_file(risk_path),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "received_human_annotations.jsonl", annotations)
    write_jsonl(
        args.out_dir / "human_unblinded_quality.jsonl",
        result["unblinded_quality_rows"],
    )
    write_jsonl(
        args.out_dir / "human_unblinded_risk.jsonl",
        result["unblinded_risk_rows"],
    )
    write_json(args.out_dir / "human_audit_analysis.json", analysis)
    print(canonical_json({"output": str(args.out_dir), **analysis}))


if __name__ == "__main__":
    main()

