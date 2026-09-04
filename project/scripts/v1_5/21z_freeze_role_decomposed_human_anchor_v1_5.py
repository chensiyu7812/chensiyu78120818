#!/usr/bin/env python3
"""Validate and freeze the pre-API role-decomposed human anchor (zero API)."""

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
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    EvidenceRiskFinding,
    evidence_excerpt_is_exact,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _persist_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.is_file() and _rows(path) != rows:
        raise RuntimeError(f"existing frozen annotation artifact drifted: {path}")
    write_jsonl(path, rows)


def _persist_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError(f"existing frozen contract artifact drifted: {path}")
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/role_decomposed_judge_packet_v1",
    )
    parser.add_argument("--completed-annotations", type=Path, required=True)
    parser.add_argument(
        "--annotations-out",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_human_annotations_v1.jsonl",
    )
    parser.add_argument(
        "--binding-out",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--contract-out",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_qualification_v1.json",
    )
    args = parser.parse_args()

    packet_path = args.packet_dir / "human_blind_packet.jsonl"
    template_path = args.packet_dir / "human_annotation_template.jsonl"
    mapping_path = args.packet_dir / "human_anchor_mapping_internal.jsonl"
    preparation_contract_path = args.packet_dir / "qualification_contract.json"
    packet = _rows(packet_path)
    template = _rows(template_path)
    mapping = _rows(mapping_path)
    annotations = _rows(args.completed_annotations)
    preparation = read_json(preparation_contract_path)
    if preparation.get("status") != "AWAITING_PRE_API_HUMAN_ANCHOR":
        raise RuntimeError("preparation contract is not awaiting a human anchor")
    if (
        len(packet) != 12
        or len(template) != 12
        or len(mapping) != 12
        or len(annotations) != 12
    ):
        raise RuntimeError("human anchor must contain exactly 12 rows")
    packet_ids = [str(row["blind_item_id"]) for row in packet]
    if [str(row["blind_item_id"]) for row in template] != packet_ids:
        raise RuntimeError("human template order drifted from packet")
    if [str(row["blind_item_id"]) for row in annotations] != packet_ids:
        raise RuntimeError("completed annotation order drifted from packet")
    if [str(row["blind_item_id"]) for row in mapping] != packet_ids:
        raise RuntimeError("human mapping order drifted from packet")
    if sha256_text(canonical_json(mapping)) != preparation[
        "source_lineage"
    ]["human_mapping_sha256"]:
        raise RuntimeError("human mapping canonical content drifted")
    if sha256_text(canonical_json(packet)) != preparation["source_lineage"][
        "human_packet_sha256"
    ]:
        raise RuntimeError("human packet canonical content drifted")

    expected_top_keys = {
        "blind_item_id",
        "overall_preference",
        "support_quality_preference",
        "quality_confidence",
        "quality_notes",
        "candidate_a_audits",
        "candidate_b_audits",
    }
    normalized: list[dict[str, Any]] = []
    for packet_row, template_row, raw in zip(packet, template, annotations):
        blind_id = str(packet_row["blind_item_id"])
        if set(raw) != expected_top_keys:
            raise RuntimeError(f"annotation top-level schema drifted: {blind_id}")
        if raw["overall_preference"] not in {"A", "B", "tie", "insufficient"}:
            raise RuntimeError(f"invalid overall preference: {blind_id}")
        if raw["support_quality_preference"] not in {
            "A",
            "B",
            "tie",
            "insufficient",
        }:
            raise RuntimeError(f"invalid support preference: {blind_id}")
        confidence = raw["quality_confidence"]
        if not isinstance(confidence, int) or not 1 <= confidence <= 5:
            raise RuntimeError(f"invalid human confidence: {blind_id}")
        if not str(raw["quality_notes"]).strip():
            raise RuntimeError(f"blank human quality notes: {blind_id}")
        normalized_row = {
            "blind_item_id": blind_id,
            "overall_preference": str(raw["overall_preference"]),
            "support_quality_preference": str(
                raw["support_quality_preference"]
            ),
            "quality_confidence": confidence,
            "quality_notes": str(raw["quality_notes"]),
        }
        for candidate_name in ("a", "b"):
            key = f"candidate_{candidate_name}_audits"
            expected_dimensions = [
                str(item["dimension"]) for item in template_row[key]
            ]
            raw_findings = list(raw[key])
            if [str(item.get("dimension")) for item in raw_findings] != (
                expected_dimensions
            ):
                raise RuntimeError(
                    f"human audit dimensions drifted: {blind_id}/{candidate_name}"
                )
            response = str(
                packet_row[f"candidate_{candidate_name}"]["response"]
            )
            evidence_surface = {
                "visible_state": packet_row["visible_state"],
                "authorized_user_context": packet_row[
                    "authorized_user_context"
                ],
                "selected_context": packet_row[
                    f"candidate_{candidate_name}"
                ]["selected_context"],
            }
            findings: list[dict[str, Any]] = []
            for raw_finding in raw_findings:
                finding = EvidenceRiskFinding.model_validate(
                    raw_finding, strict=True
                )
                if (
                    finding.response_excerpt not in {"[none]", "[omission]"}
                    and finding.response_excerpt not in response
                ):
                    raise RuntimeError(
                        f"human response quote is not exact: "
                        f"{blind_id}/{candidate_name}/{finding.dimension}"
                    )
                if (
                    not evidence_excerpt_is_exact(
                        finding.evidence_excerpt, evidence_surface
                    )
                ):
                    raise RuntimeError(
                        f"human evidence quote is not exact: "
                        f"{blind_id}/{candidate_name}/{finding.dimension}"
                    )
                findings.append(finding.model_dump(mode="json"))
            normalized_row[key] = findings
        normalized.append(normalized_row)

    args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
    _persist_rows(args.annotations_out, normalized)
    annotation_file_sha = sha256_file(args.annotations_out)
    binding_payload = {
        "protocol": "pm-v1.5-role-decomposed-human-anchor-v1",
        "status": "FROZEN_BEFORE_MODEL_CALLS",
        "annotation_role": (
            "single_researcher_independent_reference_not_automatic_gold"
        ),
        "preparation_contract_sha256": str(preparation["contract_sha256"]),
        "human_blind_packet_path": str(packet_path.relative_to(ROOT)),
        "human_blind_packet_file_sha256": sha256_file(packet_path),
        "human_blind_packet_canonical_sha256": sha256_text(
            canonical_json(packet)
        ),
        "human_annotation_template_path": str(template_path.relative_to(ROOT)),
        "human_annotation_template_file_sha256": sha256_file(template_path),
        "human_anchor_mapping_path": str(mapping_path.relative_to(ROOT)),
        "human_anchor_mapping_file_sha256": sha256_file(mapping_path),
        "human_anchor_mapping_canonical_sha256": sha256_text(
            canonical_json(mapping)
        ),
        "annotations_path": str(args.annotations_out.relative_to(ROOT)),
        "annotations_file_sha256": annotation_file_sha,
        "annotations_canonical_sha256": sha256_text(
            canonical_json(normalized)
        ),
        "annotations": len(normalized),
        "automatic_gold": False,
        "may_auto_promote_bulk_labeler": False,
        "requires_researcher_signoff": True,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    binding = {
        **binding_payload,
        "binding_sha256": sha256_text(canonical_json(binding_payload)),
    }
    _persist_json(args.binding_out, binding)

    activated_payload = {
        key: value
        for key, value in preparation.items()
        if key not in {"status", "contract_sha256"}
    }
    activated_payload.update(
        {
            "status": "READY_FOR_ZERO_API_DRY_RUN",
            "preparation_contract_sha256": str(
                preparation["contract_sha256"]
            ),
            "human_anchor_binding_path": str(
                args.binding_out.relative_to(ROOT)
            ),
            "human_anchor_binding_file_sha256": sha256_file(args.binding_out),
            "human_anchor_binding_sha256": binding["binding_sha256"],
        }
    )
    activated = {
        **activated_payload,
        "contract_sha256": sha256_text(canonical_json(activated_payload)),
    }
    _persist_json(args.contract_out, activated)
    print(
        canonical_json(
            {
                "status": "FROZEN_BEFORE_MODEL_CALLS",
                "annotations_file_sha256": annotation_file_sha,
                "human_anchor_binding_sha256": binding["binding_sha256"],
                "qualification_contract_sha256": activated[
                    "contract_sha256"
                ],
                "api_calls_made": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
