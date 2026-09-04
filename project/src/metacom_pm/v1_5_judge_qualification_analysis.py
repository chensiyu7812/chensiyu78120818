from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .io import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .pm_v2_contracts import ResponseDimensions, RiskDimensions


PREFERENCE_FIELDS = (
    "overall_preference",
    "support_quality_preference",
    "evidence_handling_preference",
    "safety_preference",
)
PREFERENCE_VALUES = frozenset({"A", "B", "tie", "insufficient"})
PAIRWISE_CONDITIONS = (
    "qwen_nonthinking_pairwise",
    "qwen_thinking_pairwise",
    "gpt_high_quality_pairwise_anchor",
)


class HumanQualificationAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    blind_item_id: str = Field(min_length=1)
    overall_preference: Literal["A", "B", "tie", "insufficient"]
    support_quality_preference: Literal["A", "B", "tie", "insufficient"]
    evidence_handling_preference: Literal["A", "B", "tie", "insufficient"]
    safety_preference: Literal["A", "B", "tie", "insufficient"]
    confidence: int = Field(ge=1, le=5)
    notes: str


def require_tracked_human_anchor(
    *,
    root: Path,
    binding_path: Path,
    pairs: Sequence[Mapping[str, Any]],
    human_packet: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen human anchor without treating it as automatic gold."""

    binding = dict(read_json(binding_path))
    binding_hash = str(binding.pop("binding_sha256", ""))
    expected_hash = sha256_text(canonical_json(binding))
    if not binding_hash or binding_hash != expected_hash:
        raise RuntimeError("human-anchor binding hash mismatch")
    if binding.get("protocol") != (
        "pm-v1.5-low-budget-judge-human-anchor-v1"
    ):
        raise RuntimeError("unexpected human-anchor binding protocol")
    if binding.get("status") != "FROZEN_BEFORE_MODEL_RESULTS":
        raise RuntimeError("human anchor was not frozen before model results")
    if binding.get("annotation_role") != (
        "independent_researcher_anchor_not_automatic_gold"
    ):
        raise RuntimeError("human-anchor role drifted")
    boundary = dict(binding.get("decision_boundary") or {})
    expected_boundary = {
        "may_be_used_as_training_labels": False,
        "may_auto_promote_bulk_labeler": False,
        "majority_vote_is_gold": False,
        "requires_researcher_signoff": True,
    }
    if boundary != expected_boundary:
        raise RuntimeError("human-anchor decision boundary drifted")

    annotation_record = dict(binding.get("annotations") or {})
    relative_path = Path(str(annotation_record.get("path") or ""))
    annotation_path = (root / relative_path).resolve()
    try:
        annotation_path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "human annotations must remain inside the project root"
        ) from exc
    if sha256_file(annotation_path) != str(
        annotation_record.get("sha256") or ""
    ):
        raise RuntimeError("tracked human annotations hash mismatch")
    annotation_rows = read_jsonl(annotation_path)
    if len(annotation_rows) != int(
        annotation_record.get("expected_rows") or 0
    ):
        raise RuntimeError("tracked human annotation count drifted")

    packet_record = dict(binding.get("human_blind_packet") or {})
    if len(human_packet) != int(packet_record.get("expected_rows") or 0):
        raise RuntimeError("human blind packet count drifted")
    if sha256_text(canonical_json(list(human_packet))) != str(
        packet_record.get("canonical_content_sha256") or ""
    ):
        raise RuntimeError("human blind packet content drifted")
    rendered_packet = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in human_packet
    )
    if sha256_text(rendered_packet) != str(
        packet_record.get("source_file_sha256") or ""
    ):
        raise RuntimeError("human blind packet serialized bytes drifted")

    normalized, validation = validate_human_annotations(
        pairs=pairs,
        human_packet=human_packet,
        annotation_rows=annotation_rows,
    )
    if validation["status"] != "COMPLETE" or len(normalized) != len(pairs):
        raise RuntimeError("human anchor is not complete")
    return {
        "protocol": str(binding["protocol"]),
        "status": str(binding["status"]),
        "annotation_role": str(binding["annotation_role"]),
        "binding_path": str(binding_path.resolve().relative_to(root.resolve())),
        "binding_file_sha256": sha256_file(binding_path),
        "binding_sha256": binding_hash,
        "annotations_path": str(relative_path),
        "annotations_file_sha256": sha256_file(annotation_path),
        "annotations": len(annotation_rows),
        "human_blind_packet_canonical_content_sha256": str(
            packet_record["canonical_content_sha256"]
        ),
        "human_blind_packet_file_sha256": str(
            packet_record["source_file_sha256"]
        ),
        "automatic_gold": False,
        "may_auto_promote_bulk_labeler": False,
        "requires_researcher_signoff": True,
    }


def flip_preference(value: str) -> str:
    if value == "A":
        return "B"
    if value == "B":
        return "A"
    if value in {"tie", "insufficient"}:
        return value
    raise ValueError(f"invalid preference: {value}")


def _normalized_pairwise(
    rows: Sequence[Mapping[str, Any]], *, condition: str
) -> dict[str, dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("condition") == condition]
    by_pair: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        record_ids = dict(row.get("record_ids") or {})
        pair_id = str(record_ids.get("pair_id") or "")
        order = int(record_ids.get("order_variant", -1))
        if not pair_id or order not in {0, 1} or order in by_pair[pair_id]:
            raise RuntimeError(f"invalid/repeated pairwise result: {condition}")
        parsed = dict(row.get("parsed") or {})
        if set(PREFERENCE_FIELDS) - set(parsed):
            raise RuntimeError(f"pairwise result fields are incomplete: {condition}")
        normalized = {
            field: (
                str(parsed[field])
                if order == 0
                else flip_preference(str(parsed[field]))
            )
            for field in PREFERENCE_FIELDS
        }
        if any(value not in PREFERENCE_VALUES for value in normalized.values()):
            raise RuntimeError(f"pairwise result preference is invalid: {condition}")
        by_pair[pair_id][order] = normalized
    result: dict[str, dict[str, Any]] = {}
    for pair_id, orders in by_pair.items():
        if set(orders) != {0, 1}:
            raise RuntimeError(
                f"pairwise AB/BA coverage is incomplete: {condition}/{pair_id}"
            )
        result[pair_id] = {
            field: {
                "order_0": orders[0][field],
                "order_1_normalized": orders[1][field],
                "order_consistent": orders[0][field] == orders[1][field],
                "canonical_preference": (
                    orders[0][field]
                    if orders[0][field] == orders[1][field]
                    else "order_disagreement"
                ),
            }
            for field in PREFERENCE_FIELDS
        }
    return result


def _pairwise_condition_report(
    rows: Sequence[Mapping[str, Any]], *, condition: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pairs = _normalized_pairwise(rows, condition=condition)
    dimensions: dict[str, Any] = {}
    for field in PREFERENCE_FIELDS:
        values = [pair[field] for pair in pairs.values()]
        consistent = sum(bool(row["order_consistent"]) for row in values)
        informative = sum(
            row["canonical_preference"] in {"A", "B"} for row in values
        )
        dimensions[field] = {
            "pairs": len(values),
            "order_consistent_pairs": consistent,
            "order_consistency_rate": consistent / len(values) if values else None,
            "informative_consistent_pairs": informative,
            "informative_rate": informative / len(values) if values else None,
            "canonical_preference_counts": dict(
                sorted(
                    Counter(
                        str(row["canonical_preference"]) for row in values
                    ).items()
                )
            ),
        }
    return {
        "condition": condition,
        "pairs": len(pairs),
        "dimensions": dimensions,
    }, pairs


def _direction_agreement(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    eligible_pair_ids: set[str] | None = None,
) -> dict[str, Any]:
    ids = set(left) & set(right)
    if eligible_pair_ids is not None:
        ids &= set(eligible_pair_ids)
    per_dimension: dict[str, Any] = {}
    for field in PREFERENCE_FIELDS:
        comparable: list[tuple[str, str]] = []
        for pair_id in sorted(ids):
            left_value = str(left[pair_id][field]["canonical_preference"])
            right_value = str(right[pair_id][field]["canonical_preference"])
            if left_value in {"A", "B"} and right_value in {"A", "B"}:
                comparable.append((left_value, right_value))
        agreements = sum(a == b for a, b in comparable)
        per_dimension[field] = {
            "eligible_pairs": len(ids),
            "comparable_pairs": len(comparable),
            "agreeing_pairs": agreements,
            "agreement_rate": (
                agreements / len(comparable) if comparable else None
            ),
        }
    return {"dimensions": per_dimension}


def _split_equivalence_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition") or "")
        if condition not in {
            "qwen_nonthinking_combined",
            "qwen_nonthinking_split_quality",
            "qwen_nonthinking_split_risk",
        }:
            continue
        record_ids = dict(row.get("record_ids") or {})
        equivalence_id = str(record_ids.get("equivalence_id") or "")
        if not equivalence_id or condition in by_id[equivalence_id]:
            raise RuntimeError("split-equivalence result coverage is malformed")
        by_id[equivalence_id][condition] = dict(row.get("parsed") or {})
    expected = {
        "qwen_nonthinking_combined",
        "qwen_nonthinking_split_quality",
        "qwen_nonthinking_split_risk",
    }
    deltas: list[dict[str, Any]] = []
    response_fields = tuple(ResponseDimensions.model_fields)
    risk_fields = tuple(RiskDimensions.model_fields)
    for equivalence_id, conditions in sorted(by_id.items()):
        if set(conditions) != expected:
            raise RuntimeError(
                f"split-equivalence trio is incomplete: {equivalence_id}"
            )
        combined = conditions["qwen_nonthinking_combined"]
        quality = conditions["qwen_nonthinking_split_quality"]
        risk = conditions["qwen_nonthinking_split_risk"]
        for field in response_fields:
            deltas.append(
                {
                    "equivalence_id": equivalence_id,
                    "dimension": f"response.{field}",
                    "absolute_delta": abs(
                        float(combined[field]) - float(quality[field])
                    ),
                }
            )
        for field in risk_fields:
            deltas.append(
                {
                    "equivalence_id": equivalence_id,
                    "dimension": f"risk.{field}",
                    "absolute_delta": abs(
                        float(combined[field]) - float(risk[field])
                    ),
                }
            )
    values = [float(row["absolute_delta"]) for row in deltas]
    return {
        "equivalence_items": len(by_id),
        "dimension_comparisons": len(values),
        "mean_absolute_error": fmean(values) if values else None,
        "within_half_point_count": sum(value <= 0.5 for value in values),
        "within_half_point_rate": (
            sum(value <= 0.5 for value in values) / len(values)
            if values
            else None
        ),
        "per_dimension": {
            dimension: {
                "n": len(dimension_values),
                "mean_absolute_error": fmean(dimension_values),
                "within_half_point_rate": (
                    sum(value <= 0.5 for value in dimension_values)
                    / len(dimension_values)
                ),
            }
            for dimension in sorted(
                {str(row["dimension"]) for row in deltas}
            )
            for dimension_values in [
                [
                    float(row["absolute_delta"])
                    for row in deltas
                    if row["dimension"] == dimension
                ]
            ]
        },
    }


def _human_pair_mapping(
    *,
    pairs: Sequence[Mapping[str, Any]],
    human_packet: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_orders: dict[str, tuple[str, int]] = {}
    pair_ids: set[str] = set()
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        if not pair_id or pair_id in pair_ids:
            raise RuntimeError("frozen pairs contain a missing/repeated pair_id")
        pair_ids.add(pair_id)
        original = sha256_text(
            canonical_json(
                {
                    "candidate_a": pair["candidate_a"],
                    "candidate_b": pair["candidate_b"],
                }
            )
        )
        reversed_hash = sha256_text(
            canonical_json(
                {
                    "candidate_a": pair["candidate_b"],
                    "candidate_b": pair["candidate_a"],
                }
            )
        )
        if original == reversed_hash:
            raise RuntimeError("frozen pair has indistinguishable candidates")
        if original in candidate_orders or reversed_hash in candidate_orders:
            raise RuntimeError("frozen pairs contain repeated candidate content")
        candidate_orders[original] = (pair_id, 0)
        candidate_orders[reversed_hash] = (pair_id, 1)
    mapping: dict[str, dict[str, Any]] = {}
    for item in human_packet:
        blind_id = str(item["blind_item_id"])
        digest = sha256_text(
            canonical_json(
                {
                    "candidate_a": item["candidate_a"],
                    "candidate_b": item["candidate_b"],
                }
            )
        )
        match = candidate_orders.get(digest)
        if match is None or blind_id in mapping:
            raise RuntimeError("human packet cannot be mapped to frozen pairs")
        mapping[blind_id] = {
            "pair_id": match[0],
            "order_variant": match[1],
        }
    mapped_pair_ids = {
        str(row["pair_id"]) for row in mapping.values()
    }
    if len(mapping) != len(pairs) or mapped_pair_ids != pair_ids:
        raise RuntimeError("human packet does not cover every frozen pair")
    return mapping


def validate_human_annotations(
    *,
    pairs: Sequence[Mapping[str, Any]],
    human_packet: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = _human_pair_mapping(pairs=pairs, human_packet=human_packet)
    parsed = [
        HumanQualificationAnnotation.model_validate(row).model_dump(
            mode="json"
        )
        for row in annotation_rows
    ]
    by_id = {str(row["blind_item_id"]): row for row in parsed}
    if len(by_id) != len(parsed) or set(by_id) != set(mapping):
        raise RuntimeError(
            "human annotations must cover each blind item exactly once"
        )
    normalized: list[dict[str, Any]] = []
    for blind_id in sorted(by_id):
        row = by_id[blind_id]
        match = mapping[blind_id]
        order = int(match["order_variant"])
        normalized.append(
            {
                **row,
                "pair_id": str(match["pair_id"]),
                "order_variant": order,
                **{
                    f"canonical_{field}": (
                        str(row[field])
                        if order == 0
                        else flip_preference(str(row[field]))
                    )
                    for field in PREFERENCE_FIELDS
                },
            }
        )
    return normalized, {
        "status": "COMPLETE",
        "annotations": len(normalized),
        "mean_confidence": fmean(
            int(row["confidence"]) for row in normalized
        ),
        "preference_counts": {
            field: dict(
                sorted(
                    Counter(
                        str(row[f"canonical_{field}"])
                        for row in normalized
                    ).items()
                )
            )
            for field in PREFERENCE_FIELDS
        },
    }


def aggregate_qualification(
    *,
    result_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    gpt_anchor_pair_ids: set[str],
    human_normalized_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if len(result_rows) != int(contract["planned_logical_calls"]):
        raise RuntimeError("qualification results are incomplete")
    expected_conditions = {
        str(key): int(value)
        for key, value in dict(contract["conditions"]).items()
    }
    observed_conditions = Counter(
        str(row.get("condition") or "") for row in result_rows
    )
    if observed_conditions != Counter(expected_conditions):
        raise RuntimeError("qualification condition coverage drifted")
    pairwise_reports: dict[str, Any] = {}
    pairwise: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in PAIRWISE_CONDITIONS:
        report, normalized = _pairwise_condition_report(
            result_rows, condition=condition
        )
        pairwise_reports[condition] = report
        pairwise[condition] = normalized
    mode_agreement = _direction_agreement(
        pairwise["qwen_nonthinking_pairwise"],
        pairwise["qwen_thinking_pairwise"],
    )
    qwen_gpt = {
        mode: _direction_agreement(
            pairwise[mode],
            pairwise["gpt_high_quality_pairwise_anchor"],
            eligible_pair_ids=gpt_anchor_pair_ids,
        )
        for mode in (
            "qwen_nonthinking_pairwise",
            "qwen_thinking_pairwise",
        )
    }
    split = _split_equivalence_report(result_rows)
    thresholds = dict(contract["pre_outcome_qualification_thresholds"])
    automatic_checks = {
        "qwen_nonthinking_informative": (
            float(
                pairwise_reports["qwen_nonthinking_pairwise"]["dimensions"][
                    "overall_preference"
                ]["informative_rate"]
            )
            >= float(thresholds["minimum_pairwise_informative_rate"])
        ),
        "qwen_thinking_informative": (
            float(
                pairwise_reports["qwen_thinking_pairwise"]["dimensions"][
                    "overall_preference"
                ]["informative_rate"]
            )
            >= float(thresholds["minimum_pairwise_informative_rate"])
        ),
        "all_qwen_order_consistency": all(
            float(dimension["order_consistency_rate"])
            >= float(thresholds["minimum_ab_ba_order_consistency"])
            for condition in (
                "qwen_nonthinking_pairwise",
                "qwen_thinking_pairwise",
            )
            for dimension in pairwise_reports[condition][
                "dimensions"
            ].values()
        ),
        "qwen_mode_overall_agreement": (
            mode_agreement["dimensions"]["overall_preference"][
                "agreement_rate"
            ]
            is not None
            and float(
                mode_agreement["dimensions"]["overall_preference"][
                    "agreement_rate"
                ]
            )
            >= float(thresholds["minimum_qwen_mode_direction_agreement"])
        ),
        "at_least_one_qwen_mode_matches_gpt_anchor": any(
            report["dimensions"]["overall_preference"]["agreement_rate"]
            is not None
            and float(
                report["dimensions"]["overall_preference"][
                    "agreement_rate"
                ]
            )
            >= float(
                thresholds[
                    "minimum_qwen_gpt_anchor_direction_agreement"
                ]
            )
            for report in qwen_gpt.values()
        ),
        "combined_split_mae": (
            split["mean_absolute_error"] is not None
            and float(split["mean_absolute_error"])
            <= float(
                thresholds[
                    "maximum_combined_vs_split_dimension_mae"
                ]
            )
        ),
        "combined_split_within_half_point": (
            split["within_half_point_rate"] is not None
            and float(split["within_half_point_rate"])
            >= float(
                thresholds[
                    "minimum_combined_vs_split_within_half_point_rate"
                ]
            )
        ),
    }
    human_report: dict[str, Any]
    if human_normalized_rows is None:
        human_report = {"status": "MISSING_REQUIRED_HUMAN_ANCHOR"}
    else:
        human_by_pair = {
            str(row["pair_id"]): row for row in human_normalized_rows
        }
        expected_human_pair_ids = set(
            pairwise["qwen_nonthinking_pairwise"]
        )
        if (
            len(human_by_pair) != len(human_normalized_rows)
            or set(human_by_pair) != expected_human_pair_ids
        ):
            raise RuntimeError(
                "human anchors must cover every frozen pair exactly once"
            )
        human_report = {
            "status": "COMPLETE_REQUIRES_RESEARCHER_SIGNOFF",
            "annotations": len(human_by_pair),
            "agreement_by_qwen_mode": {
                mode: {
                    field: {
                        "comparable_pairs": len(comparable),
                        "agreement_rate": (
                            sum(a == b for a, b in comparable)
                            / len(comparable)
                            if comparable
                            else None
                        ),
                    }
                    for field in PREFERENCE_FIELDS
                    for comparable in [
                        [
                            (
                                str(
                                    pairwise[mode][pair_id][field][
                                        "canonical_preference"
                                    ]
                                ),
                                str(
                                    human_by_pair[pair_id][
                                        f"canonical_{field}"
                                    ]
                                ),
                            )
                            for pair_id in sorted(
                                set(pairwise[mode]) & set(human_by_pair)
                            )
                            if str(
                                pairwise[mode][pair_id][field][
                                    "canonical_preference"
                                ]
                            )
                            in {"A", "B"}
                            and str(
                                human_by_pair[pair_id][f"canonical_{field}"]
                            )
                            in {"A", "B"}
                        ]
                    ]
                }
                for mode in (
                    "qwen_nonthinking_pairwise",
                    "qwen_thinking_pairwise",
                )
            },
            "automatic_human_threshold_pre_registered": False,
            "may_auto_promote_labeler": False,
        }
    if not all(automatic_checks.values()):
        status = "JUDGE_QUALIFICATION_NOT_SUPPORTED"
    elif human_normalized_rows is None:
        status = "AUTOMATIC_METRICS_PASS_HUMAN_ANCHOR_REQUIRED"
    else:
        status = "AUTOMATIC_METRICS_PASS_REQUIRES_HUMAN_SIGNOFF"
    return {
        "protocol": str(contract["protocol"]),
        "status": status,
        "pairwise": pairwise_reports,
        "qwen_mode_direction_agreement": mode_agreement,
        "qwen_gpt_anchor_direction_agreement": qwen_gpt,
        "combined_vs_split_equivalence": split,
        "automatic_checks": automatic_checks,
        "human_anchor": human_report,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
    }
