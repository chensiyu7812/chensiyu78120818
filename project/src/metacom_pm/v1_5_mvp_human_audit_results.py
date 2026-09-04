"""Validate, unblind, and summarize the minimum-RS human sanity audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import comb
from typing import Any

from .v1_5_mvp_human_audit import MVP_HUMAN_AUDIT_PROTOCOL
from .v1_5_mvp_judge import RISK_IDS


QUALITY_PREFERENCES = {"A", "B", "tie", "uncertain"}
QUALITY_CRITERIA = {
    "request_fit",
    "emotional_attunement",
    "immediate_helpfulness",
    "clarity_naturalness",
    "materially_equivalent",
    "uncertain",
}
APPLICABLE_RISK_JUDGMENTS = {"yes", "no", "uncertain"}


def _two_sided_sign_test(*, positive: int, negative: int) -> float | None:
    """Exact two-sided binomial sign test, with ties omitted."""

    total = positive + negative
    if total == 0:
        return None
    tail = sum(comb(total, k) for k in range(0, min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _count(rows: Sequence[str], values: Sequence[str]) -> dict[str, int]:
    counts = Counter(rows)
    return {value: counts.get(value, 0) for value in values}


def analyze_human_audit(
    *,
    annotations: Sequence[Mapping[str, Any]],
    packet: Sequence[Mapping[str, Any]],
    private_key: Sequence[Mapping[str, Any]],
    llm_quality_rows: Sequence[Mapping[str, Any]],
    llm_risk_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return validated unblinded rows and a deliberately bounded analysis."""

    packet_by_id = {str(row["blind_item_id"]): dict(row) for row in packet}
    key_by_id = {str(row["blind_item_id"]): dict(row) for row in private_key}
    if len(packet_by_id) != len(packet) or len(key_by_id) != len(private_key):
        raise ValueError("packet and private key must have unique blind_item_id values")
    if set(packet_by_id) != set(key_by_id):
        raise ValueError("packet and private key blind_item_id sets differ")

    annotations_by_id: dict[str, dict[str, Any]] = {}
    for raw in annotations:
        row = dict(raw)
        if row.get("protocol") != MVP_HUMAN_AUDIT_PROTOCOL:
            raise ValueError("unexpected human-audit protocol")
        blind_id = str(row.get("blind_item_id", ""))
        if blind_id in annotations_by_id:
            raise ValueError(f"duplicate annotation for {blind_id}")
        annotations_by_id[blind_id] = row
    if set(annotations_by_id) != set(packet_by_id):
        missing = sorted(set(packet_by_id) - set(annotations_by_id))
        extra = sorted(set(annotations_by_id) - set(packet_by_id))
        raise ValueError(f"annotation coverage mismatch: missing={missing}, extra={extra}")

    llm_quality_by_pair = {
        str(row["pair_id"]): dict(row) for row in llm_quality_rows
    }
    llm_risk_by_key = {
        (str(row["pair_id"]), str(row["arm"]), str(row["risk_id"])): dict(row)
        for row in llm_risk_rows
    }

    quality_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    annotator_ids: set[str] = set()
    for blind_id in sorted(packet_by_id):
        annotation = annotations_by_id[blind_id]
        item = packet_by_id[blind_id]
        key = key_by_id[blind_id]
        preference = annotation.get("quality_preference")
        criterion = annotation.get("quality_decisive_criterion")
        if preference not in QUALITY_PREFERENCES:
            raise ValueError(f"{blind_id}: invalid or missing quality preference")
        if criterion not in QUALITY_CRITERIA:
            raise ValueError(f"{blind_id}: invalid or missing quality criterion")

        label_to_arm = {
            "A": str(key["response_a_arm"]),
            "B": str(key["response_b_arm"]),
        }
        human_arm_preference = (
            label_to_arm[str(preference)]
            if preference in {"A", "B"}
            else str(preference)
        )
        llm_row = llm_quality_by_pair.get(str(key["pair_id"]))
        if llm_row is None:
            raise ValueError(f"{blind_id}: missing LLM quality row")
        llm_preference = str(llm_row["preference"])
        quality_rows.append(
            {
                "blind_item_id": blind_id,
                "pair_id": str(key["pair_id"]),
                "state_id": str(key["state_id"]),
                "user_id": str(key["user_id"]),
                "boundary_cue": str(key["boundary_cue"]),
                "selected_strategy_family": str(key["selected_strategy_family"]),
                "human_blind_preference": preference,
                "human_arm_preference": human_arm_preference,
                "quality_decisive_criterion": criterion,
                "quality_notes": str(annotation.get("quality_notes", "")),
                "llm_arm_preference": llm_preference,
                "llm_orders_consistent": bool(llm_row["orders_consistent"]),
                "human_llm_exact_agreement": human_arm_preference == llm_preference,
            }
        )

        expected_response_ids = dict(item["blind_response_ids"])
        annotated_responses = {
            str(row["blind_response_id"]): dict(row)
            for row in annotation.get("responses", [])
        }
        if len(annotated_responses) != len(annotation.get("responses", [])):
            raise ValueError(f"{blind_id}: duplicate blind response id")
        if set(annotated_responses) != set(expected_response_ids.values()):
            raise ValueError(f"{blind_id}: response coverage mismatch")

        for label in ("A", "B"):
            response_id = str(expected_response_ids[label])
            response = annotated_responses[response_id]
            findings = dict(response.get("material_risk_findings", {}))
            if set(findings) != set(RISK_IDS):
                raise ValueError(f"{response_id}: risk coverage mismatch")
            arm = label_to_arm[label]
            for risk_id in RISK_IDS:
                finding = dict(findings[risk_id])
                applicable = bool(item["risk_applicability"][risk_id]["applicable"])
                judgment = finding.get("judgment")
                allowed = (
                    APPLICABLE_RISK_JUDGMENTS
                    if applicable
                    else {"not_applicable"}
                )
                if judgment not in allowed:
                    raise ValueError(
                        f"{response_id}/{risk_id}: invalid judgment {judgment!r}"
                    )
                if judgment == "yes" and not str(
                    finding.get("evidence_excerpt", "")
                ).strip():
                    raise ValueError(
                        f"{response_id}/{risk_id}: yes requires an evidence excerpt"
                    )
                llm_risk = llm_risk_by_key.get(
                    (str(key["pair_id"]), arm, risk_id)
                )
                if llm_risk is None:
                    raise ValueError(
                        f"{response_id}/{risk_id}: missing LLM risk row"
                    )
                human_material = (
                    True
                    if judgment == "yes"
                    else False
                    if judgment == "no"
                    else None
                )
                llm_material = (
                    bool(llm_risk["material_event"]) if applicable else None
                )
                risk_rows.append(
                    {
                        "blind_item_id": blind_id,
                        "blind_response_id": response_id,
                        "pair_id": str(key["pair_id"]),
                        "user_id": str(key["user_id"]),
                        "boundary_cue": str(key["boundary_cue"]),
                        "selected_strategy_family": str(
                            key["selected_strategy_family"]
                        ),
                        "arm": arm,
                        "risk_id": risk_id,
                        "programmatically_applicable": applicable,
                        "human_judgment": judgment,
                        "human_material_event": human_material,
                        "human_evidence_excerpt": str(
                            finding.get("evidence_excerpt", "")
                        ),
                        "human_notes": str(finding.get("notes", "")),
                        "llm_material_event": llm_material,
                        "human_llm_agreement": (
                            human_material == llm_material
                            if human_material is not None
                            and llm_material is not None
                            else None
                        ),
                    }
                )

        annotator_id = str(annotation.get("annotator_id", "")).strip()
        if annotator_id:
            annotator_ids.add(annotator_id)

    human_preferences = [row["human_arm_preference"] for row in quality_rows]
    llm_preferences = [row["llm_arm_preference"] for row in quality_rows]
    human_counts = _count(human_preferences, ["RS", "R0", "tie", "uncertain"])
    llm_counts = _count(llm_preferences, ["RS", "R0", "tie", "abstain"])
    by_cue: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for field, target in (
        ("boundary_cue", by_cue),
        ("selected_strategy_family", by_family),
    ):
        values = sorted({str(row[field]) for row in quality_rows})
        for value in values:
            target[value] = _count(
                [
                    str(row["human_arm_preference"])
                    for row in quality_rows
                    if row[field] == value
                ],
                ["RS", "R0", "tie", "uncertain"],
            )

    exact_agreement = sum(
        bool(row["human_llm_exact_agreement"]) for row in quality_rows
    )
    llm_decisive = [
        row for row in quality_rows if row["llm_arm_preference"] in {"RS", "R0"}
    ]
    decisive_agreement = sum(
        bool(row["human_llm_exact_agreement"]) for row in llm_decisive
    )

    human_risk_counts: dict[str, dict[str, dict[str, int]]] = {}
    risk_agreement: dict[str, dict[str, Any]] = {}
    for risk_id in RISK_IDS:
        human_risk_counts[risk_id] = {}
        for arm in ("R0", "RS"):
            values = [
                str(row["human_judgment"])
                for row in risk_rows
                if row["risk_id"] == risk_id and row["arm"] == arm
            ]
            human_risk_counts[risk_id][arm] = _count(
                values, ["yes", "no", "uncertain", "not_applicable"]
            )
        comparable = [
            row
            for row in risk_rows
            if row["risk_id"] == risk_id
            and row["human_llm_agreement"] is not None
        ]
        risk_agreement[risk_id] = {
            "comparable_cells": len(comparable),
            "agreements": sum(
                bool(row["human_llm_agreement"]) for row in comparable
            ),
            "agreement_rate": (
                sum(bool(row["human_llm_agreement"]) for row in comparable)
                / len(comparable)
                if comparable
                else None
            ),
            "human_material_llm_nonmaterial": sum(
                row["human_material_event"] is True
                and row["llm_material_event"] is False
                for row in comparable
            ),
            "human_nonmaterial_llm_material": sum(
                row["human_material_event"] is False
                and row["llm_material_event"] is True
                for row in comparable
            ),
        }

    criterion_counts = Counter(
        str(row["quality_decisive_criterion"]) for row in quality_rows
    )
    analysis = {
        "protocol": "pm-v1.5-minimum-rs-human-audit-analysis-v1",
        "status": "HUMAN_SANITY_SIGNAL_ONLY_NOT_PM_TRAINING_LABELS",
        "data_quality": {
            "expected_pairs": len(packet_by_id),
            "received_pairs": len(annotations_by_id),
            "unique_blind_items": len(annotations_by_id),
            "responses_reviewed": len(annotations_by_id) * 2,
            "risk_cells_reviewed": len(risk_rows),
            "complete": True,
            "nonempty_annotator_ids": sorted(annotator_ids),
            "known_annotator_count": len(annotator_ids),
            "interrater_reliability_available": False,
        },
        "quality": {
            "human_arm_preferences": human_counts,
            "by_boundary_cue": by_cue,
            "by_strategy_family": by_family,
            "decisive_criterion_counts": dict(sorted(criterion_counts.items())),
            "exact_sign_test_rs_vs_r0_p_value": _two_sided_sign_test(
                positive=human_counts["RS"],
                negative=human_counts["R0"],
            ),
            "llm_arm_preferences_on_same_pairs": llm_counts,
            "human_llm_exact_agreement": {
                "pairs": len(quality_rows),
                "agreements": exact_agreement,
                "rate": exact_agreement / len(quality_rows),
            },
            "human_llm_agreement_when_llm_decisive": {
                "pairs": len(llm_decisive),
                "agreements": decisive_agreement,
                "rate": (
                    decisive_agreement / len(llm_decisive)
                    if llm_decisive
                    else None
                ),
            },
        },
        "risk": {
            "human_judgment_counts": human_risk_counts,
            "human_llm_material_event_agreement": risk_agreement,
        },
        "training_readiness": {
            "authorized": False,
            "reasons": [
                "single human annotation set with no interrater estimate",
                "all 12 quality choices are forced A/B decisions; no tie or uncertainty",
                "only six advice_welcome pairs are present",
                "the LLM quality judge failed its frozen AB/BA qualification gate",
                "the audit was designed as a sanity check, not a training-label sample",
            ],
            "safe_uses": [
                "descriptive evidence that conditional RS benefit is plausible",
                "identify advice_welcome plus Providing Suggestions as the next RS stratum",
                "calibrate and redesign the outcome measurement instrument",
            ],
            "forbidden_uses": [
                "treat all 12 forced preferences as hard PM labels",
                "claim overall RS superiority",
                "claim clinical benefit or external generalization",
            ],
        },
    }
    return {
        "analysis": analysis,
        "unblinded_quality_rows": quality_rows,
        "unblinded_risk_rows": risk_rows,
    }
