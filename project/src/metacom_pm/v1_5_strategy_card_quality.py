"""Bounded qualification audit for the minimum-RS strategy catalog."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_FAMILIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Affirmation and Reassurance",
    "Providing Suggestions",
)
HUMAN_CONTENT_FIELDS = (
    "support_move_clear",
    "when_to_use_valid",
    "when_not_to_use_valid",
    "mode_phase_goal_fit_valid",
    "burden_and_risk_flags_valid",
    "safe_general_technique",
)


def _preference_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["human_arm_preference"]) for row in rows)
    return {
        value: counts.get(value, 0)
        for value in ("RS", "R0", "tie", "uncertain")
    }


def audit_strategy_card_quality(
    *,
    cards: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    build_report: Mapping[str, Any],
    human_validation_report: Mapping[str, Any],
    selected_states: Sequence[Mapping[str, Any]],
    discovery_quality_rows: Sequence[Mapping[str, Any]],
    discovery_risk_rows: Sequence[Mapping[str, Any]],
    confirmation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine construction, human-content, and response evidence.

    This deliberately qualifies only the frozen five-card, family-gated Top-1
    treatment. It does not promote the historical raw-card Top-k bank.
    """

    card_by_id = {str(row["card_id"]): dict(row) for row in cards}
    annotation_by_id = {
        str(row["card_id"]): dict(row) for row in annotations
    }
    if len(card_by_id) != len(cards):
        raise ValueError("strategy card ids must be unique")
    if len(annotation_by_id) != len(annotations):
        raise ValueError("strategy annotation card ids must be unique")
    if set(card_by_id) != set(annotation_by_id):
        raise ValueError("strategy cards and human annotations differ")

    family_counts = Counter(
        str(row["strategy_family"]) for row in card_by_id.values()
    )
    exact_one_card_per_family = (
        set(family_counts) == set(EXPECTED_FAMILIES)
        and all(family_counts[family] == 1 for family in EXPECTED_FAMILIES)
    )
    source_construction_passed = all(
        (
            int(build_report.get("candidate_packet_source_overlap_count", -1))
            == 0,
            build_report.get("candidate_raw_examples_exposed_to_generator")
            is False,
            int(build_report.get("candidate_card_count", -1))
            == len(EXPECTED_FAMILIES),
            exact_one_card_per_family,
            all(
                row.get("content_scope") == "technique_only"
                for row in card_by_id.values()
            ),
        )
    )

    human_card_results: list[dict[str, Any]] = []
    for card_id in sorted(card_by_id):
        card = card_by_id[card_id]
        annotation = annotation_by_id[card_id]
        content_fields_passed = all(
            annotation.get(field) is True for field in HUMAN_CONTENT_FIELDS
        )
        approved = annotation.get("approve_for_train_only_pilot") is True
        confidence = int(annotation.get("confidence", 0))
        human_card_results.append(
            {
                "card_id": card_id,
                "strategy_family": str(card["strategy_family"]),
                "content_fields_passed": content_fields_passed,
                "approved_for_train_only_pilot": approved,
                "confidence": confidence,
                "reviewer_followup_recorded": bool(
                    str(annotation.get("required_corrections", "")).strip()
                ),
                "intrinsic_human_pass": (
                    content_fields_passed and approved and confidence >= 3
                ),
            }
        )
    intrinsic_human_passed = all(
        row["intrinsic_human_pass"] for row in human_card_results
    )

    selected_by_state: dict[str, dict[str, Any]] = {}
    for raw in selected_states:
        row = dict(raw)
        state_id = str(row["state_id"])
        if state_id in selected_by_state:
            raise ValueError(f"duplicate selected state: {state_id}")
        card_id = str(row["selected_strategy_card_id"])
        if card_id not in card_by_id:
            raise ValueError(f"selected unknown strategy card: {card_id}")
        selected_by_state[state_id] = row

    exposure_counts = Counter(
        str(row["selected_strategy_card_id"])
        for row in selected_by_state.values()
    )
    quality_by_card: dict[str, list[dict[str, Any]]] = {
        card_id: [] for card_id in card_by_id
    }
    for raw in discovery_quality_rows:
        row = dict(raw)
        state_id = str(row["state_id"])
        if state_id not in selected_by_state:
            raise ValueError(f"human quality row has unknown state: {state_id}")
        card_id = str(selected_by_state[state_id]["selected_strategy_card_id"])
        quality_by_card[card_id].append(row)

    rs_risk_by_card: dict[str, list[dict[str, Any]]] = {
        card_id: [] for card_id in card_by_id
    }
    for raw in discovery_risk_rows:
        row = dict(raw)
        if row.get("arm") != "RS":
            continue
        state_id = str(row.get("state_id", ""))
        if not state_id:
            # The unblinded risk rows identify pairs, not states. Recover the
            # state through the matching discovery quality row.
            pair_id = str(row["pair_id"])
            matching = [
                quality_row
                for rows in quality_by_card.values()
                for quality_row in rows
                if str(quality_row["pair_id"]) == pair_id
            ]
            if len(matching) != 1:
                raise ValueError(f"cannot bind risk pair to card: {pair_id}")
            state_id = str(matching[0]["state_id"])
        if state_id not in selected_by_state:
            raise ValueError(f"human risk row has unknown state: {state_id}")
        card_id = str(selected_by_state[state_id]["selected_strategy_card_id"])
        rs_risk_by_card[card_id].append(row)

    per_card_response_evidence: list[dict[str, Any]] = []
    all_rs_risk_rows: list[dict[str, Any]] = []
    for card_id in sorted(card_by_id):
        risk_rows = rs_risk_by_card[card_id]
        all_rs_risk_rows.extend(risk_rows)
        applicable = [
            row for row in risk_rows if row["programmatically_applicable"]
        ]
        material = [
            row for row in applicable if row["human_material_event"] is True
        ]
        per_card_response_evidence.append(
            {
                "card_id": card_id,
                "strategy_family": str(card_by_id[card_id]["strategy_family"]),
                "pilot_exposures": exposure_counts.get(card_id, 0),
                "discovery_human_pairs": len(quality_by_card[card_id]),
                "discovery_human_preferences": _preference_counts(
                    quality_by_card[card_id]
                ),
                "rs_applicable_risk_cells_reviewed": len(applicable),
                "rs_confirmed_material_risk_events": len(material),
            }
        )

    applicable_rs_risks = [
        row
        for row in all_rs_risk_rows
        if row["programmatically_applicable"]
    ]
    confirmed_rs_risks = [
        row
        for row in applicable_rs_risks
        if row["human_material_event"] is True
    ]
    preliminary_response_safety_passed = (
        bool(applicable_rs_risks) and not confirmed_rs_risks
    )
    confirmation_complete = str(
        confirmation_manifest.get("status", "")
    ).startswith("HUMAN_CONFIRMATION_COMPLETE")

    current_pilot_qualified = (
        source_construction_passed
        and intrinsic_human_passed
        and exact_one_card_per_family
        and human_validation_report.get("formal_rs_promoted") is False
    )
    return {
        "protocol": "pm-v1.5-minimum-rs-card-quality-audit-v1",
        "status": (
            "QUALIFIED_FOR_CURRENT_TRAIN_ONLY_TOP1_RS_PILOT_"
            "PENDING_RESPONSE_CONFIRMATION"
            if current_pilot_qualified and not confirmation_complete
            else "QUALIFICATION_REQUIRES_REVIEW"
        ),
        "object_being_qualified": {
            "description": (
                "five frozen technique-only cards, one per strategy family, "
                "with a family gate and one-card injection"
            ),
            "card_count": len(card_by_id),
            "retrieval_cardinality": "Top-1 after family eligibility",
            "historical_raw_topk_bank_qualified": False,
        },
        "construction_and_leakage": {
            "passed": source_construction_passed,
            "packet_source_overlap_count": int(
                build_report["candidate_packet_source_overlap_count"]
            ),
            "raw_examples_exposed_to_generator": bool(
                build_report["candidate_raw_examples_exposed_to_generator"]
            ),
            "raw_cards_screened": int(build_report["raw_card_count"]),
            "lineage_rows_retained": int(
                build_report["candidate_lineage_row_count"]
            ),
            "source_quality_statistics_role": (
                "conversation-level provenance description only; not a "
                "turn-level card utility label"
            ),
        },
        "intrinsic_human_review": {
            "passed_for_train_only_pilot": intrinsic_human_passed,
            "reviewed_cards": len(human_card_results),
            "approved_cards": sum(
                row["approved_for_train_only_pilot"]
                for row in human_card_results
            ),
            "minimum_confidence": min(
                row["confidence"] for row in human_card_results
            ),
            "cards_with_nonblocking_followups": sum(
                row["reviewer_followup_recorded"]
                for row in human_card_results
            ),
            "single_reviewer_only": True,
            "card_results": human_card_results,
        },
        "retrieval_and_coverage": {
            "exactly_one_card_per_family": exact_one_card_per_family,
            "pilot_state_count": len(selected_states),
            "families_exposed_in_pilot": sum(
                exposure_counts.get(card_id, 0) > 0 for card_id in card_by_id
            ),
            "families_not_exposed_in_pilot": sorted(
                str(card_by_id[card_id]["strategy_family"])
                for card_id in card_by_id
                if exposure_counts.get(card_id, 0) == 0
            ),
        },
        "response_level_evidence": {
            "discovery_human_pairs": len(discovery_quality_rows),
            "preliminary_no_confirmed_added_material_risk": (
                preliminary_response_safety_passed
            ),
            "rs_applicable_risk_cells_reviewed": len(applicable_rs_risks),
            "rs_confirmed_material_risk_events": len(confirmed_rs_risks),
            "conditional_benefit_confirmation_status": str(
                confirmation_manifest.get("status", "missing")
            ),
            "confirmation_complete": confirmation_complete,
            "per_card": per_card_response_evidence,
        },
        "decision": {
            "current_train_only_top1_rs_pilot_authorized": (
                current_pilot_qualified
            ),
            "formal_rs_main_table_authorized": False,
            "llm_weak_card_audit_is_promotion_gate": False,
            "why": (
                "Human content review is stronger than an unqualified weak "
                "LLM audit for this small catalog. Formal use still requires "
                "the independent response-effect confirmation and a frozen "
                "family gate."
            ),
        },
        "permitted_claims_now": [
            "the five-card catalog passed intrinsic train-only human review",
            "the catalog excludes packet-source overlap and raw examples",
            "the first 12-pair audit found no confirmed material risk event in the RS arm",
            "conditional RS benefit is plausible and awaits independent confirmation",
        ],
        "forbidden_claims_now": [
            "the historical 11,590-card Top-k RAG bank is qualified",
            "all five card families have response-level validation",
            "RS is generally superior to R0",
            "the cards are clinically validated or externally generalizable",
        ],
        "next_gate": (
            "Complete the frozen second 12-pair blind confirmation. Promote "
            "only a boundary-cue/family condition whose material-benefit "
            "direction repeats without added material risk."
        ),
    }
