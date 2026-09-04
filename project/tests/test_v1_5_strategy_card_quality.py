from __future__ import annotations

from metacom_pm.v1_5_strategy_card_quality import (
    EXPECTED_FAMILIES,
    HUMAN_CONTENT_FIELDS,
    audit_strategy_card_quality,
)


def test_card_quality_audit_bounds_top1_pilot_claim():
    cards = []
    annotations = []
    selected = []
    quality = []
    risk = []
    for index, family in enumerate(EXPECTED_FAMILIES):
        card_id = f"card_{index}"
        cards.append(
            {
                "card_id": card_id,
                "strategy_family": family,
                "content_scope": "technique_only",
            }
        )
        annotations.append(
            {
                "card_id": card_id,
                **{field: True for field in HUMAN_CONTENT_FIELDS},
                "approve_for_train_only_pilot": True,
                "confidence": 4,
                "required_corrections": "",
            }
        )
        state_id = f"state_{index}"
        pair_id = f"pair_{index}"
        selected.append(
            {
                "state_id": state_id,
                "selected_strategy_card_id": card_id,
            }
        )
        quality.append(
            {
                "state_id": state_id,
                "pair_id": pair_id,
                "human_arm_preference": "RS",
            }
        )
        risk.append(
            {
                "pair_id": pair_id,
                "arm": "RS",
                "programmatically_applicable": True,
                "human_material_event": False,
            }
        )

    report = audit_strategy_card_quality(
        cards=cards,
        annotations=annotations,
        build_report={
            "candidate_packet_source_overlap_count": 0,
            "candidate_raw_examples_exposed_to_generator": False,
            "candidate_card_count": 5,
            "raw_card_count": 100,
            "candidate_lineage_row_count": 80,
        },
        human_validation_report={"formal_rs_promoted": False},
        selected_states=selected,
        discovery_quality_rows=quality,
        discovery_risk_rows=risk,
        confirmation_manifest={
            "status": "READY_FOR_INDEPENDENT_BLIND_HUMAN_CONFIRMATION"
        },
    )

    assert report["construction_and_leakage"]["passed"] is True
    assert report["intrinsic_human_review"]["passed_for_train_only_pilot"] is True
    assert (
        report["decision"]["current_train_only_top1_rs_pilot_authorized"]
        is True
    )
    assert report["decision"]["formal_rs_main_table_authorized"] is False
    assert (
        report["object_being_qualified"]["historical_raw_topk_bank_qualified"]
        is False
    )
