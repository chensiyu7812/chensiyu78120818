from __future__ import annotations

import pytest

from metacom_pm.v1_5_v5_3_p2r_learning_blueprint import (
    CATALOG_PROTOCOL,
    P2RCurrentStateSpec,
    P2RLongitudinalCatalogItem,
    audit_blueprint_rows,
    assemble_interaction_row,
    make_state_identity,
    materialize_memory_state,
    materialize_rs_state,
    normalize_worker_catalog_row,
)
from metacom_pm.contracts import StrategyCard
from metacom_pm.io import read_jsonl


def _catalog() -> list[P2RLongitudinalCatalogItem]:
    rows: list[P2RLongitudinalCatalogItem] = [
        P2RLongitudinalCatalogItem(
            catalog_user_id="user_alpha",
            memory_id="mem_0000000000000001",
            component="MP",
            subtype="MP_PROFILE",
            created_session=1,
            valid_from_session=1,
            subject_entity_id="user_alpha",
            semantic_family="commute_disruption",
            topic_key="appointment logistics",
            literal_text="Availability: appointments are possible only in the early evening",
            source_session_ref="session_1",
            field_type="availability_window",
            field_value="early evening",
            catalog_role="target_capable",
        ),
        P2RLongitudinalCatalogItem(
            catalog_user_id="user_alpha",
            memory_id="mem_0000000000000002",
            component="MS",
            subtype="MS_SESSION",
            created_session=2,
            valid_from_session=2,
            subject_entity_id="user_alpha",
            semantic_family="commute_disruption",
            topic_key="commute disruption",
            literal_text="An earlier commute disruption observation separated schedule uncertainty from travel fatigue.",
            source_session_ref="session_2",
            catalog_role="target_capable",
        ),
        P2RLongitudinalCatalogItem(
            catalog_user_id="user_alpha",
            memory_id="mem_0000000000000003",
            component="ME",
            subtype="ME_REUSABLE_OUTCOME",
            created_session=3,
            valid_from_session=3,
            subject_entity_id="user_alpha",
            semantic_family="commute_disruption",
            topic_key="commute disruption",
            literal_text="I tried leaving ten minutes earlier, and it helped me feel calmer during the commute disruption.",
            source_session_ref="session_3",
            action_span="I tried leaving ten minutes earlier",
            outcome_span="it helped me feel calmer during the commute disruption",
            catalog_role="target_capable",
        ),
    ]
    # Realistic longitudinal depth is a property of the user history, not the
    # number of surface variants generated from it.
    for session in range(4, 14):
        rows.append(
            P2RLongitudinalCatalogItem(
                catalog_user_id="user_alpha",
                memory_id=f"mem_{session:016x}",
                component="MS",
                subtype="MS_SESSION",
                created_session=session,
                valid_from_session=session,
                subject_entity_id="user_alpha",
                semantic_family=f"other_family_{session}",
                topic_key=f"unrelated topic {session}",
                literal_text=f"A specific earlier observation concerned unrelated topic {session}.",
                source_session_ref=f"session_{session}",
                catalog_role="cross_topic_distractor",
            )
        )
    return rows


def _state(component: str, variant: str, text: str) -> P2RCurrentStateSpec:
    state_id, group_id = make_state_identity(
        component=component,
        catalog_user_id="user_alpha",
        semantic_family="commute_disruption",
        variant=variant,
    )
    return P2RCurrentStateSpec(
        state_id=state_id,
        catalog_user_id="user_alpha",
        current_subject_entity_id="user_alpha",
        component=component,
        semantic_family="commute_disruption",
        counterfactual_group_id=group_id,
        state_condition=(
            "positive_opportunity" if variant == "positive" else "current_redundant"
        ),
        current_session_index=14,
        visible_dialogue=[{"role": "user", "content": text}],
        intended_response_act="offer one bounded response",
        construction_note="fixture tests schema and isolation, not model quality",
    )


def test_catalog_requires_structured_component_payloads():
    with pytest.raises(ValueError, match="field_type"):
        P2RLongitudinalCatalogItem(
            catalog_user_id="u",
            memory_id="mem_1234567890abcdef",
            component="MP",
            subtype="MP_PROFILE",
            created_session=1,
            valid_from_session=1,
            subject_entity_id="u",
            semantic_family="f",
            topic_key="t",
            literal_text="a profile fact",
            source_session_ref="s1",
            catalog_role="target_capable",
        )


def test_worker_alias_adapter_does_not_infer_semantics():
    row = normalize_worker_catalog_row(
        {
            "protocol": CATALOG_PROTOCOL,
            "user_id": "user_alpha",
            "candidate_id": "mem_1234567890abcdef",
            "source": "MS",
            "subtype": "MS_SESSION",
            "created_session": 1,
            "subject_entity_id": "user_alpha",
            "family": "family",
            "topic_key": "topic",
            "text": "A concrete earlier observation about topic.",
            "source_session_ref": "session_1",
            "catalog_role": "target_capable",
        }
    )
    assert row.catalog_user_id == "user_alpha"
    assert row.literal_text.startswith("A concrete")


def test_materializer_restores_runtime_features_without_gold_or_ids():
    state = _state(
        "ME",
        "positive",
        "The commute disruption is back. I am open to one small optional step.",
    )
    row = materialize_memory_state(state=state, catalog=_catalog())
    assert row.model_input.candidate_present is True
    assert row.model_input.contribution_slots["past_action_result"] is True
    assert row.model_input.contribution_slots["current_action_readiness"] == "INVITES_ACTION"
    assert row.candidate_lineage.candidate_id == "mem_0000000000000003"
    assert row.candidate_lineage.candidate_id not in str(row.model_input.model_dump())
    assert row.catalog_session_count == 13


def test_counterfactual_variants_share_group_and_audit_crossing():
    positive = materialize_memory_state(
        state=_state(
            "MS",
            "positive",
            "Last time we discussed the commute disruption; can we pick that thread back up?",
        ),
        catalog=_catalog(),
    )
    redundant = materialize_memory_state(
        state=_state(
            "MS",
            "redundant",
            "The commute disruption is schedule uncertainty rather than travel fatigue; I already have that distinction here.",
        ),
        catalog=_catalog(),
    )
    assert positive.split_group_key == redundant.split_group_key
    report = audit_blueprint_rows(
        [positive, redundant], require_final_condition_crossing=True
    )
    assert report["critical_failures"] == {
        "duplicate_state_ids": 0,
        "future_or_current_candidates": 0,
        "split_leak_groups": 0,
        "model_input_gold_leaks": 0,
        "missing_runtime_feature_rows": 0,
        "families_without_condition_crossing": 0,
    }


def test_strict_past_filter_excludes_current_and_future_items():
    catalog = _catalog()
    catalog.append(
        P2RLongitudinalCatalogItem(
            catalog_user_id="user_alpha",
            memory_id="mem_ffffffffffffffff",
            component="MS",
            subtype="MS_SESSION",
            created_session=14,
            valid_from_session=14,
            subject_entity_id="user_alpha",
            semantic_family="commute_disruption",
            topic_key="commute disruption",
            literal_text="A current-session answer must never enter the candidate pool.",
            source_session_ref="session_14",
            catalog_role="target_capable",
        )
    )
    row = materialize_memory_state(
        state=_state(
            "MS",
            "positive",
            "Last time we discussed the commute disruption; can we pick that thread back up?",
        ),
        catalog=catalog,
    )
    assert "mem_ffffffffffffffff" not in row.topk_candidate_ids


def test_me_rank1_compile_failure_does_not_promote_rank2():
    catalog = _catalog()
    catalog.append(
        P2RLongitudinalCatalogItem(
            catalog_user_id="user_alpha",
            memory_id="mem_eeeeeeeeeeeeeeee",
            component="ME",
            subtype="ME_CONTEXT_EVENT",
            created_session=4,
            valid_from_session=4,
            subject_entity_id="user_alpha",
            semantic_family="commute_disruption",
            topic_key="ferry cancellation platform closure",
            literal_text="The ferry cancellation and platform closure remained upsetting and unresolved.",
            source_session_ref="session_4",
            catalog_role="same_topic_competitor",
        )
    )
    state = _state(
        "ME",
        "positive",
        "The ferry cancellation and platform closure are back; I am open to one small optional step.",
    )
    row = materialize_memory_state(state=state, catalog=catalog)
    assert row.topk_candidate_ids[0] == "mem_eeeeeeeeeeeeeeee"
    assert row.model_input.candidate_present is False
    assert row.candidate_lineage.candidate_id is None
    assert row.candidate_lineage.rank1_compiler_valid is False


def test_rs_materializer_uses_six_card_pool_and_runtime_slots():
    cards = [
        StrategyCard.model_validate(row)
        for row in read_jsonl("data/strategy/strategy_cards_v1_5_minimal.jsonl")
    ]
    state_id, group_id = make_state_identity(
        component="RS",
        catalog_user_id="user_rs",
        semantic_family="open_expression",
        variant="positive",
    )
    state = P2RCurrentStateSpec(
        state_id=state_id,
        catalog_user_id="user_rs",
        current_subject_entity_id="user_rs",
        component="RS",
        semantic_family="open_expression",
        counterfactual_group_id=group_id,
        state_condition="positive_opportunity",
        current_session_index=2,
        visible_dialogue=[
            {"role": "user", "content": "I don't know where to start; something is bothering me."}
        ],
        intended_response_act="invite open expression",
        construction_note="natural RS opportunity",
    )
    row = materialize_rs_state(state=state, cards=cards)
    assert row.model_input.candidate_present is True
    assert row.candidate_lineage.candidate_id is not None
    assert "card_precondition_met" in row.model_input.contribution_slots
    assert row.model_input.contribution_slots["observable_open_expression_opportunity"] is True


def test_interaction_assembly_preserves_each_frozen_candidate():
    text = "Last time we discussed the commute disruption. Could you suggest one practical way to arrange the appointment?"
    mp = materialize_memory_state(
        state=_state("MP", "positive", text), catalog=_catalog()
    )
    ms = materialize_memory_state(
        state=_state("MS", "positive", text), catalog=_catalog()
    )
    interaction = assemble_interaction_row(
        component_rows={"MP": mp, "MS": ms}, variant="positive"
    )
    assert set(interaction.components) == {"MP", "MS"}
    assert interaction.candidate_lineage["MP"] == mp.candidate_lineage
    assert interaction.candidate_lineage["MS"] == ms.candidate_lineage
