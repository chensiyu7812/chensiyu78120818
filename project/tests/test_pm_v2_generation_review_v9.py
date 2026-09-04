from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import shutil

import pytest

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import iter_jsonl, read_json, sha256_file
from metacom_pm.pm_v2_generation_review_v9 import (
    CURRENT_BANK_SHA256,
    LEGACY_BANK_SHA256,
    REVIEW_FIELDS,
    _finish_blinded_materials,
    _generator_messages,
    derive_strategy_rag_marginal_value,
)
from metacom_pm.retrieval import StrategyRetriever as DirtyStrategyRetriever
from metacom_pm.retrieval_v1_canonical import (
    CANONICAL_V1_RETRIEVER_SHA256,
    StrategyRetrieverV1Canonical,
    context_query,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "outputs" / "strategy_rag_v1_bank_comparison"
FROZEN = ROOT / "outputs" / "strategy_rag_v1_frozen_candidate"
V8 = ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v8"
V9 = ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v9"
V9_REVIEW = (
    ROOT
    / "outputs"
    / "pm_v2_generation_pilot_semantic_review_v9_review_protocol_v2"
)
V9_ARCHIVE_PATHS = (
    COMPARISON / "bank_comparison_traces.jsonl",
    FROZEN / "strategy_rag_manifest.json",
    V8 / "generation_pilot_semantic_review_cases.json",
    V9 / "generation_pilot_semantic_review_cases.json",
    V9 / "paired_generation_call_plan.jsonl",
    V9 / "private_condition_mapping.jsonl",
    V9_REVIEW / "generation_pilot_semantic_review_packet.csv",
)
requires_v9_archive = pytest.mark.skipif(
    not all(path.is_file() for path in V9_ARCHIVE_PATHS),
    reason="historical V8/V9 artifact-vault bundle is absent from public checkout",
)


def _cards() -> list[StrategyCard]:
    return [
        StrategyCard.model_validate(row)
        for row in iter_jsonl(ROOT / "data" / "strategy" / "strategy_cards.jsonl")
    ]


def test_canonical_wrapper_has_no_threshold_or_abstention_surface():
    parameters = inspect.signature(StrategyRetrieverV1Canonical).parameters
    assert set(parameters) == {"cards", "top_k"}
    with pytest.raises(ValueError, match="top_k=3"):
        StrategyRetrieverV1Canonical([], top_k=2)


@requires_v9_archive
def test_canonical_wrapper_matches_v1_default_behavior_on_all_v9_cases():
    cards = _cards()
    canonical = StrategyRetrieverV1Canonical(cards)
    dirty_default = DirtyStrategyRetriever(cards, top_k=3)
    for case in read_json(V9 / "generation_pilot_semantic_review_cases.json"):
        query = context_query(
            case["current_user_text"],
            case["dialogue_before_current"],
            case["session_summary"],
        )
        expected = canonical.retrieve(query)
        observed = dirty_default.retrieve(query)
        assert [row.strategy_id for row in observed] == [
            row.strategy_id for row in expected
        ]
        assert case["strategy_retrieval"]["actual_order"] == [
            row.strategy_id for row in expected
        ]


@requires_v9_archive
def test_bank_comparison_is_dual_bank_and_human_fields_are_empty():
    traces = list(iter_jsonl(COMPARISON / "bank_comparison_traces.jsonl"))
    assert len(traces) == 9
    for trace in traces:
        assert trace["v1_training_legacy_bank"]["sha256"] == LEGACY_BANK_SHA256
        assert len(trace["v1_training_legacy_bank"]["top_3"]) == 3
        assert (
            trace["v1_confirmatory_external_bank"]["sha256"]
            == CURRENT_BANK_SHA256
        )
        assert len(trace["v1_confirmatory_external_bank"]["top_3"]) == 3
        assert trace["retriever_behavior"]["threshold"] is None
        assert trace["retriever_behavior"]["reranker"] is None
        assert all(value == "" for value in trace["human_review"].values())
    with (COMPARISON / "bank_comparison_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "bank_156_relevance",
            "bank_12429_relevance",
            "bank_156_noise",
            "bank_12429_noise",
            "preferred_bank",
            "notes",
        ):
            assert row[field] == ""


@requires_v9_archive
def test_frozen_candidate_is_locked_and_separates_legacy_bank():
    manifest = read_json(FROZEN / "strategy_rag_manifest.json")
    assert manifest["gate_unlocked"] is False
    assert manifest["bank"]["sha256"] == CURRENT_BANK_SHA256
    assert manifest["bank"]["logical_name"] == "v1_confirmatory_external_bank"
    assert manifest["legacy_bank"]["logical_name"] == "v1_training_legacy_bank"
    assert manifest["legacy_bank"]["mixed_with_candidate"] is False
    assert manifest["retriever"]["canonical_source_sha256"] == (
        CANONICAL_V1_RETRIEVER_SHA256
    )
    assert manifest["retriever"]["threshold"] is None
    assert manifest["retriever"]["embedding_model"] is None
    assert manifest["retriever"]["persistent_index"] is None


@requires_v9_archive
def test_v9_removes_assumed_strategy_ground_truth_and_uses_real_top3():
    cases = read_json(V9 / "generation_pilot_semantic_review_cases.json")
    assert len(cases) == 9
    for case in cases:
        assert "strategy_target" not in case
        assert "strategy_evidence" not in case
        retrieval = case["strategy_retrieval"]
        assert retrieval["bank_sha256"] == CURRENT_BANK_SHA256
        assert retrieval["canonical_retriever_source_sha256"] == (
            CANONICAL_V1_RETRIEVER_SHA256
        )
        assert len(retrieval["cards"]) == 3
        assert retrieval["actual_order"] == [
            row["card_id"] for row in retrieval["cards"]
        ]
        assert retrieval["automatic_item_utility_labels"] is None
        assert retrieval["automatic_resource_need_label"] is None
        for card in retrieval["cards"]:
            assert "utility" not in card
            assert "marginal_value_rationale" not in card


@requires_v9_archive
def test_v9_call_plan_is_strictly_paired_and_only_strategy_input_differs():
    cases = {
        row["item_id"]: row
        for row in read_json(V9 / "generation_pilot_semantic_review_cases.json")
    }
    plan = list(iter_jsonl(V9 / "paired_generation_call_plan.jsonl"))
    assert len(plan) == 18
    by_item: dict[str, dict[str, dict]] = {}
    for row in plan:
        by_item.setdefault(row["item_id"], {})[row["condition"]] = row
    assert set(by_item) == set(cases)
    for item_id, pair in by_item.items():
        assert set(pair) == {"R0", "RS"}
        assert pair["R0"]["seed"] == pair["RS"]["seed"]
        assert pair["R0"]["temperature"] == pair["RS"]["temperature"] == 0.0
        assert pair["R0"]["max_tokens"] == pair["RS"]["max_tokens"] == 300
        assert pair["R0"]["endpoint"] == pair["RS"]["endpoint"]
        assert pair["R0"]["strategy_card_ids"] == []
        assert pair["RS"]["strategy_card_ids"] == cases[item_id][
            "strategy_retrieval"
        ]["actual_order"]
        assert pair["R0"]["messages"] == _generator_messages(cases[item_id], [])
        assert pair["RS"]["messages"] == _generator_messages(
            cases[item_id], cases[item_id]["strategy_retrieval"]["cards"]
        )
        assert pair["R0"]["messages"][0] == pair["RS"]["messages"][0]


@requires_v9_archive
def test_v9_reviewer_scores_and_card_utilities_are_blank_and_mapping_hidden():
    for filename, annotator in (
        ("reviewer_a.csv", "reviewer_a"),
        ("reviewer_b.csv", "reviewer_b"),
    ):
        with (V9_REVIEW / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        for row in rows:
            assert row["annotator_id"] == annotator
            assert row["notes"] == ""
            assert all(row[field] == "" for field in REVIEW_FIELDS)
            assert "R0" not in row and "RS" not in row
    summary = read_json(V9_REVIEW / "review_materials_summary.json")
    assert summary["status"] == "CANDIDATE_FOR_SMOKE_REVIEW_ONLY"
    assert summary["review_protocol_revision"] == 2
    assert summary["additional_api_calls"] == 0
    assert summary["memory_evidence_in_primary_packet"] is True
    assert summary["private_mapping_present"] is False
    assert summary["blind_marginal_value_field_present"] is False
    assert summary["formal_annotation_approved"] is False
    assert (V9_REVIEW / "paired_responses_blinded.jsonl").is_file()
    assert not (V9_REVIEW / "private_condition_mapping.jsonl").exists()
    status = (V9_REVIEW / "V9_STATUS.md").read_text()
    assert "PAIR GENERATION COMPLETE" in status
    assert "unapproved for formal annotation" in status


@requires_v9_archive
def test_v9_review_packet_combines_memory_strategy_and_blind_responses():
    with (V9_REVIEW / "generation_pilot_semantic_review_packet.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    required = {
        "profile_memories_json",
        "summary_memories_json",
        "event_memories_json",
        "context_provenance_json",
        "retrieved_cards_json",
        "response_A",
        "response_B",
    }
    assert required <= set(rows[0])
    for row in rows:
        assert all(row[field] for field in required)
    assert "condition" not in rows[0]
    assert "strategy_rag_marginal_value" not in rows[0]
    guidelines = (V9_REVIEW / "annotation_guidelines_ZH.md").read_text()
    assert "PAIR GENERATION PENDING" not in guidelines
    assert "pair_is_diagnostic" in guidelines


def test_strategy_rag_marginal_value_is_only_derived_after_unblinding():
    mapping = {"A": "RS", "B": "R0"}
    assert derive_strategy_rag_marginal_value(
        response_preference="A", pair_is_diagnostic=1, blind_mapping=mapping
    ) == "positive"
    assert derive_strategy_rag_marginal_value(
        response_preference="B", pair_is_diagnostic=1, blind_mapping=mapping
    ) == "negative"
    assert derive_strategy_rag_marginal_value(
        response_preference="tie", pair_is_diagnostic=1, blind_mapping=mapping
    ) == "neutral"
    assert derive_strategy_rag_marginal_value(
        response_preference="A", pair_is_diagnostic=0, blind_mapping=mapping
    ) == "uncertain"


@requires_v9_archive
def test_v8_and_formal_gate_remain_unchanged():
    assert sha256_file(V8 / "generation_pilot_semantic_review_cases.json") == (
        "cda849aabdaa9ae6c08a1cf5f66fd45dd4a6d5ef641f47714f22deb7cbd3faaf"
    )
    source = (
        ROOT / "src" / "metacom_pm" / "pm_v2_generation_review_v8.py"
    ).read_text(encoding="utf-8")
    assert "V1_STRATEGY_RAG_AUDIT_COMPLETE = False" in source


@requires_v9_archive
def test_blinded_material_finalization_never_exposes_condition_mapping(tmp_path):
    for filename in (
        "generation_pilot_semantic_review_cases.json",
        "private_condition_mapping.jsonl",
    ):
        shutil.copy2(V9 / filename, tmp_path / filename)
    cases = read_json(tmp_path / "generation_pilot_semantic_review_cases.json")
    rows = []
    for case in cases:
        for condition in ("R0", "RS"):
            rows.append(
                {
                    "call_key": f"{case['item_id']}_{condition}",
                    "item_id": case["item_id"],
                    "case_id": case["case_id"],
                    "condition": condition,
                    "response": f"synthetic response {condition} for {case['item_id']}",
                }
            )
    from metacom_pm.io import write_jsonl

    write_jsonl(tmp_path / "paired_generations.jsonl", rows)
    summary = _finish_blinded_materials(tmp_path)
    assert summary["status"] == "CANDIDATE_FOR_SMOKE_REVIEW_ONLY"
    blinded = list(iter_jsonl(tmp_path / "paired_responses_blinded.jsonl"))
    assert len(blinded) == 9
    assert all("condition" not in row for row in blinded)
    assert all(row["mapping_hidden_from_reviewer"] is True for row in blinded)
    packet = (tmp_path / "generation_pilot_semantic_review_packet.csv").read_text()
    assert "condition" not in packet.splitlines()[0]
    status = (tmp_path / "V9_STATUS.md").read_text()
    assert "PAIR GENERATION COMPLETE" in status
    assert "unapproved for formal annotation" in status
