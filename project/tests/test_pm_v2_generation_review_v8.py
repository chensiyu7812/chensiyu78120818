from __future__ import annotations

import csv
from pathlib import Path

import pytest

from metacom_pm.contracts import MemorySource
from metacom_pm.io import sha256_file
from metacom_pm.pm_v2_generation_review_v8 import (
    PACKET_FIELDS,
    PROVISIONAL_STATUS,
    PROHIBITED_SURFACE_PHRASES,
    RATING_FIELDS,
    SIMPLE_REVIEW_FIELDS,
    V8_REGIMES,
    audit_generation_pilot_semantics,
    cases_to_packet_rows,
    generate_v8_review_cases,
    prepare_generation_semantic_review_v8,
    readable_review,
    require_generation_semantic_review_v8,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_BANK = ROOT / "data" / "strategy" / "strategy_cards.jsonl"
V7_DIR = ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v7"
V7_HASHES = {
    "generation_pilot_semantic_review_manual.md": "1648e21558ae7fec2eb918c67fd0721bc8131ff5a97a979c839eb0a2c3b9dfb0",
    "generation_pilot_semantic_review_packet.csv": "2f8b6c7693b8756043391b4eeb563e2540d55510be16cbb6cbbe2a35b980d1a4",
    "generation_pilot_semantic_review_plan.json": "d1ae4e5906a63d58eec900b07ceba443c029fa2919e89033b674459c4cbc657e",
    "generation_pilot_semantic_review_readable_ZH.md": "9c654d42bef852a47335bd8be0541a25652375ba0aaa4fb687d37d2cd8eb2086",
    "reviewer_a.csv": "bc31dbee48a204336cd3f82081ca6549d6db69059efcacc497657b3084fcffdf",
    "reviewer_b.csv": "fdb09a5317300be571a084f0d01d0d2a18503414332914f936d5cfa4d0f8ffd0",
}


def _cases(*, seed: int = 20260749, cases_per_regime: int = 1):
    return generate_v8_review_cases(
        strategy_bank_path=STRATEGY_BANK,
        cases_per_regime=cases_per_regime,
        seed=seed,
    )


def test_v7_review_artifacts_are_byte_identical() -> None:
    assert V7_DIR.is_dir()
    assert {
        path.name: sha256_file(path)
        for path in sorted(V7_DIR.iterdir())
        if path.is_file()
    } == V7_HASHES


def test_v8_schema_uses_material_value_and_twelve_blank_ratings() -> None:
    cases = _cases()
    rows = cases_to_packet_rows(cases)
    assert len(cases) == len(V8_REGIMES) == 9
    assert tuple(rows[0]) == PACKET_FIELDS
    assert "candidate_materially_useful_memory_sources_json" in rows[0]
    assert rows[0]["legacy_needed_memory_sources_json"] == rows[0][
        "candidate_materially_useful_memory_sources_json"
    ]
    assert len(RATING_FIELDS) == 12
    assert all(row[field] == "" for row in rows for field in RATING_FIELDS)
    assert all(row["annotator_id"] == row["notes"] == "" for row in rows)


def test_v8_default_smoke_passes_leakage_and_grounding_audit() -> None:
    report = audit_generation_pilot_semantics(
        _cases(),
        strategy_bank_path=STRATEGY_BANK,
        v7_packet_path=V7_DIR / "generation_pilot_semantic_review_packet.csv",
    )
    assert report["status"] == PROVISIONAL_STATUS
    assert report["technical_checks_status"] == "PASS"
    assert report["ready_for_annotation"] is False
    assert report["ready_for_training"] is False
    assert all(report["checks"].values())
    assert report["youngest_item_heuristic"]["accuracy"] < 1.0
    assert report["first_row_heuristic"]["accuracy"] < 1.0
    assert abs(report["age_non_irrelevant_pearson_correlation"]) < 0.8
    assert all(
        all(values.values())
        for values in report["counterexamples_by_source"].values()
    )


def test_memory_age_and_row_order_are_reproducible_but_seed_sensitive() -> None:
    same_left = _cases(seed=20260749)
    same_right = _cases(seed=20260749)
    different = _cases(seed=20260750)
    assert [case.model_dump(mode="json") for case in same_left] == [
        case.model_dump(mode="json") for case in same_right
    ]
    left_design = [
        [
            (item.memory_id, item.age_sessions, item.utility)
            for source in MemorySource
            for item in {
                MemorySource.MP: case.profile_memories,
                MemorySource.MS: case.summary_memories,
                MemorySource.ME: case.event_memories,
            }[source]
        ]
        for case in same_left
    ]
    different_design = [
        [
            (item.memory_id, item.age_sessions, item.utility)
            for source in MemorySource
            for item in {
                MemorySource.MP: case.profile_memories,
                MemorySource.MS: case.summary_memories,
                MemorySource.ME: case.event_memories,
            }[source]
        ]
        for case in different
    ]
    assert left_design != different_design


def test_context_grounding_and_surface_leakage_fail_closed() -> None:
    grounding_cases = [case.model_copy(deep=True) for case in _cases()]
    grounding_cases[0].context_provenance[0].evidence_quote = "hidden fact"
    grounding = audit_generation_pilot_semantics(
        grounding_cases, strategy_bank_path=STRATEGY_BANK
    )
    assert grounding["status"] == "FAIL"
    assert grounding["checks"]["context_grounding_complete"] is False

    leakage_cases = [case.model_copy(deep=True) for case in _cases()]
    leakage_cases[0].current_user_text += " I need multiple memory sources."
    leakage = audit_generation_pilot_semantics(
        leakage_cases, strategy_bank_path=STRATEGY_BANK
    )
    assert leakage["status"] == "FAIL"
    assert leakage["checks"]["no_prohibited_surface_phrases"] is False


def test_strategy_evidence_is_real_and_advice_is_independent() -> None:
    cases = _cases()
    assert all(
        len(case.strategy_evidence) >= 2
        and all(
            item.card_id
            and item.strategy_type
            and item.card_text
            and item.utility
            and item.marginal_value_rationale
            for item in case.strategy_evidence
        )
        for case in cases
    )
    advice_harmful = next(case for case in cases if case.regime == "advice_harmful")
    assert advice_harmful.strategy_target.use_strategy_rag == "on"
    assert advice_harmful.strategy_target.advice_readiness == "listen_only"
    assert {item.utility for item in advice_harmful.strategy_evidence} == {
        "helpful",
        "harmful",
    }
    strategy_helpful = next(
        case for case in cases if case.regime == "strategy_helpful"
    )
    assert strategy_helpful.strategy_target.use_strategy_rag == "on"
    assert strategy_helpful.strategy_target.advice_readiness == "light_suggestion"


def test_full_bilingual_material_contains_every_dialogue_and_evidence() -> None:
    cases = _cases()
    zh = readable_review(cases, language="ZH")
    en = readable_review(cases, language="EN")
    for document in (zh, en):
        for case in cases:
            assert case.current_user_text in document
            assert case.session_summary in document
            assert case.authorized_user_context in document
            assert case.coverage_rationale in document
            assert all(turn.content in document for turn in case.dialogue_before_current)
            assert all(
                item.text in document
                for source in MemorySource
                for item in {
                    MemorySource.MP: case.profile_memories,
                    MemorySource.MS: case.summary_memories,
                    MemorySource.ME: case.event_memories,
                }[source]
            )
            assert all(item.card_text in document for item in case.strategy_evidence)
        assert all(field in document for field in RATING_FIELDS)


def test_prepare_writes_blank_reviewers_and_preserves_v7(tmp_path: Path) -> None:
    before = {name: sha256_file(V7_DIR / name) for name in V7_HASHES}
    out_dir = tmp_path / "v8"
    result = prepare_generation_semantic_review_v8(
        strategy_bank_path=STRATEGY_BANK,
        out_dir=out_dir,
        v7_dir=V7_DIR,
        cases_per_regime=1,
        seed=20260749,
    )
    assert result["status"] == PROVISIONAL_STATUS
    assert result["audit_status"] == PROVISIONAL_STATUS
    assert result["technical_checks_status"] == "PASS"
    assert result["v7_unchanged"] is True
    for reviewer in ("reviewer_a.csv", "reviewer_b.csv"):
        with (out_dir / reviewer).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == SIMPLE_REVIEW_FIELDS
            rows = list(reader)
        assert len(rows) == 9
        assert all(row[field] == "" for row in rows for field in RATING_FIELDS)
        assert all(row["notes"] == "" for row in rows)
        assert all(row["annotator_id"] for row in rows)
    assert {name: sha256_file(V7_DIR / name) for name in V7_HASHES} == before
    assert all(
        phrase not in case.current_user_text.casefold()
        for case in _cases()
        for phrase in PROHIBITED_SURFACE_PHRASES
    )


def test_formal_generation_is_blocked_pending_v1_strategy_audit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="V1 Strategy RAG"):
        require_generation_semantic_review_v8(
            tmp_path / "nonexistent_attestation.json",
            require_validation=True,
        )
