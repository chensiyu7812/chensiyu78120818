from __future__ import annotations

import numpy as np
import pytest

from metacom_pm.contracts import MemorySource, StrategyCard
from metacom_pm.hybrid_retrieval import (
    HybridMemoryRetriever,
    HybridStrategyRetriever,
    SourceScoreFloors,
)
from metacom_pm.hybrid_retrieval_diagnostics import (
    CalibrationExample,
    calibrate_source_floors,
    cluster_bootstrap_paired_diff,
    compare_fixed_token_budget,
    compare_fixed_top_k,
    evaluate_case_memory_retrieval_quality,
    evaluate_esconv_strategy_retrieval_quality,
    iter_case_calibration_examples,
    split_of_user_id,
    _is_positive_label,
)
from metacom_pm.retrieval import MemoryRetriever, StrategyRetriever


class _FakeEncoder:
    def __init__(self, vocab: dict[str, np.ndarray]):
        self.vocab = vocab

    def encode(self, texts):
        return np.stack([self.vocab[text] for text in texts])


def _one_hot(dimension: int, index: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float64)
    vector[index] = 1.0
    return vector


def _memory_row(
    text: str,
    *,
    item_utility: str = "helpful",
    stale: bool = False,
    conflicts: bool = False,
) -> dict:
    return {
        "text": text,
        "item_utility": item_utility,
        "stale": stale,
        "conflicts_with_current_state": conflicts,
    }


def _bundle(user_id: str, cases: list[dict]) -> dict:
    return {"user_id": user_id, "cases": cases}


def test_split_of_user_id_accepts_all_four_and_rejects_garbage():
    assert split_of_user_id("pmv2_train_u001") == "train"
    assert split_of_user_id("pmv2_calibration_u012") == "calibration"
    assert split_of_user_id("pmv2_internal_test_u016") == "internal_test"
    assert split_of_user_id("pmv2_external_test_u003") == "external_test"
    with pytest.raises(ValueError):
        split_of_user_id("not_a_pmv2_user_id")


def test_is_positive_label_requires_helpful_and_fresh_and_nonconflicting():
    assert _is_positive_label(_memory_row("x")) is True
    assert _is_positive_label(_memory_row("x", stale=True)) is False
    assert _is_positive_label(_memory_row("x", conflicts=True)) is False
    assert _is_positive_label(_memory_row("x", item_utility="irrelevant")) is False
    assert _is_positive_label(_memory_row("x", item_utility="harmful")) is False


def test_iter_case_calibration_examples_filters_split_and_flattens_pools():
    bundles = [
        _bundle(
            "pmv2_train_u001",
            [
                {
                    "current_user_text": "q1",
                    "profile_memories": [_memory_row("mp text")],
                    "event_memories": [_memory_row("me text")],
                    "summary_memories": [_memory_row("ms text")],
                }
            ],
        ),
        _bundle(
            "pmv2_internal_test_u016",
            [
                {
                    "current_user_text": "q2",
                    "profile_memories": [_memory_row("should not appear")],
                    "event_memories": [],
                    "summary_memories": [],
                }
            ],
        ),
    ]
    examples = iter_case_calibration_examples(bundles, allowed_splits=frozenset({"train"}))
    assert len(examples) == 3
    assert {e.source for e in examples} == {MemorySource.MP, MemorySource.ME, MemorySource.MS}
    assert all(e.split == "train" for e in examples)
    assert all("should not appear" not in e.text for e in examples)


def test_calibrate_source_floors_fits_from_train_and_reports_calibration_only():
    encoder = _FakeEncoder(
        {
            "query": _one_hot(1, 0),
            "strong positive": _one_hot(1, 0),
            "weak positive": _one_hot(1, 0),
            "clear negative": _one_hot(1, 0),
        }
    )
    examples = [
        CalibrationExample("train", "query", MemorySource.ME, "strong positive", True),
        CalibrationExample("train", "query", MemorySource.ME, "weak positive", True),
        CalibrationExample("train", "query", MemorySource.ME, "clear negative", False),
        CalibrationExample("calibration", "query", MemorySource.ME, "strong positive", True),
        CalibrationExample("calibration", "query", MemorySource.ME, "clear negative", False),
    ]
    # All texts share the same fake vector/tokens, so lexical/semantic
    # scores are identical across rows here; what matters for this test is
    # the split-membership bookkeeping and count arithmetic, not score
    # values (test_hybrid_retrieval.py already covers scoring itself).
    result = calibrate_source_floors(examples, encoder=encoder)
    me = result[MemorySource.ME]
    assert me.train_positive_count == 2
    assert me.train_negative_count == 1
    assert me.calibration_positive_count == 1
    assert me.calibration_negative_count == 1
    assert me.calibration_recall is not None
    assert me.calibration_negative_exclusion_rate is not None


def test_calibrate_source_floors_raises_without_train_positives():
    encoder = _FakeEncoder({"q": _one_hot(1, 0), "x": _one_hot(1, 0)})
    examples = [
        CalibrationExample("train", "q", MemorySource.ME, "x", False),
    ]
    with pytest.raises(RuntimeError):
        calibrate_source_floors(examples, encoder=encoder)


def _fake_user(user_id: str) -> dict:
    return {
        "id": user_id,
        "basic_info": {},
        "dialog_history": [
            {
                "id": "sess_a",
                "timestamp": "t0",
                "summary": "",
                "dialogue": [
                    {"role": "seeker", "content": "work deadline stress worry"},
                ],
            },
            {
                "id": "sess_b",
                "timestamp": "t1",
                "summary": "",
                "dialogue": [
                    {"role": "seeker", "content": "completely unrelated hobby chatter"},
                ],
            },
        ],
        "subsequent_topics": [
            {"topic": "work deadline stress worry", "related_sessions": ["sess_a"]},
        ],
    }


def test_compare_fixed_top_k_finds_the_related_session_for_both_scorers():
    user = _fake_user("u1")
    encoder = _FakeEncoder(
        {
            "work deadline stress worry": _one_hot(2, 0),
            "completely unrelated hobby chatter": _one_hot(2, 1),
        }
    )
    lexical_retriever = MemoryRetriever(top_k_by_source={MemorySource.ME: 1})
    hybrid_retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 1},
        floors_by_source={MemorySource.ME: SourceScoreFloors(-1.0, -1.0)},
    )
    result = compare_fixed_top_k(
        [user], lexical_retriever=lexical_retriever, hybrid_retriever=hybrid_retriever
    )
    assert result["lexical_only"].units_evaluated == 1
    assert result["lexical_only"].hit_rate == 1.0
    assert result["hybrid"].units_evaluated == 1
    assert result["hybrid"].hit_rate == 1.0
    assert len(result["lexical_only"].query_hashes) == 1


def test_compare_fixed_token_budget_finds_the_related_session_for_both_scorers():
    user = _fake_user("u1")
    encoder = _FakeEncoder(
        {
            "work deadline stress worry": _one_hot(2, 0),
            "completely unrelated hobby chatter": _one_hot(2, 1),
        }
    )
    hybrid_retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 5},
        floors_by_source={MemorySource.ME: SourceScoreFloors(-1.0, -1.0)},
    )
    result = compare_fixed_token_budget(
        [user], hybrid_retriever=hybrid_retriever, token_budget=100
    )
    assert result["lexical_only"].hit_rate == 1.0
    assert result["hybrid"].hit_rate == 1.0


def _case_with_memories(
    case_id: str,
    *,
    current_user_text: str,
    profile_memories: list[dict] | None = None,
    event_memories: list[dict] | None = None,
    summary_memories: list[dict] | None = None,
    recent_dialogue: list[dict] | None = None,
    session_summary: str = "",
) -> dict:
    return {
        "case_id": case_id,
        "current_user_text": current_user_text,
        "recent_dialogue": recent_dialogue or [],
        "session_summary": session_summary,
        "profile_memories": profile_memories or [],
        "event_memories": event_memories or [],
        "summary_memories": summary_memories or [],
    }


def _labeled_memory_row(
    memory_id: str,
    text: str,
    created_session: int,
    *,
    item_utility: str = "helpful",
    stale: bool = False,
    conflicts: bool = False,
) -> dict:
    return {
        "memory_id": memory_id,
        "text": text,
        "created_session": created_session,
        "item_utility": item_utility,
        "stale": stale,
        "conflicts_with_current_state": conflicts,
    }


def test_evaluate_case_memory_retrieval_quality_rejects_train_split():
    with pytest.raises(ValueError):
        evaluate_case_memory_retrieval_quality(
            [], split="train", lexical_retriever=None, hybrid_retriever=None
        )


def test_evaluate_case_memory_retrieval_quality_finds_positive_and_skips_other_splits():
    positive = _labeled_memory_row("draft_pos", "user loves hiking on weekends", 3)
    distractor = _labeled_memory_row(
        "draft_neg", "totally unrelated distractor text", 1, item_utility="irrelevant"
    )
    bundles = [
        _bundle(
            "pmv2_calibration_u001",
            [
                _case_with_memories(
                    "case_a",
                    current_user_text="user loves hiking on weekends",
                    event_memories=[positive, distractor],
                )
            ],
        ),
        _bundle(
            "pmv2_train_u001",
            [
                _case_with_memories(
                    "case_b",
                    current_user_text="should not be counted",
                    event_memories=[positive],
                )
            ],
        ),
    ]
    encoder = _FakeEncoder(
        {
            "user loves hiking on weekends": _one_hot(2, 0),
            "totally unrelated distractor text": _one_hot(2, 1),
        }
    )
    # evaluate_case_memory_retrieval_quality always retrieves across all
    # three sources at once (a real case's candidate pool spans MP/MS/ME),
    # so both retrievers must be configured for every source even though
    # this fixture only populates ME candidates.
    top_k_by_source = {MemorySource.MP: 1, MemorySource.MS: 1, MemorySource.ME: 1}
    lexical_retriever = MemoryRetriever(top_k_by_source=top_k_by_source)
    hybrid_retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source=top_k_by_source,
        floors_by_source={source: SourceScoreFloors(-1.0, -1.0) for source in MemorySource},
    )
    result = evaluate_case_memory_retrieval_quality(
        bundles,
        split="calibration",
        lexical_retriever=lexical_retriever,
        hybrid_retriever=hybrid_retriever,
    )
    by_source = result["by_method_by_source"]
    me_lexical = by_source["lexical_only"][MemorySource.ME.value]
    me_hybrid = by_source["hybrid"][MemorySource.ME.value]
    # Only the calibration-split case counts (the train-split case must be
    # skipped entirely).
    assert me_lexical.units_evaluated == 1
    assert me_hybrid.units_evaluated == 1
    assert me_lexical.hit_rate == 1.0
    assert me_hybrid.hit_rate == 1.0
    assert me_lexical.mean_precision == 1.0
    assert me_hybrid.mean_precision == 1.0
    # The distractor is "irrelevant", not "harmful", and every source has a
    # helpful candidate here, so neither negative-source metric has an
    # eligible case for this fixture.
    negative = result["negative_source_and_harmful_retrieval"]["lexical_only"][
        MemorySource.ME.value
    ]
    assert negative["harmful_eligible_cases"] == 0
    assert negative["harmful_retrieval_rate"] is None
    assert negative["negative_source_eligible_cases"] == 0
    assert negative["negative_source_false_retrieval_rate"] is None


def _write_minimal_esconv_fixture(tmp_path, *, split: str = "validation"):
    import json

    esconv_path = tmp_path / "ESConv.json"
    esconv_path.write_text(
        json.dumps(
            [
                {
                    # Left empty so context_query(current_user_text, [], "")
                    # reduces to bare current_user_text, matching the fake
                    # encoder's registered vocabulary exactly -- context_query's
                    # merging behavior is already covered by
                    # test_observable_query_matches_context_query_and_hash_is_reproducible
                    # in test_hybrid_retrieval.py.
                    "situation": "",
                    "dialog": [
                        {
                            "speaker": "seeker",
                            "content": "I've been so stressed about my job lately.",
                        },
                        {
                            "speaker": "supporter",
                            "content": "That sounds really hard to deal with.",
                            "annotation": {"strategy": "Affirmation"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "split_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {"index": 0, "split": split, "excluded_for_evoemo_overlap": False}
        )
        + "\n",
        encoding="utf-8",
    )
    return esconv_path, manifest_path


def _strategy_card(strategy_id: str, label: str, retrieval_text: str) -> StrategyCard:
    return StrategyCard(
        strategy_id=strategy_id,
        strategy_label=label,
        retrieval_text=retrieval_text,
        guidance_text="g",
        example_response="e",
        source_dialogue_id="d1",
        source_turn_index=1,
    )


def test_evaluate_esconv_strategy_retrieval_quality_hits_gold_strategy(tmp_path):
    esconv_path, manifest_path = _write_minimal_esconv_fixture(tmp_path)
    # The fixture's seeker turn (the query, since situation/history are
    # empty) is exactly "I've been so stressed about my job lately."
    query_text = "I've been so stressed about my job lately."
    matching_card = _strategy_card(
        "strat_aaaaaaaaaaaa", "Affirmation", "stressed about job"
    )
    other_card = _strategy_card(
        "strat_bbbbbbbbbbbb", "Question", "completely different unrelated topic"
    )
    encoder = _FakeEncoder(
        {
            query_text: _one_hot(2, 0),
            "stressed about job": _one_hot(2, 0),
            "completely different unrelated topic": _one_hot(2, 1),
        }
    )
    lexical_retriever = StrategyRetriever([matching_card, other_card], top_k=1)
    hybrid_retriever = HybridStrategyRetriever(
        [matching_card, other_card],
        semantic_encoder=encoder,
        top_k=1,
        floors=SourceScoreFloors(-1.0, -1.0),
    )
    result = evaluate_esconv_strategy_retrieval_quality(
        str(esconv_path),
        str(manifest_path),
        split="validation",
        lexical_retriever=lexical_retriever,
        hybrid_retriever=hybrid_retriever,
    )
    by_method = result["by_method"]
    assert by_method["lexical_only"].units_evaluated == 1
    assert by_method["hybrid"].units_evaluated == 1
    assert by_method["lexical_only"].hit_rate == 1.0
    assert by_method["hybrid"].hit_rate == 1.0
    # Both scorers hit on the one turn, so the paired hit-rate diff is 0
    # with a single dialogue cluster.
    paired = result["paired_dialogue_cluster_bootstrap"]
    assert paired["cluster_key"] == "dialogue_id"
    assert paired["hit_rate_diff"]["mean"] == 0.0
    assert paired["hit_rate_diff"]["n_clusters"] == 1


def test_evaluate_esconv_strategy_retrieval_quality_only_uses_requested_split(tmp_path):
    esconv_path, manifest_path = _write_minimal_esconv_fixture(tmp_path, split="test")
    card = _strategy_card("strat_aaaaaaaaaaaa", "Affirmation", "job stress")
    encoder = _FakeEncoder({"job stress": _one_hot(1, 0)})
    lexical_retriever = StrategyRetriever([card], top_k=1)
    hybrid_retriever = HybridStrategyRetriever(
        [card], semantic_encoder=encoder, top_k=1, floors=SourceScoreFloors(-1.0, -1.0)
    )
    result = evaluate_esconv_strategy_retrieval_quality(
        str(esconv_path),
        str(manifest_path),
        split="validation",
        lexical_retriever=lexical_retriever,
        hybrid_retriever=hybrid_retriever,
    )
    # The fixture's only row is split="test" -- requesting "validation" must
    # yield zero evaluated turns, not silently fall back to another split.
    assert result["by_method"]["lexical_only"].units_evaluated == 0
    assert result["by_method"]["lexical_only"].hit_rate is None
    assert result["paired_dialogue_cluster_bootstrap"]["hit_rate_diff"]["n_clusters"] == 0


def test_evaluate_case_memory_retrieval_quality_flags_harmful_and_negative_source():
    # ME has only a harmful candidate for this case -- no helpful item at
    # all -- so it is simultaneously "harmful-eligible" (a harmful item is
    # present) and "negative-source-eligible" (nothing helpful to retrieve).
    # With no floor/threshold filtering, top_k_by_source[ME]=1 must surface
    # it, so both rates should read 1.0 for both scorers.
    harmful_only = _labeled_memory_row(
        "draft_harm", "user should isolate from everyone", 2, item_utility="harmful"
    )
    bundles = [
        _bundle(
            "pmv2_calibration_u002",
            [
                _case_with_memories(
                    "case_harm",
                    current_user_text="q",
                    event_memories=[harmful_only],
                )
            ],
        ),
    ]
    encoder = _FakeEncoder(
        {
            "q": _one_hot(1, 0),
            "user should isolate from everyone": _one_hot(1, 0),
        }
    )
    top_k_by_source = {MemorySource.MP: 1, MemorySource.MS: 1, MemorySource.ME: 1}
    lexical_retriever = MemoryRetriever(top_k_by_source=top_k_by_source)
    hybrid_retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source=top_k_by_source,
        floors_by_source={source: SourceScoreFloors(-1.0, -1.0) for source in MemorySource},
    )
    result = evaluate_case_memory_retrieval_quality(
        bundles,
        split="calibration",
        lexical_retriever=lexical_retriever,
        hybrid_retriever=hybrid_retriever,
    )
    for method in ("lexical_only", "hybrid"):
        negative = result["negative_source_and_harmful_retrieval"][method][
            MemorySource.ME.value
        ]
        assert negative["harmful_eligible_cases"] == 1
        assert negative["harmful_retrieval_rate"] == 1.0
        assert negative["negative_source_eligible_cases"] == 1
        assert negative["negative_source_false_retrieval_rate"] == 1.0
    # A source with real ground truth and no distractor drama (e.g. MP,
    # which has no candidates at all in this fixture) has nothing eligible.
    mp_negative = result["negative_source_and_harmful_retrieval"]["lexical_only"][
        MemorySource.MP.value
    ]
    assert mp_negative["harmful_eligible_cases"] == 0
    assert mp_negative["negative_source_eligible_cases"] == 0


def test_cluster_bootstrap_paired_diff_single_cluster_is_exact_mean():
    rows = [("dlg_1", 1.0), ("dlg_1", 0.0), ("dlg_1", 1.0)]
    result = cluster_bootstrap_paired_diff(rows)
    assert result["n_clusters"] == 1
    assert result["mean"] == pytest.approx(2 / 3)
    # A single cluster is always resampled as itself -- CI collapses to the
    # observed mean exactly, no matter how many bootstrap draws.
    assert result["ci_low"] == pytest.approx(2 / 3)
    assert result["ci_high"] == pytest.approx(2 / 3)


def test_cluster_bootstrap_paired_diff_empty_rows_returns_zero_with_no_clusters():
    result = cluster_bootstrap_paired_diff([])
    assert result == {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0}


def test_cluster_bootstrap_paired_diff_multiple_clusters_ci_contains_mean():
    rows = [
        ("dlg_1", 1.0),
        ("dlg_1", 1.0),
        ("dlg_2", 0.0),
        ("dlg_3", 1.0),
        ("dlg_4", 0.0),
    ]
    result = cluster_bootstrap_paired_diff(rows, seed=7, n_boot=500)
    assert result["n_clusters"] == 4
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
