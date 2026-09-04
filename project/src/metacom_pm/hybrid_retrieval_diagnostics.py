"""Report-only calibration/comparison helpers for the Hybrid retriever diagnostic.

Called by ``scripts/v1_5/40_diagnose_hybrid_retrieval.py``, which writes the
JSON report. Every function here is report-only: results may inform a
future, separately-approved adoption decision but must never be treated as
a formal pipeline artifact, and this module must never be imported by any
real consumer (see ``tests/test_hybrid_retrieval_isolation.py``).

**Data-boundary rule (hard, per the approved plan):** ``calibrate_source_
floors`` fits floors using ONLY ``train``-split examples, then reports (never
refits against) ``calibration``-split recall/exclusion as an informational
check. Both splits are identified via the ``pmv2_{split}_u...`` user_id
convention established in
``scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py``.
``internal_test``/``external_test`` rows are never loaded by
``iter_case_calibration_examples`` in the first place. Every EvoEmo-derived
number (``evoemo_topic_diagnostic_rows``, ``compare_fixed_top_k``,
``compare_fixed_token_budget``) is computed by a structurally separate path
and must never feed back into calibration -- these are report-only
comparisons, computed only after the floors/contract are already frozen.

The EvoEmo comparison deliberately uses each user's ``topic`` field (an
EvoEmo evaluator-only field, never real deployment input) as a query
stand-in, exactly as the earlier no-API sensitivity check
(``me_chunk_sensitivity.py``) already did -- this is the one place the plan
explicitly sanctions using evaluator-only text as if it were a query, for
this diagnostic purpose only. It is intentionally kept structurally
separate from ``observable_query_with_hash`` (the real query-construction
path). Comparison scope is intentionally limited to the ME memory source:
``related_sessions`` ground truth is meaningful for session-linked episodic
(ME) items, not for MP (session-independent profile facts) or MS
(whole-session summaries), so extending the comparison to those sources
would have no real ground truth to score against.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import MemoryItem, MemorySource, StrategyCard
from .evoemo import build_evo_memory
from .hybrid_retrieval import (
    HybridMemoryRetriever,
    HybridStrategyRetriever,
    batched_encode,
    retrieve_fixed_token_budget,
)
from .io import sha256_text, stable_hex
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .strategy_bank import esconv_turn_states
from .text import lexical_score


_SPLIT_USER_ID_RE = re.compile(
    r"^pmv2_(?P<split>train|calibration|internal_test|external_test)_u\d+$"
)

MEMORY_POOL_FIELDS: dict[str, MemorySource] = {
    "profile_memories": MemorySource.MP,
    "event_memories": MemorySource.ME,
    "summary_memories": MemorySource.MS,
}

# calibrate_source_floors may fit floors using only this split -- see
# module docstring. CALIBRATION_SPLIT is loaded too, but only ever scored
# against the already-fit floor, never used to fit it.
FLOOR_FITTING_SPLIT = "train"
FLOOR_REPORTING_SPLIT = "calibration"
CALIBRATION_ALLOWED_SPLITS = frozenset({FLOOR_FITTING_SPLIT, FLOOR_REPORTING_SPLIT})


def split_of_user_id(user_id: str) -> str:
    match = _SPLIT_USER_ID_RE.match(str(user_id))
    if not match:
        raise ValueError(
            f"user_id does not match the pmv2 split-prefix convention: {user_id!r}"
        )
    return match.group("split")


def _is_positive_label(memory_row: Mapping[str, Any]) -> bool:
    """A candidate that is genuinely worth retrieving.

    Matches the definition already used elsewhere for this corpus
    (pm_v2_data.py): ``item_utility == "helpful"`` alone is not sufficient
    -- a helpful-topic item that is stale or conflicts with the current
    state is not actually a good retrieve target.
    """

    return (
        memory_row["item_utility"] == "helpful"
        and not memory_row["stale"]
        and not memory_row["conflicts_with_current_state"]
    )


@dataclass(frozen=True)
class CalibrationExample:
    split: str
    query_text: str
    source: MemorySource
    text: str
    label_positive: bool


def iter_case_calibration_examples(
    bundles: Sequence[Mapping[str, Any]],
    *,
    allowed_splits: frozenset[str] = CALIBRATION_ALLOWED_SPLITS,
) -> list[CalibrationExample]:
    """Flatten every case's memory candidates from users in ``allowed_splits``.

    ``bundles`` is the raw per-user structure produced by
    ``scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py`` (one row per
    user, each with a ``cases`` list; each case has ``current_user_text``,
    ``recent_dialogue``, ``session_summary``, and the three memory pools in
    ``MEMORY_POOL_FIELDS``). Any split not in ``allowed_splits`` (in
    particular ``internal_test``/``external_test``) is skipped entirely --
    never even flattened into memory.

    The query used for calibration is built via the same
    ``retrieval.context_query`` the real retrievers use at deployment time
    (``current_user_text`` + visible history + session summary), not bare
    ``current_user_text`` alone -- floors calibrated against a shorter,
    differently-shaped query than what real retrieval actually sees would
    not be calibrated against the query distribution they are meant to
    gate.
    """

    examples: list[CalibrationExample] = []
    for bundle in bundles:
        split = split_of_user_id(bundle["user_id"])
        if split not in allowed_splits:
            continue
        for case in bundle.get("cases") or []:
            query_text = context_query(
                str(case["current_user_text"]),
                case.get("recent_dialogue") or [],
                str(case.get("session_summary") or ""),
            )
            for pool_field, source in MEMORY_POOL_FIELDS.items():
                for memory_row in case.get(pool_field) or []:
                    examples.append(
                        CalibrationExample(
                            split=split,
                            query_text=query_text,
                            source=source,
                            text=str(memory_row["text"]),
                            label_positive=_is_positive_label(memory_row),
                        )
                    )
    return examples


@dataclass(frozen=True)
class SourceFloorCalibration:
    lexical_min_score: float
    semantic_min_score: float
    train_positive_count: int
    train_negative_count: int
    calibration_positive_count: int
    calibration_negative_count: int
    calibration_recall: float | None
    calibration_negative_exclusion_rate: float | None


def _embed_all_texts(examples: Sequence[CalibrationExample], *, encoder) -> dict[str, Any]:
    distinct_texts = sorted({e.query_text for e in examples} | {e.text for e in examples})
    if not distinct_texts:
        return {}
    vectors = batched_encode(encoder, distinct_texts)
    return dict(zip(distinct_texts, vectors))


def calibrate_source_floors(
    examples: Sequence[CalibrationExample],
    *,
    encoder,
) -> dict[MemorySource, SourceFloorCalibration]:
    """Fit one lexical+semantic floor per MemorySource from TRAIN examples only.

    Method (frozen, simple, and disclosed rather than hand-tuned): the floor
    is the minimum raw score observed among TRAIN-split *positive* examples
    for that source and scorer -- i.e. the loosest floor that would not have
    excluded a single known-good TRAIN example. The CALIBRATION split (never
    used to fit the floor) is then scored against that frozen floor purely
    to report its recall (fraction of calibration positives the floor would
    keep) and negative-exclusion rate (fraction of calibration negatives the
    floor would drop) -- informational only, not a second fitting pass.

    Raises if a source has zero TRAIN positive examples: a floor cannot be
    honestly fit from nothing, and silently defaulting would hide a real
    data problem rather than surface it.
    """

    vectors = _embed_all_texts(examples, encoder=encoder)
    scored = [
        (
            e,
            lexical_score(e.query_text, e.text),
            float(vectors[e.query_text] @ vectors[e.text]),
        )
        for e in examples
    ]
    result: dict[MemorySource, SourceFloorCalibration] = {}
    sources_present = sorted({e.source for e in examples}, key=lambda s: s.value)
    for source in sources_present:
        train_rows = [
            (lex, sem, e.label_positive)
            for e, lex, sem in scored
            if e.source is source and e.split == FLOOR_FITTING_SPLIT
        ]
        train_positives = [(lex, sem) for lex, sem, positive in train_rows if positive]
        train_negatives = [(lex, sem) for lex, sem, positive in train_rows if not positive]
        if not train_positives:
            raise RuntimeError(
                f"cannot calibrate a floor for {source.value}: zero TRAIN-split "
                "positive examples -- refusing to fabricate a floor from no data"
            )
        lexical_floor = min(lex for lex, _sem in train_positives)
        semantic_floor = min(sem for _lex, sem in train_positives)

        calibration_rows = [
            (lex, sem, e.label_positive)
            for e, lex, sem in scored
            if e.source is source and e.split == FLOOR_REPORTING_SPLIT
        ]
        calibration_positives = [
            (lex, sem) for lex, sem, positive in calibration_rows if positive
        ]
        calibration_negatives = [
            (lex, sem) for lex, sem, positive in calibration_rows if not positive
        ]
        recall = (
            sum(
                1
                for lex, sem in calibration_positives
                if lex > lexical_floor or sem > semantic_floor
            )
            / len(calibration_positives)
            if calibration_positives
            else None
        )
        exclusion_rate = (
            sum(
                1
                for lex, sem in calibration_negatives
                if lex <= lexical_floor and sem <= semantic_floor
            )
            / len(calibration_negatives)
            if calibration_negatives
            else None
        )
        result[source] = SourceFloorCalibration(
            lexical_min_score=lexical_floor,
            semantic_min_score=semantic_floor,
            train_positive_count=len(train_positives),
            train_negative_count=len(train_negatives),
            calibration_positive_count=len(calibration_positives),
            calibration_negative_count=len(calibration_negatives),
            calibration_recall=recall,
            calibration_negative_exclusion_rate=exclusion_rate,
        )
    return result


@dataclass(frozen=True)
class MethodComparisonSummary:
    units_evaluated: int
    hit_rate: float | None
    mean_precision: float | None
    query_hashes: list[str]


def _summarize(
    *, hits: int, units: int, precisions: list[float], query_hashes: list[str]
) -> MethodComparisonSummary:
    return MethodComparisonSummary(
        units_evaluated=units,
        hit_rate=(hits / units) if units else None,
        mean_precision=statistics.mean(precisions) if precisions else None,
        query_hashes=query_hashes,
    )


# ---------------------------------------------------------------------------
# Legitimate (non-EvoEmo) retrieval-quality evidence -- the only numbers this
# module produces that may inform a Part 4 adoption decision, per the
# approved plan's data-boundary rule. Both functions below use only the
# CALIBRATION split (or, for the ESConv-side check, its own held-out
# "validation" split) -- never TRAIN (already spent fitting floors above)
# and never internal_test/external_test/EvoEmo.
# ---------------------------------------------------------------------------


def evaluate_case_memory_retrieval_quality(
    bundles: Sequence[Mapping[str, Any]],
    *,
    split: str,
    lexical_retriever: MemoryRetriever,
    hybrid_retriever: HybridMemoryRetriever,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Real per-case memory retrieval quality: lexical-only vs Hybrid.

    Unlike ``calibrate_source_floors`` (which only checks whether a raw
    score would clear a floor), this runs the actual
    ``retrieve()`` pipeline -- fusion, ranking, top-k -- over each case's
    real candidate pool (positives and distractors already present in the
    synthetic corpus), using the same ``context_query``-built query as real
    retrieval. Ground truth is the same ``item_utility``/``stale``/
    ``conflicts_with_current_state`` labels used for floor calibration.

    ``split`` must not be ``"train"`` (spent fitting floors) and should
    ordinarily be ``"calibration"`` -- the one split the approved plan
    permits for adoption-relevant evidence. Passing ``"internal_test"``/
    ``"external_test"`` is not blocked (a future, already-frozen-design
    report-only pass may want them) but is not the legitimate evidence path
    and must never be used before design/floors are already frozen.

    Returns ``{"by_method_by_source": {method: {source_value:
    MethodComparisonSummary}}, "negative_source_and_harmful_retrieval":
    {method: {source_value: {harmful_retrieval_rate,
    harmful_eligible_cases, negative_source_false_retrieval_rate,
    negative_source_eligible_cases}}}}``.
    ``harmful_retrieval_rate`` is computed over cases where a
    harmful-labeled candidate exists for that source (does retrieval ever
    surface it); ``negative_source_false_retrieval_rate`` is computed over
    cases where the source has at least one candidate but none of them are
    helpful (does retrieval still surface *something* from a source with
    nothing worth retrieving) -- a source with zero candidates at all is
    deliberately excluded, since there was never anything to retrieve in
    the first place and would only dilute the rate toward 0. Both are
    ``None`` when no case is eligible.
    """

    if split == FLOOR_FITTING_SPLIT:
        raise ValueError(
            "retrieval-quality evaluation must not reuse the TRAIN split "
            "floors were fit from -- use 'calibration' for legitimate "
            "adoption-relevant evidence"
        )
    stats = {
        method: {source: {"hits": 0, "cases": 0, "precisions": []} for source in MemorySource}
        for method in ("lexical_only", "hybrid")
    }
    # harmful_retrieval_rate: among cases where a harmful-labeled candidate
    # exists for this source, how often does retrieval surface >=1 harmful
    # item. negative_source_false_retrieval_rate: among cases where this
    # source has NO helpful candidate at all (nothing worth retrieving),
    # how often does retrieval still surface *something* from that source.
    negative_stats = {
        method: {
            source: {
                "harmful_hits": 0,
                "harmful_eligible": 0,
                "false_retrievals": 0,
                "negative_eligible": 0,
            }
            for source in MemorySource
        }
        for method in ("lexical_only", "hybrid")
    }
    split_bundles = [b for b in bundles if split_of_user_id(b["user_id"]) == split]
    total_cases = sum(len(b.get("cases") or []) for b in split_bundles)
    all_sources = frozenset(MemorySource)
    processed = 0
    for bundle in split_bundles:
        for case in bundle.get("cases") or []:
            query = context_query(
                str(case["current_user_text"]),
                case.get("recent_dialogue") or [],
                str(case.get("session_summary") or ""),
            )
            items: list[MemoryItem] = []
            candidate_ids_by_source: dict[MemorySource, set[str]] = {
                source: set() for source in MemorySource
            }
            positive_ids_by_source: dict[MemorySource, set[str]] = {
                source: set() for source in MemorySource
            }
            harmful_ids_by_source: dict[MemorySource, set[str]] = {
                source: set() for source in MemorySource
            }
            for pool_field, source in MEMORY_POOL_FIELDS.items():
                for row in case.get(pool_field) or []:
                    opaque_id = "mem_" + stable_hex(
                        bundle["user_id"], case["case_id"], source.value,
                        row["memory_id"], n=20,
                    )
                    items.append(
                        MemoryItem(
                            memory_id=opaque_id,
                            source=source,
                            created_session=int(row.get("created_session") or 0),
                            text=str(row["text"]),
                        )
                    )
                    candidate_ids_by_source[source].add(opaque_id)
                    if _is_positive_label(row):
                        positive_ids_by_source[source].add(opaque_id)
                    if row["item_utility"] == "harmful":
                        harmful_ids_by_source[source].add(opaque_id)
            for method, retriever in (
                ("lexical_only", lexical_retriever),
                ("hybrid", hybrid_retriever),
            ):
                retrieved = retriever.retrieve(query, items, all_sources)
                for source in MemorySource:
                    candidates = candidate_ids_by_source[source]
                    positives = positive_ids_by_source[source]
                    harmful = harmful_ids_by_source[source]
                    retrieved_for_source = [it for it in retrieved if it.source is source]
                    retrieved_ids = {it.memory_id for it in retrieved_for_source}
                    if positives:
                        bucket = stats[method][source]
                        bucket["hits"] += int(bool(retrieved_ids & positives))
                        bucket["cases"] += 1
                        if retrieved_for_source:
                            bucket["precisions"].append(
                                sum(1 for it in retrieved_for_source if it.memory_id in positives)
                                / len(retrieved_for_source)
                            )
                    neg_bucket = negative_stats[method][source]
                    if harmful:
                        neg_bucket["harmful_eligible"] += 1
                        neg_bucket["harmful_hits"] += int(bool(retrieved_ids & harmful))
                    # "Negative source" requires candidates to actually exist
                    # for this source with none of them helpful -- a source
                    # with zero candidates at all is not a meaningful case
                    # for "did retrieval wrongly surface something," since
                    # there was never anything to retrieve in the first
                    # place (would trivially dilute the rate toward 0).
                    if candidates and not positives:
                        neg_bucket["negative_eligible"] += 1
                        neg_bucket["false_retrievals"] += int(bool(retrieved_for_source))
            processed += 1
            if progress is not None:
                progress(processed, total_cases)
    by_method_by_source = {
        method: {
            source.value: _summarize(
                hits=bucket["hits"],
                units=bucket["cases"],
                precisions=bucket["precisions"],
                query_hashes=[],
            )
            for source, bucket in per_source.items()
        }
        for method, per_source in stats.items()
    }
    negative_source_and_harmful_retrieval = {
        method: {
            source.value: {
                "harmful_retrieval_rate": (
                    neg_bucket["harmful_hits"] / neg_bucket["harmful_eligible"]
                    if neg_bucket["harmful_eligible"]
                    else None
                ),
                "harmful_eligible_cases": neg_bucket["harmful_eligible"],
                "negative_source_false_retrieval_rate": (
                    neg_bucket["false_retrievals"] / neg_bucket["negative_eligible"]
                    if neg_bucket["negative_eligible"]
                    else None
                ),
                "negative_source_eligible_cases": neg_bucket["negative_eligible"],
            }
            for source, neg_bucket in per_source.items()
        }
        for method, per_source in negative_stats.items()
    }
    return {
        "by_method_by_source": by_method_by_source,
        "negative_source_and_harmful_retrieval": negative_source_and_harmful_retrieval,
    }


def cluster_bootstrap_paired_diff(
    rows: Sequence[tuple[str, float]],
    *,
    seed: int = 42,
    n_boot: int = 5000,
) -> dict[str, float]:
    """95% CI on a paired difference, resampled by cluster (e.g. dialogue_id).

    Mirrors the cluster-bootstrap methodology already used for real ESConv
    evaluation (``esconv._cluster_bootstrap``): resample cluster KEYS with
    replacement (not individual rows), pool every row belonging to each
    resampled cluster, and report the 2.5/97.5 percentiles of the resampled
    means as the 95% CI -- the same reason real evaluation there does not
    bootstrap individual turns: turns from the same dialogue are correlated,
    so the dialogue (not the turn) is the independent unit.

    ``rows`` is ``(cluster_key, paired_diff_value)`` per unit, e.g. per
    ESConv turn: ``hybrid_hit - lexical_hit``.
    """

    rng = np.random.default_rng(seed)
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster_key, value in rows:
        clusters[cluster_key].append(value)
    keys = sorted(clusters)
    observed = float(statistics.mean(value for _key, value in rows)) if rows else 0.0
    if not keys:
        return {"mean": observed, "ci_low": observed, "ci_high": observed, "n_clusters": 0}
    resampled_means = []
    for _ in range(n_boot):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in sampled_keys for value in clusters[key]]
        resampled_means.append(float(np.mean(values)))
    return {
        "mean": observed,
        "ci_low": float(np.quantile(resampled_means, 0.025)),
        "ci_high": float(np.quantile(resampled_means, 0.975)),
        "n_clusters": len(keys),
    }


def evaluate_esconv_strategy_retrieval_quality(
    esconv_path: str,
    split_manifest_path: str,
    *,
    split: str,
    lexical_retriever: StrategyRetriever,
    hybrid_retriever: HybridStrategyRetriever,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Real ESConv-split strategy retrieval quality: lexical-only vs Hybrid.

    Mirrors the ``strategy_recall_at_k`` check already computed for real,
    paid ESConv TEST evaluation in ``esconv.run_esconv_policy_evaluation``
    (same query construction via ``context_query``, same "is gold_strategy
    among the retrieved labels" hit definition) but: (a) is diagnostic-only,
    computing both lexical-only and Hybrid rather than gating a real
    evaluation; (b) defaults to the ``"validation"`` split, never
    ``"test"`` -- ESConv test is confirmatory and must never be touched
    before the retriever design is frozen and a decision is made.

    Returns ``{"by_method": {"lexical_only": ..., "hybrid": ...},
    "paired_dialogue_cluster_bootstrap": {"hit_rate_diff": {...},
    "precision_diff": {...}}}``. The paired CIs are cluster-bootstrapped by
    ``dialogue_id`` (turns from the same dialogue are correlated, so the
    dialogue -- not the turn -- is the independent statistical unit,
    matching the same principle already applied to real ESConv evaluation).
    ``precision_diff`` only includes turns where both methods retrieved at
    least one card (precision is undefined otherwise for that method).
    """

    turns = esconv_turn_states(esconv_path, split_manifest_path, split)
    stats = {
        "lexical_only": {"hits": 0, "turns": 0, "precisions": []},
        "hybrid": {"hits": 0, "turns": 0, "precisions": []},
    }
    hit_rate_diff_rows: list[tuple[str, float]] = []
    precision_diff_rows: list[tuple[str, float]] = []
    total_turns = len(turns)
    for index, turn in enumerate(turns, start=1):
        query = context_query(
            turn["current_user_text"], turn["history"], turn["situation"]
        )
        gold_strategy = turn["gold_strategy"]
        per_method_hit: dict[str, int] = {}
        per_method_precision: dict[str, float | None] = {}
        for method, retriever in (
            ("lexical_only", lexical_retriever),
            ("hybrid", hybrid_retriever),
        ):
            retrieved = retriever.retrieve(query)
            retrieved_labels = [card.strategy_label for card in retrieved]
            bucket = stats[method]
            hit = int(gold_strategy in retrieved_labels)
            bucket["hits"] += hit
            bucket["turns"] += 1
            per_method_hit[method] = hit
            if retrieved_labels:
                precision = sum(
                    1 for label in retrieved_labels if label == gold_strategy
                ) / len(retrieved_labels)
                bucket["precisions"].append(precision)
                per_method_precision[method] = precision
            else:
                per_method_precision[method] = None
        hit_rate_diff_rows.append(
            (turn["dialogue_id"], float(per_method_hit["hybrid"] - per_method_hit["lexical_only"]))
        )
        if (
            per_method_precision["hybrid"] is not None
            and per_method_precision["lexical_only"] is not None
        ):
            precision_diff_rows.append(
                (
                    turn["dialogue_id"],
                    per_method_precision["hybrid"] - per_method_precision["lexical_only"],
                )
            )
        if progress is not None:
            progress(index, total_turns)
    by_method = {
        method: _summarize(
            hits=bucket["hits"],
            units=bucket["turns"],
            precisions=bucket["precisions"],
            query_hashes=[],
        )
        for method, bucket in stats.items()
    }
    return {
        "by_method": by_method,
        "paired_dialogue_cluster_bootstrap": {
            "cluster_key": "dialogue_id",
            "hit_rate_diff": cluster_bootstrap_paired_diff(hit_rate_diff_rows),
            "precision_diff": cluster_bootstrap_paired_diff(precision_diff_rows),
        },
    }


# ---------------------------------------------------------------------------
# Report-only EvoEmo comparisons (never feed back into calibration above, and
# never legitimate Part 4 adoption evidence -- see the two functions above
# for that).
# ---------------------------------------------------------------------------


def _session_id_by_created_session_index(user: Mapping[str, Any]) -> dict[int, str | None]:
    sessions = user.get("dialog_history") or []
    mapping: dict[int, str | None] = {0: None}
    for index, session in enumerate(sessions, 1):
        mapping[index] = str(session.get("id") or f"session_{index}")
    return mapping


def _me_related_session_topics(user: Mapping[str, Any]) -> list[tuple[str, set[str]]]:
    rows = []
    for topic in user.get("subsequent_topics") or []:
        query_text = str(topic.get("topic") or "")
        related = {str(s) for s in (topic.get("related_sessions") or [])}
        if query_text and related:
            rows.append((query_text, related))
    return rows


def compare_fixed_top_k(
    users: Sequence[Mapping[str, Any]],
    *,
    lexical_retriever: MemoryRetriever,
    hybrid_retriever: HybridMemoryRetriever,
) -> dict[str, MethodComparisonSummary]:
    """Report-only: ME retrieval hit-rate/precision, fixed top-k, both scorers.

    ``lexical_retriever``/``hybrid_retriever`` must both already be
    configured for MemorySource.ME with the same top_k so the comparison
    isolates the scoring method, not the budget. Uses each topic's
    evaluator-only text as a query stand-in (see module docstring) --
    ``query_hashes`` records a hash per topic, never the raw text.
    """

    selected = frozenset({MemorySource.ME})
    stats = {
        "lexical_only": {"hits": 0, "topics": 0, "precisions": [], "hashes": []},
        "hybrid": {"hits": 0, "topics": 0, "precisions": [], "hashes": []},
    }
    for user in users:
        items, _session_docs = build_evo_memory(user)
        session_id_by_index = _session_id_by_created_session_index(user)
        for query_text, related in _me_related_session_topics(user):
            query_hash = sha256_text(query_text)
            for name, retriever in (
                ("lexical_only", lexical_retriever),
                ("hybrid", hybrid_retriever),
            ):
                retrieved = retriever.retrieve(query_text, items, selected)
                retrieved_session_ids = {
                    session_id_by_index.get(item.created_session) for item in retrieved
                }
                bucket = stats[name]
                bucket["hits"] += int(bool(retrieved_session_ids & related))
                bucket["topics"] += 1
                bucket["hashes"].append(query_hash)
                if retrieved:
                    bucket["precisions"].append(
                        sum(
                            1
                            for item in retrieved
                            if session_id_by_index.get(item.created_session) in related
                        )
                        / len(retrieved)
                    )
    return {
        name: _summarize(
            hits=bucket["hits"],
            units=bucket["topics"],
            precisions=bucket["precisions"],
            query_hashes=bucket["hashes"],
        )
        for name, bucket in stats.items()
    }


def _lexical_rank_all_me(query: str, items: Sequence[MemoryItem]) -> list[MemoryItem]:
    """Full ME ranking by raw lexical score, same tie-break as MemoryRetriever.

    Mirrors (does not modify) the real, frozen MemoryRetriever's per-source
    scoring and deterministic tie-break, just without slicing to top_k --
    needed so the fixed-token-budget comparison can rank the complete ME
    pool for the lexical-only baseline exactly as it does for hybrid's
    ``rank_all``.
    """

    candidates = [item for item in items if item.source is MemorySource.ME]
    scored = [(lexical_score(query, item.text), item) for item in candidates]
    ranked = sorted(
        scored,
        key=lambda pair: (pair[0], pair[1].created_session, pair[1].memory_id),
        reverse=True,
    )
    return [item for _score, item in ranked]


def compare_fixed_token_budget(
    users: Sequence[Mapping[str, Any]],
    *,
    hybrid_retriever: HybridMemoryRetriever,
    token_budget: int,
) -> dict[str, MethodComparisonSummary]:
    """Report-only: ME retrieval hit-rate/precision under a fixed evidence-token
    budget (whole items only, never truncated; an overflowing item is
    skipped, not a stopping point -- see retrieve_fixed_token_budget), both
    scorers, ranking the complete ME pool first via
    ``_lexical_rank_all_me``/``hybrid_retriever.rank_all``.
    """

    selected = frozenset({MemorySource.ME})
    stats = {
        "lexical_only": {"hits": 0, "topics": 0, "precisions": [], "hashes": []},
        "hybrid": {"hits": 0, "topics": 0, "precisions": [], "hashes": []},
    }
    for user in users:
        items, _session_docs = build_evo_memory(user)
        session_id_by_index = _session_id_by_created_session_index(user)
        for query_text, related in _me_related_session_topics(user):
            query_hash = sha256_text(query_text)
            lexical_ranked = _lexical_rank_all_me(query_text, items)
            hybrid_ranked = hybrid_retriever.rank_all(query_text, items, selected)
            for name, ranked in (
                ("lexical_only", lexical_ranked),
                ("hybrid", hybrid_ranked),
            ):
                retrieved = retrieve_fixed_token_budget(
                    ranked, token_budget=token_budget, text_of=lambda item: item.text
                )
                retrieved_session_ids = {
                    session_id_by_index.get(item.created_session) for item in retrieved
                }
                bucket = stats[name]
                bucket["hits"] += int(bool(retrieved_session_ids & related))
                bucket["topics"] += 1
                bucket["hashes"].append(query_hash)
                if retrieved:
                    bucket["precisions"].append(
                        sum(
                            1
                            for item in retrieved
                            if session_id_by_index.get(item.created_session) in related
                        )
                        / len(retrieved)
                    )
    return {
        name: _summarize(
            hits=bucket["hits"],
            units=bucket["topics"],
            precisions=bucket["precisions"],
            query_hashes=bucket["hashes"],
        )
        for name, bucket in stats.items()
    }
