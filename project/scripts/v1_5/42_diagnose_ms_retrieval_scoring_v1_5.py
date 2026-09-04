"""Diagnostic: does swapping MS session-retrieval scoring from lexical to BGE-M3 help?

Status: EXPLORATORY / NOT YET A FROZEN CONTRACT ARTIFACT. Produced by an
independent audit session on 2026-08-05, outside the V5.3 machine-contract
pipeline. It does not carry a protocol hash, is not bound into
data/pm_v1_5_contracts, and must not be cited as a formal V5.3 result until a
maintainer re-runs it through the project's normal freeze process.

What this does
---------------
The real, frozen `MemoryRetriever` (src/metacom_pm/retrieval.py) ranks MP/MS/ME
candidates with `lexical_score` -- raw term-frequency cosine, no IDF, no
stopword weighting. EvoEmo's synthetic dialogues reuse the same therapeutic
register across every session regardless of topic ("anxious", "overwhelmed",
"struggling to balance", ...), so this scorer is structurally prone to ranking
a topically wrong candidate #1 whenever it happens to share more generic
affect vocabulary with the query than the topically-correct candidate does.

This script does NOT re-run BGE-M3 encoding (no new model calls). It reuses
the BGE-M3 embeddings and gold relevant-session labels the project already
computed and froze for the official ES-MemEval session-retrieval metric
(outputs/pm_v1_5_v5_2_external_e3_retrieval_v1), which draws on the same
data/external/evo_emo.json user pool that EvoEmo's MS candidates come from.
It adds two zero-marginal-cost comparison scorers on the identical
341-question, identical-document-pool task:

  1. lexical_score       -- exactly what the production MemoryRetriever uses today
  2. content_word_match_level -- already implemented in metacom_pm.text, already
     computed as a Step1 feature, but never used for ranking
  3. (reference) BGE-M3 -- already-frozen official numbers, not recomputed here

Recall@4 / nDCG@4 are computed with the exact same tie-break and metric
definitions as the official E3 script (ndcg_at_k), restricted to the same 341
"retrieval_metric_comparable" questions, so the three rows are apples-to-apples.

Why this is not the same situation as V15-OBS-19 (BGE lost to lexical for RS)
-------------------------------------------------------------------------
V15-OBS-19 tested embedding retrieval for Strategy Bank *cards* -- short,
technique-labeled text where exact terminology plausibly matters. This script
tests it for *session-summary/episodic* retrieval -- long natural-language
narrative where topical/semantic match, not literal word overlap, is what
distinguishes "this user's wedding-planning session" from "this user's job-
search session". The two content types should not be assumed to behave the
same way, and this script exists so nobody has to assume either direction.

Usage: run with either python3 (stdlib-only; no BGE re-encoding) --
no formal venv requirement, no API calls, no new hashes to freeze.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.text import content_word_match_level, lexical_score  # noqa: E402

E2_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1"

BGE_M3_OFFICIAL_REFERENCE = {
    "mean_recall_at_4": 0.6962854349951124,
    "mean_ndcg_at_4": 0.5997130750238503,
    "comparable_questions": 341,
    "source": "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1/retrieval_report.json"
    " (qa.official_session_retrieval)",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ndcg_at_k(selected: list[str], relevant: set[str], k: int) -> float | None:
    """Identical definition to the official E3 script's ndcg_at_k."""

    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, session_id in enumerate(selected[:k], start=1)
        if session_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else None


def recall_at_k(selected: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(selected[:k]) & relevant) / len(relevant)


def rank_top4(query: str, docs: list[dict], scorer) -> list[str]:
    """Same tie-break as the official BGE ranker: score desc, original
    (chronological) list position asc, session_id asc."""

    indexed = list(enumerate(docs))
    ranked = sorted(
        indexed,
        key=lambda pair: (-scorer(query, pair[1]["text"]), pair[0], pair[1]["session_id"]),
    )
    return [doc["session_id"] for _, doc in ranked[:4]]


def main() -> None:
    calls = load_jsonl(E2_DIR / "qa_logical_call_plan_private.jsonl")
    metrics = {row["question_id"]: row for row in load_jsonl(E3_DIR / "qa_session_retrieval_metrics_private.jsonl")}

    qa_calls = [c for c in calls if c.get("condition") == "official_session_rag_top4"]

    per_scorer_rows: dict[str, list[tuple[float | None, float | None]]] = {
        "lexical_score_current_production": [],
        "content_word_match_level_unused_for_ranking": [],
    }
    n_comparable = 0
    for call in qa_calls:
        metric_row = metrics.get(call["question_id"])
        if not metric_row or not metric_row.get("retrieval_metric_comparable"):
            continue
        n_comparable += 1
        relevant = set(metric_row["relevant_session_ids"])
        query = call["question"]
        docs = call["same_user_session_documents"]

        lex_top4 = rank_top4(query, docs, lexical_score)
        per_scorer_rows["lexical_score_current_production"].append(
            (recall_at_k(lex_top4, relevant, 4), ndcg_at_k(lex_top4, relevant, 4))
        )

        cw_top4 = rank_top4(query, docs, content_word_match_level)
        per_scorer_rows["content_word_match_level_unused_for_ranking"].append(
            (recall_at_k(cw_top4, relevant, 4), ndcg_at_k(cw_top4, relevant, 4))
        )

    assert n_comparable == BGE_M3_OFFICIAL_REFERENCE["comparable_questions"], (
        f"comparable-question count drifted from the frozen E3 report: "
        f"{n_comparable} vs {BGE_M3_OFFICIAL_REFERENCE['comparable_questions']}"
    )

    report: dict = {
        "status": "EXPLORATORY_NOT_A_FROZEN_CONTRACT_ARTIFACT",
        "purpose": (
            "Test whether the production MemoryRetriever's lexical_score is the "
            "bottleneck behind EvoEmo's low MS candidate topical-relevance rate, "
            "by comparing it against the already-frozen BGE-M3 numbers on the "
            "identical 341-question ES-MemEval session-retrieval task -- zero "
            "new API/model calls."
        ),
        "task": "official_session_rag_top4 (ES-MemEval session retrieval), same document pool as EvoEmo MS candidates",
        "comparable_questions": n_comparable,
        "scorers": {},
        "reference_bge_m3": BGE_M3_OFFICIAL_REFERENCE,
        "caveat_v15_obs_19": (
            "V15-OBS-19 found BGE lost to lexical for Strategy Bank (RS) card "
            "retrieval -- a different content type (short technique cards vs "
            "long narrative session summaries). Do not treat that finding as "
            "evidence against trying embeddings here; this script exists so the "
            "MS-specific question gets its own measurement instead of an analogy."
        ),
    }
    for name, rows in per_scorer_rows.items():
        recalls = [r for r, _ in rows if r is not None]
        ndcgs = [n for _, n in rows if n is not None]
        report["scorers"][name] = {
            "mean_recall_at_4": sum(recalls) / len(recalls),
            "mean_ndcg_at_4": sum(ndcgs) / len(ndcgs),
            "n": len(recalls),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ms_retrieval_scoring_diagnostic_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
