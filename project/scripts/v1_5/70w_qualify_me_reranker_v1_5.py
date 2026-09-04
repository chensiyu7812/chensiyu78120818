#!/usr/bin/env python3
"""W6: zero-API ME Rank-1 reranker qualification comparison.

Reuses the existing 32 ME superdomain states (script 68 --
outputs/pm_v1_5_v5_3_me_superdomain_v1/states_all.json): same 7-candidate
catalog, same owner (single synthetic user per state), same causal boundary
(created_session vs session_index) as when they were built. Does NOT
materialize a new P2 blueprint and does NOT generate any paired reply.

Exactly three rerankers, fixed BEFORE reading any result (no fourth method,
no reweighting after seeing scores):

  1. current_production -- discover_final_typed_memory_candidates() with
     me_subtype_hints() (lexical content-match tier + typed_tier tie-break),
     the same real production candidate discovery path used all session.
  2. bge_m3 -- pure BGE-M3 cosine over the full 7-item pool, no pre-filter
     (v1_5_v5_3_semantic_me_retrieval.rank_me_candidates_bge).
  3. hybrid_filter_then_bge -- strictly-prior + compiler-valid filter, then
     BGE-M3 cosine over the survivors
     (v1_5_v5_3_semantic_me_retrieval.rank_me_candidates_hybrid).

32 states are NOT treated as 32 independent users for significance testing
-- results are reported descriptively, mainly grouped by family (8 families,
4 states each). Zero API calls; no quality/risk/outcome read; no PM trained.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
    me_subtype_hints,
)
from metacom_pm.v1_5_v5_3_semantic_me_retrieval import (  # noqa: E402
    rank_me_candidates_bge,
    rank_me_candidates_hybrid,
)

STATES_PATH = ROOT / "outputs/pm_v1_5_v5_3_me_superdomain_v1/states_all.json"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_me_reranker_qualification_v1"

UNRELATED_MARKERS = ("garden project", "longer commute", "budget reset")

METHODS = ("current_production", "bge_m3", "hybrid_filter_then_bge")


def _load_states() -> list[dict]:
    return json.loads(STATES_PATH.read_text(encoding="utf-8"))


def _items_from_state(state: dict) -> list[MemoryItem]:
    return [
        MemoryItem(memory_id=it["memory_id"], source=MemorySource.ME,
                   created_session=it["created_session"], text=it["text"])
        for it in state["items"]
    ]


def _is_unrelated(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in UNRELATED_MARKERS)


def _rank1_current_production(state: dict, items: list[MemoryItem]) -> tuple[str | None, float]:
    start = time.perf_counter()
    queries = source_specific_memory_queries(state["current_user_text"], [], "")
    hints = me_subtype_hints(items)
    discoveries = discover_final_typed_memory_candidates(
        queries=queries, items=items, source_metadata=hints, session_index=state["session_index"],
    )
    selected = discoveries[MemorySource.ME].selected_items
    elapsed = time.perf_counter() - start
    return (selected[0].memory_id if selected else None), elapsed


def _rank1_bge(state: dict, items: list[MemoryItem], encoder) -> tuple[str | None, float]:
    start = time.perf_counter()
    ranked = rank_me_candidates_bge(state["current_user_text"], items, encoder=encoder)
    elapsed = time.perf_counter() - start
    return (ranked[0].item.memory_id if ranked else None), elapsed


def _rank1_hybrid(state: dict, items: list[MemoryItem], encoder) -> tuple[str | None, float]:
    start = time.perf_counter()
    ranked = rank_me_candidates_hybrid(
        state["current_user_text"], items, encoder=encoder, session_index=state["session_index"],
    )
    elapsed = time.perf_counter() - start
    return (ranked[0].item.memory_id if ranked else None), elapsed


def _text_by_id(state: dict, memory_id: str | None) -> str | None:
    if memory_id is None:
        return None
    for it in state["items"]:
        if it["memory_id"] == memory_id:
            return it["text"]
    return None


def main() -> None:
    states = _load_states()
    print(f"loaded {len(states)} states from {STATES_PATH}")

    print("loading BGE-M3 encoder (shared instance for both bge_m3 and hybrid methods -- "
          "this measures the marginal cost of reusing MS's already-loaded encoder for ME, "
          "not a second model load)...")
    from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder
    load_start = time.perf_counter()
    encoder = BgeM3Encoder()
    encoder_load_seconds = time.perf_counter() - load_start
    print(f"encoder loaded in {encoder_load_seconds:.2f}s")

    per_state: list[dict] = []
    for state in states:
        items = _items_from_state(state)
        target_id = state["target_item_id"]

        rank1_prod, t_prod = _rank1_current_production(state, items)
        rank1_bge, t_bge = _rank1_bge(state, items, encoder)
        rank1_hybrid, t_hybrid = _rank1_hybrid(state, items, encoder)

        row = {"state_id": state["state_id"], "family": state["family"],
               "negative_kind": state["negative_kind"], "target_item_id": target_id}
        for method, rank1_id, elapsed in (
            ("current_production", rank1_prod, t_prod),
            ("bge_m3", rank1_bge, t_bge),
            ("hybrid_filter_then_bge", rank1_hybrid, t_hybrid),
        ):
            text = _text_by_id(state, rank1_id)
            row[f"{method}_rank1_id"] = rank1_id
            row[f"{method}_matches_intended_target"] = rank1_id == target_id
            row[f"{method}_compiler_valid"] = (
                compile_atomic_reusable_outcome(text) is not None if text else False
            )
            row[f"{method}_completely_unrelated"] = (
                _is_unrelated(text) if text else False
            )
            row[f"{method}_owner_violation"] = False  # structurally impossible: single-owner pools, see doc
            row[f"{method}_future_violation"] = (
                rank1_id is not None
                and next(it["created_session"] for it in state["items"] if it["memory_id"] == rank1_id)
                >= state["session_index"]
            )
            row[f"{method}_latency_seconds"] = elapsed
        per_state.append(row)

    def _rate(rows: list[dict], key: str) -> float:
        return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0

    summary_overall = {}
    for method in METHODS:
        summary_overall[method] = {
            "exact_intended_rank1_rate": _rate(per_state, f"{method}_matches_intended_target"),
            "rank1_compiler_valid_rate": _rate(per_state, f"{method}_compiler_valid"),
            "completely_unrelated_rate": _rate(per_state, f"{method}_completely_unrelated"),
            "owner_violations": sum(r[f"{method}_owner_violation"] for r in per_state),
            "future_violations": sum(r[f"{method}_future_violation"] for r in per_state),
            "mean_latency_seconds": sum(r[f"{method}_latency_seconds"] for r in per_state) / len(per_state),
        }

    families = sorted({r["family"] for r in per_state})
    by_family = {}
    for family in families:
        rows = [r for r in per_state if r["family"] == family]
        by_family[family] = {
            method: {
                "exact_intended_rank1_rate": _rate(rows, f"{method}_matches_intended_target"),
                "n": len(rows),
            }
            for method in METHODS
        }

    pairwise = {}
    method_pairs = [
        ("current_production", "bge_m3"),
        ("current_production", "hybrid_filter_then_bge"),
        ("bge_m3", "hybrid_filter_then_bge"),
    ]
    for a, b in method_pairs:
        win = loss = tie = 0
        for r in per_state:
            a_ok = r[f"{a}_matches_intended_target"]
            b_ok = r[f"{b}_matches_intended_target"]
            if a_ok and not b_ok:
                win += 1
            elif b_ok and not a_ok:
                loss += 1
            else:
                tie += 1
        pairwise[f"{a}_vs_{b}"] = {"a_wins": win, "b_wins": loss, "tie": tie}

    report = {
        "protocol": "pm-v1.5-v5.3-me-reranker-qualification-v1",
        "n_states": len(states),
        "n_families": len(families),
        "note": (
            "32 states are 8 families x 4 current-turn variants, NOT 32 "
            "independent users -- no significance test is computed, only "
            "descriptive rates, mainly grouped by family."
        ),
        "encoder_load_seconds": encoder_load_seconds,
        "encoder_reuse_note": (
            "bge_m3 and hybrid_filter_then_bge share ONE loaded BgeM3Encoder "
            "instance in this run. If MS already holds this same encoder in "
            "memory (v1_5_v5_3_semantic_ms_retrieval.BgeM3Encoder, identical "
            "snapshot), the marginal cost of also using it for ME is the "
            "per-call encode() time below, not a second ~2GB model load."
        ),
        "summary_overall": summary_overall,
        "by_family": by_family,
        "pairwise_win_loss_tie_on_exact_intended_rank1": pairwise,
        "per_state": per_state,
        "api_calls": 0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "report.json", report)

    print(f"\nn_states={len(states)} n_families={len(families)}")
    print(f"encoder_load_seconds={encoder_load_seconds:.2f}")
    for method, stats in summary_overall.items():
        print(f"\n{method}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print("\nby_family (exact_intended_rank1_rate):")
    for family, methods in by_family.items():
        print(f"  {family}: " + ", ".join(f"{m}={s['exact_intended_rank1_rate']:.2f}" for m, s in methods.items()))
    print("\npairwise win/loss/tie (on exact intended rank1):")
    for pair, stats in pairwise.items():
        print(f"  {pair}: {stats}")
    print(f"\nfull report written to {OUT_DIR}")


if __name__ == "__main__":
    main()
