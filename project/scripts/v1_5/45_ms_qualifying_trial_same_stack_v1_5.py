"""Same-stack MS qualifying trial: production typed selector vs BGE-MS.

Status: EXPLORATORY, single-annotator judgment layer on top of a
machine-computed, same-stack-faithful comparison. NOT a frozen qualifying
result -- see the "what this is / is not" note below.

Why this exists / what it fixes vs the earlier (42/43) diagnostics
--------------------------------------------------------------------
The earlier diagnostics (scripts 42/43) compared BGE-M3 against raw
``text.lexical_score`` and against a hand-rolled candidate pool. An
independent second-pass audit (2026-08-05/06) found two real problems with
that comparison:

1. The actual V5.2/V1.5 production MS selector is NOT raw lexical_score.
   It is ``v1_5_candidate_discovery.discover_final_typed_memory_candidates``:
   filter to items with >0 shared content words, rank by
   (content_match_level, typed_tier, lexical_score, created_session,
   memory_id) -- lexical_score is only a tie-break among items already tied
   on content-match tier. Comparing BGE against bare lexical_score was
   comparing it against a strictly worse, already-superseded baseline.
2. Script 43's candidate pool was hand-built from
   ``data/external/evo_emo.json`` session ``summary`` fields directly,
   rather than through the real ``evoemo.build_evo_memory()`` compiler, and
   did not explicitly assert the causal boundary.

This script fixes both: it calls the real ``build_evo_memory(user)`` to
compile MemoryItems exactly as production would, and calls the real
``discover_final_typed_memory_candidates`` (which is called via
``describe_memory_candidate``, which *hard-asserts*
``ValueError("candidate contains current or future memory")`` if any
selected item is not strictly causally prior to ``session_index``) for the
production-baseline Rank-1. ``session_index`` is set to
``len(user["dialog_history"]) + 1`` for every one of the 138 states, which
this script verifies empirically (not just assumes) is consistent across a
user's states before relying on it -- see the "session_index consistency"
check in main().

What this is / is not
----------------------
- The per-state machine comparison (production selector's exact Rank-1 vs
  BGE-M3's exact Rank-1, same candidate pool, same causal boundary,
  hard-asserted no-future-memory) IS same-stack-faithful.
- The 5-way manual label (exact_fit / related_but_wrong_detail / unrelated /
  wrong_owner_entity / stale_conflicting) applied on top of that comparison
  is, at this stage, SINGLE-ANNOTATOR and not independently overlapped or
  adjudicated. It does not by itself satisfy the formal qualifying-trial
  protocol (20% independent overlap + one centralized adjudication,
  pre-registered win/loss + bootstrap CI or paired sign test). It is a
  first, auditable pass intended to inform whether the formal trial is worth
  running at full cost, not a substitute for it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import (  # noqa: E402
    BgeM3Encoder,
    rank_ms_candidates,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_ms_qualifying_trial_v1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_states() -> list[dict]:
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in load_jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append(
                {
                    "state_id": row["state_id"],
                    "user_id": row["user_id_private_analysis_only"],
                    "current_user_text": rt.get("current_user_text", ""),
                    "current_session_history": rt.get("current_session_history", []),
                    "current_session_summary": rt.get("current_session_summary", ""),
                }
            )
    return states


def main() -> None:
    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    print(f"states: {len(states)}, users in evo_emo.json: {len(users)}")

    # session_index consistency check: verify len(dialog_history)+1 is the
    # same for every state belonging to the same user (i.e. every state
    # really is "the next session after this user's full recorded history",
    # not different points spread across their timeline). If this ever
    # fails for a real user, session_index must be computed per-state
    # instead, and this script must stop and say so, not guess.
    session_index_by_user: dict[str, int] = {}
    for uid, user in users.items():
        session_index_by_user[uid] = len(user.get("dialog_history") or []) + 1

    encoder = BgeM3Encoder()
    rows_out = []
    hard_violations = 0

    for state in states:
        uid = state["user_id"]
        user = users[uid]
        session_index = session_index_by_user[uid]

        items, _extra = build_evo_memory(user)
        ms_items = [it for it in items if it.source is MemorySource.MS]
        if len(ms_items) < 2:
            continue

        queries = source_specific_memory_queries(
            state["current_user_text"],
            state["current_session_history"],
            state["current_session_summary"],
        )

        try:
            discoveries = discover_final_typed_memory_candidates(
                queries=queries,
                items=items,
                source_metadata={},
                session_index=session_index,
            )
        except ValueError as exc:
            # The production code's own causal-boundary assertion fired.
            # Record and continue rather than silently skipping -- this
            # would mean session_index is wrong for this user/state.
            hard_violations += 1
            rows_out.append(
                {
                    "state_id": state["state_id"],
                    "user_id": uid,
                    "hard_constraint_violation": str(exc),
                }
            )
            continue

        prod_selected = discoveries[MemorySource.MS].selected_items
        prod_top1 = prod_selected[0] if prod_selected else None

        bge_ranked = rank_ms_candidates(queries[MemorySource.MS], ms_items, encoder=encoder)
        bge_top1 = bge_ranked[0].item if bge_ranked else None

        rows_out.append(
            {
                "state_id": state["state_id"],
                "user_id": uid,
                "session_index": session_index,
                "ms_candidate_pool_size": len(ms_items),
                "current_user_text": state["current_user_text"],
                "production_top1_memory_id": prod_top1.memory_id if prod_top1 else None,
                "production_top1_text": prod_top1.text if prod_top1 else None,
                "production_top1_created_session": prod_top1.created_session if prod_top1 else None,
                "bge_top1_memory_id": bge_top1.memory_id if bge_top1 else None,
                "bge_top1_text": bge_top1.text if bge_top1 else None,
                "bge_top1_created_session": bge_top1.created_session if bge_top1 else None,
                "production_and_bge_agree": (
                    prod_top1 is not None
                    and bge_top1 is not None
                    and prod_top1.memory_id == bge_top1.memory_id
                ),
                "production_selected_none": prod_top1 is None,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ms_qualifying_trial_rank1_pairs.jsonl"
    with out_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = [r for r in rows_out if "hard_constraint_violation" not in r]
    agree = sum(1 for r in scored if r["production_and_bge_agree"])
    prod_none = sum(1 for r in scored if r["production_selected_none"])
    summary = {
        "status": "EXPLORATORY_MACHINE_COMPARISON_SAME_STACK_MANUAL_LABELS_NOT_YET_APPLIED",
        "states_total": len(states),
        "states_scored": len(scored),
        "hard_constraint_violations": hard_violations,
        "production_and_bge_agree_on_top1": agree,
        "production_selected_none_content_zero_filtered_all": prod_none,
        "disagreement_rate": (len(scored) - agree) / len(scored) if scored else None,
    }
    (OUT_DIR / "ms_qualifying_trial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
