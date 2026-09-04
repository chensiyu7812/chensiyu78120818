"""MP verification against the real production selector (not bare lexical).

Status: EXPLORATORY. Answers: does the real
discover_final_typed_memory_candidates selector have a real problem with MP,
now that we know it's not bare lexical_score? (The earlier "MP needs
hand-written keyword rules" recommendation was derived from a bare-lexical
comparison and has been retracted -- see
PM_V1_5_V5_3_MS_RETRIEVAL_SCORING_DIAGNOSTIC_ZH.md's 2026-08-06 correction
note.)

EvoEmo's build_evo_memory() never sets mp_subtype metadata (checked: no
MP_PREFERENCE/MP_PROFILE distinction exists in evoemo.py at all), so the
selector's MP_PREFERENCE special-case (preference_scope_match_level) never
fires for EvoEmo data -- MP ranking here is content-match-tier + lexical
tie-break only, structurally identical in shape to MS's ranking, just over a
much smaller (~6-8 item) static candidate pool per user.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_mp_verification_v1"


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
    session_index_by_user = {uid: len(u.get("dialog_history") or []) + 1 for uid, u in users.items()}

    rows_out = []
    hard_violations = 0
    for state in states:
        uid = state["user_id"]
        user = users[uid]
        session_index = session_index_by_user[uid]
        items, _extra = build_evo_memory(user)
        mp_items = [it for it in items if it.source is MemorySource.MP]

        queries = source_specific_memory_queries(
            state["current_user_text"],
            state["current_session_history"],
            state["current_session_summary"],
        )
        try:
            discoveries = discover_final_typed_memory_candidates(
                queries=queries, items=items, source_metadata={}, session_index=session_index,
            )
        except ValueError as exc:
            hard_violations += 1
            rows_out.append({"state_id": state["state_id"], "user_id": uid, "hard_constraint_violation": str(exc)})
            continue

        mp_selected = discoveries[MemorySource.MP].selected_items
        rows_out.append(
            {
                "state_id": state["state_id"],
                "user_id": uid,
                "mp_candidate_pool_size": len(mp_items),
                "mp_pool_texts": [it.text for it in mp_items],
                "current_user_text": state["current_user_text"],
                "mp_selected_count": len(mp_selected),
                "mp_selected_texts": [it.text for it in mp_selected],
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mp_selection_all_states.jsonl"
    with out_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = [r for r in rows_out if "hard_constraint_violation" not in r]
    zero_selected = sum(1 for r in scored if r["mp_selected_count"] == 0)
    nonzero_selected = len(scored) - zero_selected
    print(f"states: {len(states)}, scored: {len(scored)}, hard_violations: {hard_violations}")
    print(f"MP selected nothing (all pool items below content-match floor): {zero_selected}")
    print(f"MP selected >=1 candidate: {nonzero_selected}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
