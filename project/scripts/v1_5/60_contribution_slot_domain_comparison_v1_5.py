#!/usr/bin/env python3
"""Legacy-backend contribution-slot transport diagnostic.

Important scope correction (2026-08-06): the so-called training domain in
this script is the obsolete 468-card PMV2 backend, not the current V3 exact
Rank-1 effect construction and not future V5.3 FIT.  Its zero
``past_action_result`` support proves that the legacy backend must not be
reused for the new ME head.  It does *not* prove that the current formal
construction lacks ME support.  See script 61 for the current-construction
audit.

This is deliberately NOT a repeat of the earlier PMV2FeatureBuilder OOD
audit (PM_V1_5_V5_3_MASTER_STATUS_20260806_ZH.md section 9): that measured
catalog-summary statistics (count/age ratios) on a representation V5.3's
plan already deprecates. This measures the actual candidate-level slots the
plan specifies Step1 should see, computed by the SAME shared retrieval
(discover_final_typed_memory_candidates) and the SAME contribution_slot
module in both domains -- the real "same interface, different content"
comparison this whole investigation has been trying to reach.

ME only, for now (RS is structurally different -- no memory catalog -- and
MP/MS need their own follow-up; this establishes the method on the
component with the most already-validated tooling: me_subtype_hints,
compile_atomic_reusable_outcome, the real 138-state EvoEmo panel from
script 55).

Zero API calls. Training domain: the same 468-card memory_backend.jsonl +
pm_v2_states.jsonl used throughout this investigation. EvoEmo domain: the
same 138-state qualification+lockbox panel used all session.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.pm_v2_contracts import PMV2Split  # noqa: E402
from metacom_pm.pm_v2_data import load_states  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_2_atomic_memory import me_subtype_hints  # noqa: E402
from metacom_pm.v1_5_v5_3_contribution_slot_features import (  # noqa: E402
    me_contribution_slots,
)

SYNTHETIC_STATES = ROOT / "data/pm_v1_5_formal_v8_18_duplicate_repair_candidate/pm_v2_states.jsonl"
SYNTHETIC_BACKEND = ROOT / "data/pm_v1_5_formal_v8_18_duplicate_repair_candidate/memory_backend.jsonl"
EVOEMO_PATH = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_PATH = ROOT / "outputs/pm_v1_5_contribution_slot_domain_comparison_v1.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _me_observation_for_state(
    *, current_user_text: str, items: list[MemoryItem], session_index: int, history=(),
) -> dict | None:
    me_items = [it for it in items if it.source is MemorySource.ME]
    if not me_items:
        return None
    queries = source_specific_memory_queries(current_user_text, history, "")
    hints = me_subtype_hints(me_items)
    discoveries = discover_final_typed_memory_candidates(
        queries=queries, items=items, source_metadata=hints, session_index=session_index,
    )
    selected = discoveries[MemorySource.ME].selected_items
    if not selected:
        return None
    candidate_text = selected[0].text
    obs = me_contribution_slots(
        current_user_text=current_user_text, candidate_text=candidate_text,
        source_items=me_items, selected_items=selected, session_index=session_index,
    )
    return {
        "rank1_relative_age": obs.rank1_relative_age,
        "rank1_injected_tokens": obs.rank1_injected_tokens,
        "topk_top1_lexical_relevance": obs.topk_top1_lexical_relevance,
        "topk_top1_top2_lexical_margin": obs.topk_top1_top2_lexical_margin,
        "topk_median_relative_age": obs.topk_median_relative_age,
        "topk_incremental_injected_tokens": obs.topk_incremental_injected_tokens,
        "topk_capacity_fraction": obs.topk_capacity_fraction,
        "current_redundant": obs.current_redundant,
        "past_action_result": obs.past_action_result,
        "current_action_invitation": obs.current_action_invitation,
    }


def training_domain_observations() -> list[dict]:
    states = load_states(SYNTHETIC_STATES)
    train = [s for s in states if s.split == PMV2Split.TRAIN]
    backend_rows = {row["card_id"]: row for row in load_jsonl(SYNTHETIC_BACKEND)}
    observations = []
    for state in train:
        row = backend_rows.get(state.card_id)
        if row is None:
            continue
        items = [
            MemoryItem(memory_id=it["memory_id"], source=MemorySource(it["source"]),
                       created_session=it["created_session"], text=it["text"])
            for it in row["items"]
        ]
        obs = _me_observation_for_state(
            current_user_text=state.current_user_text, items=items,
            session_index=state.session_index,
        )
        if obs is not None:
            observations.append(obs)
    return observations


def evoemo_domain_observations() -> list[dict]:
    users = {str(u["id"]): u for u in json.loads(EVOEMO_PATH.read_text(encoding="utf-8"))}
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in load_jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append({
                "user_id": row["user_id_private_analysis_only"],
                "current_user_text": rt.get("current_user_text", ""),
                "current_session_history": rt.get("current_session_history", []),
                "current_session_summary": rt.get("current_session_summary", ""),
            })
    observations = []
    for state in states:
        uid = state["user_id"]
        user = users[uid]
        session_index = len(user.get("dialog_history") or []) + 1
        items, _extra = build_evo_memory(user)
        obs = _me_observation_for_state(
            current_user_text=state["current_user_text"], items=items,
            session_index=session_index, history=state["current_session_history"],
        )
        if obs is not None:
            observations.append(obs)
    return observations


def _summarize(observations: list[dict], field: str) -> dict:
    values = [o[field] for o in observations if o[field] is not None]
    bool_values = [v for v in values if isinstance(v, bool)]
    if bool_values:
        return {"n": len(bool_values), "rate_true": sum(bool_values) / len(bool_values)}
    numeric = [float(v) for v in values]
    if not numeric:
        return {"n": 0}
    return {
        "n": len(numeric), "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric), "max": max(numeric),
        "stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
    }


def main() -> None:
    print("computing training-domain ME contribution_slot observations...")
    train_obs = training_domain_observations()
    print(f"  n_train_states_with_me_candidate={len(train_obs)}")

    print("computing EvoEmo-domain ME contribution_slot observations...")
    evoemo_obs = evoemo_domain_observations()
    print(f"  n_evoemo_states_with_me_candidate={len(evoemo_obs)}")

    fields = [
        "rank1_relative_age", "rank1_injected_tokens",
        "topk_top1_lexical_relevance", "topk_top1_top2_lexical_margin",
        "topk_median_relative_age", "topk_incremental_injected_tokens", "topk_capacity_fraction",
        "current_redundant", "past_action_result", "current_action_invitation",
    ]
    comparison = {}
    for field in fields:
        t = _summarize(train_obs, field)
        e = _summarize(evoemo_obs, field)
        comparison[field] = {"train": t, "evoemo": e}
        print(f"\n{field}:")
        print(f"  train : {t}")
        print(f"  evoemo: {e}")

    write_json(OUT_PATH, {
        "protocol": "pm-v1.5-contribution-slot-domain-comparison-v1",
        "status": "LEGACY_PMV2_BACKEND_DIAGNOSTIC_NOT_CURRENT_V5_3_READINESS",
        "synthetic_source_role": "obsolete_468_card_pmv2_backend",
        "current_construction_audit": (
            "scripts/v1_5/61_audit_v5_3_me_construction_support_v1_5.py"
        ),
        "n_train": len(train_obs), "n_evoemo": len(evoemo_obs),
        "comparison": comparison,
    })
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
