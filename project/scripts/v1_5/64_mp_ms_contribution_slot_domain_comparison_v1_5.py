#!/usr/bin/env python3
"""MP and MS candidate-level contribution_slot domain comparison (roadmap
item: "MP/MS/RS各自的域对比审计" -- script 60/61 only covered ME).

Learned from V15-DATA-79's correction: use the CURRENT V3 construction
(data/pm_v1_5_v3_effect_blueprint_v1 + outputs/pm_v1_5_v3_p2_exact_rank1_v4,
dated 2026-08-03), not the older 468-card PMV2 backend, as the "training
domain" side of this comparison. That export only carries the exact-Rank-1
candidate per state (not the full Top-k pool), so rank1_* fields are computed
faithfully (same code path as production) from a single-item list, and
topk_* fields necessarily degenerate to n=1 for the training-domain side --
disclosed explicitly, not hidden. The EvoEmo side uses the real multi-item
Top-k pool via the same discover_final_typed_memory_candidates() pipeline
used all session, so topk_* IS meaningful there.

Zero API calls.
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
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_3_contribution_slot_features import (  # noqa: E402
    mp_contribution_slots,
    ms_contribution_slots,
)

V3_CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
EVOEMO_PATH = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_PATH = ROOT / "outputs/pm_v1_5_v5_3_mp_ms_contribution_slot_domain_comparison_v1/report.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def training_domain_observations(component: str) -> list[dict]:
    rows = [
        r for r in _jsonl(V3_CANDIDATES)
        if r["target_component_private_not_model_input"] == component
        and r["exact_rank1_candidate"].get("candidate_present")
    ]
    observations = []
    for row in rows:
        rank1 = row["exact_rank1_candidate"]
        text = rank1.get("candidate_text") or ""
        age = rank1.get("candidate_age_sessions") or 1
        item = MemoryItem(
            memory_id=rank1["candidate_id"],
            source=MemorySource(component),
            created_session=0,
            text=text,
        )
        # Only the exact-Rank-1 item is available in this export (no full
        # Top-k pool) -- topk_* fields degenerate to n=1 here by data
        # availability, not by design; disclosed in the report below.
        if component == "MP":
            is_pref = rank1.get("compiler_subtype_hint") == "MP_PREFERENCE"
            obs = mp_contribution_slots(
                current_user_text=row["current_user_text"], candidate_text=text,
                candidate_is_preference=is_pref, source_items=[item],
                selected_items=[item], session_index=max(age, 1),
            )
        else:
            obs = ms_contribution_slots(
                current_user_text=row["current_user_text"], candidate_text=text,
                source_items=[item], selected_items=[item], session_index=max(age, 1),
            )
        observations.append(_serialize(component, obs))
    return observations


def _serialize(component: str, obs) -> dict:
    base = {
        "current_redundant": obs.current_redundant,
        "rank1_injected_tokens": obs.rank1_injected_tokens,
    }
    if component == "MP":
        base["preference_applies_to_response_act"] = obs.preference_applies_to_response_act
        base["profile_goal_needs_advice_or_arrangement"] = obs.profile_goal_needs_advice_or_arrangement
        base["candidate_is_preference"] = obs.candidate_is_preference
    else:
        base["has_specific_prior_observation"] = obs.has_specific_prior_observation
        base["continuity_request"] = obs.continuity_request
    return base


def evoemo_domain_observations(component: str) -> list[dict]:
    users = {str(u["id"]): u for u in json.loads(EVOEMO_PATH.read_text(encoding="utf-8"))}
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in _jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append({
                "user_id": row["user_id_private_analysis_only"],
                "current_user_text": rt.get("current_user_text", ""),
                "current_session_history": rt.get("current_session_history", []),
            })
    observations = []
    for state in states:
        user = users[state["user_id"]]
        session_index = len(user.get("dialog_history") or []) + 1
        items, _extra = build_evo_memory(user)
        comp_items = [it for it in items if it.source is MemorySource(component)]
        if not comp_items:
            continue
        queries = source_specific_memory_queries(state["current_user_text"], state["current_session_history"], "")
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
        )
        selected = discoveries[MemorySource(component)].selected_items
        if not selected:
            continue
        candidate_text = selected[0].text
        if component == "MP":
            is_pref = "prefer" in candidate_text.lower() or "preference" in candidate_text.lower()
            obs = mp_contribution_slots(
                current_user_text=state["current_user_text"], candidate_text=candidate_text,
                candidate_is_preference=is_pref, source_items=comp_items,
                selected_items=selected, session_index=session_index,
            )
        else:
            obs = ms_contribution_slots(
                current_user_text=state["current_user_text"], candidate_text=candidate_text,
                source_items=comp_items, selected_items=selected, session_index=session_index,
            )
        observations.append(_serialize(component, obs))
    return observations


def _rate(observations: list[dict], field: str) -> dict:
    values = [o[field] for o in observations if o.get(field) is not None]
    bools = [v for v in values if isinstance(v, bool)]
    if bools:
        return {"n": len(bools), "rate_true": sum(bools) / len(bools)}
    nums = [float(v) for v in values]
    if not nums:
        return {"n": 0}
    return {"n": len(nums), "mean": statistics.mean(nums), "median": statistics.median(nums)}


def main() -> None:
    report: dict = {
        "protocol": "pm-v1.5-v5.3-mp-ms-contribution-slot-domain-comparison-v1",
        "caveat_training_domain_topk_degenerates_to_rank1": (
            "V3 export only carries the exact-Rank-1 candidate per state, not the "
            "full Top-k pool -- topk_* aggregates are n=1 on the training side by "
            "data availability, not by design. rank1_* and semantic slot fields "
            "(the ones this report focuses on) are computed faithfully."
        ),
        "components": {},
    }
    for component in ("MP", "MS"):
        train = training_domain_observations(component)
        evo = evoemo_domain_observations(component)
        print(f"{component}: n_train={len(train)} n_evoemo={len(evo)}")
        fields = (
            ["current_redundant", "rank1_injected_tokens", "preference_applies_to_response_act",
             "profile_goal_needs_advice_or_arrangement"]
            if component == "MP" else
            ["current_redundant", "rank1_injected_tokens", "has_specific_prior_observation",
             "continuity_request"]
        )
        comparison = {}
        for field in fields:
            t, e = _rate(train, field), _rate(evo, field)
            comparison[field] = {"train": t, "evoemo": e}
            print(f"  {field}: train={t} evoemo={e}")
        report["components"][component] = {
            "n_train": len(train), "n_evoemo": len(evo), "comparison": comparison,
        }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, report)
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
