#!/usr/bin/env python3
"""Zero-API audit of the *current* V3/V5.3 ME construction surface.

Script 60 compared EvoEmo with the legacy 468-card PMV2 backend.  That was
useful to prove the legacy backend cannot train the new candidate-level ME
head, but it is not the construction surface used by the current V3 effect
blueprint.  This audit reads the already materialized, outcome-blind exact
Rank-1 rows from the V3 blueprint and answers two narrower questions:

1. Does the current construction already provide compiler-valid, correctly
   bound ME_REUSABLE_OUTCOME candidates across FIT/confirmation/sealed?
2. Does the current transparent action-invitation observer recognize the
   action-ready goals that the private construction blueprint intended?

No quality/risk/function outcomes are read.  Construction intent is used for
audit only and is never treated as worth_opening gold.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import sha256_file, write_json  # noqa: E402
from metacom_pm.v1_5_strategy_rag_runtime import observable_flags  # noqa: E402
from metacom_pm.v1_5_v5_3_action_readiness import (  # noqa: E402
    ActionReadiness,
    observe_action_readiness,
)
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
)


PROTOCOL = "pm-v1.5-v5.3-me-current-construction-support-audit-v1"
BLUEPRINT = (
    ROOT
    / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
)
CANDIDATES = (
    ROOT
    / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
)
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_me_construction_support_audit_v1"

ACTION_READY_GOALS = {
    "one_optional_step",
    "avoid_known_failure",
    "choose_safe_alternative",
    "try_reversible_experiment",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def run(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    blueprint = {
        row["blueprint_row_id"]: row
        for row in _jsonl(BLUEPRINT)
        if row["target_component"] == "ME"
    }
    candidates = [
        row
        for row in _jsonl(CANDIDATES)
        if row["target_component_private_not_model_input"] == "ME"
    ]

    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, str]] = []
    effect_rows = 0

    for row in candidates:
        row_id = row["state_id"]
        design = blueprint[row_id]
        split = row["split_private_not_model_input"]
        counter = per_split[split]
        counter["rows"] += 1

        candidate = row["exact_rank1_candidate"]
        text = candidate.get("candidate_text") or ""
        compiler_valid = compile_atomic_reusable_outcome(text) is not None
        if compiler_valid:
            counter["compiler_valid_rank1"] += 1
        if candidate.get("compiler_subtype_hint") == design["candidate_subtype_target"]:
            counter["subtype_matches_blueprint"] += 1

        audit = row["private_retrieval_audit_not_model_input"]
        selected_ids = audit.get("selected_memory_ids") or []
        target_id = audit.get("target_construction_item_id")
        if selected_ids and selected_ids[0] == target_id == candidate.get("candidate_id"):
            counter["rank1_binding_exact"] += 1

        if design["track"] != "COMPONENT_EFFECT":
            continue
        effect_rows += 1
        counter["effect_rows"] += 1
        action_ready = design.get("explicit_current_goal") in ACTION_READY_GOALS
        if action_ready:
            counter["blueprint_action_ready"] += 1
        flags = observable_flags(
            [{"role": "user", "content": row["current_user_text"]}]
        )
        legacy_invitation = flags["explicit_advice_welcome"]
        if legacy_invitation:
            counter["legacy_observer_action_invitation"] += 1
        readiness = observe_action_readiness(row["current_user_text"])
        v53_invitation = readiness is ActionReadiness.INVITES_ACTION
        if v53_invitation:
            counter["v53_observer_action_invitation"] += 1
        if action_ready != v53_invitation:
            counter["v53_action_observer_mismatch"] += 1
            failures.append(
                {
                    "state_id": row_id,
                    "split": split,
                    "current_user_text": row["current_user_text"],
                    "blueprint_goal": str(design.get("explicit_current_goal")),
                    "legacy_observer_explicit_advice_welcome": str(
                        legacy_invitation
                    ).lower(),
                    "v53_action_readiness": readiness.value,
                }
            )

    serial = {split: dict(counts) for split, counts in sorted(per_split.items())}
    effect_compiler_valid = sum(
        counts.get("compiler_valid_rank1", 0)
        for split, counts in per_split.items()
        if split != "ELIGIBILITY_AUDIT"
    )
    effect_action_ready = sum(
        counts.get("blueprint_action_ready", 0) for counts in per_split.values()
    )
    effect_legacy_observer_ready = sum(
        counts.get("legacy_observer_action_invitation", 0)
        for counts in per_split.values()
    )
    effect_v53_observer_ready = sum(
        counts.get("v53_observer_action_invitation", 0)
        for counts in per_split.values()
    )

    result = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_ZERO_API_CURRENT_CONSTRUCTION_AUDIT",
        "source_role": (
            "consumed V3/V5.2 development construction reference; not fresh V5.3 gold"
        ),
        "inputs": {
            str(BLUEPRINT.relative_to(ROOT)): sha256_file(BLUEPRINT),
            str(CANDIDATES.relative_to(ROOT)): sha256_file(CANDIDATES),
        },
        "counts": {
            "me_rows_total": len(candidates),
            "me_effect_rows": effect_rows,
            "effect_rank1_compiler_valid": effect_compiler_valid,
            "effect_blueprint_action_ready": effect_action_ready,
            "effect_legacy_observer_action_invitation": effect_legacy_observer_ready,
            "effect_v53_observer_action_invitation": effect_v53_observer_ready,
        },
        "per_split": serial,
        "interpretation": {
            "legacy_468_card_zero_support_applies_to_current_v3_construction": False,
            "current_v3_effect_construction_has_me_action_result_support": (
                effect_compiler_valid == effect_rows
            ),
            "legacy_action_invitation_observer_failed_development_replay": (
                effect_legacy_observer_ready != effect_action_ready
            ),
            "v53_action_readiness_observer_passed_development_replay": (
                effect_v53_observer_ready == effect_action_ready
            ),
            "v53_action_readiness_observer_fresh_qualified": False,
            "construction_intent_is_worth_opening_gold": False,
            "external_prevalence_used_as_target": False,
            "quality_risk_or_function_outcomes_read": False,
        },
        "observer_mismatches": failures,
        "api_calls": 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "audit_report.json", result)
    return result


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
