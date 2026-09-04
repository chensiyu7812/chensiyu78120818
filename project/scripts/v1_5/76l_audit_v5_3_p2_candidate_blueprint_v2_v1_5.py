#!/usr/bin/env python3
"""Leader audit of the repaired W7R V5.3 P2 candidate blueprint.

Zero API, no generated responses, no quality/risk/outcome reads.  This audit
checks whether the W7R rows identify the intended Step1 learning problem,
not merely whether the previously named schema counters became zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.evoemo import load_evoemo  # noqa: E402
from metacom_pm.io import sha256_file, write_json  # noqa: E402
from metacom_pm.text import content_words  # noqa: E402
from metacom_pm.v1_5_v5_3_external_leakage_audit import (  # noqa: E402
    compare_text_surface_overlap,
    external_overlap_surfaces,
    extract_internal_text_surfaces,
)


BLUEPRINT = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2_leader_audit_v1"
EVOEMO = ROOT / "data/external/evo_emo.json"
ESCONV = ROOT / "data/external/ESConv.json"


def _load(name: str) -> list[dict]:
    return json.loads((BLUEPRINT / name).read_text(encoding="utf-8"))


def _condition(row: dict) -> str:
    return str((row.get("audit_only") or {}).get("construction_condition") or "")


def _iter_strings(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, (*path, str(index)))
    elif isinstance(value, str) and value.strip():
        yield path, value


def _esconv_surfaces() -> list[dict[str, str]]:
    value = json.loads(ESCONV.read_text(encoding="utf-8"))
    return [
        {
            "surface_id": "esconv:" + ":".join(path),
            "category": "external_esconv_text",
            "text": text,
        }
        for path, text in _iter_strings(value)
        if len(text.split()) >= 3
    ]


def _fragmented_users(rows: list[dict]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["user_id"])].add(str(row["counterfactual_group_id"]))
    return {
        user: sorted(values)
        for user, values in groups.items()
        if len(values) > 1
    }


def build_report() -> dict:
    mp = _load("mp_states.json")
    ms = _load("ms_states.json")
    me = _load("me_states.json")
    rs = _load("rs_states.json")
    interactions = _load("interaction_states.json")
    by_component = {"MP": mp, "MS": ms, "ME": me, "RS": rs}
    singles = mp + ms + me + rs
    all_rows = singles + interactions

    # W7R's original named schema fixes really did land.
    schema_repairs = {
        "mp_rows_missing_structured_fields": sum(
            any(key not in row for key in ("field_type", "field_value", "candidate_version"))
            for row in mp
        ),
        "ms_one_item_pools": sum(row["n_candidates_in_catalog"] == 1 for row in ms),
        "ms_missing_semantic_score": sum("score_top1_semantic_relevance" not in row for row in ms),
        "me_answer_feature_rows": sum(
            "rank1_matches_intended_target" in row.get("step1_features", {}) for row in me
        ),
        "rs_subtype_is_selection_mode": sum(
            row.get("exact_rank1_subtype") in {"transparent_priority", "lexical_fallback"}
            for row in rs
        ),
        "missing_surface_or_version_lineage_rows": sum(
            any(
                key not in row
                for key in (
                    "candidate_version",
                    "current_surface_sha256",
                    "candidate_surface_sha256",
                )
            )
            for row in singles
        ),
    }

    mp_preference_positive = [row for row in mp if _condition(row) == "preference_positive"]
    mp_preference_increment_already_visible = 0
    for row in mp_preference_positive:
        stored_increment = content_words(str(row["field_value"])) - {"prefers"}
        current = content_words(str(row["current_user_text"]))
        if stored_increment and stored_increment <= current:
            mp_preference_increment_already_visible += 1

    ms_continuity = [row for row in ms if _condition(row) == "continuity_positive"]
    ms_redundant_continuity = sum(
        row.get("step1_features", {}).get("current_redundant") is True
        for row in ms_continuity
    )
    ms_continuity_observer_hits = sum(
        row.get("step1_features", {}).get("continuity_request") is True
        for row in ms_continuity
    )
    # The intended target was constructed at created_session=3; the copied
    # same-family distractor is created_session=2.
    ms_continuity_rank1_is_target = sum(
        bool(row.get("topk_lineage"))
        and row["topk_lineage"][0].get("created_session") == 3
        for row in ms_continuity
    )

    me_empty_step_features = sum(not row.get("step1_features") for row in me)
    me_conditions_by_family: dict[str, set[str]] = defaultdict(set)
    for row in me:
        me_conditions_by_family[str(row["family"])].add(_condition(row))
    me_positive_only_families = sum(
        conditions == {"positive"} for conditions in me_conditions_by_family.values()
    )

    fragmented = {
        component: _fragmented_users(rows)
        for component, rows in by_component.items()
    }
    reported_groups = {
        component: len({row["counterfactual_group_id"] for row in rows})
        for component, rows in by_component.items()
    }
    user_clusters = {
        component: len({row["user_id"] for row in rows})
        for component, rows in by_component.items()
    }
    content_family_clusters = {
        component: len({row["family"] for row in rows})
        for component, rows in by_component.items()
    }

    rs_missing_required_slots = sum(
        not {
            "card_precondition_met",
            "card_already_executed_last_turn",
        }.issubset(row.get("step1_features", {}))
        for row in rs
    )
    interactions_without_mp = sum(
        not bool((row.get("components") or {}).get("MP", {}).get("available"))
        for row in interactions
    )

    internal_surfaces = extract_internal_text_surfaces(
        all_rows,
        text_fields=frozenset({"current_user_text", "model_visible_surface", "text"}),
    )
    overlap = compare_text_surface_overlap(
        internal_surfaces=internal_surfaces,
        external_surfaces=[
            *external_overlap_surfaces(load_evoemo(EVOEMO)),
            *_esconv_surfaces(),
        ],
        ngram_size=8,
    )

    blocking_findings = [
        {
            "id": "W7R-AUDIT-01",
            "severity": "critical",
            "finding": "MP preference positives still state the entire stored behavioral increment",
            "evidence": {
                "preference_positive_states": len(mp_preference_positive),
                "stored_increment_fully_visible_in_current_turn": mp_preference_increment_already_visible,
                "heuristic_current_redundant_true": sum(
                    row["step1_features"].get("current_redundant") is True
                    for row in mp_preference_positive
                ),
            },
            "cause": "Candidate text was shortened to two scope words so the three-word redundancy heuristic no longer fires.",
            "impact": "The counter reached zero without creating a hidden MP increment; ON/OFF cannot isolate stored-preference value.",
        },
        {
            "id": "W7R-AUDIT-02",
            "severity": "critical",
            "finding": "MS continuity positives reveal the requested memory and often retrieve a copied distractor",
            "evidence": {
                "continuity_positive_states": len(ms_continuity),
                "current_redundant_true": ms_redundant_continuity,
                "continuity_observer_true": ms_continuity_observer_hits,
                "rank1_is_constructed_target": ms_continuity_rank1_is_target,
                "candidate_pool_size": sorted({row["n_candidates_in_catalog"] for row in ms}),
                "observed_external_evoemo_pool_range": [13, 33],
                "external_pool_range_source": (
                    "docs/PM_V1_5_V5_3_MS_RETRIEVAL_QUALIFYING_TRIAL_FINDINGS_20260806_ZH.md"
                ),
            },
            "impact": "The state tests repetition of an answer already in the prompt, not retrieval-backed continuity under external-scale competition.",
        },
        {
            "id": "W7R-AUDIT-03",
            "severity": "critical",
            "finding": "ME construction gold was removed by deleting all component-specific runtime features",
            "evidence": {
                "me_states": len(me),
                "empty_step1_feature_rows": me_empty_step_features,
                "required_legitimate_slots": [
                    "past_action_result",
                    "current_action_readiness",
                    "current_redundancy",
                ],
            },
            "impact": "The ME value head cannot learn when an eligible past result is invited, declined, redundant, or goal-mismatched.",
        },
        {
            "id": "W7R-AUDIT-04",
            "severity": "high",
            "finding": "Counterfactual group IDs were split at state level",
            "evidence": {
                "reported_groups_by_component": reported_groups,
                "user_clusters_by_component": user_clusters,
                "content_family_clusters_by_component": content_family_clusters,
                "users_fragmented_across_multiple_groups": {
                    component: len(users) for component, users in fragmented.items()
                },
            },
            "impact": "Near-counterfactual variants could cross fit/confirmation and overstate independent sample size.",
        },
        {
            "id": "W7R-AUDIT-05",
            "severity": "high",
            "finding": "ME condition coverage is confounded with family",
            "evidence": {
                "me_families": len(me_conditions_by_family),
                "families_with_positive_only": me_positive_only_families,
            },
            "impact": "A family-held-out result could reflect condition imbalance rather than ME opportunity transfer.",
        },
        {
            "id": "W7R-AUDIT-06",
            "severity": "high",
            "finding": "RS and interaction learning surfaces remain incomplete",
            "evidence": {
                "rs_states_missing_card_precondition_and_execution_slots": rs_missing_required_slots,
                "interaction_states": len(interactions),
                "interaction_states_without_mp": interactions_without_mp,
            },
            "impact": "The full four-resource learned policy and real multi-component Step2 path are not represented by the blueprint.",
        },
    ]

    if overlap["exact_collision_count"]:
        blocking_findings.append(
            {
                "id": "W7R-AUDIT-07",
                "severity": "critical",
                "finding": "Exact external text overlap detected",
                "evidence": {"exact_collision_count": overlap["exact_collision_count"]},
                "impact": "Internal training could copy an external source surface.",
            }
        )

    return {
        "protocol": "pm-v1.5-v5.3-p2-candidate-blueprint-v2-leader-audit-v1",
        "status": "METHOD_AND_BLUEPRINT_REPAIR_REQUIRED_BEFORE_P2_READY",
        "blueprint": {
            "path": str(BLUEPRINT.relative_to(ROOT)),
            "summary_sha256": sha256_file(BLUEPRINT / "summary.json"),
            "release_identity": json.loads(
                (BLUEPRINT / "summary.json").read_text(encoding="utf-8")
            )["release_identity"],
        },
        "schema_repairs_confirmed": schema_repairs,
        "grain": {
            "single_component_states": len(singles),
            "interaction_states": len(interactions),
            "reported_groups_by_component": reported_groups,
            "user_clusters_by_component": user_clusters,
            "content_family_clusters_by_component": content_family_clusters,
        },
        "external_content_overlap": overlap,
        "blocking_findings": blocking_findings,
        "decision": {
            "w7r_schema_and_lineage_repairs_real": True,
            "w7r_learning_blueprint_accepted": False,
            "split_or_sample_size_freeze_allowed": False,
            "formal_runner_binding_allowed": False,
            "paired_generation_allowed": False,
            "worker_should_continue_iterative_surface_tuning": False,
            "next_owner": "leader",
            "next_action": "repair MP candidate semantics, MS hidden-query realistic pool, legitimate ME/RS features, and clustering in one method-level revision",
        },
        "api_calls": 0,
        "generated_response_or_quality_risk_outcome_read": False,
    }


def main() -> None:
    report = build_report()
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
