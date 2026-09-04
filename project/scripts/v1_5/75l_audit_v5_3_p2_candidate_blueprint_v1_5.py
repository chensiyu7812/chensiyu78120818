#!/usr/bin/env python3
"""Leader-side, zero-API integrity audit of the Worker W7 P2 blueprint.

This audit deliberately stops before paired generation, labels, head fitting,
or quality/risk outcomes.  It asks a narrower question: can the materialized
states support an honest Step1 learning experiment and the already frozen
external-transfer claims?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.evoemo import load_evoemo  # noqa: E402
from metacom_pm.io import sha256_file, write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_external_leakage_audit import (  # noqa: E402
    compare_text_surface_overlap,
    external_overlap_surfaces,
    extract_internal_text_surfaces,
)


BLUEPRINT = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_audit_v1"
EVOEMO = ROOT / "data/external/evo_emo.json"
ESCONV = ROOT / "data/external/ESConv.json"
COMPONENT_FILES = {
    "MP": "mp_states.json",
    "MS": "ms_states.json",
    "ME": "me_states.json",
    "RS": "rs_states.json",
}

REQUIRED_SINGLE_FIELDS = {
    "component",
    "state_id",
    "user_id",
    "family",
    "counterfactual_group_id",
    "current_user_text",
    "current_goal",
    "visible_dialogue",
    "session_index",
    "top_k_candidate_ids",
    "exact_rank1_id",
    "exact_rank1_subtype",
    "owner_id",
    "candidate_present",
    "n_candidates_in_catalog",
    "incremental_injected_tokens",
    "model_visible_surface",
    "hard_off_reason",
    "step1_features",
    "full_catalog",
}

FORMAL_LINEAGE_FIELDS = {
    "candidate_version",
    "current_surface_sha256",
    "candidate_surface_sha256",
}

# One outcome-blind, manually expanded collision.  The shared normalized
# fragment is the generic discourse surface "again and I don't want to
# feel"; it contains no external user, event, answer, evidence, or resource
# content.  Keeping the hash here makes the adjudication reproducible without
# copying external text into the report.  Any new hash remains unadjudicated.
ADJUDICATED_GENERIC_NGRAM_HASHES = {
    "0fb36b191832c6cdc6df67f3c49709e01741feacdc3867a4a1bb9089324b4d43"
}


def _load_rows() -> tuple[dict[str, list[dict]], list[dict]]:
    by_component = {
        component: json.loads((BLUEPRINT / filename).read_text(encoding="utf-8"))
        for component, filename in COMPONENT_FILES.items()
    }
    interactions = json.loads(
        (BLUEPRINT / "interaction_states.json").read_text(encoding="utf-8")
    )
    return by_component, interactions


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


def _similarity_summary(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["component"]), str(row.get("negative_kind")))].append(row)
    result = []
    for (component, kind), members in sorted(buckets.items()):
        scores = []
        high_pairs = 0
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                score = SequenceMatcher(
                    None,
                    " ".join(str(left["current_user_text"]).casefold().split()),
                    " ".join(str(right["current_user_text"]).casefold().split()),
                ).ratio()
                scores.append(score)
                high_pairs += int(score >= 0.72)
        result.append(
            {
                "component": component,
                "negative_kind": kind,
                "rows": len(members),
                "pair_count": len(scores),
                "mean_sequence_similarity": round(sum(scores) / len(scores), 4)
                if scores
                else None,
                "pairs_at_or_above_0_72": high_pairs,
            }
        )
    return result


def _is_machine_label_goal(row: dict) -> bool:
    goal = str(row.get("current_goal") or "")
    return bool(
        goal
        and (
            goal == str(row.get("negative_kind") or "")
            or goal == str(row.get("subdomain") or "")
            or goal == str(row.get("family") or "")
            or goal
            in {
                "continuity_request",
                "joint_advice_and_continuity",
                "positive",
                "current_redundant",
                "explicit_decline",
                "goal_mismatch",
            }
        )
    )


def build_report() -> dict:
    by_component, interactions = _load_rows()
    singles = [row for rows in by_component.values() for row in rows]
    all_rows = singles + interactions

    state_ids = [str(row["state_id"]) for row in all_rows]
    exact_text_counts = Counter(
        " ".join(str(row["current_user_text"]).casefold().split()) for row in all_rows
    )
    duplicate_text_rows = sum(count for count in exact_text_counts.values() if count > 1)

    missing_required = {
        component: sum(bool(REQUIRED_SINGLE_FIELDS - set(row)) for row in rows)
        for component, rows in by_component.items()
    }
    missing_lineage = {
        component: sum(bool(FORMAL_LINEAGE_FIELDS - set(row)) for row in rows)
        for component, rows in by_component.items()
    }

    groups_by_component = {
        component: len({str(row["counterfactual_group_id"]) for row in rows})
        for component, rows in by_component.items()
    }
    users_by_component = {
        component: len({str(row["user_id"]) for row in rows})
        for component, rows in by_component.items()
    }

    mp = by_component["MP"]
    mp_positive = [
        row
        for row in mp
        if row.get("negative_kind") in {"preference_positive", "profile_positive"}
    ]
    mp_profile = [row for row in mp if row.get("negative_kind") == "profile_positive"]
    mp_structured_missing = sum(
        any(field not in row for field in ("field_type", "field_value", "candidate_version"))
        for row in mp
    )
    mp_redundant_positive = sum(
        row.get("step1_features", {}).get("current_redundant") is True
        for row in mp_positive
    )
    mp_profile_need_false = sum(
        row.get("step1_features", {}).get("profile_goal_needs_advice_or_arrangement")
        is False
        for row in mp_profile
    )

    ms = by_component["MS"]
    ms_single_candidate_pool = sum(row.get("n_candidates_in_catalog") == 1 for row in ms)
    ms_semantic_score_missing = sum("score_top1_semantic_relevance" not in row for row in ms)

    me = by_component["ME"]
    me_answer_feature_rows = sum(
        "rank1_matches_intended_target" in row.get("step1_features", {}) for row in me
    )

    rs = by_component["RS"]
    rs_subtype_is_selection_mode = sum(
        row.get("exact_rank1_subtype")
        in {"transparent_priority", "lexical_fallback"}
        for row in rs
    )

    machine_label_goals = sum(_is_machine_label_goal(row) for row in all_rows)
    interactions_incomplete = sum(
        not all(
            key in component_row
            for component_row in row.get("components", {}).values()
            for key in ("candidate_present", "exact_rank1_id", "available")
        )
        or any(
            key not in row
            for key in (
                "model_visible_surfaces",
                "step1_features",
                "candidate_lineage",
            )
        )
        for row in interactions
    )

    internal_surfaces = extract_internal_text_surfaces(
        all_rows,
        text_fields=frozenset(
            {
                "current_user_text",
                "model_visible_surface",
                "text",
            }
        ),
    )
    evoemo_surfaces = external_overlap_surfaces(load_evoemo(EVOEMO))
    overlap = compare_text_surface_overlap(
        internal_surfaces=internal_surfaces,
        external_surfaces=[*evoemo_surfaces, *_esconv_surfaces()],
        ngram_size=8,
    )
    unadjudicated_ngram_collisions = [
        collision
        for collision in overlap["normalized_ngram_collisions"]
        if collision["ngram_sha256"] not in ADJUDICATED_GENERIC_NGRAM_HASHES
    ]
    overlap["adjudication"] = {
        "generic_discourse_collision_hashes": sorted(
            ADJUDICATED_GENERIC_NGRAM_HASHES
            & {
                collision["ngram_sha256"]
                for collision in overlap["normalized_ngram_collisions"]
            }
        ),
        "unadjudicated_normalized_ngram_collision_count": len(
            unadjudicated_ngram_collisions
        ),
        "rule": (
            "exact overlap blocks; normalized n-gram overlap blocks only when "
            "it remains content-specific after outcome-blind manual expansion"
        ),
        "raw_external_text_copied_into_blueprint": False,
    }

    blocking_findings = [
        {
            "id": "W7-AUDIT-01",
            "severity": "critical",
            "finding": "MP positive states restate the stored preference/profile in the current turn",
            "evidence": {
                "positive_states": len(mp_positive),
                "positive_states_marked_current_redundant": mp_redundant_positive,
                "profile_positive_states": len(mp_profile),
                "profile_need_feature_false": mp_profile_need_false,
            },
            "impact": "MP ON cannot have a clean incremental effect and the head cannot learn when hidden profile/preference changes the reply.",
        },
        {
            "id": "W7-AUDIT-02",
            "severity": "critical",
            "finding": "MS states use a one-item catalog and omit the actual BGE score",
            "evidence": {
                "ms_states": len(ms),
                "one_item_catalog_states": ms_single_candidate_pool,
                "missing_top1_semantic_score": ms_semantic_score_missing,
            },
            "impact": "The blueprint does not exercise the frozen full-causal-pool BGE selection problem that appears externally.",
        },
        {
            "id": "W7-AUDIT-03",
            "severity": "critical",
            "finding": "ME intended-target truth is stored inside Step1 features",
            "evidence": {
                "me_states": len(me),
                "rows_with_rank1_matches_intended_target_feature": me_answer_feature_rows,
            },
            "impact": "Construction gold would be available to the learned head unless moved to audit-only metadata.",
        },
        {
            "id": "W7-AUDIT-04",
            "severity": "high",
            "finding": "Current goals are machine labels rather than natural, runtime-observable goal surfaces",
            "evidence": {"rows": len(all_rows), "machine_label_goal_rows": machine_label_goals},
            "impact": "A runner that serializes current_goal could learn construction labels instead of user need.",
        },
        {
            "id": "W7-AUDIT-05",
            "severity": "high",
            "finding": "Per-head independent-group support is only 7-8 groups",
            "evidence": {
                "groups_by_component": groups_by_component,
                "users_by_component": users_by_component,
            },
            "impact": "Fit, content-disjoint confirmation, and whole-family transfer cannot all estimate a stable four-head learning claim from this blueprint.",
        },
        {
            "id": "W7-AUDIT-06",
            "severity": "high",
            "finding": "Formal lineage and interaction execution surfaces are incomplete",
            "evidence": {
                "missing_lineage_rows": missing_lineage,
                "mp_rows_missing_structured_field_type_value_version": mp_structured_missing,
                "interaction_states": len(interactions),
                "interaction_states_incomplete_for_runner": interactions_incomplete,
                "rs_rows_using_selection_mode_as_subtype": rs_subtype_is_selection_mode,
            },
            "impact": "The formal runner cannot prove same-candidate fairness, exact lineage, or faithful multi-component execution.",
        },
    ]

    if overlap["exact_collision_count"] or unadjudicated_ngram_collisions:
        blocking_findings.append(
            {
                "id": "W7-AUDIT-07",
                "severity": "critical",
                "finding": "Internal blueprint text overlaps external source surfaces",
                "evidence": {
                    "exact_collision_count": overlap["exact_collision_count"],
                    "unadjudicated_normalized_ngram_collision_count": len(
                        unadjudicated_ngram_collisions
                    ),
                },
                "impact": "Internal training text could leak EvoEmo/ES-MemEval/ESConv content.",
            }
        )

    return {
        "protocol": "pm-v1.5-v5.3-p2-candidate-blueprint-leader-audit-v1",
        "status": "REPAIR_REQUIRED_BEFORE_P2_READY",
        "blueprint": {
            "path": str(BLUEPRINT.relative_to(ROOT)),
            "summary_sha256": sha256_file(BLUEPRINT / "summary.json"),
            "release_identity": json.loads(
                (BLUEPRINT / "summary.json").read_text(encoding="utf-8")
            )["release_identity"],
        },
        "grain": {
            "single_component_states": len(singles),
            "interaction_states": len(interactions),
            "unique_state_ids": len(set(state_ids)),
            "duplicate_state_ids": len(state_ids) - len(set(state_ids)),
            "exact_duplicate_current_text_rows": duplicate_text_rows,
            "groups_by_component": groups_by_component,
            "users_by_component": users_by_component,
        },
        "schema": {
            "rows_missing_required_single_fields": missing_required,
            "rows_missing_formal_lineage_fields": missing_lineage,
        },
        "template_shape": _similarity_summary(singles),
        "external_content_overlap": overlap,
        "blocking_findings": blocking_findings,
        "minimum_remediation": [
            "MP: build the same-user structured candidate pool without copying field values/preferences into positive current turns; keep preference/profile separately and emit field_type, field_value, owner, subtype, version.",
            "MS: use realistic multi-item strictly-past same-user pools and persist actual BGE score/margin for every ranked candidate.",
            "ME: keep the frozen production ranker but move intended-target match to audit-only metadata; never expose it to Step1.",
            "RS: store atomic card/move subtype separately from selection_mode.",
            "All components: replace label-like current_goal values with natural outcome-blind goal text and add surface/lineage hashes.",
            "Expand genuinely distinct user/content groups per head before freezing fit/confirmation/whole-family-holdout splits; do not replicate the existing templates.",
            "Materialize interaction rows with full per-component candidate lineage, surfaces, features, and at least pairwise plus multi-component coverage; equal frequency over 16 actions is not required.",
        ],
        "split_freeze_allowed": False,
        "formal_runner_implementation_allowed": False,
        "paired_generation_allowed": False,
        "api_calls": 0,
        "generated_response_or_quality_risk_outcome_read": False,
    }


def main() -> None:
    report = build_report()
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
