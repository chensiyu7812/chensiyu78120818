#!/usr/bin/env python3
"""Final zero-API Leader audit before independent Worker review and split freeze."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.evoemo import load_evoemo
from metacom_pm.io import read_json, write_json
from metacom_pm.v1_5_v5_3_external_leakage_audit import (
    external_overlap_surfaces,
    normalized_ngrams,
    normalized_tokens,
)
from metacom_pm.v1_5_v5_3_p2r_learning_blueprint import (
    P2RBlueprintRow,
    P2RInteractionBlueprintRow,
    audit_blueprint_rows,
)


ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_v1"
CATALOG_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_audit_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _esconv_surfaces() -> list[dict[str, str]]:
    value = read_json(ROOT / "data/external/ESConv.json")
    rows: list[dict[str, str]] = []

    def walk(item: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(item, dict):
            for key, value in item.items():
                walk(value, (*path, str(key)))
        elif isinstance(item, list):
            for index, value in enumerate(item):
                walk(value, (*path, str(index)))
        elif isinstance(item, str) and len(item.split()) >= 3:
            rows.append(
                {
                    "surface_id": "esconv:" + ":".join(path),
                    "category": "external_esconv_text",
                    "text": item,
                }
            )

    walk(value)
    return rows


def _compact_surface_overlap(
    *,
    internal_surfaces: list[dict[str, str]],
    external_surfaces: list[dict[str, str]],
    ngram_size: int,
) -> dict[str, int]:
    """Count source overlap without materializing a collision Cartesian product.

    The shared library routine intentionally preserves every collision pair for
    small diagnostic packets.  Here the ESConv/EvoEmo surface is large and many
    ordinary n-grams repeat, so retaining every pair can exhaust memory while
    adding no value to this release-blocking audit.
    """

    external_exact = {" ".join(str(row["text"]).split()) for row in external_surfaces}
    external_ngrams: set[tuple[str, ...]] = set()
    for row in external_surfaces:
        external_ngrams.update(normalized_ngrams(str(row["text"]), n=ngram_size))

    exact_internal = 0
    ngram_internal = 0
    shared_ngram_count = 0
    for row in internal_surfaces:
        text = str(row["text"])
        if " ".join(text.split()) in external_exact:
            exact_internal += 1
        shared = normalized_ngrams(text, n=ngram_size) & external_ngrams
        if shared:
            ngram_internal += 1
            shared_ngram_count += len(shared)
    return {
        "ngram_size": ngram_size,
        "internal_surface_count": len(internal_surfaces),
        "external_surface_count": len(external_surfaces),
        "internal_surfaces_with_exact_overlap": exact_internal,
        "internal_surfaces_with_normalized_ngram_overlap": ngram_internal,
        "exact_collision_count": exact_internal,
        "normalized_ngram_collision_count": shared_ngram_count,
    }


def _condition_observability(rows: list[P2RBlueprintRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        component_rows = [row for row in rows if row.state.component == component]
        by_condition: dict[str, Any] = {}
        for condition in sorted({row.state.state_condition for row in component_rows}):
            subset = [row for row in component_rows if row.state.state_condition == condition]
            if component == "MP":
                observed = sum(
                    bool(row.model_input.contribution_slots.get("current_redundant"))
                    or bool(row.model_input.contribution_slots.get("preference_applies_to_response_act"))
                    or bool(row.model_input.contribution_slots.get("profile_goal_needs_advice_or_arrangement"))
                    for row in subset
                )
            elif component == "MS":
                observed = sum(
                    bool(row.model_input.contribution_slots.get("continuity_request"))
                    or bool(row.model_input.contribution_slots.get("current_redundant"))
                    for row in subset
                )
            elif component == "ME":
                observed = sum(
                    row.model_input.contribution_slots.get("current_action_readiness")
                    != "UNKNOWN"
                    for row in subset
                )
            else:
                observed = sum(
                    bool(row.model_input.contribution_slots.get("card_precondition_met"))
                    for row in subset
                )
            by_condition[condition] = {
                "rows": len(subset),
                "candidate_present": sum(row.model_input.candidate_present for row in subset),
                "transparent_primary_signal_observed": observed,
                "semantic_extension_or_effect_only_rows": len(subset) - observed,
            }
        result[component] = by_condition
    return result


def main() -> None:
    rows = [
        P2RBlueprintRow.model_validate(row)
        for row in _read_jsonl(BLUEPRINT_DIR / "component_rows.jsonl")
    ]
    interactions = [
        P2RInteractionBlueprintRow.model_validate(row)
        for row in _read_jsonl(BLUEPRINT_DIR / "interaction_rows.jsonl")
    ]
    structural = audit_blueprint_rows(
        rows,
        require_final_condition_crossing=True,
        require_assigned_splits=False,
    )
    catalog = {
        row["item_id"]: row
        for row in _read_jsonl(CATALOG_DIR / "catalog.jsonl")
    }

    exact_surface_groups: dict[tuple[str, ...], list[P2RBlueprintRow]] = defaultdict(list)
    internal_surfaces: list[dict[str, str]] = []
    for row in rows:
        text = row.state.current_user_text
        exact_surface_groups[normalized_tokens(text)].append(row)
        internal_surfaces.append(
            {
                "surface_id": row.state.state_id,
                "category": "p2r_current_state",
                "text": text,
            }
        )
    duplicate_groups = [group for group in exact_surface_groups.values() if len(group) > 1]
    intentional_same_state_groups = 0
    accidental_cross_user_duplicate_groups = 0
    for group in duplicate_groups:
        interaction_keys = {row.state.interaction_key for row in group}
        users = {row.state.catalog_user_id for row in group}
        if len(interaction_keys) == 1 and None not in interaction_keys and len(users) == 1:
            intentional_same_state_groups += 1
        else:
            accidental_cross_user_duplicate_groups += 1

    positive_answer_copy = 0
    family_alignment = Counter()
    for row in rows:
        candidate_id = row.candidate_lineage.candidate_id
        if candidate_id is None or candidate_id not in catalog:
            continue
        item = catalog[candidate_id]
        candidate_text = " ".join(str(item["text"]).split()).casefold()
        current_text = " ".join(row.state.current_user_text.split()).casefold()
        if row.state.state_condition == "positive_opportunity" and candidate_text in current_text:
            positive_answer_copy += 1
        state_topic = row.state.semantic_family.split("::", 1)[-1]
        if row.state.interaction_key is not None and row.state.component == "MP":
            # Interaction topics and MP field scopes use different taxonomies.
            # Treating their labels as directly comparable created 12 false
            # mismatches in the original audit.
            family_alignment[(row.state.component, "not_applicable_interaction_taxonomy")] += 1
        elif row.state.component == "MP":
            candidate_topic = (
                item.get("field_value")
                if item.get("subtype") == "MP_PREFERENCE"
                else item.get("field_type")
            )
            family_alignment[
                (row.state.component, "match" if state_topic == candidate_topic else "mismatch")
            ] += 1
        elif row.state.component in {"MS", "ME"}:
            candidate_topic = item.get("topic_thread")
            family_alignment[
                (row.state.component, "match" if state_topic == candidate_topic else "mismatch")
            ] += 1
        else:
            family_alignment[(row.state.component, "not_applicable")] += 1

    evoemo = load_evoemo(ROOT / "data/external/evo_emo.json")
    overlap = _compact_surface_overlap(
        internal_surfaces=internal_surfaces,
        external_surfaces=[*external_overlap_surfaces(evoemo), *_esconv_surfaces()],
        ngram_size=8,
    )
    full_four = sum(
        set(row.components) == {"MP", "MS", "ME", "RS"}
        and all(model_input.candidate_present for model_input in row.components.values())
        for row in interactions
    )
    interaction_candidate_alignment: list[dict[str, Any]] = []
    for row in interactions:
        topic = row.semantic_family.split("::", 1)[-1]
        mismatched_components: list[str] = []
        for component in ("MS", "ME"):
            candidate_id = row.candidate_lineage[component].candidate_id
            item = catalog.get(candidate_id or "", {})
            if item.get("topic_thread") != topic:
                mismatched_components.append(component)
        interaction_candidate_alignment.append(
            {
                "interaction_id": row.interaction_id,
                "mismatched_memory_components": mismatched_components,
            }
        )
    interactions_with_memory_topic_mismatch = sum(
        bool(row["mismatched_memory_components"])
        for row in interaction_candidate_alignment
    )
    observability = _condition_observability(rows)
    blockers = {
        **structural["critical_failures"],
        "positive_candidate_literal_copied_into_current_state": positive_answer_copy,
        "accidental_cross_user_exact_surface_duplicate_groups": accidental_cross_user_duplicate_groups,
        "external_exact_overlap": overlap["exact_collision_count"],
        "external_normalized_8gram_overlap": overlap["normalized_ngram_collision_count"],
        "interactions_missing_full_four_candidate_coverage": len(interactions) - full_four,
        "interactions_with_memory_candidate_topic_mismatch": (
            interactions_with_memory_topic_mismatch
        ),
    }
    report = {
        "protocol": "pm-v1.5-v5.3-p2r-learning-blueprint-audit-v1",
        "status": (
            "READY_FOR_WORKER_INDEPENDENT_REVIEW_NOT_SPLIT_FROZEN"
            if not any(blockers.values())
            else "STATIC_BLOCKED"
        ),
        "blockers": blockers,
        "structural": structural,
        "current_surface_duplicates": {
            "duplicate_groups": len(duplicate_groups),
            "duplicate_rows": sum(len(group) for group in duplicate_groups),
            "intentional_same_user_interaction_groups": intentional_same_state_groups,
            "accidental_cross_user_groups": accidental_cross_user_duplicate_groups,
        },
        "candidate_family_alignment": {
            f"{component}:{status}": count
            for (component, status), count in sorted(family_alignment.items())
        },
        "interaction_memory_candidate_alignment": {
            "checked_interactions": len(interactions),
            "interactions_with_mismatch": interactions_with_memory_topic_mismatch,
            "affected": [
                row
                for row in interaction_candidate_alignment
                if row["mismatched_memory_components"]
            ],
            "scope_note": (
                "MS/ME topic_thread must match the interaction topic before a row may "
                "support a joint-effect claim. MP uses a different field-scope taxonomy "
                "and is not judged by literal topic-label equality here."
            ),
        },
        "transparent_observability": observability,
        "semantic_extension_policy": (
            "Rows whose intended construct is not observed by the transparent "
            "factor remain explicit semantic-extension probes; they are not "
            "relabeled as negatives and cannot be used to claim base-head coverage."
        ),
        "external_overlap": {
            key: overlap[key]
            for key in (
                "ngram_size",
                "internal_surface_count",
                "external_surface_count",
                "internal_surfaces_with_exact_overlap",
                "internal_surfaces_with_normalized_ngram_overlap",
                "exact_collision_count",
                "normalized_ngram_collision_count",
            )
        },
        "interaction_rows": len(interactions),
        "full_four_candidate_interactions": full_four,
        "split_status": "PENDING_INDEPENDENT_WORKER_REVIEW",
        "generated_response_or_quality_risk_outcome_read": False,
        "api_calls": 0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
