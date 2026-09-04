#!/usr/bin/env python3
"""Freeze the zero-API V1.5b MP/MS anti-shortcut fit and confirmation data.

This is controlled construct data for the post-candidate opportunity decision,
not response-quality gold.  Every row contains a realized candidate surface and
a current visible state.  Labels follow the frozen MP/MS source-specific
codebook, while background action bits are crossed independently of labels.
Raw surfaces and condition names remain private audit fields and are never model
features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from metacom_pm.contracts import (
    MemoryItem,
    MemorySource,
    StrategyMode,
    canonical_action_id,
)
from metacom_pm.io import canonical_json, sha256_file, sha256_text, stable_hex, write_json, write_jsonl
from metacom_pm.text import normalize_for_hash
from metacom_pm.v1_5_candidate_discovery import MemoryCandidate, describe_memory_candidate
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-mp-ms-antishortcut-data-v1"
CONTRACT = "data/pm_v1_5_contracts/v1_5b_mp_ms_generator_root_repair_v1.json"
COMPONENTS = ("MP", "MS")

FIT_THEMES = (
    ("community choir", "missed entrances", "the rehearsal room"),
    ("garden project", "unclear task ownership", "the tool shed"),
    ("language class", "speaking practice", "the evening lesson"),
    ("volunteer shift", "several requests arriving together", "the front desk"),
    ("creative workshop", "too many revisions", "the draft table"),
    ("commute change", "an unfamiliar transfer", "the morning route"),
    ("neighborhood group", "not knowing anyone", "the monthly meeting"),
    ("music practice", "one difficult passage", "the practice room"),
)

CONFIRM_THEMES = (
    ("book club", "an idea being dismissed", "the discussion circle"),
    ("cycling route", "two confusing turns", "the riverside path"),
    ("cooking course", "several techniques at once", "the training kitchen"),
    ("library project", "vague responsibilities", "the archive desk"),
)


def _background(component: str, condition_index: int, topic_index: int, role: str) -> str:
    # Fit: every condition observes all 8 nuisance combinations exactly once.
    # Confirmation: every label observes every combination equally across its
    # four condition families.  Neither row order nor target determines a bit.
    code = topic_index if role == "fit" else (condition_index * 4 + topic_index) % 8
    target = MemorySource(component)
    others = tuple(source for source in MemorySource if source is not target)
    sources = {
        source for position, source in enumerate(others) if code & (1 << position)
    }
    strategy = StrategyMode.RS if code & 4 else StrategyMode.R0
    return canonical_action_id(sources, strategy)


def _mp_pair(
    condition: int, theme: tuple[str, str, str], role: str
) -> tuple[str, str, list[dict[str, str]], str, int, str]:
    topic, detail, place = theme
    prefix = "In my support preferences" if role == "fit" else "For supportive replies"
    if condition == 0:
        return (
            f"{prefix}, when I ask for advice about {topic}, offer one small option rather than a plan.",
            f"Could you give me one small idea about {topic}? Please keep it brief.",
            [], "MP_PREFERENCE", 1, "preference_direct_current_scope",
        )
    if condition == 1:
        return (
            "I prefer enough room to finish my thoughts before any focused question.",
            f"I need to talk through {topic} without being interrupted while I explain it.",
            [], "MP_PREFERENCE", 1, "preference_low_lexical_paraphrase",
        )
    if condition == 2:
        candidate = f"I regularly coordinate the {topic} group at {place}."
        return (
            candidate,
            f"Coordinating the {topic} group at {place} is exhausting me again.",
            [], "MP_PROFILE", 1, "profile_direct_incremental",
        )
    if condition == 3:
        return (
            f"I am the primary caregiver for my father while managing {topic}.",
            f"Dad's appointments are colliding with {topic}; I need help naming the pressure.",
            [], "MP_PROFILE", 1, "profile_low_lexical_incremental",
        )
    if condition == 4:
        return (
            f"{prefix}, when I ask for advice about {topic}, offer one small option.",
            f"I don't want advice about {topic}; just listen while I vent.",
            [], "MP_PREFERENCE", 0, "preference_high_lexical_scope_mismatch",
        )
    if condition == 5:
        candidate = f"I regularly coordinate the {topic} group at {place}."
        return (
            candidate,
            f"The {topic} is difficult today, but I already explained my role.",
            [{"role": "user", "content": candidate}],
            "MP_PROFILE", 0, "profile_already_visible",
        )
    if condition == 6:
        return (
            f"I regularly coordinate the {topic} group around {detail}.",
            f"My sibling joined a {topic} group and I only want to understand their experience with {detail}.",
            [], "MP_PROFILE", 0, "profile_high_lexical_wrong_entity",
        )
    return (
        "For household chores, I prefer a concise reply with one option.",
        f"The {topic} situation around {detail} is weighing on me today.",
        [], "MP_PREFERENCE", 0, "preference_realistic_irrelevant",
    )


def _ms_pair(
    condition: int, theme: tuple[str, str, str], role: str
) -> tuple[str, str, list[dict[str, str]], str, int, str]:
    topic, detail, place = theme
    earlier = "In the earlier session" if role == "fit" else "Previously"
    if condition == 0:
        return (
            f"{earlier}, {topic} became difficult because of {detail}; the issue remained open.",
            f"The {topic} difficulty is back, and {detail} is still getting in the way.",
            [], "MS_SESSION", 1, "same_unresolved_issue_direct",
        )
    if condition == 1:
        return (
            f"{earlier} about {topic}, {detail} was the main pressure, and naming that distinction clarified the situation.",
            f"This is back around {topic}. Which part matters most now?",
            [], "MS_SESSION", 1, "same_issue_low_lexical_paraphrase",
        )
    if condition == 2:
        return (
            f"During the prior {topic} difficulty, choosing one priority helped when {detail} became intense.",
            f"The {topic} situation feels tangled again. What helped last time?",
            [], "MS_SESSION", 1, "prior_outcome_requested",
        )
    if condition == 3:
        return (
            f"The earlier {topic} session distinguished {detail} rather than lack of effort; putting that into words helped.",
            f"I need help putting {topic} into words and understanding which part is actually hard.",
            [], "MS_SESSION", 1, "prior_distinction_matches_current_goal",
        )
    if condition == 4:
        return (
            f"Earlier {topic} planning focused on a three-step schedule for {detail}.",
            f"I do not want a plan for {topic}; I only want to talk through how {detail} feels.",
            [], "MS_SESSION", 0, "same_topic_high_lexical_wrong_goal",
        )
    if condition == 5:
        return (
            f"The {topic} issue with {detail} was fully settled and resolved at {place}.",
            f"A new part of {topic} involving {place} has started; this is not the old issue.",
            [], "MS_SESSION", 0, "old_issue_resolved_new_issue_current",
        )
    if condition == 6:
        candidate = f"{earlier}, {topic} became difficult because of {detail}."
        return (
            candidate,
            f"The {topic} issue is difficult again.",
            [{"role": "user", "content": candidate}],
            "MS_SESSION", 0, "summary_already_visible",
        )
    return (
        f"A prior {topic} check-in mentioned {detail} but contained no useful distinction.",
        f"The {topic} issue around {detail} is back; what helped last time?",
        [], "MS_SESSION", 0, "high_lexical_no_incremental_value",
    )


def _make_row(
    *, component: str, condition: int, topic_index: int, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    themes = FIT_THEMES if role == "fit" else CONFIRM_THEMES
    theme = themes[topic_index]
    pair = _mp_pair(condition, theme, role) if component == "MP" else _ms_pair(condition, theme, role)
    candidate_text, current, visible, subtype, target, family = pair
    source = MemorySource(component)
    user_id = f"pmv15b_{role}_{component.lower()}_{condition:02d}_{topic_index:02d}"
    item = MemoryItem(
        memory_id="mem_" + stable_hex(PROTOCOL, user_id, candidate_text, n=24),
        source=source,
        created_session=1,
        text=candidate_text,
    )
    descriptor = describe_memory_candidate(
        source=source,
        query=current,
        source_items=[item],
        selected_items=[item],
        session_index=3,
    )
    candidate = MemoryCandidate(source=source, selected_items=(item,), descriptor=descriptor)
    metadata = {
        item.memory_id: (
            {"mp_subtype": subtype, "outcome_read": False}
            if component == "MP"
            else {
                "summary_origin": "supplied_strictly_prior_summary",
                "outcome_read": False,
            }
        )
    }
    observation = build_source_specific_memory_opportunity_observation(
        candidate=candidate,
        catalog_items=[item],
        catalog_user_id=user_id,
        current_user_id=user_id,
        current_session_index=3,
        current_user_text=current,
        visible_dialogue=visible,
        source_metadata=metadata,
        background_action=_background(component, condition, topic_index, role),
    )
    if observation["deterministic_hard_off"]:
        raise RuntimeError(f"controlled learned row unexpectedly hard-off: {user_id}")
    semantic_family = theme[0].replace(" ", "_")
    public = {
        "protocol": PROTOCOL,
        "evidence_role": "development_fit" if role == "fit" else "sealed_confirmation",
        "component": component,
        "user_id_private_not_model_input": user_id,
        "semantic_family_private_not_model_input": semantic_family,
        "condition_family_private_not_model_input": family,
        "opportunity_target": target,
        "model_features": observation["model_features"],
        "feature_protocol": observation["protocol"],
        "deterministic_hard_off": False,
        "response_quality_risk_or_external_outcome_read": False,
    }
    private = {
        "protocol": PROTOCOL,
        "evidence_role": public["evidence_role"],
        "component": component,
        "user_id": user_id,
        "semantic_family": semantic_family,
        "condition_family": family,
        "candidate_text": candidate_text,
        "current_user_text": current,
        "visible_dialogue": visible,
        "candidate_subtype": subtype,
        "background_action": _background(component, condition, topic_index, role),
        "target": target,
    }
    return public, private


def _background_bits(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(float(value))
        for name, value in sorted(row["model_features"].items())
        if name.startswith("background_")
    )


def _nuisance_probe(rows: list[dict[str, Any]]) -> float:
    x = np.asarray([_background_bits(row) for row in rows], dtype=float)
    y = np.asarray([int(row["opportunity_target"]) for row in rows])
    model = LogisticRegression(C=0.3, solver="liblinear", random_state=20260802)
    model.fit(x, y)
    return float(balanced_accuracy_score(y, model.predict(x)))


def _feature_collision_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        key = canonical_json(dict(sorted(row["model_features"].items())))
        targets[key].add(int(row["opportunity_target"]))
    conflicts = sum(len(values) > 1 for values in targets.values())
    affected = sum(
        1
        for row in rows
        if len(targets[canonical_json(dict(sorted(row["model_features"].items())))]) > 1
    )
    return {
        "unique_feature_patterns": len(targets),
        "conflicting_feature_patterns": conflicts,
        "rows_in_conflicting_patterns": affected,
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    out_dir = root / "outputs/pm_v1_5b_mp_ms_antishortcut_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for role, themes in (("fit", FIT_THEMES), ("confirmation", CONFIRM_THEMES)):
        for component in COMPONENTS:
            for condition in range(8):
                for topic_index in range(len(themes)):
                    public_row, private_row = _make_row(
                        component=component,
                        condition=condition,
                        topic_index=topic_index,
                        role=role,
                    )
                    rows.append(public_row)
                    private.append(private_row)

    counts: dict[str, Any] = {}
    collisions: dict[str, Any] = {}
    nuisance: dict[str, float] = {}
    for role in ("development_fit", "sealed_confirmation"):
        for component in COMPONENTS:
            subset = [
                row for row in rows
                if row["evidence_role"] == role and row["component"] == component
            ]
            key = f"{role}:{component}"
            counts[key] = {
                "rows": len(subset),
                "users": len({row["user_id_private_not_model_input"] for row in subset}),
                "class_counts": dict(sorted(Counter(row["opportunity_target"] for row in subset).items())),
                "condition_counts": dict(sorted(Counter(row["condition_family_private_not_model_input"] for row in subset).items())),
                "background_by_target": {
                    str(target): {
                        "".join(str(bit) for bit in bits): count
                        for bits, count in sorted(
                            Counter(
                                _background_bits(row)
                                for row in subset
                                if row["opportunity_target"] == target
                            ).items()
                        )
                    }
                    for target in (0, 1)
                },
            }
            collisions[key] = _feature_collision_audit(subset)
            nuisance[key] = _nuisance_probe(subset)

    exact_pairs = [
        (normalize_for_hash(row["candidate_text"]), normalize_for_hash(row["current_user_text"]))
        for row in private
    ]
    fit_families = {row["semantic_family"] for row in private if row["evidence_role"] == "development_fit"}
    confirm_families = {row["semantic_family"] for row in private if row["evidence_role"] == "sealed_confirmation"}
    checks = {
        "64_fit_rows_32_on_32_off_per_head": all(
            value["rows"] == 64 and value["users"] == 64 and value["class_counts"] == {0: 32, 1: 32}
            for key, value in counts.items() if key.startswith("development_fit")
        ),
        "32_confirmation_rows_16_on_16_off_per_head": all(
            value["rows"] == 32 and value["users"] == 32 and value["class_counts"] == {0: 16, 1: 16}
            for key, value in counts.items() if key.startswith("sealed_confirmation")
        ),
        "eight_balanced_condition_families_per_split_head": all(
            len(value["condition_counts"]) == 8 and len(set(value["condition_counts"].values())) == 1
            for value in counts.values()
        ),
        "background_distribution_identical_by_target": all(
            value["background_by_target"]["0"] == value["background_by_target"]["1"]
            for value in counts.values()
        ),
        "nuisance_only_fit_ba_max_0_55": max(nuisance.values()) <= 0.55,
        "no_conflicting_exact_feature_patterns": all(
            value["conflicting_feature_patterns"] == 0 for value in collisions.values()
        ),
        "fit_confirmation_semantic_families_disjoint": not (fit_families & confirm_families),
        "all_candidate_current_pairs_unique": len(exact_pairs) == len(set(exact_pairs)),
        "same_source_specific_feature_protocol": all(
            row["feature_protocol"] == SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
            for row in rows
        ),
        "all_features_numeric": all(
            all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row["model_features"].values())
            for row in rows
        ),
        "no_raw_surface_in_model_feature_names": all(
            not any(token in name for token in ("text", "user_id", "condition", "target", "label", "dataset", "response"))
            for row in rows for name in row["model_features"]
        ),
        "zero_api_zero_response_outcome": all(
            not row["response_quality_risk_or_external_outcome_read"] for row in rows
        ),
    }
    status = "PASS_FROZEN_V1_5B_ANTISHORTCUT_DATA" if all(checks.values()) else "FAIL_V1_5B_ANTISHORTCUT_PREFLIGHT"
    rows_path = out_dir / "opportunity_rows.jsonl"
    private_path = out_dir / "private_candidate_state_surfaces.jsonl"
    write_jsonl(rows_path, rows)
    write_jsonl(private_path, private)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_role": "Controlled source-specific opportunity construct data; not response-quality or external utility gold.",
        "contract": CONTRACT,
        "api_calls_made": 0,
        "human_labels_read": False,
        "external_outcomes_read": False,
        "counts": counts,
        "feature_collision_audit": collisions,
        "nuisance_only_logistic_same_rows_ba": nuisance,
        "checks": checks,
    }
    write_json(out_dir / "preflight_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "contract_sha256": sha256_file(root / CONTRACT),
            "confirmation_frozen_before_fit": True,
            "files": {
                "opportunity_rows.jsonl": sha256_file(rows_path),
                "private_candidate_state_surfaces.jsonl": sha256_file(private_path),
                "preflight_report.json": sha256_file(out_dir / "preflight_report.json"),
            },
            "semantic_rows_sha256": sha256_text(canonical_json(rows)),
        },
    )
    if status.startswith("FAIL"):
        raise RuntimeError([key for key, value in checks.items() if not value])
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "checks": report["checks"]})


if __name__ == "__main__":
    main()
