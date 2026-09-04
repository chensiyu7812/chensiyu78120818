#!/usr/bin/env python3
"""Profile the three external exams and audit V5.3 training support.

This is a zero-API, outcome-free reverse specification.  It reads only raw
dataset structure, frozen split/surface metadata, and the W7R candidate
blueprint/audit.  It does not read generated replies, quality, risk, judge, or
human outcomes and does not create Step1 labels.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
ESCONV = ROOT / "data/external/ESConv.json"
ESCONV_SPLIT = ROOT / "data/strategy/esconv_split_manifest_v1_5.jsonl"
ESCONV_BUILD = ROOT / "data/esconv_test_v1_5/build_report.json"
EVOEMO = ROOT / "data/external/evo_emo.json"
EXTERNAL_SURFACES = (
    ROOT
    / "outputs/pm_v1_5_v5_2_external_e1_static_audit_v1"
    / "external_v5_2_response_free_surfaces_private.jsonl"
)
W7R = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2"
W7R_AUDIT = (
    ROOT
    / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2_leader_audit_v1"
    / "report.json"
)
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_external_exam_backward_training_spec_v1"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _describe(values: list[int | float]) -> dict:
    return {
        "n": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def _esconv_profile() -> dict:
    rows = _load_json(ESCONV)
    split = _load_jsonl(ESCONV_SPLIT)
    build = _load_json(ESCONV_BUILD)
    dialogue_turns = [len(row["dialog"]) for row in rows]
    supporter_turns = [
        sum(turn["speaker"] == "supporter" for turn in row["dialog"])
        for row in rows
    ]
    strategy_counts: Counter[str] = Counter()
    for row in rows:
        for turn in row["dialog"]:
            if turn["speaker"] == "supporter":
                strategy_counts[turn.get("annotation", {}).get("strategy") or "NONE"] += 1
    return {
        "raw_dialogues": len(rows),
        "dialogue_turns": _describe(dialogue_turns),
        "supporter_turns": _describe(supporter_turns),
        "supporter_strategy_counts": dict(sorted(strategy_counts.items())),
        "custom_split_counts": dict(sorted(Counter(row["split"] for row in split).items())),
        "evoemo_overlap_excluded_dialogues": sum(
            bool(row["excluded_for_evoemo_overlap"]) for row in split
        ),
        "nonoverlap_eligible_test_dialogues": build["split_audit"]["eligible_test_dialogues"],
        "all_support_eligible_test_turns": build["test_turns"],
        "training_implication": (
            "RS training must use natural multi-turn trajectories, previous supporter moves, "
            "explicit boundaries and repetition negatives; MP/MS/ME are structurally absent."
        ),
    }


def _evoemo_and_qa_profile() -> tuple[dict, dict]:
    users = _load_json(EVOEMO)
    sessions = [len(user["dialog_history"]) for user in users]
    events = [len(user["event_experience"]) for user in users]
    relations = [len(user["social_relationship"]) for user in users]
    summaries = [
        len(session.get("summary", "").split())
        for user in users
        for session in user["dialog_history"]
    ]
    qa_items: list[dict] = []
    for user in users:
        for group in user["questions"]:
            for item in group["questions"]:
                qa_items.append(item)
    capability_counts = Counter(item["capability"] for item in qa_items)
    evidence_counts = [len(item.get("evidence") or []) for item in qa_items]
    distinct_evidence_counts = [len(set(item.get("evidence") or [])) for item in qa_items]
    zero_evidence_by_capability = {
        capability: sum(
            not (item.get("evidence") or [])
            for item in qa_items
            if item["capability"] == capability
        )
        for capability in sorted(capability_counts)
    }
    evoemo = {
        "users": len(users),
        "strictly_longitudinal_sessions_total": sum(sessions),
        "sessions_per_user": _describe(sessions),
        "session_summary_words": _describe(summaries),
        "events_total": sum(events),
        "events_per_user": _describe(events),
        "relationships_total": sum(relations),
        "relationships_per_user": _describe(relations),
        "basic_profile_facts": sum(len(user["basic_info"]) for user in users),
        "basic_profile_fields": dict(
            sorted(Counter(key for user in users for key in user["basic_info"]).items())
        ),
        "stored_response_preference_items": 0,
        "training_implication": (
            "MP_PROFILE/MS/ME training must use same-user strictly-past catalogs with "
            "13-33 sessions, recurring topics, relations, temporal evolution and same-topic "
            "distractors; MP_PREFERENCE needs an internal-only experiment."
        ),
    }
    qa = {
        "released_questions": len(qa_items),
        "capability_counts": dict(sorted(capability_counts.items())),
        "evidence_per_question": _describe(evidence_counts),
        "questions_with_at_least_2_distinct_evidence": sum(
            value >= 2 for value in distinct_evidence_counts
        ),
        "questions_with_at_least_3_distinct_evidence": sum(
            value >= 3 for value in distinct_evidence_counts
        ),
        "questions_with_at_least_5_distinct_evidence": sum(
            value >= 5 for value in distinct_evidence_counts
        ),
        "zero_evidence_by_capability": zero_evidence_by_capability,
        "training_implication": (
            "The QA adapter/retriever must aggregate multiple historical evidence units, "
            "reason over time/conflict/user models and abstain. Exact-Rank1 response routing "
            "alone cannot satisfy this exam and should not absorb QA answer supervision."
        ),
    }
    return evoemo, qa


def _historical_external_surface_profile() -> dict:
    rows = _load_jsonl(EXTERNAL_SURFACES)
    profile: dict[str, dict] = {}
    for partition in sorted({row["partition"] for row in rows}):
        subset = [row for row in rows if row["partition"] == partition]
        profile[partition] = {
            "states": len(subset),
            "independent_users_or_dialogues": len(
                {row["user_id_private_analysis_only"] for row in subset}
            ),
            "candidate_present": {
                component: sum(
                    row["components"][component]["surface"]["candidate_present"]
                    for row in subset
                )
                for component in ("MP", "MS", "ME", "RS")
            },
            "structurally_executable": {
                component: sum(
                    row["components"][component]["structurally_executable"]
                    for row in subset
                )
                for component in ("MP", "MS", "ME", "RS")
            },
        }
    return profile


def _w7r_profile() -> dict:
    component_rows = {
        component: _load_json(W7R / f"{component.lower()}_states.json")
        for component in ("MP", "MS", "ME", "RS")
    }
    audit = _load_json(W7R_AUDIT)
    return {
        "states": {component: len(rows) for component, rows in component_rows.items()},
        "reported_counterfactual_groups": {
            component: len({row["counterfactual_group_id"] for row in rows})
            for component, rows in component_rows.items()
        },
        "user_or_family_clusters": {
            "MP_users": len({row["user_id"] for row in component_rows["MP"]}),
            "MS_users": len({row["user_id"] for row in component_rows["MS"]}),
            "ME_families": len({row["family"] for row in component_rows["ME"]}),
            "RS_families": len({row["family"] for row in component_rows["RS"]}),
        },
        "leader_audit_status": audit["status"],
        "blocking_findings": [item["id"] for item in audit["blocking_findings"]],
        "external_support_verdict": "NOT_READY",
    }


def main() -> None:
    evoemo, qa = _evoemo_and_qa_profile()
    report = {
        "protocol": "pm-v1.5-v5.3-external-exam-backward-training-spec-v1",
        "status": "CURRENT_W7R_NOT_READY_METHOD_LEVEL_P2R_REQUIRED",
        "api_calls": 0,
        "generated_reply_or_quality_risk_outcome_read": False,
        "grain": {
            "ESConv": "dialogue",
            "EvoEmo_response": "user cluster",
            "ES_MemEval_QA": "user cluster; question-level descriptive",
            "internal_training": "user/family counterfactual group",
        },
        "external_exam_profile": {
            "ESConv": _esconv_profile(),
            "EvoEmo": evoemo,
            "ES_MemEval": qa,
            "historical_v5_2_response_surfaces": _historical_external_surface_profile(),
        },
        "backward_training_requirements": {
            "shared": [
                "natural current utterances rather than machine instructions",
                "same-user strictly-past identity and no future/cross-user memory",
                "candidate pools with realistic density and same-topic hard negatives",
                "current answer absent from the visible query",
                "runtime-observable features only; construction target remains audit-only",
                "positive/decline/redundant/goal-mismatch crossed within every family",
                "user/family group binding across fit and confirmation",
                "paired ON/OFF outcomes from the frozen Step2, not hand-written worth-opening gold",
            ],
            "MP": [
                "structured field_type/value/owner/time representation",
                "same profile fact crossed with relevant, irrelevant, redundant and stale goals",
                "profile facts modeled externally; response preferences tested internally",
            ],
            "MS": [
                "13-33 item same-user catalogs with recurring-topic distractors",
                "natural continuity query that does not reveal the historical answer",
                "resolved/unresolved, conflict, age and owner/entity variation",
            ],
            "ME": [
                "typed past action plus result/mechanism and compiler-valid unavailable negatives",
                "current invite/decline/unknown readiness and redundancy",
                "same-topic different-action/result competitors; no intended-target feature",
            ],
            "RS": [
                "natural multi-turn context, previous support move and six-card preconditions",
                "already-executed, explicit-stop, one-point and burden negatives",
            ],
            "QA_adapter": [
                "Top-k multi-evidence aggregation rather than single Rank-1 only",
                "temporal/conflict/user-model reasoning and explicit abstention",
                "kept separate from response-PM paired-effect supervision",
            ],
        },
        "current_blueprint": _w7r_profile(),
        "decision": {
            "external_required_capabilities_are_sufficiently_specified": True,
            "current_w7r_training_blueprint_is_qualified": False,
            "main_gap_is_only_sample_count": False,
            "longitudinal_depth_and_competitive_density_gap": True,
            "label_feature_and_grouping_gaps_also_remain": True,
            "next": "one method-level P2R followed by one zero-API audit and immediate freeze",
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (
                ESCONV,
                ESCONV_SPLIT,
                ESCONV_BUILD,
                EVOEMO,
                EXTERNAL_SURFACES,
                W7R_AUDIT,
            )
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
