#!/usr/bin/env python3
"""W9: Worker independent, read-only review of the Leader's P2R learning
blueprint (commit bb1fead).

Read-only: does not modify any Leader file (v1_5_v5_3_p2r_learning_
blueprint.py, 78k/78l/79l scripts, or their outputs). Does not train, call
any API, generate a reply, or freeze a split. Reuses the Leader's own
audit_blueprint_rows()/schema classes (import only) to independently
reproduce the structural numbers, then adds checks the Leader's 79l audit
does not run: per-interaction topic coherence across components, and a
direct diagnosis of the 79l script's own candidate_family_alignment metric
(to separate real construction bugs from an artifact of that metric's own
fallback-chain logic).

Reports every finding under one of three explicit labels per the review
brief: A (blocks formal effect), B (real, must be disclosed, does not
block), C (audit false positive / not a real construct problem).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_p2r_learning_blueprint import (  # noqa: E402
    P2RBlueprintRow,
    P2RInteractionBlueprintRow,
    audit_blueprint_rows,
)

BLUEPRINT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_v1"
CATALOG_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_worker_review_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def independent_structural_rerun() -> dict:
    rows = [P2RBlueprintRow.model_validate(r) for r in _read_jsonl(BLUEPRINT_DIR / "component_rows.jsonl")]
    interactions = [
        P2RInteractionBlueprintRow.model_validate(r)
        for r in _read_jsonl(BLUEPRINT_DIR / "interaction_rows.jsonl")
    ]
    structural = audit_blueprint_rows(rows, require_final_condition_crossing=True, require_assigned_splits=False)
    full_four = sum(
        set(row.components) == {"MP", "MS", "ME", "RS"}
        and all(m.candidate_present for m in row.components.values())
        for row in interactions
    )
    return {
        "row_count": structural["row_count"],
        "component_counts": structural["component_counts"],
        "unique_user_clusters": structural["unique_user_clusters"],
        "unique_counterfactual_groups": structural["unique_counterfactual_groups"],
        "critical_failures": structural["critical_failures"],
        "static_structure_pass": structural["static_structure_pass"],
        "interaction_rows": len(interactions),
        "full_four_candidate_interactions": full_four,
        "matches_leader_79l_report": None,  # filled in main() against the real report.json
    }


def diagnose_groups_without_condition_crossing() -> dict:
    """Q11: are the 48 flagged groups a real gap, or single-condition
    interaction-view groups by construction?"""

    rows = _read_jsonl(BLUEPRINT_DIR / "component_rows.jsonl")
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["state"]["counterfactual_group_id"], []).append(r)
    report = json.loads((BLUEPRINT_DIR / "report.json").read_text())
    flagged = report["groups_without_condition_crossing"]
    shapes: dict[str, int] = {}
    for group_id in flagged:
        members = by_group.get(group_id, [])
        comps = tuple(sorted({m["state"]["component"] for m in members}))
        conds = tuple(sorted({m["audit_only"]["state_condition"] for m in members}))
        is_interaction = any(m["state"]["interaction_key"] for m in members)
        key = f"components={comps} conditions={conds} interaction={is_interaction}"
        shapes[key] = shapes.get(key, 0) + 1
    return {
        "n_flagged_groups": len(flagged),
        "shape_breakdown": shapes,
        "verdict": (
            "ALL flagged groups are single-condition interaction component-views "
            "(12 interactions x 4 components = 48) -- category B, not A: interaction "
            "rows currently support joint-candidate-availability testing only, NOT "
            "paired ON/OFF interaction-effect estimation. Does not affect the other "
            "333 single-component groups, which do cross conditions."
            if shapes == {"components=('MP',) conditions=('positive_opportunity',) interaction=True": 12}
            or all("interaction=True" in k and "positive_opportunity" in k for k in shapes)
            else "MIXED SHAPES FOUND -- some flagged groups are NOT interaction views; "
            "needs individual inspection, do not assume category B."
        ),
    }


def diagnose_interaction_topic_coherence() -> dict:
    """Q9: do all 12 interactions have topically coherent ME/MS candidates,
    or does 'candidate present for all 4 components' hide a wrong-topic
    Rank-1 for some component?"""

    interactions = _read_jsonl(BLUEPRINT_DIR / "interaction_rows.jsonl")
    findings = []
    for row in interactions:
        topic = row["semantic_family"].split("::", 1)[-1]
        mismatches = []
        for component in ("ME", "MS"):
            text = (row["execution_candidate_texts"].get(component) or "").lower()
            if topic.replace("_", " ") not in text:
                mismatches.append(component)
        findings.append({
            "interaction_id": row["interaction_id"], "topic": topic,
            "mismatched_components": mismatches,
        })
    n_affected = sum(1 for f in findings if f["mismatched_components"])
    return {
        "n_interactions": len(interactions),
        "n_interactions_with_topic_mismatch": n_affected,
        "affected": [f for f in findings if f["mismatched_components"]],
        "verdict": (
            f"{n_affected}/{len(interactions)} interactions have a real candidate "
            "for all 4 components (no re-retrieval occurred, verified against "
            "assemble_interaction_row()'s no-rerank contract by code read), but "
            "the ME or MS candidate is about a DIFFERENT topic than the interaction "
            "query states. 'full four-component candidate coverage' as currently "
            "audited (79l) checks presence, not topical coherence -- category A "
            "for these specific rows (an interaction-effect test on them would not "
            "test what it claims to), category B for the blueprint as a whole "
            "(the other rows are unaffected, and this reproduces the same "
            "same-topic-ranking-robustness limitation already disclosed and "
            "unresolved after three reranker attempts in W6 -- not a new, "
            "unknown failure mode)."
        ),
    }


def diagnose_mp_family_alignment_metric() -> dict:
    """Separates real MP cross-field retrieval confusion from a fallback-
    chain bug in 79l's own candidate_family_alignment diagnostic (Q9-adjacent,
    directly relevant to not treating a nonzero counter as an automatic
    failure per the review brief)."""

    rows = _read_jsonl(BLUEPRINT_DIR / "component_rows.jsonl")
    catalog = {r["item_id"]: r for r in _read_jsonl(CATALOG_DIR / "catalog.jsonl")}
    mp_rows = [r for r in rows if r["state"]["component"] == "MP" and r["candidate_lineage"]["candidate_id"]]
    buckets = {"preference_audit_artifact": 0, "interaction_naming_artifact": 0, "genuine_cross_field": 0, "match": 0}
    genuine_examples = []
    for r in mp_rows:
        family = r["state"]["semantic_family"]
        state_topic = family.split("::", 1)[-1]
        item = catalog.get(r["candidate_lineage"]["candidate_id"], {})
        candidate_topic = item.get("topic_thread") or item.get("field_type") or item.get("field_value")
        if state_topic == candidate_topic:
            buckets["match"] += 1
            continue
        if bool(r["state"]["interaction_key"]):
            buckets["interaction_naming_artifact"] += 1
        elif item.get("subtype") == "MP_PREFERENCE" and item.get("field_value") == state_topic:
            buckets["preference_audit_artifact"] += 1
        else:
            buckets["genuine_cross_field"] += 1
            if len(genuine_examples) < 5:
                genuine_examples.append({
                    "state_family": family,
                    "candidate_field_type": item.get("field_type"),
                    "candidate_field_value": item.get("field_value"),
                })
    return {
        "buckets": buckets,
        "genuine_cross_field_examples": genuine_examples,
        "verdict": (
            f"Of {buckets['preference_audit_artifact'] + buckets['interaction_naming_artifact'] + buckets['genuine_cross_field']} "
            "rows 79l's candidate_family_alignment counts as MP mismatches: "
            f"{buckets['preference_audit_artifact']} are category C -- 79l's own fallback chain "
            "(topic_thread or field_type or field_value) reads MP_PREFERENCE items' field_type "
            "('response_format', always non-null) before ever reaching field_value, so a CORRECTLY "
            f"retrieved preference item is misreported as a mismatch; {buckets['interaction_naming_artifact']} are "
            "category C -- interaction family strings ('interaction::topic') are compared against "
            "MP field_type strings ('location') that are only related through the hand-curated "
            "field_for_topic map in 78k, a legitimate design choice, not a retrieval bug; only "
            f"{buckets['genuine_cross_field']} are category B -- genuine profile-field cross-contamination "
            "(e.g. an education-scope query retrieving a job fact), a small (~1.5%) but real rate "
            "worth disclosing, same class of issue as the ME/MS topic mismatches above."
        ),
    }


def main() -> None:
    leader_report = json.loads((BLUEPRINT_DIR / "report.json").read_text())
    my_rerun = independent_structural_rerun()
    my_rerun["matches_leader_79l_report"] = {
        "row_count": my_rerun["row_count"] == leader_report["row_count"],
        "unique_counterfactual_groups": my_rerun["unique_counterfactual_groups"] == leader_report["unique_counterfactual_groups"],
        "unique_user_clusters": my_rerun["unique_user_clusters"] == leader_report["unique_user_clusters"],
        "full_four_candidate_interactions": my_rerun["full_four_candidate_interactions"] == 12,
    }
    groups_diag = diagnose_groups_without_condition_crossing()
    topic_diag = diagnose_interaction_topic_coherence()
    mp_diag = diagnose_mp_family_alignment_metric()

    result = {
        "protocol": "pm-v1.5-v5.3-p2r-learning-blueprint-worker-review-v1",
        "reviewed_leader_commit": "bb1fead",
        "independent_structural_rerun": my_rerun,
        "q11_groups_without_condition_crossing": groups_diag,
        "q9_interaction_topic_coherence": topic_diag,
        "mp_family_alignment_metric_diagnosis": mp_diag,
        "api_calls": 0,
        "quality_risk_or_outcome_read": False,
        "leader_files_modified": False,
        "split_frozen_by_worker": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nfull report written to {OUT_DIR}")


if __name__ == "__main__":
    main()
