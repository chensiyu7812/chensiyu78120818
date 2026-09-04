#!/usr/bin/env python3
"""Audit why qualified D3 MS retrieval produced almost no response benefit."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-ms-nonuse-root-cause-audit-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _request_cell(text: str) -> str:
    lowered = text.lower()
    if "not advice" in lowered:
        return "PUT_INTO_WORDS_NO_ADVICE"
    if "small optional suggestion" in lowered:
        return "SMALL_OPTIONAL_SUGGESTION"
    if "which part matters most" in lowered:
        return "PRIORITY_EXPLORATION"
    if "understand why" in lowered:
        return "UNDERSTAND_HEAVINESS"
    return "OTHER"


def build(*, out_dir: Path) -> dict[str, Any]:
    blueprint_dir = ROOT / "outputs/pm_v1_5_d3_ms_replacement_step0_v1"
    plan_dir = ROOT / "outputs/pm_v1_5_d3_ms_replacement_generation_v1"
    execution_dir = ROOT / "outputs/pm_v1_5_d3_ms_replacement_generation_v1_execution"
    final_dir = ROOT / "outputs/pm_v1_5_d3_ms_final_effect_labels_v1"
    pairs = _rows(blueprint_dir / "d3_ms_replacement_primary_pairs.jsonl") + _rows(
        blueprint_dir / "d3_ms_replacement_repeat_pairs.jsonl"
    )
    calls = _rows(plan_dir / "call_plan.jsonl")
    outcomes = _rows(execution_dir / "generation_outcomes.jsonl")
    final = _rows(final_dir / "final_pair_effect_labels.jsonl")
    states = {
        str(row["state_id"]): row
        for row in _rows(blueprint_dir / "runtime_states.jsonl")
    }
    sources_by_card = {
        str(row["card_id"]): {
            str(item["memory_id"]): str(item["source"])
            for item in row["items"]
        }
        for row in _rows(blueprint_dir / "memory_backend.jsonl")
    }
    pair_id = {
        (str(row["contrast_slot_id"]), str(row["pair_role"])): str(row["pair_id"])
        for row in calls
    }
    call_by = {(str(row["pair_id"]), str(row["arm"])): row for row in calls}
    outcome_by = {(str(row["pair_id"]), str(row["arm"])): row for row in outcomes}
    final_by = {str(row["pair_id"]): row for row in final}

    audit: list[dict[str, Any]] = []
    for row in pairs:
        pid = pair_id[(str(row["contrast_slot_id"]), str(row["pair_role"]))]
        control_call = call_by[(pid, "control")]
        treatment_call = call_by[(pid, "treatment")]
        control = str(outcome_by[(pid, "control")]["response"])
        treatment = str(outcome_by[(pid, "treatment")]["response"])
        current = str(control_call["messages"][-1]["content"]).split(
            "Current user message:\n", 1
        )[1].split("\n\n", 1)[0]
        selected_ids = list(treatment_call["selected_memory_ids"])
        state = states[str(row["state_id"])]
        source_index = sources_by_card[str(state["card_id"])]
        selected_ms_ids = [
            memory_id
            for memory_id in selected_ids
            if source_index[str(memory_id)] == "MS"
        ]
        audit.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pid,
                "pair_role": row["pair_role"],
                "contrast_slot_id": row["contrast_slot_id"],
                "state_id": row["state_id"],
                "request_cell": _request_cell(current),
                "background_action": row["background_action"],
                "candidate_state_match_score": row["model_features"]["candidate_state_match_score"],
                "candidate_grounding_or_nonredundancy_score": row["model_features"]["candidate_grounding_or_nonredundancy_score"],
                "selected_ms_count": len(selected_ms_ids),
                "incremental_prompt_tokens": int(treatment_call["estimated_input_tokens"]) - int(control_call["estimated_input_tokens"]),
                "quality_verdict": final_by[pid]["quality_verdict"],
                "responses_exactly_equal": control == treatment,
                "response_character_similarity": SequenceMatcher(None, control.lower(), treatment.lower()).ratio(),
                "not_an_independent_group": row["pair_role"] != "primary",
            }
        )

    by_request: dict[str, Counter[str]] = defaultdict(Counter)
    by_background: dict[str, Counter[str]] = defaultdict(Counter)
    by_grounding: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit:
        by_request[str(row["request_cell"])][str(row["quality_verdict"])] += 1
        by_background[str(row["background_action"])][str(row["quality_verdict"])] += 1
        by_grounding[str(row["candidate_grounding_or_nonredundancy_score"])][str(row["quality_verdict"])] += 1
    similarities = sorted(float(row["response_character_similarity"]) for row in audit)
    primary = [row for row in audit if row["pair_role"] == "primary"]
    checks = {
        "40_pairs": len(audit) == 40,
        "32_primary_8_repeat": len(primary) == 32 and len(audit) - len(primary) == 8,
        "all_top2_injected": all(row["selected_ms_count"] == 2 for row in audit),
        "all_four_request_cells_present": set(by_request) == {
            "PUT_INTO_WORDS_NO_ADVICE",
            "SMALL_OPTIONAL_SUGGESTION",
            "PRIORITY_EXPLORATION",
            "UNDERSTAND_HEAVINESS",
        },
        "final_verdicts_1_1_38": Counter(row["quality_verdict"] for row in audit)
        == Counter({"tie": 38, "treatment": 1, "control": 1}),
        "eight_backgrounds_present": len(by_background) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(f"MS nonuse audit failed: {checks}")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "pair_root_cause_audit.jsonl"
    write_jsonl(audit_path, audit)
    report = {
        "protocol": PROTOCOL,
        "status": "ROOT_CAUSE_LOCALIZED_TREATMENT_AND_FEATURE_REDESIGN_REQUIRED",
        "dataset_grain": {
            "pair_realizations": 40,
            "independent_states": 32,
            "repeat_realizations": 8,
        },
        "final_quality_verdict_counts": dict(sorted(Counter(row["quality_verdict"] for row in audit).items())),
        "request_cell_verdict_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(by_request.items())
        },
        "background_verdict_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(by_background.items())
        },
        "grounding_axis_verdict_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(by_grounding.items())
        },
        "response_change_profile": {
            "exactly_equal_pairs": sum(row["responses_exactly_equal"] for row in audit),
            "character_similarity_at_least_0_9": sum(row["response_character_similarity"] >= 0.9 for row in audit),
            "median_character_similarity": similarities[len(similarities) // 2],
        },
        "findings": [
            {
                "severity": "critical_for_MS_training",
                "finding": "retrieval relevance was mistaken for incremental response utility",
                "evidence": "64/64 selected summaries were topical, yet 38/40 pair realizations were material ties",
                "impact": "the current labels contain no stable positive support for a discriminative MS gate",
            },
            {
                "severity": "high",
                "finding": "the candidate grounding axis does not measure useful new information",
                "evidence": "scores 0.83333333 and 1.0 each produced 19 ties and one non-tie; the lower score was induced by supporter metacommunication although MS retrieval is seeker-only",
                "impact": "the feature cannot express why one same-topic summary should alter the reply and another should be ignored",
            },
            {
                "severity": "high",
                "finding": "current states mostly restate the same episode represented in MS",
                "evidence": "all four request cells retained the same 1/1/38 pattern; summaries predominantly repeat topic, distress, and immediate focus already visible in the current message",
                "impact": "a strong generator can answer from current context alone, so ignoring MS is often correct",
            },
            {
                "severity": "medium",
                "finding": "the prompt compiler is conservative but not itself invalid",
                "evidence": "it says to use past information only when clearly helpful; forcing use would create stale or intrusive memory behavior",
                "impact": "repair should increase candidate incremental value and expose it to PM, not force the generator to mention memory",
            },
        ],
        "smallest_valid_repair": {
            "new_estimand": "P(material MS benefit | current state, realized strictly-prior summary, other-component background)",
            "state_design": "new content-disjoint users crossing useful-new prior outcome/context with relevant-but-redundant summaries under matched current requests",
            "feature_change": "replace the supporter-derived grounding proxy with an outcome-blind current-to-candidate incremental-value/alignment feature while retaining explicit background bits",
            "prompt_change": "none required initially; keep selective-use wording so uptake must be earned by useful evidence",
            "negative_controls": "same-topic redundant summaries, prior information fully restated in current context, and safe relevant summaries that do not answer the present request",
            "hard_exclusions": "wrong user, future, conflicting, rejected or stale evidence remain deterministic off and do not become paired-effect training rows",
            "minimum_support": "at least 8 independent post-risk positive states and 8 nonpositive states before fitting an MS head",
        },
        "checks": checks,
        "inputs": {
            "blueprint_report_sha256": sha256_file(blueprint_dir / "preflight_report.json"),
            "call_plan_sha256": sha256_file(plan_dir / "call_plan.jsonl"),
            "generation_outcomes_sha256": sha256_file(execution_dir / "generation_outcomes.jsonl"),
            "finalization_report_sha256": sha256_file(final_dir / "finalization_report.json"),
        },
        "outputs": {audit_path.name: sha256_file(audit_path)},
    }
    write_json(out_dir / "report_evidence.json", report)
    return report


def main() -> None:
    report = build(out_dir=ROOT / "outputs/pm_v1_5_d3_ms_nonuse_root_cause_v1")
    print(json.dumps({key: report[key] for key in ("protocol", "status", "final_quality_verdict_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
