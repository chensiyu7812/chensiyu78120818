"""Summarize the expanded SupportNeed bakeoff without reopening sealed data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import canonical_json, read_json, sha256_file, sha256_text
from .v1_5_support_need_bakeoff import PRIMARY_FACTOR_IDS


PROTOCOL = "pm-v1.5-support-need-expanded-bakeoff-analysis-v1"


def _validate_report(report: Mapping[str, Any], *, label: str) -> None:
    core = dict(report)
    digest = str(core.pop("report_sha256", ""))
    if digest != sha256_text(canonical_json(core)):
        raise RuntimeError(f"{label} report content hash is invalid")


def _axis_summary(
    factor_id: str, head: Mapping[str, Any]
) -> dict[str, Any]:
    status = str(head["status"])
    base = {
        "factor_id": factor_id,
        "status": status,
        "labeled_rows": int(head["labeled_rows"]),
        "independent_dialogue_groups": int(
            head["independent_dialogue_groups"]
        ),
        "class_counts": dict(head["class_counts"]),
    }
    if not status.startswith("COMPLETE"):
        return base
    best_name = str(head["best_view_by_soft_log_loss"])
    best = dict(head["feature_views"][best_name])
    comparison = head["paired_loss_delta_vs_lexical"].get(best_name)
    recalls = {
        str(key): (None if value is None else float(value))
        for key, value in best["per_class_recall"].items()
    }
    return {
        **base,
        "oof_prior_soft_log_loss": float(
            head["oof_prior"]["soft_log_loss"]
        ),
        "best_view_by_soft_log_loss": best_name,
        "best_soft_log_loss": float(best["soft_log_loss"]),
        "best_accuracy": float(best["accuracy"]),
        "best_per_class_recall": recalls,
        "best_all_classes_nonzero_recall": all(
            value is not None and value > 0.0 for value in recalls.values()
        ),
        "best_noncollapsed_view_by_soft_log_loss": head.get(
            "best_noncollapsed_view_by_soft_log_loss"
        ),
        "best_vs_lexical": (
            {
                "candidate_minus_reference_soft_log_loss": float(
                    comparison[
                        "candidate_minus_reference_soft_log_loss"
                    ]
                ),
                "ci95": [
                    float(value) for value in comparison["ci95"]
                ],
                "ci_excludes_zero_in_favor_of_best": (
                    float(comparison["ci95"][1]) < 0.0
                ),
            }
            if comparison is not None
            else {
                "candidate_minus_reference_soft_log_loss": 0.0,
                "ci95": None,
                "ci_excludes_zero_in_favor_of_best": False,
                "reason": "best_view_is_lexical_reference",
            }
        ),
    }


def analyze_expanded_support_need_bakeoff(
    *,
    prior_report_path: str | Path,
    candidate_report_path: str | Path,
    repro_report_path: str | Path,
) -> dict[str, Any]:
    """Validate reproducibility and produce a compact promotion decision."""

    prior_report_path = Path(prior_report_path)
    candidate_report_path = Path(candidate_report_path)
    repro_report_path = Path(repro_report_path)
    prior = read_json(prior_report_path)
    candidate = read_json(candidate_report_path)
    repro = read_json(repro_report_path)
    _validate_report(prior, label="prior bakeoff")
    _validate_report(candidate, label="expanded candidate bakeoff")
    _validate_report(repro, label="expanded repro bakeoff")
    if sha256_file(candidate_report_path) != sha256_file(repro_report_path):
        raise RuntimeError("expanded bakeoff candidate/repro files differ")
    if candidate != repro:
        raise RuntimeError("expanded bakeoff candidate/repro content differs")
    if (
        int(candidate["non_abstaining_anchor_count"]) != 39
        or int(candidate["independent_dialogue_groups"]) != 39
        or candidate.get("expansion_human_annotations_opened") is not True
        or candidate.get("internal_test_outcomes_opened") is not False
        or candidate.get("external_outcomes_opened") is not False
    ):
        raise RuntimeError("expanded bakeoff scope or data boundary drifted")

    axes = {
        factor_id: _axis_summary(
            factor_id, candidate["heads"][factor_id]
        )
        for factor_id in (
            *PRIMARY_FACTOR_IDS,
            "planning_readiness",
            "low_interaction_burden",
        )
    }
    prior_best = {
        factor_id: prior["heads"][factor_id].get(
            "best_view_by_soft_log_loss"
        )
        for factor_id in PRIMARY_FACTOR_IDS
    }
    expanded_best = {
        factor_id: axes[factor_id].get("best_view_by_soft_log_loss")
        for factor_id in PRIMARY_FACTOR_IDS
    }
    changed_primary_best = [
        factor_id
        for factor_id in PRIMARY_FACTOR_IDS
        if prior_best[factor_id] != expanded_best[factor_id]
    ]
    nli_best_primary = [
        factor_id
        for factor_id, name in expanded_best.items()
        if isinstance(name, str) and name.startswith("nli.")
    ]
    nli_significant_primary = [
        factor_id
        for factor_id in nli_best_primary
        if axes[factor_id]["best_vs_lexical"][
            "ci_excludes_zero_in_favor_of_best"
        ]
    ]
    qwen_embedding_best_primary = [
        factor_id
        for factor_id, name in expanded_best.items()
        if isinstance(name, str)
        and name.startswith(
            "embedding.qwen3_embedding_0_6b_support_need."
        )
    ]
    collapsed_best_primary = [
        factor_id
        for factor_id in PRIMARY_FACTOR_IDS
        if not axes[factor_id]["best_all_classes_nonzero_recall"]
    ]
    flat = _axis_summary(
        "legacy.support_mode",
        candidate["heads"]["legacy.support_mode"],
    )
    report_core = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_REPRODUCIBLE_FORMAL_FIT_NOT_AUTHORIZED",
        "source_lineage": {
            "prior_report_file_sha256": sha256_file(prior_report_path),
            "expanded_candidate_report_file_sha256": sha256_file(
                candidate_report_path
            ),
            "expanded_repro_report_file_sha256": sha256_file(
                repro_report_path
            ),
            "expanded_report_content_sha256": candidate["report_sha256"],
        },
        "anchor_count_change": {
            "prior_nonabstaining_groups": int(
                prior["non_abstaining_anchor_count"]
            ),
            "expanded_nonabstaining_groups": int(
                candidate["non_abstaining_anchor_count"]
            ),
            "confirmation_groups_opened": 0,
        },
        "primary_best_view_stability": {
            "prior": prior_best,
            "expanded": expanded_best,
            "changed_factor_ids": changed_primary_best,
            "changed_factor_count": len(changed_primary_best),
        },
        "axis_summaries": axes,
        "flat_support_mode_summary": flat,
        "candidate_findings": {
            "qwen_embedding_best_primary_factor_ids": (
                qwen_embedding_best_primary
            ),
            "qwen_embedding_promoted": False,
            "nli_best_primary_factor_ids": nli_best_primary,
            "nli_best_and_ci_better_than_lexical_factor_ids": (
                nli_significant_primary
            ),
            "nli_role": "axis_specific_feature_candidate_only",
            "collapsed_best_primary_factor_ids": collapsed_best_primary,
            "predeclared_hybrid_all_primary_checks_pass": bool(
                candidate["primary_all_hybrid_checks_pass"]
            ),
            "single_representation_promoted": False,
        },
        "decision": {
            "representation_promotion_authorized": False,
            "formal_factorized_fit_authorized": False,
            "flat_support_mode_fit_authorized": False,
            "confirmation_opening_authorized": False,
            "retain_transparent_observable_baseline": True,
            "retain_bge_small_failure_fallback": True,
            "retain_nli_scores_for_axis_specific_research": True,
            "next_data_action": (
                "collect or reannotate fresh outcome-blind dialogue states "
                "for exploration, focused-question, advice, planning, and "
                "non-low-burden boundaries; do not tune on confirmation"
            ),
        },
        "interpretation": (
            "The extra fit anchors improve class coverage and reveal useful "
            "NLI signal for two axes, but four of six primary best views "
            "change relative to the 23-group run, several lowest-loss heads "
            "still collapse a class, and the predeclared hybrid fails. This "
            "is evidence for axis-specific features and more data, not a "
            "deployable SupportNeed model."
        ),
        "automatic_gold_labels_created": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }

