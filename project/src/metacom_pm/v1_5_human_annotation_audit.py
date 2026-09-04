"""Inventory retained PM-v1.5 human-annotation assets and their safe uses."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import canonical_json, iter_jsonl, sha256_file, sha256_text


PROTOCOL = "pm-v1.5-historical-human-annotation-asset-audit-v1"
SUPPORT_NEED_REQUIRED = (
    "blind_item_id",
    "support_mode",
    "goals",
    "dialogue_phase",
    "nonclinical_urgency",
    "confidence",
    "abstain",
)
PREFERENCE_REQUIRED = (
    "blind_item_id",
    "overall_preference",
    "support_quality_preference",
)
STRATEGY_REVIEW_REQUIRED = (
    "card_id",
    "safe_general_technique",
    "support_move_clear",
    "when_to_use_valid",
    "when_not_to_use_valid",
    "mode_phase_goal_fit_valid",
    "burden_and_risk_flags_valid",
    "approve_for_train_only_pilot",
    "confidence",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return True


def _completed_count(
    rows: Sequence[Mapping[str, Any]], required: Iterable[str]
) -> int:
    fields = tuple(required)
    return sum(
        all(_present(row.get(field)) for field in fields) for row in rows
    )


def _support_need_completed_count(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    return sum(
        (
            bool(row.get("abstain"))
            and _present(row.get("blind_item_id"))
            and _present(row.get("confidence"))
        )
        or all(_present(row.get(field)) for field in SUPPORT_NEED_REQUIRED)
        for row in rows
    )


def _asset(
    *,
    asset_id: str,
    target: str,
    status: str,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    completed_rows: int,
    independent_units: int,
    support_need_fit_eligible: bool,
    allowed_uses: Sequence[str],
    prohibited_uses: Sequence[str],
    duplicate_or_derivative_of: str | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "target": target,
        "status": status,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "row_count": len(rows),
        "completed_row_count": int(completed_rows),
        "independent_units": int(independent_units),
        "support_need_fit_eligible": bool(support_need_fit_eligible),
        "allowed_uses": list(allowed_uses),
        "prohibited_uses": list(prohibited_uses),
        "duplicate_or_derivative_of": duplicate_or_derivative_of,
        "notes": list(notes),
    }


def _named_human_files(root: Path) -> list[Path]:
    terms = (
        "human",
        "reviewer",
        "annotation",
        "adjudicat",
        "spot_check",
    )
    paths: list[Path] = []
    for base in (root / "data", root / "outputs"):
        if not base.is_dir():
            continue
        paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and any(term in path.name.lower() for term in terms)
        )
    return sorted(set(paths))


def _file_index(paths: Sequence[Path], *, display_root: Path) -> dict[str, Any]:
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    rows = []
    for path in paths:
        digest = sha256_file(path)
        try:
            display = str(path.relative_to(display_root))
        except ValueError:
            display = str(path)
        rows.append(
            {
                "path": display,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        hash_to_paths[digest].append(display)
    duplicate_groups = [
        {"sha256": digest, "paths": sorted(group)}
        for digest, group in sorted(hash_to_paths.items())
        if len(group) > 1
    ]
    return {
        "file_count": len(rows),
        "distinct_file_hash_count": len(hash_to_paths),
        "duplicate_hash_group_count": len(duplicate_groups),
        "files": rows,
        "duplicate_hash_groups": duplicate_groups,
    }


def _csv_completion(rows: Sequence[Mapping[str, str]]) -> int:
    if not rows:
        return 0
    ignored = {"item_id", "annotator_id", "reviewer_id", "pair_id", "notes"}
    score_fields = [
        field for field in rows[0] if field not in ignored
    ]
    return sum(
        bool(score_fields)
        and all(_present(row.get(field)) for field in score_fields)
        for row in rows
    )


def audit_historical_human_annotations(
    *,
    project_root: str | Path,
    v2_project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a content-addressed, use-restricted inventory."""

    root = Path(project_root).resolve()
    contracts = root / "data/pm_v1_5_contracts"
    outputs = root / "outputs"

    support_raw_path = contracts / "support_need_human_annotations_v1.jsonl"
    support_raw = _jsonl(support_raw_path)
    support_completed = _support_need_completed_count(support_raw)
    support_nonabstain = sum(not bool(row["abstain"]) for row in support_raw)

    adjudication_dir = (
        outputs / "pm_v1_5_support_need_fit_adjudication_v1_candidate"
    )
    adjudicated_path = adjudication_dir / "adjudicated_annotations.jsonl"
    adjudicated = _jsonl(adjudicated_path)
    adjudicated_completed = _support_need_completed_count(adjudicated)
    raw_a_path = adjudication_dir / "raw_rater_a.jsonl"
    raw_b_path = adjudication_dir / "raw_rater_b.jsonl"
    raw_a = _jsonl(raw_a_path)
    raw_b = _jsonl(raw_b_path)
    combined_path = adjudication_dir / "combined_normalized_anchors.jsonl"
    combined = _jsonl(combined_path)
    combined_packet_path = (
        adjudication_dir / "combined_human_blind_packet.jsonl"
    )
    combined_packet = _jsonl(combined_packet_path)
    combined_nonabstain = sum(
        not bool(row["raw_annotation"]["abstain"]) for row in combined
    )

    low_budget_path = (
        contracts / "low_budget_judge_human_annotations_v1.jsonl"
    )
    low_budget = _jsonl(low_budget_path)
    low_budget_packet_path = (
        outputs
        / "pm_v1_5_low_budget_judge_qualification_packet_v1"
        / "human_blind_packet.jsonl"
    )
    low_budget_packet = _jsonl(low_budget_packet_path)
    role_path = (
        contracts / "role_decomposed_judge_human_annotations_v1.jsonl"
    )
    role = _jsonl(role_path)
    role_packet_path = (
        contracts
        / "role_decomposed_judge_packet_v1"
        / "human_blind_packet.jsonl"
    )
    role_packet = _jsonl(role_packet_path)
    strict_copy_path = (
        outputs
        / "pm_v1_5_strict_judge_bakeoff_v1_semantic_aggregation_repro_a"
        / "human_annotations_normalized.jsonl"
    )
    strict_copy = _jsonl(strict_copy_path)
    weak_template_path = (
        outputs
        / "pm_v1_5_factorized_weak_learning_human_anchor_v1"
        / "human_annotation_template.jsonl"
    )
    weak_template = _jsonl(weak_template_path)
    strategy_template_path = (
        outputs
        / "pm_v1_5_strategy_bank_v2_review_candidate_v2"
        / "human_annotation_template.jsonl"
    )
    strategy_template = _jsonl(strategy_template_path)
    generation_a_path = (
        outputs
        / "pm_v1_5_generation_pilot_semantic_review_post_repair"
        / "reviewer_a.csv"
    )
    generation_b_path = generation_a_path.with_name("reviewer_b.csv")
    generation_a = _csv(generation_a_path)
    generation_b = _csv(generation_b_path)
    expansion_template_path = (
        outputs
        / "pm_v1_5_support_need_expansion_packet_v2_candidate"
        / "human_annotation_template.jsonl"
    )
    expansion_template = _jsonl(expansion_template_path)
    fit_ids = {str(row["blind_item_id"]) for row in adjudicated}
    expansion_ids = {
        str(row["blind_item_id"]) for row in expansion_template
    }
    confirmation_ids = expansion_ids - fit_ids
    if len(fit_ids & expansion_ids) != 16 or len(confirmation_ids) != 8:
        raise RuntimeError("support-need fit/confirmation partition drifted")

    def visible_states(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            canonical_json(dict(row["visible_state"])) for row in rows
        }

    current_visible = visible_states(combined_packet)
    low_visible = visible_states(low_budget_packet)
    role_visible = visible_states(role_packet)
    if (
        len(current_visible) != 40
        or len(low_visible) != 12
        or len(role_visible) != 12
    ):
        raise RuntimeError("historical visible-state uniqueness drifted")

    assets = [
        _asset(
            asset_id="support_need_human_anchor_v1",
            target="current_user_support_need",
            status="COMPLETE_BOUND_PARTIAL_ANCHOR",
            path=support_raw_path,
            rows=support_raw,
            completed_rows=support_completed,
            independent_units=support_nonabstain,
            support_need_fit_eligible=True,
            allowed_uses=(
                "train_only_factorized_support_need_fit",
                "grouped_oof_representation_diagnostic",
                "rubric_error_analysis",
            ),
            prohibited_uses=(
                "automatic_action_gold",
                "independent_test_claim",
                "clinical_label",
            ),
            notes=(
                "One abstention is retained but excluded from fitted heads.",
                "Legacy burden semantics are normalized separately and raw rows remain immutable.",
            ),
        ),
        _asset(
            asset_id="support_need_factorized_fit_adjudication_v1",
            target="current_user_support_need",
            status="COMPLETE_DUAL_REVIEW_ADJUDICATED",
            path=adjudicated_path,
            rows=adjudicated,
            completed_rows=adjudicated_completed,
            independent_units=len(adjudicated),
            support_need_fit_eligible=True,
            allowed_uses=(
                "train_only_factorized_support_need_fit",
                "grouped_oof_representation_diagnostic",
                "interrater_and_boundary_error_analysis",
            ),
            prohibited_uses=(
                "count_two_raters_as_two_dialogue_groups",
                "use_raw_rater_rows_as_independent_targets",
                "open_or_tune_on_confirmation_subset",
                "automatic_action_gold",
            ),
            notes=(
                f"Raw reviewer files are preserved at {raw_a_path} and {raw_b_path}.",
                "Only the adjudicated row may enter fitting; pre-adjudication disagreement remains provenance.",
            ),
        ),
        _asset(
            asset_id="low_budget_judge_human_anchor_v1",
            target="candidate_response_pair_preference",
            status="COMPLETE_FROZEN_JUDGE_ANCHOR",
            path=low_budget_path,
            rows=low_budget,
            completed_rows=_completed_count(
                low_budget, (*PREFERENCE_REQUIRED, "confidence")
            ),
            independent_units=len(low_budget),
            support_need_fit_eligible=False,
            allowed_uses=(
                "judge_qualification",
                "judge_human_agreement_analysis",
                "response_quality_error_analysis",
                "future_support_need_reannotation_candidate_pool",
            ),
            prohibited_uses=(
                "support_need_training_label",
                "infer_need_from_winning_response",
                "count_normalized_copies_as_new_human_units",
            ),
            notes=(
                "Candidate-relative, response-visible judgments contain outcome and treatment information.",
            ),
        ),
        _asset(
            asset_id="role_decomposed_judge_human_anchor_v1",
            target="response_preference_and_error_taxonomy",
            status="COMPLETE_FROZEN_ROLE_DECOMPOSED_ANCHOR",
            path=role_path,
            rows=role,
            completed_rows=_completed_count(
                role,
                (
                    *PREFERENCE_REQUIRED,
                    "quality_confidence",
                    "candidate_a_audits",
                    "candidate_b_audits",
                ),
            ),
            independent_units=len(role),
            support_need_fit_eligible=False,
            allowed_uses=(
                "judge_semantic_shape_qualification",
                "risk_dimension_error_analysis",
                "response_preference_analysis",
                "future_support_need_reannotation_candidate_pool",
            ),
            prohibited_uses=(
                "support_need_training_label",
                "derive_need_from_candidate_preference_or_audit",
                "automatic_response_gold",
            ),
        ),
        _asset(
            asset_id="strict_judge_normalized_human_copy_v1",
            target="candidate_response_pair_preference",
            status="DERIVATIVE_COPY_NOT_NEW_HUMAN_DATA",
            path=strict_copy_path,
            rows=strict_copy,
            completed_rows=len(strict_copy),
            independent_units=0,
            support_need_fit_eligible=False,
            allowed_uses=("strict_judge_bakeoff_aggregation",),
            prohibited_uses=(
                "count_as_additional_human_annotations",
                "support_need_training_label",
            ),
            duplicate_or_derivative_of="low_budget_judge_human_anchor_v1",
        ),
        _asset(
            asset_id="factorized_component_effect_human_template_v1",
            target="component_effect_pair_preference",
            status="TEMPLATE_UNANNOTATED",
            path=weak_template_path,
            rows=weak_template,
            completed_rows=_completed_count(
                weak_template, (*PREFERENCE_REQUIRED, "confidence")
            ),
            independent_units=0,
            support_need_fit_eligible=False,
            allowed_uses=(
                "future_component_effect_label_function_calibration_after_annotation",
            ),
            prohibited_uses=(
                "current_training",
                "support_need_training_label",
            ),
            notes=(
                "This packet targets realized component benefit, not pre-response user need.",
            ),
        ),
        _asset(
            asset_id="strategy_bank_v2_human_template",
            target="strategy_card_content_quality",
            status="TEMPLATE_UNANNOTATED",
            path=strategy_template_path,
            rows=strategy_template,
            completed_rows=_completed_count(
                strategy_template, STRATEGY_REVIEW_REQUIRED
            ),
            independent_units=0,
            support_need_fit_eligible=False,
            allowed_uses=(
                "strategy_bank_card_approval_after_annotation",
                "strategy_retrieval_applicability_audit",
            ),
            prohibited_uses=(
                "current_training",
                "current_user_support_need_label",
                "pm_component_effect_gold",
            ),
        ),
        _asset(
            asset_id="generation_pilot_semantic_review_template",
            target="synthetic_generation_surface_fidelity",
            status="TEMPLATE_UNANNOTATED",
            path=generation_a_path,
            rows=[*generation_a, *generation_b],
            completed_rows=(
                _csv_completion(generation_a)
                + _csv_completion(generation_b)
            ),
            independent_units=0,
            support_need_fit_eligible=False,
            allowed_uses=(
                "generation_surface_gate_after_independent_completion",
            ),
            prohibited_uses=(
                "current_training",
                "support_need_training_label",
                "judge_human_anchor",
            ),
        ),
        _asset(
            asset_id="support_need_confirmation_subset_v1",
            target="current_user_support_need_confirmation",
            status="SEALED_UNANNOTATED_CONFIRMATION",
            path=expansion_template_path,
            rows=expansion_template,
            completed_rows=0,
            independent_units=0,
            support_need_fit_eligible=False,
            allowed_uses=(
                "one_time_confirmation_after_model_family_and_thresholds_freeze",
            ),
            prohibited_uses=(
                "current_model_selection",
                "current_threshold_tuning",
                "current_error_analysis",
            ),
            notes=(
                "The expansion packet contains 16 now-adjudicated fit IDs and 8 still-sealed confirmation IDs.",
            ),
        ),
    ]

    external_templates: list[dict[str, Any]] = []
    v2_index: dict[str, Any] | None = None
    if v2_project_root is not None:
        v2_root = Path(v2_project_root).resolve()
        patterns = (
            "outputs/pm_v2_generation_pilot_semantic_review_v*/reviewer_?.csv",
            "outputs/v1_5_bank_mechanism_diagnostic/reviewer_?.csv",
        )
        v2_paths = sorted(
            {
                path
                for pattern in patterns
                for path in v2_root.glob(pattern)
                if path.is_file() and "template" not in path.name
            }
        )
        for path in v2_paths:
            rows = _csv(path)
            external_templates.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "row_count": len(rows),
                    "completed_row_count": _csv_completion(rows),
                    "status": (
                        "COMPLETED"
                        if _csv_completion(rows) == len(rows)
                        else "TEMPLATE_UNANNOTATED"
                    ),
                    "support_need_fit_eligible": False,
                }
            )
        v2_index = _file_index(v2_paths, display_root=v2_root)

    current_index = _file_index(
        _named_human_files(root), display_root=root
    )
    status_counts = Counter(asset["status"] for asset in assets)
    report_core = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_USE_RESTRICTED_INVENTORY",
        "scope": {
            "project_root": str(root),
            "v2_project_root": (
                str(Path(v2_project_root).resolve())
                if v2_project_root is not None
                else None
            ),
            "retained_files_only": True,
            "deleted_archives_reconstructed": False,
        },
        "summary": {
            "logical_asset_count": len(assets),
            "status_counts": dict(sorted(status_counts.items())),
            "completed_support_need_human_rows": (
                support_completed + adjudicated_completed
            ),
            "support_need_nonabstaining_independent_groups": (
                combined_nonabstain
            ),
            "completed_non_support_need_human_rows": (
                len(low_budget) + len(role)
            ),
            "current_project_named_human_files": current_index[
                "file_count"
            ],
            "current_project_distinct_named_human_file_hashes": (
                current_index["distinct_file_hash_count"]
            ),
            "external_completed_rows_found": sum(
                row["completed_row_count"] for row in external_templates
            ),
        },
        "support_need_fit_decision": {
            "eligible_asset_ids": [
                "support_need_human_anchor_v1",
                "support_need_factorized_fit_adjudication_v1",
            ],
            "combined_rows": len(combined),
            "combined_nonabstaining_rows": combined_nonabstain,
            "independent_dialogue_groups": combined_nonabstain,
            "raw_rater_rows": len(raw_a) + len(raw_b),
            "raw_rater_rows_add_independent_groups": False,
            "confirmation_ids_opened": False,
            "confirmation_id_count": len(confirmation_ids),
            "non_support_need_human_assets_may_enter_fit": False,
        },
        "historical_judge_state_reannotation_opportunity": {
            "status": "ELIGIBLE_AS_NEW_BLINDED_ACTIVE_LEARNING_POOL_ONLY",
            "candidate_dialogue_states": len(low_visible | role_visible),
            "low_budget_unique_states": len(low_visible),
            "role_decomposed_unique_states": len(role_visible),
            "low_budget_role_exact_overlap": len(low_visible & role_visible),
            "judge_state_current_support_need_exact_overlap": len(
                (low_visible | role_visible) & current_visible
            ),
            "source_packet_file_sha256": {
                "low_budget": sha256_file(low_budget_packet_path),
                "role_decomposed": sha256_file(role_packet_path),
                "current_support_need": sha256_file(
                    combined_packet_path
                ),
            },
            "reuse_rule": (
                "strip candidates, selected context, authorized context, "
                "old annotations, and old preferences; reannotate visible "
                "dialogue only under the current factorized rubric"
            ),
            "sampling_role": (
                "response-difference-enriched active-learning pool, not an "
                "unbiased prevalence sample"
            ),
            "old_human_labels_reused_as_need_targets": False,
            "may_enter_current_fit_without_reannotation": False,
        },
        "logical_assets": assets,
        "current_project_file_index": current_index,
        "external_v2_review_files": external_templates,
        "external_v2_file_index": v2_index,
        "global_use_boundaries": {
            "human_annotation_is_not_automatic_gold": True,
            "task_mismatched_human_labels_must_not_be_pooled": True,
            "candidate_or_response_visible_reviews_are_post_treatment_for_need_diagnosis": True,
            "candidate_repro_and_normalized_copies_do_not_increase_ess": True,
            "one_dialogue_state_contributes_at_most_one_independent_group": True,
            "confirmation_subset_remains_sealed": True,
        },
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
