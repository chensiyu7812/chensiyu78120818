"""Outcome-blind learnability checks for the SupportNeed human anchors.

This is intentionally a diagnostic, not a formal PM fit.  It asks whether a
small set of independent, train-only human anchors contains recoverable signal
under the exact frozen BAAI representation and under transparent baselines.
No ESConv metadata, target supporter response, strategy annotation, survey,
memory item, internal-test outcome, or external outcome is read.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupKFold

from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import SemanticTextEncoder
from .v1_5_support_need import (
    DIALOGUE_PHASE_IDS,
    GOAL_IDS,
    NONCLINICAL_URGENCY_IDS,
    SUPPORT_MODE_IDS,
    cross_fit_categorical_soft_head,
    explicit_support_boundaries,
)


PROTOCOL = "pm-v1.5-support-need-human-anchor-learnability-v1"
FEATURE_VIEW_NAMES = (
    "observable_structure",
    "lexical_hash",
    "baai_current",
    "baai_multiview",
    "baai_multiview_plus_observable",
)


def _normalized(value: Any, *, empty: str = "[none]") -> str:
    text = " ".join(str(value or "").split())
    return text or empty


def _visible_texts(visible_state: Mapping[str, Any]) -> tuple[str, ...]:
    current = _normalized(visible_state.get("current_user_text"))
    turns = list(visible_state.get("recent_dialogue") or [])
    last_assistant = next(
        (
            _normalized(turn.get("content"))
            for turn in reversed(turns)
            if turn.get("role") == "assistant"
            and str(turn.get("content") or "").strip()
        ),
        "[none]",
    )
    recent = "\n".join(
        f"{turn.get('role')}: {_normalized(turn.get('content'))}"
        for turn in turns
        if turn.get("role") in {"user", "assistant"}
        and str(turn.get("content") or "").strip()
    )
    summary = _normalized(visible_state.get("session_summary"))
    return current, last_assistant, _normalized(recent), summary


def _observable_features(visible_state: Mapping[str, Any]) -> np.ndarray:
    current, last_assistant, recent, summary = _visible_texts(visible_state)
    turns = list(visible_state.get("recent_dialogue") or [])
    boundaries = explicit_support_boundaries(current)
    values = [
        math.log1p(len(current.split())),
        math.log1p(len(last_assistant.split())),
        math.log1p(len(recent.split())),
        math.log1p(len(summary.split())),
        math.log1p(len(turns)),
        math.log1p(sum(turn.get("role") == "user" for turn in turns)),
        math.log1p(sum(turn.get("role") == "assistant" for turn in turns)),
        float("?" in current),
        float("!" in current),
        float(
            current.lower().startswith(
                ("yes", "no", "yeah", "nope", "okay", "ok")
            )
        ),
        float(boundaries.advice_rejected),
        float(boundaries.advice_requested),
        float(boundaries.one_small_step_requested),
        float(boundaries.listen_first_requested),
        float(boundaries.question_or_task_burden_limit),
    ]
    return np.asarray(values, dtype=float)


def _compose_lexical_text(visible_state: Mapping[str, Any]) -> str:
    current, last_assistant, recent, summary = _visible_texts(visible_state)
    return (
        f"[current] {current}\n[last_assistant] {last_assistant}\n"
        f"[history] {recent}\n[summary] {summary}"
    )


def _one_hot(value: str, classes: Sequence[str]) -> dict[str, float]:
    if value not in classes:
        raise ValueError("human anchor target is outside the frozen classes")
    return {class_id: float(class_id == value) for class_id in classes}


def _oof_prior_baseline(
    *,
    targets: Sequence[Mapping[str, float]],
    groups: Sequence[str],
    classes: Sequence[str],
    folds: int = 5,
) -> dict[str, Any]:
    y = np.asarray(
        [[float(row[class_id]) for class_id in classes] for row in targets],
        dtype=float,
    )
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    predictions = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    for train_index, test_index in splitter.split(
        y, np.argmax(y, axis=1), group_values
    ):
        prior = np.mean(y[train_index], axis=0)
        prior = np.maximum(prior, 1e-6)
        prior = prior / prior.sum()
        predictions[test_index] = prior
    truth = np.argmax(y, axis=1)
    predicted = np.argmax(predictions, axis=1)
    log_loss = float(
        np.mean(
            -np.sum(y * np.log(np.clip(predictions, 1e-12, 1.0)), axis=1)
        )
    )
    recalls = {}
    for index, class_id in enumerate(classes):
        mask = truth == index
        recalls[class_id] = (
            float(np.mean(predicted[mask] == index)) if np.any(mask) else None
        )
    return {
        "protocol": "pm-v1.5-support-need-oof-prior-baseline-v1",
        "soft_log_loss": log_loss,
        "accuracy": float(np.mean(predicted == truth)),
        "per_class_recall": recalls,
    }


def _encode_feature_views(
    *,
    encoder: SemanticTextEncoder,
    packet_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    visible_states = [dict(row["visible_state"]) for row in packet_rows]
    view_text_rows = [_visible_texts(row) for row in visible_states]
    flat_texts = [text for row in view_text_rows for text in row]
    encoded = np.asarray(encoder.encode(flat_texts), dtype=float)
    if encoded.shape != (
        len(packet_rows) * 4,
        int(encoder.spec.output_dimension),
    ):
        raise RuntimeError("support-need anchor BAAI matrix shape drifted")
    matrices = encoded.reshape(
        len(packet_rows), 4, int(encoder.spec.output_dimension)
    )
    deltas = matrices[:, 0, :] - np.mean(matrices[:, 1:, :], axis=1)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    deltas = deltas / np.maximum(norms, 1e-12)
    semantic_multiview = np.concatenate(
        [matrices.reshape(len(packet_rows), -1), deltas], axis=1
    )
    observable = np.vstack(
        [_observable_features(row) for row in visible_states]
    )
    lexical = HashingVectorizer(
        n_features=256,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
    ).transform([_compose_lexical_text(row) for row in visible_states])
    lexical_dense = np.asarray(lexical.toarray(), dtype=float)
    feature_views = {
        "observable_structure": observable,
        "lexical_hash": np.concatenate([lexical_dense, observable], axis=1),
        "baai_current": matrices[:, 0, :],
        "baai_multiview": semantic_multiview,
        "baai_multiview_plus_observable": np.concatenate(
            [semantic_multiview, observable], axis=1
        ),
    }
    if set(feature_views) != set(FEATURE_VIEW_NAMES):
        raise RuntimeError("support-need feature-view contract drifted")
    if any(
        matrix.shape[0] != len(packet_rows)
        or not np.all(np.isfinite(matrix))
        for matrix in feature_views.values()
    ):
        raise RuntimeError("support-need feature view is malformed")
    return feature_views


def run_human_anchor_learnability_diagnostic(
    *,
    encoder: SemanticTextEncoder,
    packet_rows: Sequence[Mapping[str, Any]],
    normalized_anchor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare transparent feature views using only held-out anchor rows."""

    packet_by_id = {
        str(row["blind_item_id"]): dict(row) for row in packet_rows
    }
    anchor_by_id = {
        str(row["blind_item_id"]): dict(row)
        for row in normalized_anchor_rows
    }
    if len(packet_by_id) != len(packet_rows):
        raise RuntimeError("support-need packet contains duplicate IDs")
    if set(anchor_by_id) != set(packet_by_id):
        raise RuntimeError("normalized anchors do not exactly cover the packet")
    ordered_ids = [
        str(row["blind_item_id"])
        for row in packet_rows
        if not bool(anchor_by_id[str(row["blind_item_id"])]["raw_annotation"][
            "abstain"
        ])
    ]
    if len(ordered_ids) < 15:
        raise RuntimeError("too few non-abstaining anchors for learnability check")
    ordered_packet = [packet_by_id[row_id] for row_id in ordered_ids]
    ordered_anchors = [anchor_by_id[row_id] for row_id in ordered_ids]
    feature_views = _encode_feature_views(
        encoder=encoder,
        packet_rows=ordered_packet,
    )
    groups = ordered_ids
    weights = [
        float(row["raw_annotation"]["confidence"]) / 5.0
        for row in ordered_anchors
    ]
    target_specs: dict[str, tuple[tuple[str, ...], list[dict[str, float]]]] = {
        "support_mode": (
            SUPPORT_MODE_IDS,
            [
                _one_hot(
                    str(row["raw_annotation"]["support_mode"]),
                    SUPPORT_MODE_IDS,
                )
                for row in ordered_anchors
            ],
        ),
        "interaction_posture": (
            ("non_directive", "explore", "directive"),
            [
                _one_hot(
                    (
                        "non_directive"
                        if row["raw_annotation"]["support_mode"]
                        in {"listen", "comfort_reassure"}
                        else (
                            "explore"
                            if row["raw_annotation"]["support_mode"] == "explore"
                            else "directive"
                        )
                    ),
                    ("non_directive", "explore", "directive"),
                )
                for row in ordered_anchors
            ],
        ),
        "dialogue_phase": (
            DIALOGUE_PHASE_IDS,
            [
                _one_hot(
                    str(row["raw_annotation"]["dialogue_phase"]),
                    DIALOGUE_PHASE_IDS,
                )
                for row in ordered_anchors
            ],
        ),
        "nonclinical_urgency": (
            NONCLINICAL_URGENCY_IDS,
            [
                _one_hot(
                    str(row["raw_annotation"]["nonclinical_urgency"]),
                    NONCLINICAL_URGENCY_IDS,
                )
                for row in ordered_anchors
            ],
        ),
    }
    for goal in GOAL_IDS:
        target_specs[f"goal.{goal}"] = (
            ("negative", "positive"),
            [
                _one_hot(
                    (
                        "positive"
                        if goal in row["raw_annotation"]["goals"]
                        else "negative"
                    ),
                    ("negative", "positive"),
                )
                for row in ordered_anchors
            ],
        )

    head_reports: dict[str, Any] = {}
    for head_name, (classes, targets) in target_specs.items():
        hard_counts = Counter(
            classes[int(np.argmax([target[key] for key in classes]))]
            for target in targets
        )
        if any(hard_counts.get(class_id, 0) < 2 for class_id in classes):
            head_reports[head_name] = {
                "status": "UNSUPPORTED_CLASS_HAS_FEWER_THAN_TWO_ANCHORS",
                "class_counts": dict(sorted(hard_counts.items())),
            }
            continue
        prior = _oof_prior_baseline(
            targets=targets,
            groups=groups,
            classes=classes,
        )
        views = {}
        for view_name in FEATURE_VIEW_NAMES:
            report = cross_fit_categorical_soft_head(
                features=feature_views[view_name],
                targets=targets,
                groups=groups,
                class_ids=classes,
                base_weights=weights,
                projection_dimensions=(2, 4, 8),
                alphas=(10.0, 100.0),
                folds=5,
            )
            views[view_name] = {
                **report,
                "log_loss_improvement_over_oof_prior": (
                    prior["soft_log_loss"] - report["soft_log_loss"]
                ),
                "accuracy_improvement_over_oof_prior": (
                    report["accuracy"] - prior["accuracy"]
                ),
            }
        best_view = min(
            views,
            key=lambda name: (
                views[name]["soft_log_loss"],
                FEATURE_VIEW_NAMES.index(name),
            ),
        )
        best_report = views[best_view]
        represented_class_recall_count = sum(
            recall is not None and float(recall) > 0.0
            for recall in best_report["per_class_recall"].values()
        )
        signal_checks = {
            "positive_oof_log_loss_improvement": (
                best_report["log_loss_improvement_over_oof_prior"] > 0.0
            ),
            "positive_oof_accuracy_improvement": (
                best_report["accuracy_improvement_over_oof_prior"] > 0.0
            ),
            "represented_class_recall_count": (
                represented_class_recall_count
            ),
            "class_count": len(classes),
            "all_classes_have_nonzero_oof_recall": (
                represented_class_recall_count == len(classes)
            ),
        }
        head_reports[head_name] = {
            "status": "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD",
            "class_counts": dict(sorted(hard_counts.items())),
            "oof_prior": prior,
            "feature_views": views,
            "best_view_by_soft_log_loss": best_view,
            "best_log_loss_improvement_over_oof_prior": views[best_view][
                "log_loss_improvement_over_oof_prior"
            ],
            "signal_checks": signal_checks,
        }

    primary_heads = (
        "support_mode",
        "interaction_posture",
        "dialogue_phase",
        "nonclinical_urgency",
    )
    primary_improved = [
        name
        for name in primary_heads
        if head_reports[name]["status"] == "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD"
        and head_reports[name]["signal_checks"][
            "positive_oof_log_loss_improvement"
        ]
    ]
    flat_mode_ready = (
        head_reports["support_mode"]["status"]
        == "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD"
        and head_reports["support_mode"]["signal_checks"][
            "positive_oof_log_loss_improvement"
        ]
        and head_reports["support_mode"]["signal_checks"][
            "positive_oof_accuracy_improvement"
        ]
        and head_reports["support_mode"]["signal_checks"][
            "all_classes_have_nonzero_oof_recall"
        ]
    )
    status = (
        "PROVISIONAL_FLAT_MODE_LEARNABILITY_MORE_ANCHORS_REQUIRED"
        if flat_mode_ready
        else "INSUFFICIENT_EVIDENCE_FOR_FORMAL_FIVE_WAY_MODE_FIT"
    )
    report_core = {
        "protocol": PROTOCOL,
        "status": status,
        "non_abstaining_anchor_count": len(ordered_ids),
        "independent_dialogue_groups": len(set(groups)),
        "feature_view_names": list(FEATURE_VIEW_NAMES),
        "encoder_spec_sha256": encoder.binding.spec_sha256,
        "encoder_snapshot_tree_sha256": encoder.binding.snapshot_tree_sha256,
        "heads": head_reports,
        "primary_heads_with_positive_oof_log_loss_improvement": (
            primary_improved
        ),
        "flat_five_way_support_mode_ready_for_formal_fit": flat_mode_ready,
        "hierarchical_interaction_posture_is_diagnostic_only": True,
        "interpretation": (
            "diagnostic_only_not_formal_fit; positive held-out signal does not "
            "authorize training, and a negative result requires more or revised "
            "anchors before scaling weak supervision"
        ),
        "esconv_metadata_used_as_features": False,
        "target_supporter_responses_used": False,
        "target_strategy_annotations_used": False,
        "survey_outcomes_used": False,
        "automatic_gold_labels_created": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
