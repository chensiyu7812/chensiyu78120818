#!/usr/bin/env python3
"""Audit the existing 468-state corpus for candidate-admission reuse."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from metacom_pm.io import iter_jsonl, write_json
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-candidate-admission-data-audit-v1"
SPLITS = ("train", "calibration", "internal_test")
SYNTHETIC_EXPLICIT_NEGATIVE_MARKERS = (
    "news",
    "unrelated",
    "outside topic",
    "does not describe",
    "not a recurring personal pattern",
    "explicitly said it was unrelated",
    "general news stories",
)
MAX_METADATA_BALANCED_ACCURACY = 0.70


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _has_explicit_negative_marker(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered for marker in SYNTHETIC_EXPLICIT_NEGATIVE_MARKERS
    )


def _best_threshold(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = sorted(set(x.tolist()))
    thresholds = [values[0] - 1e-9, values[-1] + 1e-9]
    thresholds.extend(
        (left + right) / 2 for left, right in zip(values, values[1:])
    )
    candidates: list[tuple[float, str, float]] = []
    for threshold in thresholds:
        candidates.append(
            (
                float(balanced_accuracy_score(y, x >= threshold)),
                "ge",
                float(threshold),
            )
        )
        candidates.append(
            (
                float(balanced_accuracy_score(y, x <= threshold)),
                "le",
                float(threshold),
            )
        )
    score, direction, threshold = max(candidates)
    return {
        "balanced_accuracy": score,
        "direction": direction,
        "threshold": threshold,
        "positive_mean": float(x[y == 1].mean()),
        "negative_mean": float(x[y == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_candidate_admission_data_audit_v1"
        / "candidate_admission_data_audit.json",
    )
    args = parser.parse_args()

    states = {
        str(row["card_id"]): row
        for row in _rows(args.data_dir / "pm_v2_states.jsonl")
    }
    contexts = {
        str(row["card_id"]): row
        for row in _rows(args.data_dir / "evaluator_contexts.jsonl")
    }
    backends = {
        str(row["card_id"]): row
        for row in _rows(args.data_dir / "memory_backend.jsonl")
    }
    if not (set(states) == set(contexts) == set(backends)):
        raise RuntimeError("state/context/backend card identities differ")

    candidates: list[dict[str, Any]] = []
    for card_id, state in states.items():
        context = contexts[card_id]
        annotations = {
            str(row["memory_id"]): row
            for row in context["memory_annotations"]
        }
        needed = set(context["needed_memory_sources"])
        for item in backends[card_id]["items"]:
            annotation = annotations[str(item["memory_id"])]
            label = int(
                annotation["item_utility"] == "helpful"
                and item["source"] in needed
                and not bool(annotation["stale"])
                and not bool(
                    annotation["conflicts_with_current_state"]
                )
            )
            candidates.append(
                {
                    "split": state["split"],
                    "user_id": state["user_id"],
                    "source": item["source"],
                    "utility": annotation["item_utility"],
                    "label": label,
                    "item_tokens": estimate_tokens(str(item["text"])),
                    "item_age": (
                        int(state["session_index"])
                        - int(item["created_session"])
                    ),
                    "explicit_negative_marker": (
                        _has_explicit_negative_marker(str(item["text"]))
                    ),
                }
            )

    split_reports: dict[str, Any] = {}
    raw_marker_scores: dict[str, float] = {}
    filtered_metadata_scores: list[float] = []
    for split in SPLITS:
        rows = [row for row in candidates if row["split"] == split]
        y_raw = np.asarray([row["label"] for row in rows], dtype=int)
        marker_action = np.asarray(
            [not row["explicit_negative_marker"] for row in rows],
            dtype=bool,
        )
        marker_ba = float(balanced_accuracy_score(y_raw, marker_action))
        raw_marker_scores[split] = marker_ba

        filtered = [
            row for row in rows if not row["explicit_negative_marker"]
        ]
        y = np.asarray([row["label"] for row in filtered], dtype=int)
        metadata = {}
        for field in ("item_tokens", "item_age"):
            result = _best_threshold(
                np.asarray([row[field] for row in filtered], dtype=float), y
            )
            metadata[field] = result
            filtered_metadata_scores.append(result["balanced_accuracy"])
        split_reports[split] = {
            "raw_candidates": len(rows),
            "raw_positive": int(y_raw.sum()),
            "raw_utility_counts": dict(
                sorted(Counter(row["utility"] for row in rows).items())
            ),
            "raw_explicit_negative_marker_rows": int(
                sum(row["explicit_negative_marker"] for row in rows)
            ),
            "raw_marker_only_balanced_accuracy": marker_ba,
            "marker_neutral_candidates": len(filtered),
            "marker_neutral_positive": int(y.sum()),
            "marker_neutral_negative": int((y == 0).sum()),
            "marker_neutral_users": len(
                {str(row["user_id"]) for row in filtered}
            ),
            "marker_neutral_metadata_thresholds": metadata,
        }

    raw_eligible = (
        max(raw_marker_scores.values())
        < MAX_METADATA_BALANCED_ACCURACY
    )
    filtered_eligible = (
        max(filtered_metadata_scores)
        < MAX_METADATA_BALANCED_ACCURACY
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "RAW_POOL_INELIGIBLE_MARKER_NEUTRAL_DEVELOPMENT_POOL_REUSABLE"
            if not raw_eligible and filtered_eligible
            else (
                "RAW_POOL_ELIGIBLE"
                if raw_eligible and filtered_eligible
                else "CANDIDATE_POOL_REBUILD_REQUIRED"
            )
        ),
        "state_count": len(states),
        "candidate_count": len(candidates),
        "user_count": len(
            {str(row["user_id"]) for row in candidates}
        ),
        "maximum_allowed_metadata_balanced_accuracy": (
            MAX_METADATA_BALANCED_ACCURACY
        ),
        "raw_pool_eligible": raw_eligible,
        "marker_neutral_pool_eligible_for_development": filtered_eligible,
        "split_reports": split_reports,
        "decision": {
            "reuse_raw_pool": False,
            "reuse_marker_neutral_rows_as_development": (
                not raw_eligible and filtered_eligible
            ),
            "treat_explicit_negative_markers_as": (
                "deterministic synthetic hard exclusions, never learned "
                "evidence of generalization"
            ),
            "fresh_holdout_still_required": True,
            "internal_test_is_development_after_prior_consumption": True,
        },
        "interpretation": (
            "The raw candidate pool contains compiler-owned explicit-negative "
            "phrases that cross users and splits. A trivial absence-of-marker "
            "rule exceeds the frozen shortcut ceiling. Removing those rows "
            "leaves sufficient positive/nonpositive examples and item token/age "
            "statistics below the ceiling, so existing content can be reused "
            "for development only. It cannot replace a fresh holdout."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
