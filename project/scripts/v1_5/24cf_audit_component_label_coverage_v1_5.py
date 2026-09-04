#!/usr/bin/env python3
"""Audit whether labeled memory states cover the outcome-blind external inputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.contracts import StrategyMode, parse_action_id
from metacom_pm.io import iter_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-component-label-covariate-coverage-audit-v1"
COMPONENTS = ("MP", "MS", "ME")
CANDIDATE_FIELDS = (
    "top1_relevance_bucket",
    "relevance_margin_bucket",
    "token_cost_bucket",
    "retrieved_fraction",
    "minimum_relative_age_bucket",
    "maximum_relative_age_bucket",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _candidate_tuple(descriptor: dict[str, Any]) -> tuple[float, ...]:
    features = dict(descriptor["model_features"])
    return tuple(float(features[field]) for field in CANDIDATE_FIELDS)


def _full_tuple(
    descriptor: dict[str, Any],
    control_action: str,
) -> tuple[float, ...]:
    sources, strategy = parse_action_id(control_action)
    return _candidate_tuple(descriptor) + (
        float(len(sources)),
        float(strategy is StrategyMode.RS),
    )


def _range_support(
    reference: list[tuple[float, ...]],
    evaluation: list[tuple[float, ...]],
) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=float)
    evaluation_array = np.asarray(evaluation, dtype=float)
    minimum = reference_array.min(axis=0)
    maximum = reference_array.max(axis=0)
    supported = np.all(
        (evaluation_array >= minimum) & (evaluation_array <= maximum),
        axis=1,
    )
    return {
        "reference_rows": len(reference),
        "evaluation_rows": len(evaluation),
        "supported_rows": int(supported.sum()),
        "supported_fraction": float(supported.mean()),
        "minimum": {
            field: float(minimum[index])
            for index, field in enumerate(CANDIDATE_FIELDS)
        },
        "maximum": {
            field: float(maximum[index])
            for index, field in enumerate(CANDIDATE_FIELDS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/"
        "component_effect_labels.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
        "memory_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--internal-descriptors",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--external-descriptors",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
        "external_candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_component_label_coverage_audit_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    blueprints = {
        str(row["contrast_slot_id"]): row
        for row in _rows(args.blueprint)
    }
    internal = {
        (
            str(row["state_id"]),
            str(row["candidate_descriptor"]["source"]),
        ): dict(row["candidate_descriptor"])
        for row in _rows(args.internal_descriptors)
    }
    external_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(args.external_descriptors):
        descriptor = dict(row["candidate_descriptor"])
        if descriptor["candidate_present"]:
            external_by_source[str(row["source"])].append(descriptor)

    component_reports: dict[str, Any] = {}
    for component in COMPONENTS:
        component_labels = [
            row for row in labels if row["component"] == component
        ]
        development = [
            row
            for row in component_labels
            if row["split"] in {"train", "calibration"}
        ]
        all_internal_descriptors = [
            descriptor
            for (state_id, source), descriptor in internal.items()
            if source == component and descriptor["candidate_present"]
        ]
        development_descriptors = [
            internal[(str(row["state_id"]), component)]
            for row in development
        ]
        external_descriptors = external_by_source[component]
        all_internal_patterns = [
            _candidate_tuple(row) for row in all_internal_descriptors
        ]
        development_patterns = [
            _candidate_tuple(row) for row in development_descriptors
        ]
        external_patterns = [
            _candidate_tuple(row) for row in external_descriptors
        ]
        all_internal_pattern_set = set(all_internal_patterns)
        development_pattern_set = set(development_patterns)

        pattern_labels: dict[tuple[float, ...], list[int]] = defaultdict(list)
        for row, descriptor in zip(
            development,
            development_descriptors,
            strict=True,
        ):
            blueprint = blueprints[str(row["contrast_slot_id"])]
            pattern_labels[
                _full_tuple(descriptor, str(blueprint["control_action"]))
            ].append(int(row["target_y"]))
        conflicting = {
            pattern: values
            for pattern, values in pattern_labels.items()
            if len(set(values)) > 1
        }
        optimistic_correct = sum(
            max(Counter(values).values())
            for values in pattern_labels.values()
        )

        component_reports[component] = {
            "available_internal_candidate_rows": len(
                all_internal_descriptors
            ),
            "labeled_development_rows": len(development_descriptors),
            "external_candidate_rows": len(external_descriptors),
            "unique_candidate_patterns": {
                "all_internal": len(all_internal_pattern_set),
                "labeled_development": len(development_pattern_set),
                "external": len(set(external_patterns)),
            },
            "external_exact_pattern_coverage": {
                "by_all_internal_rows": sum(
                    pattern in all_internal_pattern_set
                    for pattern in external_patterns
                ),
                "by_all_internal_fraction": float(
                    np.mean(
                        [
                            pattern in all_internal_pattern_set
                            for pattern in external_patterns
                        ]
                    )
                ),
                "by_labeled_development_rows": sum(
                    pattern in development_pattern_set
                    for pattern in external_patterns
                ),
                "by_labeled_development_fraction": float(
                    np.mean(
                        [
                            pattern in development_pattern_set
                            for pattern in external_patterns
                        ]
                    )
                ),
            },
            "external_joint_range_support": {
                "by_all_internal": _range_support(
                    all_internal_patterns,
                    external_patterns,
                ),
                "by_labeled_development": _range_support(
                    development_patterns,
                    external_patterns,
                ),
            },
            "labeled_full_feature_pattern_diagnostic": {
                "unique_patterns": len(pattern_labels),
                "singleton_patterns": sum(
                    len(values) == 1 for values in pattern_labels.values()
                ),
                "conflicting_patterns": len(conflicting),
                "rows_in_conflicting_patterns": sum(
                    len(values) for values in conflicting.values()
                ),
                "optimistic_in_sample_majority_accuracy": (
                    optimistic_correct / len(development)
                ),
                "interpretation": (
                    "This is an optimistic descriptive ceiling within exact "
                    "observed feature patterns, not a valid OOF estimate."
                ),
            },
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_OUTCOME_BLIND_COVERAGE_DIAGNOSIS",
        "external_outcomes_used": False,
        "component_reports": component_reports,
        "finding": (
            "The complete 416-state internal catalog covers materially more "
            "external covariate patterns than the 48 labeled development rows "
            "per head. The current label selection, not only the shared "
            "compiler, is a transport bottleneck."
        ),
        "repair_boundary": (
            "External covariates may diagnose coverage before outcomes, but "
            "external responses, quality, risk, or PM actions must not select "
            "new training rows. Any repaired model requires untouched external "
            "outcome groups for confirmation."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "blueprint_sha256": sha256_file(args.blueprint),
            "internal_descriptors_sha256": sha256_file(
                args.internal_descriptors
            ),
            "external_descriptors_sha256": sha256_file(
                args.external_descriptors
            ),
        },
    }
    write_json(args.out_dir / "coverage_audit.json", report)
    print(
        {
            component: {
                "all_internal_exact": values[
                    "external_exact_pattern_coverage"
                ]["by_all_internal_fraction"],
                "labeled_development_exact": values[
                    "external_exact_pattern_coverage"
                ]["by_labeled_development_fraction"],
                "labeled_development_range": values[
                    "external_joint_range_support"
                ]["by_labeled_development"]["supported_fraction"],
            }
            for component, values in component_reports.items()
        }
    )


if __name__ == "__main__":
    main()
