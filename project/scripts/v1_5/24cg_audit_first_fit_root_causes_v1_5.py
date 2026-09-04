#!/usr/bin/env python3
"""Audit why the first formal four-component PM fit did not learn."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.contracts import StrategyMode, parse_action_id
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-first-four-component-fit-root-cause-audit-v1"
COMPONENTS = ("RS", "MP", "MS", "ME")
MEMORY_COMPONENTS = ("MP", "MS", "ME")
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
N_SPLITS = 5
THRESHOLD = 0.5
C_VALUE = 0.03


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    result = np.asarray(
        [1.0 / counts[value] for value in groups],
        dtype=float,
    )
    return result * (len(result) / result.sum())


def _model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=C_VALUE,
                    penalty="l2",
                    solver="liblinear",
                    class_weight=None,
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prior: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    weights = _weights(groups)
    action = probability >= THRESHOLD
    brier = float(
        brier_score_loss(y, probability, sample_weight=weights)
    )
    prior_brier = float(
        brier_score_loss(y, prior, sample_weight=weights)
    )
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                action,
                sample_weight=weights,
            )
        ),
        "positive_recall": float(
            recall_score(
                y,
                action,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "brier": brier,
        "prior_brier": prior_brier,
        "brier_gain_vs_prior": prior_brier - brier,
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
    }


def _grouped_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    for fold, (train, test) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("user leakage across root-cause OOF fold")
        model = _model(seed + fold)
        model.fit(
            x[train],
            y[train],
            logistic__sample_weight=_weights(groups[train]),
        )
        probability[test] = model.predict_proba(x[test])[:, 1]
        prior[test] = float(
            np.average(y[train], weights=_weights(groups[train]))
        )
    return {
        "probability": probability,
        "prior": prior,
        "metrics": _metrics(y, probability, prior, groups),
    }


def _explicit_background_probe(
    component: str,
    labels: list[dict[str, Any]],
    blueprints: dict[str, dict[str, Any]],
    descriptors: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in labels if row["component"] == component]
    other_sources = [
        value for value in MEMORY_COMPONENTS if value != component
    ]
    values: list[list[float]] = []
    for row in rows:
        descriptor = descriptors[(str(row["state_id"]), component)]
        sources, strategy = parse_action_id(
            str(blueprints[str(row["contrast_slot_id"])]["control_action"])
        )
        values.append(
            [
                float(descriptor["candidate_state_bge_similarity_optional"]),
                float(descriptor["source_catalog_bge_similarity_optional"]),
                float(any(x.value == other_sources[0] for x in sources)),
                float(any(x.value == other_sources[1] for x in sources)),
                float(strategy is StrategyMode.RS),
            ]
        )
    x = np.asarray(values, dtype=float)
    y = np.asarray([int(row["target_y"]) for row in rows], dtype=int)
    groups = np.asarray([str(row["user_id"]) for row in rows], dtype=object)
    splits = np.asarray([str(row["split"]) for row in rows], dtype=object)
    development = np.flatnonzero(
        np.isin(splits, ["train", "calibration"])
    )
    runs = [
        _grouped_oof(
            x[development],
            y[development],
            groups[development],
            seed,
        )
        for seed in SEEDS
    ]
    probability = np.mean(
        np.stack([run["probability"] for run in runs]),
        axis=0,
    )
    prior = np.mean(
        np.stack([run["prior"] for run in runs]),
        axis=0,
    )
    metrics = _metrics(
        y[development],
        probability,
        prior,
        groups[development],
    )
    seed_ba = [
        run["metrics"]["balanced_accuracy"] for run in runs
    ]
    return {
        "role": (
            "posthoc root-cause probe only; cannot promote a model or tune "
            "the existing internal/external evaluation"
        ),
        "features": [
            "candidate_state_bge_similarity",
            "source_catalog_bge_similarity",
            f"background_{other_sources[0]}_on",
            f"background_{other_sources[1]}_on",
            "background_RS_on",
        ],
        "feature_count": 5,
        "metrics": metrics,
        "balanced_accuracy_by_seed": seed_ba,
        "balanced_accuracy_seed_std": float(np.std(seed_ba, ddof=1)),
    }


def _background_rates(
    component: str,
    labels: list[dict[str, Any]],
    blueprints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, list[int]] = defaultdict(list)
    for row in labels:
        if (
            row["component"] == component
            and row["split"] in {"train", "calibration"}
        ):
            action = str(
                blueprints[str(row["contrast_slot_id"])]["control_action"]
            )
            result[action].append(int(row["target_y"]))
    return {
        action: {
            "rows": len(values),
            "on": sum(values),
            "on_fraction": sum(values) / len(values),
        }
        for action, values in sorted(result.items())
    }


def _normalized_mp_template(text: str) -> str:
    return re.sub(
        r"about .*?, with one gentle question at a time",
        "about <TOPIC>, with one gentle question at a time",
        text,
    )


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
        "--memory-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
        "memory_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--rs-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1/"
        "rs_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--descriptors",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--memory-backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "memory_backend.jsonl",
    )
    parser.add_argument(
        "--primary-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/"
        "training_report.json",
    )
    parser.add_argument(
        "--bge-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_bge_challenger_v1/"
        "bge_challenger_report.json",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_quality_aggregation_v1/"
        "quality_aggregation.json",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_component_label_coverage_audit_v1/"
        "coverage_audit.json",
    )
    parser.add_argument(
        "--transport-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
        "equivalence_audit.json",
    )
    parser.add_argument(
        "--ms-construct-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_ms_external_subdomain_audit_v1/"
        "ms_external_subdomain_audit.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_first_fit_root_cause_audit_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    blueprints = {
        str(row["contrast_slot_id"]): row
        for row in (
            _rows(args.memory_blueprint) + _rows(args.rs_blueprint)
        )
    }
    descriptors = {
        (
            str(row["state_id"]),
            str(row["candidate_descriptor"]["source"]),
        ): dict(row["candidate_descriptor"])
        for row in _rows(args.descriptors)
    }
    states = _rows(args.states)
    memory_backend = _rows(args.memory_backend)
    primary = read_json(args.primary_report)
    bge = read_json(args.bge_report)
    quality = read_json(args.quality_report)
    coverage = read_json(args.coverage_report)
    transport = read_json(args.transport_report)
    ms_construct = read_json(args.ms_construct_report)

    actual_feature_counts = {
        component: len(
            primary["head_reports"][component]["feature_names"]
        )
        for component in COMPONENTS
    }
    capacity: dict[str, Any] = {}
    for component in COMPONENTS:
        development = [
            row
            for row in labels
            if row["component"] == component
            and row["split"] in {"train", "calibration"}
        ]
        users = {str(row["user_id"]) for row in development}
        cap = math.floor(len(users) / 5)
        capacity[component] = {
            "development_rows": len(development),
            "independent_user_groups": len(users),
            "plan_feature_cap_floor_groups_div_5": cap,
            "actual_primary_feature_count": actual_feature_counts[component],
            "capacity_contract_passed": actual_feature_counts[component] <= cap,
            "minimum_48_independent_groups_passed": len(users) >= 48,
        }

    state_pool = {
        split: {
            "rows": sum(row["split"] == split for row in states),
            "users": len(
                {
                    str(row["user_id"])
                    for row in states
                    if row["split"] == split
                }
            ),
        }
        for split in ("train", "calibration", "internal_test")
    }
    development_pool_users = len(
        {
            str(row["user_id"])
            for row in states
            if row["split"] in {"train", "calibration"}
        }
    )

    variance: dict[str, Any] = {}
    for component in MEMORY_COMPONENTS:
        component_labels = [
            row
            for row in labels
            if row["component"] == component
            and row["split"] in {"train", "calibration"}
        ]
        names = list(primary["head_reports"][component]["feature_names"])
        arrays: list[list[float]] = []
        for row in component_labels:
            descriptor = descriptors[(str(row["state_id"]), component)]
            features = dict(descriptor["model_features"])
            sources, strategy = parse_action_id(
                str(
                    blueprints[str(row["contrast_slot_id"])][
                        "control_action"
                    ]
                )
            )
            values = {
                "top1_relevance_bucket": float(
                    features["top1_relevance_bucket"]
                ),
                "relevance_margin_bucket": float(
                    features["relevance_margin_bucket"]
                ),
                "token_cost_bucket": float(
                    features["token_cost_bucket"]
                ),
                "retrieved_fraction": float(
                    features["retrieved_fraction"]
                ),
                "minimum_relative_age_bucket": float(
                    features["minimum_relative_age_bucket"]
                ),
                "maximum_relative_age_bucket": float(
                    features["maximum_relative_age_bucket"]
                ),
                "background_other_memory_count": float(len(sources)),
                "background_strategy_on": float(
                    strategy is StrategyMode.RS
                ),
            }
            arrays.append([values[name] for name in names])
        x = np.asarray(arrays, dtype=float)
        variance[component] = {
            "zero_variance_features": [
                name
                for index, name in enumerate(names)
                if float(np.ptp(x[:, index])) == 0.0
            ],
            "nonzero_variance_feature_count": int(
                sum(float(np.ptp(x[:, index])) > 0.0 for index in range(x.shape[1]))
            ),
        }

    explicit_background_probes = {
        component: _explicit_background_probe(
            component,
            labels,
            blueprints,
            descriptors,
        )
        for component in MEMORY_COMPONENTS
    }

    mp_items = [
        item
        for row in memory_backend
        for item in row["items"]
        if item["source"] == "MP"
    ]
    mp_texts = [str(item["text"]) for item in mp_items]
    mp_profile = {
        "historical_mp_ontology": (
            "stable profile, preference, or boundary memory"
        ),
        "state_item_records": len(mp_items),
        "unique_memory_ids": len(
            {str(item["memory_id"]) for item in mp_items}
        ),
        "unique_exact_texts": len(set(mp_texts)),
        "normalized_template_count": len(
            {_normalized_mp_template(text) for text in mp_texts}
        ),
        "acknowledgement_before_suggestions_rate": (
            sum(
                "prefers acknowledgement before suggestions" in text
                for text in mp_texts
            )
            / len(mp_texts)
        ),
        "one_gentle_question_rate": (
            sum(
                "with one gentle question at a time" in text
                for text in mp_texts
            )
            / len(mp_texts)
        ),
        "internal_profile_fields": transport["content_profiles"][
            "repaired_internal"
        ]["profile_field_names"],
        "external_profile_fields": transport["content_profiles"][
            "evoemo_external"
        ]["profile_field_names"],
        "field_name_intersection": sorted(
            set(
                transport["content_profiles"]["repaired_internal"][
                    "profile_field_names"
                ]
            )
            & set(
                transport["content_profiles"]["evoemo_external"][
                    "profile_field_names"
                ]
            )
        ),
        "corrected_interpretation": (
            "The two domains cover different valid MP subtypes under the "
            "project's broader historical MP ontology. The defect is subtype "
            "coverage and severe internal preference templating, not evidence "
            "that preference memory should be deleted or replaced by profile "
            "facts."
        ),
    }

    background_rates = {
        component: _background_rates(component, labels, blueprints)
        for component in COMPONENTS
    }

    report = {
        "protocol": PROTOCOL,
        "status": "ROOT_CAUSES_SEPARATED_REPAIR_ORDER_IDENTIFIED",
        "formal_model_promotion_allowed": False,
        "external_outcomes_used_for_new_model_selection": False,
        "technical_conclusion": (
            "The first-fit failure cannot be attributed to BAAI or to "
            "intrinsically unlearnable labels yet. The implemented primary "
            "model violated its own low-capacity and state×candidate×background "
            "feature contract; the labeled subset failed external covariate "
            "coverage; and the old MS treatment did not implement a qualified "
            "prior-session-summary construct. Label reproducibility remains "
            "unmeasured."
        ),
        "data_and_grain": {
            "labels": len(labels),
            "contrasts_per_component": dict(
                Counter(str(row["component"]) for row in labels)
            ),
            "state_pool": state_pool,
            "available_development_users": development_pool_users,
            "selected_development_capacity": capacity,
        },
        "confirmed_method_failures": {
            "capacity_contract": {
                "plan_rule": (
                    "primary feature count <= floor(effective independent "
                    "groups / 5)"
                ),
                "by_component": capacity,
            },
            "memory_background_identity_collapsed": {
                "actual": (
                    "MP/MS/ME primary heads use only "
                    "background_other_memory_count plus strategy on/off"
                ),
                "required": (
                    "other-component background bits because the estimand is "
                    "conditional on the fixed background action"
                ),
                "observed_background_effect_rates": background_rates,
                "posthoc_explicit_background_probe": (
                    explicit_background_probes
                ),
                "probe_result": (
                    "Restoring explicit background identity raises ME from "
                    "BGE BA 0.482 to about 0.561 but does not reach the gate; "
                    "the representation bug is real but not the only cause."
                ),
            },
            "state_view_incomplete": {
                "plan_required_common_block": [
                    "turn/text length",
                    "explicit request/boundary",
                    "question/task",
                    "one-step/low-burden",
                    "history length",
                ],
                "actual_memory_primary": (
                    "candidate retrieval/age/cost summaries and collapsed "
                    "background only"
                ),
                "actual_rs_primary": (
                    "two coarse request/burden flags, candidate family/score, "
                    "and explicit memory-background bits"
                ),
                "missing_candidate_safety_descriptors": [
                    "current context already covers the memory",
                    "conflict flag",
                    "turn-specific request versus stable preference",
                    "profile stability",
                ],
            },
            "declared_but_zero_variance_features": variance,
        },
        "confirmed_data_design_failures": {
            "effective_groups_below_plan_minimum": {
                "plan_minimum": 48,
                "selected_independent_users_by_component": {
                    component: values["independent_user_groups"]
                    for component, values in capacity.items()
                },
                "maximum_available_train_calibration_users": (
                    development_pool_users
                ),
                "interpretation": (
                    "The current 36-user development pool cannot satisfy a "
                    "48-independent-group formal claim without new users or a "
                    "pre-outcome protocol amendment."
                ),
            },
            "blueprint_selection_axis_too_narrow": {
                "actual_axis": "component_top1_lexical_score quantiles only",
                "external_coverage_report": coverage["component_reports"],
                "interpretation": (
                    "The shared compiler passed on all 416 states, but the "
                    "48 labeled development rows per head did not preserve "
                    "that coverage."
                ),
            },
            "mp_construct_mismatch": mp_profile,
            "ms_construct_and_current_session_mismatch": {
                "ontology": ms_construct["historical_ontology"],
                "implementation_findings": ms_construct[
                    "implementation_findings"
                ],
                "severity_decision": ms_construct["severity_decision"],
                "interpretation": (
                    "The old 64 MS labels are diagnostic only. V1.5 must use "
                    "MS_SESSION with supplied strictly-prior summaries and one "
                    "shared complete-current-session visibility policy before "
                    "D2 or retraining."
                ),
            },
        },
        "label_quality_status": {
            "supported": {
                "schema_complete": all(quality["checks"].values()),
                "within_annotator_reversed_position_agreement": quality[
                    "reliability"
                ]["semantic_direction_agreement"],
                "blind_position_profile": quality[
                    "blind_position_profile"
                ],
                "risk_wins_reviewed": 109,
            },
            "not_measured": {
                "independent_second_annotator": True,
                "same_state_independent_generation_replicates": True,
                "expected_component_benefit_probability": True,
            },
            "protocol_gap": (
                "The final plan required second review of all candidate "
                "positives and a stratified subset of nonpositives; the "
                "completed 256-pair quality packet used one annotator plus "
                "within-annotator position reversals."
            ),
            "interpretation": (
                "The labels are usable as single-human realized-pair proxies, "
                "but data noise versus generator sampling cannot be separated "
                "without a bounded replication/inter-rater audit."
            ),
        },
        "model_family_evidence": {
            "primary_status": primary["status"],
            "primary_development_ba": {
                component: primary["head_reports"][component][
                    "grouped_oof"
                ]["metrics"]["group_weighted_balanced_accuracy"]
                for component in COMPONENTS
            },
            "bge_status": bge["status"],
            "bge_development_ba": {
                component: bge["head_reports"][component][
                    "grouped_oof_metrics"
                ]["group_weighted_balanced_accuracy"]
                for component in COMPONENTS
            },
            "interpretation": (
                "BGE-small does not rescue the current design. This rejects "
                "'embedding absence is the main bottleneck'; it does not prove "
                "that every richer semantic model would fail."
            ),
        },
        "root_cause_assessment": [
            {
                "rank": 1,
                "cause": (
                    "primary implementation does not match the frozen "
                    "low-capacity state×candidate×background contract"
                ),
                "severity": "critical",
                "confidence": "high",
                "type": "method implementation",
            },
            {
                "rank": 2,
                "cause": (
                    "labeled subset and available independent-user count do "
                    "not support the intended formal transport/complexity claim"
                ),
                "severity": "critical",
                "confidence": "high",
                "type": "data design and sampling",
            },
            {
                "rank": 3,
                "cause": (
                    "MP subtype coverage is disjoint: internal data contains "
                    "highly templated support preferences while EvoEmo contains "
                    "demographic/life profile facts"
                ),
                "severity": "high",
                "confidence": "high",
                "type": "construct subtype coverage",
            },
            {
                "rank": 4,
                "cause": (
                    "MS treatment and state transport are construct-confounded: "
                    "59/64 old contrasts inject utterance fallback and internal "
                    "versus EvoEmo current-session visibility differs"
                ),
                "severity": "high",
                "confidence": "high",
                "type": "MS construct and state transport",
            },
            {
                "rank": 5,
                "cause": (
                    "single-generator-draw and single-annotator labels may be "
                    "too noisy for individual state-level gating"
                ),
                "severity": "high if confirmed",
                "confidence": "unresolved",
                "type": "measurement",
            },
            {
                "rank": 6,
                "cause": "BAAI representation is too weak",
                "severity": "not established as primary",
                "confidence": "low",
                "type": "model family",
            },
        ],
        "repair_order": [
            {
                "step": "D0",
                "action": (
                    "Freeze the current fit as invalid for promotion but valid "
                    "as a root-cause baseline; do not run paid external quality."
                ),
                "new_human_reviews": 0,
                "new_api_calls": 0,
            },
            {
                "step": "D1",
                "action": (
                    "Write a new outcome-blind feature/blueprint contract with "
                    "at most five or six features, explicit background bits, "
                    "typed MP preference/profile candidates, and only "
                    "pre-injection state/candidate safety descriptors."
                ),
                "new_human_reviews": 0,
                "new_api_calls": 0,
            },
            {
                "step": "D1b",
                "action": (
                    "Freeze MS_SESSION as the only V1.5 MS subtype, forbid "
                    "last-message fallback, align complete current-session "
                    "visibility, downgrade all old MS pairs to diagnostic, and "
                    "build eight fresh outcome-blind MS states."
                ),
                "new_human_reviews": 0,
                "new_api_calls": 0,
            },
            {
                "step": "D2",
                "action": (
                    "Before rebuilding all labels, run a bounded 32-state "
                    "replication and independent-review audit: 24 eligible "
                    "RS/MP/ME states receive two new pairs each, and eight "
                    "fresh qualified MS states receive three pairs each."
                ),
                "human_review_plan": {
                    "primary_reviewer_new_pair_decisions": 72,
                    "second_reviewer_fixed_overlap_decisions": 32,
                    "total_decisions": 104,
                },
                "new_api_calls": 144,
                "gate": (
                    "majority direction reproducibility >=0.70, shared-pair "
                    "inter-annotator agreement >=0.75, and uncertain <=0.10; "
                    "otherwise do not train hard single-pair labels"
                ),
            },
            {
                "step": "D3",
                "action": (
                    "Only if D2 passes, add content-disjoint users and select "
                    "coverage-aware contrasts before outcomes. Do not reuse "
                    "consumed internal/external groups as confirmation."
                ),
                "new_human_reviews": "conditional",
                "new_api_calls": "conditional",
            },
        ],
        "claim_decision": {
            "research_goal_changed": False,
            "preserved_claim": (
                "A low-capacity auditable pre-generation PM attempts to learn "
                "component-specific expected marginal-benefit gates and then "
                "applies separate risk and deterministic cost constraints."
            ),
            "current_evidence": (
                "The claim is not yet supported, but the first failure is not "
                "a clean test of the claim because implementation, sampling, "
                "construct transport, and measurement gates were incomplete."
            ),
        },
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "memory_blueprint_sha256": sha256_file(args.memory_blueprint),
            "rs_blueprint_sha256": sha256_file(args.rs_blueprint),
            "descriptors_sha256": sha256_file(args.descriptors),
            "states_sha256": sha256_file(args.states),
            "memory_backend_sha256": sha256_file(args.memory_backend),
            "primary_report_sha256": sha256_file(args.primary_report),
            "bge_report_sha256": sha256_file(args.bge_report),
            "quality_report_sha256": sha256_file(args.quality_report),
            "coverage_report_sha256": sha256_file(args.coverage_report),
            "transport_report_sha256": sha256_file(args.transport_report),
            "ms_construct_report_sha256": sha256_file(
                args.ms_construct_report
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "root_cause_audit.json", report)
    print(
        {
            "status": report["status"],
            "root_causes": [
                (row["rank"], row["type"], row["confidence"])
                for row in report["root_cause_assessment"]
            ],
            "feature_capacity_pass": {
                component: values["capacity_contract_passed"]
                for component, values in capacity.items()
            },
        }
    )


if __name__ == "__main__":
    main()
