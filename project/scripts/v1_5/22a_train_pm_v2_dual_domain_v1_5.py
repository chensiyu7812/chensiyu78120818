#!/usr/bin/env python3
"""Train one PM-v1.5 router on longitudinal and ESConv auxiliary domains.

The two domains share one fitted model, feature contract, rule router and
selection configuration.  They do not share calibration comparators or an
internal-test consumption ledger.  Internal outcomes are opened only after the
joint candidate family and both opaque seals have been frozen.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.internal_holdout import (
    begin_internal_test_consumption,
    finish_internal_test_consumption,
    freeze_candidate_manifest,
    require_sealed_internal_label_bundle,
)
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v1_5_algorithm_selection import select_routing_algorithm_group_cv
from metacom_pm.pm_v1_5_rule_router import (
    transparent_rule_candidates,
    tune_transparent_rule_router,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v2_contracts import ActionLabel, CompositeSpec, PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_model import (
    PMV2Model,
    SelectionConfig,
    calibrate_uncertainty_multiplier,
    evaluate_policy,
    evaluate_prediction_coverage,
    tune_selection_config,
)
from metacom_pm.v1_5_dual_domain_training import (
    DUAL_DOMAIN_GATE_PROTOCOL,
    DUAL_DOMAIN_TRAINING_PROTOCOL,
    ESCONV_AUXILIARY_DOMAIN,
    LONGITUDINAL_DOMAIN,
    audit_domain_label_matrix,
    calibration_action_viability,
    domain_internal_gate,
    fixed_action_metrics,
    require_equal_domain_training_weight,
    training_domain_for_state,
    validate_dual_domain_training_inputs,
    validate_internal_domain_labels,
)


ROOT = Path(__file__).resolve().parents[2]


def _selection_from_config(config: dict[str, Any]) -> SelectionConfig:
    quality = config["quality_composite"]
    composite = CompositeSpec(
        version=str(quality["version"]),
        weights={str(key): float(value) for key, value in quality["weights"].items()},
    )
    return SelectionConfig(**dict(config["selection"]), composite_spec=composite)


def _load_labels(path: Path) -> list[ActionLabel]:
    return [ActionLabel.model_validate(row) for row in iter_jsonl(path)]


def _labels_for_states(labels, states):
    state_ids = {state.state_id for state in states}
    return [label for label in labels if label.state_id in state_ids]


def _domain_rows(states, labels, domain: str):
    selected_states = [
        state for state in states if training_domain_for_state(state) == domain
    ]
    return selected_states, _labels_for_states(labels, selected_states)


@contextmanager
def _cache_selection_inference(model: PMV2Model):
    """Cache selector-independent model inference during the frozen grid scan.

    The calibration candidates alter only risk/cost/gain thresholds.  They do
    not alter fitted heads, conformal radii, ``uncertainty_z`` or the state
    features.  Recomputing all bootstrap-head predictions for every candidate
    is therefore both wasteful and exactly redundant.
    """

    original_bundle = model._prediction_bundle
    original_routing = model._routing_scores
    bundle_cache: dict[str, Any] = {}
    routing_cache: dict[str, Any] = {}

    def cached_bundle(state):
        if state.state_id not in bundle_cache:
            bundle_cache[state.state_id] = original_bundle(state)
        return bundle_cache[state.state_id]

    def cached_routing(state):
        if state.state_id not in routing_cache:
            routing_cache[state.state_id] = original_routing(state)
        return routing_cache[state.state_id]

    model._prediction_bundle = cached_bundle
    model._routing_scores = cached_routing
    try:
        yield {
            "prediction_bundle_cache": bundle_cache,
            "routing_score_cache": routing_cache,
        }
    finally:
        del model._prediction_bundle
        del model._routing_scores


def _require_internal_state_commitment(path: Path, states, *, domain: str) -> dict:
    commitment = read_json(path)
    core = {
        key: value
        for key, value in commitment.items()
        if key != "commitment_sha256"
    }
    internal_states = sorted(
        (state for state in states if state.split is PMV2Split.INTERNAL_TEST),
        key=lambda state: state.state_id,
    )
    action_matrix = [
        {
            "state_id": state.state_id,
            "allowed_actions": sorted(state.allowed_actions),
        }
        for state in internal_states
    ]
    if (
        commitment.get("protocol")
        != "pm-v1.5-internal-state-universe-commitment-v1"
        or commitment.get("status")
        != "COMMITTED_WITHOUT_OUTCOMES_BEFORE_FIT"
        or commitment.get("domain") != domain
        or commitment.get("commitment_sha256")
        != sha256_text(canonical_json(core))
        or int(commitment.get("state_count") or -1) != len(internal_states)
        or int(commitment.get("expected_label_rows") or -1)
        != sum(len(state.allowed_actions) for state in internal_states)
        or commitment.get("state_universe_sha256")
        != sha256_text(
            canonical_json(sorted(state.state_id for state in internal_states))
        )
        or commitment.get("action_matrix_sha256")
        != sha256_text(canonical_json(action_matrix))
        or commitment.get("internal_label_values_deserialized") is not False
    ):
        raise RuntimeError(f"{domain} internal state commitment is invalid")
    return commitment


def _algorithm_selection_checkpoint_binding(
    *, args, config_path: Path
) -> dict[str, Any]:
    paths = {
        "pm_v1_5_config": config_path,
        "dual_preflight_report": args.dual_preflight_report,
        "longitudinal_states": args.longitudinal_states,
        "longitudinal_train_calibration_labels": (
            args.longitudinal_train_calibration_labels
        ),
        "auxiliary_train_labels": args.auxiliary_train_labels,
        "algorithm_selection_code": (
            ROOT / "src" / "metacom_pm" / "pm_v1_5_algorithm_selection.py"
        ),
        "pm_v2_model_code": ROOT / "src" / "metacom_pm" / "pm_v2_model.py",
        "training_script": Path(__file__),
    }
    return {
        "protocol": "pm-v1.5-train-only-algorithm-selection-checkpoint-v1",
        "run_identity": args.run_identity,
        "seed": int(args.seed),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }


def _require_algorithm_selection_checkpoint(
    path: Path, *, expected_binding: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    record = read_json(path)
    core = {key: value for key, value in record.items() if key != "binding_sha256"}
    if core.get("binding") != expected_binding:
        raise RuntimeError("algorithm-selection checkpoint input binding drifted")
    if record.get("binding_sha256") != sha256_text(canonical_json(core)):
        raise RuntimeError("algorithm-selection checkpoint self-hash mismatch")
    selected = str(core.get("selected_algorithm") or "")
    report = dict(core.get("algorithm_selection") or {})
    if (
        not selected
        or report.get("selected_algorithm") != selected
        or report.get("selection_data_role") != "train_only"
    ):
        raise RuntimeError("algorithm-selection checkpoint content is invalid")
    return selected, report


def _ood_report(model, states) -> dict[str, Any]:
    rows = [model.feature_builder.ood_report(state) for state in states]
    return {
        "n": len(rows),
        "severe_semantic_ood_rate": sum(
            bool(row["severe_semantic_ood"]) for row in rows
        )
        / len(rows),
        "severe_metadata_ood_rate": sum(
            bool(row["severe_metadata_ood"]) for row in rows
        )
        / len(rows),
        "maximum_semantic_ood_score": max(
            float(row["semantic_ood_score"]) for row in rows
        ),
        "maximum_metadata_ood_score": max(
            float(row["metadata_ood_score"]) for row in rows
        ),
    }


def _calibration_fixed_frontier(model, states, labels, actions):
    policy = evaluate_policy(model, states, labels)
    target_tokens = float(policy["mean_observed_input_tokens"])
    rows = []
    for action_id in actions:
        row = fixed_action_metrics(model, states, labels, action_id)
        if row is None:
            continue
        row["observed_token_relative_deviation"] = abs(
            float(row["mean_observed_input_tokens"]) / max(target_tokens, 1.0) - 1.0
        )
        rows.append(row)
    if not rows:
        raise RuntimeError("domain calibration has no legal fixed comparator")
    # Selection is bound to the MAD-adjusted conservative utility (lambda=
    # 1.0 fixed) -- the same shared helper and constant used by every other
    # comparator in this audit -- never the nominal utility.
    cost_matched = min(
        rows,
        key=lambda row: (
            row["observed_token_relative_deviation"],
            -row["mean_conservative_utility"],
            row["action_id"],
        ),
    )
    best = max(
        rows,
        key=lambda row: (
            row["mean_conservative_utility"],
            row["mean_conservative_quality"],
            -row["mean_conservative_risk"],
            row["action_id"],
        ),
    )
    return {
        "policy": {key: value for key, value in policy.items() if key != "rows"},
        "frontier": [
            {key: value for key, value in row.items() if key != "rows"}
            for row in rows
        ],
        "cost_matched_action": cost_matched["action_id"],
        "best_fixed_action": best["action_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--dual-preflight-report", type=Path, required=True)
    parser.add_argument("--longitudinal-states", type=Path, required=True)
    parser.add_argument("--longitudinal-train-calibration-labels", type=Path, required=True)
    parser.add_argument("--longitudinal-internal-test-labels", type=Path)
    parser.add_argument("--longitudinal-internal-seal", type=Path)
    parser.add_argument("--longitudinal-internal-commitment", type=Path)
    parser.add_argument("--auxiliary-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-train-labels", type=Path, required=True)
    parser.add_argument("--auxiliary-calibration-labels", type=Path, required=True)
    parser.add_argument("--auxiliary-internal-test-labels", type=Path)
    parser.add_argument("--auxiliary-internal-seal", type=Path)
    parser.add_argument("--auxiliary-internal-commitment", type=Path)
    parser.add_argument("--shortcut-audit-report", type=Path, required=True)
    parser.add_argument("--shortcut-audit-attestation", type=Path)
    parser.add_argument("--rule-grid-report", type=Path, required=True)
    parser.add_argument("--rule-grid-attestation", type=Path)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-nonreportable", action="store_true")
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "Fit/calibrate and freeze the candidate, then stop before opening "
            "or requiring any internal-test label artifact."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    if config.get("version") != "pm-v1.5":
        raise RuntimeError("dual-domain trainer requires pm-v1.5")
    preflight = read_json(args.dual_preflight_report)
    if (
        preflight.get("protocol") != DUAL_DOMAIN_TRAINING_PROTOCOL
        or preflight.get("status") != "PASS"
        or preflight.get("internal_label_values_deserialized") is not False
        or preflight.get("pm_v1_5_config_sha256") != sha256_file(args.pm_v1_5_config)
        or bool(preflight.get("fit_only")) != bool(args.fit_only)
    ):
        raise RuntimeError("dual-domain training preflight is missing or stale")
    measurement_contract = dict(
        preflight.get("frozen_esconv_auxiliary_measurement_contract") or {}
    )
    weak_supervision_mode = (
        measurement_contract.get("protocol") == "pm-v1.5-llm-weak-supervision-v1"
        and measurement_contract.get("status")
        == "COMPLETE_WEAK_SUPERVISION_NOT_GOLD"
        and measurement_contract.get("automatic_gold_label_claimed") is False
    )
    if args.fit_only and not weak_supervision_mode:
        raise RuntimeError(
            "fit-only training currently requires the content-addressed "
            "non-gold weak-supervision contract"
        )
    shortcut = read_json(args.shortcut_audit_report)
    rule_grid = read_json(args.rule_grid_report)
    expected_states_sha256 = sha256_file(args.longitudinal_states)
    expected_config_sha256 = sha256_file(args.pm_v1_5_config)
    if (
        shortcut.get("status") != "PASS"
        or (shortcut.get("input_hashes") or {}).get("states")
        != expected_states_sha256
        or (shortcut.get("input_hashes") or {}).get("pm_v1_5_config")
        != expected_config_sha256
    ):
        raise RuntimeError("shortcut audit is not the exact longitudinal PASS artifact")
    if (
        rule_grid.get("status") != "PASS"
        or rule_grid.get("states_sha256") != expected_states_sha256
        or rule_grid.get("pm_v1_5_config_sha256") != expected_config_sha256
    ):
        raise RuntimeError("rule-grid preflight is not the exact longitudinal PASS artifact")
    shortcut_attestation = (
        args.shortcut_audit_attestation
        or args.shortcut_audit_report.parent / "artifact_attestation.json"
    )
    rule_grid_attestation = (
        args.rule_grid_attestation
        or args.rule_grid_report.parent / "artifact_attestation.json"
    )
    require_artifact_attestation(
        shortcut_attestation,
        required_stage="pm_v1_5_step0_shortcut_audit",
        required_output_paths={"shortcut_audit_report": args.shortcut_audit_report},
    )
    require_artifact_attestation(
        rule_grid_attestation,
        required_stage="pm_v1_5_pre_training_rule_grid_diagnostic",
        required_output_paths={"rule_grid_report": args.rule_grid_report},
    )

    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    semantic_runtime = require_semantic_runtime_contract(config, encoder)
    del encoder

    longitudinal_states = load_states(args.longitudinal_states)
    auxiliary_state_paths = {
        split: args.auxiliary_dir / split / "pm_v2_states.jsonl"
        for split in ("train", "calibration", "internal_test")
    }
    auxiliary_states = [
        state
        for split in ("train", "calibration", "internal_test")
        for state in load_states(auxiliary_state_paths[split])
    ]
    if args.fit_only:
        forbidden_internal_args = (
            args.longitudinal_internal_test_labels,
            args.longitudinal_internal_seal,
            args.auxiliary_internal_test_labels,
            args.auxiliary_internal_seal,
        )
        if any(value is not None for value in forbidden_internal_args):
            raise RuntimeError(
                "fit-only training forbids internal label/seal path arguments"
            )
        if (
            args.longitudinal_internal_commitment is None
            or args.auxiliary_internal_commitment is None
        ):
            raise RuntimeError(
                "fit-only training requires both state-universe commitments"
            )
        long_seal = None
        aux_seal = None
        long_commitment = _require_internal_state_commitment(
            args.longitudinal_internal_commitment,
            longitudinal_states,
            domain=LONGITUDINAL_DOMAIN,
        )
        aux_commitment = _require_internal_state_commitment(
            args.auxiliary_internal_commitment,
            auxiliary_states,
            domain=ESCONV_AUXILIARY_DOMAIN,
        )
        preflight_commitments = dict(
            preflight.get("internal_state_commitments") or {}
        )
        if preflight_commitments.get(LONGITUDINAL_DOMAIN) != long_commitment:
            raise RuntimeError(
                "longitudinal internal commitment differs from dual preflight"
            )
        if preflight_commitments.get(ESCONV_AUXILIARY_DOMAIN) != aux_commitment:
            raise RuntimeError(
                "ESConv auxiliary internal commitment differs from dual preflight"
            )
    else:
        if any(
            value is None
            for value in (
                args.longitudinal_internal_test_labels,
                args.longitudinal_internal_seal,
                args.auxiliary_internal_test_labels,
                args.auxiliary_internal_seal,
            )
        ):
            raise RuntimeError(
                "legacy joint training/evaluation requires internal labels and seals"
            )
        long_commitment = None
        aux_commitment = None
        long_seal = require_sealed_internal_label_bundle(
            args.longitudinal_internal_seal,
            internal_labels_path=args.longitudinal_internal_test_labels,
        )
        aux_seal = require_sealed_internal_label_bundle(
            args.auxiliary_internal_seal,
            internal_labels_path=args.auxiliary_internal_test_labels,
        )
        preflight_seals = dict(preflight.get("sealed_internal_bundles") or {})
        if preflight_seals.get(LONGITUDINAL_DOMAIN) != long_seal:
            raise RuntimeError(
                "longitudinal internal seal differs from dual preflight"
            )
        if preflight_seals.get(ESCONV_AUXILIARY_DOMAIN) != aux_seal:
            raise RuntimeError(
                "ESConv auxiliary internal seal differs from dual preflight"
            )
    longitudinal_labels = _load_labels(args.longitudinal_train_calibration_labels)
    auxiliary_label_paths = {
        "train": args.auxiliary_train_labels,
        "calibration": args.auxiliary_calibration_labels,
    }
    auxiliary_labels = [
        label
        for split in ("train", "calibration")
        for label in _load_labels(auxiliary_label_paths[split])
    ]
    inputs = validate_dual_domain_training_inputs(
        longitudinal_states=longitudinal_states,
        auxiliary_states=auxiliary_states,
        longitudinal_train_calibration_labels=longitudinal_labels,
        auxiliary_train_calibration_labels=auxiliary_labels,
        pm_config=config,
    )
    states_by_split, labels_by_split = inputs.fit_and_calibration_views()
    reliable_cfg = config["reliable_action_matrix_gate"]
    domain_audits = {
        domain: {
            split.value: audit_domain_label_matrix(
                *_domain_rows(states_by_split[split], labels_by_split[split], domain),
                domain=domain,
                low_mad_threshold=float(config["labeling"]["reliable_mad_threshold"]),
                minimum_reliable_rate=float(config["data_label_gate"]["minimum_reliable_rate"]),
                minimum_low_mad_coverage_per_dimension=float(
                    reliable_cfg[split.value]["minimum_low_mad_coverage_per_dimension"]
                ),
                minimum_low_mad_coverage_per_action_dimension=float(
                    reliable_cfg[split.value]["minimum_low_mad_coverage_per_action_dimension"]
                ),
                enforce_low_mad_coverage_gate=not weak_supervision_mode,
            )
            for split in (PMV2Split.TRAIN, PMV2Split.CALIBRATION)
        }
        for domain in (LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN)
    }

    initial_selection = _selection_from_config(config)
    model_cfg, feature_cfg = config["model"], config["features"]
    labeling_cfg = config["labeling"]
    algorithm_cfg = config["algorithm_selection"]
    rule_cfg = config["transparent_rule_router"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    algorithm_checkpoint_path = (
        args.out_dir / "algorithm_selection_checkpoint.json"
    )
    algorithm_checkpoint_binding = _algorithm_selection_checkpoint_binding(
        args=args, config_path=args.pm_v1_5_config
    )
    if algorithm_checkpoint_path.exists():
        selected_algorithm, algorithm_selection = (
            _require_algorithm_selection_checkpoint(
                algorithm_checkpoint_path,
                expected_binding=algorithm_checkpoint_binding,
            )
        )
    else:
        selected_algorithm, algorithm_selection = select_routing_algorithm_group_cv(
            states=states_by_split[PMV2Split.TRAIN],
            labels=labels_by_split[PMV2Split.TRAIN],
            selection_config=initial_selection,
            candidates=algorithm_cfg["candidates"],
            folds=int(algorithm_cfg["folds"]),
            n_models=int(algorithm_cfg["cv_bootstrap_models"]),
            seed=args.seed,
            dimension_mad_scale=float(labeling_cfg["reliable_mad_threshold"]),
            bootstrap_group_key=str(model_cfg.get("group_bootstrap_key", "user_id")),
            use_precomputed_embeddings=bool(feature_cfg["optional_precomputed_semantic_embedding"]),
            require_precomputed_embeddings=bool(feature_cfg["require_precomputed_semantic_embedding"]),
            semantic_projection_dimensions=int(feature_cfg["semantic_projection_dimensions"]),
            word_features=int(feature_cfg["word_hash_features"]),
            char_features=int(feature_cfg["char_hash_features"]),
            rule_grid=rule_cfg["grid"],
            rule_minimum_quality=float(rule_cfg["train_minimum_quality"]),
            rule_maximum_risk=float(rule_cfg["train_maximum_risk"]),
            minimum_validation_quality=float(algorithm_cfg["minimum_validation_quality"]),
            maximum_validation_risk=float(algorithm_cfg["maximum_validation_risk"]),
            safe_residual_thresholds=algorithm_cfg["safe_residual_thresholds"],
            simplicity_order=algorithm_cfg["simplicity_order"],
            domain_key=training_domain_for_state,
            parallel_folds=2,
        )
        checkpoint_core = {
            "binding": algorithm_checkpoint_binding,
            "selected_algorithm": selected_algorithm,
            "algorithm_selection": algorithm_selection,
        }
        write_json(
            algorithm_checkpoint_path,
            {
                **checkpoint_core,
                "binding_sha256": sha256_text(
                    canonical_json(checkpoint_core)
                ),
            },
        )
    rule_router, rule_tuning = tune_transparent_rule_router(
        states=states_by_split[PMV2Split.TRAIN],
        labels=labels_by_split[PMV2Split.TRAIN],
        selection_config=initial_selection,
        candidates=transparent_rule_candidates(rule_cfg["grid"]),
        minimum_quality=float(rule_cfg["train_minimum_quality"]),
        maximum_risk=float(rule_cfg["train_maximum_risk"]),
        selection_data_role="train",
        domain_key=training_domain_for_state,
    )
    model = PMV2Model.train(
        states_by_split[PMV2Split.TRAIN],
        labels_by_split[PMV2Split.TRAIN],
        selection_config=initial_selection,
        n_models=int(model_cfg["bootstrap_models"]),
        seed=args.seed,
        dimension_mad_scale=float(labeling_cfg["reliable_mad_threshold"]),
        bootstrap_group_key=str(model_cfg.get("group_bootstrap_key", "user_id")),
        use_precomputed_embeddings=bool(feature_cfg["optional_precomputed_semantic_embedding"]),
        require_precomputed_embeddings=bool(feature_cfg["require_precomputed_semantic_embedding"]),
        semantic_projection_dimensions=int(feature_cfg["semantic_projection_dimensions"]),
        word_features=int(feature_cfg["word_hash_features"]),
        char_features=int(feature_cfg["char_hash_features"]),
        domain_key=training_domain_for_state,
    )
    model.fit_routing_objective(
        states_by_split[PMV2Split.TRAIN],
        labels_by_split[PMV2Split.TRAIN],
        algorithm=selected_algorithm,
        n_models=int(model_cfg["bootstrap_models"]),
        seed=args.seed,
        bootstrap_group_key=str(model_cfg.get("group_bootstrap_key", "user_id")),
        domain_key=training_domain_for_state,
        rule_router=(rule_router if selected_algorithm == "rule_relative_safe_residual_hgb" else None),
        safe_thresholds=(algorithm_cfg["safe_residual_thresholds"] if selected_algorithm == "rule_relative_safe_residual_hgb" else None),
    )
    equal_domain_weight = require_equal_domain_training_weight(model.training_report)

    ood_cfg = config["ood_calibration"]
    # Calibrate each domain independently, then keep the conservative envelope.
    # A pooled empirical quantile would let the 170-state auxiliary calibration
    # split silently dominate the 108-state longitudinal split.
    ood_by_domain = {}
    ood_thresholds = {}
    for domain in (LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN):
        domain_states, _ = _domain_rows(
            states_by_split[PMV2Split.CALIBRATION],
            labels_by_split[PMV2Split.CALIBRATION],
            domain,
        )
        ood_by_domain[domain] = model.feature_builder.calibrate_ood(
            domain_states,
            semantic_false_positive_quantile=float(ood_cfg["semantic_false_positive_quantile"]),
            metadata_false_positive_quantile=float(ood_cfg["metadata_false_positive_quantile"]),
            maximum_joint_in_distribution_fallback_rate=float(ood_cfg["maximum_joint_in_distribution_fallback_rate"]),
            minimum_semantic_challenge_detection_rate=float(ood_cfg["minimum_semantic_challenge_detection_rate"]),
            minimum_metadata_challenge_detection_rate=float(ood_cfg["minimum_metadata_challenge_detection_rate"]),
        )
        ood_thresholds[domain] = {
            "semantic": float(model.feature_builder.semantic_ood_threshold),
            "metadata": float(model.feature_builder.metadata_ood_threshold),
        }
    model.feature_builder.semantic_ood_threshold = max(
        row["semantic"] for row in ood_thresholds.values()
    )
    model.feature_builder.metadata_ood_threshold = max(
        row["metadata"] for row in ood_thresholds.values()
    )
    ood_calibration = {
        "protocol": "pm-v1.5-equal-domain-conservative-ood-envelope-v1",
        "status": "COMPLETE",
        "domain_weight": {
            LONGITUDINAL_DOMAIN: 0.5,
            ESCONV_AUXILIARY_DOMAIN: 0.5,
        },
        "per_domain_calibration": ood_by_domain,
        "per_domain_thresholds": ood_thresholds,
        "final_thresholds": {
            "semantic": float(model.feature_builder.semantic_ood_threshold),
            "metadata": float(model.feature_builder.metadata_ood_threshold),
        },
        "selection_rule": "elementwise_maximum_no_domain_can_tighten_another",
    }
    per_domain_ood = {
        domain: _ood_report(model, _domain_rows(states_by_split[PMV2Split.CALIBRATION], labels_by_split[PMV2Split.CALIBRATION], domain)[0])
        for domain in (LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN)
    }
    uncertainty_cfg = config["uncertainty_calibration"]
    uncertainty = calibrate_uncertainty_multiplier(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
        z_candidates=[float(value) for value in uncertainty_cfg["z_candidates"]],
        target_coverage=float(uncertainty_cfg["target_coverage"]),
        minimum_quality_coverage_lower_bound=float(uncertainty_cfg["minimum_quality_coverage_lower_bound"]),
        minimum_response_coverage_lower_bound=float(uncertainty_cfg["minimum_response_coverage_lower_bound"]),
        minimum_risk_coverage_lower_bound=float(uncertainty_cfg["minimum_risk_coverage_lower_bound"]),
        coverage_confidence_level=float(uncertainty_cfg["confidence_level"]),
    )
    per_domain_uncertainty = {}
    for domain in (LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN):
        domain_states, domain_labels = _domain_rows(
            states_by_split[PMV2Split.CALIBRATION],
            labels_by_split[PMV2Split.CALIBRATION],
            domain,
        )
        per_domain_uncertainty[domain] = evaluate_prediction_coverage(
            model,
            domain_states,
            domain_labels,
            confidence_level=float(uncertainty_cfg["confidence_level"]),
        )
    grid_cfg = config["calibration_grid"]
    with _cache_selection_inference(model) as inference_cache:
        tuning = tune_selection_config(
            model,
            states_by_split[PMV2Split.CALIBRATION],
            labels_by_split[PMV2Split.CALIBRATION],
            cost_weights=[float(value) for value in grid_cfg["cost_weights"]],
            risk_weights=[float(value) for value in grid_cfg["risk_weights"]],
            resource_gains=[float(value) for value in grid_cfg["resource_gains"]],
            strategy_gains=[float(value) for value in grid_cfg["strategy_gains"]],
            max_risks=[float(value) for value in grid_cfg["max_risks"]],
            minimum_quality=float(grid_cfg["minimum_quality"]),
            objective_risk_weight=float(grid_cfg["objective_risk_weight"]),
            objective_cost_weight=float(grid_cfg["objective_cost_weight"]),
            objective_version=str(grid_cfg["objective_version"]),
            domain_key=training_domain_for_state,
        )
        tuning["inference_cache"] = {
            "protocol": "selector-independent-state-inference-cache-v1",
            "prediction_bundle_state_count": len(
                inference_cache["prediction_bundle_cache"]
            ),
            "routing_score_state_count": len(
                inference_cache["routing_score_cache"]
            ),
            "candidate_scientific_contract_changed": False,
        }
    rule_router.selection_config = model.selection_config

    long_cal_states, long_cal_labels = _domain_rows(
        states_by_split[PMV2Split.CALIBRATION], labels_by_split[PMV2Split.CALIBRATION], LONGITUDINAL_DOMAIN
    )
    aux_cal_states, aux_cal_labels = _domain_rows(
        states_by_split[PMV2Split.CALIBRATION], labels_by_split[PMV2Split.CALIBRATION], ESCONV_AUXILIARY_DOMAIN
    )
    long_actions = sorted(set.intersection(*(set(state.allowed_actions) for state in long_cal_states)))
    long_frontier = _calibration_fixed_frontier(model, long_cal_states, long_cal_labels, long_actions)
    aux_frontier = _calibration_fixed_frontier(model, aux_cal_states, aux_cal_labels, ["M0+R0", "M0+RS"])
    action_preflight = config["external_evaluation"]["action_preflight"]
    pre_internal_action_viability = {
        LONGITUDINAL_DOMAIN: calibration_action_viability(
            long_frontier["policy"],
            action_preflight=action_preflight,
        ),
        ESCONV_AUXILIARY_DOMAIN: calibration_action_viability(
            aux_frontier["policy"],
            action_preflight=action_preflight,
        ),
    }
    calibration_candidate_supported = all(
        row["status"] == "PASS"
        for row in pre_internal_action_viability.values()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.fit_only and not calibration_candidate_supported:
        failure_report = {
            "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
            "status": "NOT_SUPPORTED_FOR_INTERNAL_TEST_CONSUMPTION",
            "fit_only": False,
            "internal_test_outcomes_opened": False,
            "run_identity": args.run_identity,
            "pre_internal_calibration_action_viability": (
                pre_internal_action_viability
            ),
            "stopping_rule": (
                "internal outcomes must not be deserialized or consumed when "
                "either domain fails the frozen calibration viability gate"
            ),
        }
        write_json(
            args.out_dir / "pre_internal_viability_failure.json",
            failure_report,
        )
        raise RuntimeError(
            "dual-domain candidate failed the pre-internal calibration "
            "action-viability gate; internal consumption is forbidden"
        )
    checkpoint = args.out_dir / "pm_v1_5_dual_domain.joblib"
    rule_checkpoint = args.out_dir / "pm_v1_5_dual_domain_transparent_rule.joblib"
    model.save(checkpoint)
    joblib.dump(rule_router, rule_checkpoint)
    candidate_path = args.out_dir / "candidate_manifest.json"
    candidate = freeze_candidate_manifest(
        candidate_path,
        run_identity=args.run_identity,
        artifacts={
            "pm_v1_5_config": args.pm_v1_5_config,
            "dual_preflight_report": args.dual_preflight_report,
            "longitudinal_states": args.longitudinal_states,
            "longitudinal_train_calibration_labels": args.longitudinal_train_calibration_labels,
            "auxiliary_train_states": auxiliary_state_paths["train"],
            "auxiliary_calibration_states": auxiliary_state_paths["calibration"],
            "auxiliary_internal_states": auxiliary_state_paths["internal_test"],
            "auxiliary_train_labels": args.auxiliary_train_labels,
            "auxiliary_calibration_labels": args.auxiliary_calibration_labels,
            "algorithm_selection_checkpoint": algorithm_checkpoint_path,
            "primary_checkpoint": checkpoint,
            "transparent_rule_checkpoint": rule_checkpoint,
            **(
                {
                    "internal_state_commitment_longitudinal": (
                        args.longitudinal_internal_commitment
                    ),
                    "internal_state_commitment_esconv_auxiliary": (
                        args.auxiliary_internal_commitment
                    ),
                }
                if args.fit_only
                else {
                    "sealed_internal_longitudinal": (
                        args.longitudinal_internal_seal
                    ),
                    "sealed_internal_esconv_auxiliary": (
                        args.auxiliary_internal_seal
                    ),
                }
            ),
            "training_script": Path(__file__),
            "shortcut_audit_report": args.shortcut_audit_report,
            "shortcut_audit_attestation": shortcut_attestation,
            "rule_grid_report": args.rule_grid_report,
            "rule_grid_attestation": rule_grid_attestation,
        },
        parameters={
            "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
            "selected_algorithm": selected_algorithm,
            "equal_top_level_domain_weight": equal_domain_weight,
            "longitudinal_cost_matched_action": long_frontier["cost_matched_action"],
            "longitudinal_best_fixed_action": long_frontier["best_fixed_action"],
            "auxiliary_fixed_actions": ["M0+R0", "M0+RS"],
            "selection_config_sha256": model.selection_config.digest(),
            **(
                {
                    "longitudinal_internal_commitment_sha256": (
                        long_commitment["commitment_sha256"]
                    ),
                    "auxiliary_internal_commitment_sha256": (
                        aux_commitment["commitment_sha256"]
                    ),
                    "internal_test_outcomes_opened": False,
                }
                if args.fit_only
                else {
                    "longitudinal_internal_seal_sha256": (
                        long_seal["seal_sha256"]
                    ),
                    "auxiliary_internal_seal_sha256": aux_seal["seal_sha256"],
                }
            ),
        },
    )

    if args.fit_only:
        fit_report = {
            "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
            "status": (
                "CANDIDATE_FROZEN_BEFORE_INTERNAL_TEST"
                if calibration_candidate_supported
                else "CANDIDATE_NOT_SUPPORTED_BEFORE_INTERNAL_TEST"
            ),
            "fit_only": True,
            "supervision_status": (
                "LLM_WEAK_SUPERVISION_NOT_GOLD"
                if weak_supervision_mode
                else "AUTOMATIC_LABELS"
            ),
            "weak_supervision_contract": measurement_contract,
            "internal_test_outcomes_opened": False,
            "run_identity": args.run_identity,
            "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
            "semantic_runtime": semantic_runtime,
            "dual_input_report": inputs.report,
            "domain_label_audits": domain_audits,
            "algorithm_selection": algorithm_selection,
            "algorithm_selection_checkpoint": str(
                algorithm_checkpoint_path
            ),
            "algorithm_selection_checkpoint_sha256": sha256_file(
                algorithm_checkpoint_path
            ),
            "selected_algorithm": selected_algorithm,
            "training": model.training_report,
            "equal_top_level_domain_weight": equal_domain_weight,
            "pooled_ood_calibration": ood_calibration,
            "per_domain_ood": per_domain_ood,
            "pooled_uncertainty_calibration": uncertainty,
            "per_domain_uncertainty": per_domain_uncertainty,
            "selection_calibration": tuning,
            "transparent_rule_train_only_tuning": rule_tuning,
            "calibration_comparators": {
                LONGITUDINAL_DOMAIN: long_frontier,
                ESCONV_AUXILIARY_DOMAIN: aux_frontier,
            },
            "pre_internal_calibration_action_viability": (
                pre_internal_action_viability
            ),
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "transparent_rule_checkpoint": str(rule_checkpoint),
            "transparent_rule_checkpoint_sha256": sha256_file(
                rule_checkpoint
            ),
            "next_stage": (
                "generate/seal internal labels only after this candidate "
                "manifest is immutable, then consume each domain once"
                if calibration_candidate_supported
                else (
                    "stop before internal; redesign using train/calibration "
                    "only and fit a new candidate identity"
                )
            ),
        }
        write_json(args.out_dir / "fit_report.json", fit_report)
        return

    ledger_paths = {
        LONGITUDINAL_DOMAIN: args.out_dir / "internal_longitudinal_ledger.jsonl",
        ESCONV_AUXILIARY_DOMAIN: args.out_dir / "internal_esconv_auxiliary_ledger.jsonl",
    }
    starts = {
        LONGITUDINAL_DOMAIN: begin_internal_test_consumption(
            ledger_paths[LONGITUDINAL_DOMAIN],
            candidate_manifest_path=candidate_path,
            internal_labels_path=args.longitudinal_internal_test_labels,
            sealed_artifact_name="sealed_internal_longitudinal",
            consumption_domain=LONGITUDINAL_DOMAIN,
        ),
        ESCONV_AUXILIARY_DOMAIN: begin_internal_test_consumption(
            ledger_paths[ESCONV_AUXILIARY_DOMAIN],
            candidate_manifest_path=candidate_path,
            internal_labels_path=args.auxiliary_internal_test_labels,
            sealed_artifact_name="sealed_internal_esconv_auxiliary",
            consumption_domain=ESCONV_AUXILIARY_DOMAIN,
        ),
    }
    long_internal_states, long_internal_labels = validate_internal_domain_labels(
        states=[state for state in longitudinal_states if state.split is PMV2Split.INTERNAL_TEST],
        labels=_load_labels(args.longitudinal_internal_test_labels),
        domain=LONGITUDINAL_DOMAIN,
    )
    aux_internal_states, aux_internal_labels = validate_internal_domain_labels(
        states=[state for state in auxiliary_states if state.split is PMV2Split.INTERNAL_TEST],
        labels=_load_labels(args.auxiliary_internal_test_labels),
        domain=ESCONV_AUXILIARY_DOMAIN,
    )
    gate_cfg = config["protocol_gates"]
    internal_gate_cfg = config["internal_reportability_gate"]
    long_fixed = sorted({long_frontier["cost_matched_action"], "ME+R0"})
    gates = {
        LONGITUDINAL_DOMAIN: domain_internal_gate(
            model=model,
            rule_router=rule_router,
            states=long_internal_states,
            labels=long_internal_labels,
            domain=LONGITUDINAL_DOMAIN,
            fixed_actions=long_fixed,
            gate_config=gate_cfg,
            uncertainty_confidence_level=float(uncertainty_cfg["confidence_level"]),
            bootstrap_replicates=int(internal_gate_cfg["paired_bootstrap_replicates"]),
            bootstrap_confidence_level=float(internal_gate_cfg["paired_bootstrap_confidence_level"]),
            bootstrap_seed=int(internal_gate_cfg["paired_bootstrap_seed"]),
        ),
        ESCONV_AUXILIARY_DOMAIN: domain_internal_gate(
            model=model,
            rule_router=rule_router,
            states=aux_internal_states,
            labels=aux_internal_labels,
            domain=ESCONV_AUXILIARY_DOMAIN,
            fixed_actions=["M0+R0", "M0+RS"],
            gate_config=gate_cfg,
            uncertainty_confidence_level=float(uncertainty_cfg["confidence_level"]),
            bootstrap_replicates=int(internal_gate_cfg["paired_bootstrap_replicates"]),
            bootstrap_confidence_level=float(internal_gate_cfg["paired_bootstrap_confidence_level"]),
            bootstrap_seed=int(internal_gate_cfg["paired_bootstrap_seed"]) + 10000,
        ),
    }
    joint_status = "PASS" if all(row["status"] == "PASS" for row in gates.values()) else "NOT_SUPPORTED"
    report = {
        "protocol": DUAL_DOMAIN_GATE_PROTOCOL,
        "status": "COMPLETE" if joint_status == "PASS" else "NONREPORTABLE",
        "joint_gate_status": joint_status,
        "one_domain_cannot_compensate_for_another": True,
        "run_identity": args.run_identity,
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "semantic_runtime": semantic_runtime,
        "dual_input_report": inputs.report,
        "domain_label_audits": domain_audits,
        "algorithm_selection": algorithm_selection,
        "selected_algorithm": selected_algorithm,
        "training": model.training_report,
        "equal_top_level_domain_weight": equal_domain_weight,
        "pooled_ood_calibration": ood_calibration,
        "per_domain_ood": per_domain_ood,
        "pooled_uncertainty_calibration": uncertainty,
        "per_domain_uncertainty": per_domain_uncertainty,
        "selection_calibration": tuning,
        "transparent_rule_train_only_tuning": rule_tuning,
        "calibration_comparators": {
            LONGITUDINAL_DOMAIN: long_frontier,
            ESCONV_AUXILIARY_DOMAIN: aux_frontier,
        },
        "internal_gates": gates,
        "internal_consumption_started": starts,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transparent_rule_checkpoint": str(rule_checkpoint),
        "transparent_rule_checkpoint_sha256": sha256_file(rule_checkpoint),
    }
    report_path = args.out_dir / "training_report.json"
    write_json(report_path, report)
    for domain, ledger_path in ledger_paths.items():
        finish_internal_test_consumption(
            ledger_path,
            report_path=report_path,
            expected_domain=domain,
        )
    if joint_status != "PASS" and not args.allow_nonreportable:
        raise RuntimeError(
            "dual-domain PM failed at least one independent internal gate; external evaluation is forbidden"
        )


if __name__ == "__main__":
    main()
