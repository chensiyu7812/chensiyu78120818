#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from metacom_pm.config import load_config
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, write_json
from metacom_pm.pm_v2_contracts import ResponseDimensions
from metacom_pm.pm_v2_judging import (
    JUDGE_RUBRIC_VERSION,
    composite_spec_from_config,
    composite_weights_hash,
    prompt_contract_hash,
)

ROOT = Path(__file__).resolve().parents[1]

RESPONSE_FIELDS = (
    "emotional_support",
    "personalization",
    "memory_appropriateness",
    "factual_grounding",
    "temporal_consistency",
    "non_intrusiveness",
)
RISK_FIELDS = (
    "selected_context_misuse",
    "unnecessary_exposure",
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "memory_omission",
    "strategy_overuse",
    "strategy_omission",
)
MANUAL_VERSION = "pm-v2-human-rating-manual-v2-independent-dimensions"
SAMPLE_PLAN_VERSION = "pm-v2-human-sample-plan-v1"


def read_completed(paths):
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("item_id") or not row.get("annotator_id"):
                    continue
                for field in (*RESPONSE_FIELDS, *RISK_FIELDS):
                    if row.get(field, "").strip() == "":
                        raise ValueError(
                            f"missing {field} for item {row['item_id']} annotator {row['annotator_id']}"
                        )
                    row[field] = float(row[field])
                    if not row[field].is_integer():
                        raise ValueError(
                            f"{field} must use the manual's whole-number scale for "
                            f"item {row['item_id']} annotator {row['annotator_id']}"
                        )
                if any(not 1.0 <= row[field] <= 5.0 for field in RESPONSE_FIELDS):
                    raise ValueError("response score outside 1-5")
                if any(not 0.0 <= row[field] <= 3.0 for field in RISK_FIELDS):
                    raise ValueError("risk score outside 0-3")
                rows.append(row)
    return rows


def safe_spearman(left, right):
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = spearmanr(left, right).statistic
    return None if np.isnan(value) else float(value)


def agreement_metrics(human, llm):
    human_array = np.asarray(human, dtype=float)
    llm_array = np.asarray(llm, dtype=float)
    return {
        "n": len(human),
        "mae": float(np.mean(np.abs(human_array - llm_array))),
        "within_one_rate": float(np.mean(np.abs(human_array - llm_array) <= 1.0)),
        "exact_rate": float(np.mean(human_array == llm_array)),
        "spearman": safe_spearman(human, llm),
        "human_mean": float(np.mean(human_array)),
        "llm_mean": float(np.mean(llm_array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed", type=Path, nargs="+", required=True)
    parser.add_argument("--key", type=Path, default=ROOT / "outputs" / "pm_v2_human_audit" / "human_rating_key.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v2_human_audit" / "human_audit_report.json")
    parser.add_argument(
        "--manual",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_human_audit" / "human_rating_manual.md",
    )
    parser.add_argument(
        "--sample-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_human_audit" / "human_sample_plan.json",
    )
    parser.add_argument(
        "--judge-manifest",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_judging" / "run_manifest.json",
    )
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    audit_cfg = config["human_label_audit"]
    minimum_annotators = int(audit_cfg["minimum_annotators"])
    maximum_response_mae = float(audit_cfg["maximum_response_mae"])
    minimum_response_spearman = float(audit_cfg["minimum_response_spearman"])
    maximum_risk_mae = float(audit_cfg["maximum_risk_mae"])
    minimum_within_one_rate = float(audit_cfg["minimum_within_one_rate"])
    minimum_interrater_kappa = float(audit_cfg["minimum_interrater_kappa"])
    composite_spec = composite_spec_from_config(config)
    key_data = json.loads(args.key.read_text(encoding="utf-8"))
    if key_data.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("human audit key was prepared under a different PM-v2 config")
    if key_data.get("human_audit_config") != audit_cfg:
        raise RuntimeError("human audit key thresholds do not match frozen PM-v2 YAML")
    if key_data.get("quality_composite_version") != composite_spec.version:
        raise RuntimeError("human audit key composite version does not match PM-v2 YAML")
    expected_manual = {
        "manual": str(args.manual),
        "manual_version": MANUAL_VERSION,
        "manual_sha256": sha256_file(args.manual),
    }
    for field, expected in expected_manual.items():
        actual = key_data.get(field)
        if field == "manual":
            if Path(str(actual)).resolve() != args.manual.resolve():
                raise RuntimeError("human audit manual path mismatch")
        elif actual != expected:
            raise RuntimeError(f"human audit manual contract mismatch: {field}")
    manual_text = args.manual.read_text(encoding="utf-8")
    required_manual_fragments = [MANUAL_VERSION]
    for field in (
        "response_dimension_definitions",
        "response_dimension_boundaries",
        "response_scale_anchors",
        "risk_dimension_definitions",
        "risk_scale_anchors",
    ):
        value = key_data.get(field)
        if not isinstance(value, dict) or not value:
            raise RuntimeError(f"human audit key lacks manual anchors: {field}")
        for name, description in value.items():
            required_manual_fragments.append(str(name))
            if isinstance(description, list):
                required_manual_fragments.extend(str(item) for item in description)
            else:
                required_manual_fragments.append(str(description))
    missing_manual_fragments = sorted(
        fragment for fragment in required_manual_fragments if fragment not in manual_text
    )
    if missing_manual_fragments:
        raise RuntimeError(
            "human rating manual is inconsistent with its frozen anchors: "
            + str(missing_manual_fragments[:10])
        )

    sample_wrapper = key_data.get("sample_plan") or {}
    sample_plan = read_json(args.sample_plan)
    expected_sample_wrapper = {
        "version": SAMPLE_PLAN_VERSION,
        "sha256": sha256_text(canonical_json(sample_plan)),
        "file_sha256": sha256_file(args.sample_plan),
        "sample_seed": int(audit_cfg["sample_seed"]),
        "seed_source": "pm_v2.yaml:human_label_audit.sample_seed",
    }
    if Path(str(sample_wrapper.get("path") or "")).resolve() != args.sample_plan.resolve():
        raise RuntimeError("human sample-plan path mismatch")
    for field, expected in expected_sample_wrapper.items():
        if sample_wrapper.get(field) != expected:
            raise RuntimeError(f"human sample-plan wrapper mismatch: {field}")
    expected_sample_plan_keys = {
        "version",
        "sample_seed",
        "seed_source",
        "target_items",
        "regime_order",
        "initial_per_regime_quota",
        "candidate_pool_count",
        "candidate_pool_sha256",
        "ordered_selection",
        "input_bindings",
    }
    if set(sample_plan) != expected_sample_plan_keys:
        raise RuntimeError("human sample plan has an unexpected schema")
    if (
        sample_plan["version"] != SAMPLE_PLAN_VERSION
        or int(sample_plan["sample_seed"]) != int(audit_cfg["sample_seed"])
        or sample_plan["seed_source"]
        != "pm_v2.yaml:human_label_audit.sample_seed"
        or int(sample_plan["target_items"]) != int(audit_cfg["items"])
        or len(sample_plan["ordered_selection"]) != int(audit_cfg["items"])
    ):
        raise RuntimeError("human sample plan violates frozen YAML")
    sample_inputs = sample_plan.get("input_bindings") or {}
    expected_sample_inputs = {
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "judge_manifest_sha256": sha256_file(args.judge_manifest),
        "llm_rubric_version": JUDGE_RUBRIC_VERSION,
        "llm_prompt_contract_sha256": prompt_contract_hash(),
        "quality_composite_version": composite_spec.version,
        "quality_composite_weights_sha256": composite_weights_hash(
            composite_spec
        ),
    }
    for field, expected in expected_sample_inputs.items():
        if sample_inputs.get(field) != expected:
            raise RuntimeError(f"human sample-plan input mismatch: {field}")

    rubric = key_data.get("llm_rubric_contract") or {}
    judge_manifest = read_json(args.judge_manifest)
    expected_rubric = {
        "version": JUDGE_RUBRIC_VERSION,
        "prompt_contract_sha256": prompt_contract_hash(),
        "requests_overall_field": False,
        "judge_manifest_sha256": sha256_file(args.judge_manifest),
    }
    if Path(str(rubric.get("judge_manifest") or "")).resolve() != args.judge_manifest.resolve():
        raise RuntimeError("human audit LLM-rubric judge-manifest path mismatch")
    for field, expected in expected_rubric.items():
        if rubric.get(field) != expected:
            raise RuntimeError(f"human audit LLM-rubric mismatch: {field}")
    expected_judge_bindings = {
        "prompt_contract_hash": prompt_contract_hash(),
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "composite_weights_sha256": composite_weights_hash(composite_spec),
    }
    for field, expected in expected_judge_bindings.items():
        if judge_manifest.get(field) != expected:
            raise RuntimeError(f"human audit judge-manifest mismatch: {field}")
    key_rows = {row["item_id"]: row for row in key_data["key_rows"]}
    ordered_item_ids = [str(row["item_id"]) for row in sample_plan["ordered_selection"]]
    if len(set(ordered_item_ids)) != len(ordered_item_ids) or set(ordered_item_ids) != set(
        key_rows
    ):
        raise RuntimeError("human sample plan and rating key item matrix differ")
    for selection in sample_plan["ordered_selection"]:
        key_row = key_rows[str(selection["item_id"])]
        for field in (
            "state_id",
            "card_id",
            "action_id",
            "regime",
            "outcome_request_hash",
        ):
            if selection.get(field) != key_row.get(field):
                raise RuntimeError(
                    "human sample plan/rating-key mismatch: "
                    f"{selection['item_id']}/{field}"
                )
    human_rows = read_completed(args.completed)
    by_item = defaultdict(list)
    seen = set()
    for row in human_rows:
        key = (row["item_id"], row["annotator_id"])
        if key in seen:
            raise RuntimeError(f"duplicate human rating: {key}")
        seen.add(key)
        if row["item_id"] not in key_rows:
            raise RuntimeError(f"unknown human audit item: {row['item_id']}")
        by_item[row["item_id"]].append(row)
    missing = [
        item_id
        for item_id in key_rows
        if len(by_item[item_id]) < minimum_annotators
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} audit items have fewer than {minimum_annotators} ratings"
        )

    human_median = {}
    for item_id, rows in by_item.items():
        human_median[item_id] = {
            field: float(np.median([row[field] for row in rows]))
            for field in (*RESPONSE_FIELDS, *RISK_FIELDS)
        }

    dimensions = {}
    for group, fields, llm_key in (
        ("response", RESPONSE_FIELDS, "llm_response"),
        ("risk", RISK_FIELDS, "llm_risk"),
    ):
        for field in fields:
            human = [human_median[item_id][field] for item_id in sorted(key_rows)]
            llm = [float(key_rows[item_id][llm_key][field]) for item_id in sorted(key_rows)]
            dimensions[f"{group}.{field}"] = agreement_metrics(human, llm)

    annotators = sorted({str(row["annotator_id"]) for row in human_rows})
    kappas_by_dimension = {
        f"response.{field}": [] for field in RESPONSE_FIELDS
    } | {f"risk.{field}": [] for field in RISK_FIELDS}
    for left_index in range(len(annotators)):
        for right_index in range(left_index + 1, len(annotators)):
            left_id, right_id = annotators[left_index], annotators[right_index]
            common = []
            for item_id in sorted(key_rows):
                left = next(
                    (row for row in by_item[item_id] if row["annotator_id"] == left_id),
                    None,
                )
                right = next(
                    (row for row in by_item[item_id] if row["annotator_id"] == right_id),
                    None,
                )
                if left is not None and right is not None:
                    common.append((left, right))
            for group, fields in (("response", RESPONSE_FIELDS), ("risk", RISK_FIELDS)):
                for field in fields:
                    if len(common) < 3:
                        continue
                    left_values = [round(row[0][field]) for row in common]
                    right_values = [round(row[1][field]) for row in common]
                    value = cohen_kappa_score(
                        left_values, right_values, weights="quadratic"
                    )
                    if not np.isnan(value):
                        kappas_by_dimension[f"{group}.{field}"].append(float(value))
    dimension_kappa = {
        name: (float(np.mean(values)) if values else None)
        for name, values in kappas_by_dimension.items()
    }
    observed_kappas = [value for value in dimension_kappa.values() if value is not None]
    mean_kappa = float(np.mean(observed_kappas)) if observed_kappas else None
    response_rows = [dimensions[f"response.{field}"] for field in RESPONSE_FIELDS]
    risk_rows = [dimensions[f"risk.{field}"] for field in RISK_FIELDS]
    response_spearman = [
        row["spearman"] for row in response_rows if row["spearman"] is not None
    ]
    sorted_item_ids = sorted(key_rows)
    human_composite = []
    llm_composite = []
    for item_id in sorted_item_ids:
        human_dimensions = ResponseDimensions(
            **{field: human_median[item_id][field] for field in RESPONSE_FIELDS}
        )
        llm_dimensions = ResponseDimensions.model_validate(key_rows[item_id]["llm_response"])
        human_composite.append(1.0 + 4.0 * composite_spec.score(human_dimensions))
        llm_composite.append(1.0 + 4.0 * composite_spec.score(llm_dimensions))
    composite_validity = agreement_metrics(human_composite, llm_composite)

    checks = {}
    for field in RESPONSE_FIELDS:
        row = dimensions[f"response.{field}"]
        checks[f"response.{field}.mae"] = row["mae"] <= maximum_response_mae
        checks[f"response.{field}.spearman"] = (
            row["spearman"] is not None
            and row["spearman"] >= minimum_response_spearman
        )
        checks[f"response.{field}.within_one"] = (
            row["within_one_rate"] >= minimum_within_one_rate
        )
        checks[f"response.{field}.kappa"] = (
            dimension_kappa[f"response.{field}"] is not None
            and dimension_kappa[f"response.{field}"] >= minimum_interrater_kappa
        )
    for field in RISK_FIELDS:
        row = dimensions[f"risk.{field}"]
        checks[f"risk.{field}.mae"] = row["mae"] <= maximum_risk_mae
        checks[f"risk.{field}.within_one"] = (
            row["within_one_rate"] >= minimum_within_one_rate
        )
        checks[f"risk.{field}.kappa"] = (
            dimension_kappa[f"risk.{field}"] is not None
            and dimension_kappa[f"risk.{field}"] >= minimum_interrater_kappa
        )
    checks.update(
        {
            "quality_composite.mae": composite_validity["mae"]
            <= maximum_response_mae,
            "quality_composite.spearman": composite_validity["spearman"] is not None
            and composite_validity["spearman"] >= minimum_response_spearman,
            "quality_composite.within_one": composite_validity["within_one_rate"]
            >= minimum_within_one_rate,
        }
    )
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "n_items": len(key_rows),
        "n_human_rows": len(human_rows),
        "annotators": annotators,
        "dimensions": dimensions,
        "mean_response_mae": float(np.mean([row["mae"] for row in response_rows])),
        "mean_response_spearman": (
            float(np.mean(response_spearman)) if response_spearman else None
        ),
        "mean_response_within_one_rate": float(
            np.mean([row["within_one_rate"] for row in response_rows])
        ),
        "mean_risk_mae": float(np.mean([row["mae"] for row in risk_rows])),
        "mean_pairwise_quadratic_kappa": mean_kappa,
        "interrater_kappa_by_dimension": dimension_kappa,
        "quality_composite": {
            "version": composite_spec.version,
            "spec_sha256": sha256_text(
                canonical_json(composite_spec.model_dump(mode="json"))
            ),
            "validity": composite_validity,
        },
        "checks": checks,
        "thresholds": {
            "minimum_annotators": minimum_annotators,
            "maximum_response_mae": maximum_response_mae,
            "minimum_response_spearman": minimum_response_spearman,
            "maximum_risk_mae": maximum_risk_mae,
            "minimum_within_one_rate": minimum_within_one_rate,
            "minimum_interrater_kappa": minimum_interrater_kappa,
        },
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "human_key_sha256": sha256_file(args.key),
        "manual": str(args.manual.resolve()),
        "manual_version": MANUAL_VERSION,
        "manual_sha256": sha256_file(args.manual),
        "sample_plan": {
            **expected_sample_wrapper,
            "path": str(args.sample_plan.resolve()),
        },
        "llm_rubric_contract": {
            **expected_rubric,
            "judge_manifest": str(args.judge_manifest.resolve()),
        },
        "completed_inputs_sha256": {
            str(path.resolve()): sha256_file(path) for path in sorted(args.completed)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report)
    if report["status"] != "PASS":
        raise RuntimeError("PM-v2 human judge calibration gate failed")


if __name__ == "__main__":
    main()
