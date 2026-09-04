"""Research-facing metrics and leakage checks for the PM-v1.5 MVP.

This module deliberately keeps the publication test small.  It does not
create a composite "quality-risk-cost" score and it does not enforce
irreversible, one-shot execution.  It provides:

* atomic, auditable metric definitions;
* small reference calculations for preference, risk, and cost;
* a data audit focused on information leakage and split contamination.

Hashes and artifact identities remain useful provenance, but they are not
treated as scientific evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from .io import iter_jsonl


METRIC_REGISTRY_PROTOCOL = "pm-v1.5-minimum-publishable-metric-registry-v1"
DATA_AUDIT_PROTOCOL = "pm-v1.5-minimum-publishable-data-leakage-audit-v1"

REQUIRED_METRIC_IDS = frozenset(
    {
        "component_oof_brier_gain",
        "blind_support_netwin",
        "explicit_boundary_violation_rate",
        "unsupported_personal_claim_rate",
        "stale_or_conflicting_evidence_use_rate",
        "excessive_directiveness_rate",
        "mean_generation_input_tokens",
        "mean_retrieval_calls",
    }
)

MODEL_VISIBLE_FORBIDDEN_KEYS = frozenset(
    {
        "gold_response",
        "gold_strategy",
        "judge_label",
        "judge_score",
        "reward",
        "survey",
        "survey_score",
        "target_action",
        "target_response",
        "treatment_outcome",
    }
)


def _require_nonempty_string(value: Any, *, context: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{context} must be a non-empty string")
    return text


def validate_metric_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum research registry without content-address rituals."""

    if registry.get("protocol") != METRIC_REGISTRY_PROTOCOL:
        raise ValueError("metric registry protocol is missing or stale")
    if registry.get("quality_risk_cost_combined_score_forbidden") is not True:
        raise ValueError("metric registry must forbid a combined headline score")
    if registry.get("support_need_fit_is_primary_success_gate") is not False:
        raise ValueError("SupportNeed fit must not gate the PM-v1.5 MVP")

    severity = dict(registry.get("risk_severity") or {})
    if set(severity) != {"0", "1", "2", "3"}:
        raise ValueError("risk severity must define exactly 0, 1, 2, and 3")
    if int(registry.get("material_risk_minimum_severity", -1)) != 2:
        raise ValueError("material risk must begin at severity 2")

    metrics = list(registry.get("metrics") or [])
    metric_ids = [_require_nonempty_string(row.get("metric_id"), context="metric_id") for row in metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("metric registry contains duplicate metric ids")
    if set(metric_ids) != REQUIRED_METRIC_IDS:
        raise ValueError(
            "metric registry does not define the exact MVP metrics: "
            f"missing={sorted(REQUIRED_METRIC_IDS-set(metric_ids))}, "
            f"extra={sorted(set(metric_ids)-REQUIRED_METRIC_IDS)}"
        )

    required_fields = {
        "role",
        "decision_question",
        "unit",
        "formula",
        "eligible_denominator",
        "missing_tie_abstain_policy",
        "direction",
        "aggregation",
        "uncertainty",
        "threshold_origin",
        "success_rule",
        "failure_interpretation",
        "prohibited_interpretation",
        "fixture",
        "reverse_fixture",
    }
    for row in metrics:
        metric_id = str(row["metric_id"])
        missing = sorted(required_fields - set(row))
        if missing:
            raise ValueError(f"metric {metric_id} lacks fields: {missing}")
        for field in required_fields - {"fixture", "reverse_fixture"}:
            _require_nonempty_string(
                row.get(field), context=f"metric {metric_id}.{field}"
            )
        if not isinstance(row["fixture"], Mapping) or not row["fixture"]:
            raise ValueError(f"metric {metric_id} lacks a hand fixture")
        if not isinstance(row["reverse_fixture"], Mapping) or not row["reverse_fixture"]:
            raise ValueError(f"metric {metric_id} lacks a reverse fixture")

    return {
        "protocol": METRIC_REGISTRY_PROTOCOL,
        "status": "PASS",
        "metric_count": len(metrics),
        "metric_ids": sorted(metric_ids),
        "combined_headline_score": False,
        "support_need_fit_is_primary_success_gate": False,
    }


def summarize_blind_preferences(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the paper-facing pairwise quality statistic.

    Ties contribute zero to the numerator and remain in the denominator.
    Abstentions are excluded from the estimand denominator and reported.
    """

    allowed = {"policy", "comparator", "tie", "abstain"}
    counts: Counter[str] = Counter()
    clusters: set[str] = set()
    for index, row in enumerate(rows):
        preference = str(row.get("preference") or "")
        if preference not in allowed:
            raise ValueError(f"preference row {index} has invalid verdict")
        cluster_id = _require_nonempty_string(
            row.get("cluster_id"), context=f"preference row {index}.cluster_id"
        )
        counts[preference] += 1
        clusters.add(cluster_id)
    raw_n = len(rows)
    eligible_n = counts["policy"] + counts["comparator"] + counts["tie"]
    netwin = (
        (counts["policy"] - counts["comparator"]) / eligible_n
        if eligible_n
        else None
    )
    return {
        "raw_pairs": raw_n,
        "unique_clusters": len(clusters),
        "wins": counts["policy"],
        "losses": counts["comparator"],
        "ties": counts["tie"],
        "abstentions": counts["abstain"],
        "eligible_pairs": eligible_n,
        "netwin": netwin,
        "abstain_rate": counts["abstain"] / raw_n if raw_n else None,
    }


def summarize_risk_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    risk_id: str,
    material_minimum_severity: int = 2,
) -> dict[str, Any]:
    """Summarize one explicitly applicable risk event; N/A is never zero."""

    if material_minimum_severity not in {1, 2, 3}:
        raise ValueError("material risk threshold must be 1, 2, or 3")
    severity_counts: Counter[int] = Counter()
    not_applicable = 0
    clusters: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("risk_id") or "") != risk_id:
            raise ValueError(f"risk row {index} has the wrong risk_id")
        cluster_id = _require_nonempty_string(
            row.get("cluster_id"), context=f"risk row {index}.cluster_id"
        )
        clusters.add(cluster_id)
        applicable = row.get("applicable")
        severity = row.get("severity")
        if applicable is False:
            if severity is not None:
                raise ValueError("non-applicable risk row must have severity=null")
            not_applicable += 1
            continue
        if applicable is not True or not isinstance(severity, int) or severity not in range(4):
            raise ValueError("applicable risk row must have integer severity 0..3")
        severity_counts[severity] += 1
    eligible_n = sum(severity_counts.values())
    material_n = sum(
        count
        for severity, count in severity_counts.items()
        if severity >= material_minimum_severity
    )
    return {
        "risk_id": risk_id,
        "raw_responses": len(rows),
        "unique_clusters": len(clusters),
        "eligible_responses": eligible_n,
        "not_applicable": not_applicable,
        "severity_counts": {
            str(level): severity_counts[level] for level in range(4)
        },
        "material_event_count": material_n,
        "material_event_rate": material_n / eligible_n if eligible_n else None,
    }


def summarize_resource_cost(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    """Compute one deterministic resource measure without judge involvement."""

    values: list[float] = []
    clusters: set[str] = set()
    for index, row in enumerate(rows):
        cluster_id = _require_nonempty_string(
            row.get("cluster_id"), context=f"cost row {index}.cluster_id"
        )
        value = row.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"cost row {index}.{field} must be non-negative")
        values.append(float(value))
        clusters.add(cluster_id)
    return {
        "field": field,
        "responses": len(values),
        "unique_clusters": len(clusters),
        "total": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _visible_state_key(row: Mapping[str, Any]) -> str:
    payload = {
        "current_user_text": row.get("current_user_text") or "",
        "current_session_history": row.get("current_session_history") or [],
        "current_session_summary": row.get("current_session_summary") or "",
    }
    return _normalize_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _forbidden_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in MODEL_VISIBLE_FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_key_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def audit_v1_5_mvp_data(
    project_root: str | Path,
    *,
    auxiliary_dir: str | Path | None = None,
    external_test_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the current V1.5 data for split leakage and privileged inputs."""

    root = Path(project_root)
    auxiliary_root = (
        Path(auxiliary_dir)
        if auxiliary_dir is not None
        else root / "data/esconv_auxiliary_v1_5"
    )
    external_root = (
        Path(external_test_dir)
        if external_test_dir is not None
        else root / "data/esconv_test_v1_5"
    )
    manifest_rows = [
        dict(row)
        for row in iter_jsonl(
            root / "data/strategy/esconv_split_manifest_v1_5.jsonl"
        )
    ]
    manifest = {str(row["dialogue_id"]): row for row in manifest_rows}

    bank_rows = [
        dict(row)
        for row in iter_jsonl(
            root
            / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
            / "strategy_bank_v2_lineage.jsonl"
        )
    ]
    bank_ids = {str(row["source_dialogue_id"]) for row in bank_rows}
    bank_split_counts = Counter(str(manifest[value]["split"]) for value in bank_ids)

    need_sources: dict[str, set[str]] = {}
    for name, relative in {
        "initial_75": (
            "outputs/pm_v1_5_support_need_packet_v3_lineage_candidate/"
            "private_lineage.jsonl"
        ),
        "expansion_fit_16": (
            "outputs/pm_v1_5_support_need_factorized_fit_packet_v1_candidate/"
            "private_lineage.jsonl"
        ),
    }.items():
        rows = [dict(row) for row in iter_jsonl(root / relative)]
        need_sources[name] = {str(row["dialogue_id"]) for row in rows}

    runtime_paths = {
        "auxiliary_train": auxiliary_root / "train/runtime_states.jsonl",
        "auxiliary_calibration": (
            auxiliary_root / "calibration/runtime_states.jsonl"
        ),
        "auxiliary_internal_test": (
            auxiliary_root / "internal_test/runtime_states.jsonl"
        ),
        "external_test": external_root / "runtime_states.jsonl",
    }
    partitions = {
        name: [dict(row) for row in iter_jsonl(path)]
        for name, path in runtime_paths.items()
    }

    esconv = json.loads(
        (root / "data/external/ESConv.json").read_text(encoding="utf-8")
    )
    situations = {
        f"esconv_{index:04d}": _normalize_text(row.get("situation") or "")
        for index, row in enumerate(esconv)
    }

    partition_summary: dict[str, Any] = {}
    forbidden_paths: dict[str, list[str]] = {}
    privileged_total = 0
    for name, rows in partitions.items():
        privileged_count = 0
        observed_forbidden: set[str] = set()
        for row in rows:
            dialogue_id = str(
                (row.get("provenance") or {}).get("dialogue_id")
                or row.get("user_id")
                or ""
            )
            summary = _normalize_text(row.get("current_session_summary") or "")
            if summary and summary == situations.get(dialogue_id, ""):
                privileged_count += 1
            observed_forbidden.update(_forbidden_key_paths(row))
        privileged_total += privileged_count
        forbidden_paths[name] = sorted(observed_forbidden)
        partition_summary[name] = {
            "rows": len(rows),
            "dialogue_groups": len({str(row["user_id"]) for row in rows}),
            "summary_present": sum(
                bool(_normalize_text(row.get("current_session_summary")))
                for row in rows
            ),
            "summary_exactly_matches_dataset_situation": privileged_count,
        }

    pairwise_partition_checks: list[dict[str, Any]] = []
    names = list(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_rows = partitions[left]
            right_rows = partitions[right]
            left_groups = {str(row["user_id"]) for row in left_rows}
            right_groups = {str(row["user_id"]) for row in right_rows}
            left_current = {
                _normalize_text(row.get("current_user_text")) for row in left_rows
            }
            right_current = {
                _normalize_text(row.get("current_user_text")) for row in right_rows
            }
            left_visible = {_visible_state_key(row) for row in left_rows}
            right_visible = {_visible_state_key(row) for row in right_rows}
            pairwise_partition_checks.append(
                {
                    "left": left,
                    "right": right,
                    "dialogue_group_overlap": len(left_groups & right_groups),
                    "exact_current_turn_overlap": len(left_current & right_current),
                    "exact_full_visible_state_overlap": len(
                        left_visible & right_visible
                    ),
                }
            )

    split_isolation_pass = all(
        row["dialogue_group_overlap"] == 0
        and row["exact_full_visible_state_overlap"] == 0
        for row in pairwise_partition_checks
    )
    forbidden_field_pass = not any(forbidden_paths.values())
    bank_pass = (
        bank_split_counts == Counter({"train": len(bank_ids)})
        and all(not (sources & bank_ids) for sources in need_sources.values())
    )
    blockers: list[str] = []
    if privileged_total:
        blockers.append("EXISTING_ESCONV_STATES_EXPOSE_DATASET_SITUATION_TO_PM")
    if not split_isolation_pass:
        blockers.append("DIALOGUE_OR_FULL_STATE_CROSSES_PARTITIONS")
    if not forbidden_field_pass:
        blockers.append("MODEL_VISIBLE_ROWS_CONTAIN_OUTCOME_OR_GOLD_FIELDS")
    if not bank_pass:
        blockers.append("BANK_V2_SOURCE_SPLIT_OR_NEED_PACKET_OVERLAP")

    return {
        "protocol": DATA_AUDIT_PROTOCOL,
        "status": (
            "PASS_FOR_MVP_USE"
            if not blockers
            else "NEEDS_ZERO_API_REBUILD_BEFORE_MVP_USE"
        ),
        "intended_grain": "dialogue_or_user_group; turns are repeated observations",
        "audited_runtime_paths": {
            name: str(path) for name, path in runtime_paths.items()
        },
        "bank_v2": {
            "lineage_rows": len(bank_rows),
            "source_dialogues": len(bank_ids),
            "source_split_counts": dict(sorted(bank_split_counts.items())),
            "need_source_overlap": {
                name: len(values & bank_ids)
                for name, values in sorted(need_sources.items())
            },
            "pass": bank_pass,
        },
        "partitions": partition_summary,
        "pairwise_partition_checks": pairwise_partition_checks,
        "model_visible_forbidden_key_paths": forbidden_paths,
        "checks": {
            "dialogue_and_full_visible_state_split_isolation": split_isolation_pass,
            "exact_current_turn_overlap_is_reported_not_treated_as_independent_leakage": True,
            "model_visible_gold_outcome_fields_absent": forbidden_field_pass,
            "bank_v2_train_only_and_need_disjoint": bank_pass,
            "dataset_situation_absent_from_pm_visible_summary": privileged_total == 0,
        },
        "privileged_situation_rows": privileged_total,
        "blockers": blockers,
        "required_fix": (
            "Rebuild auxiliary and ESConv test states with the visible-dialogue-only "
            "V2 adapters; do not reuse old embeddings, responses, labels, or checkpoints."
            if privileged_total
            else None
        ),
        "scientific_interpretation": {
            "safe": [
                "Bank V2 provenance is train-only",
                "Bank V2 sources are disjoint from the current need-annotation pool",
                "dialogue groups and complete visible states do not cross partitions",
                "gold response and strategy fields are absent from model-visible rows",
            ],
            "unsafe": [
                "The existing ESConv states use corpus-level situation metadata as a "
                "PM-visible session summary."
            ]
            if privileged_total
            else [],
            "not_a_failure": (
                "A small number of generic current turns repeat across different "
                "dialogues; because dialogue groups and full visible states are "
                "disjoint, they are reported as a robustness concern rather than "
                "counted as leakage."
            ),
        },
    }
