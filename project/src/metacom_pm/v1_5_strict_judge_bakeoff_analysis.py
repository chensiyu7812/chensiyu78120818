from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any, Mapping, Sequence

from .v1_5_judge_qualification_analysis import (
    PREFERENCE_FIELDS,
    _direction_agreement,
    _pairwise_condition_report,
)
from .v1_5_strict_judge_bakeoff import (
    EXPECTED_CANDIDATES,
    EXPECTED_ORDERED_PAIR_COUNT,
    EXPECTED_PAIR_COUNT,
    StrictBakeoffPairwiseOutput,
)


STRICT_BAKEOFF_AGGREGATION_PROTOCOL = (
    "pm-v1.5-train-only-strict-pairwise-judge-bakeoff-aggregation-v1"
)


def _validated_result_rows(
    *,
    result_rows: Sequence[Mapping[str, Any]],
    call_plan_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(result_rows) != (
        len(EXPECTED_CANDIDATES) * EXPECTED_ORDERED_PAIR_COUNT
    ):
        raise RuntimeError("strict bake-off result matrix is incomplete")
    if len(call_plan_rows) != len(result_rows):
        raise RuntimeError("strict bake-off call plan/result count drifted")

    expected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in call_plan_rows:
        row = dict(raw)
        candidate = str(row.get("candidate_key") or "")
        record_ids = dict(row.get("record_ids") or {})
        pair_id = str(record_ids.get("pair_id") or "")
        order = int(record_ids.get("order_variant", -1))
        key = (candidate, pair_id, order)
        if (
            candidate not in EXPECTED_CANDIDATES
            or not pair_id
            or order not in {0, 1}
            or key in expected
        ):
            raise RuntimeError("strict bake-off call plan coverage is malformed")
        if str(row.get("condition") or "") != f"{candidate}_pairwise":
            raise RuntimeError("strict bake-off call plan condition drifted")
        expected[key] = row

    normalized: list[dict[str, Any]] = []
    observed: set[tuple[str, str, int]] = set()
    for raw in result_rows:
        row = dict(raw)
        candidate = str(row.get("candidate_key") or "")
        record_ids = dict(row.get("record_ids") or {})
        pair_id = str(record_ids.get("pair_id") or "")
        order = int(record_ids.get("order_variant", -1))
        key = (candidate, pair_id, order)
        if key not in expected or key in observed:
            raise RuntimeError("strict bake-off result coverage drifted")
        plan = expected[key]
        if record_ids != dict(plan["record_ids"]):
            raise RuntimeError("strict bake-off result record_ids drifted")
        if (
            str(row.get("judge_family") or "") != candidate
            or str(row.get("judge_family") or "")
            != str(plan["judge_family"])
            or str(row.get("judge_model") or "")
            != str(plan["judge_model"])
        ):
            raise RuntimeError("strict bake-off result judge identity drifted")
        if len(str(row.get("request_hash") or "")) != 64:
            raise RuntimeError("strict bake-off result request hash is absent")
        parsed = StrictBakeoffPairwiseOutput.model_validate(
            row.get("parsed"), strict=True
        ).model_dump(mode="json")
        normalized.append(
            {
                "condition": candidate,
                "candidate_key": candidate,
                "record_ids": record_ids,
                "parsed": parsed,
            }
        )
        observed.add(key)
    if observed != set(expected):
        raise RuntimeError("strict bake-off result matrix is incomplete")

    counts = Counter(row["candidate_key"] for row in normalized)
    if counts != Counter(
        {candidate: EXPECTED_ORDERED_PAIR_COUNT for candidate in EXPECTED_CANDIDATES}
    ):
        raise RuntimeError("strict bake-off candidate coverage drifted")
    return normalized


def _validated_gpt_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligible_pair_ids: set[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if row.get("condition") != "gpt_high_quality_pairwise_anchor":
            continue
        if (
            str(row.get("judge_family") or "") != "openai_gpt_5_6_sol"
            or str(row.get("judge_model") or "") != "gpt-5.6-sol"
        ):
            raise RuntimeError("GPT anchor identity drifted")
        record_ids = dict(row.get("record_ids") or {})
        pair_id = str(record_ids.get("pair_id") or "")
        if pair_id not in eligible_pair_ids:
            raise RuntimeError("GPT anchor pair is outside the frozen packet")
        parsed = StrictBakeoffPairwiseOutput.model_validate(
            row.get("parsed"), strict=True
        ).model_dump(mode="json")
        normalized.append(
            {
                "condition": "gpt_anchor",
                "record_ids": record_ids,
                "parsed": parsed,
            }
        )
    if len(normalized) != 16:
        raise RuntimeError("GPT anchor must contain exactly 16 selected calls")
    return normalized


def _human_agreement(
    *,
    candidate_pairs: Mapping[str, Mapping[str, Any]],
    human_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    human_by_pair = {str(row["pair_id"]): dict(row) for row in human_rows}
    if (
        len(human_by_pair) != len(human_rows)
        or set(human_by_pair) != set(candidate_pairs)
    ):
        raise RuntimeError("human anchor does not cover the strict bake-off pairs")

    dimensions: dict[str, Any] = {}
    for field in PREFERENCE_FIELDS:
        comparable: list[tuple[str, str, int]] = []
        for pair_id in sorted(candidate_pairs):
            model_value = str(
                candidate_pairs[pair_id][field]["canonical_preference"]
            )
            human_value = str(
                human_by_pair[pair_id][f"canonical_{field}"]
            )
            if model_value in {"A", "B"} and human_value in {"A", "B"}:
                comparable.append(
                    (
                        model_value,
                        human_value,
                        int(human_by_pair[pair_id]["confidence"]),
                    )
                )
        dimensions[field] = {
            "eligible_pairs": len(candidate_pairs),
            "comparable_pairs": len(comparable),
            "agreeing_pairs": sum(a == b for a, b, _ in comparable),
            "agreement_rate": (
                sum(a == b for a, b, _ in comparable) / len(comparable)
                if comparable
                else None
            ),
            "by_human_confidence": {
                str(confidence): {
                    "comparable_pairs": len(group),
                    "agreeing_pairs": sum(a == b for a, b, _ in group),
                    "agreement_rate": (
                        sum(a == b for a, b, _ in group) / len(group)
                    ),
                }
                for confidence in sorted({value[2] for value in comparable})
                for group in [
                    [value for value in comparable if value[2] == confidence]
                ]
            },
        }
    return {
        "status": "DESCRIPTIVE_ONLY_REQUIRES_RESEARCHER_SIGNOFF",
        "dimensions": dimensions,
        "automatic_threshold_pre_registered": False,
        "automatic_gold": False,
        "may_auto_promote_labeler": False,
    }


def aggregate_strict_bakeoff_qualification(
    *,
    result_rows: Sequence[Mapping[str, Any]],
    call_plan_rows: Sequence[Mapping[str, Any]],
    gpt_anchor_rows: Sequence[Mapping[str, Any]],
    human_normalized_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the pre-outcome criteria frozen in the bake-off contract.

    Metric semantics deliberately inherit the already-existing V2 qualification
    aggregation: AB/BA consistency is required on every preference dimension;
    informative rate and anchor-direction agreement use overall preference.
    Human agreement remains descriptive and cannot automatically promote a
    bulk labeler.
    """

    if str(contract.get("protocol") or "") != (
        "pm-v1.5-train-only-strict-pairwise-judge-bakeoff-v1"
    ):
        raise RuntimeError("unexpected strict bake-off contract protocol")
    if tuple(contract.get("candidate_keys") or ()) != EXPECTED_CANDIDATES:
        raise RuntimeError("strict bake-off contract candidate set drifted")
    if int(contract.get("pair_count") or 0) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("strict bake-off contract pair count drifted")

    rows = _validated_result_rows(
        result_rows=result_rows,
        call_plan_rows=call_plan_rows,
    )
    pair_ids = {
        str(dict(row["record_ids"])["pair_id"]) for row in rows
    }
    if len(pair_ids) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("strict bake-off pair coverage drifted")
    gpt_rows = _validated_gpt_anchor_rows(
        gpt_anchor_rows,
        eligible_pair_ids=pair_ids,
    )

    candidate_reports: dict[str, Any] = {}
    candidate_pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in EXPECTED_CANDIDATES:
        report, pairs = _pairwise_condition_report(
            rows, condition=candidate
        )
        candidate_reports[candidate] = report
        candidate_pairs[candidate] = pairs
    gpt_report, gpt_pairs = _pairwise_condition_report(
        gpt_rows, condition="gpt_anchor"
    )
    gpt_pair_ids = set(gpt_pairs)

    thresholds = dict(contract.get("pre_outcome_criteria") or {})
    schema_threshold = float(thresholds["required_schema_valid_rate"])
    order_threshold = float(
        thresholds["minimum_ab_ba_order_consistency"]
    )
    informative_threshold = float(
        thresholds["minimum_pairwise_informative_rate"]
    )
    gpt_threshold = float(
        thresholds["minimum_gpt_anchor_direction_agreement_secondary"]
    )

    gpt_agreement: dict[str, Any] = {}
    human_agreement: dict[str, Any] = {}
    automatic_checks: dict[str, Any] = {}
    for candidate in EXPECTED_CANDIDATES:
        gpt_agreement[candidate] = _direction_agreement(
            candidate_pairs[candidate],
            gpt_pairs,
            eligible_pair_ids=gpt_pair_ids,
        )
        human_agreement[candidate] = _human_agreement(
            candidate_pairs=candidate_pairs[candidate],
            human_rows=human_normalized_rows,
        )
        dimensions = candidate_reports[candidate]["dimensions"]
        gpt_overall = gpt_agreement[candidate]["dimensions"][
            "overall_preference"
        ]["agreement_rate"]
        checks = {
            "schema_valid_rate": {
                "value": 1.0,
                "threshold": schema_threshold,
                "pass": 1.0 >= schema_threshold,
            },
            "all_dimension_ab_ba_order_consistency": {
                "values": {
                    field: float(values["order_consistency_rate"])
                    for field, values in dimensions.items()
                },
                "threshold": order_threshold,
                "pass": all(
                    float(values["order_consistency_rate"])
                    >= order_threshold
                    for values in dimensions.values()
                ),
            },
            "overall_pairwise_informative_rate": {
                "value": float(
                    dimensions["overall_preference"]["informative_rate"]
                ),
                "threshold": informative_threshold,
                "pass": float(
                    dimensions["overall_preference"]["informative_rate"]
                )
                >= informative_threshold,
            },
            "overall_gpt_anchor_direction_agreement_secondary": {
                "value": gpt_overall,
                "comparable_pairs": gpt_agreement[candidate]["dimensions"][
                    "overall_preference"
                ]["comparable_pairs"],
                "threshold": gpt_threshold,
                "pass": (
                    gpt_overall is not None
                    and float(gpt_overall) >= gpt_threshold
                ),
            },
        }
        automatic_checks[candidate] = {
            **checks,
            "all_pre_outcome_automatic_checks_pass": all(
                bool(value["pass"]) for value in checks.values()
            ),
        }

    cross_candidate = {
        f"{left}__vs__{right}": _direction_agreement(
            candidate_pairs[left], candidate_pairs[right]
        )
        for index, left in enumerate(EXPECTED_CANDIDATES)
        for right in EXPECTED_CANDIDATES[index + 1 :]
    }
    automatically_supported = [
        candidate
        for candidate in EXPECTED_CANDIDATES
        if automatic_checks[candidate][
            "all_pre_outcome_automatic_checks_pass"
        ]
    ]
    status = (
        "AUTOMATIC_CRITERIA_PASS_REQUIRES_RESEARCHER_SIGNOFF"
        if automatically_supported
        else "JUDGE_QUALIFICATION_NOT_SUPPORTED"
    )
    return {
        "protocol": STRICT_BAKEOFF_AGGREGATION_PROTOCOL,
        "status": status,
        "metric_semantics": {
            "source": (
                "pre-existing v1_5_judge_qualification_analysis semantics "
                "that predate strict-bakeoff outcomes"
            ),
            "all_dimensions_used_for_order_consistency": True,
            "overall_preference_used_for_informative_rate": True,
            "overall_preference_used_for_gpt_anchor_gate": True,
            "human_anchor_is_descriptive_only": True,
        },
        "candidate_reports": candidate_reports,
        "gpt_anchor_report": gpt_report,
        "gpt_anchor_direction_agreement": gpt_agreement,
        "human_anchor_agreement": human_agreement,
        "cross_candidate_direction_agreement_diagnostic": cross_candidate,
        "automatic_checks": automatic_checks,
        "automatically_supported_candidates": automatically_supported,
        "researcher_signoff_required": True,
        "automatic_promotion": False,
        "majority_vote_is_gold": False,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
    }
