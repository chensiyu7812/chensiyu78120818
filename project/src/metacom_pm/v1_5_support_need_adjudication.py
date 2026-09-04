"""Bind, adjudicate, and normalize two reviews of SupportNeed fit anchors.

The two review files remain immutable evidence.  Adjudication creates one
state-level target per blind item, records every disagreement, and never turns
two reviews of the same dialogue into two independent training groups.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json, iter_jsonl, sha256_file, sha256_text
from .text import normalize_space
from .v1_5_support_need import explicit_support_boundaries
from .v1_5_support_need_packet import (
    HUMAN_BOUNDARY_FIELDS,
    SupportNeedHumanAnnotationV2,
    validate_support_need_expansion_annotations,
)


ADJUDICATION_PROTOCOL = "pm-v1.5-support-need-fit-adjudication-v1"
ADJUDICATION_CONFIG_PROTOCOL = (
    "pm-v1.5-support-need-fit-adjudication-decisions-v1"
)
COMBINED_ANCHOR_PROTOCOL = (
    "pm-v1.5-support-need-combined-fit-anchors-v1"
)
SCALAR_FIELDS = (
    "support_mode",
    "dialogue_phase",
    "nonclinical_urgency",
    "recommended_response_burden",
)
COMPARISON_FIELDS = (*SCALAR_FIELDS, "goals")
GOAL_IDS = ("be_heard", "make_sense", "stabilize", "decide", "act")


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _by_id(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result = {str(row["blind_item_id"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{label} contains duplicate blind item IDs")
    return result


def _categorical_distribution(values: Sequence[str]) -> dict[str, float]:
    counts = Counter(str(value) for value in values)
    total = float(len(values))
    return {
        value: float(count / total)
        for value, count in sorted(counts.items())
    }


def _categorical_kappa(
    first: Sequence[str], second: Sequence[str]
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("categorical kappa inputs are invalid")
    categories = sorted(set(first) | set(second))
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[value] / len(first))
        * (second_counts[value] / len(second))
        for value in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


def _evidence_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row["boundary_type"]),
        normalize_space(row["exact_user_quote"]),
    )


def _validate_visible_quotes(
    *,
    annotation: SupportNeedHumanAnnotationV2,
    visible_state: Mapping[str, Any],
) -> None:
    visible_user_texts = [
        normalize_space(visible_state.get("current_user_text") or "")
    ] + [
        normalize_space(turn.get("content") or "")
        for turn in visible_state.get("recent_dialogue") or []
        if turn.get("role") == "user"
    ]
    for evidence in annotation.active_explicit_boundary_evidence:
        quote = normalize_space(evidence.exact_user_quote)
        if not any(quote in text for text in visible_user_texts):
            raise RuntimeError(
                "adjudicated boundary quote is not an exact visible user "
                f"substring: {annotation.blind_item_id}"
            )


def build_support_need_fit_adjudication(
    *,
    packet_dir: str | Path,
    rater_a_path: str | Path,
    rater_b_path: str | Path,
    decisions_path: str | Path,
) -> dict[str, Any]:
    """Create one audited annotation per fit item from two review files."""

    packet_dir = Path(packet_dir)
    rater_a_path = Path(rater_a_path)
    rater_b_path = Path(rater_b_path)
    decisions_path = Path(decisions_path)
    packet_contract = json.loads(
        (packet_dir / "qualification_contract.json").read_text(
            encoding="utf-8"
        )
    )
    packet_contract_core = dict(packet_contract)
    packet_contract_sha256 = str(
        packet_contract_core.pop("contract_sha256", "")
    )
    if packet_contract_sha256 != sha256_text(
        canonical_json(packet_contract_core)
    ):
        raise RuntimeError("support-need fit packet contract hash is invalid")
    if (
        packet_contract.get("confirmation_rows_exposed") is not False
        or int(packet_contract.get("packet_size", -1))
        != int(packet_contract.get("fit_anchor_count", -2))
    ):
        raise RuntimeError("support-need fit packet does not remain sealed")
    packet_rows = _rows(packet_dir / "human_blind_packet.jsonl")
    packet_by_id = _by_id(packet_rows, label="support-need fit packet")

    rater_validations = {
        "rater_a": validate_support_need_expansion_annotations(
            packet_dir=packet_dir,
            annotations_path=rater_a_path,
        ),
        "rater_b": validate_support_need_expansion_annotations(
            packet_dir=packet_dir,
            annotations_path=rater_b_path,
        ),
    }
    rater_a_rows = _rows(rater_a_path)
    rater_b_rows = _rows(rater_b_path)
    rater_a_by_id = _by_id(rater_a_rows, label="rater A")
    rater_b_by_id = _by_id(rater_b_rows, label="rater B")
    if set(rater_a_by_id) != set(packet_by_id) or set(
        rater_b_by_id
    ) != set(packet_by_id):
        raise RuntimeError("support-need rater coverage does not match packet")

    config = json.loads(decisions_path.read_text(encoding="utf-8"))
    if config.get("protocol") != ADJUDICATION_CONFIG_PROTOCOL:
        raise RuntimeError("support-need adjudication config protocol drifted")
    expected_hashes = config.get("expected_source_file_sha256", {})
    if (
        expected_hashes.get("rater_a") != sha256_file(rater_a_path)
        or expected_hashes.get("rater_b") != sha256_file(rater_b_path)
        or config.get("packet_contract_sha256")
        != packet_contract_sha256
    ):
        raise RuntimeError("support-need adjudication input hash drifted")
    decision_rows = [dict(row) for row in config.get("adjudications", [])]
    decision_by_id = _by_id(
        decision_rows, label="support-need adjudication decisions"
    )
    if set(decision_by_id) != set(packet_by_id):
        raise RuntimeError(
            "support-need adjudication decisions do not exactly cover packet"
        )

    ordered_ids = [str(row["blind_item_id"]) for row in packet_rows]
    adjudicated_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for blind_item_id in ordered_ids:
        first = SupportNeedHumanAnnotationV2.model_validate(
            rater_a_by_id[blind_item_id]
        ).model_dump(mode="json")
        second = SupportNeedHumanAnnotationV2.model_validate(
            rater_b_by_id[blind_item_id]
        ).model_dump(mode="json")
        decision = decision_by_id[blind_item_id]
        final = SupportNeedHumanAnnotationV2.model_validate(
            decision.get("final_annotation")
        )
        if final.blind_item_id != blind_item_id or final.abstain:
            raise RuntimeError(
                "support-need adjudication ID/abstain decision is invalid"
            )
        final_row = final.model_dump(mode="json")
        for field in SCALAR_FIELDS:
            if final_row[field] not in {first[field], second[field]}:
                raise RuntimeError(
                    "adjudicated scalar must come from a source review: "
                    f"{blind_item_id}/{field}"
                )
        if frozenset(final_row["goals"]) not in {
            frozenset(first["goals"]),
            frozenset(second["goals"]),
        }:
            raise RuntimeError(
                "adjudicated goals must come from a source review: "
                f"{blind_item_id}"
            )
        source_evidence = {
            _evidence_key(row)
            for row in (
                first["active_explicit_boundary_evidence"]
                + second["active_explicit_boundary_evidence"]
            )
        }
        final_evidence = {
            _evidence_key(row)
            for row in final_row["active_explicit_boundary_evidence"]
        }
        if not final_evidence <= source_evidence:
            raise RuntimeError(
                "adjudication cannot invent explicit boundary evidence: "
                f"{blind_item_id}"
            )
        visible_state = dict(packet_by_id[blind_item_id]["visible_state"])
        _validate_visible_quotes(
            annotation=final,
            visible_state=visible_state,
        )
        disagreement_fields = [
            field
            for field in COMPARISON_FIELDS
            if (
                frozenset(first[field]) != frozenset(second[field])
                if field == "goals"
                else first[field] != second[field]
            )
        ]
        first_evidence = {
            _evidence_key(row)
            for row in first["active_explicit_boundary_evidence"]
        }
        second_evidence = {
            _evidence_key(row)
            for row in second["active_explicit_boundary_evidence"]
        }
        if first_evidence != second_evidence:
            disagreement_fields.append("active_explicit_boundary_evidence")
        trace_core = {
            "protocol": ADJUDICATION_PROTOCOL,
            "blind_item_id": blind_item_id,
            "rater_a_annotation": first,
            "rater_b_annotation": second,
            "final_annotation": final_row,
            "disagreement_fields": disagreement_fields,
            "pre_adjudication_distributions": {
                "support_mode": _categorical_distribution(
                    [first["support_mode"], second["support_mode"]]
                ),
                "dialogue_phase": _categorical_distribution(
                    [
                        first["dialogue_phase"],
                        second["dialogue_phase"],
                    ]
                ),
                "nonclinical_urgency": _categorical_distribution(
                    [
                        first["nonclinical_urgency"],
                        second["nonclinical_urgency"],
                    ]
                ),
                "recommended_response_burden": _categorical_distribution(
                    [
                        first["recommended_response_burden"],
                        second["recommended_response_burden"],
                    ]
                ),
                "goal_probabilities": {
                    goal: float(
                        (
                            int(goal in first["goals"])
                            + int(goal in second["goals"])
                        )
                        / 2.0
                    )
                    for goal in GOAL_IDS
                },
            },
            "source_evidence_union": [
                {
                    "boundary_type": boundary_type,
                    "exact_user_quote": quote,
                }
                for boundary_type, quote in sorted(source_evidence)
            ],
            "final_evidence": [
                {
                    "boundary_type": boundary_type,
                    "exact_user_quote": quote,
                }
                for boundary_type, quote in sorted(final_evidence)
            ],
            "removed_evidence": [
                {
                    "boundary_type": boundary_type,
                    "exact_user_quote": quote,
                }
                for boundary_type, quote in sorted(
                    source_evidence - final_evidence
                )
            ],
            "adjudication_rationale": normalize_space(
                decision.get("adjudication_rationale") or ""
            ),
            "independent_dialogue_group_mass": 1,
            "review_rows_do_not_increase_ess": True,
        }
        if not trace_core["adjudication_rationale"]:
            raise RuntimeError(
                f"support-need adjudication rationale missing: {blind_item_id}"
            )
        trace = {
            **trace_core,
            "trace_sha256": sha256_text(canonical_json(trace_core)),
        }
        trace_rows.append(trace)
        adjudicated_rows.append(final_row)

        boundaries = explicit_support_boundaries(
            str(visible_state["current_user_text"])
        ).model_dump(mode="json")
        burden = str(final_row["recommended_response_burden"])
        normalized_core = {
            "protocol": (
                "pm-v1.5-support-need-fit-anchor-normalized-v1"
            ),
            "blind_item_id": blind_item_id,
            "visible_state_sha256": sha256_text(
                canonical_json(visible_state)
            ),
            "raw_annotation": final_row,
            "deterministic_current_turn_explicit_boundaries": boundaries,
            "adjudicated_explicit_boundary_evidence": final_row[
                "active_explicit_boundary_evidence"
            ],
            "human_recommended_low_interaction_burden": (
                burden in {"minimal_presence", "one_focus"}
            ),
            "recommended_response_burden": burden,
            "pre_adjudication_distributions": trace_core[
                "pre_adjudication_distributions"
            ],
            "adjudication_disagreement_fields": disagreement_fields,
            "adjudication_trace_sha256": trace["trace_sha256"],
            "semantic_roles": {
                "deterministic_current_turn_explicit_boundaries": (
                    "observable_pm_input_not_label"
                ),
                "adjudicated_explicit_boundary_evidence": (
                    "human_reviewed_visible_user_evidence_not_action_gold"
                ),
                "support_mode_goals_phase_urgency_burden": (
                    "adjudicated_human_partial_labels_not_pm_action_gold"
                ),
                "pre_adjudication_distributions": (
                    "disagreement_provenance_not_additional_groups"
                ),
            },
            "automatic_gold_label": False,
            "pm_action_gold": False,
            "internal_test_outcomes_opened": False,
            "external_outcomes_opened": False,
        }
        normalized_rows.append(
            {
                **normalized_core,
                "normalized_row_sha256": sha256_text(
                    canonical_json(normalized_core)
                ),
            }
        )

    agreement: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        first_values = [str(rater_a_by_id[row_id][field]) for row_id in ordered_ids]
        second_values = [
            str(rater_b_by_id[row_id][field]) for row_id in ordered_ids
        ]
        agree = sum(
            first_value == second_value
            for first_value, second_value in zip(
                first_values, second_values
            )
        )
        agreement[field] = {
            "agreement_count": agree,
            "agreement_rate": float(agree / len(ordered_ids)),
            "cohen_kappa": _categorical_kappa(
                first_values, second_values
            ),
        }
    exact_goal_agreement = sum(
        frozenset(rater_a_by_id[row_id]["goals"])
        == frozenset(rater_b_by_id[row_id]["goals"])
        for row_id in ordered_ids
    )
    agreement["goals_exact_set"] = {
        "agreement_count": exact_goal_agreement,
        "agreement_rate": float(exact_goal_agreement / len(ordered_ids)),
    }
    report_core = {
        "protocol": ADJUDICATION_PROTOCOL,
        "status": "COMPLETE_ADJUDICATED_FIT_ANCHORS_NOT_AUTOMATIC_GOLD",
        "packet_contract_sha256": packet_contract_sha256,
        "source_review_file_sha256": {
            "rater_a": sha256_file(rater_a_path),
            "rater_b": sha256_file(rater_b_path),
        },
        "source_review_binding_sha256": {
            key: value["binding_sha256"]
            for key, value in rater_validations.items()
        },
        "decisions_file_sha256": sha256_file(decisions_path),
        "annotation_count": len(adjudicated_rows),
        "source_review_row_count": len(rater_a_rows) + len(rater_b_rows),
        "independent_dialogue_groups": len(adjudicated_rows),
        "source_review_rows_do_not_increase_ess": True,
        "items_with_any_disagreement": sum(
            bool(row["disagreement_fields"]) for row in trace_rows
        ),
        "source_evidence_count": sum(
            len(row["source_evidence_union"]) for row in trace_rows
        ),
        "final_evidence_count": sum(
            len(row["final_evidence"]) for row in trace_rows
        ),
        "removed_evidence_count": sum(
            len(row["removed_evidence"]) for row in trace_rows
        ),
        "inter_rater_agreement": agreement,
        "adjudicated_annotations_sha256": sha256_text(
            canonical_json(adjudicated_rows)
        ),
        "adjudication_trace_sha256": sha256_text(
            canonical_json(trace_rows)
        ),
        "normalized_fit_anchors_sha256": sha256_text(
            canonical_json(normalized_rows)
        ),
        "confirmation_rows_exposed": False,
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        "rater_a_rows": rater_a_rows,
        "rater_b_rows": rater_b_rows,
        "adjudicated_rows": adjudicated_rows,
        "trace_rows": trace_rows,
        "normalized_rows": normalized_rows,
        "report": {
            **report_core,
            "report_sha256": sha256_text(canonical_json(report_core)),
        },
    }


def combine_support_need_fit_anchors(
    *,
    initial_packet_dir: str | Path,
    initial_normalized_path: str | Path,
    initial_normalization_binding_path: str | Path,
    fit_packet_rows: Sequence[Mapping[str, Any]],
    fit_normalized_rows: Sequence[Mapping[str, Any]],
    adjudication_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine 23+1 initial rows with 16 fit rows without opening confirmation."""

    initial_packet_dir = Path(initial_packet_dir)
    initial_normalized_path = Path(initial_normalized_path)
    initial_normalization_binding_path = Path(
        initial_normalization_binding_path
    )
    initial_contract = json.loads(
        (initial_packet_dir / "qualification_contract.json").read_text(
            encoding="utf-8"
        )
    )
    initial_contract_core = dict(initial_contract)
    initial_contract_sha256 = str(
        initial_contract_core.pop("contract_sha256", "")
    )
    if initial_contract_sha256 != sha256_text(
        canonical_json(initial_contract_core)
    ):
        raise RuntimeError("initial support-need packet contract is invalid")
    initial_packet_rows = _rows(
        initial_packet_dir / "human_blind_packet.jsonl"
    )
    if (
        sha256_text(canonical_json(initial_packet_rows))
        != initial_contract.get("human_anchor_packet_sha256")
    ):
        raise RuntimeError("initial support-need packet drifted")
    initial_normalized_rows = _rows(initial_normalized_path)
    normalization_binding = json.loads(
        initial_normalization_binding_path.read_text(encoding="utf-8")
    )
    binding_core = dict(normalization_binding)
    binding_sha256 = str(
        binding_core.pop("normalization_binding_sha256", "")
    )
    if (
        binding_sha256 != sha256_text(canonical_json(binding_core))
        or sha256_text(canonical_json(initial_normalized_rows))
        != normalization_binding.get("normalized_rows_sha256")
    ):
        raise RuntimeError("initial normalized support-need anchors drifted")
    initial_packet_by_id = _by_id(
        initial_packet_rows, label="initial support-need packet"
    )
    initial_normalized_by_id = _by_id(
        initial_normalized_rows, label="initial normalized anchors"
    )
    fit_packet_by_id = _by_id(
        fit_packet_rows, label="fit support-need packet"
    )
    fit_normalized_by_id = _by_id(
        fit_normalized_rows, label="fit normalized anchors"
    )
    if set(initial_packet_by_id) != set(initial_normalized_by_id):
        raise RuntimeError("initial packet/normalized anchor IDs differ")
    if set(fit_packet_by_id) != set(fit_normalized_by_id):
        raise RuntimeError("fit packet/normalized anchor IDs differ")
    if set(initial_packet_by_id) & set(fit_packet_by_id):
        raise RuntimeError("initial and fit support-need IDs overlap")
    if (
        adjudication_report.get("confirmation_rows_exposed") is not False
        or adjudication_report.get("status")
        != "COMPLETE_ADJUDICATED_FIT_ANCHORS_NOT_AUTOMATIC_GOLD"
    ):
        raise RuntimeError("fit adjudication is not eligible for combination")
    combined_packet_rows = [
        *initial_packet_rows,
        *[dict(row) for row in fit_packet_rows],
    ]
    combined_normalized_rows = [
        *initial_normalized_rows,
        *[dict(row) for row in fit_normalized_rows],
    ]
    non_abstaining = sum(
        not bool(row["raw_annotation"]["abstain"])
        for row in combined_normalized_rows
    )
    report_core = {
        "protocol": COMBINED_ANCHOR_PROTOCOL,
        "status": "COMPLETE_TRAIN_ONLY_INITIAL_PLUS_FIT_CONFIRMATION_SEALED",
        "initial_packet_contract_sha256": initial_contract_sha256,
        "initial_normalization_binding_sha256": binding_sha256,
        "initial_normalized_file_sha256": sha256_file(
            initial_normalized_path
        ),
        "fit_adjudication_report_sha256": adjudication_report[
            "report_sha256"
        ],
        "initial_packet_rows": len(initial_packet_rows),
        "fit_packet_rows": len(fit_packet_rows),
        "combined_packet_rows": len(combined_packet_rows),
        "non_abstaining_anchor_count": non_abstaining,
        "independent_dialogue_groups": non_abstaining,
        "source_review_rows_do_not_increase_ess": True,
        "combined_packet_sha256": sha256_text(
            canonical_json(combined_packet_rows)
        ),
        "combined_normalized_anchors_sha256": sha256_text(
            canonical_json(combined_normalized_rows)
        ),
        "confirmation_rows_exposed": False,
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        "combined_packet_rows": combined_packet_rows,
        "combined_normalized_rows": combined_normalized_rows,
        "report": {
            **report_core,
            "report_sha256": sha256_text(canonical_json(report_core)),
        },
    }
