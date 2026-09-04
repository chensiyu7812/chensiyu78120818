from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActionOutcome, MemoryBackendRecord
from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import SemanticTextEncoder
from .pm_v2_contracts import PMV2State
from .prompts import common_context
from .text import jaccard, lexical_score, normalize_space, shingle_set
from .v1_5_oracle_memory_pilot import (
    CONTROL_ARM,
    HARMFUL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_PROTOCOL,
)


ORACLE_MEMORY_UPTAKE_PROTOCOL = (
    "pm-v1.5-longitudinal-oracle-memory-uptake-analysis-v1"
)


def _encode_lookup(
    encoder: SemanticTextEncoder, texts: Sequence[str]
) -> dict[str, np.ndarray]:
    distinct = sorted({normalize_space(text) for text in texts})
    if not distinct or any(not text for text in distinct):
        raise RuntimeError("oracle-memory analysis cannot encode empty text")
    matrix = encoder.encode(distinct)
    if matrix.shape != (len(distinct), encoder.spec.output_dimension):
        raise RuntimeError("oracle-memory semantic matrix shape drifted")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("oracle-memory semantic matrix is non-finite")
    return dict(zip(distinct, matrix))


def _cosine(vectors: Mapping[str, np.ndarray], left: str, right: str) -> float:
    return float(
        np.clip(
            vectors[normalize_space(left)] @ vectors[normalize_space(right)],
            -1.0,
            1.0,
        )
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["target_evidence_bge_cosine_delta"]) for row in rows]
    lexical = [
        float(row["target_evidence_lexical_cosine_delta"]) for row in rows
    ]
    return {
        "pairs": len(rows),
        "mean_target_evidence_bge_cosine_delta": float(np.mean(deltas)),
        "positive_pairs": sum(value > 0.0 for value in deltas),
        "negative_pairs": sum(value < 0.0 for value in deltas),
        "zero_pairs": sum(value == 0.0 for value in deltas),
        "nonpositive_pairs": sum(value <= 0.0 for value in deltas),
        "mean_target_evidence_lexical_cosine_delta": float(np.mean(lexical)),
        "mean_visible_dialogue_bge_cosine_delta": float(
            np.mean(
                [
                    float(row["visible_dialogue_bge_cosine_delta"])
                    for row in rows
                ]
            )
        ),
        "maximum_treatment_evidence_5gram_jaccard": max(
            float(row["treatment_evidence_5gram_jaccard"]) for row in rows
        ),
    }


def _status(
    helpful_summary: Mapping[str, Any],
    harmful_summary: Mapping[str, Any],
) -> tuple[str, bool, bool]:
    helpful_pass = (
        int(helpful_summary["pairs"]) == 12
        and float(helpful_summary["mean_target_evidence_bge_cosine_delta"]) > 0.0
        and int(helpful_summary["positive_pairs"]) >= 8
    )
    harmful_pass = (
        int(harmful_summary["pairs"]) == 6
        and float(harmful_summary["mean_target_evidence_bge_cosine_delta"]) <= 0.0
        and int(harmful_summary["nonpositive_pairs"]) >= 4
    )
    if helpful_pass and harmful_pass:
        status = "ORACLE_MEMORY_UPTAKE_CAPACITY_SUPPORTED_REPORT_ONLY"
    elif helpful_pass:
        status = "ORACLE_MEMORY_UPTAKE_NONSELECTIVE_REPORT_ONLY"
    else:
        status = "ORACLE_MEMORY_UPTAKE_CAPACITY_NOT_ESTABLISHED_REPORT_ONLY"
    return status, helpful_pass, harmful_pass


def analyze_oracle_memory_uptake(
    *,
    pilot_contract: Mapping[str, Any],
    states: Sequence[PMV2State],
    backend_by_card: Mapping[str, MemoryBackendRecord],
    outcomes: Sequence[ActionOutcome],
    encoder: SemanticTextEncoder,
) -> dict[str, Any]:
    if pilot_contract.get("protocol") != ORACLE_MEMORY_PILOT_PROTOCOL:
        raise RuntimeError("unexpected oracle-memory pilot protocol")
    contract_sha = pilot_contract.get("contract_sha256")
    without_sha = {
        key: value
        for key, value in pilot_contract.items()
        if key != "contract_sha256"
    }
    if contract_sha != sha256_text(canonical_json(without_sha)):
        raise RuntimeError("oracle-memory pilot contract hash mismatch")
    selected = {
        str(row["state_id"]): dict(row)
        for row in pilot_contract.get("selected_states") or []
    }
    if len(selected) != 18:
        raise RuntimeError("oracle-memory analysis requires 18 selected states")
    state_by_id = {state.state_id: state for state in states}
    outcome_by_key: dict[tuple[str, str], ActionOutcome] = {}
    for outcome in outcomes:
        arm = str(outcome.provenance.get("oracle_memory_pilot_arm") or "")
        key = (outcome.state_id, arm)
        if key in outcome_by_key:
            raise RuntimeError(f"duplicate oracle-memory outcome: {key}")
        outcome_by_key[key] = outcome
    expected_keys = {
        (state_id, CONTROL_ARM) for state_id in selected
    } | {
        (state_id, str(row["treatment_arm"]))
        for state_id, row in selected.items()
    }
    if set(outcome_by_key) != expected_keys:
        raise RuntimeError(
            "oracle-memory outcome coverage mismatch: "
            f"missing={sorted(expected_keys-set(outcome_by_key))}, "
            f"extra={sorted(set(outcome_by_key)-expected_keys)}"
        )

    prepared: list[dict[str, Any]] = []
    all_texts: list[str] = []
    for state_id, selected_row in selected.items():
        state = state_by_id.get(state_id)
        if state is None or state.split.value != "train":
            raise RuntimeError(f"oracle-memory state is absent/non-train: {state_id}")
        backend = backend_by_card.get(state.card_id)
        if backend is None:
            raise RuntimeError(f"oracle-memory backend is absent: {state.card_id}")
        control = outcome_by_key[(state_id, CONTROL_ARM)]
        arm = str(selected_row["treatment_arm"])
        treatment = outcome_by_key[(state_id, arm)]
        if control.action_id != "M0+R0":
            raise RuntimeError(f"oracle-memory control action drifted: {state_id}")
        if treatment.action_id != str(selected_row["target_action_id"]):
            raise RuntimeError(f"oracle-memory treatment action drifted: {state_id}")
        if control.memory_view or control.strategy_view or treatment.strategy_view:
            raise RuntimeError(f"oracle-memory evidence surface drifted: {state_id}")
        target_ids = set(
            str(value) for value in selected_row["target_memory_ids"]
        )
        if set(treatment.selected_memory_ids) != target_ids:
            raise RuntimeError(f"oracle-memory target coverage drifted: {state_id}")
        if {
            item.memory_id for item in backend.items
        } != target_ids:
            raise RuntimeError(f"oracle-memory backend target set drifted: {state_id}")
        for key in (
            "supporter_generation_treatment_sha256",
            "temperature",
            "seed",
        ):
            if control.provenance.get(key) != treatment.provenance.get(key):
                raise RuntimeError(
                    f"oracle-memory generation treatment drifted: {state_id}/{key}"
                )
        if control.model_name != treatment.model_name:
            raise RuntimeError(f"oracle-memory model drifted: {state_id}")
        reference = normalize_space(
            "\n".join(
                item.text
                for item in sorted(
                    backend.items,
                    key=lambda item: (item.source.value, item.memory_id),
                )
            )
        )
        visible = common_context(state)
        all_texts.extend([reference, visible, control.response, treatment.response])
        prepared.append(
            {
                "state_id": state_id,
                "user_id": state.user_id,
                "regime": selected_row["regime"],
                "treatment_arm": arm,
                "target_item_utility": selected_row["target_item_utility"],
                "reference": reference,
                "visible": visible,
                "control": control,
                "treatment": treatment,
                "target_ids": target_ids,
            }
        )
    vectors = _encode_lookup(encoder, all_texts)
    pair_rows: list[dict[str, Any]] = []
    for row in prepared:
        control = row["control"]
        treatment = row["treatment"]
        reference = row["reference"]
        visible = row["visible"]
        control_semantic = _cosine(vectors, control.response, reference)
        treatment_semantic = _cosine(vectors, treatment.response, reference)
        control_lexical = lexical_score(reference, control.response)
        treatment_lexical = lexical_score(reference, treatment.response)
        pair_rows.append(
            {
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "regime": row["regime"],
                "treatment_arm": row["treatment_arm"],
                "target_item_utility": row["target_item_utility"],
                "target_reference_sha256": sha256_text(reference),
                "control_response_sha256": sha256_text(control.response),
                "treatment_response_sha256": sha256_text(treatment.response),
                "control_target_evidence_bge_cosine": control_semantic,
                "treatment_target_evidence_bge_cosine": treatment_semantic,
                "target_evidence_bge_cosine_delta": (
                    treatment_semantic - control_semantic
                ),
                "control_target_evidence_lexical_cosine": control_lexical,
                "treatment_target_evidence_lexical_cosine": treatment_lexical,
                "target_evidence_lexical_cosine_delta": (
                    treatment_lexical - control_lexical
                ),
                "visible_dialogue_bge_cosine_delta": (
                    _cosine(vectors, treatment.response, visible)
                    - _cosine(vectors, control.response, visible)
                ),
                "control_evidence_5gram_jaccard": jaccard(
                    shingle_set(control.response),
                    shingle_set(reference),
                ),
                "treatment_evidence_5gram_jaccard": jaccard(
                    shingle_set(treatment.response),
                    shingle_set(reference),
                ),
                "treatment_selected_memory_ids": sorted(
                    treatment.selected_memory_ids
                ),
                "annotated_target_memory_ids": sorted(row["target_ids"]),
            }
        )
    helpful = [
        row for row in pair_rows if row["treatment_arm"] == HELPFUL_ARM
    ]
    harmful = [
        row for row in pair_rows if row["treatment_arm"] == HARMFUL_ARM
    ]
    if len(helpful) != 12 or len(harmful) != 6:
        raise RuntimeError("oracle-memory helpful/harmful balance drifted")
    helpful_summary = _summary(helpful)
    harmful_summary = _summary(harmful)
    status, helpful_pass, harmful_pass = _status(
        helpful_summary, harmful_summary
    )
    return {
        "protocol": ORACLE_MEMORY_UPTAKE_PROTOCOL,
        "status": status,
        "pilot_contract_sha256": contract_sha,
        "semantic_encoder_spec_sha256": encoder.spec.digest(),
        "pairs": len(pair_rows),
        "helpful": helpful_summary,
        "harmful": harmful_summary,
        "helpful_rule_passed": helpful_pass,
        "harmful_selectivity_guardrail_passed": harmful_pass,
        "pair_rows": pair_rows,
        "api_judges_used": False,
        "training_labels_created": False,
        "judge_qualification_authorized": False,
        "pm_training_authorized": False,
        "interpretation": (
            "Report-only oracle-evidence uptake upper bound. This result "
            "localizes a mechanism bottleneck but does not establish response "
            "quality, deployable routing, judge validity, or PM efficacy."
        ),
    }
