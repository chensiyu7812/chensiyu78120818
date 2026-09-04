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


UPTAKE_MEASUREMENT_PROTOCOL = (
    "pm-v1.5-longitudinal-response-mechanism-uptake-measurement-v1"
)
CONTROL_ARM = "frozen_current_control"
TREATMENT_ARM = "top1_evidence_surface_same_prompt"
HELPFUL_REGIMES = frozenset(
    {
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "strategy_helpful",
    }
)
HARMFUL_REGIMES = frozenset({"memory_harmful", "strategy_harmful"})


def uptake_measurement_contract(
    *,
    pilot_contract_sha256: str,
    semantic_encoder_spec_sha256: str,
) -> dict[str, Any]:
    payload = {
        "protocol": UPTAKE_MEASUREMENT_PROTOCOL,
        "status": "FROZEN_BEFORE_PILOT_OUTCOMES",
        "scope": "longitudinal_train_only_report_only",
        "pilot_contract_sha256": str(pilot_contract_sha256),
        "semantic_encoder_spec_sha256": str(semantic_encoder_spec_sha256),
        "paired_unit": "state_id",
        "arms": [CONTROL_ARM, TREATMENT_ARM],
        "reference_construction": {
            "memory_helpful_regimes": (
                "concatenate evaluator-annotated item_utility=helpful memory "
                "texts in (source,memory_id) order; identical reference for both arms"
            ),
            "memory_harmful_regime": (
                "concatenate evaluator-annotated item_utility=harmful memory "
                "texts in (source,memory_id) order; identical reference for both arms"
            ),
            "strategy_regimes": (
                "use the treatment arm's top-1 prompt-visible strategy guidance "
                "plus example as the identical reference for both arms"
            ),
            "visible_dialogue_guardrail": (
                "prompts.common_context(state), identical for both arms"
            ),
        },
        "primary_metric": {
            "name": "paired_target_evidence_bge_cosine_delta",
            "formula": (
                "cosine(treatment_response,target_reference)-"
                "cosine(control_response,target_reference)"
            ),
            "helpful_expected_direction": "positive",
            "harmful_expected_direction": "negative",
        },
        "supportive_metrics": [
            {
                "name": "paired_target_evidence_lexical_cosine_delta",
                "formula": (
                    "lexical_score(target_reference,treatment_response)-"
                    "lexical_score(target_reference,control_response)"
                ),
            },
            {
                "name": "paired_visible_dialogue_bge_cosine_delta",
                "role": "grounding_guardrail_report_only",
            },
            {
                "name": "response_target_evidence_5gram_jaccard",
                "role": "verbatim_copy_guardrail_report_only",
            },
            {
                "name": "selected_annotated_item_coverage",
                "role": "retrieval-versus-generator-localization",
            },
        ],
        "directional_signal_rule": {
            "helpful_pair_count": 10,
            "minimum_helpful_positive_pairs": 6,
            "helpful_mean_delta_must_be": "greater_than_zero",
            "harmful_pair_count": 4,
            "minimum_harmful_negative_pairs": 3,
            "harmful_mean_delta_must_be": "less_than_zero",
            "primary_metric_only": True,
            "lexical_and_guardrail_metrics": "reported_not_gating",
        },
        "allowed_statuses": [
            "MECHANISM_SIGNAL_PRESENT_REPORT_ONLY",
            "MECHANISM_SIGNAL_NOT_ESTABLISHED_REPORT_ONLY",
        ],
        "forbidden_consequences": [
            "judge qualification",
            "action label creation",
            "PM training authorization",
            "internal-test access",
            "external efficacy claim",
        ],
        "api_judges_used": False,
        "training_labels_created": False,
    }
    return {
        **payload,
        "contract_sha256": sha256_text(canonical_json(payload)),
    }


def _strategy_reference(outcome: ActionOutcome) -> str:
    if len(outcome.strategy_view) != 1:
        raise RuntimeError(
            f"strategy pilot treatment must expose exactly one strategy: "
            f"{outcome.state_id}"
        )
    card = outcome.strategy_view[0]
    return normalize_space(card.guidance_text + "\n" + card.example_response)


def _memory_reference(
    *,
    state_id: str,
    evaluator: Mapping[str, Any],
    backend: MemoryBackendRecord,
    utility: str,
) -> tuple[str, set[str]]:
    annotations = {
        str(row["memory_id"]): row
        for row in evaluator.get("memory_annotations") or []
        if str(row.get("item_utility")) == utility
    }
    items = [
        item
        for item in backend.items
        if item.memory_id in annotations
    ]
    items.sort(key=lambda item: (item.source.value, item.memory_id))
    if not items:
        raise RuntimeError(
            f"pilot state lacks annotated {utility} memory reference: {state_id}"
        )
    return (
        "\n".join(normalize_space(item.text) for item in items),
        {item.memory_id for item in items},
    )


def _encode_lookup(
    encoder: SemanticTextEncoder, texts: Sequence[str]
) -> dict[str, np.ndarray]:
    distinct = sorted({normalize_space(text) for text in texts})
    if not distinct or any(not text for text in distinct):
        raise RuntimeError("uptake diagnostic cannot encode empty text")
    matrix = encoder.encode(distinct)
    if matrix.shape != (len(distinct), encoder.spec.output_dimension):
        raise RuntimeError("uptake diagnostic semantic matrix shape drifted")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("uptake diagnostic semantic matrix is non-finite")
    return dict(zip(distinct, matrix))


def _cosine(vectors: Mapping[str, np.ndarray], left: str, right: str) -> float:
    return float(
        np.clip(
            vectors[normalize_space(left)] @ vectors[normalize_space(right)],
            -1.0,
            1.0,
        )
    )


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["target_evidence_bge_cosine_delta"]) for row in rows]
    lexical = [
        float(row["target_evidence_lexical_cosine_delta"]) for row in rows
    ]
    return {
        "pairs": len(rows),
        "mean_target_evidence_bge_cosine_delta": float(np.mean(deltas)),
        "positive_bge_pairs": sum(value > 0.0 for value in deltas),
        "negative_bge_pairs": sum(value < 0.0 for value in deltas),
        "zero_bge_pairs": sum(value == 0.0 for value in deltas),
        "mean_target_evidence_lexical_cosine_delta": float(np.mean(lexical)),
        "positive_lexical_pairs": sum(value > 0.0 for value in lexical),
        "negative_lexical_pairs": sum(value < 0.0 for value in lexical),
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


def _directional_status(
    helpful_summary: Mapping[str, Any],
    harmful_summary: Mapping[str, Any],
) -> str:
    signal = (
        int(helpful_summary["pairs"]) == 10
        and float(helpful_summary["mean_target_evidence_bge_cosine_delta"]) > 0.0
        and int(helpful_summary["positive_bge_pairs"]) >= 6
        and int(harmful_summary["pairs"]) == 4
        and float(harmful_summary["mean_target_evidence_bge_cosine_delta"]) < 0.0
        and int(harmful_summary["negative_bge_pairs"]) >= 3
    )
    return (
        "MECHANISM_SIGNAL_PRESENT_REPORT_ONLY"
        if signal
        else "MECHANISM_SIGNAL_NOT_ESTABLISHED_REPORT_ONLY"
    )


def analyze_uptake(
    *,
    measurement_contract: Mapping[str, Any],
    pilot_contract: Mapping[str, Any],
    states: Sequence[PMV2State],
    evaluator_by_state: Mapping[str, Mapping[str, Any]],
    backend_by_card: Mapping[str, MemoryBackendRecord],
    outcomes: Sequence[ActionOutcome],
    encoder: SemanticTextEncoder,
) -> dict[str, Any]:
    if measurement_contract.get("protocol") != UPTAKE_MEASUREMENT_PROTOCOL:
        raise RuntimeError("unexpected uptake measurement protocol")
    expected_measurement_sha = measurement_contract.get("contract_sha256")
    without_sha = {
        key: value
        for key, value in measurement_contract.items()
        if key != "contract_sha256"
    }
    if expected_measurement_sha != sha256_text(canonical_json(without_sha)):
        raise RuntimeError("uptake measurement contract hash mismatch")
    if measurement_contract.get("pilot_contract_sha256") != pilot_contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("uptake measurement/pilot contract mismatch")
    if measurement_contract.get(
        "semantic_encoder_spec_sha256"
    ) != encoder.spec.digest():
        raise RuntimeError("uptake measurement encoder binding mismatch")

    state_by_id = {state.state_id: state for state in states}
    selected = {
        str(row["state_id"]): dict(row)
        for row in pilot_contract.get("selected_states") or []
    }
    if len(selected) != 14:
        raise RuntimeError("uptake diagnostic requires exactly 14 selected states")
    outcome_by_key: dict[tuple[str, str], ActionOutcome] = {}
    for outcome in outcomes:
        arm = str(outcome.provenance.get("response_mechanism_pilot_arm") or "")
        key = (outcome.state_id, arm)
        if key in outcome_by_key:
            raise RuntimeError(f"duplicate pilot outcome: {key}")
        outcome_by_key[key] = outcome
    expected_keys = {
        (state_id, arm)
        for state_id in selected
        for arm in (CONTROL_ARM, TREATMENT_ARM)
    }
    if set(outcome_by_key) != expected_keys:
        raise RuntimeError(
            "pilot outcome key coverage mismatch: "
            f"missing={sorted(expected_keys-set(outcome_by_key))}, "
            f"extra={sorted(set(outcome_by_key)-expected_keys)}"
        )

    prepared: list[dict[str, Any]] = []
    all_texts: list[str] = []
    for state_id, selected_row in selected.items():
        state = state_by_id.get(state_id)
        if state is None:
            raise RuntimeError(f"pilot state is absent: {state_id}")
        if state.split.value != "train":
            raise RuntimeError(f"pilot state is not train-only: {state_id}")
        evaluator = evaluator_by_state.get(state_id)
        if evaluator is None:
            raise RuntimeError(f"pilot evaluator context is absent: {state_id}")
        backend = backend_by_card.get(state.card_id)
        if backend is None:
            raise RuntimeError(f"pilot backend is absent: {state.card_id}")
        regime = str(selected_row["regime"])
        control = outcome_by_key[(state_id, CONTROL_ARM)]
        treatment = outcome_by_key[(state_id, TREATMENT_ARM)]
        if (
            control.action_id != treatment.action_id
            or control.action_id != str(selected_row["target_action_id"])
        ):
            raise RuntimeError(f"paired pilot action drifted: {state_id}")
        for key in (
            "supporter_generation_treatment_sha256",
            "temperature",
            "seed",
        ):
            if control.provenance.get(key) != treatment.provenance.get(key):
                raise RuntimeError(
                    f"paired pilot generation treatment drifted: {state_id}/{key}"
                )
        if control.model_name != treatment.model_name:
            raise RuntimeError(f"paired pilot model drifted: {state_id}")
        memory_source_counts: dict[str, int] = defaultdict(int)
        for item in treatment.memory_view:
            memory_source_counts[item.source.value] += 1
        if (
            len(treatment.memory_view) > 3
            or any(value > 1 for value in memory_source_counts.values())
            or len(treatment.strategy_view) > 1
        ):
            raise RuntimeError(f"pilot top1 evidence surface drifted: {state_id}")
        if regime == "strategy_helpful" or regime == "strategy_harmful":
            reference = _strategy_reference(treatment)
            annotated_ids: set[str] = set()
            reference_kind = "top1_strategy_prompt_surface"
        elif regime == "memory_harmful":
            reference, annotated_ids = _memory_reference(
                state_id=state_id,
                evaluator=evaluator,
                backend=backend,
                utility="harmful",
            )
            reference_kind = "annotated_harmful_memory"
        else:
            reference, annotated_ids = _memory_reference(
                state_id=state_id,
                evaluator=evaluator,
                backend=backend,
                utility="helpful",
            )
            reference_kind = "annotated_helpful_memory"
        visible = common_context(state)
        all_texts.extend(
            [reference, visible, control.response, treatment.response]
        )
        prepared.append(
            {
                "state_id": state_id,
                "user_id": state.user_id,
                "regime": regime,
                "expected_direction": (
                    "positive" if regime in HELPFUL_REGIMES else "negative"
                ),
                "reference_kind": reference_kind,
                "reference": reference,
                "visible": visible,
                "annotated_ids": annotated_ids,
                "control": control,
                "treatment": treatment,
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
        selected_ids = set(treatment.selected_memory_ids)
        annotated_ids = set(row["annotated_ids"])
        pair_rows.append(
            {
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "regime": row["regime"],
                "expected_direction": row["expected_direction"],
                "reference_kind": row["reference_kind"],
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
                "treatment_selected_memory_ids": treatment.selected_memory_ids,
                "annotated_target_memory_ids": sorted(annotated_ids),
                "selected_annotated_item_count": len(
                    selected_ids & annotated_ids
                ),
                "annotated_item_count": len(annotated_ids),
                "selected_strategy_count": len(
                    treatment.selected_strategy_ids
                ),
            }
        )
    helpful = [
        row for row in pair_rows if row["regime"] in HELPFUL_REGIMES
    ]
    harmful = [
        row for row in pair_rows if row["regime"] in HARMFUL_REGIMES
    ]
    if len(helpful) != 10 or len(harmful) != 4:
        raise RuntimeError("pilot helpful/harmful pair balance drifted")
    helpful_summary = _group_summary(helpful)
    harmful_summary = _group_summary(harmful)
    status = _directional_status(helpful_summary, harmful_summary)
    return {
        "protocol": UPTAKE_MEASUREMENT_PROTOCOL,
        "status": status,
        "measurement_contract_sha256": expected_measurement_sha,
        "pilot_contract_sha256": pilot_contract["contract_sha256"],
        "semantic_encoder_spec_sha256": encoder.spec.digest(),
        "pairs": len(pair_rows),
        "helpful": helpful_summary,
        "harmful": harmful_summary,
        "pair_rows": pair_rows,
        "api_judges_used": False,
        "training_labels_created": False,
        "judge_qualification_authorized": False,
        "pm_training_authorized": False,
        "interpretation": (
            "report-only localization of evidence-surface dilution; this "
            "result cannot establish response quality or PM efficacy"
        ),
    }
