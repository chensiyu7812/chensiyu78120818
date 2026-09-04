from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActionOutcome, MemoryBackendRecord
from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import SemanticTextEncoder
from .text import normalize_space
from .v1_5_multisource_decomposition import (
    MULTISOURCE_DECOMPOSITION_PROTOCOL,
    SINGLE_SOURCE_ACTIONS,
    validate_existing_multisource_outcomes,
)


MULTISOURCE_UPTAKE_PROTOCOL = (
    "pm-v1.5-longitudinal-multisource-uptake-analysis-v1"
)


def _encode_lookup(
    encoder: SemanticTextEncoder,
    texts: Sequence[str],
) -> dict[str, np.ndarray]:
    distinct = sorted({normalize_space(text) for text in texts})
    if not distinct or any(not text for text in distinct):
        raise RuntimeError("multisource analysis cannot encode empty text")
    matrix = encoder.encode(distinct)
    if matrix.shape != (len(distinct), encoder.spec.output_dimension):
        raise RuntimeError("multisource semantic matrix shape drifted")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("multisource semantic matrix is non-finite")
    return dict(zip(distinct, matrix))


def _cosine(
    vectors: Mapping[str, np.ndarray],
    left: str,
    right: str,
) -> float:
    return float(
        np.clip(
            vectors[normalize_space(left)] @ vectors[normalize_space(right)],
            -1.0,
            1.0,
        )
    )


def classify_multisource_interference(
    state_rows: Sequence[Mapping[str, Any]],
) -> str:
    clear = sum(bool(row["clear_multisource_interference"]) for row in state_rows)
    any_positive = sum(int(row["positive_single_source_count"]) > 0 for row in state_rows)
    if len(state_rows) != 3:
        raise RuntimeError("multisource classification requires three states")
    if clear >= 2:
        return "MULTISOURCE_INTERFERENCE_SUPPORTED_REPORT_ONLY"
    if any_positive == 0:
        return "NO_SINGLE_SOURCE_UPTAKE_REPORT_ONLY"
    return "SOURCE_SPECIFIC_OR_INCONCLUSIVE_REPORT_ONLY"


def analyze_multisource_uptake(
    *,
    contract: Mapping[str, Any],
    backend_by_card: Mapping[str, MemoryBackendRecord],
    existing_outcomes: Sequence[ActionOutcome],
    single_source_outcomes: Sequence[ActionOutcome],
    encoder: SemanticTextEncoder,
) -> dict[str, Any]:
    if contract.get("protocol") != MULTISOURCE_DECOMPOSITION_PROTOCOL:
        raise RuntimeError("unexpected multisource decomposition protocol")
    without_sha = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if contract.get("contract_sha256") != sha256_text(
        canonical_json(without_sha)
    ):
        raise RuntimeError("multisource decomposition contract hash mismatch")
    existing_dicts = [
        row.model_dump(mode="json") for row in existing_outcomes
    ]
    reused = validate_existing_multisource_outcomes(
        contract=contract,
        outcome_rows=existing_dicts,
    )
    existing_by_key = {
        (str(row["state_id"]), str(row["action_id"])): row
        for row in reused
    }

    expected_new: dict[tuple[str, str], dict[str, Any]] = {}
    for state in contract.get("states") or []:
        state_id = str(state["state_id"])
        for arm in state["single_source_arms"]:
            expected_new[(state_id, str(arm["source"]))] = dict(arm)
    found_new: dict[tuple[str, str], ActionOutcome] = {}
    for outcome in single_source_outcomes:
        source = str(
            outcome.provenance.get("multisource_decomposition_source") or ""
        )
        key = (outcome.state_id, source)
        if key in found_new:
            raise RuntimeError(f"duplicate multisource outcome: {key}")
        if key not in expected_new:
            raise RuntimeError(f"unexpected multisource outcome: {key}")
        arm = expected_new[key]
        if (
            outcome.action_id != str(arm["action_id"])
            or set(outcome.selected_memory_ids)
            != {str(arm["target_memory_id"])}
            or outcome.strategy_view
        ):
            raise RuntimeError(f"multisource outcome surface drifted: {key}")
        found_new[key] = outcome
    if set(found_new) != set(expected_new):
        raise RuntimeError(
            "multisource outcome coverage mismatch: "
            f"missing={sorted(set(expected_new)-set(found_new))}"
        )

    prepared: list[dict[str, Any]] = []
    texts: list[str] = []
    for state in contract["states"]:
        state_id = str(state["state_id"])
        card_id = str(state["card_id"])
        backend = backend_by_card.get(card_id)
        if backend is None:
            raise RuntimeError(f"multisource backend absent: {card_id}")
        items_by_source = {item.source.value: item for item in backend.items}
        if set(items_by_source) != set(SINGLE_SOURCE_ACTIONS):
            raise RuntimeError(f"multisource backend sources drifted: {state_id}")
        references = {
            source: normalize_space(item.text)
            for source, item in items_by_source.items()
        }
        combined_reference = normalize_space(
            "\n".join(references[source] for source in sorted(references))
        )
        control = str(existing_by_key[(state_id, "M0+R0")]["response"])
        all_three = str(
            existing_by_key[(state_id, "MPMSME+R0")]["response"]
        )
        singles = {
            source: found_new[(state_id, source)].response
            for source in sorted(SINGLE_SOURCE_ACTIONS)
        }
        texts.extend(
            [
                control,
                all_three,
                combined_reference,
                *references.values(),
                *singles.values(),
            ]
        )
        prepared.append(
            {
                "state_id": state_id,
                "user_id": str(state["user_id"]),
                "semantic_family": str(state["semantic_family"]),
                "control": control,
                "all_three": all_three,
                "singles": singles,
                "references": references,
                "combined_reference": combined_reference,
            }
        )
    vectors = _encode_lookup(encoder, texts)

    state_reports: list[dict[str, Any]] = []
    for row in prepared:
        source_rows: list[dict[str, Any]] = []
        for source in sorted(SINGLE_SOURCE_ACTIONS):
            reference = row["references"][source]
            control_score = _cosine(vectors, row["control"], reference)
            single_score = _cosine(
                vectors, row["singles"][source], reference
            )
            all_score = _cosine(vectors, row["all_three"], reference)
            source_rows.append(
                {
                    "source": source,
                    "control_source_bge_cosine": control_score,
                    "single_source_bge_cosine": single_score,
                    "all_three_source_bge_cosine": all_score,
                    "single_minus_control_source_bge_delta": (
                        single_score - control_score
                    ),
                    "all_three_minus_control_source_bge_delta": (
                        all_score - control_score
                    ),
                    "single_minus_all_three_source_bge_delta": (
                        single_score - all_score
                    ),
                }
            )
        positive_count = sum(
            float(value["single_minus_control_source_bge_delta"]) > 0.0
            for value in source_rows
        )
        single_beats_all_count = sum(
            float(value["single_minus_all_three_source_bge_delta"]) > 0.0
            for value in source_rows
        )
        combined_reference = row["combined_reference"]
        combined_control = _cosine(
            vectors, row["control"], combined_reference
        )
        combined_all = _cosine(
            vectors, row["all_three"], combined_reference
        )
        state_reports.append(
            {
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "semantic_family": row["semantic_family"],
                "source_reports": source_rows,
                "positive_single_source_count": positive_count,
                "single_source_beats_all_three_count": single_beats_all_count,
                "clear_multisource_interference": (
                    positive_count >= 1 and single_beats_all_count >= 2
                ),
                "combined_reference_control_bge_cosine": combined_control,
                "combined_reference_all_three_bge_cosine": combined_all,
                "combined_reference_all_three_minus_control_delta": (
                    combined_all - combined_control
                ),
            }
        )
    status = classify_multisource_interference(state_reports)
    return {
        "protocol": MULTISOURCE_UPTAKE_PROTOCOL,
        "status": status,
        "contract_sha256": contract["contract_sha256"],
        "semantic_encoder_spec_sha256": encoder.spec.digest(),
        "states": state_reports,
        "summary": {
            "state_count": len(state_reports),
            "clear_multisource_interference_states": sum(
                bool(row["clear_multisource_interference"])
                for row in state_reports
            ),
            "states_with_any_positive_single_source": sum(
                int(row["positive_single_source_count"]) > 0
                for row in state_reports
            ),
            "positive_single_source_contrasts": sum(
                int(row["positive_single_source_count"])
                for row in state_reports
            ),
            "single_source_beats_all_three_contrasts": sum(
                int(row["single_source_beats_all_three_count"])
                for row in state_reports
            ),
        },
        "api_judges_used": False,
        "training_labels_created": False,
        "pm_training_authorized": False,
    }
