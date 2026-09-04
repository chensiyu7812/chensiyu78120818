#!/usr/bin/env python3
"""No-API PM-v1.5 semantic runtime and readiness preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metacom_pm.config import load_config
from metacom_pm.contracts import DialogueTurn
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json
from metacom_pm.pm_v2_contracts import PMV2Split, ResourceNeedRegime
from metacom_pm.pm_v2_data import (
    READINESS_SURFACE_PROTOCOL,
    _READINESS_SURFACE_CLAUSES,
    GeneratedStateCase,
    case_to_state,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    require_unified_semantic_query_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v1_5_step0 import (
    _build_readiness_vectors,
    readiness_natural_language_challenge,
)


ROOT = Path(__file__).resolve().parents[2]


def compiled_readiness_surface_smoke(encoder) -> dict[str, object]:
    """Prove every compiler-owned readiness phrase is visible to frozen BGE."""

    anchors = _build_readiness_vectors(
        semantic_encoder=encoder, n_features=encoder.spec.output_dimension
    )
    cases = [
        (expected, clause)
        for expected, clauses in _READINESS_SURFACE_CLAUSES.items()
        for clause in clauses
    ]
    matrix = np.asarray(encoder.encode([clause for _, clause in cases]), dtype=float)
    rows = []
    for (expected, clause), query in zip(cases, matrix, strict=True):
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = {
            name: float(np.clip(query @ np.asarray(vector), -1.0, 1.0))
            for name, vector in anchors.items()
        }
        top1 = max(scores, key=scores.get)
        rows.append(
            {
                "expected": expected,
                "top1": top1,
                "matches_expected": top1 == expected,
                "clause_sha256": sha256_text(clause),
                "scores": scores,
            }
        )
    status = "PASS" if all(row["matches_expected"] for row in rows) else "FAIL"
    return {
        "status": status,
        "protocol": READINESS_SURFACE_PROTOCOL,
        "outcome_labels_used": False,
        "strategy_resource_target_used": False,
        "case_count": len(rows),
        "match_count": sum(row["matches_expected"] for row in rows),
        "rows": rows,
    }


def strict_state_construction_smoke(encoder) -> dict[str, object]:
    """Exercise the real section assembler and strict PMV2State schema together."""

    history = [
        DialogueTurn(
            role="user" if index % 2 == 0 else "assistant",
            content=" ".join(
                f"public_canary_history_{index}_{token}" for token in range(60)
            ),
        )
        for index in range(12)
    ]
    case = GeneratedStateCase(
        case_id="semantic_runtime_strict_state_canary",
        semantic_family="public_runtime_canary",
        surface_form_id="surface_semantic_runtime_strict_state_canary",
        regime=ResourceNeedRegime.CONTEXT_ONLY,
        current_user_text="I want to think carefully about one next step.",
        recent_dialogue=history,
        session_summary=" ".join(
            f"public_canary_summary_{index}" for index in range(80)
        ),
        session_index=2,
        profile_memories=[],
        summary_memories=[],
        event_memories=[],
        needed_memory_sources=[],
        authorized_user_context="Public synthetic runtime canary only.",
        coverage_rationale="Exercises bounded long-context state construction.",
    )
    state = case_to_state(
        user_id="public_semantic_runtime_canary_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    audit = state.provenance["semantic_observation"]
    allocation = audit["tokenization"]["section_allocation"]
    recent = allocation["sections"]["recent_dialogue"]
    if (
        audit["step0_semantic_query_sha256"]
        != audit["state_embedding_query_sha256"]
        or audit["step0_semantic_query_vector_sha256"]
        != audit["state_embedding_query_vector_sha256"]
        or audit["tokenization"]["views"]["visible_dialogue_state"]["truncated"]
        is not False
        or int(recent["dropped_token_count"]) <= 0
    ):
        raise RuntimeError("strict semantic state-construction smoke failed")
    return {
        "status": "PASS",
        "protocol": audit["semantic_query_protocol"],
        "state_contract_sha256": sha256_text(
            canonical_json(state.model_dump(mode="json"))
        ),
        "semantic_query_sha256": audit["state_embedding_query_sha256"],
        "semantic_query_vector_sha256": audit[
            "state_embedding_query_vector_sha256"
        ],
        "combined_dimension": len(state.text_embedding),
        "final_visible_state_token_count": allocation[
            "final_visible_state_token_count"
        ],
        "recent_dialogue_dropped_token_count": recent["dropped_token_count"],
        "implicit_tokenizer_truncation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_semantic_runtime_preflight.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    semantic_query_contract = require_unified_semantic_query_contract(config)
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    semantic_runtime = require_semantic_runtime_contract(config, encoder)
    readiness = readiness_natural_language_challenge(encoder)
    readiness_surface = compiled_readiness_surface_smoke(encoder)
    if readiness_surface["status"] != "PASS":
        raise RuntimeError("compiler-owned readiness surface is not BGE-observable")
    strict_state_smoke = strict_state_construction_smoke(encoder)
    report = {
        "status": "PASS_RUNTIME_WITH_REPORT_ONLY_READINESS_DIAGNOSTIC",
        "protocol": "pm-v1.5-semantic-runtime-preflight-v1",
        "pm_v1_5_config_sha256": sha256_file(args.config),
        "semantic_runtime_status": semantic_runtime["status"],
        "readiness_challenge_status": readiness["status"],
        "compiled_readiness_surface_status": readiness_surface["status"],
        "semantic_runtime": semantic_runtime,
        "semantic_query_contract": semantic_query_contract,
        "strict_state_construction_smoke": strict_state_smoke,
        "readiness_natural_language_challenge": readiness,
        "compiled_readiness_surface_smoke": readiness_surface,
        "api_calls": 0,
        "outcome_labels_used": False,
    }
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
