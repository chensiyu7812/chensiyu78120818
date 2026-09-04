from __future__ import annotations

from copy import deepcopy

import pytest

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.v1_5_claim_stack_audit import (
    REQUIRED_CONSUMER_ROLES,
    validate_same_stack_consumer_matrix,
)


def _mechanism(*, bank: str = "a" * 64) -> dict:
    body = {
        "protocol": "pm-v1.5-response-mechanism-contract-v1",
        "supporter_generation_treatment_sha256": "b" * 64,
        "generator_endpoint_sha256": "c" * 64,
        "strategy_bank_sha256": bank,
        "retrieval_settings": {
            "memory_min_score": 0.0,
            "strategy_min_score": 0.0,
            "strategy_top_k": 3,
        },
        "memory_top_k_by_source": {"ME": 3, "MP": 2, "MS": 2},
        "evidence_filter_enabled": False,
        "shared_code_manifest": {"src/metacom_pm/retrieval.py": "d" * 64},
        "canary_prompts": {"M0+R0": "e" * 64},
    }
    return {
        **body,
        "contract_sha256": sha256_text(canonical_json(body)),
    }


def _consumers(mechanism: dict) -> list[dict]:
    return [
        {
            "role": role,
            "response_mechanism_contract": deepcopy(mechanism),
        }
        for role in REQUIRED_CONSUMER_ROLES
    ]


def test_same_stack_matrix_requires_one_exact_mechanism_for_all_roles():
    mechanism = _mechanism()
    report = validate_same_stack_consumer_matrix(_consumers(mechanism))
    assert report["status"] == "PASS"
    assert (
        report["response_mechanism_contract_sha256"]
        == mechanism["contract_sha256"]
    )


def test_same_stack_matrix_rejects_bank_drift_between_policy_conditions():
    rows = _consumers(_mechanism())
    rows[-1]["response_mechanism_contract"] = _mechanism(bank="f" * 64)
    with pytest.raises(RuntimeError, match="does not match the frozen value"):
        validate_same_stack_consumer_matrix(rows)


def test_same_stack_matrix_rejects_missing_or_duplicate_roles():
    rows = _consumers(_mechanism())
    with pytest.raises(RuntimeError, match="roles are not exact"):
        validate_same_stack_consumer_matrix(rows[:-1])
    rows[-1]["role"] = rows[0]["role"]
    with pytest.raises(RuntimeError, match="duplicate roles"):
        validate_same_stack_consumer_matrix(rows)


def test_same_stack_matrix_recomputes_declared_contract_hash():
    rows = _consumers(_mechanism())
    rows[2]["response_mechanism_contract"]["strategy_bank_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="declared contract_sha256"):
        validate_same_stack_consumer_matrix(rows)

