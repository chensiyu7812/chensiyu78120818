from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from metacom_pm.config import load_config
from metacom_pm.generation_contract import (
    FINISH_REASON_PROTOCOL_VERSION,
    OUTPUT_NORMALIZATION_VERSION,
    SUPPORTER_GENERATION_CONTRACT_VERSION,
    SupporterGenerationContract,
)
from metacom_pm.prompts import (
    SELECTIVE_ESMEM_SYSTEM,
    resolve_supporter_system_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def _configured_contract() -> SupporterGenerationContract:
    config = load_config(ROOT / "configs" / "pm_v2.yaml")
    return SupporterGenerationContract.from_config(config)


def test_pm_v22_config_has_one_valid_supporter_generation_contract() -> None:
    config = load_config(ROOT / "configs" / "pm_v2.yaml")
    contract = SupporterGenerationContract.from_config(config)

    assert config["version"] == "pm-v2.2"
    assert contract.version == SUPPORTER_GENERATION_CONTRACT_VERSION
    assert contract.finish_reason_protocol == FINISH_REASON_PROTOCOL_VERSION
    assert contract.output_normalization == OUTPUT_NORMALIZATION_VERSION
    assert contract.system_prompt_id == "selective_esmem_v1"
    assert contract.system_prompt == SELECTIVE_ESMEM_SYSTEM
    assert contract.temperature == 0.0
    assert contract.max_output_tokens == 300
    assert contract.accepted_normalized_finish_reasons == ("complete",)
    assert len(contract.digest()) == 64
    assert contract.payload() == config["supporter_generation_treatment"]
    assert {
        "generator_endpoint",
        "temperature",
        "max_output_tokens",
    }.isdisjoint(config["development_sweep"])
    assert {
        "generator_endpoint",
        "generation_protocol",
        "system_prompt_id",
        "temperature",
        "max_output_tokens",
    }.isdisjoint(config["external_evaluation"])


def test_supporter_generation_contract_digest_changes_with_treatment() -> None:
    original = _configured_contract()
    changed_payload = deepcopy(original.payload())
    changed_payload["max_output_tokens"] = original.max_output_tokens + 1
    changed = SupporterGenerationContract.from_mapping(changed_payload)

    assert changed.digest() != original.digest()


def test_supporter_generation_contract_rejects_prompt_hash_drift() -> None:
    payload = deepcopy(_configured_contract().payload())
    payload["system_prompt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="system prompt hash mismatch"):
        SupporterGenerationContract.from_mapping(payload)


def test_supporter_generation_contract_rejects_schema_drift() -> None:
    payload = deepcopy(_configured_contract().payload())
    payload["unreviewed_override"] = True

    with pytest.raises(ValueError, match="extra=.*unreviewed_override"):
        SupporterGenerationContract.from_mapping(payload)


def test_pm_v1_5_config_uses_same_supporter_generation_contract() -> None:
    """PM-v1.5's honestly-versioned config ("pm-v1.5", not disguised as
    "pm-v2.2") must load through the same shared contract loader, and its
    supporter treatment must match PM-v2.2's byte-for-byte -- that
    consistency is the entire point of this shared loader."""

    v1_5_config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    assert v1_5_config["version"] == "pm-v1.5"
    v1_5_contract = SupporterGenerationContract.from_config(v1_5_config)
    assert v1_5_contract.digest() == _configured_contract().digest()


def test_supporter_generation_contract_rejects_wrong_config_version() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "pm_v2.yaml"))
    config["version"] = "pm-v2.1"

    with pytest.raises(ValueError, match="requires one of these config versions"):
        SupporterGenerationContract.from_config(config)


def test_supporter_generation_contract_rejects_nonfinite_temperature() -> None:
    payload = deepcopy(_configured_contract().payload())
    payload["temperature"] = float("nan")

    with pytest.raises(ValueError, match="finite and non-negative"):
        SupporterGenerationContract.from_mapping(payload)


def test_supporter_generation_contract_normalizes_output_and_gates_finish_reason() -> None:
    contract = _configured_contract()

    assert contract.normalize_output("  First line.\n\nSecond   line.  ") == (
        "First line. Second line."
    )
    assert (
        contract.completion_gate_error(
            normalized_finish_reason="complete",
            provider_finish_reason="stop",
        )
        is None
    )
    assert "output-token limit" in str(
        contract.completion_gate_error(
            normalized_finish_reason="length",
            provider_finish_reason="length",
        )
    )
    assert "missing or unknown" in str(
        contract.completion_gate_error(
            normalized_finish_reason="unknown",
            provider_finish_reason=None,
        )
    )


def test_supporter_prompt_resolver_is_allowlisted() -> None:
    assert resolve_supporter_system_prompt("selective_esmem_v1") == (
        SELECTIVE_ESMEM_SYSTEM
    )
    with pytest.raises(ValueError, match="unknown supporter system prompt"):
        resolve_supporter_system_prompt("unreviewed_prompt")
