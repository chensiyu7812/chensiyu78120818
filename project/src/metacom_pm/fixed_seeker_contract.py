from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Any, Mapping

from .api import Endpoint
from .generation_contract import (
    FINISH_REASON_PROTOCOL_VERSION,
    NORMALIZED_FINISH_REASONS,
    OUTPUT_NORMALIZATION_VERSION,
)
from .io import canonical_json, sha256_text
from .text import normalize_space


FIXED_SEEKER_GENERATION_CONTRACT_VERSION = (
    "pm-v2.2-fixed-seeker-generation-v1"
)
FIXED_SEEKER_SYSTEM_PROMPT_ID = "evoemo-fixed-seeker-v1"
FIXED_SEEKER_SEED_PROTOCOL = "base-seed-plus-turn-index-v1"
FIXED_SEEKER_SCAFFOLD_PROTOCOL = "deterministic-generic-open-loop-v1"

# The 60-token limit is an instruction to the simulator, not the provider's
# output-token cap.  Keeping the template here lets the configuration bind its
# exact text without pretending that the private scenario is static.
FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE = """Role-play the emotional-support seeker. Stay faithful to the
private profile and topic. Do not mention that this is a benchmark, do not
reveal the entire hidden card at once, and do not discuss retrieval, policies,
or experimental conditions. Respond naturally to the supporter's latest
message in at most {response_instruction_token_limit} tokens. Do not become artificially agreeable merely
because the supporter suggests something.

Private scenario:
{private_scenario_json}
"""
FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256 = sha256_text(
    FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE
)

_CONTRACT_KEYS = frozenset(
    {
        "version",
        "seeker_endpoint",
        "system_prompt_id",
        "system_prompt_template_sha256",
        "response_instruction_token_limit",
        "temperature",
        "max_output_tokens",
        "output_normalization",
        "finish_reason_protocol",
        "accepted_normalized_finish_reasons",
        "maximum_physical_attempts_per_logical_call",
        "seed_protocol",
        "elicitation_scaffold_protocol",
    }
)


@dataclass(frozen=True)
class BoundFixedSeekerGenerationContract:
    """A frozen seeker treatment bound to the actual endpoint identity."""

    contract: "FixedSeekerGenerationContract"
    endpoint_id: str
    endpoint: Endpoint

    def __post_init__(self) -> None:
        if self.endpoint_id != self.contract.seeker_endpoint:
            raise ValueError(
                "fixed-seeker endpoint alias differs from the frozen contract: "
                f"configured={self.contract.seeker_endpoint!r}, "
                f"observed={self.endpoint_id!r}"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "treatment": self.contract.payload(),
            "endpoint": {
                "endpoint_id": self.endpoint_id,
                "base_url": self.endpoint.base_url,
                "model": self.endpoint.model,
                "family": self.endpoint.family,
                "api_key_env": self.endpoint.api_key_env,
                "timeout_seconds": self.endpoint.timeout_seconds,
            },
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))


@dataclass(frozen=True)
class FixedSeekerGenerationContract:
    """Frozen PM-v2.2 treatment for policy-independent EvoEmo tracks."""

    version: str
    seeker_endpoint: str
    system_prompt_id: str
    system_prompt_template_sha256: str
    response_instruction_token_limit: int
    temperature: float
    max_output_tokens: int
    output_normalization: str
    finish_reason_protocol: str
    accepted_normalized_finish_reasons: tuple[str, ...]
    maximum_physical_attempts_per_logical_call: int
    seed_protocol: str
    elicitation_scaffold_protocol: str

    def __post_init__(self) -> None:
        if self.version != FIXED_SEEKER_GENERATION_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported fixed-seeker contract version: {self.version!r}"
            )
        if not self.seeker_endpoint:
            raise ValueError("fixed-seeker endpoint alias must be non-empty")
        if self.system_prompt_id != FIXED_SEEKER_SYSTEM_PROMPT_ID:
            raise ValueError(
                f"unsupported fixed-seeker system prompt: {self.system_prompt_id!r}"
            )
        if (
            self.system_prompt_template_sha256
            != FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256
        ):
            raise ValueError(
                "fixed-seeker system-prompt template hash mismatch: "
                f"configured={self.system_prompt_template_sha256}, "
                f"actual={FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256}"
            )
        if self.response_instruction_token_limit != 60:
            raise ValueError(
                "fixed-seeker natural-language response instruction must equal "
                "60 tokens"
            )
        if not isfinite(self.temperature) or self.temperature != 0.2:
            raise ValueError("fixed-seeker temperature must equal 0.2")
        if self.max_output_tokens != 300:
            raise ValueError(
                "fixed-seeker API max_output_tokens must equal 300; the 60-token "
                "instruction is not an API truncation cap"
            )
        if self.output_normalization != OUTPUT_NORMALIZATION_VERSION:
            raise ValueError(
                f"unsupported fixed-seeker output normalization: "
                f"{self.output_normalization!r}"
            )
        if self.finish_reason_protocol != FINISH_REASON_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported finish-reason protocol: "
                f"{self.finish_reason_protocol!r}"
            )
        reasons = self.accepted_normalized_finish_reasons
        if not reasons or len(reasons) != len(set(reasons)):
            raise ValueError(
                "accepted normalized finish reasons must be non-empty and unique"
            )
        unknown_reasons = sorted(set(reasons) - NORMALIZED_FINISH_REASONS)
        if unknown_reasons:
            raise ValueError(
                f"unknown normalized finish reasons in contract: {unknown_reasons}"
            )
        if set(reasons) != {"complete"}:
            raise ValueError(
                "PM-v2.2 fixed-seeker generation must accept only complete responses"
            )
        if self.maximum_physical_attempts_per_logical_call != 1:
            raise ValueError(
                "PM-v2.2 fixed-seeker generation permits exactly one physical "
                "attempt per logical call"
            )
        if self.seed_protocol != FIXED_SEEKER_SEED_PROTOCOL:
            raise ValueError(
                f"unsupported fixed-seeker seed protocol: {self.seed_protocol!r}"
            )
        if self.elicitation_scaffold_protocol != FIXED_SEEKER_SCAFFOLD_PROTOCOL:
            raise ValueError(
                "unsupported fixed-seeker elicitation scaffold: "
                f"{self.elicitation_scaffold_protocol!r}"
            )

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any]
    ) -> "FixedSeekerGenerationContract":
        data = dict(raw)
        if set(data) != _CONTRACT_KEYS:
            missing = sorted(_CONTRACT_KEYS - set(data))
            extra = sorted(set(data) - _CONTRACT_KEYS)
            raise ValueError(
                "fixed_seeker_generation_treatment keys do not match the frozen "
                f"contract: missing={missing}, extra={extra}"
            )
        string_keys = (
            "version",
            "seeker_endpoint",
            "system_prompt_id",
            "system_prompt_template_sha256",
            "output_normalization",
            "finish_reason_protocol",
            "seed_protocol",
            "elicitation_scaffold_protocol",
        )
        for key in string_keys:
            if not isinstance(data[key], str):
                raise ValueError(f"{key} must be a string")
        integer_keys = (
            "response_instruction_token_limit",
            "max_output_tokens",
            "maximum_physical_attempts_per_logical_call",
        )
        for key in integer_keys:
            if isinstance(data[key], bool) or not isinstance(data[key], int):
                raise ValueError(f"{key} must be an integer")
        if isinstance(data["temperature"], bool) or not isinstance(
            data["temperature"], (int, float)
        ):
            raise ValueError("fixed-seeker temperature must be numeric")
        reasons = data["accepted_normalized_finish_reasons"]
        if not isinstance(reasons, (list, tuple)) or any(
            not isinstance(value, str) for value in reasons
        ):
            raise ValueError(
                "accepted_normalized_finish_reasons must be a list of strings"
            )
        return cls(
            version=data["version"],
            seeker_endpoint=data["seeker_endpoint"],
            system_prompt_id=data["system_prompt_id"],
            system_prompt_template_sha256=data["system_prompt_template_sha256"],
            response_instruction_token_limit=int(
                data["response_instruction_token_limit"]
            ),
            temperature=float(data["temperature"]),
            max_output_tokens=int(data["max_output_tokens"]),
            output_normalization=data["output_normalization"],
            finish_reason_protocol=data["finish_reason_protocol"],
            accepted_normalized_finish_reasons=tuple(reasons),
            maximum_physical_attempts_per_logical_call=int(
                data["maximum_physical_attempts_per_logical_call"]
            ),
            seed_protocol=data["seed_protocol"],
            elicitation_scaffold_protocol=data["elicitation_scaffold_protocol"],
        )

    def bind_endpoint(
        self, endpoint_id: str, endpoint: Endpoint
    ) -> BoundFixedSeekerGenerationContract:
        return BoundFixedSeekerGenerationContract(self, endpoint_id, endpoint)

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seeker_endpoint": self.seeker_endpoint,
            "system_prompt_id": self.system_prompt_id,
            "system_prompt_template_sha256": self.system_prompt_template_sha256,
            "response_instruction_token_limit": (
                self.response_instruction_token_limit
            ),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "output_normalization": self.output_normalization,
            "finish_reason_protocol": self.finish_reason_protocol,
            "accepted_normalized_finish_reasons": list(
                self.accepted_normalized_finish_reasons
            ),
            "maximum_physical_attempts_per_logical_call": (
                self.maximum_physical_attempts_per_logical_call
            ),
            "seed_protocol": self.seed_protocol,
            "elicitation_scaffold_protocol": self.elicitation_scaffold_protocol,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))

    def render_system_prompt(self, private_scenario: Mapping[str, Any]) -> str:
        return FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE.format(
            response_instruction_token_limit=self.response_instruction_token_limit,
            private_scenario_json=json.dumps(
                dict(private_scenario), ensure_ascii=False, indent=2
            ),
        )

    def normalize_output(self, text: str) -> str:
        return normalize_space(text)

    def completion_gate_error(
        self,
        *,
        normalized_finish_reason: str | None,
        provider_finish_reason: str | None,
    ) -> str | None:
        normalized = normalized_finish_reason or "unknown"
        if normalized not in self.accepted_normalized_finish_reasons:
            return (
                "fixed-seeker completion finish reason is not allowed by the "
                f"frozen contract: normalized={normalized!r}, "
                f"provider={provider_finish_reason!r}"
            )
        return None
