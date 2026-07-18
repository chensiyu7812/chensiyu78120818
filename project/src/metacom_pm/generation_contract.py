from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from .io import canonical_json, sha256_text
from .prompts import resolve_supporter_system_prompt
from .text import normalize_space


SUPPORTER_GENERATION_CONTRACT_VERSION = "pm-v2.2-supporter-generation-v1"
PM_V2_CONFIG_VERSION = "pm-v2.2"
# PM-v1.5 and PM-v1.6 are honestly versioned conference-track configs that
# intentionally reuse the exact same supporter-generation treatment. Keeping
# one loader prevents development/external prompt, cap, temperature, and finish
# reason drift across protocol versions.
ACCEPTED_PM_CONFIG_VERSIONS = frozenset(
    {PM_V2_CONFIG_VERSION, "pm-v1.5", "pm-v1.6"}
)
FINISH_REASON_PROTOCOL_VERSION = "pm-v2-finish-reason-v1"
OUTPUT_NORMALIZATION_VERSION = "normalize_space_v1"
NORMALIZED_FINISH_REASONS = frozenset(
    {"complete", "length", "tool_call", "content_filter", "unknown"}
)

_CONTRACT_KEYS = frozenset(
    {
        "version",
        "generator_endpoint",
        "system_prompt_id",
        "system_prompt_sha256",
        "temperature",
        "max_output_tokens",
        "output_normalization",
        "finish_reason_protocol",
        "accepted_normalized_finish_reasons",
        "reject_length",
        "reject_missing_or_unknown",
    }
)


@dataclass(frozen=True)
class SupporterGenerationContract:
    """Frozen response-generation treatment shared by development and external runs."""

    version: str
    generator_endpoint: str
    system_prompt_id: str
    system_prompt_sha256: str
    temperature: float
    max_output_tokens: int
    output_normalization: str
    finish_reason_protocol: str
    accepted_normalized_finish_reasons: tuple[str, ...]
    reject_length: bool
    reject_missing_or_unknown: bool

    def __post_init__(self) -> None:
        if self.version != SUPPORTER_GENERATION_CONTRACT_VERSION:
            raise ValueError(
                "unsupported supporter-generation contract version: "
                f"{self.version!r}"
            )
        if not self.generator_endpoint:
            raise ValueError("supporter generator endpoint must be non-empty")
        if self.finish_reason_protocol != FINISH_REASON_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported finish-reason protocol: "
                f"{self.finish_reason_protocol!r}"
            )
        if self.output_normalization != OUTPUT_NORMALIZATION_VERSION:
            raise ValueError(
                "unsupported supporter output normalization: "
                f"{self.output_normalization!r}"
            )
        if not isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError(
                "supporter generation temperature must be finite and non-negative"
            )
        if self.max_output_tokens < 1:
            raise ValueError("supporter max_output_tokens must be positive")
        if len(self.system_prompt_sha256) != 64:
            raise ValueError("supporter system prompt SHA-256 must contain 64 hex digits")
        try:
            int(self.system_prompt_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "supporter system prompt SHA-256 must contain 64 hex digits"
            ) from exc

        prompt = resolve_supporter_system_prompt(self.system_prompt_id)
        actual_prompt_sha256 = sha256_text(prompt)
        if self.system_prompt_sha256 != actual_prompt_sha256:
            raise ValueError(
                "supporter system prompt hash mismatch: "
                f"configured={self.system_prompt_sha256}, actual={actual_prompt_sha256}"
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
                "supporter generation must accept only complete responses"
            )
        if not self.reject_length or not self.reject_missing_or_unknown:
            raise ValueError(
                "supporter generation must reject length and missing/unknown "
                "finish reasons"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SupporterGenerationContract":
        data = dict(raw)
        if set(data) != _CONTRACT_KEYS:
            missing = sorted(_CONTRACT_KEYS - set(data))
            extra = sorted(set(data) - _CONTRACT_KEYS)
            raise ValueError(
                "supporter_generation_treatment keys do not match the frozen "
                f"contract: missing={missing}, extra={extra}"
            )
        string_keys = (
            "version",
            "generator_endpoint",
            "system_prompt_id",
            "system_prompt_sha256",
            "output_normalization",
            "finish_reason_protocol",
        )
        for key in string_keys:
            if not isinstance(data[key], str):
                raise ValueError(f"{key} must be a string")
        if isinstance(data["temperature"], bool) or not isinstance(
            data["temperature"], (int, float)
        ):
            raise ValueError("supporter generation temperature must be numeric")
        if isinstance(data["max_output_tokens"], bool) or not isinstance(
            data["max_output_tokens"], int
        ):
            raise ValueError("supporter max_output_tokens must be an integer")
        for key in ("reject_length", "reject_missing_or_unknown"):
            if not isinstance(data[key], bool):
                raise ValueError(f"{key} must be boolean")
        raw_reasons = data["accepted_normalized_finish_reasons"]
        if not isinstance(raw_reasons, (list, tuple)) or any(
            not isinstance(value, str) for value in raw_reasons
        ):
            raise ValueError(
                "accepted_normalized_finish_reasons must be a list of strings"
            )
        return cls(
            version=data["version"],
            generator_endpoint=data["generator_endpoint"],
            system_prompt_id=data["system_prompt_id"],
            system_prompt_sha256=data["system_prompt_sha256"],
            temperature=float(data["temperature"]),
            max_output_tokens=int(data["max_output_tokens"]),
            output_normalization=data["output_normalization"],
            finish_reason_protocol=data["finish_reason_protocol"],
            accepted_normalized_finish_reasons=tuple(raw_reasons),
            reject_length=data["reject_length"],
            reject_missing_or_unknown=data["reject_missing_or_unknown"],
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SupporterGenerationContract":
        if config.get("version") not in ACCEPTED_PM_CONFIG_VERSIONS:
            raise ValueError(
                "supporter generation contract requires one of these config "
                f"versions: {sorted(ACCEPTED_PM_CONFIG_VERSIONS)}"
            )
        raw = config.get("supporter_generation_treatment")
        if not isinstance(raw, Mapping):
            raise ValueError("PM config lacks supporter_generation_treatment")
        return cls.from_mapping(raw)

    @property
    def system_prompt(self) -> str:
        return resolve_supporter_system_prompt(self.system_prompt_id)

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator_endpoint": self.generator_endpoint,
            "system_prompt_id": self.system_prompt_id,
            "system_prompt_sha256": self.system_prompt_sha256,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "output_normalization": self.output_normalization,
            "finish_reason_protocol": self.finish_reason_protocol,
            "accepted_normalized_finish_reasons": list(
                self.accepted_normalized_finish_reasons
            ),
            "reject_length": self.reject_length,
            "reject_missing_or_unknown": self.reject_missing_or_unknown,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))

    def normalize_output(self, text: str) -> str:
        if self.output_normalization != OUTPUT_NORMALIZATION_VERSION:
            raise RuntimeError("supporter output normalization contract changed")
        return normalize_space(text)

    def completion_gate_error(
        self,
        *,
        normalized_finish_reason: str | None,
        provider_finish_reason: str | None,
    ) -> str | None:
        normalized = normalized_finish_reason or "unknown"
        if normalized == "length" and self.reject_length:
            return (
                "supporter completion reached the frozen output-token limit: "
                f"provider_finish_reason={provider_finish_reason!r}"
            )
        if normalized == "unknown" and self.reject_missing_or_unknown:
            return (
                "supporter completion has a missing or unknown finish reason: "
                f"provider_finish_reason={provider_finish_reason!r}"
            )
        if normalized not in self.accepted_normalized_finish_reasons:
            return (
                "supporter completion finish reason is not allowed by the frozen "
                f"contract: normalized={normalized!r}, "
                f"provider={provider_finish_reason!r}"
            )
        return None
