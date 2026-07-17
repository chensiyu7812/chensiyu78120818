from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .api import Endpoint
from .io import canonical_json, sha256_text

PROTOCOL = "pm-v1.6-development-final-judge-isolation-v1"

DEFAULT_FINAL_FAMILIES = frozenset({"openai_gpt4o", "anthropic_claude"})
DEFAULT_FINAL_MODEL_MARKERS = (
    "gpt-4o",
    "gpt4o",
    "claude",
)
DEFAULT_DEVELOPMENT_FAMILIES = frozenset(
    {"google_gemini", "deepseek"}
)


@dataclass(frozen=True)
class JudgeIsolationPolicy:
    development_families: frozenset[str] = DEFAULT_DEVELOPMENT_FAMILIES
    final_families: frozenset[str] = DEFAULT_FINAL_FAMILIES
    final_model_markers: tuple[str, ...] = DEFAULT_FINAL_MODEL_MARKERS

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "development_families": sorted(self.development_families),
            "final_families": sorted(self.final_families),
            "final_model_markers": list(self.final_model_markers),
            "match_fields": ["endpoint_alias", "family", "model", "base_url"],
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))


def _endpoint_fields(alias: str, endpoint: Endpoint | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(endpoint, Endpoint):
        return {
            "endpoint_alias": str(alias),
            "family": str(endpoint.family or ""),
            "model": str(endpoint.model or ""),
            "base_url": str(endpoint.base_url or ""),
        }
    return {
        "endpoint_alias": str(alias),
        "family": str(endpoint.get("family") or ""),
        "model": str(endpoint.get("model") or ""),
        "base_url": str(endpoint.get("base_url") or ""),
    }


def require_development_judge_isolation(
    endpoints: Mapping[str, Endpoint | Mapping[str, Any]],
    selected_aliases: Sequence[str],
    *,
    policy: JudgeIsolationPolicy | None = None,
) -> dict[str, Any]:
    """Require two distinct development-only judge families.

    The check is based on resolved endpoint metadata, not aliases alone. Final
    GPT/Claude endpoints cannot enter semantic review, label generation, tuning,
    or calibration even if they are renamed.
    """

    frozen = policy or JudgeIsolationPolicy()
    aliases = [str(value) for value in selected_aliases]
    if len(aliases) < 2 or len(aliases) != len(set(aliases)):
        raise RuntimeError("development judging requires at least two unique endpoints")
    resolved = []
    for alias in aliases:
        if alias not in endpoints:
            raise KeyError(f"unknown development judge endpoint: {alias}")
        fields = _endpoint_fields(alias, endpoints[alias])
        family = fields["family"].lower()
        combined = " ".join(fields.values()).lower()
        if family in {value.lower() for value in frozen.final_families}:
            raise RuntimeError(
                f"final judge family is forbidden in development: {fields}"
            )
        if any(marker.lower() in combined for marker in frozen.final_model_markers):
            raise RuntimeError(
                f"final judge model/provider marker is forbidden in development: {fields}"
            )
        if family not in {value.lower() for value in frozen.development_families}:
            raise RuntimeError(
                f"development judge family is outside the frozen allow-list: {fields}"
            )
        resolved.append(fields)
    families = [row["family"] for row in resolved]
    if len(set(families)) != len(families):
        raise RuntimeError("development judges must use distinct model families")
    return {
        "status": "PASS",
        "protocol": PROTOCOL,
        "policy_sha256": frozen.digest(),
        "resolved_endpoints": resolved,
    }


def legacy_semantic_review_invalidation(
    *,
    legacy_report_sha256: str,
    reason: str = "final GPT-4o family participated in development semantic review",
) -> dict[str, Any]:
    if len(str(legacy_report_sha256)) != 64:
        raise ValueError("legacy report SHA-256 is required")
    return {
        "status": "INVALID_FOR_PM_V1_6_PROTOCOL",
        "protocol": PROTOCOL,
        "legacy_report_sha256": str(legacy_report_sha256),
        "reason": str(reason),
        "reusable_rows": 0,
        "reusable_gate_status": False,
    }
