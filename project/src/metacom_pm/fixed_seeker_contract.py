from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping

from .api import Endpoint
from .generation_contract import (
    FINISH_REASON_PROTOCOL_VERSION,
    NORMALIZED_FINISH_REASONS,
    OUTPUT_NORMALIZATION_VERSION,
)
from .io import canonical_json, read_json, sha256_file, sha256_text
from .text import normalize_space


FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V1 = (
    "pm-v2.2-fixed-seeker-generation-v1"
)
# v2 lifts the exactly-one-physical-attempt restriction: configs/pm_v2.yaml
# still uses v1 and must keep its historical exactly-one-attempt behavior
# unchanged, so the two versions are validated differently below rather than
# repointing the shared constant (which would silently loosen pm_v2.yaml too).
FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V2 = (
    "pm-v2.2-fixed-seeker-generation-v2"
)
FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3 = (
    "pm-v2.2-fixed-seeker-generation-v3-bounded-surface"
)
SUPPORTED_FIXED_SEEKER_GENERATION_CONTRACT_VERSIONS = frozenset(
    {
        FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V1,
        FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V2,
        FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3,
    }
)
# configs/pm_v1_5.yaml stays on the historical V2 treatment: a real
# longitudinal dry-run showed that editing it directly invalidates the
# already-qualified V8.19.2 lineage via a pm_v1_5_config hash mismatch. V3 is
# therefore read only from this separately-tracked sidecar file by every V1.5
# consumer that needs it (never from configs/pm_v1_5.yaml). Each consumer
# verifies this exact frozen file hash before trusting its contents -- update
# deliberately if the sidecar is ever revised (a genuine V3 protocol change),
# never to silence a real mismatch.
FIXED_SEEKER_V3_SIDECAR_CONTRACT_SHA256 = (
    "8386e31e996a6621f293fb813812bf3882dd4eaf4bb337bd7fde7b536d4b8b21"
)
FIXED_SEEKER_SYSTEM_PROMPT_ID = "evoemo-fixed-seeker-v1"
FIXED_SEEKER_SYSTEM_PROMPT_ID_V3 = "evoemo-fixed-seeker-bounded-surface-v1"
FIXED_SEEKER_SEED_PROTOCOL = "base-seed-plus-turn-index-v1"
FIXED_SEEKER_SCAFFOLD_PROTOCOL = "deterministic-generic-open-loop-v1"
FIXED_SEEKER_SURFACE_SELECTION_PROTOCOL = (
    "normalized-longest-complete-sentence-prefix-v1"
)

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

FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_V3 = """Role-play the emotional-support seeker. Stay faithful to the
private profile and topic. Do not mention that this is a benchmark, do not
reveal the entire hidden card at once, and do not discuss retrieval, policies,
or experimental conditions. Respond naturally to the supporter's latest
message in one or two short sentences using at most
{response_instruction_word_limit} whitespace-delimited words. Do not become
artificially agreeable merely because the supporter suggests something.

Private scenario:
{private_scenario_json}
"""
FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256_V3 = sha256_text(
    FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_V3
)

_LEGACY_CONTRACT_KEYS = frozenset(
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
_V3_CONTRACT_KEYS = frozenset(
    {
        "version",
        "seeker_endpoint",
        "system_prompt_id",
        "system_prompt_template_sha256",
        "response_instruction_word_limit",
        "surface_selection_protocol",
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
class FixedSeekerSurface:
    """The exact normalized surface admitted to a fixed seeker track."""

    text: str
    original_word_count: int
    selected_word_count: int
    sentence_count: int
    prefix_selected: bool
    provider_finish_reason: str | None
    normalized_finish_reason: str
    protocol: str

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "original_word_count": self.original_word_count,
            "selected_word_count": self.selected_word_count,
            "sentence_count": self.sentence_count,
            "prefix_selected": self.prefix_selected,
            "provider_finish_reason": self.provider_finish_reason,
            "normalized_finish_reason": self.normalized_finish_reason,
        }


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
    response_instruction_token_limit: int | None
    response_instruction_word_limit: int | None
    surface_selection_protocol: str | None
    temperature: float
    max_output_tokens: int
    output_normalization: str
    finish_reason_protocol: str
    accepted_normalized_finish_reasons: tuple[str, ...]
    maximum_physical_attempts_per_logical_call: int
    seed_protocol: str
    elicitation_scaffold_protocol: str

    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_FIXED_SEEKER_GENERATION_CONTRACT_VERSIONS:
            raise ValueError(
                f"unsupported fixed-seeker contract version: {self.version!r}"
            )
        if not self.seeker_endpoint:
            raise ValueError("fixed-seeker endpoint alias must be non-empty")
        if self.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3:
            if self.system_prompt_id != FIXED_SEEKER_SYSTEM_PROMPT_ID_V3:
                raise ValueError(
                    "unsupported V3 fixed-seeker system prompt: "
                    f"{self.system_prompt_id!r}"
                )
            if (
                self.system_prompt_template_sha256
                != FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_SHA256_V3
            ):
                raise ValueError(
                    "V3 fixed-seeker system-prompt template hash mismatch"
                )
            if self.response_instruction_token_limit is not None:
                raise ValueError("V3 fixed seeker cannot claim a provider token limit")
            if self.response_instruction_word_limit != 60:
                raise ValueError(
                    "V3 fixed-seeker normalized response limit must equal 60 words"
                )
            if (
                self.surface_selection_protocol
                != FIXED_SEEKER_SURFACE_SELECTION_PROTOCOL
            ):
                raise ValueError("unsupported V3 fixed-seeker surface selector")
        else:
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
            if self.response_instruction_word_limit is not None:
                raise ValueError("legacy fixed seeker cannot claim a word limit")
            if self.surface_selection_protocol is not None:
                raise ValueError("legacy fixed seeker cannot select a bounded prefix")
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
        expected_reasons = (
            {"complete", "length"}
            if self.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
            else {"complete"}
        )
        if set(reasons) != expected_reasons:
            if self.version != FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3:
                raise ValueError(
                    "PM-v2.2 fixed-seeker generation must accept only complete "
                    "responses"
                )
            raise ValueError(
                "V3 fixed-seeker accepted finish reasons must equal complete "
                "plus length"
            )
        if self.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V1:
            if self.maximum_physical_attempts_per_logical_call != 1:
                raise ValueError(
                    "PM-v2.2 fixed-seeker generation v1 permits exactly one "
                    "physical attempt per logical call"
                )
        elif self.maximum_physical_attempts_per_logical_call < 1:
            raise ValueError(
                "PM-v2.2 fixed-seeker generation v2 requires a positive "
                "physical attempt budget per logical call"
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
        version = data.get("version")
        expected_keys = (
            _V3_CONTRACT_KEYS
            if version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
            else _LEGACY_CONTRACT_KEYS
        )
        if set(data) != expected_keys:
            missing = sorted(expected_keys - set(data))
            extra = sorted(set(data) - expected_keys)
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
        integer_keys = [
            "max_output_tokens",
            "maximum_physical_attempts_per_logical_call",
        ]
        integer_keys.append(
            "response_instruction_word_limit"
            if version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
            else "response_instruction_token_limit"
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
            response_instruction_token_limit=(
                None
                if version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
                else int(data["response_instruction_token_limit"])
            ),
            response_instruction_word_limit=(
                int(data["response_instruction_word_limit"])
                if version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
                else None
            ),
            surface_selection_protocol=(
                str(data["surface_selection_protocol"])
                if version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
                else None
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
        payload = {
            "version": self.version,
            "seeker_endpoint": self.seeker_endpoint,
            "system_prompt_id": self.system_prompt_id,
            "system_prompt_template_sha256": self.system_prompt_template_sha256,
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
        if self.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3:
            payload.update(
                {
                    "response_instruction_word_limit": (
                        self.response_instruction_word_limit
                    ),
                    "surface_selection_protocol": self.surface_selection_protocol,
                }
            )
        else:
            payload["response_instruction_token_limit"] = (
                self.response_instruction_token_limit
            )
        return payload

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))

    def render_system_prompt(self, private_scenario: Mapping[str, Any]) -> str:
        template = (
            FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE_V3
            if self.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
            else FIXED_SEEKER_SYSTEM_PROMPT_TEMPLATE
        )
        return template.format(
            response_instruction_token_limit=self.response_instruction_token_limit,
            response_instruction_word_limit=self.response_instruction_word_limit,
            private_scenario_json=json.dumps(
                dict(private_scenario), ensure_ascii=False, indent=2
            ),
        )

    def normalize_output(self, text: str) -> str:
        return normalize_space(text)

    def select_surface(
        self,
        text: str,
        *,
        normalized_finish_reason: str | None,
        provider_finish_reason: str | None,
    ) -> tuple[FixedSeekerSurface | None, str | None]:
        """Select the exact final seeker surface without mid-sentence truncation."""

        normalized_reason = normalized_finish_reason or "unknown"
        finish_error = self.completion_gate_error(
            normalized_finish_reason=normalized_reason,
            provider_finish_reason=provider_finish_reason,
        )
        if finish_error is not None:
            return None, finish_error
        normalized = self.normalize_output(text)
        if not normalized:
            return None, "fixed-seeker completion is empty after frozen normalization"
        if self.version != FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3:
            return (
                FixedSeekerSurface(
                    text=normalized,
                    original_word_count=len(normalized.split()),
                    selected_word_count=len(normalized.split()),
                    sentence_count=0,
                    prefix_selected=False,
                    provider_finish_reason=provider_finish_reason,
                    normalized_finish_reason=normalized_reason,
                    protocol=self.output_normalization,
                ),
                None,
            )

        assert self.response_instruction_word_limit is not None
        limit = int(self.response_instruction_word_limit)
        original_words = len(normalized.split())
        if normalized_reason == "complete" and original_words <= limit:
            return (
                FixedSeekerSurface(
                    text=normalized,
                    original_word_count=original_words,
                    selected_word_count=original_words,
                    sentence_count=1,
                    prefix_selected=False,
                    provider_finish_reason=provider_finish_reason,
                    normalized_finish_reason=normalized_reason,
                    protocol=str(self.surface_selection_protocol),
                ),
                None,
            )

        complete_sentences = [
            match.group(0).strip()
            for match in re.finditer(r".+?[.!?](?=\s|$)", normalized)
        ]
        selected: list[str] = []
        selected_words = 0
        for sentence in complete_sentences:
            sentence_words = len(sentence.split())
            if selected_words + sentence_words > limit:
                break
            selected.append(sentence)
            selected_words += sentence_words
        if not selected:
            return (
                None,
                "fixed-seeker V3 has no complete sentence prefix within its "
                f"{limit}-word bound",
            )
        final_text = " ".join(selected)
        if not final_text.endswith((".", "!", "?")):
            raise RuntimeError("fixed-seeker V3 selector produced an incomplete surface")
        return (
            FixedSeekerSurface(
                text=final_text,
                original_word_count=original_words,
                selected_word_count=selected_words,
                sentence_count=len(selected),
                prefix_selected=final_text != normalized,
                provider_finish_reason=provider_finish_reason,
                normalized_finish_reason=normalized_reason,
                protocol=str(self.surface_selection_protocol),
            ),
            None,
        )

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


FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL = (
    "pm-v1.5-fixed-seeker-v3-formal-bundle-binding-v1"
)


def require_fixed_seeker_v3_formal_bundle(
    binding_path: str | Path,
    *,
    root: Path,
    required_stage: str,
) -> dict[str, Any]:
    """Fail closed unless the tracked binding matches the real bundle on disk.

    The formal V3 fixed-seeker bundle is located and verified through this
    one tracked, content-addressed binding file -- never by a hardcoded
    output-directory basename. A caller that hardcodes
    ``.../evoemo_fixed_tracks_v1_5_v3_formal_candidate`` cannot survive the
    bundle living under a different, freshly-approved directory name (as
    happens whenever a prior identity is permanently consumed and a new one
    is approved into a new directory); binding on content instead of a path
    name survives that rename for free. Every check here is independent of
    the bundle's own internal attestation, so a forged directory with a
    self-consistent-but-wrong attestation (same stage, different real bytes)
    is still rejected: the tracks file and the attestation file themselves
    must match the exact digests recorded in the binding at approval time,
    not merely be internally consistent with each other.
    """

    path = Path(binding_path)
    binding = read_json(path)
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if binding.get("binding_sha256") != sha256_text(canonical_json(payload)):
        raise RuntimeError(
            f"fixed-seeker V3 formal-bundle binding hash mismatch: path={path}"
        )
    if binding.get("protocol") != FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL:
        raise RuntimeError(
            f"unexpected fixed-seeker V3 formal-bundle binding protocol: path={path}"
        )
    if binding.get("stage") != required_stage:
        raise RuntimeError(
            "fixed-seeker V3 formal-bundle binding stage does not match the "
            f"required stage: bound={binding.get('stage')!r}, "
            f"required={required_stage!r}"
        )
    output_directory = (root / str(binding["output_directory"])).resolve()
    tracks_path = output_directory / "fixed_seeker_tracks.jsonl"
    attestation_path = output_directory / "artifact_attestation.json"
    observed_tracks_sha256 = sha256_file(tracks_path)
    if observed_tracks_sha256 != binding["fixed_seeker_tracks_sha256"]:
        raise RuntimeError(
            "fixed-seeker V3 formal-bundle tracks file hash mismatch: "
            f"path={tracks_path}, expected={binding['fixed_seeker_tracks_sha256']}, "
            f"observed={observed_tracks_sha256}"
        )
    observed_attestation_sha256 = sha256_file(attestation_path)
    if observed_attestation_sha256 != binding["artifact_attestation_sha256"]:
        raise RuntimeError(
            "fixed-seeker V3 formal-bundle attestation file hash mismatch: "
            f"path={attestation_path}, "
            f"expected={binding['artifact_attestation_sha256']}, "
            f"observed={observed_attestation_sha256}"
        )
    attestation = read_json(attestation_path)
    if attestation.get("stage") != required_stage:
        raise RuntimeError(
            "fixed-seeker V3 formal-bundle attestation stage mismatch: "
            f"path={attestation_path}, stage={attestation.get('stage')!r}, "
            f"required={required_stage!r}"
        )
    summary = read_json(output_directory / "summary.json")
    expected_tracks = int(binding["expected_tracks"])
    expected_logical_calls = int(binding["expected_logical_calls"])
    if (
        int(summary.get("expected_tracks") or -1) != expected_tracks
        or int(summary.get("completed_tracks") or -1) != expected_tracks
        or int(summary.get("expected_logical_calls") or -1) != expected_logical_calls
        or int(summary.get("successful_logical_calls") or -1) != expected_logical_calls
    ):
        raise RuntimeError(
            "fixed-seeker V3 formal-bundle track/call counts do not match the "
            f"binding: path={output_directory / 'summary.json'}"
        )
    return {
        "output_directory": output_directory,
        "fixed_tracks_path": tracks_path,
        "artifact_attestation_path": attestation_path,
        "binding": dict(binding),
    }


def require_fixed_seeker_v3_sidecar_contract(
    sidecar_path: str | Path,
) -> "FixedSeekerGenerationContract":
    """Fail closed unless the V3 sidecar file is exactly the frozen contract.

    Every V1.5 consumer that needs the V3 bounded-surface treatment loads it
    through this one function, never by reading configs/pm_v1_5.yaml (which
    stays on the historical V2 treatment -- see FIXED_SEEKER_V3_SIDECAR_
    CONTRACT_SHA256's docstring for why). Verifies both the exact frozen file
    hash and that the file actually declares itself V3, so a hand-edited or
    swapped-in sidecar is rejected before any downstream generation/freeze
    binding check even runs.
    """

    path = Path(sidecar_path)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != FIXED_SEEKER_V3_SIDECAR_CONTRACT_SHA256:
        raise RuntimeError(
            "fixed-seeker V3 sidecar contract hash mismatch: "
            f"path={path}, expected={FIXED_SEEKER_V3_SIDECAR_CONTRACT_SHA256}, "
            f"observed={observed_sha256}"
        )
    contract = FixedSeekerGenerationContract.from_mapping(read_json(path))
    if contract.version != FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3:
        raise RuntimeError(
            f"fixed-seeker V3 sidecar contract is not V3: path={path}, "
            f"version={contract.version!r}"
        )
    return contract
