"""Frozen, local semantic observations for the PM-v1.5 pre-retrieval router.

The encoder is a deployable feature extractor, not a judge and not an oracle.  It
may read only text already visible to the router.  Reportable runs resolve an
exact local Hugging Face snapshot and never download or update weights at run
time.  Model identity and file hashes are audit metadata; neither is exposed as
a numerical PM feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import platform
from pathlib import Path
import site
import sys
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np
from pydantic import Field, model_validator

from .contracts import StrictModel
from .io import canonical_json, sha256_file, sha256_text


SEMANTIC_ENCODER_PROTOCOL = "pm-v1.5-frozen-visible-text-encoder-v1"
SEMANTIC_INPUT_PROTOCOL = "pm-v1.5-visible-dialogue-state-v2-section-aware"
UNIFIED_SEMANTIC_QUERY_PROTOCOL = (
    "pm-v1.5-unified-step0-state-semantic-query-v1"
)
SEMANTIC_SNAPSHOT_HASH_PROTOCOL = "relative-path-tab-sha256-v1"
SEMANTIC_RUNTIME_PROTOCOL = "pm-v1.5-semantic-runtime-canary-v1"
SEMANTIC_CANARY_TEXTS = (
    "I only want someone to listen right now.",
    "Could you help me think through a small next step?",
    "Maybe, I guess.",
)
SEMANTIC_RUNTIME_PACKAGES = (
    "numpy",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
)


class FrozenSemanticEncoderSpec(StrictModel):
    protocol: Literal["pm-v1.5-frozen-visible-text-encoder-v1"] = (
        SEMANTIC_ENCODER_PROTOCOL
    )
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pooling: Literal["cls"] = "cls"
    normalize: Literal[True] = True
    max_length: int = Field(ge=64, le=4096)
    output_dimension: int = Field(ge=16, le=4096)
    output_round_decimals: int = Field(default=8, ge=4, le=12)
    local_files_only: Literal[True] = True
    trust_remote_code: Literal[False] = False
    language_scope: Literal["english"] = "english"
    visible_state_input_protocol: Literal[
        "pm-v1.5-visible-dialogue-state-v2-section-aware"
    ] = SEMANTIC_INPUT_PROTOCOL
    current_user_state_token_budget: int = Field(default=32, ge=32, le=512)
    session_summary_token_budget: int = Field(default=16, ge=16, le=512)
    history_retention_policy: Literal[
        "most-recent-token-suffix-preserve-chronology"
    ] = "most-recent-token-suffix-preserve-chronology"
    implicit_full_state_truncation: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def valid_visible_state_budgets(self):
        # Leave room for section headers, special tokens, and useful history.
        if (
            self.current_user_state_token_budget
            + self.session_summary_token_budget
            > self.max_length - 16
        ):
            raise ValueError("visible-state section budgets leave too little history room")
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class SemanticEncoderBinding(StrictModel):
    protocol: Literal["pm-v1.5-semantic-encoder-binding-v1"] = (
        "pm-v1.5-semantic-encoder-binding-v1"
    )
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_file_count: int = Field(ge=1)
    implementation: Literal["transformers-auto-model-cls-float32"]


class FrozenSemanticRuntimeContract(StrictModel):
    """Exact numerical environment required for reportable BGE observations."""

    protocol: Literal["pm-v1.5-semantic-runtime-canary-v1"] = (
        SEMANTIC_RUNTIME_PROTOCOL
    )
    encoder_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canary_texts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canary_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canary_shape: list[int] = Field(min_length=2, max_length=2)
    python_version: str = Field(min_length=1)
    platform_system: str = Field(min_length=1)
    platform_machine: str = Field(min_length=1)
    user_site_enabled: Literal[False]
    user_site_on_sys_path: Literal[False]
    packages: dict[str, str]
    device: Literal["cpu"]
    dtype: Literal["float32"]

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class SemanticTextEncoder(Protocol):
    spec: FrozenSemanticEncoderSpec
    binding: SemanticEncoderBinding

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a finite, row-normalized ``[n, dimension]`` float matrix."""


@dataclass(frozen=True)
class PreparedVisibleSemanticState:
    """One bounded semantic input shared by Step-0 and state features."""

    query_text: str
    current_user_vector: tuple[float, ...]
    state_query_vector: tuple[float, ...]
    combined_embedding: tuple[float, ...]
    semantic_query_sha256: str
    semantic_query_vector_sha256: str
    audit: Mapping[str, Any]


def require_unified_semantic_query_contract(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Fail before work if the frozen config permits two full-state BGE inputs."""

    step0 = config.get("step0_observation") or {}
    diagnostics = config.get("semantic_diagnostics") or {}
    expected = {
        "step0_protocol": (
            "pm-v1.5-step0-semantic-source-observation-v3-unified-state-query"
        ),
        "semantic_query_protocol": UNIFIED_SEMANTIC_QUERY_PROTOCOL,
    }
    if (
        step0.get("protocol") != expected["step0_protocol"]
        or step0.get("stage") != "pre_item_retrieval"
        or step0.get("full_state_semantic_query")
        != expected["semantic_query_protocol"]
        or diagnostics.get("unified_step0_state_semantic_query_required")
        is not True
        or diagnostics.get("section_allocation_required_for_every_state")
        is not True
    ):
        raise RuntimeError(
            "PM-v1.5 config lacks the unified section-aware Step-0/state "
            "semantic-query contract"
        )
    return expected


def resolve_semantic_encoder_binding(
    spec: FrozenSemanticEncoderSpec,
) -> tuple[Path, SemanticEncoderBinding]:
    """Resolve and hash the local snapshot without loading model weights."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - deployment preflight
        raise RuntimeError(
            "frozen semantic encoding requires installed huggingface_hub"
        ) from exc
    snapshot_path = Path(
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_files_only=True,
        )
    ).resolve()
    observed_hash = semantic_snapshot_tree_sha256(snapshot_path)
    if observed_hash != spec.snapshot_tree_sha256:
        raise RuntimeError(
            "semantic encoder snapshot hash mismatch: "
            f"expected={spec.snapshot_tree_sha256}, observed={observed_hash}"
        )
    rows = _snapshot_file_manifest(snapshot_path)
    return snapshot_path, SemanticEncoderBinding(
        spec_sha256=spec.digest(),
        snapshot_tree_sha256=observed_hash,
        snapshot_file_count=len(rows),
        implementation="transformers-auto-model-cls-float32",
    )


def semantic_encoder_spec_from_config(
    config: Mapping[str, Any],
) -> FrozenSemanticEncoderSpec:
    raw = config.get("semantic_encoder")
    if not isinstance(raw, Mapping):
        raise RuntimeError("PM-v1.5 config lacks the frozen semantic_encoder contract")
    if raw.get("enabled") is not True:
        raise RuntimeError("reportable PM-v1.5 requires semantic_encoder.enabled=true")
    payload = {key: value for key, value in raw.items() if key != "enabled"}
    return FrozenSemanticEncoderSpec.model_validate(payload)


def semantic_runtime_contract_from_config(
    config: Mapping[str, Any],
) -> FrozenSemanticRuntimeContract:
    raw = config.get("semantic_runtime")
    if not isinstance(raw, Mapping):
        raise RuntimeError("PM-v1.5 config lacks the frozen semantic runtime contract")
    contract = FrozenSemanticRuntimeContract.model_validate(raw)
    if set(contract.packages) != set(SEMANTIC_RUNTIME_PACKAGES):
        raise RuntimeError(
            "semantic runtime package set differs from the frozen contract"
        )
    expected_text_hash = sha256_text(canonical_json(list(SEMANTIC_CANARY_TEXTS)))
    if contract.canary_texts_sha256 != expected_text_hash:
        raise RuntimeError("semantic runtime canary text contract is stale")
    return contract


def _snapshot_file_manifest(snapshot_path: str | Path) -> list[dict[str, str]]:
    root = Path(snapshot_path).resolve()
    if not root.is_dir():
        raise RuntimeError(f"semantic encoder snapshot is absent: {root}")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not rows:
        raise RuntimeError("semantic encoder snapshot contains no files")
    return rows


def semantic_snapshot_tree_sha256(snapshot_path: str | Path) -> str:
    """Hash resolved snapshot files without including a machine-local path."""

    rows = _snapshot_file_manifest(snapshot_path)
    lines = "".join(f"{row['path']}\t{row['sha256']}\n" for row in rows)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


@dataclass
class FrozenTransformerSemanticEncoder:
    spec: FrozenSemanticEncoderSpec
    binding: SemanticEncoderBinding
    tokenizer: Any
    model: Any

    @classmethod
    def load(cls, spec: FrozenSemanticEncoderSpec) -> "FrozenTransformerSemanticEncoder":
        """Resolve and load one already-cached immutable snapshot.

        ``snapshot_download(..., local_files_only=True)`` is a resolver only.  It
        cannot contact the network under this contract.
        """

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by deployment preflight
            raise RuntimeError(
                "frozen semantic encoding requires installed transformers and "
                "huggingface_hub dependencies"
            ) from exc
        snapshot_path, binding = resolve_semantic_encoder_binding(spec)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.eval()
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        if hidden_size != spec.output_dimension:
            raise RuntimeError(
                "semantic encoder output dimension mismatch: "
                f"expected={spec.output_dimension}, observed={hidden_size}"
            )
        return cls(spec=spec, binding=binding, tokenizer=tokenizer, model=model)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [" ".join(str(text).split()) for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("semantic encoder requires non-empty normalized text")
        import torch

        tokens = self.tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(**tokens).last_hidden_state[:, 0]
        matrix = output.detach().cpu().to(torch.float32).numpy().astype(np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("semantic encoder returned invalid vectors")
        matrix = matrix / norms
        matrix = np.round(matrix, decimals=self.spec.output_round_decimals)
        renorm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(renorm, 1e-12)
        return matrix.astype(np.float64)

    def tokenization_telemetry(
        self, texts: Sequence[str], *, view_names: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Measure right-truncation without exposing token IDs or text."""

        normalized = [" ".join(str(text).split()) for text in texts]
        names = list(view_names or [f"view_{index}" for index in range(len(texts))])
        if len(names) != len(normalized) or len(set(names)) != len(names):
            raise ValueError("semantic token telemetry requires unique aligned view names")
        rows: dict[str, dict[str, Any]] = {}
        for name, text in zip(names, normalized, strict=True):
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            original = len(encoded["input_ids"])
            retained = min(original, int(self.spec.max_length))
            rows[name] = {
                "original_token_count": int(original),
                "encoded_token_count": int(retained),
                "truncated": bool(original > retained),
                "truncated_token_count": int(original - retained),
                "input_sha256": sha256_text(text),
            }
        return {
            "protocol": "pm-v1.5-semantic-tokenization-telemetry-v1",
            "max_length": int(self.spec.max_length),
            "truncation_side": str(getattr(self.tokenizer, "truncation_side", "right")),
            "views": rows,
        }

    def assemble_visible_dialogue_state(
        self,
        *,
        current_user_text: str,
        current_session_history: Sequence[Any],
        current_session_summary: str,
    ) -> tuple[str, dict[str, Any]]:
        """Build a <=max_length state while preserving recent context explicitly.

        Current text and summary receive frozen prefix budgets.  The remaining
        capacity is filled with the most-recent suffix of the chronological
        history.  The final string is re-tokenized without truncation and is
        shortened until it fits, so model-side implicit right truncation is
        never part of the reportable input contract.
        """

        def normalized(value: Any) -> str:
            return " ".join(str(value).split())

        def token_ids(text: str, *, special: bool = False) -> list[int]:
            encoded = self.tokenizer(
                text,
                add_special_tokens=special,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            return [int(value) for value in encoded["input_ids"]]

        def decode(ids: Sequence[int]) -> str:
            if not ids:
                return "[none]"
            return normalized(
                self.tokenizer.decode(
                    list(ids),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )

        current = normalized(current_user_text)
        summary = normalized(current_session_summary) or "[none]"
        turns: list[str] = []
        for turn in current_session_history:
            if isinstance(turn, Mapping):
                role = str(turn.get("role") or "")
                content = str(turn.get("content") or "")
            else:
                role = str(getattr(turn, "role", ""))
                content = str(getattr(turn, "content", ""))
            if role not in {"user", "assistant"} or not content.strip():
                raise ValueError("visible dialogue state contains an invalid turn")
            turns.append(f"{role}: {normalized(content)}")
        history = "\n".join(turns) or "[none]"

        original_ids = {
            "current_user": token_ids(current),
            "session_summary": token_ids(summary),
            "recent_dialogue": token_ids(history),
        }
        retained = {
            "current_user": original_ids["current_user"][: int(
                self.spec.current_user_state_token_budget
            )],
            "session_summary": original_ids["session_summary"][: int(
                self.spec.session_summary_token_budget
            )],
            # Start with the largest possible recent suffix; the exact final
            # capacity is established by the loop below, including headers.
            "recent_dialogue": original_ids["recent_dialogue"][-int(
                self.spec.max_length
            ):],
        }

        def render() -> str:
            return (
                f"CURRENT_USER:\n{decode(retained['current_user'])}\n\n"
                f"SESSION_SUMMARY:\n{decode(retained['session_summary'])}\n\n"
                "RECENT_DIALOGUE_MOST_RECENT_SUFFIX_CHRONOLOGICAL:\n"
                f"{decode(retained['recent_dialogue'])}"
            )

        state_text = render()
        while len(token_ids(state_text, special=True)) > int(self.spec.max_length):
            if retained["recent_dialogue"]:
                retained["recent_dialogue"].pop(0)
            elif retained["session_summary"]:
                retained["session_summary"].pop()
            elif retained["current_user"]:
                retained["current_user"].pop()
            else:  # pragma: no cover - fixed headers fit under every valid spec
                raise RuntimeError("visible-state headers exceed semantic max_length")
            state_text = render()
        final_tokens = len(token_ids(state_text, special=True))
        sections = {
            name: {
                "original_token_count": len(original_ids[name]),
                "retained_token_count": len(retained[name]),
                "dropped_token_count": len(original_ids[name]) - len(retained[name]),
                "input_sha256": sha256_text(
                    current
                    if name == "current_user"
                    else summary if name == "session_summary" else history
                ),
            }
            for name in ("current_user", "session_summary", "recent_dialogue")
        }
        return state_text, {
            "protocol": SEMANTIC_INPUT_PROTOCOL,
            "max_length": int(self.spec.max_length),
            "current_user_state_token_budget": int(
                self.spec.current_user_state_token_budget
            ),
            "session_summary_token_budget": int(
                self.spec.session_summary_token_budget
            ),
            "history_retention_policy": self.spec.history_retention_policy,
            "implicit_full_state_truncation": self.spec.implicit_full_state_truncation,
            "final_visible_state_token_count": final_tokens,
            "sections": sections,
        }


def semantic_runtime_attestation(
    encoder: FrozenTransformerSemanticEncoder,
) -> FrozenSemanticRuntimeContract:
    """Encode a public canary and bind it to exact runtime versions."""

    matrix = encoder.encode(SEMANTIC_CANARY_TEXTS)
    try:
        packages = {name: package_version(name) for name in SEMANTIC_RUNTIME_PACKAGES}
    except PackageNotFoundError as exc:
        raise RuntimeError(f"semantic runtime package is missing: {exc}") from exc
    try:
        parameter = next(encoder.model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("semantic runtime cannot determine model device/dtype") from exc
    dtype = str(parameter.dtype).removeprefix("torch.")
    return FrozenSemanticRuntimeContract(
        encoder_spec_sha256=encoder.binding.spec_sha256,
        encoder_snapshot_tree_sha256=encoder.binding.snapshot_tree_sha256,
        canary_texts_sha256=sha256_text(canonical_json(list(SEMANTIC_CANARY_TEXTS))),
        canary_matrix_sha256=sha256_text(canonical_json(matrix.tolist())),
        canary_shape=[int(value) for value in matrix.shape],
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        user_site_enabled=bool(site.ENABLE_USER_SITE),
        user_site_on_sys_path=site.getusersitepackages() in sys.path,
        packages=packages,
        device=str(parameter.device.type),
        dtype=dtype,
    )


def require_semantic_runtime_contract(
    config: Mapping[str, Any],
    encoder: FrozenTransformerSemanticEncoder,
) -> dict[str, Any]:
    expected = semantic_runtime_contract_from_config(config)
    observed = semantic_runtime_attestation(encoder)
    expected_payload = expected.model_dump(mode="json")
    observed_payload = observed.model_dump(mode="json")
    if observed_payload != expected_payload:
        mismatches = {
            key: {"expected": expected_payload.get(key), "observed": observed_payload.get(key)}
            for key in sorted(set(expected_payload) | set(observed_payload))
            if expected_payload.get(key) != observed_payload.get(key)
        }
        raise RuntimeError(
            "semantic runtime contract mismatch: " + canonical_json(mismatches)
        )
    return {
        "status": "PASS",
        "contract": observed_payload,
        "contract_sha256": observed.digest(),
    }


def require_recorded_semantic_runtime(
    config: Mapping[str, Any], recorded: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a development-data runtime record without re-encoding text."""

    expected = semantic_runtime_contract_from_config(config)
    if recorded.get("status") != "PASS":
        raise RuntimeError("development semantic runtime did not PASS")
    payload = recorded.get("contract")
    if payload != expected.model_dump(mode="json"):
        raise RuntimeError("development semantic runtime differs from frozen config")
    if recorded.get("contract_sha256") != expected.digest():
        raise RuntimeError("development semantic runtime digest mismatch")
    return {
        "status": "PASS",
        "contract_sha256": expected.digest(),
    }


def require_runtime_verification_matches_encoder(
    recorded: Mapping[str, Any], encoder: SemanticTextEncoder
) -> dict[str, Any]:
    """Validate a live verification before embedding it in a stage artifact."""

    if recorded.get("status") != "PASS":
        raise RuntimeError("semantic runtime verification did not PASS")
    contract = FrozenSemanticRuntimeContract.model_validate(recorded.get("contract"))
    if (
        contract.encoder_spec_sha256 != encoder.binding.spec_sha256
        or contract.encoder_snapshot_tree_sha256
        != encoder.binding.snapshot_tree_sha256
        or recorded.get("contract_sha256") != contract.digest()
    ):
        raise RuntimeError("semantic runtime verification/encoder binding mismatch")
    return {
        "status": "PASS",
        "contract": contract.model_dump(mode="json"),
        "contract_sha256": contract.digest(),
    }


def visible_dialogue_state_text(
    *,
    current_user_text: str,
    current_session_history: Sequence[Any],
    current_session_summary: str,
) -> str:
    turns: list[str] = []
    for turn in current_session_history:
        if isinstance(turn, Mapping):
            role = str(turn.get("role") or "")
            content = str(turn.get("content") or "")
        else:
            role = str(getattr(turn, "role", ""))
            content = str(getattr(turn, "content", ""))
        if role not in {"user", "assistant"} or not content.strip():
            raise ValueError("visible dialogue state contains an invalid turn")
        turns.append(f"{role}: {' '.join(content.split())}")
    return (
        f"CURRENT_USER:\n{' '.join(current_user_text.split())}\n\n"
        f"RECENT_DIALOGUE:\n{chr(10).join(turns)}\n\n"
        f"SESSION_SUMMARY:\n{' '.join(current_session_summary.split())}"
    )


def semantic_vector_similarity(
    query_vector: Sequence[float], centroid: Sequence[float]
) -> float:
    """Return a finite clipped cosine for already normalized semantic vectors."""

    query = np.asarray(query_vector, dtype=float)
    target = np.asarray(centroid, dtype=float)
    if (
        query.ndim != 1
        or target.ndim != 1
        or query.shape != target.shape
        or query.size == 0
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(target))
    ):
        raise RuntimeError("semantic query/catalog vectors are invalid or misaligned")
    query_norm = float(np.linalg.norm(query))
    target_norm = float(np.linalg.norm(target))
    if query_norm <= 0.0 or target_norm <= 0.0:
        raise RuntimeError("semantic query/catalog vectors must be non-zero")
    return float(
        np.clip((query / query_norm) @ (target / target_norm), -1.0, 1.0)
    )


def prepare_visible_semantic_state(
    encoder: SemanticTextEncoder,
    *,
    current_user_text: str,
    current_session_history: Sequence[Any],
    current_session_summary: str,
) -> PreparedVisibleSemanticState:
    """Assemble and encode the sole full-state query used by reportable PM-v1.5."""

    assembly_method = getattr(encoder, "assemble_visible_dialogue_state", None)
    if callable(assembly_method):
        state_text, section_allocation = assembly_method(
            current_user_text=current_user_text,
            current_session_history=current_session_history,
            current_session_summary=current_session_summary,
        )
    else:
        state_text = visible_dialogue_state_text(
            current_user_text=current_user_text,
            current_session_history=current_session_history,
            current_session_summary=current_session_summary,
        )
        section_allocation = None
    matrix = encoder.encode([current_user_text, state_text])
    expected = (2, encoder.spec.output_dimension)
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"semantic visible-state matrix has shape {matrix.shape}, expected {expected}"
        )
    current_vector = tuple(float(value) for value in matrix[0])
    state_vector = tuple(float(value) for value in matrix[1])
    flattened = tuple(float(value) for value in matrix.reshape(-1))
    telemetry_method = getattr(encoder, "tokenization_telemetry", None)
    tokenization = (
        telemetry_method(
            [current_user_text, state_text],
            view_names=["current_user_text", "visible_dialogue_state"],
        )
        if callable(telemetry_method)
        else {
            "protocol": "nonreportable-test-encoder-no-token-telemetry",
            "max_length": int(encoder.spec.max_length),
            "truncation_side": "unknown",
            "views": {},
        }
    )
    if section_allocation is not None:
        tokenization = {**tokenization, "section_allocation": section_allocation}
        visible = (tokenization.get("views") or {}).get("visible_dialogue_state") or {}
        if visible.get("truncated") is not False:
            raise RuntimeError(
                "section-aware visible state must not reach implicit tokenizer truncation"
            )
    visible_view = (tokenization.get("views") or {}).get(
        "visible_dialogue_state"
    ) or {}
    query_sha256 = visible_view.get("input_sha256")
    if not isinstance(query_sha256, str) or len(query_sha256) != 64:
        # Nonreportable fixture encoders need not implement telemetry, but their
        # semantic input must still be content addressed deterministically.
        query_sha256 = sha256_text(" ".join(state_text.split()))
    vector_sha256 = sha256_text(canonical_json(list(state_vector)))
    audit = {
        "protocol": SEMANTIC_INPUT_PROTOCOL,
        "semantic_query_protocol": UNIFIED_SEMANTIC_QUERY_PROTOCOL,
        "encoder_spec_sha256": encoder.binding.spec_sha256,
        "encoder_snapshot_tree_sha256": encoder.binding.snapshot_tree_sha256,
        "views": ["current_user_text", "visible_dialogue_state"],
        "per_view_dimension": encoder.spec.output_dimension,
        "combined_dimension": len(flattened),
        "tokenization": tokenization,
        "step0_semantic_query_sha256": query_sha256,
        "state_embedding_query_sha256": query_sha256,
        "step0_semantic_query_vector_sha256": vector_sha256,
        "state_embedding_query_vector_sha256": vector_sha256,
        "visible_input_sha256": sha256_text(
            canonical_json(
                {
                    "current_user_text": " ".join(current_user_text.split()),
                    "visible_dialogue_state": state_text,
                }
            )
        ),
    }
    audit["observation_sha256"] = sha256_text(canonical_json(audit))
    return PreparedVisibleSemanticState(
        query_text=state_text,
        current_user_vector=current_vector,
        state_query_vector=state_vector,
        combined_embedding=flattened,
        semantic_query_sha256=query_sha256,
        semantic_query_vector_sha256=vector_sha256,
        audit=audit,
    )


def encode_visible_state(
    encoder: SemanticTextEncoder,
    *,
    current_user_text: str,
    current_session_history: Sequence[Any],
    current_session_summary: str,
) -> tuple[list[float], dict[str, Any]]:
    """Compatibility wrapper around the unified semantic-state preparation."""

    prepared = prepare_visible_semantic_state(
        encoder,
        current_user_text=current_user_text,
        current_session_history=current_session_history,
        current_session_summary=current_session_summary,
    )
    return list(prepared.combined_embedding), dict(prepared.audit)


def summarize_semantic_truncation(states: Sequence[Any]) -> dict[str, Any]:
    """Aggregate per-state visible-text truncation without reading hidden text."""

    audits: list[Mapping[str, Any]] = []
    for state in states:
        provenance = getattr(state, "provenance", {}) or {}
        audits.append(provenance.get("semantic_observation") or {})
    return summarize_semantic_truncation_audits(audits)


def summarize_semantic_truncation_audits(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_view: dict[str, list[Mapping[str, Any]]] = {}
    by_section: dict[str, list[Mapping[str, Any]]] = {}
    unavailable = 0
    for semantic in audits:
        tokenization = semantic.get("tokenization") or {}
        views = tokenization.get("views") or {}
        if not views:
            unavailable += 1
            continue
        for name, row in views.items():
            by_view.setdefault(str(name), []).append(row)
        sections = (tokenization.get("section_allocation") or {}).get("sections") or {}
        for name, row in sections.items():
            by_section.setdefault(str(name), []).append(row)

    summaries: dict[str, Any] = {}
    for name, rows in sorted(by_view.items()):
        truncated = [bool(row.get("truncated")) for row in rows]
        lost = [int(row.get("truncated_token_count", 0)) for row in rows]
        original = [int(row.get("original_token_count", 0)) for row in rows]
        summaries[name] = {
            "state_count": len(rows),
            "truncated_state_count": int(sum(truncated)),
            "truncation_rate": float(np.mean(truncated)) if rows else 0.0,
            "total_truncated_tokens": int(sum(lost)),
            "maximum_truncated_tokens": max(lost, default=0),
            "maximum_original_tokens": max(original, default=0),
        }
    current = summaries.get("current_user_text") or {}
    visible = summaries.get("visible_dialogue_state") or {}
    section_summaries = {
        name: {
            "state_count": len(rows),
            "states_with_deliberate_drop": int(
                sum(int(row.get("dropped_token_count", 0)) > 0 for row in rows)
            ),
            "total_dropped_tokens": int(
                sum(int(row.get("dropped_token_count", 0)) for row in rows)
            ),
            "maximum_dropped_tokens": max(
                (int(row.get("dropped_token_count", 0)) for row in rows),
                default=0,
            ),
        }
        for name, rows in sorted(by_section.items())
    }
    return {
        "protocol": "pm-v1.5-semantic-truncation-summary-v1",
        "state_count": len(audits),
        "telemetry_unavailable_state_count": unavailable,
        "views": summaries,
        "section_allocation": section_summaries,
        "current_user_text_complete": (
            unavailable == 0 and int(current.get("truncated_state_count", -1)) == 0
        ),
        "implicit_visible_state_truncation_complete": (
            unavailable == 0 and int(visible.get("truncated_state_count", -1)) == 0
        ),
    }


def semantic_centroid(
    encoder: SemanticTextEncoder, texts: Sequence[str]
) -> tuple[float, ...]:
    if not texts:
        return tuple(0.0 for _ in range(encoder.spec.output_dimension))
    matrix = encoder.encode(texts)
    centroid = np.mean(matrix, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0.0 or not np.all(np.isfinite(centroid)):
        raise RuntimeError("semantic centroid is invalid")
    centroid = centroid / norm
    return tuple(
        round(float(value), encoder.spec.output_round_decimals) for value in centroid
    )


def semantic_query_similarity(
    encoder: SemanticTextEncoder,
    query_text: str,
    centroid: Sequence[float],
) -> float:
    vector = np.asarray(centroid, dtype=np.float64)
    if vector.shape != (encoder.spec.output_dimension,):
        raise ValueError("semantic centroid dimension does not match the encoder")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return 0.0
    query = encoder.encode([query_text])[0]
    return float(np.clip(query @ (vector / norm), -1.0, 1.0))
