"""Factorized, train-only representation bakeoff for SupportNeedObservation.

This module is a diagnostic sidecar.  It deliberately does not change the
frozen PM-v1.5 semantic encoder or produce deployable labels.  It compares:

* transparent visible-text/structure baselines;
* frozen embedding candidates, including instruction-aware encoders;
* fixed natural-language-inference hypotheses; and
* one predeclared embedding + NLI + observable hybrid view.

All targets come from the already-bound, train-only human anchors.  Flat
five-way support mode is retained as a legacy diagnostic, while the primary
targets are overlapping, observable interaction axes.  No memory item,
retrieved strategy, supporter target, survey, outcome, internal split, or
external split is read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import math
import platform
import site
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator
from sklearn.feature_extraction.text import HashingVectorizer

from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import semantic_snapshot_tree_sha256
from .pm_v2_contracts import StrictModel
from .v1_5_support_need import (
    DIALOGUE_PHASE_IDS,
    NONCLINICAL_URGENCY_IDS,
    SUPPORT_MODE_IDS,
    cross_fit_categorical_soft_head,
    explicit_support_boundaries,
)


PROTOCOL = "pm-v1.5-support-need-factorized-representation-bakeoff-v1"
EMBEDDING_SPEC_PROTOCOL = (
    "pm-v1.5-support-need-frozen-embedding-candidate-v1"
)
NLI_SPEC_PROTOCOL = "pm-v1.5-support-need-frozen-nli-candidate-v1"
MODEL_BINDING_PROTOCOL = "pm-v1.5-support-need-candidate-binding-v1"
NLI_LABEL_IDS = ("contradiction", "entailment", "neutral")
PRIMARY_FACTOR_IDS = (
    "need_to_be_heard",
    "need_for_emotional_containment",
    "need_for_exploration",
    "advance_readiness",
    "focused_question_readiness",
    "advice_readiness",
)
DIAGNOSTIC_FACTOR_IDS = (
    "planning_readiness",
    "low_interaction_burden",
)
LEGACY_HEAD_IDS = (
    "legacy.support_mode",
    "legacy.dialogue_phase",
    "legacy.nonclinical_urgency",
)


FACTOR_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "factor_id": "need_to_be_heard",
        "target_rule": "human_goal_contains:be_heard",
        "positive_hypothesis": (
            "The user currently wants the supporter to listen and acknowledge "
            "their experience."
        ),
        "negative_hypothesis": (
            "The user currently wants the conversation to move away from "
            "listening toward analysis or action."
        ),
    },
    {
        "factor_id": "need_for_emotional_containment",
        "target_rule": "human_goal_contains:stabilize",
        "positive_hypothesis": (
            "The user currently needs emotional comfort or stabilization "
            "before more problem solving."
        ),
        "negative_hypothesis": (
            "The user is emotionally steady enough to focus on analysis or "
            "action without first needing comfort."
        ),
    },
    {
        "factor_id": "need_for_exploration",
        "target_rule": "human_goal_contains:make_sense",
        "positive_hypothesis": (
            "The user currently needs help making sense of their feelings, "
            "situation, or goal."
        ),
        "negative_hypothesis": (
            "The user's situation and immediate goal are already clear enough "
            "that further sense-making is not the main need."
        ),
    },
    {
        "factor_id": "advance_readiness",
        "target_rule": (
            "human_support_mode_in:explore,light_guidance,"
            "structured_planning"
        ),
        "positive_hypothesis": (
            "The user is ready for the conversation to advance beyond simple "
            "listening or reassurance."
        ),
        "negative_hypothesis": (
            "The conversation should currently stay with listening, "
            "acknowledgement, or reassurance rather than advance."
        ),
    },
    {
        "factor_id": "focused_question_readiness",
        "target_rule": "human_support_mode_equals:explore",
        "positive_hypothesis": (
            "One focused, low-burden question would fit the user's current need."
        ),
        "negative_hypothesis": (
            "Another exploratory question would not fit the user's current "
            "need."
        ),
    },
    {
        "factor_id": "advice_readiness",
        "target_rule": (
            "human_support_mode_in:light_guidance,structured_planning"
        ),
        "positive_hypothesis": (
            "The user is currently ready to receive an optional suggestion or "
            "guidance."
        ),
        "negative_hypothesis": (
            "The user is not currently ready for advice or guidance."
        ),
    },
    {
        "factor_id": "planning_readiness",
        "target_rule": "human_support_mode_equals:structured_planning",
        "positive_hypothesis": (
            "The user is ready to work through a structured multi-step plan."
        ),
        "negative_hypothesis": (
            "The user is not ready for a structured multi-step plan right now."
        ),
    },
    {
        "factor_id": "low_interaction_burden",
        "target_rule": "human_partial:recommended_low_interaction_burden",
        "positive_hypothesis": (
            "The next response should keep interaction burden very low, with "
            "at most one focus."
        ),
        "negative_hypothesis": (
            "The user can currently engage with multiple questions or steps."
        ),
    },
)


def factor_definition_sha256() -> str:
    return sha256_text(canonical_json(list(FACTOR_DEFINITIONS)))


class FrozenSupportNeedEmbeddingSpec(StrictModel):
    protocol: Literal[
        "pm-v1.5-support-need-frozen-embedding-candidate-v1"
    ] = EMBEDDING_SPEC_PROTOCOL
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pooling: Literal["cls", "last_token"]
    padding_side: Literal["left", "right"]
    instruction: str | None = None
    max_length: int = Field(ge=64, le=4096)
    hidden_dimension: int = Field(ge=16, le=8192)
    output_dimension: int = Field(ge=16, le=8192)
    output_round_decimals: int = Field(default=8, ge=4, le=12)
    local_files_only: Literal[True] = True
    trust_remote_code: Literal[False] = False

    @model_validator(mode="after")
    def coherent_embedding_contract(self):
        if self.output_dimension > self.hidden_dimension:
            raise ValueError("embedding output dimension exceeds hidden dimension")
        if self.instruction is not None:
            normalized = " ".join(self.instruction.split())
            if not normalized or normalized != self.instruction:
                raise ValueError(
                    "embedding instruction must be non-empty normalized text"
                )
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class FrozenSupportNeedNLISpec(StrictModel):
    protocol: Literal[
        "pm-v1.5-support-need-frozen-nli-candidate-v1"
    ] = NLI_SPEC_PROTOCOL
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_length: int = Field(ge=64, le=4096)
    label_ids: list[
        Literal["contradiction", "entailment", "neutral"]
    ] = list(NLI_LABEL_IDS)
    output_round_decimals: int = Field(default=10, ge=6, le=14)
    local_files_only: Literal[True] = True
    trust_remote_code: Literal[False] = False

    @model_validator(mode="after")
    def exact_nli_label_order(self):
        if tuple(self.label_ids) != NLI_LABEL_IDS:
            raise ValueError("NLI label IDs/order differ from frozen contract")
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class SupportNeedCandidateBinding(StrictModel):
    protocol: Literal[
        "pm-v1.5-support-need-candidate-binding-v1"
    ] = MODEL_BINDING_PROTOCOL
    candidate_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_file_count: int = Field(ge=1)
    implementation: str = Field(min_length=1)
    device: str = Field(min_length=1)
    parameter_dtype: Literal["float32"]


def _snapshot_binding(
    *,
    model_id: str,
    revision: str,
    expected_tree_sha256: str,
) -> tuple[Path, int]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - deployment preflight
        raise RuntimeError("candidate binding requires huggingface_hub") from exc
    root = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    observed = semantic_snapshot_tree_sha256(root)
    if observed != expected_tree_sha256:
        raise RuntimeError(
            "support-need candidate snapshot hash mismatch: "
            f"expected={expected_tree_sha256}, observed={observed}"
        )
    file_count = sum(path.is_file() for path in root.rglob("*"))
    if file_count < 1:
        raise RuntimeError("support-need candidate snapshot is empty")
    return root, file_count


def _model_device(model: Any) -> str:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("candidate model has no parameters") from exc
    return str(parameter.device)


@dataclass
class FrozenSupportNeedEmbeddingEncoder:
    spec: FrozenSupportNeedEmbeddingSpec
    binding: SupportNeedCandidateBinding
    tokenizer: Any
    model: Any
    batch_size: int

    @classmethod
    def load(
        cls,
        spec: FrozenSupportNeedEmbeddingSpec,
        *,
        device: str,
        batch_size: int,
    ) -> "FrozenSupportNeedEmbeddingEncoder":
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime preflight
            raise RuntimeError(
                "embedding candidate requires torch and transformers"
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA embedding runtime is unavailable")
        root, file_count = _snapshot_binding(
            model_id=spec.model_id,
            revision=spec.revision,
            expected_tree_sha256=spec.snapshot_tree_sha256,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            padding_side=spec.padding_side,
        )
        model = AutoModel.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        if hidden_size != spec.hidden_dimension:
            raise RuntimeError(
                "embedding hidden dimension mismatch: "
                f"expected={spec.hidden_dimension}, observed={hidden_size}"
            )
        binding = SupportNeedCandidateBinding(
            candidate_id=spec.candidate_id,
            model_id=spec.model_id,
            revision=spec.revision,
            spec_sha256=spec.digest(),
            snapshot_tree_sha256=spec.snapshot_tree_sha256,
            snapshot_file_count=file_count,
            implementation=(
                "transformers-auto-model-float32-"
                f"{spec.pooling}-normalized"
            ),
            device=_model_device(model),
            parameter_dtype="float32",
        )
        return cls(
            spec=spec,
            binding=binding,
            tokenizer=tokenizer,
            model=model,
            batch_size=int(batch_size),
        )

    def _model_text(self, text: str) -> str:
        normalized = " ".join(str(text).split())
        if not normalized:
            raise ValueError("embedding candidate received empty text")
        if self.spec.instruction is None:
            return normalized
        return f"Instruct: {self.spec.instruction}\nQuery:{normalized}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        prepared = [self._model_text(text) for text in texts]
        if not prepared:
            raise ValueError("embedding candidate requires at least one text")
        rows: list[np.ndarray] = []
        for start in range(0, len(prepared), self.batch_size):
            batch_text = prepared[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(self.binding.device)
                for key, value in tokens.items()
            }
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
            if self.spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = tokens["attention_mask"]
                if bool(torch.all(mask[:, -1] == 1)):
                    pooled = hidden[:, -1]
                else:
                    lengths = mask.sum(dim=1) - 1
                    pooled = hidden[
                        torch.arange(hidden.shape[0], device=hidden.device),
                        lengths,
                    ]
            pooled = pooled[:, : self.spec.output_dimension]
            matrix = pooled.detach().cpu().to(torch.float32).numpy()
            rows.append(np.asarray(matrix, dtype=np.float64))
        matrix = np.concatenate(rows, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("embedding candidate returned invalid vectors")
        matrix = matrix / norms
        matrix = np.round(matrix, decimals=self.spec.output_round_decimals)
        matrix = matrix / np.maximum(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
        )
        return np.asarray(matrix, dtype=np.float64)


@dataclass
class FrozenSupportNeedNLIScorer:
    spec: FrozenSupportNeedNLISpec
    binding: SupportNeedCandidateBinding
    tokenizer: Any
    model: Any
    batch_size: int

    @classmethod
    def load(
        cls,
        spec: FrozenSupportNeedNLISpec,
        *,
        device: str,
        batch_size: int,
    ) -> "FrozenSupportNeedNLIScorer":
        if batch_size < 1:
            raise ValueError("NLI batch size must be positive")
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - runtime preflight
            raise RuntimeError("NLI candidate requires transformers") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA NLI runtime is unavailable")
        root, file_count = _snapshot_binding(
            model_id=spec.model_id,
            revision=spec.revision,
            expected_tree_sha256=spec.snapshot_tree_sha256,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        observed_labels = tuple(
            str(model.config.id2label[index]).lower()
            for index in range(len(model.config.id2label))
        )
        if observed_labels != tuple(spec.label_ids):
            raise RuntimeError(
                "NLI label order mismatch: "
                f"expected={spec.label_ids}, observed={observed_labels}"
            )
        binding = SupportNeedCandidateBinding(
            candidate_id=spec.candidate_id,
            model_id=spec.model_id,
            revision=spec.revision,
            spec_sha256=spec.digest(),
            snapshot_tree_sha256=spec.snapshot_tree_sha256,
            snapshot_file_count=file_count,
            implementation=(
                "transformers-auto-sequence-classification-float32-softmax"
            ),
            device=_model_device(model),
            parameter_dtype="float32",
        )
        return cls(
            spec=spec,
            binding=binding,
            tokenizer=tokenizer,
            model=model,
            batch_size=int(batch_size),
        )

    def score(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> np.ndarray:
        import torch

        if len(premises) != len(hypotheses) or not premises:
            raise ValueError("NLI premises and hypotheses must be non-empty/aligned")
        normalized_premises = [" ".join(str(text).split()) for text in premises]
        normalized_hypotheses = [
            " ".join(str(text).split()) for text in hypotheses
        ]
        if any(not text for text in normalized_premises + normalized_hypotheses):
            raise ValueError("NLI candidate received empty text")
        rows: list[np.ndarray] = []
        for start in range(0, len(premises), self.batch_size):
            batch_premises = normalized_premises[
                start : start + self.batch_size
            ]
            batch_hypotheses = normalized_hypotheses[
                start : start + self.batch_size
            ]
            tokens = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(self.binding.device)
                for key, value in tokens.items()
            }
            with torch.inference_mode():
                logits = self.model(**tokens).logits
                probabilities = torch.softmax(logits, dim=1)
            rows.append(
                probabilities.detach().cpu().to(torch.float32).numpy()
            )
        matrix = np.asarray(np.concatenate(rows, axis=0), dtype=np.float64)
        matrix = np.round(matrix, decimals=self.spec.output_round_decimals)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        if matrix.shape != (len(premises), len(NLI_LABEL_IDS)):
            raise RuntimeError("NLI score matrix shape drifted")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("NLI candidate returned non-finite scores")
        return matrix


def candidate_runtime_attestation(
    binding: SupportNeedCandidateBinding,
) -> dict[str, Any]:
    package_names = (
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = "MISSING"
    payload = {
        "protocol": "pm-v1.5-support-need-candidate-runtime-v1",
        "binding": binding.model_dump(mode="json"),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "user_site_on_sys_path": site.getusersitepackages() in sys.path,
        "packages": packages,
        "device": binding.device,
        "parameter_dtype": binding.parameter_dtype,
    }
    return {
        **payload,
        "runtime_sha256": sha256_text(canonical_json(payload)),
    }


def _normalized(value: Any, *, empty: str = "[none]") -> str:
    text = " ".join(str(value or "").split())
    return text or empty


def _visible_texts(
    visible_state: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    current = _normalized(visible_state.get("current_user_text"))
    turns = list(visible_state.get("recent_dialogue") or [])
    last_assistant = next(
        (
            _normalized(turn.get("content"))
            for turn in reversed(turns)
            if turn.get("role") == "assistant"
            and str(turn.get("content") or "").strip()
        ),
        "[none]",
    )
    recent = "\n".join(
        f"{turn.get('role')}: {_normalized(turn.get('content'))}"
        for turn in turns
        if turn.get("role") in {"user", "assistant"}
        and str(turn.get("content") or "").strip()
    )
    summary = _normalized(visible_state.get("session_summary"))
    return current, last_assistant, _normalized(recent), summary


def _full_visible_text(visible_state: Mapping[str, Any]) -> str:
    current, _last_assistant, recent, summary = _visible_texts(visible_state)
    return (
        f"CURRENT_USER:\n{current}\n\n"
        f"RECENT_DIALOGUE:\n{recent}\n\n"
        f"SESSION_SUMMARY:\n{summary}"
    )


def _observable_features(visible_state: Mapping[str, Any]) -> np.ndarray:
    current, last_assistant, recent, summary = _visible_texts(visible_state)
    turns = list(visible_state.get("recent_dialogue") or [])
    boundaries = explicit_support_boundaries(current)
    return np.asarray(
        [
            math.log1p(len(current.split())),
            math.log1p(len(last_assistant.split())),
            math.log1p(len(recent.split())),
            math.log1p(len(summary.split())),
            math.log1p(len(turns)),
            math.log1p(sum(turn.get("role") == "user" for turn in turns)),
            math.log1p(
                sum(turn.get("role") == "assistant" for turn in turns)
            ),
            float("?" in current),
            float("!" in current),
            float(
                current.lower().startswith(
                    ("yes", "no", "yeah", "nope", "okay", "ok")
                )
            ),
            float(boundaries.advice_rejected),
            float(boundaries.advice_requested),
            float(boundaries.one_small_step_requested),
            float(boundaries.listen_first_requested),
            float(boundaries.question_or_task_burden_limit),
        ],
        dtype=float,
    )


def _lexical_text(visible_state: Mapping[str, Any]) -> str:
    current, last_assistant, recent, summary = _visible_texts(visible_state)
    return (
        f"[current] {current}\n[last_assistant] {last_assistant}\n"
        f"[history] {recent}\n[summary] {summary}"
    )


def transparent_feature_views(
    packet_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    visible_states = [dict(row["visible_state"]) for row in packet_rows]
    observable = np.vstack(
        [_observable_features(state) for state in visible_states]
    )
    lexical = HashingVectorizer(
        n_features=256,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
    ).transform([_lexical_text(state) for state in visible_states])
    return {
        "observable_structure": observable,
        "lexical_hash": np.concatenate(
            [np.asarray(lexical.toarray(), dtype=float), observable], axis=1
        ),
    }


def embedding_candidate_feature_views(
    encoder: FrozenSupportNeedEmbeddingEncoder,
    packet_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    visible_states = [dict(row["visible_state"]) for row in packet_rows]
    text_rows = [_visible_texts(state) for state in visible_states]
    flat_texts = [text for row in text_rows for text in row]
    encoded = np.asarray(encoder.encode(flat_texts), dtype=float)
    expected_shape = (
        len(packet_rows) * 4,
        int(encoder.spec.output_dimension),
    )
    if encoded.shape != expected_shape:
        raise RuntimeError(
            "embedding candidate matrix shape drifted: "
            f"expected={expected_shape}, observed={encoded.shape}"
        )
    matrices = encoded.reshape(
        len(packet_rows), 4, int(encoder.spec.output_dimension)
    )
    deltas = matrices[:, 0, :] - np.mean(matrices[:, 1:, :], axis=1)
    deltas = deltas / np.maximum(
        np.linalg.norm(deltas, axis=1, keepdims=True), 1e-12
    )
    multiview = np.concatenate(
        [matrices.reshape(len(packet_rows), -1), deltas], axis=1
    )
    observable = transparent_feature_views(packet_rows)[
        "observable_structure"
    ]
    prefix = f"embedding.{encoder.spec.candidate_id}"
    return {
        f"{prefix}.current": matrices[:, 0, :],
        f"{prefix}.multiview": multiview,
        f"{prefix}.multiview_plus_observable": np.concatenate(
            [multiview, observable], axis=1
        ),
    }


def nli_candidate_feature_bundle(
    scorer: FrozenSupportNeedNLIScorer,
    packet_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    visible_states = [dict(row["visible_state"]) for row in packet_rows]
    current_texts = [_visible_texts(state)[0] for state in visible_states]
    full_texts = [_full_visible_text(state) for state in visible_states]
    hypotheses = [
        definition[key]
        for definition in FACTOR_DEFINITIONS
        for key in ("negative_hypothesis", "positive_hypothesis")
    ]

    def score_context(contexts: Sequence[str]) -> np.ndarray:
        premises = [
            context for context in contexts for _ in range(len(hypotheses))
        ]
        aligned_hypotheses = hypotheses * len(contexts)
        scores = scorer.score(premises, aligned_hypotheses)
        return scores.reshape(
            len(contexts),
            len(FACTOR_DEFINITIONS),
            2,
            len(NLI_LABEL_IDS),
        )

    current_scores = score_context(current_texts)
    full_scores = score_context(full_texts)
    current_features = current_scores.reshape(len(packet_rows), -1)
    full_features = full_scores.reshape(len(packet_rows), -1)
    observable = transparent_feature_views(packet_rows)[
        "observable_structure"
    ]
    feature_views = {
        "nli.current": current_features,
        "nli.full_visible": full_features,
        "nli.current_plus_full": np.concatenate(
            [current_features, full_features], axis=1
        ),
        "nli.current_plus_full_plus_observable": np.concatenate(
            [current_features, full_features, observable], axis=1
        ),
    }
    return {
        "feature_views": feature_views,
        "current_scores": current_scores,
        "full_scores": full_scores,
        "factor_ids": [
            definition["factor_id"] for definition in FACTOR_DEFINITIONS
        ],
        "label_ids": list(NLI_LABEL_IDS),
        "hypothesis_contract_sha256": factor_definition_sha256(),
    }


def _one_hot(value: str, classes: Sequence[str]) -> dict[str, float]:
    if value not in classes:
        raise ValueError("support-need target is outside frozen classes")
    return {class_id: float(class_id == value) for class_id in classes}


def _binary_target(value: bool) -> dict[str, float]:
    return {
        "negative": 0.0 if value else 1.0,
        "positive": 1.0 if value else 0.0,
    }


def _factor_value(
    factor_id: str,
    normalized_anchor: Mapping[str, Any],
) -> bool | None:
    raw = dict(normalized_anchor["raw_annotation"])
    mode = str(raw.get("support_mode") or "")
    goals = {str(value) for value in raw.get("goals") or []}
    if factor_id == "need_to_be_heard":
        return "be_heard" in goals
    if factor_id == "need_for_emotional_containment":
        return "stabilize" in goals
    if factor_id == "need_for_exploration":
        return "make_sense" in goals
    if factor_id == "advance_readiness":
        return mode in {
            "explore",
            "light_guidance",
            "structured_planning",
        }
    if factor_id == "focused_question_readiness":
        return mode == "explore"
    if factor_id == "advice_readiness":
        return mode in {"light_guidance", "structured_planning"}
    if factor_id == "planning_readiness":
        return mode == "structured_planning"
    if factor_id == "low_interaction_burden":
        value = normalized_anchor.get(
            "human_recommended_low_interaction_burden"
        )
        return None if value is None else bool(value)
    raise ValueError(f"unknown factor target: {factor_id}")


def _target_specs(
    ordered_anchors: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for definition in FACTOR_DEFINITIONS:
        factor_id = definition["factor_id"]
        values = [_factor_value(factor_id, row) for row in ordered_anchors]
        indices = [index for index, value in enumerate(values) if value is not None]
        specs[factor_id] = {
            "class_ids": ("negative", "positive"),
            "indices": indices,
            "targets": [
                _binary_target(bool(values[index])) for index in indices
            ],
            "target_rule": definition["target_rule"],
            "target_role": (
                "human_partial_label"
                if factor_id == "low_interaction_burden"
                else "transparent_factorization_of_human_anchor"
            ),
        }
    specs["legacy.support_mode"] = {
        "class_ids": SUPPORT_MODE_IDS,
        "indices": list(range(len(ordered_anchors))),
        "targets": [
            _one_hot(
                str(row["raw_annotation"]["support_mode"]),
                SUPPORT_MODE_IDS,
            )
            for row in ordered_anchors
        ],
        "target_rule": "human_support_mode",
        "target_role": "legacy_flat_diagnostic_only",
    }
    specs["legacy.dialogue_phase"] = {
        "class_ids": DIALOGUE_PHASE_IDS,
        "indices": list(range(len(ordered_anchors))),
        "targets": [
            _one_hot(
                str(row["raw_annotation"]["dialogue_phase"]),
                DIALOGUE_PHASE_IDS,
            )
            for row in ordered_anchors
        ],
        "target_rule": "human_dialogue_phase",
        "target_role": "secondary_diagnostic",
    }
    specs["legacy.nonclinical_urgency"] = {
        "class_ids": NONCLINICAL_URGENCY_IDS,
        "indices": list(range(len(ordered_anchors))),
        "targets": [
            _one_hot(
                str(row["raw_annotation"]["nonclinical_urgency"]),
                NONCLINICAL_URGENCY_IDS,
            )
            for row in ordered_anchors
        ],
        "target_rule": "human_nonclinical_urgency",
        "target_role": "secondary_diagnostic",
    }
    return specs


def _prior_oof(
    targets: Sequence[Mapping[str, float]],
    groups: Sequence[str],
    class_ids: Sequence[str],
    weights: Sequence[float],
    folds: int = 5,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    y = np.asarray(
        [
            [float(target[class_id]) for class_id in class_ids]
            for target in targets
        ],
        dtype=float,
    )
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    sample_weights = np.asarray(weights, dtype=float)
    if (
        sample_weights.shape != (len(targets),)
        or np.any(sample_weights <= 0.0)
        or not np.all(np.isfinite(sample_weights))
    ):
        raise ValueError("OOF prior weights are invalid")
    predictions = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    for train_index, test_index in splitter.split(
        y, np.argmax(y, axis=1), group_values
    ):
        prior = np.maximum(
            np.average(
                y[train_index],
                axis=0,
                weights=sample_weights[train_index],
            ),
            1e-6,
        )
        predictions[test_index] = prior / prior.sum()
    return _classification_metrics(
        y=y,
        predictions=predictions,
        weights=sample_weights,
        class_ids=class_ids,
        include_predictions=True,
        protocol="pm-v1.5-support-need-oof-prior-v2",
    )


def _classification_metrics(
    *,
    y: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    class_ids: Sequence[str],
    include_predictions: bool,
    protocol: str,
) -> dict[str, Any]:
    clipped = np.clip(np.asarray(predictions, dtype=float), 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    truth_class = np.argmax(y, axis=1)
    predicted_class = np.argmax(clipped, axis=1)
    recalls: dict[str, float | None] = {}
    for index, class_id in enumerate(class_ids):
        mask = truth_class == index
        recalls[class_id] = (
            float(
                np.average(
                    predicted_class[mask] == index,
                    weights=weights[mask],
                )
            )
            if np.any(mask)
            else None
        )
    valid_recalls = [value for value in recalls.values() if value is not None]
    payload: dict[str, Any] = {
        "protocol": protocol,
        "soft_log_loss": float(
            np.average(-np.sum(y * np.log(clipped), axis=1), weights=weights)
        ),
        "brier_score": float(
            np.average(np.sum((clipped - y) ** 2, axis=1), weights=weights)
        ),
        "accuracy": float(
            np.average(predicted_class == truth_class, weights=weights)
        ),
        "balanced_accuracy": (
            float(np.mean(valid_recalls)) if valid_recalls else None
        ),
        "per_class_recall": recalls,
    }
    if include_predictions:
        payload["oof_predictions"] = [
            {
                class_id: float(clipped[row_index, class_index])
                for class_index, class_id in enumerate(class_ids)
            }
            for row_index in range(len(clipped))
        ]
    return payload


def _paired_group_bootstrap_loss_delta(
    *,
    targets: Sequence[Mapping[str, float]],
    class_ids: Sequence[str],
    groups: Sequence[str],
    weights: Sequence[float],
    reference_predictions: Sequence[Mapping[str, float]],
    candidate_predictions: Sequence[Mapping[str, float]],
    seed_key: str,
    resamples: int = 2000,
) -> dict[str, Any]:
    y = np.asarray(
        [
            [float(target[class_id]) for class_id in class_ids]
            for target in targets
        ],
        dtype=float,
    )

    def matrix(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        value = np.asarray(
            [
                [float(row[class_id]) for class_id in class_ids]
                for row in rows
            ],
            dtype=float,
        )
        return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)

    reference = matrix(reference_predictions)
    candidate = matrix(candidate_predictions)
    sample_weights = np.asarray(weights, dtype=float)
    if (
        sample_weights.shape != (len(groups),)
        or np.any(sample_weights <= 0.0)
        or not np.all(np.isfinite(sample_weights))
    ):
        raise ValueError("paired bootstrap weights are invalid")
    per_row_delta = -np.sum(y * np.log(np.clip(candidate, 1e-12, 1.0)), axis=1)
    per_row_delta -= -np.sum(
        y * np.log(np.clip(reference, 1e-12, 1.0)), axis=1
    )
    unique_groups = sorted(set(groups))
    by_group_delta = np.asarray(
        [
            np.average(
                per_row_delta[
                    [group == group_id for group in groups]
                ],
                weights=sample_weights[
                    [group == group_id for group in groups]
                ],
            )
            for group_id in unique_groups
        ],
        dtype=float,
    )
    by_group_weight = np.asarray(
        [
            np.mean(
                sample_weights[
                    [group == group_id for group in groups]
                ]
            )
            for group_id in unique_groups
        ],
        dtype=float,
    )
    seed = int(sha256_text(seed_key)[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(by_group_delta),
        size=(int(resamples), len(by_group_delta)),
    )
    samples = np.asarray(
        [
            np.average(
                by_group_delta[indices],
                weights=by_group_weight[indices],
            )
            for indices in sampled_indices
        ],
        dtype=float,
    )
    return {
        "protocol": "pm-v1.5-paired-dialogue-group-bootstrap-v1",
        "candidate_minus_reference_soft_log_loss": float(
            np.average(by_group_delta, weights=by_group_weight)
        ),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "resamples": int(resamples),
        "independent_groups": len(unique_groups),
        "negative_favors_candidate": True,
    }


def _nli_binary_probabilities(scores: np.ndarray) -> np.ndarray:
    contradiction_index = NLI_LABEL_IDS.index("contradiction")
    entailment_index = NLI_LABEL_IDS.index("entailment")
    support = (
        scores[..., entailment_index] - scores[..., contradiction_index]
    )
    shifted = support - np.max(support, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _direct_nli_reports(
    *,
    nli_bundle: Mapping[str, Any],
    target_specs: Mapping[str, Mapping[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    current_scores = np.asarray(nli_bundle["current_scores"], dtype=float)
    full_scores = np.asarray(nli_bundle["full_scores"], dtype=float)
    factor_ids = list(nli_bundle["factor_ids"])
    reports: dict[str, Any] = {}
    for factor_index, factor_id in enumerate(factor_ids):
        spec = target_specs[factor_id]
        indices = list(spec["indices"])
        targets = list(spec["targets"])
        counts = Counter(
            max(target, key=target.get) for target in targets
        )
        if len(indices) < 4 or any(
            counts.get(class_id, 0) < 2
            for class_id in ("negative", "positive")
        ):
            reports[factor_id] = {
                "status": "UNSUPPORTED_CLASS_HAS_FEWER_THAN_TWO_ANCHORS",
                "class_counts": dict(sorted(counts.items())),
                "labeled_rows": len(indices),
            }
            continue
        y = np.asarray(
            [
                [target["negative"], target["positive"]]
                for target in targets
            ],
            dtype=float,
        )
        selected_weights = np.asarray(
            [weights[index] for index in indices], dtype=float
        )
        current = _nli_binary_probabilities(
            current_scores[indices, factor_index]
        )
        full = _nli_binary_probabilities(full_scores[indices, factor_index])
        mean = (current + full) / 2.0
        reports[factor_id] = {
            "status": "COMPLETE_FIXED_HYPOTHESIS_DIAGNOSTIC",
            "class_counts": dict(sorted(counts.items())),
            "probability_mapping": (
                "softmax_over_negative_positive_hypotheses_of_"
                "entailment_minus_contradiction"
            ),
            "current": _classification_metrics(
                y=y,
                predictions=current,
                weights=selected_weights,
                class_ids=("negative", "positive"),
                include_predictions=False,
                protocol="pm-v1.5-fixed-nli-hypothesis-score-v1",
            ),
            "full_visible": _classification_metrics(
                y=y,
                predictions=full,
                weights=selected_weights,
                class_ids=("negative", "positive"),
                include_predictions=False,
                protocol="pm-v1.5-fixed-nli-hypothesis-score-v1",
            ),
            "current_full_mean": _classification_metrics(
                y=y,
                predictions=mean,
                weights=selected_weights,
                class_ids=("negative", "positive"),
                include_predictions=False,
                protocol="pm-v1.5-fixed-nli-hypothesis-score-v1",
            ),
            "calibration_fitted": False,
            "automatic_gold_label": False,
        }
    return reports


def run_factorized_support_need_bakeoff(
    *,
    packet_rows: Sequence[Mapping[str, Any]],
    normalized_anchor_rows: Sequence[Mapping[str, Any]],
    feature_views: Mapping[str, np.ndarray],
    nli_bundle: Mapping[str, Any],
    candidate_runtimes: Sequence[Mapping[str, Any]],
    hybrid_embedding_candidate_id: str,
    source_lineage: Mapping[str, Any] | None = None,
    expansion_human_annotations_opened: bool = False,
) -> dict[str, Any]:
    """Evaluate factorized targets using only group-held-out train anchors."""

    packet_by_id = {
        str(row["blind_item_id"]): dict(row) for row in packet_rows
    }
    anchor_by_id = {
        str(row["blind_item_id"]): dict(row)
        for row in normalized_anchor_rows
    }
    if len(packet_by_id) != len(packet_rows):
        raise RuntimeError("support-need bakeoff packet contains duplicate IDs")
    if set(packet_by_id) != set(anchor_by_id):
        raise RuntimeError(
            "support-need bakeoff anchors do not exactly cover packet"
        )
    ordered_packet_ids = [str(row["blind_item_id"]) for row in packet_rows]
    non_abstain_positions = [
        index
        for index, row_id in enumerate(ordered_packet_ids)
        if not bool(anchor_by_id[row_id]["raw_annotation"]["abstain"])
    ]
    if len(non_abstain_positions) < 15:
        raise RuntimeError("too few non-abstaining anchors for bakeoff")
    ordered_ids = [ordered_packet_ids[index] for index in non_abstain_positions]
    ordered_anchors = [anchor_by_id[row_id] for row_id in ordered_ids]
    row_count = len(packet_rows)
    parsed_views: dict[str, np.ndarray] = {}
    for name, value in feature_views.items():
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != row_count:
            raise RuntimeError(f"bakeoff feature view {name} has invalid shape")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"bakeoff feature view {name} is non-finite")
        parsed_views[str(name)] = matrix[non_abstain_positions]
    required_baselines = {"observable_structure", "lexical_hash"}
    if not required_baselines.issubset(parsed_views):
        raise RuntimeError("bakeoff lacks transparent baseline feature views")
    hybrid_embedding_view = (
        f"embedding.{hybrid_embedding_candidate_id}.multiview"
    )
    if hybrid_embedding_view not in parsed_views:
        raise RuntimeError("bakeoff hybrid embedding candidate is absent")
    nli_hybrid_source = "nli.current_plus_full"
    if nli_hybrid_source not in parsed_views:
        raise RuntimeError("bakeoff NLI hybrid source is absent")
    hybrid_view_name = (
        f"hybrid.{hybrid_embedding_candidate_id}.nli_observable"
    )
    parsed_views[hybrid_view_name] = np.concatenate(
        [
            parsed_views[hybrid_embedding_view],
            parsed_views[nli_hybrid_source],
            parsed_views["observable_structure"],
        ],
        axis=1,
    )
    view_order = list(parsed_views)
    weights = [
        float(row["raw_annotation"]["confidence"]) / 5.0
        for row in ordered_anchors
    ]
    target_specs = _target_specs(ordered_anchors)
    head_reports: dict[str, Any] = {}
    for head_id in (
        *PRIMARY_FACTOR_IDS,
        *DIAGNOSTIC_FACTOR_IDS,
        *LEGACY_HEAD_IDS,
    ):
        spec = target_specs[head_id]
        indices = list(spec["indices"])
        targets = list(spec["targets"])
        class_ids = tuple(spec["class_ids"])
        groups = [ordered_ids[index] for index in indices]
        selected_weights = [weights[index] for index in indices]
        class_counts = Counter(
            max(target, key=target.get) for target in targets
        )
        base = {
            "target_rule": spec["target_rule"],
            "target_role": spec["target_role"],
            "class_ids": list(class_ids),
            "class_counts": dict(sorted(class_counts.items())),
            "labeled_rows": len(indices),
            "independent_dialogue_groups": len(set(groups)),
        }
        if len(set(groups)) < 5 or any(
            class_counts.get(class_id, 0) < 2 for class_id in class_ids
        ):
            head_reports[head_id] = {
                **base,
                "status": "UNSUPPORTED_CLASS_HAS_FEWER_THAN_TWO_ANCHORS",
            }
            continue
        prior = _prior_oof(
            targets,
            groups,
            class_ids,
            selected_weights,
        )
        views: dict[str, Any] = {}
        for view_name in view_order:
            report = cross_fit_categorical_soft_head(
                features=parsed_views[view_name][indices],
                targets=targets,
                groups=groups,
                class_ids=class_ids,
                base_weights=selected_weights,
                projection_dimensions=(2, 4, 8),
                alphas=(10.0, 100.0),
                folds=5,
            )
            views[view_name] = {
                **report,
                "log_loss_improvement_over_oof_prior": (
                    prior["soft_log_loss"] - report["soft_log_loss"]
                ),
                "accuracy_improvement_over_oof_prior": (
                    report["accuracy"] - prior["accuracy"]
                ),
            }
        reference_predictions = views["lexical_hash"]["oof_predictions"]
        comparisons = {
            view_name: _paired_group_bootstrap_loss_delta(
                targets=targets,
                class_ids=class_ids,
                groups=groups,
                weights=selected_weights,
                reference_predictions=reference_predictions,
                candidate_predictions=report["oof_predictions"],
                seed_key=f"{PROTOCOL}:{head_id}:{view_name}",
            )
            for view_name, report in views.items()
            if view_name != "lexical_hash"
        }
        best_view = min(
            views,
            key=lambda name: (
                views[name]["soft_log_loss"],
                view_order.index(name),
            ),
        )
        best_report = views[best_view]
        represented_recall_count = sum(
            value is not None and float(value) > 0.0
            for value in best_report["per_class_recall"].values()
        )
        noncollapsed_views = [
            name
            for name, report in views.items()
            if all(
                value is not None and float(value) > 0.0
                for value in report["per_class_recall"].values()
            )
        ]
        best_noncollapsed_view = (
            min(
                noncollapsed_views,
                key=lambda name: (
                    views[name]["soft_log_loss"],
                    view_order.index(name),
                ),
            )
            if noncollapsed_views
            else None
        )
        hybrid_report = views[hybrid_view_name]
        hybrid_comparison = comparisons[hybrid_view_name]
        head_reports[head_id] = {
            **base,
            "status": "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD_DIAGNOSTIC",
            "oof_prior": prior,
            "feature_views": views,
            "paired_loss_delta_vs_lexical": comparisons,
            "best_view_by_soft_log_loss": best_view,
            "best_view_all_classes_nonzero_recall": (
                represented_recall_count == len(class_ids)
            ),
            "best_noncollapsed_view_by_soft_log_loss": (
                best_noncollapsed_view
            ),
            "best_noncollapsed_soft_log_loss": (
                views[best_noncollapsed_view]["soft_log_loss"]
                if best_noncollapsed_view is not None
                else None
            ),
            "predeclared_hybrid_view": hybrid_view_name,
            "predeclared_hybrid_checks": {
                "positive_log_loss_improvement_over_prior": (
                    hybrid_report["log_loss_improvement_over_oof_prior"] > 0.0
                ),
                "positive_log_loss_improvement_over_lexical": (
                    hybrid_comparison[
                        "candidate_minus_reference_soft_log_loss"
                    ]
                    < 0.0
                ),
                "paired_ci_excludes_zero_in_favor_of_hybrid": (
                    hybrid_comparison["ci95"][1] < 0.0
                ),
                "all_classes_have_nonzero_recall": all(
                    value is not None and float(value) > 0.0
                    for value in hybrid_report["per_class_recall"].values()
                ),
            },
        }

    direct_nli = _direct_nli_reports(
        nli_bundle=nli_bundle,
        target_specs=target_specs,
        weights=weights,
    )
    primary_hybrid_checks = {
        factor_id: head_reports[factor_id]["predeclared_hybrid_checks"]
        for factor_id in PRIMARY_FACTOR_IDS
        if head_reports[factor_id]["status"]
        == "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD_DIAGNOSTIC"
    }
    primary_all_hybrid_checks_pass = bool(primary_hybrid_checks) and all(
        all(checks.values()) for checks in primary_hybrid_checks.values()
    )
    report_core = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_DIAGNOSTIC_NOT_FORMAL_FIT",
        "non_abstaining_anchor_count": len(ordered_ids),
        "independent_dialogue_groups": len(set(ordered_ids)),
        "factor_definition_sha256": factor_definition_sha256(),
        "primary_factor_ids": list(PRIMARY_FACTOR_IDS),
        "diagnostic_factor_ids": list(DIAGNOSTIC_FACTOR_IDS),
        "legacy_head_ids": list(LEGACY_HEAD_IDS),
        "feature_view_order": view_order,
        "predeclared_hybrid_view": hybrid_view_name,
        "heads": head_reports,
        "fixed_nli_hypothesis_diagnostics": direct_nli,
        "primary_factor_predeclared_hybrid_checks": primary_hybrid_checks,
        "primary_all_hybrid_checks_pass": primary_all_hybrid_checks_pass,
        "representation_promotion_authorized": False,
        "formal_fit_authorized": False,
        "expansion_human_annotations_opened": bool(
            expansion_human_annotations_opened
        ),
        "automatic_gold_labels_created": False,
        "candidate_runtimes": [dict(row) for row in candidate_runtimes],
        "source_lineage": dict(source_lineage or {}),
        "data_roles": {
            "human_mode_goal_phase_urgency": (
                "train_only_partial_anchor_not_action_gold"
            ),
            "low_interaction_burden": (
                "human_contextual_partial_label_missing_is_not_negative"
            ),
            "embedding_outputs": "representation_features_not_labels",
            "nli_outputs": (
                "fixed_hypothesis_scores_and_features_not_labels"
            ),
        },
        "esconv_metadata_used_as_features": False,
        "target_supporter_responses_used": False,
        "target_strategy_annotations_used": False,
        "survey_outcomes_used": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "interpretation": (
            (
                "expanded train-only anchor representation diagnostic after "
                "human fit-subset adjudication; the untouched confirmation "
                "subset remains sealed, and no result here authorizes a "
                "formal need head or PM fit"
            )
            if expansion_human_annotations_opened
            else (
                "same-anchor representation diagnostic only; the prepared "
                "fresh expansion packet remains unopened until this report "
                "is reviewed, and no result here authorizes a formal need "
                "head or PM fit"
            )
        ),
    }
    return {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
