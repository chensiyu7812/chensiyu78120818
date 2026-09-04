from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .contracts import MemoryItem, MemorySource
from .artifacts import require_artifact_attestation
from .io import canonical_json, read_json, sha256_file, sha256_text
from .pm_v2_contracts import PMV2State
from .retrieval import context_query
from .text import estimate_tokens, lexical_score


EVIDENCE_FILTER_MODEL_VERSION = "pm-v2-evidence-filter-model-v1"
EVIDENCE_FILTER_TRAINING_STAGE = "pm_v2_evidence_filter_training"


@dataclass(frozen=True)
class MemoryFilterExample:
    state_id: str
    card_id: str
    user_id: str
    current_user_text: str
    context_query_text: str
    session_index: int
    item: MemoryItem
    label: int
    label_reason: str


@dataclass
class PMV2EvidenceFilterModel:
    word_features: int = 512
    char_features: int = 512
    seed: int = 8841
    threshold: float = 0.5
    classifier: LogisticRegression | None = None
    training_report: dict[str, Any] = field(default_factory=dict)
    calibration_report: dict[str, Any] = field(default_factory=dict)
    internal_report: dict[str, Any] = field(default_factory=dict)
    format_version: str = EVIDENCE_FILTER_MODEL_VERSION
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        self._word = HashingVectorizer(
            n_features=self.word_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
            analyzer="word",
        )
        self._char = HashingVectorizer(
            n_features=self.char_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
        )

    @staticmethod
    def _pair_text(
        current_user_text: str,
        context_query_text: str,
        item: MemoryItem,
    ) -> str:
        return (
            f"CURRENT_USER:\n{current_user_text}\n\n"
            f"VISIBLE_CONTEXT:\n{context_query_text}\n\n"
            f"CANDIDATE_MEMORY[{item.source.value}]:\n{item.text}"
        )

    @staticmethod
    def _numeric(
        current_user_text: str,
        context_query_text: str,
        item: MemoryItem,
        session_index: int,
    ) -> list[float]:
        source_bits = [float(item.source is source) for source in MemorySource]
        age = max(0, int(session_index) - int(item.created_session))
        return [
            *source_bits,
            lexical_score(current_user_text, item.text),
            lexical_score(context_query_text, item.text),
            min(float(estimate_tokens(item.text)) / 256.0, 1.0),
            min(float(age) / max(float(session_index), 1.0), 1.0),
        ]

    def _matrix(
        self,
        rows: Sequence[tuple[str, str, MemoryItem, int]],
    ):
        if not rows:
            raise ValueError("evidence-filter feature matrix cannot be empty")
        texts = [self._pair_text(current, context, item) for current, context, item, _ in rows]
        numeric = np.asarray(
            [
                self._numeric(current, context, item, session_index)
                for current, context, item, session_index in rows
            ],
            dtype=np.float64,
        )
        return hstack(
            [self._word.transform(texts), self._char.transform(texts), csr_matrix(numeric)],
            format="csr",
        )

    def fit(self, examples: Sequence[MemoryFilterExample]) -> "PMV2EvidenceFilterModel":
        labels = np.asarray([int(row.label) for row in examples], dtype=int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("evidence-filter training requires both label classes")
        matrix = self._matrix(
            [
                (
                    row.current_user_text,
                    row.context_query_text,
                    row.item,
                    row.session_index,
                )
                for row in examples
            ]
        )
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=self.seed,
            solver="liblinear",
        )
        classifier.fit(matrix, labels)
        self.classifier = classifier
        self.training_report = {
            "status": "COMPLETE",
            "protocol": EVIDENCE_FILTER_MODEL_VERSION,
            "n_examples": len(examples),
            "n_states": len({row.state_id for row in examples}),
            "n_users": len({row.user_id for row in examples}),
            "positive_rate": float(np.mean(labels)),
            "label_contract": (
                "item_helpful_and_needed_source_and_not_stale_and_not_conflicting"
            ),
            "inference_feature_boundary": (
                "current_context_candidate_text_source_age_tokens_only"
            ),
        }
        return self

    def predict_probabilities(
        self, examples: Sequence[MemoryFilterExample]
    ) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("evidence-filter model is not fitted")
        matrix = self._matrix(
            [
                (
                    row.current_user_text,
                    row.context_query_text,
                    row.item,
                    row.session_index,
                )
                for row in examples
            ]
        )
        return self.classifier.predict_proba(matrix)[:, 1]

    def predict_helpfulness(
        self,
        *,
        current_user_text: str,
        context_query_text: str,
        item: MemoryItem,
        session_index: int,
    ) -> float:
        if self.classifier is None:
            raise RuntimeError("evidence-filter model is not fitted")
        matrix = self._matrix(
            [(current_user_text, context_query_text, item, int(session_index))]
        )
        return float(self.classifier.predict_proba(matrix)[0, 1])

    @staticmethod
    def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
        predicted = probabilities >= float(threshold)
        positive = labels == 1
        negative = ~positive
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & negative))
        tn = int(np.sum(~predicted & negative))
        fn = int(np.sum(~predicted & positive))
        return {
            "threshold": float(threshold),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
            "kept_rate": float(np.mean(predicted)),
            "accuracy": (tp + tn) / max(len(labels), 1),
        }

    def calibrate(
        self,
        examples: Sequence[MemoryFilterExample],
        *,
        thresholds: Sequence[float],
        minimum_recall: float,
        maximum_false_positive_rate: float,
    ) -> dict[str, Any]:
        labels = np.asarray([row.label for row in examples], dtype=int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("evidence-filter calibration requires both classes")
        probabilities = self.predict_probabilities(examples)
        candidates = [
            self._metrics(labels, probabilities, float(threshold))
            for threshold in thresholds
        ]
        feasible = [
            row
            for row in candidates
            if row["recall"] >= float(minimum_recall)
            and row["false_positive_rate"] <= float(maximum_false_positive_rate)
        ]
        if not feasible:
            report = {
                "status": "FAIL",
                "protocol": "pm-v2-evidence-filter-calibration-v1",
                "candidates": candidates,
                "minimum_recall": float(minimum_recall),
                "maximum_false_positive_rate": float(maximum_false_positive_rate),
            }
            self.calibration_report = report
            raise RuntimeError("evidence-filter calibration has no feasible threshold")
        selected = max(
            feasible,
            key=lambda row: (
                row["precision"],
                row["specificity"],
                -row["kept_rate"],
                row["threshold"],
            ),
        )
        self.threshold = float(selected["threshold"])
        self.calibration_report = {
            "status": "PASS",
            "protocol": "pm-v2-evidence-filter-calibration-v1",
            "n_examples": len(examples),
            "n_users": len({row.user_id for row in examples}),
            "selected": selected,
            "candidates": candidates,
            "minimum_recall": float(minimum_recall),
            "maximum_false_positive_rate": float(maximum_false_positive_rate),
        }
        return self.calibration_report

    def evaluate_internal(
        self,
        examples: Sequence[MemoryFilterExample],
        *,
        minimum_recall: float,
        minimum_precision: float,
        minimum_specificity: float,
    ) -> dict[str, Any]:
        labels = np.asarray([row.label for row in examples], dtype=int)
        probabilities = self.predict_probabilities(examples)
        metrics = self._metrics(labels, probabilities, self.threshold)
        checks = {
            "minimum_recall": metrics["recall"] >= float(minimum_recall),
            "minimum_precision": metrics["precision"] >= float(minimum_precision),
            "minimum_specificity": metrics["specificity"] >= float(minimum_specificity),
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "protocol": "pm-v2-evidence-filter-internal-gate-v1",
            "n_examples": len(examples),
            "n_users": len({row.user_id for row in examples}),
            "metrics": metrics,
            "checks": checks,
            "claim_boundary": (
                "synthetic item labels validate filter plumbing, not external usefulness"
            ),
        }
        self.internal_report = report
        if report["status"] != "PASS":
            raise RuntimeError("evidence-filter internal gate failed: " + canonical_json(report))
        return report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_sha256 = None
        joblib.dump(self, path)
        self.checkpoint_sha256 = sha256_file(path)

    @classmethod
    def load(cls, path: str | Path) -> "PMV2EvidenceFilterModel":
        path = Path(path)
        value = joblib.load(path)
        if not isinstance(value, cls) or value.format_version != EVIDENCE_FILTER_MODEL_VERSION:
            raise RuntimeError("unsupported PM-v2 evidence-filter checkpoint")
        if (
            value.classifier is None
            or value.training_report.get("status") != "COMPLETE"
            or value.calibration_report.get("status") != "PASS"
            or value.internal_report.get("status") != "PASS"
        ):
            raise RuntimeError("evidence-filter checkpoint lacks completed gates")
        value.checkpoint_sha256 = sha256_file(path)
        return value

    def contract_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "format_version": self.format_version,
                    "word_features": self.word_features,
                    "char_features": self.char_features,
                    "threshold": self.threshold,
                    "training_protocol": self.training_report.get("protocol"),
                    "calibration_protocol": self.calibration_report.get("protocol"),
                    "internal_protocol": self.internal_report.get("protocol"),
                }
            )
        )


def build_memory_filter_examples(
    states: Sequence[PMV2State],
    backends_by_card: Mapping[str, Sequence[MemoryItem]],
    evaluator_by_state: Mapping[str, Mapping[str, Any]],
) -> list[MemoryFilterExample]:
    examples: list[MemoryFilterExample] = []
    for state in states:
        context = evaluator_by_state[state.state_id]
        annotations = {
            str(row["memory_id"]): row for row in context["memory_annotations"]
        }
        needed = {str(value) for value in context["needed_memory_sources"]}
        query = context_query(
            state.current_user_text,
            [row.model_dump(mode="json") for row in state.current_session_history],
            state.current_session_summary,
        )
        items = list(backends_by_card[state.card_id])
        if set(annotations) != {item.memory_id for item in items}:
            raise RuntimeError(f"filter annotation/backend mismatch for {state.state_id}")
        for item in items:
            annotation = annotations[item.memory_id]
            useful = (
                annotation["item_utility"] == "helpful"
                and item.source.value in needed
                and not bool(annotation["stale"])
                and not bool(annotation["conflicts_with_current_state"])
            )
            reasons = []
            if item.source.value not in needed:
                reasons.append("source_not_needed")
            if annotation["item_utility"] != "helpful":
                reasons.append("item_" + str(annotation["item_utility"]))
            if annotation["stale"]:
                reasons.append("stale")
            if annotation["conflicts_with_current_state"]:
                reasons.append("conflicting")
            examples.append(
                MemoryFilterExample(
                    state_id=state.state_id,
                    card_id=state.card_id,
                    user_id=state.user_id,
                    current_user_text=state.current_user_text,
                    context_query_text=query,
                    session_index=state.session_index,
                    item=item,
                    label=int(useful),
                    label_reason="useful" if useful else "+".join(reasons),
                )
            )
    return examples


def require_evidence_filter_artifacts(
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    pm_v2_config_path: str | Path,
) -> tuple[PMV2EvidenceFilterModel, dict[str, Any]]:
    """Load only a content-verified, fully gated filter checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    report_path = Path(report_path)
    attestation_path = Path(attestation_path)
    pm_v2_config_path = Path(pm_v2_config_path)
    attestation = require_artifact_attestation(
        attestation_path,
        required_stage=EVIDENCE_FILTER_TRAINING_STAGE,
        required_output_paths={
            "checkpoint": checkpoint_path,
            "report": report_path,
        },
    )
    report = read_json(report_path)
    if report.get("status") != "COMPLETE":
        raise RuntimeError("evidence-filter training report is not COMPLETE")
    if (
        (report.get("calibration") or {}).get("status") != "PASS"
        or (report.get("internal") or {}).get("status") != "PASS"
    ):
        raise RuntimeError("evidence-filter calibration/internal gates did not PASS")
    if (report.get("semantic_sanity") or {}).get("status") != "PASS":
        raise RuntimeError("evidence-filter labels lack a PASS semantic-sanity audit")
    if report.get("pm_v2_config_sha256") != sha256_file(pm_v2_config_path):
        raise RuntimeError("evidence-filter checkpoint was trained under another PM-v2 config")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("evidence-filter checkpoint hash differs from its report")
    model = PMV2EvidenceFilterModel.load(checkpoint_path)
    model_contract_sha256 = model.contract_hash()
    if report.get("model_contract_sha256") != model_contract_sha256:
        raise RuntimeError("evidence-filter model contract differs from its report")
    binding = {
        "protocol": EVIDENCE_FILTER_MODEL_VERSION,
        "training_stage": EVIDENCE_FILTER_TRAINING_STAGE,
        "checkpoint_sha256": checkpoint_sha256,
        "model_contract_sha256": model_contract_sha256,
        "training_report_sha256": sha256_file(report_path),
        "attestation_sha256": attestation["attestation_sha256"],
        "synthetic_internal_gate_status": "PASS",
        "external_usefulness_status": "UNPROVEN",
    }
    return model, binding
