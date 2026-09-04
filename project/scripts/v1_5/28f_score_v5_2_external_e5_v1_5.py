#!/usr/bin/env python3
"""Freeze, then apply, the zero-API E5 external automatic metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import sys
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e5-automatic-scoring-v1"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_external_e5_scoring_v1.json"
E4_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1"
E4_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_execution_v1"
E2 = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3 = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
GOLD = E2 / "qa_evaluator_only_gold_private.jsonl"
BERT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/models--bert-base-uncased/snapshots"
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def official_prediction_surface(text: str) -> str:
    return str(text).replace("*", "").replace("#", "").strip("Answer:").strip()


def official_token_f1(gold: str, prediction: str) -> float:
    def tokens(text: str) -> list[str]:
        return re.sub(r"\W+", " ", str(text).lower()).strip().split()

    gold_tokens = tokens(gold)
    prediction_tokens = tokens(prediction)
    if not gold_tokens or not prediction_tokens:
        return 0.0
    common = set(gold_tokens) & set(prediction_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def snapshot_dir() -> Path:
    if not BERT_MODEL.is_dir():
        raise RuntimeError(
            "official E5 BERTScore requires a locally pinned bert-base-uncased snapshot"
        )
    candidates = sorted(path for path in BERT_MODEL.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError("E5 requires exactly one pinned bert-base-uncased snapshot")
    return candidates[0]


def snapshot_tree_sha256(path: Path) -> str:
    """Hash all resolved model files without machine-local path material."""

    records = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not records:
        raise RuntimeError("empty bert-base-uncased snapshot")
    return sha256_text(canonical_json(records))


class OfficialBertScore:
    def __init__(self) -> None:
        try:
            import bert_score
            import bert_score.utils
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("E5 requires the frozen bert-score==0.3.13 dependency") from exc
        if getattr(bert_score, "__version__", "") != "0.3.12":
            # PyPI 0.3.13 historically exposes __version__ 0.3.12. This oddity
            # is part of the package itself and is explicitly checked.
            raise RuntimeError("unexpected bert-score runtime version surface")
        self.bert_score = bert_score
        self.torch = torch
        self.device = "cpu"
        path = snapshot_dir()
        self.snapshot_id = path.name
        self.snapshot_tree_sha256 = snapshot_tree_sha256(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True)
        layers = int(bert_score.utils.model2layers["bert-base-uncased"])
        self.model.encoder.layer = torch.nn.ModuleList(
            list(self.model.encoder.layer[:layers])
        )
        self.model.eval().to(self.device)

    def compute(self, gold: str, prediction: str) -> float:
        idf = defaultdict(lambda: 1.0)
        idf[self.tokenizer.sep_token_id] = 0
        idf[self.tokenizer.cls_token_id] = 0
        result = self.bert_score.bert_cos_score_idf(
            self.model,
            [gold],
            [prediction],
            self.tokenizer,
            idf,
            device=self.device,
        ).cpu()
        return float(result[..., 2])


def freeze(out_dir: Path) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    if contract.get("status") != "FROZEN_BEFORE_E4_OUTCOMES":
        raise RuntimeError("E5 metric contract is not pre-outcome frozen")
    seal = {
        "protocol": PROTOCOL,
        "status": "E5_METRICS_FROZEN_BEFORE_E4_OUTCOMES",
        "contract_sha256": sha256_file(CONTRACT),
        "implementation_sha256": sha256_file(Path(__file__)),
        "gold_sha256": sha256_file(GOLD),
        "e4_execution_seal_sha256": sha256_file(E4_PLAN / "execution_seal.json"),
        "e3_retrieval_report_sha256": sha256_file(E3 / "retrieval_report.json"),
        "e4_outcomes_read": False,
        "api_calls": 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metric_freeze.json", seal)
    return seal


def score(out_dir: Path, outcomes_dir: Path) -> dict[str, Any]:
    frozen = read_json(out_dir / "metric_freeze.json")
    if frozen.get("status") != "E5_METRICS_FROZEN_BEFORE_E4_OUTCOMES":
        raise RuntimeError("E5 must be frozen before scoring")
    if frozen["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("E5 contract drifted after freeze")
    if frozen["implementation_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("E5 implementation drifted after freeze")
    e4_summary = read_json(outcomes_dir / "execution_summary.json")
    _e4_complete_statuses = {
        "COMPLETE_READY_FOR_E5_ZERO_API_SCORING",
        # 2026-08-05: e3raw_1f526d93a4d7a51e2161c504 (state_79702dbcf1f5ea0e6318)
        # was independently confirmed twice (cap 300 and cap 512) as deterministic
        # single-request repetition degeneration and permanently registered invalid
        # rather than retried, imputed, or adopted truncated. This status denotes
        # the resulting 4217-valid/1-registered-invalid completion; it is not a
        # numeric-result-driven change. See registered_invalid_generation.json in
        # the same outcomes-dir and V15-ENG-22/23 in the global failure ledger.
        "COMPLETE_WITH_ONE_REGISTERED_INVALID_RAW_GENERATION_READY_FOR_VERSIONED_E5_ZERO_API_SCORING",
    }
    if e4_summary.get("status") not in _e4_complete_statuses:
        raise RuntimeError("E5 requires complete E4 outcomes")

    qa = rows(outcomes_dir / "qa_outcomes_private.jsonl")
    gold_by_question = {str(row["question_id"]): row for row in rows(GOLD)}
    if len(qa) != 2090 or len(gold_by_question) != 418:
        raise RuntimeError("E5 QA denominator drifted")
    bert = OfficialBertScore()
    bert_binding = {
        "model_type": "bert-base-uncased",
        "snapshot_id": bert.snapshot_id,
        "snapshot_tree_sha256": bert.snapshot_tree_sha256,
        "local_files_only": True,
        "bound_before_first_item_scoring": True,
    }
    write_json(out_dir / "bert_model_binding.json", bert_binding)
    item_scores: list[dict[str, Any]] = []
    for outcome in qa:
        gold = gold_by_question[str(outcome["question_id"])]
        prediction = official_prediction_surface(str(outcome["final_output"]))
        item_scores.append(
            {
                "question_id": outcome["question_id"],
                "condition": outcome["condition"],
                "capability": gold["capability"],
                "raw_output": outcome["final_output"],
                "normalized_prediction": prediction,
                "token_f1": official_token_f1(str(gold["answer"]), prediction),
                "bertscore_f1": bert.compute(str(gold["answer"]), prediction),
            }
        )
    write_jsonl(out_dir / "qa_item_scores_private.jsonl", item_scores)

    aggregates: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in item_scores}):
        subset = [row for row in item_scores if row["condition"] == condition]
        for capability in ["ALL", *sorted({str(row["capability"]) for row in subset})]:
            block = (
                subset
                if capability == "ALL"
                else [row for row in subset if row["capability"] == capability]
            )
            aggregates.append(
                {
                    "condition": condition,
                    "capability": capability,
                    "n": len(block),
                    "mean_token_f1": sum(row["token_f1"] for row in block) / len(block),
                    "mean_bertscore_f1": sum(row["bertscore_f1"] for row in block) / len(block),
                }
            )
    write_jsonl(out_dir / "qa_aggregate_scores.jsonl", aggregates)

    response = rows(outcomes_dir / "response_core_outcomes_private.jsonl")
    bindings = rows(E2 / "response_core_policy_bindings_private.jsonl")
    core_by_state_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in response:
        core_by_state_action[(str(row["state_id"]), str(row["requested_action_id"]))].append(row)
    policy_rows = []
    for binding in bindings:
        calls = core_by_state_action[
            (str(binding["state_id"]), str(binding["realized_action_id"]))
        ]
        if len(calls) != 2:
            raise RuntimeError("E5 response policy binding does not resolve to two seeds")
        policy_rows.append(
            {
                **binding,
                "seed_calls": 2,
                "reported_prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in calls),
                "reported_completion_tokens": sum(
                    row["usage"]["completion_tokens"] for row in calls
                ),
            }
        )
    write_jsonl(out_dir / "response_policy_cost_rows_private.jsonl", policy_rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_READY_FOR_E6_E7",
        "qa_items": len(item_scores),
        "qa_questions": len(gold_by_question),
        "response_policy_bindings": len(policy_rows),
        "qa_item_scores_sha256": sha256_file(out_dir / "qa_item_scores_private.jsonl"),
        "qa_aggregate_scores_sha256": sha256_file(out_dir / "qa_aggregate_scores.jsonl"),
        "response_policy_cost_sha256": sha256_file(
            out_dir / "response_policy_cost_rows_private.jsonl"
        ),
        "bert_model_snapshot_id": bert.snapshot_id,
        "bert_model_snapshot_tree_sha256": bert.snapshot_tree_sha256,
        "bert_model_binding_sha256": sha256_file(out_dir / "bert_model_binding.json"),
        "api_calls": 0,
        "external_outcome_used_to_change_method": False,
    }
    write_json(out_dir / "scoring_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_external_e5_scoring_v1",
    )
    parser.add_argument("--outcomes-dir", type=Path, default=E4_EXECUTION)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E5 requires {FORMAL_PYTHON}; got {sys.executable}")
    if args.score:
        result = score(args.out_dir, args.outcomes_dir)
    else:
        result = freeze(args.out_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
