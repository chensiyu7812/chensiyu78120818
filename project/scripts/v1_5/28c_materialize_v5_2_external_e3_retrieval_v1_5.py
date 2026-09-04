#!/usr/bin/env python3
"""Run the zero-API E3 retrieval audit and seal exact external prompts.

The only learned model loaded here is the already-cached official BAAI/bge-m3
session retriever.  It selects sessions without gold.  Question answers,
capabilities and evidence are joined only after every retrieval decision and
generator message has been finalized, solely to compute evaluator-side
retrieval diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_external_memory_adapter import (
    qa_messages,
    raw_response_messages,
    render_indexed_session_fragments,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e3-retrieval-and-prompt-seal-v1"
E2_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E2_MANIFEST = E2_DIR / "plan_manifest.json"
E2_CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_external_e2_plan_v1.json"
V52_CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"
EVOEMO = ROOT / "data/external/evo_emo.json"
E1_HELPER = ROOT / "scripts/v1_5/28a_audit_v5_2_external_e1_v1_5.py"
SURFACE_HELPER = ROOT / "scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py"
BGE_M3 = Path(
    "/home/tokkio/.cache/huggingface/hub/models--BAAI--bge-m3/"
    "snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)
FORBIDDEN_GENERATION_KEYS = frozenset(
    {"answer", "answers", "evidence", "capability", "summaries", "question_group"}
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_tree_sha256(path: Path) -> str:
    records = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            records.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "sha256": sha256_file(item),
                }
            )
    if not records:
        raise RuntimeError("empty BGE snapshot")
    return sha256_text(canonical_json(records))


class BgeM3:
    def __init__(self, path: Path) -> None:
        if not path.is_dir():
            raise RuntimeError(f"local BGE-M3 snapshot missing: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True)
        if torch.cuda.is_available():
            index = 1 if torch.cuda.device_count() > 1 else 0
            self.device = torch.device(f"cuda:{index}")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        pieces: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = self.tokenizer(
                    list(texts[start : start + 8]),
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                vector = self.model(**batch).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
                pieces.append(vector.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(pieces, axis=0)


def fit_fragments(
    *,
    fragments: Sequence[str],
    message_builder: Callable[[Sequence[str]], list[dict[str, str]]],
    maximum_bound: int,
    safety_factor: float,
) -> tuple[list[str], list[dict[str, str]], int]:
    selected: list[str] = []
    messages = message_builder(selected)
    for fragment in fragments:
        candidate = [*selected, str(fragment)]
        candidate_messages = message_builder(candidate)
        bound = conservative_token_bound(
            canonical_json(candidate_messages), safety_factor=safety_factor
        )
        if bound > maximum_bound:
            break
        selected = candidate
        messages = candidate_messages
    final_bound = conservative_token_bound(
        canonical_json(messages), safety_factor=safety_factor
    )
    if final_bound > maximum_bound:
        raise RuntimeError("base prompt exceeds frozen context bound")
    return selected, messages, final_bound


def ranked_documents(
    *,
    query_vector: np.ndarray,
    documents: Sequence[Mapping[str, Any]],
    document_vectors: Mapping[tuple[str, str], np.ndarray],
    user_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    scored = []
    for chronological_index, document in enumerate(documents):
        session_id = str(document["session_id"])
        score = float(np.dot(query_vector, document_vectors[(user_id, session_id)]))
        scored.append((score, chronological_index, session_id, dict(document)))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    output: list[dict[str, Any]] = []
    for rank, (score, chronological_index, _session_id, document) in enumerate(
        scored[:top_k], start=1
    ):
        output.append(
            {
                **document,
                "rank": rank,
                "score": score,
                "chronological_index": chronological_index,
            }
        )
    return output


def typed_candidate_fragments(
    *,
    user: Mapping[str, Any],
    question_id: str,
    question: str,
    surface_helper: Any,
    e1_helper: Any,
    v52_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    state = {
        "state_id": question_id,
        "user_id": str(user["id"]),
        "visible_dialogue": [],
        "current_user_text": question,
        "current_session_index": len(user.get("dialog_history") or []) + 1,
    }
    sanitized_user = {
        "id": str(user["id"]),
        "basic_info": dict(user.get("basic_info") or {}),
        "dialog_history": list(user.get("dialog_history") or []),
    }
    components, audit = surface_helper.materialize_memory_components(
        state=state, user=sanitized_user
    )
    fixed_components: list[str] = []
    learned_components: list[str] = []
    details: dict[str, Any] = {}
    fragments_by_component: dict[str, str] = {}
    for component in ("MP", "MS", "ME"):
        item = components[component]
        candidate, failure = e1_helper.v52_candidate(
            component=component, item=item, user_id=str(user["id"])
        )
        features = {str(key): float(value) for key, value in item["model_features"].items()}
        probability = e1_helper.learned_probability(
            component=component, features=features, contract=v52_contract
        )
        threshold = float(v52_contract["frozen_method"]["heads"][component]["threshold"])
        executable = bool(
            candidate is not None
            and not bool(item["deterministic_hard_off"])
            and all(float(value) in {0.0, 0.5, 1.0} for value in features.values())
        )
        if candidate is not None:
            if component == "MP":
                literal = candidate.preference or candidate.profile_fact
            elif component == "MS":
                literal = candidate.prior_observation
            else:
                literal = " ".join(
                    value
                    for value in (
                        candidate.past_action,
                        candidate.observed_outcome,
                        candidate.mechanism,
                    )
                    if value
                )
            fragments_by_component[component] = (
                f"[{component}; strictly prior same-user record] {literal}"
            )
        if executable:
            fixed_components.append(component)
            if probability >= threshold:
                learned_components.append(component)
        details[component] = {
            "surface": item["surface"],
            "model_features": features,
            "typed_candidate": asdict(candidate) if candidate is not None else None,
            "v5_2_typed_failure": failure,
            "structurally_executable": executable,
            "learned_probability": probability,
            "learned_threshold": threshold,
            "learned_on": bool(executable and probability >= threshold),
        }
    return (
        {
            "question_id": question_id,
            "user_id": str(user["id"]),
            "components": details,
            "memory_transport_audit": audit,
            "RS": "STRUCTURALLY_UNAVAILABLE_FOR_FACTUAL_QA",
            "answer_evidence_capability_read": False,
        },
        [fragments_by_component[item] for item in fixed_components],
        [fragments_by_component[item] for item in learned_components],
    )


def evidence_sessions(user: Mapping[str, Any], evidence: Sequence[str]) -> tuple[set[str], list[str]]:
    sessions = {str(item["id"]) for item in user.get("dialog_history") or []}
    events = {
        str(item["id"]): str(item.get("conv_id") or "")
        for item in user.get("event_experience") or []
    }
    mapped: set[str] = set()
    missing: list[str] = []
    for raw in evidence:
        value = str(raw)
        prefix = value.split(":", 1)[0]
        if prefix in sessions:
            mapped.add(prefix)
        elif value in events and events[value] in sessions:
            mapped.add(events[value])
        else:
            missing.append(value)
    return mapped, missing


def ndcg_at_k(selected: Sequence[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, session_id in enumerate(selected[:k], start=1)
        if session_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E3 requires {FORMAL_PYTHON}; got {sys.executable}")
    e2_manifest = read_json(E2_MANIFEST)
    if e2_manifest.get("status") != "PASS_READY_FOR_E3_RETRIEVAL":
        raise RuntimeError("E3 requires the passed E2 manifest")
    contract = read_json(E2_CONTRACT)
    v52_contract = read_json(V52_CONTRACT)
    safety_factor = 1.5
    raw_cap = int(contract["response"]["raw_context_token_limit"])
    qa_cap = int(contract["qa"]["context_token_limit"])
    raw_requests = rows(E2_DIR / "response_raw_retrieval_requests_private.jsonl")
    qa_calls = rows(E2_DIR / "qa_logical_call_plan_private.jsonl")
    gold_rows = rows(E2_DIR / "qa_evaluator_only_gold_private.jsonl")
    gold_by_id = {str(row["question_id"]): row for row in gold_rows}
    evo_users = {
        str(row["id"]): row
        for row in json.loads(EVOEMO.read_text(encoding="utf-8"))
    }
    e1_helper = load_module("external_e1_helper", E1_HELPER)
    surface_helper = load_module("external_surface_helper", SURFACE_HELPER)

    # Build the complete text universe before loading gold into any retrieval
    # calculation.  Documents and queries are the only BGE inputs.
    documents_by_user: dict[str, list[dict[str, Any]]] = {}
    for request in raw_requests:
        user_id = str(request["user_id_private_analysis_only"])
        documents_by_user[user_id] = [dict(item) for item in request["same_user_session_documents"]]
    for call in qa_calls:
        user_id = str(call["user_id_private_analysis_only"])
        documents_by_user[user_id] = [dict(item) for item in call["same_user_session_documents"]]

    doc_keys: list[tuple[str, str]] = []
    doc_texts: list[str] = []
    for user_id, documents in sorted(documents_by_user.items()):
        for document in documents:
            doc_keys.append((user_id, str(document["session_id"])))
            doc_texts.append(str(document["text"]))
    query_keys: list[tuple[str, str]] = []
    query_texts: list[str] = []
    for request in raw_requests:
        if str(request["condition"]).startswith("raw_session_top4"):
            query_keys.append(("raw", str(request["retrieval_request_id"])))
            query_texts.append(str(request["query"]))
    for call in qa_calls:
        if str(call["condition"]) == "official_session_rag_top4":
            query_keys.append(("qa", str(call["question_id"])))
            query_texts.append(str(call["question"]))
    if len(query_keys) != len(set(query_keys)):
        raise RuntimeError("duplicate BGE query key")

    encoder = BgeM3(BGE_M3)
    doc_matrix = encoder.encode(doc_texts)
    query_matrix = encoder.encode(query_texts)
    document_vectors = {key: vector for key, vector in zip(doc_keys, doc_matrix, strict=True)}
    query_vectors = {key: vector for key, vector in zip(query_keys, query_matrix, strict=True)}

    raw_final_calls: list[dict[str, Any]] = []
    raw_retrieval_rows: list[dict[str, Any]] = []
    for request in raw_requests:
        user_id = str(request["user_id_private_analysis_only"])
        documents = [dict(item) for item in request["same_user_session_documents"]]
        condition = str(request["condition"])
        if condition.startswith("raw_session_top4"):
            ranked = ranked_documents(
                query_vector=query_vectors[("raw", str(request["retrieval_request_id"]))],
                documents=documents,
                document_vectors=document_vectors,
                user_id=user_id,
                top_k=4,
            )
            indexed = True
        else:
            ranked = [
                {**document, "rank": index, "score": None, "chronological_index": index - 1}
                for index, document in enumerate(documents, start=1)
            ]
            indexed = False
        fragments = render_indexed_session_fragments(ranked, indexed=indexed)
        selected_fragments, messages, input_bound = fit_fragments(
            fragments=fragments,
            message_builder=lambda values, request=request: raw_response_messages(
                current_context=str(request["query"]),
                session_fragments=values,
                strategy_instruction=str(request["strategy_instruction"]),
            ),
            maximum_bound=raw_cap,
            safety_factor=safety_factor,
        )
        kept = len(selected_fragments)
        selected_docs = ranked[:kept]
        raw_retrieval_rows.append(
            {
                "protocol": PROTOCOL,
                "retrieval_request_id": str(request["retrieval_request_id"]),
                "state_id": str(request["state_id"]),
                "user_id_private_analysis_only": user_id,
                "condition": condition,
                "selected_session_ids": [str(item["session_id"]) for item in selected_docs],
                "selected_scores": [item["score"] for item in selected_docs],
                "candidate_session_count": len(documents),
                "selected_before_context_fit": len(ranked),
                "selected_after_context_fit": kept,
                "owner_isolation": True,
                "gold_or_outcome_read": False,
            }
        )
        for seed_label in request["generation_seed_labels"]:
            seed_hex = stable_hex(
                PROTOCOL, str(request["state_id"]), condition, seed_label, n=16
            )
            raw_final_calls.append(
                {
                    "protocol": PROTOCOL,
                    "call_id": "e3raw_"
                    + stable_hex(PROTOCOL, str(request["state_id"]), condition, seed_label, n=24),
                    "experiment": str(request["experiment"]),
                    "partition": str(request["partition"]),
                    "state_id": str(request["state_id"]),
                    "user_id_private_analysis_only": user_id,
                    "condition": condition,
                    "selected_session_ids": [str(item["session_id"]) for item in selected_docs],
                    "strategy_candidate_id": request["strategy_candidate_id"],
                    "messages": messages,
                    "messages_sha256": sha256_text(canonical_json(messages)),
                    "input_token_upper_bound": input_bound,
                    "output_token_cap": int(request["output_token_cap"]),
                    "seed_label": seed_label,
                    "seed_hex": seed_hex,
                    "seed": (int(seed_hex, 16) % 2_147_483_647) or 1,
                    "retrieval_status": "E3_MATERIALIZED_AND_CONTEXT_FIT",
                    "response_or_outcome_read": False,
                }
            )

    typed_cache: dict[str, tuple[dict[str, Any], list[str], list[str]]] = {}
    qa_final_calls: list[dict[str, Any]] = []
    qa_retrieval_rows: list[dict[str, Any]] = []
    for call in qa_calls:
        condition = str(call["condition"])
        user_id = str(call["user_id_private_analysis_only"])
        question_id = str(call["question_id"])
        user = evo_users[user_id]
        documents = [dict(item) for item in call["same_user_session_documents"]]
        selected_session_ids: list[str] = []
        selected_scores: list[float] = []
        typed_components: list[str] = []
        if condition == "no_memory":
            messages = list(call["messages"])
            input_bound = int(call["input_token_upper_bound"])
        elif condition in {"full_history", "official_session_rag_top4"}:
            if condition == "full_history":
                ranked = [
                    {**document, "rank": index, "score": None, "chronological_index": index - 1}
                    for index, document in enumerate(documents, start=1)
                ]
                indexed = False
            else:
                ranked = ranked_documents(
                    query_vector=query_vectors[("qa", question_id)],
                    documents=documents,
                    document_vectors=document_vectors,
                    user_id=user_id,
                    top_k=4,
                )
                indexed = True
            fragments = render_indexed_session_fragments(ranked, indexed=indexed)
            selected_fragments, messages, input_bound = fit_fragments(
                fragments=fragments,
                message_builder=lambda values, call=call: qa_messages(
                    question=str(call["question"]), memory_fragments=values
                ),
                maximum_bound=qa_cap,
                safety_factor=safety_factor,
            )
            ranked = ranked[: len(selected_fragments)]
            selected_session_ids = [str(item["session_id"]) for item in ranked]
            selected_scores = [float(item["score"]) for item in ranked if item["score"] is not None]
        else:
            if question_id not in typed_cache:
                typed_cache[question_id] = typed_candidate_fragments(
                    user=user,
                    question_id=question_id,
                    question=str(call["question"]),
                    surface_helper=surface_helper,
                    e1_helper=e1_helper,
                    v52_contract=v52_contract,
                )
            typed_audit, fixed_fragments, learned_fragments = typed_cache[question_id]
            fragments = (
                fixed_fragments
                if condition == "typed_memory_fixed_high"
                else learned_fragments
            )
            typed_components = [fragment[1:3] for fragment in fragments]
            _selected, messages, input_bound = fit_fragments(
                fragments=fragments,
                message_builder=lambda values, call=call: qa_messages(
                    question=str(call["question"]), memory_fragments=values
                ),
                maximum_bound=qa_cap,
                safety_factor=safety_factor,
            )
            qa_retrieval_rows.append(
                {
                    "protocol": PROTOCOL,
                    "question_id": question_id,
                    "user_id_private_analysis_only": user_id,
                    "condition": condition,
                    "selected_typed_components": typed_components,
                    "typed_candidate_audit": typed_audit,
                    "gold_or_outcome_read": False,
                }
            )
        finalized = {
            key: value
            for key, value in call.items()
            if key not in {"same_user_session_documents", "messages", "messages_sha256", "input_token_upper_bound"}
        }
        finalized.update(
            {
                "protocol": PROTOCOL,
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "input_token_upper_bound": input_bound,
                "retrieval_status": "E3_MATERIALIZED_AND_CONTEXT_FIT",
                "selected_session_ids": selected_session_ids,
                "selected_session_scores": selected_scores,
                "selected_typed_components": typed_components,
                "answer_evidence_capability_visible_to_generation": False,
            }
        )
        qa_final_calls.append(finalized)

    # Evaluator-side retrieval scoring begins only after every generation row
    # has exact messages and hashes.  This join cannot change retrieval.
    top4_by_question = {
        str(row["question_id"]): row
        for row in qa_final_calls
        if str(row["condition"]) == "official_session_rag_top4"
    }
    retrieval_metrics: list[dict[str, Any]] = []
    for question_id, gold in gold_by_id.items():
        selected = list(top4_by_question[question_id]["selected_session_ids"])
        relevant, missing = evidence_sessions(
            evo_users[str(gold["user_id"])], list(gold["evidence"])
        )
        comparable = bool(relevant) and not missing
        retrieval_metrics.append(
            {
                "protocol": PROTOCOL,
                "question_id": question_id,
                "capability": str(gold["capability"]),
                "selected_session_ids": selected,
                "relevant_session_ids": sorted(relevant),
                "unmapped_evidence_ids": missing,
                "retrieval_metric_comparable": comparable,
                "recall_at_4": (
                    len(set(selected) & relevant) / len(relevant) if comparable else None
                ),
                "ndcg_at_4": ndcg_at_k(selected, relevant, 4) if comparable else None,
                "used_to_change_retrieval_or_prompt": False,
            }
        )

    forbidden = lambda value: any(
        str(key) in FORBIDDEN_GENERATION_KEYS
        or (isinstance(item, (dict, list)) and forbidden(item))
        for key, item in value.items()
    ) if isinstance(value, dict) else any(forbidden(item) for item in value) if isinstance(value, list) else False
    component_counts = Counter(
        component
        for row in qa_final_calls
        if str(row["condition"]) == "typed_memory_learned_pm"
        for component in row["selected_typed_components"]
    )
    comparable = [row for row in retrieval_metrics if row["retrieval_metric_comparable"]]
    checks = {
        "e2_passed": True,
        "bge_m3_snapshot_local": BGE_M3.is_dir(),
        "raw_retrieval_rows_276": len(raw_retrieval_rows) == 276,
        "raw_final_calls_552": len(raw_final_calls) == 552,
        "qa_final_calls_2090": len(qa_final_calls) == 2090,
        "qa_all_exact_messages": all(row.get("messages") for row in qa_final_calls),
        "qa_all_within_context_bound": all(
            int(row["input_token_upper_bound"]) <= qa_cap for row in qa_final_calls
        ),
        "raw_all_within_context_bound": all(
            int(row["input_token_upper_bound"]) <= raw_cap for row in raw_final_calls
        ),
        "generation_rows_exclude_gold_fields": not forbidden(qa_final_calls),
        "typed_memory_owner_isolation": all(
            all(
                item.get("typed_candidate") is None
                or str(item["typed_candidate"].get("owner_id")) == str(row["user_id_private_analysis_only"])
                for item in row["typed_candidate_audit"]["components"].values()
            )
            for row in qa_retrieval_rows
        ),
        "top4_exactly_four_sessions": all(
            len(row["selected_session_ids"]) == 4
            for row in qa_final_calls
            if str(row["condition"]) == "official_session_rag_top4"
        ),
        "retrieval_metrics_418": len(retrieval_metrics) == 418,
        "zero_api_calls": True,
        "zero_response_quality_risk_human_or_judge_outcomes_read": True,
    }
    status = "PASS_READY_FOR_E4_PAID_GENERATION" if all(checks.values()) else "FAIL_E3"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "response_raw_retrieval_audit_private.jsonl", raw_retrieval_rows)
    write_jsonl(args.out_dir / "response_raw_call_plan_private.jsonl", raw_final_calls)
    write_jsonl(args.out_dir / "qa_retrieval_and_typed_audit_private.jsonl", qa_retrieval_rows)
    write_jsonl(args.out_dir / "qa_final_call_plan_private.jsonl", qa_final_calls)
    write_jsonl(args.out_dir / "qa_session_retrieval_metrics_private.jsonl", retrieval_metrics)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "checks": checks,
        "input_sha256": {
            str(E2_MANIFEST.relative_to(ROOT)): sha256_file(E2_MANIFEST),
            str(E2_CONTRACT.relative_to(ROOT)): sha256_file(E2_CONTRACT),
            str(V52_CONTRACT.relative_to(ROOT)): sha256_file(V52_CONTRACT),
            str(EVOEMO.relative_to(ROOT)): sha256_file(EVOEMO),
        },
        "implementation_sha256": {
            "e3": sha256_file(Path(__file__)),
            "e1_helper": sha256_file(E1_HELPER),
            "surface_helper": sha256_file(SURFACE_HELPER),
            "external_adapter": sha256_file(
                ROOT / "src/metacom_pm/v1_5_external_memory_adapter.py"
            ),
        },
        "bge_m3": {
            "model_id": "BAAI/bge-m3",
            "snapshot_revision": BGE_M3.name,
            "snapshot_tree_sha256": snapshot_tree_sha256(BGE_M3),
            "pooling": "CLS",
            "normalize": True,
            "max_length": 8192,
            "device": str(encoder.device),
            "official_alignment": "same released model, session unit and Top-4; local deterministic tie break replaces FAISS unspecified ties",
        },
        "response": {
            "raw_retrieval_requests": len(raw_retrieval_rows),
            "raw_calls": len(raw_final_calls),
            "input_token_upper_bound": sum(
                int(row["input_token_upper_bound"]) for row in raw_final_calls
            ),
            "output_token_upper_bound": sum(
                int(row["output_token_cap"]) for row in raw_final_calls
            ),
            "maximum_input_token_upper_bound": max(
                int(row["input_token_upper_bound"]) for row in raw_final_calls
            ),
        },
        "qa": {
            "calls": len(qa_final_calls),
            "input_token_upper_bound": sum(
                int(row["input_token_upper_bound"]) for row in qa_final_calls
            ),
            "output_token_upper_bound": sum(
                int(row["output_token_cap"]) for row in qa_final_calls
            ),
            "maximum_input_token_upper_bound": max(
                int(row["input_token_upper_bound"]) for row in qa_final_calls
            ),
            "typed_learned_selected_component_counts": dict(sorted(component_counts.items())),
            "official_session_retrieval": {
                "comparable_questions": len(comparable),
                "not_comparable_questions": len(retrieval_metrics) - len(comparable),
                "mean_recall_at_4": (
                    sum(float(row["recall_at_4"]) for row in comparable) / len(comparable)
                    if comparable
                    else None
                ),
                "mean_ndcg_at_4": (
                    sum(float(row["ndcg_at_4"]) for row in comparable) / len(comparable)
                    if comparable
                    else None
                ),
                "metric_is_evaluator_side_only": True,
            },
        },
        "paid_runner_requirement": {
            "new_e4_runner_required": True,
            "require_paid_run_release": True,
            "e3_authorizes_no_api_call_by_itself": True,
        },
        "api_calls": 0,
        "human_or_llm_outcomes_read": 0,
    }
    write_json(args.out_dir / "retrieval_report.json", report)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "raw_calls": len(raw_final_calls),
                "qa_calls": len(qa_final_calls),
                "comparable_retrieval_questions": len(comparable),
                "api_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
