#!/usr/bin/env python3
"""Use one audited LLM search pass to prepare the final 36-core relink page."""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from metacom_pm.api import make_client
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import iter_jsonl, sha256_file, stable_hex


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h1-source-relink-llm-preselection-v1"


class CoreCandidateRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_candidate_numbers: list[int]
    estimated_direct_fit_count: int
    selection_confidence: Literal["high", "medium", "low"]
    short_reason: str


def _load_prepare_module():
    path = ROOT / "scripts/v1_5/24ak_prepare_h1_source_relink_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("h1_source_prepare", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_surface(row: dict[str, Any], number: int) -> str:
    dialogue = list(row.get("recent_dialogue") or [])[-3:]
    context = "\n".join(
        f"{turn.get('speaker')}: {str(turn.get('content') or '').strip()}"
        for turn in dialogue
    )
    return (
        f"CANDIDATE {number}\n"
        f"VISIBLE CONTEXT:\n{context}\n"
        f"TARGET SUPPORTER RESPONSE:\n"
        f"{str(row['supporter_response']).strip()}"
    )


def _messages(
    core: dict[str, Any],
    numbered_rows: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, str]]:
    candidates = "\n\n".join(
        _candidate_surface(row, number) for number, row in numbered_rows
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a source-evidence search assistant, not the final "
                "annotator. Rank examples only by whether the TARGET SUPPORTER "
                "RESPONSE directly and literally performs the specified atomic "
                "support move in the visible context. Generic friendliness, "
                "topical similarity, a good overall response, or an action "
                "appearing only in the context do not count. Respect the "
                "when-to-use cue. Do not reward unsafe promises, prescriptions, "
                "unsupported inference, or a different support move. Return "
                "exactly 10 unique candidate numbers, best direct evidence "
                "first. If fewer than 10 are true direct fits, rank the closest "
                "remaining candidates last and report the estimated true count."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ATOMIC MOVE: {core['support_move']}\n"
                f"WHEN TO USE: {core['when_to_use']}\n"
                f"DO NOT USE: {core['when_not_to_use']}\n\n"
                f"{candidates}"
            ),
        },
    ]


def _validate_ranking(
    ranking: CoreCandidateRanking,
    *,
    available_numbers: set[int],
) -> None:
    values = list(ranking.ranked_candidate_numbers)
    if len(values) != 10 or len(set(values)) != 10:
        raise RuntimeError("preselector must return exactly 10 unique candidates")
    if not set(values) <= available_numbers:
        raise RuntimeError("preselector returned an unknown candidate number")
    if not 0 <= ranking.estimated_direct_fit_count <= len(available_numbers):
        raise RuntimeError("estimated_direct_fit_count out of range")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _final_selection(
    *,
    cores: list[str],
    shuffled_by_core: dict[str, list[dict[str, Any]]],
    outcomes: dict[str, dict[str, Any]],
    examples_per_core: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_dialogues: set[str] = set()
    used_responses: set[str] = set()
    problem_counts: dict[str, defaultdict[str, int]] = {
        core: defaultdict(int) for core in cores
    }
    ordered_rows: dict[str, list[dict[str, Any]]] = {}
    for core_id in cores:
        rows = shuffled_by_core[core_id]
        by_number = {index: row for index, row in enumerate(rows, start=1)}
        preferred = list(outcomes[core_id]["ranked_candidate_numbers"])
        preferred.extend(
            number
            for number in range(1, len(rows) + 1)
            if number not in preferred
        )
        ordered_rows[core_id] = [by_number[number] for number in preferred]

    # Allocate one example per core per round, so an early core cannot consume
    # all globally unique dialogues before later cores are considered.
    for _ in range(examples_per_core):
        for core_id in sorted(cores):
            choice = None
            for row in ordered_rows[core_id]:
                dialogue = str(row["source_dialogue_id"])
                response = " ".join(
                    str(row["supporter_response"]).casefold().split()
                )
                problem = str(row.get("problem_type") or "unknown")
                if dialogue in used_dialogues or response in used_responses:
                    continue
                if row in selected[core_id]:
                    continue
                if problem_counts[core_id][problem] >= 2:
                    continue
                choice = row
                break
            if choice is None:
                raise RuntimeError(f"no globally unique finalist for {core_id}")
            selected[core_id].append(choice)
            used_dialogues.add(str(choice["source_dialogue_id"]))
            used_responses.add(
                " ".join(str(choice["supporter_response"]).casefold().split())
            )
            problem_counts[core_id][
                str(choice.get("problem_type") or "unknown")
            ] += 1
    return dict(selected)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--endpoint", default="final_judge")
    parser.add_argument("--examples-per-core", type=int, default=5)
    parser.add_argument("--max-new-calls", type=int, default=36)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    module = _load_prepare_module()
    pool = [dict(row) for row in iter_jsonl(
        args.candidate_dir / "private_preselection_pool.jsonl"
    )]
    by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_core[str(row["core_submove_id"])].append(row)
    cores = sorted(by_core)
    if len(cores) != 36 or any(len(by_core[core]) != 30 for core in cores):
        raise RuntimeError("expected 36 cores with 30 preselection rows each")

    # Hide all ranker positions from the LLM with a fixed hash ordering.
    shuffled_by_core = {
        core: sorted(
            rows,
            key=lambda row: stable_hex(
                PROTOCOL, core, row["strategy_id"], n=32
            ),
        )
        for core, rows in by_core.items()
    }
    outcomes_path = args.candidate_dir / "llm_preselection_outcomes.jsonl"
    existing = {
        str(row["core_submove_id"]): dict(row)
        for row in (
            iter_jsonl(outcomes_path) if outcomes_path.is_file() else []
        )
    }
    pending = [core for core in cores if core not in existing]
    config = load_config(args.experiment_config)
    endpoint = endpoint_from_config(config, args.endpoint)
    preflight = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_EXECUTION"
            if len(pending) <= args.max_new_calls
            else "BLOCKED_BY_MAX_NEW_CALLS"
        ),
        "endpoint_name": args.endpoint,
        "endpoint_family": endpoint.family,
        "endpoint_model": endpoint.model,
        "planned_total_calls": len(cores),
        "completed_calls": len(existing),
        "pending_calls": len(pending),
        "max_new_calls": args.max_new_calls,
        "run_requested": args.run,
    }
    _write_json(args.candidate_dir / "llm_preselection_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if len(pending) > args.max_new_calls:
        raise RuntimeError(json.dumps(preflight, sort_keys=True))

    client = make_client(endpoint)
    try:
        for index, core_id in enumerate(pending, start=1):
            rows = shuffled_by_core[core_id]
            core = rows[0]
            numbered = list(enumerate(rows, start=1))
            call, parsed = client.chat(
                _messages(core, numbered),
                temperature=0.0,
                max_tokens=900,
                seed=101,
                response_schema=CoreCandidateRanking,
                retries=3,
            )
            if parsed is None:
                raise RuntimeError("structured preselector returned no object")
            _validate_ranking(
                parsed,
                available_numbers=set(range(1, len(rows) + 1)),
            )
            row = {
                "protocol": PROTOCOL,
                "core_submove_id": core_id,
                **parsed.model_dump(mode="json"),
                "endpoint_family": endpoint.family,
                "endpoint_model": endpoint.model,
                "request_hash": call.request_hash,
                "usage": call.usage,
                "latency_ms": round(call.latency_ms, 3),
            }
            _append_jsonl(outcomes_path, row)
            existing[core_id] = row
            print(
                {
                    "completed_now": index,
                    "remaining": len(pending) - index,
                    "core_submove_id": core_id,
                    "estimated_direct_fit_count": (
                        parsed.estimated_direct_fit_count
                    ),
                },
                flush=True,
            )
    finally:
        client.close()

    selected = _final_selection(
        cores=cores,
        shuffled_by_core=shuffled_by_core,
        outcomes=existing,
        examples_per_core=args.examples_per_core,
    )
    packet: list[dict[str, Any]] = []
    private_scores: list[dict[str, Any]] = []
    for core_id in cores:
        core = by_core[core_id][0]
        public = [
            module._public_candidate(row, number)
            for number, row in enumerate(selected[core_id], start=1)
        ]
        review_item_id = stable_hex(
            "h1-source-relink-review", core_id, n=24
        )
        packet.append(
            {
                "protocol": module.PROTOCOL,
                "review_item_id": review_item_id,
                "core_submove_id": core_id,
                "strategy_family": core["strategy_family"],
                "support_move": core["support_move"],
                "when_to_use": core["when_to_use"],
                "when_not_to_use": core["when_not_to_use"],
                "candidates": public,
            }
        )
        preferred = set(existing[core_id]["ranked_candidate_numbers"])
        shuffled_number = {
            str(row["strategy_id"]): index
            for index, row in enumerate(
                shuffled_by_core[core_id], start=1
            )
        }
        for number, row in enumerate(selected[core_id], start=1):
            private_scores.append(
                {
                    "core_submove_id": core_id,
                    "candidate_number": number,
                    "candidate_id": public[number - 1]["candidate_id"],
                    "strategy_id": row["strategy_id"],
                    "source_dialogue_id": row["source_dialogue_id"],
                    "problem_type": row["problem_type"],
                    "nli_entailment": row["nli_entailment"],
                    "lexical_score": row["lexical_score"],
                    "old_bge_assignment_score": row[
                        "old_bge_assignment_score"
                    ],
                    "llm_preselection_number": shuffled_number[
                        str(row["strategy_id"])
                    ],
                    "in_llm_ranked_top10": (
                        shuffled_number[str(row["strategy_id"])] in preferred
                    ),
                }
            )

    _write_jsonl(args.candidate_dir / "source_relink_packet.jsonl", packet)
    _write_jsonl(
        args.candidate_dir / "private_candidate_scores.jsonl",
        private_scores,
    )
    template = [
        {
            "protocol": module.PROTOCOL,
            "review_item_id": row["review_item_id"],
            "core_submove_id": row["core_submove_id"],
            "accepted_candidate_numbers": [],
            "reviewed": False,
            "notes": "",
            "annotator_id": "",
        }
        for row in packet
    ]
    _write_jsonl(
        args.candidate_dir / "human_annotation_template.jsonl", template
    )
    manifest_path = args.candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "READY_FOR_ONE_BOUNDED_HUMAN_REVIEW"
    manifest["llm_search_assistant"] = {
        "protocol": PROTOCOL,
        "endpoint_name": args.endpoint,
        "endpoint_family": endpoint.family,
        "endpoint_model": endpoint.model,
        "calls": len(existing),
        "role": (
            "candidate search/ranking only; hidden from reviewer and never "
            "used as a source-validity label"
        ),
        "final_human_candidates": len(private_scores),
        "all_scores_hidden_from_human": True,
    }
    manifest.pop("intermediate_warning", None)
    manifest.pop("outputs", None)
    (args.candidate_dir / "human_review.html").write_text(
        module._render_html(packet, manifest),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in sorted(args.candidate_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(manifest_path, manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": "READY_FOR_ONE_BOUNDED_HUMAN_REVIEW",
            "api_calls": len(existing),
            "cores": len(packet),
            "final_candidates": len(private_scores),
            "human_review": str(args.candidate_dir / "human_review.html"),
        }
    )


if __name__ == "__main__":
    main()
