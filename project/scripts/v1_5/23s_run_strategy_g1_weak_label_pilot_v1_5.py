#!/usr/bin/env python3
"""Run a bounded G1 Coder-B weak multi-label packet.

The model proposes atomic moves and source-exclusion flags. It does not create
gold labels, approve cards, estimate prevalence, or produce PM targets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from metacom_pm.api import (
    endpoint_transport,
    make_client,
    require_reported_usage,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-bounded-weak-label-runner-v1"
ALLOWED_EXCLUSIONS = {
    "platform_or_crowdwork_mechanics",
    "pure_phatic_or_off_topic",
    "substantive_supporter_self_disclosure",
    "external_factual_or_specialist_content",
    "high_stakes_guidance",
    "person_labeling_shaming_or_values_imposition",
    "unsupported_deterministic_prediction_or_false_hope",
    "confidentiality_or_permanent_availability_overclaim",
}


class MoveEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move_id: str
    response_excerpt: str


class ExclusionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclusion_flag: str
    excerpt: str
    location: Literal["supporter_response", "recent_dialogue"]


class WeakLabelItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blind_item_id: str
    recognized_move_ids: list[str]
    move_evidence: list[MoveEvidence]
    hard_exclusion_flags: list[str]
    exclusion_evidence: list[ExclusionEvidence]
    confidence: Literal["high", "medium", "low"]
    notes: str


class WeakLabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[WeakLabelItem]


def _normalize(text: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", str(text).casefold().replace("’", "'"))
    )


def _is_literal(excerpt: str, source: str) -> bool:
    normalized_excerpt = _normalize(excerpt)
    return bool(normalized_excerpt and normalized_excerpt in _normalize(source))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("blind_item_id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label}: empty or duplicate blind_item_id")
        result[item_id] = row
    return result


def _system_prompt(codebook: dict[str, Any]) -> str:
    moves = [
        {
            "move_id": move["move_id"],
            "name": move["name"],
            "definition": move["definition"],
        }
        for move in codebook["moves"]
    ]
    exclusions = {
        "platform_or_crowdwork_mechanics": (
            "interface, timing, survey, quit, or crowdwork instructions"
        ),
        "pure_phatic_or_off_topic": (
            "greeting, goodbye, or casual content with no support progress"
        ),
        "substantive_supporter_self_disclosure": (
            "support effect relies on the supporter's identity, relationship, "
            "location, work, coping history, or lived experience"
        ),
        "external_factual_or_specialist_content": (
            "main value relies on externally verifiable factual, resource, "
            "medical, legal, financial, or mechanistic content"
        ),
        "high_stakes_guidance": (
            "specific medical, legal, financial, clinical, DV, self-harm, "
            "emergency, or comparable high-stakes guidance"
        ),
        "person_labeling_shaming_or_values_imposition": (
            "labels or shames the user, asserts hidden motives, or imposes the "
            "supporter's religion or values"
        ),
        "unsupported_deterministic_prediction_or_false_hope": (
            "promises or predicts an outcome without support, or contradicts "
            "explicit evidence to offer hope"
        ),
        "confidentiality_or_permanent_availability_overclaim": (
            "promises confidentiality, secrecy, or indefinite availability "
            "that the system or one-off supporter cannot guarantee"
        ),
    }
    return (
        "You are doing source-screening weak multi-label annotation for a "
        "topic-agnostic emotional-support technique bank. This is not response "
        "quality scoring and not PM action labeling.\n\n"
        "Assign zero or more recognized atomic moves from the frozen codebook. "
        "A response may contain several moves. Do not invent a move that is not "
        "in the list. Separately flag every applicable hard source exclusion. "
        "A recognizable move may coexist with an exclusion; the move then helps "
        "taxonomy discovery but the original response is not automatically a "
        "safe card source.\n\n"
        "For every recognized move, quote one exact literal excerpt from the "
        "target supporter response. For every exclusion, quote one exact "
        "literal excerpt and identify whether it comes from the target response "
        "or recent dialogue. Do not quote or paraphrase text that is absent. "
        "If no move or exclusion applies, return empty lists. Low confidence is "
        "allowed. Never decide final card eligibility.\n\n"
        f"FROZEN MOVES:\n{canonical_json(moves)}\n\n"
        f"HARD EXCLUSIONS:\n{canonical_json(exclusions)}"
    )


def _user_prompt(rows: list[dict[str, Any]]) -> str:
    return (
        "Label every item exactly once. Preserve blind_item_id exactly. Return "
        "only the requested structured object.\n\n"
        + canonical_json(rows)
    )


def _validate_batch(
    *,
    expected_rows: list[dict[str, Any]],
    parsed: WeakLabelBatch,
    allowed_move_ids: set[str],
) -> list[dict[str, Any]]:
    expected = _by_id(expected_rows, label="expected_batch")
    items = [item.model_dump(mode="json") for item in parsed.items]
    actual = _by_id(items, label="parsed_batch")
    if set(expected) != set(actual):
        raise ValueError("parsed IDs do not exactly match the requested batch")

    validated: list[dict[str, Any]] = []
    for item_id in sorted(actual):
        row = actual[item_id]
        source = expected[item_id]
        move_ids = [str(value) for value in row["recognized_move_ids"]]
        exclusion_flags = [str(value) for value in row["hard_exclusion_flags"]]
        if len(move_ids) != len(set(move_ids)):
            raise ValueError(f"{item_id}: duplicate move IDs")
        if len(exclusion_flags) != len(set(exclusion_flags)):
            raise ValueError(f"{item_id}: duplicate exclusion flags")
        invalid_moves = set(move_ids) - allowed_move_ids
        invalid_exclusions = set(exclusion_flags) - ALLOWED_EXCLUSIONS
        if invalid_moves:
            raise ValueError(f"{item_id}: invalid move IDs {sorted(invalid_moves)}")
        if invalid_exclusions:
            raise ValueError(
                f"{item_id}: invalid exclusions {sorted(invalid_exclusions)}"
            )

        move_evidence = list(row["move_evidence"])
        evidence_move_ids = [str(value["move_id"]) for value in move_evidence]
        if sorted(evidence_move_ids) != sorted(move_ids):
            raise ValueError(f"{item_id}: move evidence does not match move IDs")
        response = str(source["supporter_response_to_label"])
        valid_move_evidence: list[dict[str, Any]] = []
        dropped_nonliteral_move_ids: list[str] = []
        for evidence in move_evidence:
            if _is_literal(str(evidence["response_excerpt"]), response):
                valid_move_evidence.append(evidence)
            else:
                # A model can anchor on a supporter turn in recent dialogue
                # even though moves must be executed in the target response.
                # Deterministically dropping that move is safer and cheaper
                # than accepting contamination or repeatedly calling with the
                # same seed. This repair can only remove weak positives.
                dropped_nonliteral_move_ids.append(str(evidence["move_id"]))
        if dropped_nonliteral_move_ids:
            row["recognized_move_ids"] = [
                move_id
                for move_id in move_ids
                if move_id not in set(dropped_nonliteral_move_ids)
            ]
            row["move_evidence"] = valid_move_evidence
            move_ids = list(row["recognized_move_ids"])

        exclusion_evidence = list(row["exclusion_evidence"])
        evidence_flags = [
            str(value["exclusion_flag"]) for value in exclusion_evidence
        ]
        if sorted(evidence_flags) != sorted(exclusion_flags):
            raise ValueError(
                f"{item_id}: exclusion evidence does not match exclusion flags"
            )
        recent_dialogue = "\n".join(
            str(turn.get("content") or "")
            for turn in source["recent_visible_dialogue"]
        )
        for evidence in exclusion_evidence:
            location = str(evidence["location"])
            evidence_source = response if location == "supporter_response" else recent_dialogue
            if not _is_literal(str(evidence["excerpt"]), evidence_source):
                raise ValueError(f"{item_id}: nonliteral exclusion evidence")

        validated.append(
            {
                **row,
                "recognized_move_ids": sorted(move_ids),
                "hard_exclusion_flags": sorted(exclusion_flags),
                "dropped_nonliteral_target_move_ids": sorted(
                    dropped_nonliteral_move_ids
                ),
                "target_response_grounding_repair_applied": bool(
                    dropped_nonliteral_move_ids
                ),
                "source_compatible_weak_proposal": bool(
                    move_ids and not exclusion_flags
                ),
                "evidence_literal_validation_passed": True,
            }
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
        / "public_packet.jsonl",
    )
    parser.add_argument(
        "--private-lineage",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
        / "private_lineage.jsonl",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=ROOT
        / "data/strategy"
        / "pm_v1_5_strategy_atomic_move_codebook_v1.json",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--endpoint", default="training_judge_deepseek_flash")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=3200)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_run_v1",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    packet_rows = _read_jsonl(args.packet)
    lineage_rows = _read_jsonl(args.private_lineage)
    packet = _by_id(packet_rows, label="packet")
    lineage = _by_id(lineage_rows, label="private_lineage")
    if not packet or set(packet) != set(lineage):
        raise ValueError("runner requires nonempty matching public/private ID sets")
    allowed_public_fields = {
        "blind_item_id",
        "recent_visible_dialogue",
        "supporter_response_to_label",
    }
    if any(set(row) != allowed_public_fields for row in packet_rows):
        raise ValueError("pilot public packet contains unexpected or private fields")

    codebook = _read_json(args.codebook)
    allowed_move_ids = {str(move["move_id"]) for move in codebook["moves"]}
    if len(allowed_move_ids) != 17:
        raise ValueError("runner requires the frozen 17-move codebook")
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("batch size must be between 1 and 8")

    experiment = load_config(args.experiment_config)
    endpoint = endpoint_from_config(experiment, args.endpoint)
    batches = [
        packet_rows[start : start + args.batch_size]
        for start in range(0, len(packet_rows), args.batch_size)
    ]
    system_prompt = _system_prompt(codebook)
    call_plan: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_prompt(batch)},
        ]
        call_plan.append(
            {
                "batch_index": batch_index,
                "blind_item_ids": [str(row["blind_item_id"]) for row in batch],
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "weak_labels.jsonl"
    ledger_path = args.out_dir / "call_ledger.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if ledger_path.exists():
        for row in iter_jsonl(ledger_path):
            if row.get("status") != "complete":
                continue
            batch_index = int(row["batch_index"])
            if batch_index in completed:
                raise ValueError("call ledger repeats a completed batch")
            completed[batch_index] = dict(row)
    pending = [
        plan for plan in call_plan if int(plan["batch_index"]) not in completed
    ]
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY" if pending else "COMPLETE",
        "packet_rows": len(packet_rows),
        "private_lineage_rows": len(lineage_rows),
        "frozen_move_count": len(allowed_move_ids),
        "endpoint_name": args.endpoint,
        "endpoint_family": endpoint.family,
        "endpoint_model": endpoint.model,
        "transport": endpoint_transport(endpoint),
        "batch_size": args.batch_size,
        "total_calls": len(call_plan),
        "completed_calls": len(completed),
        "pending_calls": len(pending),
        "native_strategy_problem_emotion_labels_visible": False,
        "BGE_or_lexical_candidate_scores_visible": False,
        "model_outputs_are_gold": False,
        "automatic_card_promotion_authorized": False,
        "packet_sha256": sha256_file(args.packet),
        "codebook_sha256": sha256_file(args.codebook),
    }
    write_json(args.out_dir / "preflight.json", preflight)
    if not args.run:
        print(canonical_json(preflight))
        return
    if args.max_new_calls < 1:
        raise ValueError("--run requires positive --max-new-calls")
    if len(pending) > args.max_new_calls:
        raise ValueError(
            f"{len(pending)} calls remain but max-new-calls={args.max_new_calls}"
        )

    client = make_client(endpoint)
    try:
        for plan in pending:
            batch_index = int(plan["batch_index"])
            batch = batches[batch_index]
            parsed_debug: dict[str, Any] | None = None
            try:
                result, parsed = client.chat(
                    list(plan["messages"]),
                    temperature=0.0,
                    max_tokens=args.max_output_tokens,
                    seed=7601 + batch_index,
                    response_schema=WeakLabelBatch,
                    retries=3,
                )
                if parsed is None:
                    raise RuntimeError("weak-label call returned no parsed output")
                if result.normalized_finish_reason != "complete":
                    raise RuntimeError(
                        "weak-label call did not finish completely: "
                        f"{result.provider_finish_reason}"
                    )
                parsed_debug = parsed.model_dump(mode="json")
                validated = _validate_batch(
                    expected_rows=batch,
                    parsed=parsed,
                    allowed_move_ids=allowed_move_ids,
                )
                usage = require_reported_usage(
                    result.usage,
                    stage="strategy_g1_bounded_weak_label",
                )
                for row in validated:
                    append_jsonl(
                        output_path,
                        {
                            "protocol": PROTOCOL,
                            "endpoint_name": args.endpoint,
                            "model_family": endpoint.family,
                            "model": endpoint.model,
                            "batch_index": batch_index,
                            **row,
                        },
                    )
                append_jsonl(
                    ledger_path,
                    {
                        "protocol": PROTOCOL,
                        "status": "complete",
                        "endpoint_name": args.endpoint,
                        "model_family": endpoint.family,
                        "model": endpoint.model,
                        "batch_index": batch_index,
                        "blind_item_ids": plan["blind_item_ids"],
                        "messages_sha256": plan["messages_sha256"],
                        "request_hash": result.request_hash,
                        "provider_finish_reason": result.provider_finish_reason,
                        "normalized_finish_reason": result.normalized_finish_reason,
                        "usage": usage,
                        "latency_ms": round(float(result.latency_ms), 3),
                    },
                )
            except Exception as exc:
                append_jsonl(
                    ledger_path,
                    {
                        "protocol": PROTOCOL,
                        "status": "failed",
                        "endpoint_name": args.endpoint,
                        "model_family": endpoint.family,
                        "model": endpoint.model,
                        "batch_index": batch_index,
                        "blind_item_ids": plan["blind_item_ids"],
                        "messages_sha256": plan["messages_sha256"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "parsed_output_for_validation_debug": parsed_debug,
                    },
                )
                raise
    finally:
        client.close()

    output_rows = _read_jsonl(output_path)
    output_ids = [str(row["blind_item_id"]) for row in output_rows]
    if len(output_rows) != len(packet_rows) or set(output_ids) != set(packet):
        raise RuntimeError("output lacks exact one-row-per-item coverage")
    move_counts = Counter(
        move_id
        for row in output_rows
        for move_id in row["recognized_move_ids"]
    )
    exclusion_counts = Counter(
        flag for row in output_rows for flag in row["hard_exclusion_flags"]
    )
    ledger_rows = [
        dict(row)
        for row in iter_jsonl(ledger_path)
        if row.get("status") == "complete"
    ]
    usage_totals = {
        key: sum(int(row["usage"].get(key, 0)) for row in ledger_rows)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    report = {
        **preflight,
        "status": "COMPLETE_PENDING_COVERAGE_AGGREGATION",
        "completed_calls": len(call_plan),
        "pending_calls": 0,
        "output_rows": len(output_rows),
        "unique_output_ids": len(set(output_ids)),
        "all_evidence_literal_validation_passed": all(
            row["evidence_literal_validation_passed"] for row in output_rows
        ),
        "rows_with_any_move": sum(
            bool(row["recognized_move_ids"]) for row in output_rows
        ),
        "rows_with_any_exclusion": sum(
            bool(row["hard_exclusion_flags"]) for row in output_rows
        ),
        "source_compatible_weak_proposal_rows": sum(
            row["source_compatible_weak_proposal"] for row in output_rows
        ),
        "move_counts": dict(sorted(move_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "reported_usage": usage_totals,
        "next_gate": (
            "Join private lineage, count independent clean weak-source "
            "dialogues per move, and expand only deficient moves. Outputs "
            "remain non-gold until the bounded source/card human audit."
        ),
    }
    write_json(args.out_dir / "run_report.json", report)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
