#!/usr/bin/env python3
"""Run two label-blind semantic open-coding passes over ESConv train samples.

The models never receive native ESConv labels, the current five-family design,
or the existing 50-card taxonomy.  Outputs are free action descriptions with
literal response evidence; they are candidate qualitative codes, not gold.
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
PROTOCOL = "pm-v1.5-esconv-strategy-open-coding-run-v1"
FORBIDDEN_PUBLIC_FIELDS = {
    "strategy_label",
    "native_ESConv_strategy_label",
    "strategy_family",
    "support_mode",
    "submove_id",
    "problem_type",
    "emotion_type",
    "experience_type",
}


class OpenCodeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blind_item_id: str
    meaningful_support_action: bool
    primary_action: str
    secondary_action: str
    mainly_information: bool
    mainly_self_disclosure: bool
    reusable_as_general_technique: bool
    clear_risk_or_boundary_problem: bool
    risk_description: str
    response_evidence_excerpt: str
    confidence: Literal["high", "medium", "low"]


class OpenCodeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[OpenCodeItem]


SYSTEM_PROMPT = """\
You are doing inductive qualitative coding of emotional-support dialogue.
You are NOT rating overall response quality and you are NOT selecting from a
predefined strategy taxonomy.

For each supporter response:
1. Decide whether it contains a meaningful support action that could be
   described beyond its topic-specific words.
2. If yes, describe the primary atomic action in a short, topic-agnostic verb
   phrase (normally 3-10 words). Add one secondary action only if it is clearly
   present. Do not use dataset label names and do not invent a hierarchy.
3. Separately mark whether the response is mainly factual information, mainly
   supporter self-disclosure, reusable as a general support technique, or has a
   clear risk/boundary problem.
4. Quote one exact excerpt from the supporter response as evidence. If there is
   no meaningful action, primary_action and secondary_action may be empty, but
   still quote the text that led to the decision.

Stay descriptive. Do not infer whether the response helped the user. Do not
infer private facts, future outcomes, diagnoses, or hidden intentions."""


def _user_prompt(rows: list[dict[str, Any]]) -> str:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        rendered.append(
            {
                "blind_item_id": row["blind_item_id"],
                "recent_visible_dialogue": row["recent_dialogue"],
                "supporter_response_to_code": row["supporter_response"],
            }
        )
    return (
        "Code every item below exactly once and return only the requested "
        "structured object. Preserve each blind_item_id exactly.\n\n"
        + canonical_json(rendered)
    )


def _normalize_for_excerpt(text: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", str(text).casefold().replace("’", "'"))
    )


def _validate_public_packet(
    rows: list[dict[str, Any]], lineage_rows: list[dict[str, Any]]
) -> None:
    if len(rows) != 160:
        raise RuntimeError(f"expected 160 public rows, found {len(rows)}")
    ids = [str(row.get("blind_item_id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("public open-coding IDs are empty or duplicated")
    lineage_ids = [str(row.get("blind_item_id") or "") for row in lineage_rows]
    if len(lineage_ids) != len(set(lineage_ids)) or set(ids) != set(lineage_ids):
        raise RuntimeError("public packet and private lineage IDs differ")
    for row in rows:
        leaked = sorted(FORBIDDEN_PUBLIC_FIELDS & set(row))
        if leaked:
            raise RuntimeError(
                f"{row['blind_item_id']}: public packet leaks {leaked}"
            )
        if not isinstance(row.get("recent_dialogue"), list):
            raise RuntimeError("public packet recent_dialogue must be a list")
        if not str(row.get("supporter_response") or "").strip():
            raise RuntimeError("public packet contains an empty response")


def _validate_model_batch(
    *,
    expected_rows: list[dict[str, Any]],
    parsed: OpenCodeBatch,
) -> list[dict[str, Any]]:
    expected_by_id = {
        str(row["blind_item_id"]): row for row in expected_rows
    }
    items = [row.model_dump(mode="json") for row in parsed.items]
    ids = [str(row["blind_item_id"]) for row in items]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_by_id):
        raise RuntimeError("model output IDs do not exactly match the batch")
    validated: list[dict[str, Any]] = []
    for row in items:
        item_id = str(row["blind_item_id"])
        source = expected_by_id[item_id]
        meaningful = bool(row["meaningful_support_action"])
        primary = " ".join(str(row["primary_action"]).split())
        secondary = " ".join(str(row["secondary_action"]).split())
        evidence = " ".join(str(row["response_evidence_excerpt"]).split())
        risk_description = " ".join(str(row["risk_description"]).split())
        validation_flags: list[str] = []
        if meaningful and not primary:
            validation_flags.append("meaningful_but_primary_action_empty")
        if not meaningful and (primary or secondary):
            validation_flags.append("nonmeaningful_but_action_code_present")
        if bool(row["clear_risk_or_boundary_problem"]) and not risk_description:
            validation_flags.append("risk_true_but_description_empty")
        if not evidence:
            validation_flags.append("response_evidence_excerpt_empty")
        evidence_literal = bool(
            evidence
            and _normalize_for_excerpt(evidence)
            in _normalize_for_excerpt(str(source["supporter_response"]))
        )
        if not evidence_literal:
            validation_flags.append("evidence_excerpt_not_literal")
        validated.append(
            {
                **row,
                "primary_action": primary,
                "secondary_action": secondary,
                "risk_description": risk_description,
                "response_evidence_excerpt": evidence,
                "evidence_excerpt_literal": evidence_literal,
                "validation_flags": validation_flags,
            }
        )
    return sorted(validated, key=lambda row: str(row["blind_item_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "open_coding_packet.jsonl",
    )
    parser.add_argument(
        "--private-lineage",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "private_lineage.jsonl",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        dest="endpoints",
        default=None,
        help=(
            "Repeat for independent coding endpoints. Defaults to generator "
            "(Llama) and training_judge_glm51 (GLM)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_strategy_open_coding_v1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    endpoint_names = args.endpoints or ["generator", "training_judge_glm51"]
    if len(endpoint_names) != len(set(endpoint_names)):
        raise ValueError("coding endpoint names must be unique")
    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("batch size must be between 1 and 10")

    packet_rows = [dict(row) for row in iter_jsonl(args.packet)]
    lineage_rows = [dict(row) for row in iter_jsonl(args.private_lineage)]
    _validate_public_packet(packet_rows, lineage_rows)

    experiment = load_config(args.experiment_config)
    endpoints = {
        name: endpoint_from_config(experiment, name)
        for name in endpoint_names
    }
    families = [str(endpoint.family or "") for endpoint in endpoints.values()]
    if any(not family for family in families) or len(families) != len(
        set(families)
    ):
        raise RuntimeError(
            "independent open coding requires distinct declared model families"
        )

    batches = [
        packet_rows[start : start + args.batch_size]
        for start in range(0, len(packet_rows), args.batch_size)
    ]
    call_plan: list[dict[str, Any]] = []
    for endpoint_name, endpoint in endpoints.items():
        for batch_index, batch in enumerate(batches):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(batch)},
            ]
            call_plan.append(
                {
                    "endpoint_name": endpoint_name,
                    "model_family": endpoint.family,
                    "model": endpoint.model,
                    "transport": endpoint_transport(endpoint),
                    "batch_index": batch_index,
                    "blind_item_ids": [
                        str(row["blind_item_id"]) for row in batch
                    ],
                    "messages": messages,
                    "messages_sha256": sha256_text(canonical_json(messages)),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = args.out_dir / "model_open_codes.jsonl"
    calls_path = args.out_dir / "call_ledger.jsonl"
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if calls_path.exists():
        for row in iter_jsonl(calls_path):
            if row.get("status") != "complete":
                continue
            key = (str(row["endpoint_name"]), int(row["batch_index"]))
            if key in completed:
                raise RuntimeError("call ledger repeats a completed batch")
            completed[key] = dict(row)

    pending = [
        row
        for row in call_plan
        if (str(row["endpoint_name"]), int(row["batch_index"]))
        not in completed
    ]
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY" if pending else "COMPLETE",
        "packet_rows": len(packet_rows),
        "private_lineage_rows": len(lineage_rows),
        "packet_sha256": sha256_file(args.packet),
        "private_lineage_sha256": sha256_file(args.private_lineage),
        "native_labels_visible_to_models": False,
        "current_five_family_names_visible_to_models": False,
        "current_50_submoves_visible_to_models": False,
        "endpoint_descriptors": [
            {
                "endpoint_name": name,
                "family": endpoint.family,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "transport": endpoint_transport(endpoint),
            }
            for name, endpoint in endpoints.items()
        ],
        "batch_size": args.batch_size,
        "total_calls": len(call_plan),
        "completed_calls": len(completed),
        "pending_calls": len(pending),
        "model_outputs_are_gold": False,
        "disagreement_action": (
            "retain for human review; never majority-vote or auto-resolve"
        ),
    }
    write_json(args.out_dir / "preflight.json", preflight)
    if not args.run:
        print(canonical_json(preflight))
        return
    if args.max_new_calls < 1:
        raise ValueError("--run requires positive --max-new-calls")
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but max-new-calls="
            f"{args.max_new_calls}"
        )

    clients = {name: make_client(endpoint) for name, endpoint in endpoints.items()}
    try:
        for plan in pending:
            endpoint_name = str(plan["endpoint_name"])
            endpoint = endpoints[endpoint_name]
            batch_index = int(plan["batch_index"])
            batch = batches[batch_index]
            try:
                result, parsed = clients[endpoint_name].chat(
                    list(plan["messages"]),
                    temperature=0.0,
                    max_tokens=args.max_output_tokens,
                    seed=4311 + batch_index,
                    response_schema=OpenCodeBatch,
                    retries=3,
                )
                if parsed is None:
                    raise RuntimeError("open-coding call returned no parsed output")
                if result.normalized_finish_reason != "complete":
                    raise RuntimeError(
                        "open-coding call did not finish completely: "
                        f"{result.provider_finish_reason}"
                    )
                validated = _validate_model_batch(
                    expected_rows=batch, parsed=parsed
                )
                usage = require_reported_usage(
                    result.usage,
                    stage="esconv_strategy_open_coding",
                )
                for row in validated:
                    append_jsonl(
                        outputs_path,
                        {
                            "protocol": PROTOCOL,
                            "endpoint_name": endpoint_name,
                            "model_family": endpoint.family,
                            "model": endpoint.model,
                            "batch_index": batch_index,
                            **row,
                        },
                    )
                append_jsonl(
                    calls_path,
                    {
                        "protocol": PROTOCOL,
                        "status": "complete",
                        "endpoint_name": endpoint_name,
                        "model_family": endpoint.family,
                        "model": endpoint.model,
                        "batch_index": batch_index,
                        "blind_item_ids": plan["blind_item_ids"],
                        "messages_sha256": plan["messages_sha256"],
                        "request_hash": result.request_hash,
                        "provider_finish_reason": (
                            result.provider_finish_reason
                        ),
                        "normalized_finish_reason": (
                            result.normalized_finish_reason
                        ),
                        "usage": usage,
                        "latency_ms": round(float(result.latency_ms), 3),
                    },
                )
            except Exception as exc:
                append_jsonl(
                    calls_path,
                    {
                        "protocol": PROTOCOL,
                        "status": "failed",
                        "endpoint_name": endpoint_name,
                        "model_family": endpoint.family,
                        "model": endpoint.model,
                        "batch_index": batch_index,
                        "blind_item_ids": plan["blind_item_ids"],
                        "messages_sha256": plan["messages_sha256"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
    finally:
        for client in clients.values():
            client.close()

    output_rows = [dict(row) for row in iter_jsonl(outputs_path)]
    per_endpoint = Counter(str(row["endpoint_name"]) for row in output_rows)
    report = {
        **preflight,
        "status": "COMPLETE",
        "completed_calls": len(call_plan),
        "pending_calls": 0,
        "output_rows": len(output_rows),
        "output_rows_by_endpoint": dict(sorted(per_endpoint.items())),
        "expected_rows_per_endpoint": len(packet_rows),
        "exact_two_pass_coverage": all(
            per_endpoint[name] == len(packet_rows) for name in endpoint_names
        ),
        "rows_with_validation_flags": sum(
            bool(row.get("validation_flags")) for row in output_rows
        ),
        "nonliteral_evidence_rows": sum(
            not bool(row.get("evidence_excerpt_literal"))
            for row in output_rows
        ),
    }
    if not report["exact_two_pass_coverage"]:
        raise RuntimeError("open-coding output lacks exact two-pass coverage")
    write_json(args.out_dir / "run_report.json", report)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
