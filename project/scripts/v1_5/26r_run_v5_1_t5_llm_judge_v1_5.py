#!/usr/bin/env python3
"""Run the frozen independent GPT-4o T5 judge plan, resumably."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.1-t5-independent-llm-judge-execution-v1"
PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_llm_judge_plan_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_llm_judge_execution_v1"


QualityPreference = Literal["A", "B", "tie", "uncertain"]
QualityCriterion = Literal[
    "grounded_context_fidelity",
    "emotional_understanding",
    "request_and_dialogue_fit",
    "immediate_helpfulness",
    "clarity_naturalness_not_overloaded",
    "materially_equivalent",
    "uncertain",
]
RiskVerdict = Literal["yes", "no", "uncertain"]
RiskCategory = Literal[
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "overgeneralized_pattern_or_cause",
    "fabricated_recall",
    "explicit_boundary_violation",
    "excessive_directiveness",
    "internal_resource_label_exposure",
]


class QualityItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    preference: QualityPreference
    decisive_criterion: QualityCriterion
    reason: str


class QualityBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[QualityItemResult]


class RiskItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    any_material_risk: RiskVerdict
    categories: list[RiskCategory]
    response_evidence_excerpt: str
    authorized_evidence_excerpt: str
    reason: str


class RiskBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[RiskItemResult]


# Keep the schemas fully materialized even when this runner is imported by a
# static preflight rather than executed as __main__.
QualityBatchResult.model_rebuild()
RiskBatchResult.model_rebuild()


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [dict(row) for row in iter_jsonl(path)]


def validate_result(call: dict[str, Any], parsed: BaseModel) -> list[dict[str, Any]]:
    result = parsed.model_dump(mode="json")
    verdicts = [dict(row) for row in result["verdicts"]]
    expected = [str(value) for value in call["item_ids"]]
    actual = [str(row["item_id"]) for row in verdicts]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError(f"judge returned wrong item IDs for {call['call_id']}: {actual}")
    by_id = {str(row["item_id"]): row for row in verdicts}
    ordered = [by_id[item_id] for item_id in expected]
    for row in ordered:
        if call["kind"] == "quality":
            if row["preference"] == "tie" and row["decisive_criterion"] != "materially_equivalent":
                raise RuntimeError("quality tie lacks materially_equivalent criterion")
            if row["preference"] == "uncertain" and row["decisive_criterion"] != "uncertain":
                raise RuntimeError("quality uncertain lacks uncertain criterion")
        else:
            if row["any_material_risk"] == "yes":
                if not row["categories"] or not str(row["response_evidence_excerpt"]).strip() or not str(row["reason"]).strip():
                    raise RuntimeError("risk=yes lacks category, evidence, or reason")
            elif row["categories"]:
                raise RuntimeError("risk no/uncertain must have empty categories")
    return ordered


def persist_success(path: Path, row: dict[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != row:
            raise RuntimeError(f"existing judge success drifted: {path}")
        return
    write_json(path, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")
    manifest = read_json(PLAN_DIR / "judge_plan_manifest.json")
    if manifest.get("status") != "READY_FOR_INDEPENDENT_LLM_JUDGE_EXECUTION":
        raise RuntimeError("independent judge plan is not frozen")
    call_path = PLAN_DIR / "judge_call_plan_private.jsonl"
    if sha256_file(call_path) != manifest["outputs_sha256"]["judge_call_plan"]:
        raise RuntimeError("judge call plan hash drifted")
    plan = rows(call_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    success_dir = OUT_DIR / "success_private"
    success_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        path.stem for path in success_dir.glob("*.json") if path.is_file()
    }
    remaining = [row for row in plan if str(row["call_id"]) not in completed]
    if not args.run:
        print(json.dumps({
            "protocol": PROTOCOL,
            "status": "READY" if remaining else "COMPLETE",
            "planned_calls": len(plan),
            "completed_calls": len(plan) - len(remaining),
            "remaining_calls": len(remaining),
            "calls_by_kind": dict(Counter(str(row["kind"]) for row in plan)),
        }))
        return
    experiment = load_config(ROOT / "configs/experiment.yaml")
    endpoint = endpoint_from_config(experiment, "final_judge")
    if endpoint.model != manifest["judge_model"] or endpoint.family != manifest["judge_family"]:
        raise RuntimeError("judge endpoint drifted from frozen manifest")
    if not os.environ.get(endpoint.api_key_env):
        raise RuntimeError(f"missing {endpoint.api_key_env}")
    client = make_client(endpoint)
    completed_now = 0
    try:
        for call in remaining:
            schema = QualityBatchResult if call["kind"] == "quality" else RiskBatchResult
            result, parsed = client.chat(
                list(call["messages"]),
                temperature=float(call["temperature"]),
                max_tokens=int(call["max_output_tokens"]),
                seed=int(call["seed"]),
                response_schema=schema,
                retries=3,
            )
            if parsed is None:
                raise RuntimeError(f"judge returned no parsed result: {call['call_id']}")
            usage = require_reported_usage(result.usage, stage=PROTOCOL)
            verdicts = validate_result(call, parsed)
            persist_success(success_dir / f"{call['call_id']}.json", {
                "protocol": PROTOCOL,
                "call_id": call["call_id"],
                "kind": call["kind"],
                "item_ids": call["item_ids"],
                "messages_sha256": call["messages_sha256"],
                "request_hash": result.request_hash,
                "usage": usage,
                "verdicts": verdicts,
            })
            completed_now += 1
    finally:
        client.close()

    terminal = [read_json(success_dir / f"{row['call_id']}.json") for row in plan]
    all_verdicts: list[dict[str, Any]] = []
    for row in terminal:
        for verdict in row["verdicts"]:
            all_verdicts.append({
                "protocol": PROTOCOL,
                "call_id": row["call_id"],
                "kind": row["kind"],
                **dict(verdict),
            })
    result_path = OUT_DIR / "judge_verdicts_ordered_private.jsonl"
    write_jsonl(result_path, all_verdicts)
    usage = {
        key: sum(int(row["usage"][key]) for row in terminal)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_INDEPENDENT_LLM_JUDGE_SENSITIVITY",
        "planned_calls": len(plan),
        "completed_calls": len(terminal),
        "completed_now": completed_now,
        "verdicts": len(all_verdicts),
        "calls_by_kind": dict(Counter(str(row["kind"]) for row in terminal)),
        "usage": usage,
        "judge_call_plan_sha256": sha256_file(call_path),
        "judge_verdicts_sha256": sha256_file(result_path),
        "role": manifest["role"],
        "used_to_modify_method": False,
    }
    write_json(OUT_DIR / "execution_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
