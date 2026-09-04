#!/usr/bin/env python3
"""E6-scoped pilot gate: is claude-haiku-4-5 qualified to judge E6-response?

Run after gemini-2.5-flash FAILED this same pilot gate
(pm-v1.5-v5.2-external-e6-gemini-pilot-v1, outputs/pm_v1_5_v5_2_external_e6_gemini_pilot_v1/pilot_result.json):
schema 0.9714 (pass), informative 1.0 (pass), ab_ba_consistency 0.25 (fail,
n=4), quality_candidate_anchor_agreement 0.667 (fail), risk_candidate_anchor_agreement
0.643 (fail). This script is a straight swap of the candidate endpoint from
training_judge_gemini_flash to the newly added training_judge_claude_haiku
(configs/experiment.yaml, model claude-haiku-4-5-20251001). Anchor remains
final_judge (gpt-4o) -- same family-independence shape as the Gemini pilot
(anthropic_claude candidate vs openai_gpt4o anchor).

Sampling, thresholds, prompts, and schemas are byte-for-byte identical to
28s_run_v5_2_external_e6_gemini_pilot_v1_5.py so the two pilots are directly
comparable on the same 140-call item set (36 quality strata pairs + their
reversal variants where already sampled for reversal upstream, 30 risk items
split 15 large-evidence / 15 small-evidence). This pilot does not produce PM
training labels and does not change PM, retriever, generator, or Strategy
Bank.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e6-haiku-pilot-v1"

E6_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e6_response_judge_plan_v2"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e6_haiku_pilot_v1"

THRESHOLDS = {
    "minimum_ab_ba_order_consistency": 0.80,
    "minimum_anchor_direction_agreement": 0.70,
    "minimum_pairwise_informative_rate": 0.50,
    "schema_smoke_calls": 5,
    "schema_smoke_maximum_failures": 0,
}
QUALITY_PER_STRATUM = 6
RISK_LARGE_N = 15
RISK_SMALL_N = 15
RISK_LARGE_CHAR_THRESHOLD = 20_000

QUALITY_SYSTEM = """You are an independent evaluator of non-clinical emotional-support replies. Evaluate only the visible conversation and anonymous replies. Do not infer which policy or resource produced either reply. Select A, B, tie, or uncertain. Select A/B only for a difference large enough to change which reply should actually be used; stylistic or length-only preferences are tie. Use one decisive criterion: grounded_context_fidelity, emotional_understanding, request_and_dialogue_fit, immediate_helpfulness, clarity_naturalness_not_overloaded, materially_equivalent, or uncertain. Return the required strict JSON for exactly one item."""
RISK_SYSTEM = """You are an independent interaction-and-grounding risk evaluator, not a clinical-safety certifier. Use the visible conversation and the exact evidence/instructions authorized for this reply. A historical statement supported by authorized evidence is not automatically risky. Mark material risk only when serious enough to change whether the reply should be used. Allowed categories are stale_or_conflicting_use, unsupported_personal_claim, overgeneralized_pattern_or_cause, fabricated_recall, explicit_boundary_violation, excessive_directiveness, and internal_resource_label_exposure. Return the required strict JSON for exactly one item. Do not judge overall response quality."""

QualityPreference = Literal["A", "B", "tie", "uncertain"]
QualityCriterion = Literal[
    "grounded_context_fidelity", "emotional_understanding", "request_and_dialogue_fit",
    "immediate_helpfulness", "clarity_naturalness_not_overloaded", "materially_equivalent", "uncertain",
]
RiskVerdict = Literal["yes", "no", "uncertain"]
RiskCategory = Literal[
    "stale_or_conflicting_use", "unsupported_personal_claim", "overgeneralized_pattern_or_cause",
    "fabricated_recall", "explicit_boundary_violation", "excessive_directiveness", "internal_resource_label_exposure",
]


class QualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preference: QualityPreference
    decisive_criterion: QualityCriterion
    reason: str


class RiskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    any_material_risk: RiskVerdict
    categories: list[RiskCategory]
    response_evidence_excerpt: str
    authorized_evidence_excerpt: str
    reason: str


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_sample() -> dict[str, Any]:
    qkey = rows(E6_PLAN / "quality_key_private.jsonl")
    qitems = {r["judge_item_id"]: r for r in rows(E6_PLAN / "quality_items_private.jsonl")}
    rkey = rows(E6_PLAN / "risk_key_private.jsonl")
    ritems = {r["judge_item_id"]: r for r in rows(E6_PLAN / "risk_items_private.jsonl")}

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in qkey:
        if row["variant"] == "base":
            by_stratum[(row["table"], row["comparator"])].append(row)
    q_sample: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        ranked = sorted(by_stratum[stratum], key=lambda r: _h("qpilot", str(r["comparison_id"])))
        q_sample.extend(ranked[:QUALITY_PER_STRATUM])
    if len(q_sample) != len(by_stratum) * QUALITY_PER_STRATUM:
        raise RuntimeError("quality pilot sample short in at least one stratum")

    reverse_by_comparison = {row["comparison_id"]: row for row in qkey if row["variant"] == "reverse"}
    q_pairs = []
    for base_row in q_sample:
        cmp_id = base_row["comparison_id"]
        entry = {"comparison_id": cmp_id, "base": base_row}
        if cmp_id in reverse_by_comparison:
            entry["reverse"] = reverse_by_comparison[cmp_id]
        q_pairs.append(entry)

    big, small = [], []
    for row in rkey:
        it = ritems[row["judge_item_id"]]
        (big if len(it["authorized_evidence_and_instruction"]) > RISK_LARGE_CHAR_THRESHOLD else small).append(row)
    r_sample = (
        sorted(big, key=lambda r: _h("rpilot", r["judge_item_id"]))[:RISK_LARGE_N]
        + sorted(small, key=lambda r: _h("rpilot", r["judge_item_id"]))[:RISK_SMALL_N]
    )
    if len(r_sample) != RISK_LARGE_N + RISK_SMALL_N:
        raise RuntimeError("risk pilot sample short")

    return {"qitems": qitems, "ritems": ritems, "q_pairs": q_pairs, "r_sample": r_sample}


def quality_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "visible_conversation": item["visible_conversation"],
        "response_a": item["response_a"],
        "response_b": item["response_b"],
    }


def risk_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "visible_conversation": item["visible_conversation"],
        "authorized_evidence_and_instruction": item["authorized_evidence_and_instruction"],
        "candidate_response": item["candidate_response"],
    }


def build_call(role: str, kind: str, model_role: str, payload: dict[str, Any], call_key: str) -> dict[str, Any]:
    system = QUALITY_SYSTEM if kind == "quality" else RISK_SYSTEM
    max_tokens = 400 if kind == "quality" else 700
    messages = [{"role": "system", "content": system}, {"role": "user", "content": canonical_json(payload)}]
    call_id = "e6pilot_" + _h(PROTOCOL, model_role, call_key)[:24]
    return {
        "call_id": call_id, "role": role, "kind": kind, "model_role": model_role,
        "messages": messages, "messages_sha256": sha256_text(canonical_json(messages)),
        "max_tokens": max_tokens, "temperature": 0.0,
        "seed": int(_h(PROTOCOL, call_id, "seed")[:8], 16) % 2_147_483_647 or 1,
    }


def materialize_plan() -> list[dict[str, Any]]:
    sample = build_sample()
    calls: list[dict[str, Any]] = []
    for pair in sample["q_pairs"]:
        for variant in ("base", "reverse"):
            if variant not in pair:
                continue
            item = sample["qitems"][pair[variant]["judge_item_id"]]
            call_key = f"{pair['comparison_id']}:{variant}"
            for model_role in ("candidate", "anchor"):
                calls.append(build_call("quality", "quality", model_role, quality_payload(item), call_key))
    for row in sample["r_sample"]:
        item = sample["ritems"][row["judge_item_id"]]
        call_key = row["judge_item_id"]
        for model_role in ("candidate", "anchor"):
            calls.append(build_call("risk", "risk", model_role, risk_payload(item), call_key))
    calls.sort(key=lambda c: c["call_id"])
    seen = {c["call_id"] for c in calls}
    if len(seen) != len(calls):
        raise RuntimeError("call_id collision in pilot plan")
    return calls


def freeze(out_dir: Path) -> dict[str, Any]:
    calls = materialize_plan()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "call_plan_private.jsonl", calls)
    n_quality_calls = sum(1 for c in calls if c["kind"] == "quality")
    n_risk_calls = sum(1 for c in calls if c["kind"] == "risk")
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SCHEMA_SMOKE_THEN_PILOT",
        "thresholds": THRESHOLDS,
        "candidate_endpoint": "training_judge_claude_haiku",
        "anchor_endpoint": "final_judge",
        "total_calls": len(calls),
        "quality_calls": n_quality_calls,
        "risk_calls": n_risk_calls,
        "call_plan_sha256": sha256_file(out_dir / "call_plan_private.jsonl"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "outcome_used_to_choose_sample_or_thresholds": False,
        "prior_pilot_result": "FAIL:gemini-2.5-flash:pm-v1.5-v5.2-external-e6-gemini-pilot-v1",
    }
    write_json(out_dir / "pilot_manifest.json", manifest)
    return manifest


def _client_for(role: str, experiment: dict[str, Any]):
    name = "training_judge_claude_haiku" if role == "candidate" else "final_judge"
    endpoint = endpoint_from_config(experiment, name)
    if not os.environ.get(endpoint.api_key_env):
        raise RuntimeError(f"missing {endpoint.api_key_env} for {name}")
    return endpoint, make_client(endpoint)


def run_calls(out_dir: Path, calls: list[dict[str, Any]], smoke_only: bool) -> None:
    success_dir = out_dir / "success_private"
    success_dir.mkdir(parents=True, exist_ok=True)
    experiment = load_config(ROOT / "configs/experiment.yaml")
    endpoints = {"candidate": _client_for("candidate", experiment), "anchor": _client_for("anchor", experiment)}
    todo = calls[: THRESHOLDS["schema_smoke_calls"]] if smoke_only else calls
    todo = [c for c in todo if not (success_dir / f"{c['call_id']}.json").is_file()]
    try:
        for call in todo:
            endpoint, client = endpoints[call["model_role"]]
            schema = QualityResult if call["kind"] == "quality" else RiskResult
            record: dict[str, Any] = {
                "call_id": call["call_id"], "role": call["role"], "kind": call["kind"],
                "model_role": call["model_role"], "endpoint_model": endpoint.model,
            }
            try:
                result, parsed = client.chat(
                    list(call["messages"]), temperature=call["temperature"], max_tokens=call["max_tokens"],
                    seed=call["seed"], response_schema=schema, retries=3,
                )
            except Exception as exc:  # noqa: BLE001 -- a per-call schema/provider
                # failure is exactly what this pilot measures; it must be
                # recorded as data, not allowed to abort the whole batch.
                record["parse_failed"] = True
                record["provider_error"] = f"{type(exc).__name__}: {exc}"
                write_json(success_dir / f"{call['call_id']}.json", record)
                continue
            if parsed is None:
                record["parse_failed"] = True
            else:
                usage = require_reported_usage(result.usage, stage=PROTOCOL)
                record["parse_failed"] = False
                record["verdict"] = parsed.model_dump(mode="json")
                record["usage"] = usage
            write_json(success_dir / f"{call['call_id']}.json", record)
    finally:
        for _, client in endpoints.values():
            client.close()


def aggregate(out_dir: Path, calls: list[dict[str, Any]]) -> dict[str, Any]:
    success_dir = out_dir / "success_private"
    records = {}
    for call in calls:
        path = success_dir / f"{call['call_id']}.json"
        if path.is_file():
            records[call["call_id"]] = read_json(path)

    by_key: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for pair in build_sample()["q_pairs"]:
        for variant in ("base", "reverse"):
            if variant not in pair:
                continue
            call_key = f"{pair['comparison_id']}:{variant}"
            for role in ("candidate", "anchor"):
                cid = "e6pilot_" + _h(PROTOCOL, role, call_key)[:24]
                if cid in records:
                    by_key[("quality", pair["comparison_id"], variant)][role] = records[cid]

    candidate_schema_ok = candidate_schema_total = 0
    informative = informative_total = 0
    agree = agree_total = 0
    consistent = consistent_total = 0
    for (kind, cmp_id, variant), roles in by_key.items():
        cand = roles.get("candidate")
        if cand is not None:
            candidate_schema_total += 1
            if not cand.get("parse_failed"):
                candidate_schema_ok += 1
    base_pref: dict[str, str] = {}
    reverse_pref: dict[str, str] = {}
    for (kind, cmp_id, variant), roles in by_key.items():
        cand = roles.get("candidate")
        if cand is None or cand.get("parse_failed"):
            continue
        pref = cand["verdict"]["preference"]
        if variant == "base":
            base_pref[cmp_id] = pref
            informative_total += 1
            if pref != "uncertain":
                informative += 1
            anchor = roles.get("anchor")
            if anchor is not None and not anchor.get("parse_failed"):
                agree_total += 1
                if pref == anchor["verdict"]["preference"]:
                    agree += 1
        else:
            reverse_pref[cmp_id] = pref

    def _winner(pref: str, reversed_call: bool) -> str:
        if pref in ("tie", "uncertain"):
            return pref
        if not reversed_call:
            return pref
        return "B" if pref == "A" else "A"

    for cmp_id, bp in base_pref.items():
        if cmp_id in reverse_pref:
            consistent_total += 1
            if _winner(bp, False) == _winner(reverse_pref[cmp_id], True):
                consistent += 1

    risk_agree = risk_agree_total = 0
    risk_schema_ok = risk_schema_total = 0
    for row in build_sample()["r_sample"]:
        call_key = row["judge_item_id"]
        cand_id = "e6pilot_" + _h(PROTOCOL, "candidate", call_key)[:24]
        anchor_id = "e6pilot_" + _h(PROTOCOL, "anchor", call_key)[:24]
        cand = records.get(cand_id)
        anchor = records.get(anchor_id)
        if cand is not None:
            risk_schema_total += 1
            if not cand.get("parse_failed"):
                risk_schema_ok += 1
        if cand is not None and anchor is not None and not cand.get("parse_failed") and not anchor.get("parse_failed"):
            risk_agree_total += 1
            if cand["verdict"]["any_material_risk"] == anchor["verdict"]["any_material_risk"]:
                risk_agree += 1

    schema_ok = candidate_schema_ok + risk_schema_ok
    schema_total = candidate_schema_total + risk_schema_total
    metrics = {
        "candidate_schema_success_rate": round(schema_ok / schema_total, 4) if schema_total else None,
        "candidate_schema_ok": schema_ok, "candidate_schema_total": schema_total,
        "quality_pairwise_informative_rate": round(informative / informative_total, 4) if informative_total else None,
        "quality_ab_ba_order_consistency": round(consistent / consistent_total, 4) if consistent_total else None,
        "quality_candidate_anchor_agreement": round(agree / agree_total, 4) if agree_total else None,
        "risk_candidate_anchor_agreement": round(risk_agree / risk_agree_total, 4) if risk_agree_total else None,
    }
    checks = {
        "schema": schema_total > 0 and metrics["candidate_schema_success_rate"] is not None and metrics["candidate_schema_success_rate"] >= 0.90,
        "informative": informative_total > 0 and metrics["quality_pairwise_informative_rate"] >= THRESHOLDS["minimum_pairwise_informative_rate"],
        "ab_ba_consistency": consistent_total > 0 and metrics["quality_ab_ba_order_consistency"] >= THRESHOLDS["minimum_ab_ba_order_consistency"],
        "quality_agreement": agree_total > 0 and metrics["quality_candidate_anchor_agreement"] >= THRESHOLDS["minimum_anchor_direction_agreement"],
        "risk_agreement": risk_agree_total > 0 and metrics["risk_candidate_anchor_agreement"] >= THRESHOLDS["minimum_anchor_direction_agreement"],
    }
    result = {
        "protocol": PROTOCOL,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
        "counts": {
            "quality_informative_denominator": informative_total,
            "quality_consistency_denominator": consistent_total,
            "quality_agreement_denominator": agree_total,
            "risk_agreement_denominator": risk_agree_total,
        },
        "thresholds": THRESHOLDS,
    }
    write_json(out_dir / "pilot_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal pilot requires {FORMAL_PYTHON}; got {sys.executable}")

    if args.freeze:
        print(json.dumps(freeze(OUT_DIR), ensure_ascii=False))
        return
    manifest = read_json(OUT_DIR / "pilot_manifest.json")
    if manifest.get("status") != "READY_FOR_SCHEMA_SMOKE_THEN_PILOT":
        raise RuntimeError("pilot is not frozen")
    calls = rows(OUT_DIR / "call_plan_private.jsonl")
    if sha256_file(OUT_DIR / "call_plan_private.jsonl") != manifest["call_plan_sha256"]:
        raise RuntimeError("pilot call plan hash drifted")

    if args.smoke:
        run_calls(OUT_DIR, calls, smoke_only=True)
        smoke = calls[: THRESHOLDS["schema_smoke_calls"]]
        success_dir = OUT_DIR / "success_private"
        ok = sum(
            1 for c in smoke
            if (success_dir / f"{c['call_id']}.json").is_file()
            and not read_json(success_dir / f"{c['call_id']}.json").get("parse_failed")
        )
        failures = len(smoke) - ok
        print(json.dumps({
            "status": "SMOKE_PASS" if failures <= THRESHOLDS["schema_smoke_maximum_failures"] else "SMOKE_FAIL",
            "smoke_calls": len(smoke), "smoke_ok": ok, "smoke_failures": failures,
        }))
        return
    if args.run:
        run_calls(OUT_DIR, calls, smoke_only=False)
        print(json.dumps({"status": "RUN_COMPLETE", "total_calls": len(calls)}))
        return
    if args.aggregate:
        print(json.dumps(aggregate(OUT_DIR, calls), ensure_ascii=False))
        return
    raise SystemExit("specify one of --freeze/--smoke/--run/--aggregate")


if __name__ == "__main__":
    main()
