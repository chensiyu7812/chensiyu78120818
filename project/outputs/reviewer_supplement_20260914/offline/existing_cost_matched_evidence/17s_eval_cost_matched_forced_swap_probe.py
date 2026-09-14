#!/usr/bin/env python3
"""Forced-swap blind probe for PM vs cost-matched ME+R0.

This is a small, targeted robustness check for the cost-matched fixed-action
comparison.  Unlike the main pointwise V4 scorer, each sampled unit is judged in
both candidate orders:

  order_variant=0: PM first, fixed baseline second
  order_variant=1: fixed baseline first, PM second

The summary reports ordinary score deltas as well as a dual-order preference
that resolves order disagreement to tie.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import Field

from metacom_pm.api import make_client
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import StrictModel
from metacom_pm.evo_response_v4 import _count_message_tokens, _ground_truth
from metacom_pm.evoemo import load_evoemo
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    load_done_keys,
    read_json,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.judging import _call_with_semantic_retry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED_CONDITION = "cost_matched_me_r0"
CONDITIONS = ("pm", DEFAULT_FIXED_CONDITION)
ORDER_BY_VARIANT = {
    0: ("pm", DEFAULT_FIXED_CONDITION),
    1: (DEFAULT_FIXED_CONDITION, "pm"),
}
SCORE_FIELDS = ("support", "personalization")


class CandidateProbeScore(StrictModel):
    candidate_id: str
    support: int = Field(ge=1, le=5)
    personalization: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=500)


class ForcedSwapProbeJudgment(StrictModel):
    candidate_scores: list[CandidateProbeScore] = Field(min_length=2, max_length=2)
    preferred_candidate_id: str = Field(pattern="^(C1|C2|tie)$")
    preference_reason: str = Field(min_length=1, max_length=700)


SYSTEM_PROMPT = """You are an evaluator-only judge for longitudinal emotional-support responses.
Two anonymous candidates answered the same fixed-context seeker turn.
Score both candidates independently on 1-5 integer scales, then give a preference.
Do not reward mentioning more personal facts. Use authorized user context only to
check whether personalization is relevant, grounded, and non-intrusive. Prefer
tie when the two responses are close or involve a tradeoff. Return only JSON
matching the schema."""


RUBRIC = {
    "support": "empathy, validation, warmth, and helpful emotional presence for the current seeker turn",
    "personalization": "uses relevant authorized personal context when useful, without overfitting or exposing unnecessary details",
    "preference": "choose C1 or C2 only if clearly better overall for support and grounded personalization; otherwise choose tie",
    "score_scale": {
        "1": "harmful, mismatched, or unsafe",
        "2": "weak or noticeably flawed",
        "3": "adequate but limited",
        "4": "good and supportive",
        "5": "excellent, deeply supportive, grounded, and non-intrusive",
    },
}


def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _unit_key(row: Mapping[str, Any]) -> tuple[str, int, int, str, str, int]:
    return (
        str(row["user_id"]),
        int(row["topic_index"]),
        int(row["seed"]),
        str(row.get("simulator_id") or "seeker_main"),
        str(row.get("interaction_mode") or "fixed"),
        int(row["turn_index"]),
    )


def _topic_by_index(user: Mapping[str, Any], topic_index: int) -> dict[str, Any]:
    for topic in user.get("subsequent_topics") or []:
        if int(topic["idx"]) == int(topic_index):
            return dict(topic)
    raise KeyError(f"missing topic idx={topic_index} for user={user.get('id')}")


def _load_pm_turns(path: Path) -> dict[tuple[str, int, int, str, str, int], dict[str, Any]]:
    out: dict[tuple[str, int, int, str, str, int], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("condition") != "pm":
            continue
        out[_unit_key(row)] = row
    return out


def build_units(
    *,
    evoemo_path: Path,
    pm_turns_path: Path,
    fixed_turns_path: Path,
    fixed_condition: str,
    ground_truth_mode: str,
) -> list[dict[str, Any]]:
    users = {str(user["id"]): user for user in load_evoemo(evoemo_path)}
    pm_turns = _load_pm_turns(pm_turns_path)
    units: list[dict[str, Any]] = []
    for fixed_turn in iter_jsonl(fixed_turns_path):
        if fixed_turn.get("condition") != fixed_condition:
            continue
        key = _unit_key(fixed_turn)
        pm_turn = pm_turns.get(key)
        if pm_turn is None:
            raise RuntimeError(f"missing PM turn for fixed baseline unit: {key}")
        for field in ("context_sha256", "seeker_message", "state_id", "track_id"):
            if fixed_turn.get(field) != pm_turn.get(field):
                raise RuntimeError(f"fixed baseline context mismatch on {field}: {key}")
        if (fixed_turn.get("context_before_turn") or []) != (pm_turn.get("context_before_turn") or []):
            raise RuntimeError(f"fixed baseline context_before_turn mismatch: {key}")
        user = users[str(fixed_turn["user_id"])]
        topic = _topic_by_index(user, int(fixed_turn["topic_index"]))
        unit_id = fixed_turn.get("unit_id") or (
            "unit_" + stable_hex(
                fixed_turn["user_id"],
                fixed_turn["topic_index"],
                fixed_turn["seed"],
                fixed_turn.get("simulator_id") or "seeker_main",
                fixed_turn.get("interaction_mode") or "fixed",
                fixed_turn["turn_index"],
                n=20,
            )
        )
        units.append({
            "unit_id": unit_id,
            "user_id": str(fixed_turn["user_id"]),
            "topic_index": int(fixed_turn["topic_index"]),
            "seed": int(fixed_turn["seed"]),
            "simulator_id": str(fixed_turn.get("simulator_id") or "seeker_main"),
            "interaction_mode": str(fixed_turn.get("interaction_mode") or "fixed"),
            "track_id": fixed_turn.get("track_id"),
            "state_id": fixed_turn.get("state_id"),
            "context_sha256": fixed_turn.get("context_sha256"),
            "turn_index": int(fixed_turn["turn_index"]),
            "context_before_turn": fixed_turn.get("context_before_turn") or [],
            "seeker_message": fixed_turn["seeker_message"],
            "candidate_turns": {
                "pm": pm_turn,
                fixed_condition: fixed_turn,
            },
            "authorized_ground_truth": _ground_truth(user, topic, mode=ground_truth_mode),
        })
    if not units:
        raise RuntimeError(f"no fixed baseline turns found for condition={fixed_condition}: {fixed_turns_path}")
    return units


def select_units(units: Sequence[dict[str, Any]], sample_units: int, sample_seed: int) -> list[dict[str, Any]]:
    if sample_units <= 0:
        raise ValueError("sample_units must be positive")
    if sample_units >= len(units):
        return list(units)
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_turn[int(unit["turn_index"])].append(unit)
    selected: list[dict[str, Any]] = []
    remaining = sample_units
    turns = sorted(by_turn)
    for i, turn_index in enumerate(turns):
        quota = remaining if i == len(turns) - 1 else sample_units // len(turns)
        remaining -= quota
        pool = sorted(
            by_turn[turn_index],
            key=lambda u: sha256_text(f"{sample_seed}:{u['unit_id']}"),
        )
        selected.extend(pool[:quota])
    return sorted(selected, key=lambda u: (int(u["turn_index"]), str(u["unit_id"])))


def messages_for_unit(unit: Mapping[str, Any], order: Sequence[str]) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for position, condition in enumerate(order):
        candidate_id = f"C{position + 1}"
        turn = unit["candidate_turns"][condition]
        candidates.append({
            "candidate_id": candidate_id,
            "response": turn["supporter_message"],
        })
        mapping[candidate_id] = {
            "condition": condition,
            "position": position + 1,
            "action_id": turn.get("action_id"),
            "input_tokens": turn.get("input_tokens"),
            "output_tokens": turn.get("output_tokens"),
        }
    payload = {
        "task": "forced-swap blind comparison of two fixed-context emotional-support responses",
        "rubric": RUBRIC,
        "authorized_ground_truth": unit["authorized_ground_truth"],
        "case": {
            "unit_id": unit["unit_id"],
            "turn_index": unit["turn_index"],
            "context_before_turn": unit["context_before_turn"],
            "current_seeker_message": unit["seeker_message"],
        },
        "candidates": candidates,
        "output_json_shape": {
            "candidate_scores": [
                {
                    "candidate_id": "C1",
                    "support": 1,
                    "personalization": 1,
                    "reason": "brief evidence-based reason",
                },
                {
                    "candidate_id": "C2",
                    "support": 1,
                    "personalization": 1,
                    "reason": "brief evidence-based reason",
                },
            ],
            "preferred_candidate_id": "C1 | C2 | tie",
            "preference_reason": "brief reason for the preference or tie",
        },
    }
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        mapping,
    )


def _validate_judgment(parsed: ForcedSwapProbeJudgment) -> None:
    ids = [score.candidate_id for score in parsed.candidate_scores]
    if sorted(ids) != ["C1", "C2"]:
        raise ValueError(f"candidate IDs must be exactly C1/C2: {ids}")


def _hash_record(record: dict[str, Any], key: str) -> dict[str, Any]:
    out = dict(record)
    out[key] = ""
    out[key] = sha256_text(canonical_json(out))
    return out


def estimate_cost(
    *,
    units: Sequence[Mapping[str, Any]],
    order_variants: Sequence[int],
    model: str,
    estimated_output_tokens_per_call: int,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    ground_truth_mode: str,
    sample_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    token_counts = []
    for unit in units:
        for order_variant in order_variants:
            order = ORDER_BY_VARIANT[order_variant]
            messages, mapping = messages_for_unit(unit, order)
            tokens = _count_message_tokens(messages, model)
            token_counts.append(tokens)
            rows.append({
                "unit_id": unit["unit_id"],
                "order_variant": order_variant,
                "input_tokens_est": tokens,
                "candidate_order": list(order),
                "candidate_mapping": mapping,
                "prompt_hash": sha256_text(canonical_json(messages)),
            })
    if not token_counts:
        raise ValueError("no calls to estimate")
    sorted_counts = sorted(token_counts)
    p95 = sorted_counts[min(len(sorted_counts) - 1, int(0.95 * (len(sorted_counts) - 1)))]
    total_input = sum(token_counts)
    total_output = len(token_counts) * int(estimated_output_tokens_per_call)
    estimate = {
        "status": "ESTIMATED",
        "protocol": "cost_matched_forced_swap_probe",
        "api_calls": len(token_counts),
        "sample_units": len(units),
        "order_variants": list(order_variants),
        "conditions": list(CONDITIONS),
        "ground_truth_mode": ground_truth_mode,
        "sample_seed": int(sample_seed),
        "input_tokens": {
            "total": total_input,
            "mean": total_input / len(token_counts),
            "min": min(token_counts),
            "p95": p95,
            "max": max(token_counts),
        },
        "estimated_output_tokens": {
            "per_call": int(estimated_output_tokens_per_call),
            "total": total_output,
        },
        "estimated_cost_usd": (
            total_input / 1_000_000 * float(input_usd_per_mtok)
            + total_output / 1_000_000 * float(output_usd_per_mtok)
        ),
        "pricing": {
            "input_usd_per_mtok": float(input_usd_per_mtok),
            "output_usd_per_mtok": float(output_usd_per_mtok),
        },
        "model": model,
        "cost_estimate_sha256": "",
    }
    return _hash_record(estimate, "cost_estimate_sha256"), rows


def require_cost_acceptance(estimate: Mapping[str, Any], accepted: str | None) -> None:
    expected = str(estimate["cost_estimate_sha256"])
    if not accepted:
        raise RuntimeError(
            "missing --accept-cost-estimate-sha256. Run --dry-run first and pass "
            f"the current hash: {expected}"
        )
    if str(accepted) != expected:
        raise RuntimeError(
            "accepted forced-swap probe estimate hash does not match current "
            f"parameters: accepted={accepted}, current={expected}"
        )


def _condition_from_candidate(mapping: Mapping[str, Mapping[str, Any]], candidate_id: str) -> str:
    if candidate_id == "tie":
        return "tie"
    return str(mapping[candidate_id]["condition"])


def write_rows(
    *,
    score_path: Path,
    judgment_path: Path,
    unit: Mapping[str, Any],
    order_variant: int,
    mapping: Mapping[str, Mapping[str, Any]],
    parsed: ForcedSwapProbeJudgment,
) -> None:
    preferred_condition = _condition_from_candidate(mapping, parsed.preferred_candidate_id)
    judgment = {
        "unit_id": unit["unit_id"],
        "order_variant": order_variant,
        "user_id": unit["user_id"],
        "topic_index": unit["topic_index"],
        "seed": unit["seed"],
        "turn_index": unit["turn_index"],
        "preferred_candidate_id": parsed.preferred_candidate_id,
        "preferred_condition": preferred_condition,
        "preference_reason": parsed.preference_reason,
        "candidate_mapping": dict(mapping),
        "judgment": parsed.model_dump(mode="json"),
    }
    append_jsonl(judgment_path, judgment)
    for score in parsed.candidate_scores:
        info = mapping[score.candidate_id]
        append_jsonl(score_path, {
            "unit_id": unit["unit_id"],
            "order_variant": order_variant,
            "user_id": unit["user_id"],
            "topic_index": unit["topic_index"],
            "seed": unit["seed"],
            "turn_index": unit["turn_index"],
            "condition": info["condition"],
            "candidate_id": score.candidate_id,
            "position": info["position"],
            "action_id": info.get("action_id"),
            "support": score.support,
            "personalization": score.personalization,
            "preferred_condition": preferred_condition,
            "reason": score.reason,
        })


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_ci(values: Sequence[float], *, seed: int = 12345, reps: int = 5000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "ci95": [means[int(0.025 * reps)], means[min(reps - 1, int(0.975 * reps))]],
    }


def summarize(score_path: Path, judgment_path: Path) -> dict[str, Any]:
    scores = list(iter_jsonl(score_path))
    judgments = list(iter_jsonl(judgment_path))
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_position: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_condition_position: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scores:
        condition = str(row["condition"])
        position = int(row["position"])
        for field in SCORE_FIELDS:
            by_condition[condition][field].append(float(row[field]))
            by_position[position][field].append(float(row[field]))
            by_condition_position[(condition, position)][field].append(float(row[field]))

    unit_order_scores: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in scores:
        unit_order_scores[(str(row["unit_id"]), int(row["order_variant"]))][str(row["condition"])] = row

    order_deltas: dict[str, dict[str, Any]] = {}
    for order_variant in sorted({key[1] for key in unit_order_scores}):
        diffs_by_field: dict[str, list[float]] = defaultdict(list)
        wins = ties = losses = 0
        for (unit_id, ov), conds in unit_order_scores.items():
            if ov != order_variant or set(conds) != set(CONDITIONS):
                continue
            support_diff = float(conds["pm"]["support"]) - float(conds[DEFAULT_FIXED_CONDITION]["support"])
            if support_diff > 0:
                wins += 1
            elif support_diff < 0:
                losses += 1
            else:
                ties += 1
            for field in SCORE_FIELDS:
                diffs_by_field[field].append(
                    float(conds["pm"][field]) - float(conds[DEFAULT_FIXED_CONDITION][field])
                )
        order_deltas[str(order_variant)] = {
            field: _bootstrap_ci(values)
            for field, values in diffs_by_field.items()
        }
        order_deltas[str(order_variant)]["support_wtl_by_score"] = {
            "pm_win": wins,
            "tie": ties,
            "pm_loss": losses,
        }

    judgments_by_unit: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in judgments:
        judgments_by_unit[str(row["unit_id"])][int(row["order_variant"])] = row

    dual_counts = {"pm": 0, DEFAULT_FIXED_CONDITION: 0, "tie": 0, "order_disagreement": 0, "missing": 0}
    dual_units = []
    for unit_id, by_order in judgments_by_unit.items():
        if 0 not in by_order or 1 not in by_order:
            dual_counts["missing"] += 1
            continue
        pref0 = str(by_order[0]["preferred_condition"])
        pref1 = str(by_order[1]["preferred_condition"])
        if pref0 == pref1:
            resolved = pref0
        elif pref0 == "tie":
            resolved = pref1
        elif pref1 == "tie":
            resolved = pref0
        else:
            resolved = "tie"
            dual_counts["order_disagreement"] += 1
        dual_counts[resolved] = dual_counts.get(resolved, 0) + 1
        dual_units.append({
            "unit_id": unit_id,
            "order0_preference": pref0,
            "order1_preference": pref1,
            "resolved_preference": resolved,
        })

    averaged_diffs: dict[str, list[float]] = defaultdict(list)
    units = sorted({key[0] for key in unit_order_scores})
    for unit_id in units:
        by_cond_field: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for order_variant in (0, 1):
            conds = unit_order_scores.get((unit_id, order_variant), {})
            if set(conds) != set(CONDITIONS):
                continue
            for condition, row in conds.items():
                for field in SCORE_FIELDS:
                    by_cond_field[condition][field].append(float(row[field]))
        if set(by_cond_field) != set(CONDITIONS):
            continue
        for field in SCORE_FIELDS:
            pm_mean = _mean(by_cond_field["pm"][field])
            fixed_mean = _mean(by_cond_field[DEFAULT_FIXED_CONDITION][field])
            if pm_mean is not None and fixed_mean is not None:
                averaged_diffs[field].append(pm_mean - fixed_mean)

    return {
        "condition_summary": {
            condition: {
                field: _mean(values)
                for field, values in fields.items()
            }
            for condition, fields in sorted(by_condition.items())
        },
        "position_summary": {
            str(position): {
                field: _mean(values)
                for field, values in fields.items()
            }
            for position, fields in sorted(by_position.items())
        },
        "condition_position_summary": {
            f"{condition}@{position}": {
                field: _mean(values)
                for field, values in fields.items()
            }
            for (condition, position), fields in sorted(by_condition_position.items())
        },
        "order_deltas_pm_minus_fixed": order_deltas,
        "dual_order_mean_deltas_pm_minus_fixed": {
            field: _bootstrap_ci(values)
            for field, values in averaged_diffs.items()
        },
        "dual_order_preference": dual_counts,
        "dual_order_units": dual_units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--endpoint", default="final_judge")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data/external/evo_emo.json")
    parser.add_argument("--pm-turns", type=Path, default=ROOT / "outputs/evoemo_selective/turns.jsonl")
    parser.add_argument("--fixed-turns", type=Path, default=ROOT / "outputs/evoemo_cost_matched_me_r0_generation/turns.jsonl")
    parser.add_argument("--fixed-condition", default=DEFAULT_FIXED_CONDITION)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/evoemo_cost_matched_forced_swap_probe")
    parser.add_argument("--ground-truth-mode", choices=["full", "compact_related"], default="full")
    parser.add_argument("--sample-units", type=int, default=40)
    parser.add_argument("--sample-seed", type=int, default=17019)
    parser.add_argument("--order-variants", default="0,1")
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--max-api-calls", type=int, default=100)
    parser.add_argument("--max-estimated-usd", type=float, default=2.5)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--estimated-output-tokens-per-call", type=int, default=350)
    parser.add_argument("--input-usd-per-mtok", type=float, default=2.50)
    parser.add_argument("--output-usd-per-mtok", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.evoemo, args.pm_turns, args.fixed_turns):
        if not path.is_file():
            raise FileNotFoundError(path)
    order_variants = _csv_ints(args.order_variants)
    unknown = [x for x in order_variants if x not in ORDER_BY_VARIANT]
    if unknown:
        raise ValueError(f"unsupported forced-swap order variants: {unknown}")

    all_units = build_units(
        evoemo_path=args.evoemo,
        pm_turns_path=args.pm_turns,
        fixed_turns_path=args.fixed_turns,
        fixed_condition=args.fixed_condition,
        ground_truth_mode=args.ground_truth_mode,
    )
    units = select_units(all_units, args.sample_units, args.sample_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint_from_config(load_config(args.config), args.endpoint)
    estimate, rows = estimate_cost(
        units=units,
        order_variants=order_variants,
        model=endpoint.model,
        estimated_output_tokens_per_call=args.estimated_output_tokens_per_call,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        ground_truth_mode=args.ground_truth_mode,
        sample_seed=args.sample_seed,
    )
    budget_checks = {
        "api_calls": int(estimate["api_calls"]) <= int(args.max_api_calls),
        "estimated_cost_usd": float(estimate["estimated_cost_usd"]) <= float(args.max_estimated_usd),
        "max_input_tokens_per_call": int(estimate["input_tokens"]["max"]) <= int(args.max_input_tokens_per_call),
    }
    estimate["budget_gate"] = {
        "status": "PASS" if all(budget_checks.values()) else "FAILED",
        "checks": budget_checks,
        "limits": {
            "max_api_calls": args.max_api_calls,
            "max_estimated_usd": args.max_estimated_usd,
            "max_input_tokens_per_call": args.max_input_tokens_per_call,
        },
    }
    estimate["sample_unit_ids"] = [unit["unit_id"] for unit in units]
    write_json(args.out_dir / "cost_estimate.json", estimate)
    write_jsonl(args.out_dir / "cost_estimate_calls.jsonl", rows)
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("forced-swap probe budget gate failed: " + json.dumps(estimate["budget_gate"], ensure_ascii=False))
    if args.dry_run:
        print({
            "status": "DRY_RUN_ONLY",
            "protocol": "cost_matched_forced_swap_probe",
            "sample_units": len(units),
            "expected_calls": estimate["api_calls"],
            "estimated_cost_usd": round(float(estimate["estimated_cost_usd"]), 6),
            "input_tokens": estimate["input_tokens"],
            "cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "cost_estimate": str(args.out_dir / "cost_estimate.json"),
        })
        return

    require_cost_acceptance(estimate, args.accept_cost_estimate_sha256)
    score_path = args.out_dir / "response_scores.jsonl"
    judgment_path = args.out_dir / "response_judgments.jsonl"
    raw_path = args.out_dir / "response_raw_calls.jsonl"
    summary_path = args.out_dir / "response_summary.json"
    if args.overwrite:
        for path in (score_path, judgment_path, raw_path, summary_path):
            if path.exists():
                path.unlink()
    done = load_done_keys(judgment_path, ("unit_id", "order_variant"))
    client = make_client(endpoint)
    try:
        for unit in units:
            for order_variant in order_variants:
                done_key = (unit["unit_id"], order_variant)
                if done_key in done:
                    continue
                order = ORDER_BY_VARIANT[order_variant]
                messages, mapping = messages_for_unit(unit, order)
                parsed = _call_with_semantic_retry(
                    client,
                    endpoint,
                    messages,
                    ForcedSwapProbeJudgment,
                    _validate_judgment,
                    stage="cost_matched_forced_swap_probe",
                    record_ids={
                        "unit_id": unit["unit_id"],
                        "order_variant": order_variant,
                        "user_id": unit["user_id"],
                        "topic_index": unit["topic_index"],
                        "seed": unit["seed"],
                        "turn_index": unit["turn_index"],
                    },
                    raw_log_path=raw_path,
                    max_tokens=args.max_tokens,
                )
                assert isinstance(parsed, ForcedSwapProbeJudgment)
                write_rows(
                    score_path=score_path,
                    judgment_path=judgment_path,
                    unit=unit,
                    order_variant=order_variant,
                    mapping=mapping,
                    parsed=parsed,
                )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    expected_score_rows = len(units) * len(order_variants) * len(CONDITIONS)
    score_rows = list(iter_jsonl(score_path))
    judgment_rows = list(iter_jsonl(judgment_path))
    raw_rows = list(iter_jsonl(raw_path))
    validation = {
        "ok": (
            len(score_rows) == expected_score_rows
            and len(judgment_rows) == len(units) * len(order_variants)
            and len([r for r in raw_rows if not r.get("error")]) == len(units) * len(order_variants)
        ),
        "expected_calls": len(units) * len(order_variants),
        "judgment_rows": len(judgment_rows),
        "score_rows": len(score_rows),
        "raw_rows": len(raw_rows),
        "successful_raw_calls": len([r for r in raw_rows if not r.get("error")]),
        "expected_score_rows": expected_score_rows,
    }
    summary = {
        "status": "COMPLETE",
        "protocol": "cost_matched_forced_swap_probe",
        "conditions": list(CONDITIONS),
        "sample_units": len(units),
        "order_variants": list(order_variants),
        "output_validation": validation,
        "score_summary": summarize(score_path, judgment_path),
        "cost_estimate": estimate,
        "outputs": {
            "scores": str(score_path),
            "judgments": str(judgment_path),
            "raw_calls": str(raw_path),
            "cost_estimate": str(args.out_dir / "cost_estimate.json"),
        },
    }
    write_json(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
