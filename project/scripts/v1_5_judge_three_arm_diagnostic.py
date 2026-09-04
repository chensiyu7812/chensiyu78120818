#!/usr/bin/env python3
"""Position-bias-safe blind judging across all three bank-mechanism arms.

Fixes the position-bias problem found in the earlier single-order judge
(`v1_5_judge_bank_mechanism_diagnostic.py`, which only judged one order per
pair). This script judges every pair in BOTH orders (AB and BA) for all three
pairwise comparisons: legacy-vs-full (already generated), legacy-vs-R0, and
full-vs-R0. A pair only counts as decisive if both orders agree on the winner
after de-anonymizing; disagreement is recorded as "unresolved", not forced
into a winner or a tie. Ad-hoc, not part of the PM-v2.2 gated pipeline;
descriptive/mechanistic only, not confirmatory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import append_jsonl, iter_jsonl, utc_now, write_json
from metacom_pm.prompts import RESPONSE_JUDGE_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = ("legacy_vs_full", "legacy_vs_r0", "full_vs_r0")
CONDITION_KEY = {
    "legacy_vs_full": ("legacy_156", "full_12429"),
    "legacy_vs_r0": ("legacy_156", "r0_no_strategy"),
    "full_vs_r0": ("full_12429", "r0_no_strategy"),
}


def build_context(row: dict) -> str:
    history = row.get("visible_dialogue_before_current_turn") or []
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    return (
        f"Current-session summary:\n(none)\n\n"
        f"Recent dialogue:\n{history_text or '(none)'}\n\n"
        f"Current user message:\n{row['current_seeker_message']}"
    )


def judge_messages(context: str, response_a: str, response_b: str) -> list[dict[str, str]]:
    payload = {
        "shared_visible_context": context,
        "response_A": response_a,
        "response_B": response_b,
    }
    return [
        {"role": "system", "content": RESPONSE_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_all_items(out_dir: Path) -> list[dict]:
    """Build (comparison, pair_id, order, response_A, response_B, condition_A,
    condition_B, context) rows for both orders of all three comparisons."""

    contexts_by_pair: dict[str, str] = {}
    for row in iter_jsonl(out_dir / "dual_order_review_items.jsonl"):
        if row["pair_id"] not in contexts_by_pair:
            contexts_by_pair[row["pair_id"]] = build_context(row)

    responses: dict[tuple[str, str], str] = {}
    for row in iter_jsonl(out_dir / "paired_generations.jsonl"):
        responses[(row["pair_id"], row["bank_condition"])] = row["response"]
    for row in iter_jsonl(out_dir / "r0_generations.jsonl"):
        responses[(row["pair_id"], row["bank_condition"])] = row["response"]

    pair_ids = sorted(contexts_by_pair)
    items = []
    for comparison in COMPARISONS:
        cond_1, cond_2 = CONDITION_KEY[comparison]
        for pair_id in pair_ids:
            if (pair_id, cond_1) not in responses or (pair_id, cond_2) not in responses:
                continue
            context = contexts_by_pair[pair_id]
            r1, r2 = responses[(pair_id, cond_1)], responses[(pair_id, cond_2)]
            items.append(
                {
                    "comparison": comparison,
                    "pair_id": pair_id,
                    "order": "order1",
                    "response_A": r1,
                    "response_B": r2,
                    "condition_A": cond_1,
                    "condition_B": cond_2,
                    "context": context,
                }
            )
            items.append(
                {
                    "comparison": comparison,
                    "pair_id": pair_id,
                    "order": "order2",
                    "response_A": r2,
                    "response_B": r1,
                    "condition_A": cond_2,
                    "condition_B": cond_1,
                    "context": context,
                }
            )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--endpoint", default="synthetic_generator")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "v1_5_bank_mechanism_diagnostic"
    )
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.config)
    endpoint = endpoint_from_config(experiment_config, args.endpoint)

    items = build_all_items(args.out_dir)
    if len(items) > args.max_api_calls:
        raise RuntimeError(f"{len(items)} judge calls exceed --max-api-calls={args.max_api_calls}")

    estimated_usd = 0.0
    for item in items:
        messages = judge_messages(item["context"], item["response_A"], item["response_B"])
        approx_input_tokens = sum(len(m["content"]) for m in messages) / 4.0
        estimated_usd += (
            approx_input_tokens / 1_000_000 * args.input_usd_per_million_tokens
            + 300 / 1_000_000 * args.output_usd_per_million_tokens
        )
    plan_summary = {
        "status": "READY",
        "planned_calls": len(items),
        "by_comparison": dict(Counter(i["comparison"] for i in items)),
        "maximum_estimated_usd": round(estimated_usd, 6),
    }
    if estimated_usd > args.max_estimated_usd:
        raise RuntimeError(
            f"estimated cost ${estimated_usd:.4f} exceeds --max-estimated-usd={args.max_estimated_usd}"
        )
    if args.dry_run:
        print(plan_summary)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(endpoint)
    raw_log_path = args.out_dir / "three_arm_judge_raw_api_calls.jsonl"
    verdicts_path = args.out_dir / "three_arm_judge_verdicts.jsonl"

    verdicts = []
    for item in items:
        messages = judge_messages(item["context"], item["response_A"], item["response_B"])
        result, _ = client.chat(
            messages, temperature=0.0, max_tokens=300, seed=13, response_schema=None, retries=3
        )
        require_reported_usage(result.usage, stage="v1_5_three_arm_judge")
        append_jsonl(
            raw_log_path,
            {
                "timestamp": utc_now(),
                "comparison": item["comparison"],
                "pair_id": item["pair_id"],
                "order": item["order"],
                "model": endpoint.model,
                "usage": result.usage,
                "normalized_finish_reason": result.normalized_finish_reason,
                "raw_text": result.text,
            },
        )
        try:
            parsed = json.loads(result.text)
            preference = str(parsed.get("preference"))
            if preference not in ("A", "B", "tie"):
                preference = "unparseable"
        except (json.JSONDecodeError, AttributeError):
            preference = "unparseable"
        winner_condition = None
        if preference == "A":
            winner_condition = item["condition_A"]
        elif preference == "B":
            winner_condition = item["condition_B"]
        verdict = {
            "comparison": item["comparison"],
            "pair_id": item["pair_id"],
            "order": item["order"],
            "raw_preference": preference,
            "winner_condition": winner_condition,
        }
        verdicts.append(verdict)
        append_jsonl(verdicts_path, verdict)

    by_comparison_pair: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for v in verdicts:
        by_comparison_pair[v["comparison"]][v["pair_id"]].append(v)

    report: dict[str, dict] = {}
    for comparison in COMPARISONS:
        cond_1, cond_2 = CONDITION_KEY[comparison]
        decisive_counts = Counter()
        n_decisive = 0
        n_unresolved = 0
        n_both_tie = 0
        for pair_id, pair_verdicts in by_comparison_pair[comparison].items():
            winners = {v["order"]: v["winner_condition"] for v in pair_verdicts}
            raw = {v["order"]: v["raw_preference"] for v in pair_verdicts}
            if len(winners) != 2:
                continue
            w1, w2 = winners.get("order1"), winners.get("order2")
            r1, r2 = raw.get("order1"), raw.get("order2")
            if r1 == "tie" and r2 == "tie":
                n_both_tie += 1
            elif w1 is not None and w1 == w2:
                decisive_counts[w1] += 1
                n_decisive += 1
            else:
                n_unresolved += 1
        n_pairs = len(by_comparison_pair[comparison])
        report[comparison] = {
            "n_pairs": n_pairs,
            "n_decisive": n_decisive,
            "n_both_tie": n_both_tie,
            "n_unresolved_order_disagreement": n_unresolved,
            "decisive_win_counts": dict(decisive_counts),
            f"{cond_1}_decisive_win_rate": decisive_counts.get(cond_1, 0) / max(n_pairs, 1),
            f"{cond_2}_decisive_win_rate": decisive_counts.get(cond_2, 0) / max(n_pairs, 1),
        }

    summary = {
        "status": "COMPLETE",
        "protocol": "v1-5-three-arm-dual-order-judge-v1",
        "confirmatory": False,
        "note": (
            "Single LLM judge family (not multi-family), but full AB/BA dual "
            "order per pair. A comparison only counts as decisive when both "
            "orders agree on the winner after de-anonymizing; order "
            "disagreement is reported separately as unresolved, not forced "
            "into a winner. Still directional, not confirmatory."
        ),
        "by_comparison": report,
    }
    write_json(args.out_dir / "three_arm_judge_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
