#!/usr/bin/env python3
"""Ad-hoc blind pairwise LLM judging of the V1 bank-mechanism diagnostic.

This is NOT part of the PM-v2.2 gated pipeline and produces no training
labels. It reuses the project's existing blind pairwise response-judge
prompt (`RESPONSE_JUDGE_SYSTEM`) to substitute for the dual-annotator human
review that `34_posthoc_v1_bank_mechanism_diagnostic.py` also generated
(`reviewer_a.csv` / `reviewer_b.csv`), because the user explicitly opted for
an automated read instead of spending human-annotator time. The diagnostic
itself remains descriptive/mechanistic, not confirmatory.
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
from metacom_pm.paid_run_release import require_paid_run_release

ROOT = Path(__file__).resolve().parents[1]


def build_context(row: dict) -> str:
    history = row.get("visible_dialogue_before_current_turn") or []
    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in history
    )
    return (
        f"Current-session summary:\n(none)\n\n"
        f"Recent dialogue:\n{history_text or '(none)'}\n\n"
        f"Current user message:\n{row['current_seeker_message']}"
    )


def judge_messages(row: dict) -> list[dict[str, str]]:
    payload = {
        "shared_visible_context": build_context(row),
        "response_A": row["response_A"],
        "response_B": row["response_B"],
    }
    return [
        {"role": "system", "content": RESPONSE_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--paid-run-identity")
    parser.add_argument("--endpoint", default="synthetic_generator")
    parser.add_argument(
        "--review-items",
        type=Path,
        default=ROOT / "outputs" / "v1_5_bank_mechanism_diagnostic" / "dual_order_review_items.jsonl",
    )
    parser.add_argument(
        "--review-plan",
        type=Path,
        default=ROOT / "outputs" / "v1_5_bank_mechanism_diagnostic" / "private_blind_review_plan.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "v1_5_bank_mechanism_diagnostic")
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.config)
    require_paid_run_release(
        load_config(args.pm_v1_5_config),
        config_path=args.pm_v1_5_config,
        stage="diagnostic_bank_mechanism_judging",
        run=bool(args.run),
        run_identity=args.paid_run_identity,
    )
    endpoint = endpoint_from_config(experiment_config, args.endpoint)

    items = list(iter_jsonl(args.review_items))
    plan_by_item = {
        row["item_id"]: row for row in iter_jsonl(args.review_plan)
    }
    items_a_only = [row for row in items if plan_by_item[row["item_id"]]["reviewer_id"] == "reviewer_a"]

    if len(items_a_only) > args.max_api_calls:
        raise RuntimeError(
            f"{len(items_a_only)} judge calls exceed --max-api-calls={args.max_api_calls}"
        )

    estimated_usd = 0.0
    for row in items_a_only:
        messages = judge_messages(row)
        approx_input_tokens = sum(len(m["content"]) for m in messages) / 4.0
        estimated_usd += (
            approx_input_tokens / 1_000_000 * args.input_usd_per_million_tokens
            + 300 / 1_000_000 * args.output_usd_per_million_tokens
        )
    if estimated_usd > args.max_estimated_usd:
        raise RuntimeError(
            f"estimated cost ${estimated_usd:.4f} exceeds --max-estimated-usd={args.max_estimated_usd}"
        )

    plan_summary = {
        "status": "READY",
        "planned_calls": len(items_a_only),
        "maximum_estimated_usd": round(estimated_usd, 6),
    }
    if args.dry_run:
        print(plan_summary)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(endpoint)
    raw_log_path = args.out_dir / "judge_raw_api_calls.jsonl"
    verdicts_path = args.out_dir / "judge_verdicts.jsonl"

    verdicts = []
    for row in items_a_only:
        item_id = row["item_id"]
        messages = judge_messages(row)
        result, _ = client.chat(
            messages,
            temperature=0.0,
            max_tokens=300,
            seed=13,
            response_schema=None,
            retries=3,
        )
        require_reported_usage(result.usage, stage="v1_5_bank_mechanism_judge")
        append_jsonl(
            raw_log_path,
            {
                "timestamp": utc_now(),
                "item_id": item_id,
                "model": endpoint.model,
                "usage": result.usage,
                "normalized_finish_reason": result.normalized_finish_reason,
                "raw_text": result.text,
            },
        )
        try:
            parsed = json.loads(result.text)
            preference = str(parsed.get("preference"))
        except (json.JSONDecodeError, AttributeError):
            preference = "unparseable"
        plan_row = plan_by_item[item_id]
        winner_condition = None
        if preference in ("A", "B"):
            winner_condition = plan_row[f"response_{preference}_condition"]
        verdict = {
            "item_id": item_id,
            "pair_id": row["pair_id"],
            "order_variant": row["order_variant"],
            "raw_preference": preference,
            "winner_condition": winner_condition,
            "response_A_condition": plan_row["response_A_condition"],
            "response_B_condition": plan_row["response_B_condition"],
        }
        verdicts.append(verdict)
        append_jsonl(verdicts_path, verdict)

    counts = Counter(v["winner_condition"] or "tie_or_unparseable" for v in verdicts)
    by_pair = defaultdict(list)
    for v in verdicts:
        by_pair[v["pair_id"]].append(v)

    summary = {
        "status": "COMPLETE",
        "protocol": "v1-5-bank-mechanism-judge-v1",
        "confirmatory": False,
        "note": (
            "Automated single-family LLM judge substituting for the dual-"
            "annotator human review this diagnostic also prepared "
            "(reviewer_a.csv/reviewer_b.csv). This is a single-order read "
            "per pair (reviewer_a's order assignment only), not the full "
            "dual-order design; treat as directional, not confirmatory."
        ),
        "n_pairs_judged": len(by_pair),
        "winner_counts": dict(counts),
        "legacy_156_win_rate": counts.get("legacy_156", 0) / max(len(by_pair), 1),
        "full_12429_win_rate": counts.get("full_12429", 0) / max(len(by_pair), 1),
        "tie_or_unparseable_rate": counts.get("tie_or_unparseable", 0) / max(len(by_pair), 1),
    }
    write_json(args.out_dir / "judge_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
