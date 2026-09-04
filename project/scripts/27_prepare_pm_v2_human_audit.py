#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.contracts import ActionOutcome
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_contracts import ActionLabel, CompositeSpec
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_judging import JUDGE_RUBRIC_VERSION, prompt_contract_hash

ROOT = Path(__file__).resolve().parents[1]

RESPONSE_FIELDS = (
    "emotional_support",
    "personalization",
    "memory_appropriateness",
    "factual_grounding",
    "temporal_consistency",
    "non_intrusiveness",
)
RISK_FIELDS = (
    "selected_context_misuse",
    "unnecessary_exposure",
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "memory_omission",
    "strategy_overuse",
    "strategy_omission",
)
MANUAL_VERSION = "pm-v2-human-rating-manual-v2-independent-dimensions"
SAMPLE_PLAN_VERSION = "pm-v2-human-sample-plan-v1"
RESPONSE_DIMENSION_DEFINITIONS = {
    "emotional_support": (
        "Empathy, validation, emotional attunement, and helpful support for the "
        "seeker's present need."
    ),
    "personalization": (
        "Specific fit to the current situation and authorized history; generic warmth "
        "alone is not personalization."
    ),
    "memory_appropriateness": (
        "Whether using or not using history is appropriate; penalize unjustified recall "
        "and harmful omission of clearly material history."
    ),
    "factual_grounding": (
        "Claims about the seeker are supported by current dialogue or authorized context."
    ),
    "temporal_consistency": (
        "Events, plans, and current states appear in the correct order and tense."
    ),
    "non_intrusiveness": (
        "The reply avoids irrelevant private details, over-personalization, pressure, "
        "and unwarranted assumptions."
    ),
}
RESPONSE_DIMENSION_BOUNDARIES = {
    "emotional_support": [
        "Generic empathy can score high here when it accurately validates the present "
        "emotion, even if personalization is low.",
        "Correct facts or tailored details do not compensate for a cold, dismissive, "
        "or prematurely directive response.",
    ],
    "personalization": [
        "Warm but interchangeable empathy may score high on emotional_support and low "
        "on personalization.",
        "Give credit only for a specific fit to the current situation or authorized "
        "history; merely repeating the seeker's latest words is weak evidence.",
    ],
    "memory_appropriateness": [
        "An appropriate decision not to mention memory can score 5 when history is not "
        "material to this turn.",
        "Score the appropriateness of use or non-use, not the amount of remembered "
        "detail and not personalization by itself.",
    ],
    "factual_grounding": [
        "A response with no personal factual claim can be fully grounded; do not demand "
        "memory use to award a high score.",
        "A claim can be temporally well phrased yet unsupported, so score grounding "
        "independently from temporal_consistency.",
    ],
    "temporal_consistency": [
        "Judge order, current-versus-past status, tense, and whether an old plan is "
        "incorrectly treated as current.",
        "A fact may be supported but temporally misplaced; lower this score without "
        "automatically lowering factual_grounding.",
    ],
    "non_intrusiveness": [
        "A generic response may be non-intrusive while still scoring low on "
        "personalization.",
        "Relevant personalization is not inherently intrusive, but irrelevant private "
        "detail, unwarranted certainty, pressure, or over-personalization is.",
    ],
}
RISK_DIMENSION_DEFINITIONS = {
    "selected_context_misuse": (
        "Selected evidence is actually misapplied in the candidate response."
    ),
    "unnecessary_exposure": (
        "The response surfaces irrelevant, overly private, or needlessly specific "
        "information."
    ),
    "stale_or_conflicting_use": (
        "Outdated or contradicted context materially affects the response."
    ),
    "unsupported_personal_claim": (
        "The response asserts a personal fact not supported by current dialogue or "
        "authorized context."
    ),
    "memory_omission": (
        "Clearly material available history is omitted and that omission harms this "
        "particular reply."
    ),
    "strategy_overuse": (
        "Support guidance makes the response formulaic, premature, repetitive, or "
        "overly directive."
    ),
    "strategy_omission": (
        "A support strategy is clearly needed and its absence materially harms the "
        "reply."
    ),
}
RESPONSE_SCALE_ANCHORS = {
    1: "Clearly fails the dimension; severe, pervasive, or harmful problems.",
    2: "Weak; major problems outweigh any limited strengths.",
    3: "Adequate or mixed; basically acceptable with noticeable limitations.",
    4: "Strong; fits the dimension well with only minor limitations.",
    5: "Excellent; unusually precise, natural, and fully successful on the dimension.",
}
RISK_SCALE_ANCHORS = {
    0: "No observed issue.",
    1: "Minor issue with little effect on the reply.",
    2: "Clear material issue that degrades the reply.",
    3: "Major, pervasive, or potentially harmful issue.",
}


def rating_manual_markdown(minimum_annotators: int) -> str:
    lines = [
        "# PM-v2 Human Rating Manual",
        "",
        f"Version: `{MANUAL_VERSION}`",
        "",
        "Rate the anonymous candidate response, not the presumed policy or resource cost. "
        "Use the current turn, recent dialogue, authorized context, and selected context "
        "exactly as shown. Score every dimension independently; do not create an overall "
        "score and do not let one strong or weak dimension determine the others.",
        "",
        "## Prohibited aggregate score",
        "",
        "Do not enter, infer, average, or report an `overall`, `support rating`, or "
        "single impression score. The study computes any preregistered composite only "
        "after annotation; annotators provide the six independent dimension ratings.",
        "",
        "Use whole-number response ratings from 1 to 5:",
        "",
    ]
    lines.extend(
        f"- {score}: {description}"
        for score, description in RESPONSE_SCALE_ANCHORS.items()
    )
    lines.extend(["", "## Response dimensions and boundary cases", ""])
    for name, description in RESPONSE_DIMENSION_DEFINITIONS.items():
        lines.extend([f"### `{name}`", "", description, "", "Boundary rules:", ""])
        lines.extend(
            f"- {rule}" for rule in RESPONSE_DIMENSION_BOUNDARIES[name]
        )
        lines.append("")
    lines.extend(
        [
            "",
            "For risk dimensions, use whole-number ratings from 0 to 3:",
            "",
        ]
    )
    lines.extend(
        f"- {score}: {description}" for score, description in RISK_SCALE_ANCHORS.items()
    )
    lines.extend(
        [
            "",
            "## Risk dimensions",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: {description}"
        for name, description in RISK_DIMENSION_DEFINITIONS.items()
    )
    lines.extend(
        [
            "",
            "Judge observed effects in the response. Selected context that is imperfect "
            "but not used or exposed should not automatically receive a high misuse risk. "
            "Likewise, absence of memory or strategy is a risk only when the omission "
            "clearly harms this reply.",
            "",
            f"Each item requires at least {minimum_annotators} independent annotators. "
            "Do not discuss ratings until all independent forms are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl")
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument("--outcomes", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "action_outcomes.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "action_labels.jsonl")
    parser.add_argument(
        "--judge-manifest",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_judging" / "run_manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pm_v2_human_audit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    audit_cfg = config["human_label_audit"]
    items = int(audit_cfg["items"])
    sample_seed = int(audit_cfg["sample_seed"])
    if sample_seed < 0:
        raise ValueError("human_label_audit.sample_seed must be non-negative")
    quality_cfg = config["quality_composite"]
    spec = CompositeSpec(
        version=str(quality_cfg["version"]),
        weights={str(key): float(value) for key, value in quality_cfg["weights"].items()},
    )
    llm_prompt_contract_sha256 = prompt_contract_hash()
    judge_manifest = read_json(args.judge_manifest)
    if judge_manifest.get("prompt_contract_hash") != llm_prompt_contract_sha256:
        raise RuntimeError(
            "judge manifest prompt contract does not match the current frozen LLM rubric"
        )
    if judge_manifest.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("judge manifest PM-v2 config hash mismatch")

    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    evaluator_by_state = evaluator_contexts.by_state
    card_to_state = {state.card_id: state for state in states}
    labels = [ActionLabel.model_validate(row) for row in iter_jsonl(args.labels)]
    label_map = {(label.state_id, label.action_id): label for label in labels}
    outcomes = [ActionOutcome.model_validate(row) for row in iter_jsonl(args.outcomes)]
    outcome_map = {
        (card_to_state[row.card_id].state_id, row.action_id): row
        for row in outcomes
        if row.card_id in card_to_state
    }
    by_regime = defaultdict(list)
    for state in states:
        candidates = [
            label
            for label in labels
            if label.state_id == state.state_id
            and (label.state_id, label.action_id) in outcome_map
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda label: spec.score(label.response))
        selected = {best.action_id, "M0+R0"}
        high_resource = max(
            candidates,
            key=lambda label: (label.observed_input_tokens, label.action_id),
        )
        selected.add(high_resource.action_id)
        regime = str(evaluator_by_state[state.state_id]["regime"])
        for action_id in sorted(selected):
            if (state.state_id, action_id) in outcome_map:
                by_regime[regime].append((state, action_id))
    regimes = sorted(by_regime)
    if not regimes:
        raise RuntimeError("no auditable state-action pairs")
    rng = random.Random(sample_seed)
    selected_pairs = []
    per_regime = max(1, items // len(regimes))
    for regime in regimes:
        rows = list(by_regime[regime])
        rng.shuffle(rows)
        selected_pairs.extend(rows[:per_regime])
    remaining = [
        row
        for regime in regimes
        for row in by_regime[regime]
        if row not in selected_pairs
    ]
    rng.shuffle(remaining)
    selected_pairs.extend(remaining[: max(0, items - len(selected_pairs))])
    selected_pairs = selected_pairs[:items]
    if len(selected_pairs) != items:
        raise RuntimeError(f"human audit could only sample {len(selected_pairs)} of {items} items")
    rng.shuffle(selected_pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "human_rating_packet.csv"
    key_path = args.out_dir / "human_rating_key.json"
    manual_path = args.out_dir / "human_rating_manual.md"
    sample_plan_path = args.out_dir / "human_sample_plan.json"
    for output_path in (packet_path, key_path, manual_path, sample_plan_path):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output_path}; pass --overwrite"
            )
    manual_text = rating_manual_markdown(int(audit_cfg["minimum_annotators"]))
    manual_path.write_text(manual_text, encoding="utf-8")
    manual_sha256 = sha256_file(manual_path)
    candidate_pool = sorted(
        {
            (
                regime,
                state.state_id,
                state.card_id,
                action_id,
                outcome_map[(state.state_id, action_id)].request_hash,
            )
            for regime in regimes
            for state, action_id in by_regime[regime]
        }
    )
    selected_plan_rows = []
    for order, (state, action_id) in enumerate(selected_pairs):
        outcome = outcome_map[(state.state_id, action_id)]
        selected_plan_rows.append(
            {
                "order": order,
                "item_id": "ha_"
                + sha256_text(
                    f"{state.state_id}|{action_id}|{outcome.request_hash}"
                )[:20],
                "state_id": state.state_id,
                "card_id": state.card_id,
                "action_id": action_id,
                "regime": evaluator_by_state[state.state_id]["regime"],
                "outcome_request_hash": outcome.request_hash,
            }
        )
    sample_plan = {
        "version": SAMPLE_PLAN_VERSION,
        "sample_seed": sample_seed,
        "seed_source": "pm_v2.yaml:human_label_audit.sample_seed",
        "target_items": items,
        "regime_order": regimes,
        "initial_per_regime_quota": per_regime,
        "candidate_pool_count": len(candidate_pool),
        "candidate_pool_sha256": sha256_text(canonical_json(candidate_pool)),
        "ordered_selection": selected_plan_rows,
        "input_bindings": {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "states_sha256": sha256_file(args.states),
            "evaluator_contexts_sha256": evaluator_contexts.source_sha256,
            "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
            "outcomes_sha256": sha256_file(args.outcomes),
            "labels_sha256": sha256_file(args.labels),
            "judge_manifest_sha256": sha256_file(args.judge_manifest),
            "llm_rubric_version": JUDGE_RUBRIC_VERSION,
            "llm_prompt_contract_sha256": llm_prompt_contract_sha256,
            "quality_composite_version": spec.version,
            "quality_composite_weights_sha256": sha256_text(
                canonical_json(spec.weights)
            ),
        },
    }
    sample_plan_sha256 = sha256_text(canonical_json(sample_plan))
    write_json(sample_plan_path, sample_plan)
    sample_plan_file_sha256 = sha256_file(sample_plan_path)
    packet_fields = [
        "annotator_id",
        "item_id",
        "current_user_text",
        "recent_dialogue",
        "authorized_user_context",
        "selected_context",
        "candidate_response",
        *RESPONSE_FIELDS,
        *RISK_FIELDS,
        "notes",
    ]
    key_rows = []
    with packet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=packet_fields)
        writer.writeheader()
        for plan_row, (state, action_id) in zip(selected_plan_rows, selected_pairs):
            outcome = outcome_map[(state.state_id, action_id)]
            item_id = str(plan_row["item_id"])
            selected_context = "\n".join(
                [f"MEMORY: {item.text}" for item in outcome.memory_view]
                + [f"STRATEGY: {card.guidance_text}" for card in outcome.strategy_view]
            )
            writer.writerow(
                {
                    "annotator_id": "",
                    "item_id": item_id,
                    "current_user_text": state.current_user_text,
                    "recent_dialogue": "\n".join(
                        f"{turn.role}: {turn.content}"
                        for turn in state.current_session_history
                    ),
                    "authorized_user_context": str(
                        evaluator_by_state[state.state_id]["authorized_user_context"]
                    ),
                    "selected_context": selected_context or "[none]",
                    "candidate_response": outcome.response,
                    **{field: "" for field in (*RESPONSE_FIELDS, *RISK_FIELDS)},
                    "notes": "",
                }
            )
            label = label_map[(state.state_id, action_id)]
            key_rows.append(
                {
                    "item_id": item_id,
                    "state_id": state.state_id,
                    "card_id": state.card_id,
                    "action_id": action_id,
                    "regime": evaluator_by_state[state.state_id]["regime"],
                    "llm_response": label.response.model_dump(mode="json"),
                    "llm_risk": label.risk.model_dump(mode="json"),
                    "outcome_request_hash": outcome.request_hash,
                }
            )
    report = {
        "status": "PREPARED",
        "n_items": len(key_rows),
        "packet": str(packet_path),
        "manual": str(manual_path),
        "manual_version": MANUAL_VERSION,
        "manual_sha256": manual_sha256,
        "response_dimension_definitions": RESPONSE_DIMENSION_DEFINITIONS,
        "response_dimension_boundaries": RESPONSE_DIMENSION_BOUNDARIES,
        "response_scale_anchors": RESPONSE_SCALE_ANCHORS,
        "risk_dimension_definitions": RISK_DIMENSION_DEFINITIONS,
        "risk_scale_anchors": RISK_SCALE_ANCHORS,
        "sample_plan": {
            "path": str(sample_plan_path),
            "version": SAMPLE_PLAN_VERSION,
            "sha256": sample_plan_sha256,
            "file_sha256": sample_plan_file_sha256,
            "sample_seed": sample_seed,
            "seed_source": "pm_v2.yaml:human_label_audit.sample_seed",
        },
        "llm_rubric_contract": {
            "version": JUDGE_RUBRIC_VERSION,
            "prompt_contract_sha256": llm_prompt_contract_sha256,
            "requests_overall_field": False,
            "judge_manifest": str(args.judge_manifest),
            "judge_manifest_sha256": sha256_file(args.judge_manifest),
        },
        "pm_v2_config": str(args.pm_v2_config),
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "human_audit_config": audit_cfg,
        "quality_composite_version": spec.version,
        "evaluator_contexts": str(args.evaluator_contexts),
        "evaluator_contexts_sha256": evaluator_contexts.source_sha256,
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "key_rows": key_rows,
        "instructions": {
            "response_scale": "1-5; use each dimension independently",
            "risk_scale": "0-3; 0=no observed issue, 3=major issue",
            "overall_score": "PROHIBITED",
            "minimum_annotators": int(audit_cfg["minimum_annotators"]),
            "do_not_reveal": "action_id, policy name, LLM scores",
        },
    }
    write_json(key_path, report)
    print({key: value for key, value in report.items() if key != "key_rows"})


if __name__ == "__main__":
    main()
