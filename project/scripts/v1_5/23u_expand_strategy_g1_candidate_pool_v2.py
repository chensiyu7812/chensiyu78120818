#!/usr/bin/env python3
"""Expand G1 recall with train-only native strata and transparent patterns.

Native ESConv labels and regexes select source-review candidates only. They are
hidden from Coder B and never treated as atomic-move gold.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ID_PROTOCOL = "pm-v1.5-strategy-g1-full-universe-candidate-pool-v1"
MOVE_NATIVE_LABELS = {
    "AM01_invite_open_expression": {"Question", "Others"},
    "AM02_ask_one_focused_clarification": {"Question"},
    "AM03_check_feeling_or_coping": {"Question", "Reflection of feelings"},
    "AM04_tentative_paraphrase_check": {"Restatement or Paraphrasing"},
    "AM05_grounded_validation": {
        "Reflection of feelings",
        "Affirmation and Reassurance",
    },
    "AM06_bounded_normalization": {
        "Affirmation and Reassurance",
        "Reflection of feelings",
    },
    "AM07_acknowledge_effort_strength_or_resource": {
        "Affirmation and Reassurance"
    },
    "AM08_repair_misunderstanding": {
        "Others",
        "Restatement or Paraphrasing",
    },
    "AM09_offer_collaborative_structuring": {
        "Question",
        "Providing Suggestions",
        "Others",
    },
    "AM10_offer_one_optional_micro_step": {"Providing Suggestions"},
    "AM11_reinforce_user_generated_plan": {
        "Affirmation and Reassurance",
        "Providing Suggestions",
    },
    "AM12_connect_to_trusted_support": {
        "Providing Suggestions",
        "Information",
    },
    "AM13_offer_bounded_hope": {"Affirmation and Reassurance"},
    "AM14_supportive_transition": {"Others"},
    "AM15_explore_interpersonal_boundary": {
        "Providing Suggestions",
        "Question",
    },
    "AM16_gentle_behavior_focused_challenge": {"Question", "Others"},
    "AM17_explain_brief_non_domain_rationale": {
        "Information",
        "Providing Suggestions",
    },
}
MOVE_PATTERNS = {
    "AM01_invite_open_expression": [
        r"\btell me\b",
        r"\bwant to talk\b",
        r"\bwhat (?:is|s) (?:the )?(?:problem|matter)\b",
        r"\bhow (?:may|can) i help\b",
    ],
    "AM02_ask_one_focused_clarification": [
        r"\b(?:what|when|where|which|who|why|how)\b",
        r"\?",
    ],
    "AM03_check_feeling_or_coping": [
        r"\bfeel(?:ing|s)?\b",
        r"\bcope|coping\b",
        r"\btake care of\b",
        r"\bhow are you doing\b",
    ],
    "AM04_tentative_paraphrase_check": [
        r"\bsounds like\b",
        r"\bif i understand\b",
        r"\byou mean\b",
        r"\bis that (?:right|correct)\b",
    ],
    "AM05_grounded_validation": [
        r"\bunderstand\b",
        r"\b(?:hard|difficult|frustrating|scary|painful)\b",
        r"\bsorry to hear\b",
    ],
    "AM06_bounded_normalization": [
        r"\bnormal\b",
        r"\bunderstandable\b",
        r"\bnot alone\b",
        r"\bmany people\b",
    ],
    "AM07_acknowledge_effort_strength_or_resource": [
        r"\bstrong\b",
        r"\beffort\b",
        r"\bproud\b",
        r"\bgood (?:job|idea|plan)\b",
    ],
    "AM08_repair_misunderstanding": [
        r"\bi apologize\b",
        r"\bsorry[, ]",
        r"\bmisunderst",
        r"\bi (?:see|understand) now\b",
    ],
    "AM09_offer_collaborative_structuring": [
        r"\bwe can\b",
        r"\btogether\b",
        r"\bfigure (?:it|this|that) out\b",
        r"\bfind a solution\b",
    ],
    "AM10_offer_one_optional_micro_step": [
        r"\b(?:maybe|could|try|consider|perhaps|how about)\b",
        r"\brecommend\b",
        r"\bsuggest\b",
    ],
    "AM11_reinforce_user_generated_plan": [
        r"\bgood idea\b",
        r"\bsounds like a (?:good )?plan\b",
        r"\bglad you\b",
        r"\bthat (?:could|may|might) help\b",
    ],
    "AM12_connect_to_trusted_support": [
        r"\b(?:friend|family|trusted|support group|counsel)\w*\b",
    ],
    "AM13_offer_bounded_hope": [
        r"\bhope\b",
        r"\bpossible\b",
        r"\b(?:may|might|could) (?:improve|get better|work)\b",
    ],
    "AM14_supportive_transition": [
        r"\btalk about something else\b",
        r"\bcontinue\b",
        r"\banything else\b",
        r"\btake care\b",
    ],
    "AM15_explore_interpersonal_boundary": [
        r"\bboundar",
        r"\blimit\b",
        r"\b(?:distance|space)\b",
        r"\bsay no\b",
    ],
    "AM16_gentle_behavior_focused_challenge": [
        r"\bwhat if\b",
        r"\bcould it be\b",
        r"\bdo you think\b",
        r"\bconsider\b",
    ],
    "AM17_explain_brief_non_domain_rationale": [
        r"\bbecause\b",
        r"\bso that\b",
        r"\bcan help\b",
        r"\bthe reason\b",
    ],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate_id(row: dict[str, Any]) -> str:
    source = (
        f"{ID_PROTOCOL}|{row['source_dialogue_id']}|"
        f"{row['source_turn_index']}|{row['strategy_id']}"
    )
    return "strategy_g1_" + hashlib.sha256(source.encode()).hexdigest()[:20]


def main() -> None:
    universe_path = (
        ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "clean_train_strategy_universe.jsonl"
    )
    pilot_dir = ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
    pilot_labels_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_run_v1"
        / "weak_labels.jsonl"
    )
    out_dir = ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v2"
    per_move = 60

    universe = _read_jsonl(universe_path)
    if len(universe) != 9148:
        raise ValueError("expected 9148 clean train rows")
    pilot_public = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pilot_dir / "public_packet.jsonl")
    }
    pilot_private = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pilot_dir / "private_lineage.jsonl")
    }
    completed_pilot_ids = {
        str(row["blind_item_id"]) for row in _read_jsonl(pilot_labels_path)
    }
    if len(completed_pilot_ids) != 122:
        raise ValueError("completed pilot must contain 122 items")

    selected_by_source_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_move_report: dict[str, Any] = {}
    for move_id, native_labels in MOVE_NATIVE_LABELS.items():
        patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in MOVE_PATTERNS[move_id]]
        candidates: list[tuple[int, int, int, str, int]] = []
        for index, row in enumerate(universe):
            if str(row["strategy_label"]) not in native_labels:
                continue
            response = str(row["supporter_response"])
            pattern_hits = sum(bool(pattern.search(response)) for pattern in patterns)
            candidates.append(
                (
                    -pattern_hits,
                    -int(bool(row["has_question_mark"])),
                    int(row["word_count"]),
                    str(row["source_dialogue_id"]),
                    index,
                )
            )
        candidates.sort()
        chosen: list[int] = []
        seen_dialogues: set[str] = set()
        for neg_pattern_hits, _, _, dialogue_id, source_index in candidates:
            if dialogue_id in seen_dialogues:
                continue
            rank = len(chosen) + 1
            chosen.append(source_index)
            seen_dialogues.add(dialogue_id)
            selected_by_source_index[source_index].append(
                {
                    "move_id": move_id,
                    "native_pattern_rank": rank,
                    "transparent_pattern_hit_count": -neg_pattern_hits,
                    "native_label_used_for_candidate_recall": str(
                        universe[source_index]["strategy_label"]
                    ),
                }
            )
            if len(chosen) == per_move:
                break
        if len(chosen) != per_move:
            raise ValueError(f"{move_id}: insufficient native-stratum candidates")
        per_move_report[move_id] = {
            "native_pattern_distinct_dialogues": len(chosen),
            "rows_with_at_least_one_pattern_hit": sum(
                any(pattern.search(str(universe[index]["supporter_response"])) for pattern in patterns)
                for index in chosen
            ),
        }

    public: dict[str, dict[str, Any]] = dict(pilot_public)
    private: dict[str, dict[str, Any]] = dict(pilot_private)
    for source_index, move_proposals in selected_by_source_index.items():
        row = universe[source_index]
        item_id = _candidate_id(row)
        public[item_id] = {
            "blind_item_id": item_id,
            "recent_visible_dialogue": row["recent_dialogue"],
            "supporter_response_to_label": row["supporter_response"],
        }
        existing = private.get(item_id, {})
        prior_proposals = list(existing.get("candidate_move_proposals") or [])
        private[item_id] = {
            "blind_item_id": item_id,
            "source_dialogue_id": row["source_dialogue_id"],
            "source_turn_index": row["source_turn_index"],
            "strategy_id": row["strategy_id"],
            "native_strategy_label_audit_only": row["strategy_label"],
            "problem_type_audit_only": row["problem_type"],
            "emotion_type_audit_only": row["emotion_type"],
            "experience_type_audit_only": row["experience_type"],
            "first_person_language_flag": bool(row["first_person_language_flag"]),
            "domain_claim_keyword_flag": bool(row["domain_claim_keyword_flag"]),
            "candidate_move_proposals": prior_proposals,
            "native_pattern_candidate_proposals": sorted(
                move_proposals, key=lambda value: str(value["move_id"])
            ),
            "low_score_control_for_move_ids": list(
                existing.get("low_score_control_for_move_ids") or []
            ),
        }

    ordered_ids = sorted(public)
    if set(public) != set(private):
        raise ValueError("v2 public/private candidate IDs differ")
    new_ids = [item_id for item_id in ordered_ids if item_id not in completed_pilot_ids]
    public_rows = [public[item_id] for item_id in ordered_ids]
    private_rows = [private[item_id] for item_id in ordered_ids]
    new_public_rows = [public[item_id] for item_id in new_ids]
    new_private_rows = [private[item_id] for item_id in new_ids]
    report = {
        "protocol": "pm-v1.5-strategy-g1-candidate-pool-v2",
        "status": "TARGETED_RECALL_EXPANSION_READY_NO_NEW_API_CALLS",
        "full_train_universe_rows_considered": len(universe),
        "completed_pilot_rows_reused": len(completed_pilot_ids),
        "native_pattern_candidates_per_move": per_move,
        "targeted_pool_rows_including_pilot": len(public_rows),
        "new_unlabeled_rows": len(new_public_rows),
        "planned_new_single_coder_calls_at_batch_8": (
            len(new_public_rows) + 7
        )
        // 8,
        "targeted_pool_dialogues": len(
            {str(row["source_dialogue_id"]) for row in private_rows}
        ),
        "per_move": per_move_report,
        "native_labels_visible_to_coder": False,
        "native_labels_are_gold_atomic_moves": False,
        "transparent_patterns_are_gold_atomic_moves": False,
        "selection_role": "candidate_recall_only",
        "next_step": (
            "Run the same qualified Coder B and literal-evidence validator on "
            "new_unlabeled_public_packet, then stop per move once 20 distinct "
            "source-compatible dialogues survive literal-source audit."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "targeted_public_packet.jsonl", public_rows)
    _write_jsonl(out_dir / "targeted_private_lineage.jsonl", private_rows)
    _write_jsonl(out_dir / "new_unlabeled_public_packet.jsonl", new_public_rows)
    _write_jsonl(out_dir / "new_unlabeled_private_lineage.jsonl", new_private_rows)
    _write_json(out_dir / "candidate_pool_v2_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
