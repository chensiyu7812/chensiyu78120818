#!/usr/bin/env python3
"""Freeze a zero-outcome MS incremental-value repair blueprint."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import runpy
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, MemorySource, RuntimeState
from metacom_pm.contracts import StrategyMode, canonical_action_id
from metacom_pm.io import iter_jsonl, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_d3_features import ms_incremental_value_observation


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-ms-incremental-value-blueprint-v1"
STATUS = "PASS_READY_TO_PREPARE_80_MS_INCREMENTAL_VALUE_CALLS"
REQUEST_SUFFIXES = (
    " I only want help putting this into words, not advice.",
    " Could you offer one small optional suggestion?",
    " I am unsure which part matters most right now.",
    " I want to understand why this feels different from the earlier session today.",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _load_legacy() -> dict[str, Any]:
    return runpy.run_path(
        str(ROOT / "scripts/v1_5/24ct_freeze_d3_ms_replacement_blueprint_v1_5.py")
    )


def _make_user_factory(legacy: dict[str, Any]):
    themes = tuple(legacy["THEMES"])
    distractors = tuple(legacy["DISTRACTORS"])

    def _make_user(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        theme_index = index % len(themes)
        block = index // len(themes)
        theme = themes[theme_index]
        request_index = (theme_index + block) % len(REQUEST_SUFFIXES)
        high_incremental = block % 2 == 0
        core = str(theme["high"] if request_index < 2 else theme["low"])
        if high_incremental:
            relevant = (
                f"An earlier {theme['key']} session clarified one concrete difference and helped name what felt hardest.",
                f"A small optional response to {theme['anchors']} reduced pressure without turning it into a large plan.",
                f"Comparing the current concern with the earlier {theme['key']} session made the immediate priority easier to describe.",
                f"The latest {theme['key']} check-in recorded what worked: slow down and name the one need that is present now.",
            )
            design_cell = "USEFUL_NONREDUNDANT_PRIOR_OUTCOME"
        else:
            relevant = (
                core,
                f"A prior {theme['key']} check-in covered the same general topic but recorded no response or outcome.",
                f"The {theme['anchors']} situation was mentioned in an earlier session.",
                f"The latest {theme['key']} summary repeated the current concern without adding a new distinction.",
            )
            design_cell = "RELEVANT_BUT_NO_INCREMENTAL_VALUE"
        sessions: list[dict[str, Any]] = []
        relevant_sessions = {1, 3, 5, 6}
        relevant_index = 0
        distractor_index = 0
        for session_no in range(1, 7):
            if session_no in relevant_sessions:
                summary = relevant[relevant_index]
                event = f"{theme['anchors']}. {summary}"
                relevant_index += 1
            else:
                summary = (
                    f"Another {theme['key']} check-in found that naming one "
                    f"immediate need helped with {theme['anchors']}."
                    if high_incremental
                    else f"An ordinary {theme['key']} check-in covered "
                    f"{theme['anchors']} but recorded no response or outcome."
                )
                event = summary
                distractor_index += 1
            sessions.append(
                {
                    "id": f"msiv_u{index + 1:03d}_s{session_no}",
                    "timestamp": f"session-{session_no}",
                    "summary": summary,
                    "dialogue": [
                        {"role": "seeker", "content": event},
                        {
                            "role": "supporter",
                            "content": "Thank you for explaining what happened in that session.",
                        },
                    ],
                }
            )
        user_id = f"pmv15_msiv_u{index + 1:03d}"
        user = {
            "id": user_id,
            "basic_info": {"profile_context": theme["profile"]},
            "dialog_history": sessions,
        }
        state = {
            "state_id": "state_" + stable_hex(PROTOCOL, user_id, n=24),
            "card_id": "card_" + stable_hex(PROTOCOL, user_id, "backend", n=24),
            "user_id": user_id,
            "split": "development",
            "semantic_family": f"msiv_{theme['key']}",
            "current_user_text": normalize_space(core + REQUEST_SUFFIXES[request_index]),
            "current_session_history": [
                {
                    "role": "user",
                    "content": f"This week the {theme['anchors']} situation has come up again.",
                },
                {
                    "role": "assistant",
                    "content": "I'm listening. Tell me what feels most relevant today.",
                },
            ],
            "current_session_summary": "",
            "session_index": 7,
            "private_design": {
                "design_cell": design_cell,
                "request_cell": request_index,
                "theme_key": theme["key"],
                "not_a_pm_feature": True,
            },
        }
        return user, state

    return _make_user


def build(*, cards_path: Path, out_dir: Path) -> dict[str, Any]:
    legacy = _load_legacy()
    legacy_globals = legacy["build"].__globals__
    legacy_globals["PROTOCOL"] = PROTOCOL
    legacy_globals["STATUS"] = STATUS
    legacy_globals["INCREMENTAL_WRAPPER_REPLACES_STRATA"] = True
    legacy_globals["_make_user"] = _make_user_factory(legacy_globals)
    # RuntimeState forbids the private construction stratum, so retain it in a
    # separate file while the legacy builder receives only deployable fields.
    design_by_user: dict[str, dict[str, Any]] = {}
    original_make_user = legacy_globals["_make_user"]

    def _deployable_user(index: int):
        user, state = original_make_user(index)
        private = dict(state.pop("private_design"))
        design_by_user[str(user["id"])] = private
        return user, state

    legacy_globals["_make_user"] = _deployable_user
    original_metadata = legacy_globals["_metadata"]

    def _all_ms_summaries_are_topically_relevant(items):
        metadata = original_metadata(items)
        for item in items:
            if item.source is MemorySource.MS:
                metadata[item.memory_id]["expected_relevant_for_current_state"] = True
        return metadata

    legacy_globals["_metadata"] = _all_ms_summaries_are_topically_relevant
    cards = _rows(cards_path)
    available: dict[int, bool] = {}
    for index in range(32):
        _, raw_state = original_make_user(index)
        raw_state.pop("private_design", None)
        available[index] = legacy_globals["_strategy_candidate"](raw_state, cards) is not None
    high = [index for index in range(32) if (index // 8) % 2 == 0]
    low = [index for index in range(32) if (index // 8) % 2 == 1]
    rs_high = [index for index in high if available[index]][:8]
    rs_low = [index for index in low if available[index]][:8]
    if len(rs_high) != 8 or len(rs_low) != 8:
        raise RuntimeError("insufficient RS-applicable states for balanced backgrounds")
    rs_indices = set(rs_high + rs_low)
    slot_by_index: dict[int, int] = {}
    for cohort, slots in (
        (sorted(rs_high), (4, 5, 6, 7)),
        (sorted(rs_low), (4, 5, 6, 7)),
        (sorted(set(high) - rs_indices), (0, 1, 2, 3)),
        (sorted(set(low) - rs_indices), (0, 1, 2, 3)),
    ):
        for position, index in enumerate(cohort):
            slot_by_index[index] = slots[position % len(slots)]

    background_call_index = 0

    def _balanced_background(_: int) -> tuple[str, str]:
        nonlocal background_call_index
        index = background_call_index
        background_call_index += 1
        slot = slot_by_index[index]
        sources = frozenset(
            source
            for source, bit in (
                (MemorySource.MP, 1),
                (MemorySource.ME, 2),
            )
            if slot & bit
        )
        strategy = StrategyMode.RS if slot & 4 else StrategyMode.R0
        return (
            canonical_action_id(sources, strategy),
            canonical_action_id(frozenset({*sources, MemorySource.MS}), strategy),
        )

    legacy_globals["_background"] = _balanced_background
    legacy_report = legacy["build"](cards_path=cards_path, out_dir=out_dir)

    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(out_dir / "runtime_states.jsonl")
        )
    }
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(out_dir / "memory_backend.jsonl")
        )
    }
    contrasts = _rows(out_dir / "d3_ms_replacement_contrasts.jsonl")
    design_rows: list[dict[str, Any]] = []
    by_slot: dict[str, dict[str, Any]] = {}
    for row in contrasts:
        state = states[str(row["state_id"])]
        selected_ids = row["selected_memory_ids_by_action_generation_only"][
            row["treatment_action"]
        ]
        selected_ms = [
            backends[state.card_id][str(memory_id)]
            for memory_id in selected_ids
            if backends[state.card_id][str(memory_id)].source is MemorySource.MS
        ]
        visible = [
            {
                "speaker": "seeker" if turn.role == "user" else "supporter",
                "content": turn.content,
            }
            for turn in state.current_session_history
        ] + [{"speaker": "seeker", "content": state.current_user_text}]
        incremental = ms_incremental_value_observation(
            selected_items=selected_ms,
            current_user_text=state.current_user_text,
            visible_dialogue=visible,
        )
        design = design_by_user[state.user_id]
        expected_alignment = 1.0 if design["design_cell"].startswith("USEFUL") else 0.0
        if incremental["candidate_incremental_alignment_score"] != expected_alignment:
            raise RuntimeError(
                f"incremental alignment construction failed: {state.user_id} "
                f"expected={expected_alignment} actual={incremental}"
            )
        row["component_candidate_observation"]["incremental_value"] = incremental
        row["model_features"].pop("candidate_grounding_or_nonredundancy_score")
        row["model_features"]["candidate_incremental_alignment_score"] = expected_alignment
        by_slot[str(row["contrast_slot_id"])] = row
        design_rows.append(
            {
                "protocol": PROTOCOL,
                "user_id": state.user_id,
                "state_id": state.state_id,
                "contrast_slot_id": row["contrast_slot_id"],
                **design,
                "candidate_incremental_alignment_score": expected_alignment,
                "outcome_read": False,
                "private_construction_only_not_pm_input": True,
            }
        )
    primary = _rows(out_dir / "d3_ms_replacement_primary_pairs.jsonl")
    repeats = _rows(out_dir / "d3_ms_replacement_repeat_pairs.jsonl")
    for collection in (primary, repeats):
        for index, pair in enumerate(collection):
            base = by_slot[str(pair["contrast_slot_id"])]
            collection[index] = {
                "pair_role": pair["pair_role"],
                "pair_seed": pair["pair_seed"],
                **base,
            }

    alignment_counts = Counter(
        row["model_features"]["candidate_incremental_alignment_score"]
        for row in contrasts
    )
    request_counts = Counter(row["request_cell"] for row in design_rows)
    theme_counts = Counter(row["theme_key"] for row in design_rows)
    background_alignment = Counter(
        (
            row["background_action"],
            row["model_features"]["candidate_incremental_alignment_score"],
        )
        for row in contrasts
    )
    checks = {
        "32_new_users": len(design_rows) == 32,
        "alignment_16_high_16_low": alignment_counts == Counter({0.0: 16, 1.0: 16}),
        "four_request_cells_eight_each": len(request_counts) == 4 and set(request_counts.values()) == {8},
        "eight_themes_four_each": len(theme_counts) == 8 and set(theme_counts.values()) == {4},
        "each_background_has_two_high_two_low": all(
            background_alignment[(background, value)] == 2
            for background in {row["background_action"] for row in contrasts}
            for value in (0.0, 1.0)
        ),
        "five_numeric_features": all(
            len(row["model_features"]) == 5
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row["model_features"].values())
            for row in contrasts
        ),
        "no_outcome_read": all(not row["outcome_read"] for row in contrasts),
    }
    if not all(checks.values()):
        raise RuntimeError(f"MS incremental blueprint checks failed: {checks}")

    write_jsonl(out_dir / "d3_ms_replacement_contrasts.jsonl", contrasts)
    write_jsonl(out_dir / "d3_ms_replacement_primary_pairs.jsonl", primary)
    write_jsonl(out_dir / "d3_ms_replacement_repeat_pairs.jsonl", repeats)
    write_jsonl(out_dir / "private_design_strata.jsonl", design_rows)
    report = {
        **legacy_report,
        "protocol": PROTOCOL,
        "status": STATUS,
        "scope": "one_bounded_ms_incremental_value_repair",
        "prior_ms_result": "DIAGNOSTIC_OLD_TREATMENT_NOT_TRAIN_QUALIFIED",
        "candidate_incremental_alignment_counts": {str(key): value for key, value in sorted(alignment_counts.items())},
        "request_cell_counts": {str(key): value for key, value in sorted(request_counts.items())},
        "theme_counts": dict(sorted(theme_counts.items())),
        "incremental_checks": checks,
    }
    for name in (
        "d3_ms_replacement_contrasts.jsonl",
        "d3_ms_replacement_primary_pairs.jsonl",
        "d3_ms_replacement_repeat_pairs.jsonl",
        "private_design_strata.jsonl",
    ):
        report["files"][name] = sha256_file(out_dir / name)
    write_json(out_dir / "preflight_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "preflight_report_sha256": sha256_file(out_dir / "preflight_report.json"),
            "primary_pairs_sha256": sha256_file(out_dir / "d3_ms_replacement_primary_pairs.jsonl"),
            "repeat_pairs_sha256": sha256_file(out_dir / "d3_ms_replacement_repeat_pairs.jsonl"),
            "strategy_cards_sha256": sha256_file(cards_path),
            "repair_contract_sha256": sha256_file(ROOT / "data/pm_v1_5_contracts/ms_incremental_value_repair_v1.json"),
            "outcome_blind": True,
            "api_calls_made": 0,
        },
    )
    return report


def main() -> None:
    cards = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    out_dir = ROOT / "outputs/pm_v1_5_ms_incremental_value_step0_v1"
    report = build(cards_path=cards, out_dir=out_dir)
    print(json.dumps({key: report[key] for key in ("protocol", "status", "users", "total_pairs", "planned_response_calls", "candidate_incremental_alignment_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
