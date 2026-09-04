#!/usr/bin/env python3
"""Freeze the untouched ESConv/EvoEmo panels for the final V1.5b system test.

ESConv contributes one support-eligible turn per independent test dialogue,
chosen by a protocol hash without reading candidates or outcomes.  EvoEmo uses
the already-frozen open-loop tracks at the preregistered turns 3 and 8.  The
script performs no response generation and reads no quality/risk judgments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.config import load_config
from metacom_pm.evoemo import _fixed_context_before_turn, load_evoemo, make_evo_runtime_state
from metacom_pm.io import iter_jsonl, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.v1_5_memory_transport import compile_bounded_memory_with_metadata


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-final-external-panel-freeze-v1"
ESCONV_SELECTION_SEED = "pm-v1.5b-esconv-one-state-per-dialogue-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(*, root: Path = ROOT) -> dict[str, Any]:
    esconv_path = root / "data/esconv_test_v1_5/runtime_states.jsonl"
    esconv_audit_path = root / "data/esconv_test_v1_5/split_audit.json"
    tracks_path = root / "outputs/evoemo_fixed_tracks_v1_5_v3_formal_fresh_candidate/fixed_seeker_tracks.jsonl"
    evoemo_path = root / "data/external/evo_emo.json"
    config_path = root / "configs/pm_v1_5.yaml"
    config = load_config(config_path)
    turn_indices = tuple(int(value) for value in config["external_evaluation"]["turn_indices"])
    if turn_indices != (3, 8):
        raise RuntimeError("final EvoEmo turns must remain frozen at 3 and 8")

    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(esconv_path):
        by_dialogue[str(row["user_id"])].append(row)
    esconv_selected: list[dict[str, Any]] = []
    for dialogue_id, values in sorted(by_dialogue.items()):
        selected = min(
            values,
            key=lambda row: stable_hex(
                ESCONV_SELECTION_SEED,
                dialogue_id,
                str(row["state_id"]),
                n=32,
            ),
        )
        esconv_selected.append(
            {
                "protocol": PROTOCOL,
                "domain": "ESConv",
                "panel_id": "esconv_" + stable_hex(PROTOCOL, selected["state_id"], n=20),
                "state_id": selected["state_id"],
                "dialogue_id_private_analysis_only": dialogue_id,
                "turn_index_private_analysis_only": selected["provenance"]["turn_index"],
                "runtime_state": selected,
                "selection_key": "minimum_protocol_hash_within_dialogue",
                "selection_read_candidate_response_quality_risk_or_judge": False,
            }
        )

    tracks = _rows(tracks_path)
    evo_users = load_evoemo(evoemo_path)
    users_by_id = {str(user["id"]): user for user in evo_users}
    topics = {
        (str(user["id"]), int(topic["idx"])): topic
        for user in evo_users
        for topic in (user.get("subsequent_topics") or [])
    }
    compiled = {
        user_id: compile_bounded_memory_with_metadata(user)[0]
        for user_id, user in users_by_id.items()
    }
    evoemo_selected: list[dict[str, Any]] = []
    for track in tracks:
        user_id = str(track["user_id"])
        topic_index = int(track["topic_index"])
        for turn_index in turn_indices:
            state = make_evo_runtime_state(
                users_by_id[user_id],
                topics[(user_id, topic_index)],
                _fixed_context_before_turn(track, turn_index),
                str(track["seeker_turns"][turn_index - 1]),
                compiled[user_id],
                turn_index,
                "pm_v1_5b_final_system_panel",
                track_id=str(track["track_id"]),
                fixed_open_loop=True,
            )
            evoemo_selected.append(
                {
                    "protocol": PROTOCOL,
                    "domain": "EvoEmo",
                    "panel_id": "evoemo_" + stable_hex(
                        PROTOCOL, track["track_id"], turn_index, n=20
                    ),
                    "track_id": track["track_id"],
                    "state_id": state.state_id,
                    "user_id_private_analysis_only": user_id,
                    "topic_index_private_analysis_only": topic_index,
                    "turn_index": turn_index,
                    "current_user_text": state.current_user_text,
                    "runtime_state": state.model_dump(mode="json"),
                    "selection_key": "all_frozen_tracks_at_preregistered_turns_3_and_8",
                    "selection_read_candidate_response_quality_risk_or_judge": False,
                }
            )

    out_dir = root / "outputs/pm_v1_5b_final_external_panel_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    esconv_out = out_dir / "esconv_panel_private.jsonl"
    evoemo_out = out_dir / "evoemo_panel_private.jsonl"
    write_jsonl(esconv_out, esconv_selected)
    write_jsonl(evoemo_out, evoemo_selected)
    checks = {
        "esconv_169_independent_test_dialogues": len(esconv_selected) == 169
        and len({row["dialogue_id_private_analysis_only"] for row in esconv_selected}) == 169,
        "esconv_exactly_one_state_per_dialogue": Counter(
            row["dialogue_id_private_analysis_only"] for row in esconv_selected
        ) == Counter({dialogue_id: 1 for dialogue_id in by_dialogue}),
        "evoemo_204_states": len(evoemo_selected) == 204,
        "evoemo_18_users": len({row["user_id_private_analysis_only"] for row in evoemo_selected}) == 18,
        "evoemo_turns_exactly_3_and_8": {row["turn_index"] for row in evoemo_selected} == {3, 8},
        "selection_is_outcome_and_candidate_blind": all(
            not row["selection_read_candidate_response_quality_risk_or_judge"]
            for row in [*esconv_selected, *evoemo_selected]
        ),
        "panel_ids_unique": len({row["panel_id"] for row in [*esconv_selected, *evoemo_selected]}) == len(esconv_selected) + len(evoemo_selected),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_FINAL_EXTERNAL_PANELS_FROZEN" if all(checks.values()) else "FAIL_EXTERNAL_PANEL_FREEZE",
        "panels": {
            "ESConv": {
                "states": len(esconv_selected),
                "independent_dialogues": len({row["dialogue_id_private_analysis_only"] for row in esconv_selected}),
                "sampling": "one outcome-blind protocol-hash state per formal test dialogue",
                "reason": "avoid weighting long dialogues as many independent observations while retaining all 169 held-out dialogue groups",
            },
            "EvoEmo": {
                "states": len(evoemo_selected),
                "independent_users": len({row["user_id_private_analysis_only"] for row in evoemo_selected}),
                "tracks": len(tracks),
                "turn_indices": list(turn_indices),
                "sampling": "all previously frozen open-loop tracks at preregistered turns",
            },
        },
        "checks": checks,
        "response_generation_calls": 0,
        "external_quality_risk_or_judge_outcome_read": False,
        "further_panel_selection_from_external_results_allowed": False,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (esconv_path, esconv_audit_path, tracks_path, evoemo_path, config_path)
        },
        "outputs": {
            str(esconv_out.relative_to(root)): sha256_file(esconv_out),
            str(evoemo_out.relative_to(root)): sha256_file(evoemo_out),
        },
    }
    write_json(out_dir / "panel_freeze_report.json", report)
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "panels": report["panels"], "checks": report["checks"]})


if __name__ == "__main__":
    main()
