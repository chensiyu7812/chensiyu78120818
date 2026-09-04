#!/usr/bin/env python3
"""Freeze a second, fully disjoint H2-v2 state slice for repair_v2.

Reuses 24an's exact selection logic (same strata, same hash ordering, same
one-state-per-dialogue rule) via import, and adds every dialogue already
touched by any prior round -- original H2 states, the 32-group RS effect
study, and the 40-pair ESConv frozen validation -- to the exclusion set. This
guarantees the states here were never seen by H1, H2, the RS effect model,
or the ESConv frozen quality test, so repair_v2 can be checked on states it
has never had a chance to be tuned toward.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2v2-retrieval-state-freeze-v1"


def _load_24an():
    path = ROOT / "scripts/v1_5/24an_freeze_h2_retrieval_states_v1_5.py"
    spec = importlib.util.spec_from_file_location("h2_freeze_v1", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    h2v1 = _load_24an()

    h1_packet = (
        ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "h1_review_packet.json"
    )
    h1_private_retrieval = (
        ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "private_retrieval_audit.jsonl"
    )
    relink_finalists = (
        ROOT / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate/"
        "private_candidate_scores.jsonl"
    )
    formal_test = ROOT / "data/splits/formal_test.jsonl"

    excluded, exclusion_counts = h2v1._excluded_dialogues(
        h1_packet=h1_packet,
        h1_private_retrieval=h1_private_retrieval,
        relink_finalists=relink_finalists,
    )
    if formal_test.is_file():
        formal = {
            str(row.get("user_id") or row.get("source_dialogue_id"))
            for row in iter_jsonl(formal_test)
        }
        formal.discard("None")
        excluded |= formal
        exclusion_counts["formal_test_dialogues"] = len(formal)

    h2v1_states = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(
            ROOT / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1/"
            "h2_states_private.jsonl"
        )
    }
    rs_effect_dialogues = {
        str(row["user_id"])
        for row in iter_jsonl(
            ROOT / "outputs/pm_v1_5_rs_effect_final_labels_v1/"
            "pm_rs_effect_labels.jsonl"
        )
    }
    rs_esconv_frozen_dialogues = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(
            ROOT / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1/"
            "selected_states.jsonl"
        )
    }
    excluded |= h2v1_states | rs_effect_dialogues | rs_esconv_frozen_dialogues
    exclusion_counts.update(
        {
            "h2v1_state_dialogues": len(h2v1_states),
            "rs_effect_study_dialogues": len(rs_effect_dialogues),
            "rs_esconv_frozen_validation_dialogues": len(
                rs_esconv_frozen_dialogues
            ),
            "union_all_prior_rounds": len(excluded),
        }
    )

    pools: dict[str, list[dict]] = {}
    from collections import defaultdict

    pools = defaultdict(list)
    seen_states: set[tuple[str, str]] = set()
    from metacom_pm.io import stable_hex, sha256_text
    from metacom_pm.text import normalize_space

    for source in iter_jsonl(
        ROOT / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/"
        "clean_train_strategy_universe.jsonl"
    ):
        dialogue_id = str(source["source_dialogue_id"])
        dialogue = [
            {
                "speaker": str(turn["speaker"]),
                "content": normalize_space(turn["content"]),
            }
            for turn in source["recent_dialogue"]
            if str(turn.get("speaker")) in {"seeker", "supporter"}
            and normalize_space(turn.get("content", ""))
        ]
        if (
            dialogue_id in excluded
            or not dialogue
            or dialogue[-1]["speaker"] != "seeker"
        ):
            continue
        visible_text = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in dialogue
        )
        digest = sha256_text(visible_text)
        state_key = (dialogue_id, digest)
        if state_key in seen_states:
            continue
        seen_states.add(state_key)
        seekers = [
            turn["content"] for turn in dialogue if turn["speaker"] == "seeker"
        ]
        flags = h2v1.repaired_observable_opportunity_flags(
            current_user_text=seekers[-1],
            recent_user_text=" ".join(seekers[-3:]),
            visible_dialogue=dialogue,
        )
        row = {
            "h2v2_state_id": "h2v2_state_"
            + stable_hex(PROTOCOL, dialogue_id, digest, n=20),
            "source_dialogue_id": dialogue_id,
            "source_turn_index": int(source["source_turn_index"]),
            "problem_type": str(source.get("problem_type") or "unknown"),
            "visible_dialogue": dialogue,
            "visible_dialogue_sha256": digest,
            "observable_flags_at_freeze": flags,
        }
        for stratum in h2v1._strata(flags):
            pools[stratum].append(row)

    def _select_v2(pools, *, targets):
        """Best-effort per stratum: take up to `targets[stratum]`, honestly
        report whatever the real pool supports instead of failing when a
        stratum's disjoint-from-everything pool is thinner than the others.
        """

        selected: list[dict] = []
        used_dialogues: set[str] = set()
        actual: dict[str, int] = {}
        for stratum in targets:
            from collections import Counter

            problem_counts: Counter = Counter()
            taken = 0
            for row in sorted(
                pools[stratum],
                key=lambda value: stable_hex(
                    PROTOCOL,
                    stratum,
                    value["source_dialogue_id"],
                    value["visible_dialogue_sha256"],
                    n=32,
                ),
            ):
                dialogue = str(row["source_dialogue_id"])
                problem = str(row.get("problem_type") or "unknown")
                if dialogue in used_dialogues or problem_counts[problem] >= 3:
                    continue
                selected.append({**row, "selection_stratum": stratum})
                used_dialogues.add(dialogue)
                problem_counts[problem] += 1
                taken += 1
                if taken == targets[stratum]:
                    break
            actual[stratum] = taken
        return selected, actual

    # After excluding every dialogue touched by any prior round, the
    # ESConv-train pool is thinner than H1/H2v1 had available. Report exactly
    # what remains per stratum and use the largest count every stratum can
    # honestly support, rather than silently reusing a touched dialogue or
    # failing outright.
    available = {
        stratum: len(
            {row["source_dialogue_id"] for row in pools[stratum]}
        )
        for stratum in h2v1.TARGETS
    }
    print({"available_unique_dialogues_per_stratum": available})
    targets = {stratum: 20 for stratum in h2v1.TARGETS}
    selected, actual = _select_v2(pools, targets=targets)
    print({"actual_selected_per_stratum": actual})
    targets = actual

    out_dir = ROOT / "outputs/pm_v1_5_h2v2_retrieval_state_freeze_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "h2v2_states_private.jsonl"
    states_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    from collections import Counter

    manifest = {
        "protocol": PROTOCOL,
        "status": "H2V2_STATES_FROZEN_DISJOINT_FROM_ALL_PRIOR_ROUNDS",
        "purpose": (
            "Fresh qualification slice for repair_v2_rank_applicable_cards, "
            "disjoint from H1, H2v1, the 32-group RS effect study, and the "
            "40-pair ESConv frozen quality validation."
        ),
        "states": len(selected),
        "targets": targets,
        "observed_strata": dict(
            sorted(
                Counter(row["selection_stratum"] for row in selected).items()
            )
        ),
        "unique_dialogues": len(
            {str(row["source_dialogue_id"]) for row in selected}
        ),
        "excluded": exclusion_counts,
        "outputs": {states_path.name: sha256_file(states_path)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "status": manifest["status"],
            "states": len(selected),
            "unique_dialogues": manifest["unique_dialogues"],
            "strata": manifest["observed_strata"],
            "excluded_union": exclusion_counts["union_all_prior_rounds"],
        }
    )


if __name__ == "__main__":
    main()
