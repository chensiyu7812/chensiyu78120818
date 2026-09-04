from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

from .contracts import StrategyCard
from .io import canonical_json, sha256_text, stable_hex, write_jsonl, write_json, sha256_file
from .text import dialogue_text, jaccard, normalize_for_hash, normalize_space, shingle_set


def stable_dialogue_split(dialogue_index: int, seed: int = 13) -> str:
    # Frozen custom split for the expanded 1,300-dialogue release.  It is not
    # presented as the official split of the original 1,053-dialogue corpus.
    # Hashing the dialogue index keeps complete dialogues together and makes
    # the assignment reproducible before any turn-level outcome is inspected.
    value = int(stable_hex("esconv_split", seed, dialogue_index, n=12), 16) / float(16**12)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def load_esconv(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("ESConv must be a non-empty JSON list")
    for idx, row in enumerate(data):
        if not isinstance(row, dict) or not isinstance(row.get("dialog"), list):
            raise ValueError(f"invalid ESConv dialogue at index {idx}")
    return data


def evoemo_dialogue_texts(evoemo_path: str | Path) -> list[str]:
    data = json.loads(Path(evoemo_path).read_text(encoding="utf-8"))
    texts: list[str] = []
    for user in data:
        for session in user.get("dialog_history") or []:
            dialogue = session.get("dialogue") or []
            texts.append(dialogue_text([
                {"speaker": turn.get("role"), "content": turn.get("content")}
                for turn in dialogue
            ]))
    return texts


def find_esconv_evoemo_overlaps(
    esconv: list[dict[str, Any]],
    evoemo_path: str | Path,
    *,
    jaccard_threshold: float = 0.88,
    include_deterministic_source_ids: bool = False,
) -> dict[int, dict[str, Any]]:
    evo_texts = evoemo_dialogue_texts(evoemo_path)
    exact = {normalize_for_hash(text) for text in evo_texts if text}
    evo_shingles = [shingle_set(text) for text in evo_texts if text]
    overlaps: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(esconv):
        text = dialogue_text(row["dialog"])
        normalized = normalize_for_hash(text)
        if normalized in exact:
            overlaps[idx] = {"reason": "exact", "similarity": 1.0}
            continue
        shingles = shingle_set(text)
        if not shingles:
            continue
        best = max((jaccard(shingles, other) for other in evo_shingles), default=0.0)
        if best >= jaccard_threshold:
            overlaps[idx] = {"reason": "shingle_jaccard", "similarity": best}
    if include_deterministic_source_ids:
        # The whole-dialogue Jaccard check can miss a real source match when
        # unrelated portions of the dialogue were edited enough to drag the
        # overall similarity below threshold, even though large verbatim
        # spans remain. Some EvoEmo sessions are named `escN`, directly
        # encoding their ESConv source index; that mapping is a deterministic
        # ground truth for those sessions and should be unioned in regardless
        # of what the fuzzy check found (or missed).
        for idx, reason in find_deterministic_esconv_evoemo_overlaps(evoemo_path).items():
            overlaps.setdefault(idx, reason)
    return overlaps


def find_deterministic_esconv_evoemo_overlaps(
    evoemo_path: str | Path,
) -> dict[int, dict[str, Any]]:
    """ESConv indices directly named by an EvoEmo session id (`escN`).

    This is a small, known subset of EvoEmo sessions whose id literally
    encodes their ESConv source dialogue index. It is a deterministic ground
    truth for those specific sessions, independent of and more reliable than
    the fuzzy whole-dialogue Jaccard check for exactly this subset -- it does
    not replace that check for the remaining sessions, which use unrelated
    naming and have no equivalent deterministic signal.
    """

    data = json.loads(Path(evoemo_path).read_text(encoding="utf-8"))
    overlaps: dict[int, dict[str, Any]] = {}
    for user in data:
        for session in user.get("dialog_history") or []:
            session_id = str(session.get("id") or "")
            match = re.match(r"^esc(\d+)$", session_id)
            if match:
                overlaps[int(match.group(1))] = {
                    "reason": "deterministic_session_id",
                    "evoemo_session_id": session_id,
                    "similarity": 1.0,
                }
    return overlaps


def _preceding_context(dialogue: list[dict[str, Any]], turn_index: int, max_turns: int = 6) -> str:
    begin = max(0, turn_index - max_turns)
    rows = dialogue[begin:turn_index]
    return "\n".join(
        f"{turn.get('speaker')}: {normalize_space(turn.get('content') or '')}"
        for turn in rows
    )


def build_strategy_bank(
    esconv_path: str | Path,
    evoemo_path: str | Path,
    out_bank_path: str | Path,
    out_split_path: str | Path,
    out_overlap_path: str | Path,
    *,
    seed: int = 13,
    jaccard_threshold: float = 0.88,
    include_deterministic_source_ids: bool = False,
    excluded_development_seed_ids: Iterable[str] = (),
) -> dict[str, Any]:
    esconv = load_esconv(esconv_path)
    overlaps = find_esconv_evoemo_overlaps(
        esconv,
        evoemo_path,
        jaccard_threshold=jaccard_threshold,
        include_deterministic_source_ids=include_deterministic_source_ids,
    )
    development_seed_ids = {str(value) for value in excluded_development_seed_ids}
    if "" in development_seed_ids:
        raise ValueError("development seed exclusions cannot contain an empty ID")
    cards: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(esconv):
        split = stable_dialogue_split(idx, seed)
        dialogue_id = f"esconv_{idx:04d}"
        split_rows.append({
            "dialogue_id": dialogue_id,
            "index": idx,
            "split": split,
            "excluded_for_evoemo_overlap": idx in overlaps,
        })
        if split != "train" or idx in overlaps or dialogue_id in development_seed_ids:
            continue
        dialogue = row["dialog"]
        for turn_index, turn in enumerate(dialogue):
            if turn.get("speaker") != "supporter":
                continue
            annotation = turn.get("annotation") or {}
            strategy = normalize_space(annotation.get("strategy") or "Others")
            response = normalize_space(turn.get("content") or "")
            if not response:
                continue
            context = _preceding_context(dialogue, turn_index)
            situation = normalize_space(row.get("situation") or "")
            retrieval_text = "\n".join(x for x in [situation, context] if x)
            guidance = (
                f"Use the emotional-support strategy '{strategy}' when it fits the "
                "visible dialogue. Adapt it naturally; do not copy private facts or "
                "assume information that the user has not disclosed."
            )
            card = StrategyCard(
                strategy_id=f"strat_{stable_hex(dialogue_id, turn_index, strategy, response, n=20)}",
                strategy_label=strategy,
                retrieval_text=retrieval_text or situation or response,
                guidance_text=guidance,
                example_response=response,
                source_dialogue_id=dialogue_id,
                source_turn_index=turn_index,
            )
            cards.append(card.model_dump(mode="json"))

    # Remove exact duplicates without using validation/test content.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        key = (
            normalize_for_hash(card["retrieval_text"]),
            normalize_for_hash(card["example_response"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)

    write_jsonl(out_bank_path, deduped)
    write_jsonl(out_split_path, split_rows)
    source_ids = {str(card["source_dialogue_id"]) for card in deduped}
    family_counts: dict[str, int] = defaultdict(int)
    for card in deduped:
        family_counts[str(card["strategy_label"])] += 1
    write_json(out_overlap_path, {
        "esconv_sha256": sha256_file(esconv_path),
        "evoemo_sha256": sha256_file(evoemo_path),
        "strategy_bank_sha256": sha256_file(out_bank_path),
        "split_manifest_sha256": sha256_file(out_split_path),
        "threshold": jaccard_threshold,
        "deterministic_source_id_rule_enabled": bool(
            include_deterministic_source_ids
        ),
        "n_esconv_dialogues": len(esconv),
        "n_excluded_overlap": len(overlaps),
        "overlaps": {f"esconv_{k:04d}": v for k, v in sorted(overlaps.items())},
        "development_seed_exclusion": {
            "protocol": "pm-v1.5-dialogue-instance-disjoint-strategy-bank-v1",
            "excluded_source_ids": sorted(development_seed_ids),
            "excluded_source_ids_sha256": sha256_text(
                canonical_json(sorted(development_seed_ids))
            ),
            "n_excluded_sources": len(development_seed_ids),
            "bank_source_intersection": sorted(development_seed_ids & source_ids),
        },
        "n_strategy_source_dialogues": len(source_ids),
        "strategy_family_card_counts": dict(sorted(family_counts.items())),
        "n_strategy_cards": len(deduped),
    })
    return {
        "n_esconv_dialogues": len(esconv),
        "n_strategy_cards": len(deduped),
        "n_excluded_overlap": len(overlaps),
        "n_excluded_development_seed_sources": len(development_seed_ids),
        "n_strategy_source_dialogues": len(source_ids),
    }


def esconv_turn_states(
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    esconv = load_esconv(esconv_path)
    split_rows = {
        int(row["index"]): row
        for row in __import__("metacom_pm.io", fromlist=["iter_jsonl"]).iter_jsonl(split_manifest_path)
    }
    states: list[dict[str, Any]] = []
    for idx, dialogue_row in enumerate(esconv):
        meta = split_rows[idx]
        if meta["split"] != split or meta["excluded_for_evoemo_overlap"]:
            continue
        dialogue = dialogue_row["dialog"]
        for turn_index, turn in enumerate(dialogue):
            if turn.get("speaker") != "supporter":
                continue
            # Require at least one visible seeker message.
            previous = dialogue[:turn_index]
            current_user_index = next(
                (
                    index
                    for index in range(len(previous) - 1, -1, -1)
                    if previous[index].get("speaker") == "seeker"
                ),
                None,
            )
            if current_user_index is None:
                continue
            current_user = normalize_space(
                previous[current_user_index].get("content") or ""
            )
            if not current_user:
                continue
            # The final seeker utterance is represented once as current_user_text,
            # never duplicated inside history.  This matches the EvoEmo runtime
            # and the formal visible-state semantic query contract.
            prior_history = previous[:current_user_index]
            history = [
                {
                    "role": "user" if x.get("speaker") == "seeker" else "assistant",
                    "content": normalize_space(x.get("content") or ""),
                }
                for x in prior_history[-8:]
                if normalize_space(x.get("content") or "")
            ]
            states.append({
                "dialogue_id": f"esconv_{idx:04d}",
                "turn_index": turn_index,
                "current_user_text": current_user,
                "history": history,
                "gold_response": normalize_space(turn.get("content") or ""),
                "gold_strategy": normalize_space(
                    (turn.get("annotation") or {}).get("strategy") or "Others"
                ),
                "situation": normalize_space(dialogue_row.get("situation") or ""),
            })
    return states
