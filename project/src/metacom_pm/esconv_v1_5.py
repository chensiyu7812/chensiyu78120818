from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from .contracts import (
    DialogueTurn,
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyCard,
)
from .io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from .pm_v1_5_rule_router import RULE_ROUTER_PROTOCOL, TransparentRuleRouter
from .pm_v1_5_semantic import (
    UNIFIED_SEMANTIC_QUERY_PROTOCOL,
    SemanticTextEncoder,
    prepare_visible_semantic_state,
)
from .pm_v1_5_step0 import build_strategy_family_catalog
from .pm_v2_contracts import PMV2Split, PMV2State
from .pm_v2_data import (
    OBSERVABLE_HISTORY_TURN_TARGETS,
    compare_external_observable_state_support,
    runtime_to_pmv2_state,
)
from .pm_v2_model import PMV2Model, decision_fallback_kind
from .strategy_bank import esconv_turn_states, load_esconv


ESCONV_V1_5_SPLIT_PROTOCOL = (
    "pm-v1.5-esconv-expanded1300-custom-hash-70-15-15-dialogue-level-v1"
)
ESCONV_V1_5_ADAPTER_PROTOCOL = (
    "pm-v1.5-esconv-single-session-same-checkpoint-adapter-v1"
)
ESCONV_V1_5_TURN_PROTOCOL = (
    "pm-v1.5-esconv-all-support-eligible-turns-recent-even-window-v1"
)
ESCONV_V1_5_ALLOWED_ACTIONS = ("M0+R0", "M0+RS")
ESCONV_V1_5_AUXILIARY_ADAPTER_PROTOCOL = (
    "pm-v1.5-esconv-auxiliary-bank-disjoint-seed-training-support-v1"
)
ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY = "esconv_auxiliary_strategy_routing"
# Mirrors scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py's own
# seeds[global_index % len(seeds)] assignment for the 52 synthetic users:
# split_specs iterates TRAIN, then CALIBRATION, then INTERNAL_TEST, consuming
# the seed list in that same fixed order.  This is *not* a random sample --
# it is the first 24 dialogues (by selection_ordinal) for train, next 12 for
# calibration, final 16 for internal_test.  Disclose as an outcome-free
# deterministic bank-disjoint auxiliary subset, never as a random sample.
ESCONV_V1_5_AUXILIARY_SPLIT_ORDER = (
    (PMV2Split.TRAIN, 24),
    (PMV2Split.CALIBRATION, 12),
    (PMV2Split.INTERNAL_TEST, 16),
)
EXPECTED_EXPANDED_SPLIT_COUNTS = {
    "train": 934,
    "validation": 186,
    "test": 180,
}
EXPECTED_NONOVERLAP_SPLIT_COUNTS = {
    "train": 875,
    "validation": 172,
    "test": 169,
}


def audit_esconv_v1_5_split(
    *,
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    strategy_bank_path: str | Path,
    selected_seed_sources_path: str | Path,
    require_frozen_counts: bool = True,
) -> dict[str, Any]:
    """Prove dialogue-level test isolation for the expanded ESConv release."""

    esconv = load_esconv(esconv_path)
    split_rows = [dict(row) for row in iter_jsonl(split_manifest_path)]
    if len(split_rows) != len(esconv):
        raise RuntimeError("ESConv split manifest does not cover the corpus")
    if [int(row.get("index", -1)) for row in split_rows] != list(
        range(len(esconv))
    ):
        raise RuntimeError("ESConv split manifest is incomplete or reordered")
    if any(
        str(row.get("dialogue_id")) != f"esconv_{index:04d}"
        or row.get("split") not in {"train", "validation", "test"}
        for index, row in enumerate(split_rows)
    ):
        raise RuntimeError("ESConv split row IDs or labels are invalid")

    raw_counts = Counter(str(row["split"]) for row in split_rows)
    eligible_rows = [
        row
        for row in split_rows
        if not bool(row.get("excluded_for_evoemo_overlap"))
    ]
    eligible_counts = Counter(str(row["split"]) for row in eligible_rows)
    if require_frozen_counts and (
        dict(raw_counts) != EXPECTED_EXPANDED_SPLIT_COUNTS
        or dict(eligible_counts) != EXPECTED_NONOVERLAP_SPLIT_COUNTS
    ):
        raise RuntimeError(
            "expanded ESConv split counts drifted from the frozen V1.5 design"
        )

    manifest_by_id = {str(row["dialogue_id"]): row for row in split_rows}
    test_ids = {
        str(row["dialogue_id"])
        for row in eligible_rows
        if row["split"] == "test"
    }
    validation_ids = {
        str(row["dialogue_id"])
        for row in eligible_rows
        if row["split"] == "validation"
    }
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)]
    bank_source_ids = {card.source_dialogue_id for card in cards}
    seed_ids = {
        str(row["dialogue_id"])
        for row in iter_jsonl(selected_seed_sources_path)
    }
    checks = {
        "strategy_bank_sources_are_train_only": all(
            source_id in manifest_by_id
            and manifest_by_id[source_id]["split"] == "train"
            and not bool(
                manifest_by_id[source_id]["excluded_for_evoemo_overlap"]
            )
            for source_id in bank_source_ids
        ),
        "strategy_bank_has_no_validation_or_test_sources": not bool(
            bank_source_ids & (validation_ids | test_ids)
        ),
        "development_seeds_are_train_only": all(
            seed_id in manifest_by_id
            and manifest_by_id[seed_id]["split"] == "train"
            and not bool(manifest_by_id[seed_id]["excluded_for_evoemo_overlap"])
            for seed_id in seed_ids
        ),
        "development_seeds_have_no_test_sources": not bool(seed_ids & test_ids),
        "development_seed_and_strategy_instances_are_disjoint": not bool(
            seed_ids & bank_source_ids
        ),
        "test_excludes_evoemo_overlap": all(
            not bool(row["excluded_for_evoemo_overlap"])
            for row in eligible_rows
            if row["split"] == "test"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "ESConv V1.5 data-isolation contract failed: "
            + canonical_json(checks)
        )
    return {
        "protocol": ESCONV_V1_5_SPLIT_PROTOCOL,
        "status": "PASS",
        "corpus_release": "expanded_1300_dialogues",
        "split_is_official_original_1053_release": False,
        "split_role": "frozen_custom_dialogue_level_external_partition",
        "assignment": "stable_hash(seed=13, dialogue_index)",
        "nominal_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "raw_counts": dict(sorted(raw_counts.items())),
        "nonoverlap_counts": dict(sorted(eligible_counts.items())),
        "eligible_test_dialogues": len(test_ids),
        "strategy_bank_source_dialogues": len(bank_source_ids),
        "development_seed_dialogues": len(seed_ids),
        "outcomes_used_for_split": False,
        "checks": checks,
        "esconv_sha256": sha256_file(esconv_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "selected_seed_sources_sha256": sha256_file(selected_seed_sources_path),
    }


def build_esconv_v1_5_test_artifacts(
    *,
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    strategy_bank_path: str | Path,
    selected_seed_sources_path: str | Path,
    out_dir: str | Path,
    semantic_encoder: SemanticTextEncoder,
    strategy_estimated_tokens: int,
    development_observable_state_support: dict[str, Any],
    require_frozen_split_counts: bool = True,
) -> dict[str, Any]:
    """Build ESConv states under the exact PM-v1.5 feature/Step-0 contract."""

    split_audit = audit_esconv_v1_5_split(
        esconv_path=esconv_path,
        split_manifest_path=split_manifest_path,
        strategy_bank_path=strategy_bank_path,
        selected_seed_sources_path=selected_seed_sources_path,
        require_frozen_counts=require_frozen_split_counts,
    )
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)]
    if not cards:
        raise RuntimeError("V1.5 ESConv adapter requires a non-empty Strategy Bank")
    strategy_family_catalog = build_strategy_family_catalog(
        cards,
        semantic_encoder=semantic_encoder,
        require_all_families=True,
    )
    turns = esconv_turn_states(esconv_path, split_manifest_path, "test")
    runtime_rows: list[dict[str, Any]] = []
    pm_state_rows: list[dict[str, Any]] = []
    pm_states: list[PMV2State] = []
    backend_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    raw_history_counts: Counter[int] = Counter()
    excluded_short_history_count = 0
    for turn in turns:
        raw_history_count = len(turn["history"])
        raw_history_counts[raw_history_count] += 1
        eligible_targets = [
            target
            for target in OBSERVABLE_HISTORY_TURN_TARGETS
            if target <= raw_history_count
        ]
        if not eligible_targets:
            # The PM was never trained on zero/one-history states.  Excluding
            # them by a frozen observable-only rule is safer than padding,
            # inventing dialogue, or silently asking the classifier to
            # extrapolate outside its declared support.
            excluded_short_history_count += 1
            continue
        history_turn_target = max(eligible_targets)
        state_id = "state_" + stable_hex(
            ESCONV_V1_5_ADAPTER_PROTOCOL,
            turn["dialogue_id"],
            turn["turn_index"],
            n=24,
        )
        card_id = "card_" + stable_hex(state_id, "esconv-v1.5", n=24)
        history = [
            DialogueTurn.model_validate(row)
            for row in turn["history"][-history_turn_target:]
        ]
        prepared = prepare_visible_semantic_state(
            semantic_encoder,
            current_user_text=turn["current_user_text"],
            current_session_history=history,
            current_session_summary=turn["situation"],
        )
        runtime = RuntimeState(
            state_id=state_id,
            card_id=card_id,
            user_id=turn["dialogue_id"],
            split="esconv_test",
            semantic_family="esconv_single_session_strategy_routing",
            current_user_text=turn["current_user_text"],
            current_session_history=history,
            current_session_summary=turn["situation"],
            session_index=1,
            inventory={
                source: SourceCatalog(
                    available=False,
                    count=0,
                    estimated_tokens=0,
                    catalog_fingerprint=[],
                    semantic_query_similarity=0.0,
                    semantic_representation_valid=False,
                )
                for source in MemorySource
            },
            allowed_actions=list(ESCONV_V1_5_ALLOWED_ACTIONS),
            provenance={
                "protocol": ESCONV_V1_5_ADAPTER_PROTOCOL,
                "dialogue_id": turn["dialogue_id"],
                "turn_index": int(turn["turn_index"]),
                "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
                "raw_history_turn_count": raw_history_count,
                "history_turn_target": history_turn_target,
                "current_user_duplicated_in_history": False,
                "memory_sources_structurally_unavailable": True,
                "semantic_query_protocol": UNIFIED_SEMANTIC_QUERY_PROTOCOL,
                "semantic_query_sha256": prepared.semantic_query_sha256,
                "semantic_query_vector_sha256": (
                    prepared.semantic_query_vector_sha256
                ),
            },
        )
        pm_state = runtime_to_pmv2_state(
            runtime,
            split=PMV2Split.EXTERNAL_TEST,
            strategy_catalog_count=len(cards),
            strategy_estimated_tokens=int(strategy_estimated_tokens),
            strategy_family_catalog=strategy_family_catalog,
            include_step0_observation=True,
            semantic_encoder=semantic_encoder,
        )
        if tuple(pm_state.allowed_actions) != ESCONV_V1_5_ALLOWED_ACTIONS:
            raise RuntimeError("ESConv PM-v1.5 action mask drifted")
        pm_states.append(pm_state)
        runtime_rows.append(runtime.model_dump(mode="json"))
        pm_state_rows.append(pm_state.model_dump(mode="json"))
        backend_rows.append(
            MemoryBackendRecord(card_id=card_id, items=[]).model_dump(mode="json")
        )
        audit_rows.append(
            {
                "card_id": card_id,
                "state_id": state_id,
                "dialogue_id": turn["dialogue_id"],
                "turn_index": int(turn["turn_index"]),
                "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
                "raw_history_turn_count": raw_history_count,
                "history_turn_target": history_turn_target,
                "gold_response": turn["gold_response"],
                "gold_strategy": turn["gold_strategy"],
                "evaluator_only": True,
            }
        )

    observable_support = compare_external_observable_state_support(
        development=development_observable_state_support,
        external_states=pm_states,
    )
    if observable_support["status"] != "PASS":
        raise RuntimeError("ESConv states fall outside development observable support")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runtime_states": out_dir / "runtime_states.jsonl",
        "pm_v2_states": out_dir / "pm_v2_states.jsonl",
        "memory_backend": out_dir / "memory_backend.jsonl",
        "audit_only": out_dir / "audit_only.jsonl",
        "split_audit": out_dir / "split_audit.json",
    }
    write_jsonl(paths["runtime_states"], runtime_rows)
    write_jsonl(paths["pm_v2_states"], pm_state_rows)
    write_jsonl(paths["memory_backend"], backend_rows)
    write_jsonl(paths["audit_only"], audit_rows)
    write_json(paths["split_audit"], split_audit)
    report = {
        "protocol": ESCONV_V1_5_ADAPTER_PROTOCOL,
        "status": "COMPLETE",
        "same_pm_checkpoint_required": True,
        "retraining_or_esconv_outcome_tuning_authorized": False,
        "memory_capability_claim_authorized": False,
        "legal_actions": list(ESCONV_V1_5_ALLOWED_ACTIONS),
        "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
        "turn_selection_uses_response_or_judge_outcomes": False,
        "all_support_eligible_test_turns_included": True,
        "raw_supporter_turns": len(turns),
        "excluded_history_below_two_count": excluded_short_history_count,
        "raw_history_turn_counts": dict(sorted(raw_history_counts.items())),
        "test_dialogues": len({row["dialogue_id"] for row in audit_rows}),
        "test_turns": len(pm_state_rows),
        "current_user_duplicated_in_history_count": 0,
        "summary_present_count": sum(
            bool(row["current_session_summary"].strip()) for row in runtime_rows
        ),
        "history_turn_counts": dict(
            sorted(
                Counter(
                    len(row["current_session_history"])
                    for row in runtime_rows
                ).items()
            )
        ),
        "split_audit": split_audit,
        "observable_state_support": observable_support,
        "development_observable_state_support_sha256": sha256_text(
            canonical_json(development_observable_state_support)
        ),
        "semantic_encoder_spec_sha256": semantic_encoder.binding.spec_sha256,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(out_dir / "build_report.json", report)
    return report


def esconv_v1_5_auxiliary_seed_split_assignment(
    selected_seed_sources_path: str | Path,
) -> dict[str, tuple[PMV2Split, int]]:
    """Freeze which of the 52 bank-disjoint seed dialogues maps to which PM-v2 split.

    Returns ``{dialogue_id: (split, local_index)}`` where ``local_index`` is the
    0-based position within that split's own dialogue list (e.g. train's 24
    dialogues get local_index 0..23), matching the ``pmv2_{split}_u{NNN}``
    numbering already used by the 52 synthetic users.
    """

    rows = sorted(
        (dict(row) for row in iter_jsonl(selected_seed_sources_path)),
        key=lambda row: int(row["selection_ordinal"]),
    )
    expected_total = sum(count for _, count in ESCONV_V1_5_AUXILIARY_SPLIT_ORDER)
    if len(rows) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} bank-disjoint seed sources, found {len(rows)}"
        )
    ordinals = [int(row["selection_ordinal"]) for row in rows]
    if ordinals != list(range(expected_total)):
        raise RuntimeError(
            "seed source selection_ordinal is not a dense 0.." + str(expected_total - 1)
        )
    if any(bool(row.get("excluded_for_evoemo_overlap")) for row in rows):
        raise RuntimeError(
            "a bank-disjoint seed source is marked excluded_for_evoemo_overlap"
        )
    dialogue_ids = [str(row["dialogue_id"]) for row in rows]
    if len(set(dialogue_ids)) != expected_total:
        raise RuntimeError("bank-disjoint seed sources contain duplicate dialogue IDs")
    assignment: dict[str, tuple[PMV2Split, int]] = {}
    cursor = 0
    for split, count in ESCONV_V1_5_AUXILIARY_SPLIT_ORDER:
        for local_index in range(count):
            dialogue_id = dialogue_ids[cursor]
            assignment[dialogue_id] = (split, local_index)
            cursor += 1
    return assignment


def build_esconv_v1_5_auxiliary_training_artifacts(
    *,
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    strategy_bank_path: str | Path,
    selected_seed_sources_path: str | Path,
    out_dir: str | Path,
    semantic_encoder: SemanticTextEncoder,
    strategy_estimated_tokens: int,
    require_frozen_split_counts: bool = True,
) -> dict[str, Any]:
    """Build ESConv "no cross-session memory" auxiliary training/calibration/
    internal-test states from the 52 dialogues already used to seed the 52
    synthetic development users -- and only those 52.

    These 52 dialogues are, by construction (see
    ``require_instance_disjoint_seed_strategy_sources`` in
    scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py), instance-disjoint
    from the Strategy Bank's 823 source dialogues: generating and retrieving
    against these states can never surface a strategy card derived from the
    state's own dialogue. The held-out 169-dialogue ESConv *test* split
    (data/esconv_test_v1_5/) is untouched by this function and never read here.

    Gold response/gold strategy text is written only to a separate
    evaluator-only audit file, exactly like the test builder; the generator,
    judges, and PM-facing state/runtime rows never see it.
    """

    split_audit = audit_esconv_v1_5_split(
        esconv_path=esconv_path,
        split_manifest_path=split_manifest_path,
        strategy_bank_path=strategy_bank_path,
        selected_seed_sources_path=selected_seed_sources_path,
        require_frozen_counts=require_frozen_split_counts,
    )
    seed_assignment = esconv_v1_5_auxiliary_seed_split_assignment(
        selected_seed_sources_path
    )
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)]
    if not cards:
        raise RuntimeError("V1.5 ESConv adapter requires a non-empty Strategy Bank")
    strategy_family_catalog = build_strategy_family_catalog(
        cards,
        semantic_encoder=semantic_encoder,
        require_all_families=True,
    )
    # The 52 seeds are train-split ESConv sources (source_split: "train" in
    # pm_v1_5_selected_seed_sources.jsonl); they are never drawn from
    # validation or test.
    turns = esconv_turn_states(esconv_path, split_manifest_path, "train")
    turns = [turn for turn in turns if turn["dialogue_id"] in seed_assignment]
    rows_by_split: dict[PMV2Split, dict[str, list[dict[str, Any]]]] = {
        split: {
            "runtime_states": [],
            "pm_v2_states": [],
            "memory_backend": [],
            "audit_only": [],
        }
        for split, _ in ESCONV_V1_5_AUXILIARY_SPLIT_ORDER
    }
    raw_history_counts: Counter[int] = Counter()
    excluded_short_history_count = 0
    dialogues_touched_by_split: dict[PMV2Split, set[str]] = {
        split: set() for split, _ in ESCONV_V1_5_AUXILIARY_SPLIT_ORDER
    }
    for turn in turns:
        split, _local_index = seed_assignment[turn["dialogue_id"]]
        raw_history_count = len(turn["history"])
        raw_history_counts[raw_history_count] += 1
        eligible_targets = [
            target
            for target in OBSERVABLE_HISTORY_TURN_TARGETS
            if target <= raw_history_count
        ]
        if not eligible_targets:
            excluded_short_history_count += 1
            continue
        history_turn_target = max(eligible_targets)
        state_id = "state_" + stable_hex(
            ESCONV_V1_5_AUXILIARY_ADAPTER_PROTOCOL,
            turn["dialogue_id"],
            turn["turn_index"],
            n=24,
        )
        card_id = "card_" + stable_hex(state_id, "esconv-v1.5-auxiliary", n=24)
        history = [
            DialogueTurn.model_validate(row)
            for row in turn["history"][-history_turn_target:]
        ]
        prepared = prepare_visible_semantic_state(
            semantic_encoder,
            current_user_text=turn["current_user_text"],
            current_session_history=history,
            current_session_summary=turn["situation"],
        )
        runtime = RuntimeState(
            state_id=state_id,
            card_id=card_id,
            user_id=turn["dialogue_id"],
            split="esconv_auxiliary_" + split.value,
            semantic_family=ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY,
            current_user_text=turn["current_user_text"],
            current_session_history=history,
            current_session_summary=turn["situation"],
            session_index=1,
            inventory={
                source: SourceCatalog(
                    available=False,
                    count=0,
                    estimated_tokens=0,
                    catalog_fingerprint=[],
                    semantic_query_similarity=0.0,
                    semantic_representation_valid=False,
                )
                for source in MemorySource
            },
            allowed_actions=list(ESCONV_V1_5_ALLOWED_ACTIONS),
            provenance={
                "protocol": ESCONV_V1_5_AUXILIARY_ADAPTER_PROTOCOL,
                "dialogue_id": turn["dialogue_id"],
                "turn_index": int(turn["turn_index"]),
                "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
                "raw_history_turn_count": raw_history_count,
                "history_turn_target": history_turn_target,
                "current_user_duplicated_in_history": False,
                "memory_sources_structurally_unavailable": True,
                "semantic_query_protocol": UNIFIED_SEMANTIC_QUERY_PROTOCOL,
                "semantic_query_sha256": prepared.semantic_query_sha256,
                "semantic_query_vector_sha256": (
                    prepared.semantic_query_vector_sha256
                ),
                "bank_disjoint_auxiliary_seed": True,
            },
        )
        pm_state = runtime_to_pmv2_state(
            runtime,
            split=split,
            strategy_catalog_count=len(cards),
            strategy_estimated_tokens=int(strategy_estimated_tokens),
            strategy_family_catalog=strategy_family_catalog,
            include_step0_observation=True,
            semantic_encoder=semantic_encoder,
        )
        if tuple(pm_state.allowed_actions) != ESCONV_V1_5_ALLOWED_ACTIONS:
            raise RuntimeError("ESConv auxiliary PM-v1.5 action mask drifted")
        dialogues_touched_by_split[split].add(turn["dialogue_id"])
        bucket = rows_by_split[split]
        bucket["runtime_states"].append(runtime.model_dump(mode="json"))
        bucket["pm_v2_states"].append(pm_state.model_dump(mode="json"))
        bucket["memory_backend"].append(
            MemoryBackendRecord(card_id=card_id, items=[]).model_dump(mode="json")
        )
        bucket["audit_only"].append(
            {
                "card_id": card_id,
                "state_id": state_id,
                "dialogue_id": turn["dialogue_id"],
                "turn_index": int(turn["turn_index"]),
                "split": split.value,
                "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
                "raw_history_turn_count": raw_history_count,
                "history_turn_target": history_turn_target,
                "gold_response": turn["gold_response"],
                "gold_strategy": turn["gold_strategy"],
                "evaluator_only": True,
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for split, _ in ESCONV_V1_5_AUXILIARY_SPLIT_ORDER:
        bucket = rows_by_split[split]
        split_dir = out_dir / split.value
        split_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "runtime_states": split_dir / "runtime_states.jsonl",
            "pm_v2_states": split_dir / "pm_v2_states.jsonl",
            "memory_backend": split_dir / "memory_backend.jsonl",
            "audit_only": split_dir / "audit_only.jsonl",
        }
        write_jsonl(paths["runtime_states"], bucket["runtime_states"])
        write_jsonl(paths["pm_v2_states"], bucket["pm_v2_states"])
        write_jsonl(paths["memory_backend"], bucket["memory_backend"])
        write_jsonl(paths["audit_only"], bucket["audit_only"])
        outputs[split.value] = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        }
        split_reports[split.value] = {
            "dialogue_count": len(dialogues_touched_by_split[split]),
            "state_count": len(bucket["pm_v2_states"]),
        }
    report = {
        "protocol": ESCONV_V1_5_AUXILIARY_ADAPTER_PROTOCOL,
        "status": "COMPLETE",
        "semantic_family": ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY,
        "legal_actions": list(ESCONV_V1_5_ALLOWED_ACTIONS),
        "turn_selection_protocol": ESCONV_V1_5_TURN_PROTOCOL,
        "turn_selection_uses_response_or_judge_outcomes": False,
        "sample_selection_rule": "first_eligible_train_sources_in_manifest_order",
        "sample_selection_is_outcome_free_deterministic_not_random": True,
        "bank_disjoint": True,
        "strategy_bank_source_dialogue_overlap_count": len(
            {str(row["source_dialogue_id"]) for row in iter_jsonl(strategy_bank_path)}
            & set(seed_assignment)
        ),
        "seed_dialogue_count": len(seed_assignment),
        "excluded_history_below_two_count": excluded_short_history_count,
        "raw_history_turn_counts": dict(sorted(raw_history_counts.items())),
        "split_reports": split_reports,
        "split_audit": split_audit,
        "semantic_encoder_spec_sha256": semantic_encoder.binding.spec_sha256,
        "outputs": outputs,
    }
    write_json(out_dir / "build_report.json", report)
    return report


def choose_esconv_v1_5_policies(
    *,
    pm_v2_states_path: str | Path,
    learned_checkpoint_path: str | Path,
    transparent_rule_checkpoint_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    """Run one frozen learned PM and its frozen transparent-rule comparator."""

    states = [
        PMV2State.model_validate_json(line)
        for line in Path(pm_v2_states_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    learned = PMV2Model.load(learned_checkpoint_path)
    rule = joblib.load(transparent_rule_checkpoint_path)
    if not isinstance(rule, TransparentRuleRouter) or rule.format_version != RULE_ROUTER_PROTOCOL:
        raise RuntimeError("transparent-rule checkpoint is stale")
    rows: list[dict[str, Any]] = []
    for state in states:
        if tuple(state.allowed_actions) != ESCONV_V1_5_ALLOWED_ACTIONS:
            raise RuntimeError("ESConv policy input exposes a non-Strategy action")
        learned_decision = learned.choose(state)
        rule_decision = rule.choose(state)
        rows.append(
            {
                "state_id": state.state_id,
                "card_id": state.card_id,
                "dialogue_id": state.user_id,
                "learned_action": learned_decision.chosen_action,
                "learned_fallback_type": decision_fallback_kind(
                    learned_decision
                ),
                "learned_semantic_ood_score": (
                    learned_decision.semantic_ood_score
                ),
                "learned_metadata_ood_score": (
                    learned_decision.metadata_ood_score
                ),
                "transparent_rule_action": rule_decision.chosen_action,
                "always_r0_action": "M0+R0",
                "always_rs_action": "M0+RS",
                "esconv_outcome_used_for_choice": False,
            }
        )
    write_jsonl(out_path, rows)
    fallback_counts = Counter(
        str(row["learned_fallback_type"] or "none") for row in rows
    )
    learned_actions = Counter(str(row["learned_action"]) for row in rows)
    rule_actions = Counter(str(row["transparent_rule_action"]) for row in rows)
    return {
        "protocol": ESCONV_V1_5_ADAPTER_PROTOCOL,
        "status": "COMPLETE",
        "state_count": len(rows),
        "same_frozen_checkpoint": True,
        "esconv_train_validation_or_test_outcomes_used_for_choice": False,
        "learned_checkpoint_sha256": sha256_file(learned_checkpoint_path),
        "transparent_rule_checkpoint_sha256": sha256_file(
            transparent_rule_checkpoint_path
        ),
        "pm_v2_states_sha256": sha256_file(pm_v2_states_path),
        "policy_choices_sha256": sha256_file(out_path),
        "learned_action_distribution": dict(sorted(learned_actions.items())),
        "transparent_rule_action_distribution": dict(sorted(rule_actions.items())),
        "learned_fallback_distribution": dict(sorted(fallback_counts.items())),
    }
