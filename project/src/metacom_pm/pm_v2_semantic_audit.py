from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import create_artifact_attestation, require_artifact_attestation
from .contracts import MemoryBackendRecord, RuntimeState
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from .pm_v2_contracts import PMV2Split, PMV2State, ResourceNeedRegime
from .pm_v2_data import (
    EvaluatorContextIndex,
    REGIME_INSTRUCTIONS,
    load_states,
    state_to_v1_runtime,
)


SEMANTIC_AUDIT_PROTOCOL = "pm-v2-semantic-sanity-audit-v3-item-utility"
SEMANTIC_AUDIT_PLAN_VERSION = "pm-v2-semantic-sanity-plan-v3-item-utility"
SEMANTIC_AUDIT_MANUAL_VERSION = "pm-v2-semantic-sanity-manual-v3-item-utility"
SEMANTIC_AUDIT_ATTESTATION_STAGE = "pm_v2_semantic_sanity_audit"

AUDIT_FIELDS = (
    "semantic_family_match",
    "regime_match",
    "needed_memory_sources_match",
    "memory_item_utility_match",
)
PROTECTED_PACKET_FIELDS = (
    "item_id",
    "current_user_text",
    "recent_dialogue_json",
    "current_session_summary",
    "authorized_user_context",
    "source_inventory_json",
    "source_memory_items_json",
    "candidate_semantic_family",
    "candidate_regime",
    "candidate_needed_memory_sources_json",
)
PACKET_FIELDS = (
    *PROTECTED_PACKET_FIELDS,
    *AUDIT_FIELDS,
    "annotator_id",
    "notes",
)
AUDITED_SPLITS = (
    PMV2Split.TRAIN,
    PMV2Split.CALIBRATION,
    PMV2Split.INTERNAL_TEST,
)

_CONFIG_KEYS = {
    "required_before_development_api",
    "sample_seed",
    "states_per_regime_per_split",
    "minimum_states_per_family_per_split",
    "minimum_annotators",
    "minimum_affirmative_rate_per_field",
    "minimum_pairwise_agreement_per_field",
    "minimum_affirmative_rate_per_regime",
    "minimum_affirmative_rate_per_split",
    "minimum_affirmative_rate_per_family",
}


def semantic_audit_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("semantic_sanity_audit")
    if not isinstance(raw, Mapping):
        raise RuntimeError("PM-v2 config lacks semantic_sanity_audit")
    if set(raw) != _CONFIG_KEYS:
        raise RuntimeError(
            "semantic_sanity_audit must use the frozen schema: "
            f"missing={sorted(_CONFIG_KEYS-set(raw))}, "
            f"extra={sorted(set(raw)-_CONFIG_KEYS)}"
        )
    if raw["required_before_development_api"] is not True:
        raise RuntimeError(
            "semantic_sanity_audit.required_before_development_api must be the "
            "boolean true"
        )
    for name in (
        "sample_seed",
        "states_per_regime_per_split",
        "minimum_states_per_family_per_split",
        "minimum_annotators",
    ):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int):
            raise TypeError(f"semantic audit setting {name} must be an integer")
    for name in (
        "minimum_affirmative_rate_per_field",
        "minimum_pairwise_agreement_per_field",
        "minimum_affirmative_rate_per_regime",
        "minimum_affirmative_rate_per_split",
        "minimum_affirmative_rate_per_family",
    ):
        if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
            raise TypeError(f"semantic audit setting {name} must be numeric")
    settings = {
        "required_before_development_api": True,
        "sample_seed": int(raw["sample_seed"]),
        "states_per_regime_per_split": int(
            raw["states_per_regime_per_split"]
        ),
        "minimum_states_per_family_per_split": int(
            raw["minimum_states_per_family_per_split"]
        ),
        "minimum_annotators": int(raw["minimum_annotators"]),
        "minimum_affirmative_rate_per_field": float(
            raw["minimum_affirmative_rate_per_field"]
        ),
        "minimum_pairwise_agreement_per_field": float(
            raw["minimum_pairwise_agreement_per_field"]
        ),
        "minimum_affirmative_rate_per_regime": float(
            raw["minimum_affirmative_rate_per_regime"]
        ),
        "minimum_affirmative_rate_per_split": float(
            raw["minimum_affirmative_rate_per_split"]
        ),
        "minimum_affirmative_rate_per_family": float(
            raw["minimum_affirmative_rate_per_family"]
        ),
    }
    if settings["sample_seed"] < 0:
        raise ValueError("semantic audit sample_seed must be non-negative")
    if settings["states_per_regime_per_split"] < 1:
        raise ValueError("semantic audit cell size must be positive")
    if settings["minimum_states_per_family_per_split"] < 1:
        raise ValueError("semantic audit family coverage must be positive")
    if settings["minimum_annotators"] < 2:
        raise ValueError("semantic audit requires at least two annotators")
    for name in (
        "minimum_affirmative_rate_per_field",
        "minimum_pairwise_agreement_per_field",
        "minimum_affirmative_rate_per_regime",
        "minimum_affirmative_rate_per_split",
        "minimum_affirmative_rate_per_family",
    ):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"semantic audit threshold {name} must be in [0, 1]")
    return settings


def semantic_audit_manual(settings: Mapping[str, Any]) -> str:
    lines = [
        "# PM-v2 Semantic Sanity Audit Manual",
        "",
        f"Version: `{SEMANTIC_AUDIT_MANUAL_VERSION}`",
        "",
        "This is a no-API data audit. Judge whether each displayed synthetic state "
        "supports the three candidate annotations. Do not judge response quality, "
        "choose a PM action, or infer the hidden split, user, state ID, or generator.",
        "",
        "Work independently. Enter `1` only when the displayed evidence supports the "
        "candidate annotation, and `0` when it is wrong, incoherent, or the evidence "
        "is insufficient. Fill every field, add a non-empty `annotator_id`, and do not "
        "change any other CSV cell.",
        "",
        "## Required fields",
        "",
        "- `semantic_family_match`: the current situation primarily matches the "
        "candidate topic family (underscores are word separators).",
        "- `regime_match`: the current turn, dialogue, authorized context, source "
        "summaries, and stale/conflict annotations jointly support the candidate "
        "resource-need regime.",
        "- `needed_memory_sources_match`: the exact displayed MP/MS/ME set is the "
        "minimal materially useful long-term context. `[]` means no long-term memory "
        "is needed; do not award a match merely because a source is available.",
        "- `memory_item_utility_match`: every displayed memory item's `helpful`, "
        "`irrelevant`, or `harmful` label is correct for this exact current state. "
        "A source-level need does not make every item from that source helpful.",
        "",
        "Source meanings: `MP` is stable profile/preference/boundary memory; `MS` is "
        "a cross-session summary or pattern; `ME` is a concrete past event.",
        "",
        "## Regime definitions",
        "",
    ]
    for regime in ResourceNeedRegime:
        lines.append(f"- `{regime.value}`: {REGIME_INSTRUCTIONS[regime]}")
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            f"Every sampled item needs at least {int(settings['minimum_annotators'])} "
            "independent annotations. The preregistered minimum affirmative rate per "
            f"field is {float(settings['minimum_affirmative_rate_per_field']):.2f}; "
            "minimum pairwise agreement per field is "
            f"{float(settings['minimum_pairwise_agreement_per_field']):.2f}; minimum "
            "affirmative rates per regime, split, and semantic family are respectively "
            f"{float(settings['minimum_affirmative_rate_per_regime']):.2f} and "
            f"{float(settings['minimum_affirmative_rate_per_split']):.2f}, and "
            f"{float(settings['minimum_affirmative_rate_per_family']):.2f}.",
            "",
            "The `coverage_rationale`, original IDs, user ID, and split are withheld "
            "from the packet. The hashed sample plan retains them for verification.",
            "",
        ]
    )
    return "\n".join(lines)


def _selection_rank(seed: int, split: str, regime: str, state_id: str) -> str:
    return sha256_text(f"{SEMANTIC_AUDIT_PLAN_VERSION}|{seed}|{split}|{regime}|{state_id}")


def select_balanced_semantic_audit_states(
    states: Sequence[PMV2State],
    evaluator_contexts: EvaluatorContextIndex,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evaluator_by_state = evaluator_contexts.require_states(states, exact=True)
    regime_minimum = int(settings["states_per_regime_per_split"])
    family_minimum = int(settings["minimum_states_per_family_per_split"])
    seed = int(settings["sample_seed"])
    groups: dict[tuple[str, str], list[PMV2State]] = defaultdict(list)
    for state in states:
        if state.split not in AUDITED_SPLITS:
            raise RuntimeError(
                f"semantic audit received unsupported split {state.split.value}"
            )
        regime = str(evaluator_by_state[state.state_id]["regime"])
        groups[(state.split.value, regime)].append(state)

    expected_cells = {
        (split.value, regime.value)
        for split in AUDITED_SPLITS
        for regime in ResourceNeedRegime
    }
    if set(groups) != expected_cells:
        raise RuntimeError(
            "semantic audit source does not exactly cover the 3x9 split/regime grid: "
            f"missing={sorted(expected_cells-set(groups))}, "
            f"extra={sorted(set(groups)-expected_cells)}"
        )

    selected: list[dict[str, Any]] = []
    for split in AUDITED_SPLITS:
        candidates = [state for state in states if state.split is split]
        families = sorted({state.semantic_family for state in candidates})
        remaining_regimes = {
            regime.value: regime_minimum for regime in ResourceNeedRegime
        }
        remaining_families = {
            family: family_minimum for family in families
        }
        chosen: list[PMV2State] = []
        chosen_ids: set[str] = set()
        used_users: set[str] = set()
        while any(remaining_regimes.values()) or any(remaining_families.values()):
            ranked = []
            for state in candidates:
                if state.state_id in chosen_ids:
                    continue
                regime = str(evaluator_by_state[state.state_id]["regime"])
                gain = int(remaining_regimes[regime] > 0) + int(
                    remaining_families[state.semantic_family] > 0
                )
                if gain == 0:
                    continue
                ranked.append(
                    (
                        -gain,
                        state.user_id in used_users,
                        _selection_rank(
                            seed, split.value, regime, state.state_id
                        ),
                        state,
                    )
                )
            if not ranked:
                raise RuntimeError(
                    "semantic audit deterministic set-cover cannot satisfy "
                    f"split={split.value}, remaining_regimes={remaining_regimes}, "
                    f"remaining_families={remaining_families}"
                )
            state = min(ranked, key=lambda row: row[:3])[3]
            regime = str(evaluator_by_state[state.state_id]["regime"])
            chosen.append(state)
            chosen_ids.add(state.state_id)
            used_users.add(state.user_id)
            remaining_regimes[regime] = max(0, remaining_regimes[regime] - 1)
            remaining_families[state.semantic_family] = max(
                0, remaining_families[state.semantic_family] - 1
            )
        for state in chosen:
            context = evaluator_by_state[state.state_id]
            item_id = "sem_" + sha256_text(
                f"{SEMANTIC_AUDIT_PLAN_VERSION}|{seed}|{state.state_id}"
            )[:20]
            selected.append(
                {
                    "item_id": item_id,
                    "state": state,
                    "context": context,
                }
            )
    if len({row["item_id"] for row in selected}) != len(selected):
        raise RuntimeError("semantic audit produced duplicate blinded item IDs")
    return selected


def _inventory_for_packet(state: PMV2State) -> dict[str, Any]:
    return {
        source.value: {
            "available": summary.available,
            "count": summary.count,
            "min_age_sessions": summary.min_age_sessions,
            "median_age_sessions": summary.median_age_sessions,
            "max_age_sessions": summary.max_age_sessions,
            "estimated_tokens": summary.estimated_tokens,
            "query_similarity_mean": summary.query_similarity_mean,
        }
        for source, summary in sorted(
            state.inventory.items(), key=lambda pair: pair[0].value
        )
    }


def load_semantic_audit_backend(
    path: str | Path, states: Sequence[PMV2State]
) -> dict[str, MemoryBackendRecord]:
    by_card: dict[str, MemoryBackendRecord] = {}
    for row_number, row in enumerate(iter_jsonl(path), 1):
        record = MemoryBackendRecord.model_validate(row)
        if record.card_id in by_card:
            raise RuntimeError(
                f"semantic audit backend has duplicate card {record.card_id} "
                f"at row {row_number}"
            )
        by_card[record.card_id] = record
    expected_cards = {state.card_id for state in states}
    if set(by_card) != expected_cards:
        raise RuntimeError(
            "semantic audit backend/state card matrix mismatch: "
            f"missing={sorted(expected_cards-set(by_card))}, "
            f"extra={sorted(set(by_card)-expected_cards)}"
        )
    return by_card


def require_pmv2_runtime_state_lineage(
    runtime_path: str | Path, states: Sequence[PMV2State]
) -> dict[str, Any]:
    expected = {
        state.state_id: state_to_v1_runtime(state).model_dump(mode="json")
        for state in states
    }
    observed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(iter_jsonl(runtime_path), 1):
        runtime = RuntimeState.model_validate(row)
        state_id = str((runtime.provenance or {}).get("pm_v2_state_id") or "")
        if not state_id:
            raise RuntimeError(
                f"PM-v2 runtime row {row_number} lacks provenance.pm_v2_state_id"
            )
        if state_id in observed:
            raise RuntimeError(f"PM-v2 runtime repeats state {state_id}")
        observed[state_id] = runtime.model_dump(mode="json")
    if set(observed) != set(expected):
        raise RuntimeError(
            "PM-v2 runtime/state matrix mismatch: "
            f"missing={sorted(set(expected)-set(observed))}, "
            f"extra={sorted(set(observed)-set(expected))}"
        )
    mismatched = sorted(
        state_id
        for state_id in expected
        if canonical_json(observed[state_id]) != canonical_json(expected[state_id])
    )
    if mismatched:
        raise RuntimeError(
            "PM-v2 runtime rows do not exactly derive from audited PM-v2 states: "
            + str(mismatched[:20])
        )
    return {
        "status": "PASS",
        "runtime_sha256": sha256_file(runtime_path),
        "state_count": len(expected),
        "state_ids_sha256": sha256_text(canonical_json(sorted(expected))),
    }


def semantic_packet_rows(
    selected: Sequence[Mapping[str, Any]],
    backend_by_card: Mapping[str, MemoryBackendRecord],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for selection in selected:
        state: PMV2State = selection["state"]
        context = dict(selection["context"])
        annotation_by_id = {
            str(row["memory_id"]): row for row in context["memory_annotations"]
        }
        if len(annotation_by_id) != len(context["memory_annotations"]):
            raise RuntimeError(
                f"semantic audit context has duplicate memory IDs: {state.state_id}"
            )
        backend = backend_by_card.get(state.card_id)
        if backend is None:
            raise RuntimeError(
                f"semantic audit backend lacks card {state.card_id}"
            )
        backend_ids = {item.memory_id for item in backend.items}
        if backend_ids != set(annotation_by_id):
            raise RuntimeError(
                f"semantic audit backend/evaluator memory matrix mismatch for "
                f"{state.state_id}"
            )
        memory_items = []
        for item in sorted(
            backend.items, key=lambda value: (value.source.value, value.memory_id)
        ):
            annotation = annotation_by_id[item.memory_id]
            if (
                annotation["source"] != item.source.value
                or int(annotation["created_session"]) != item.created_session
            ):
                raise RuntimeError(
                    f"semantic audit memory metadata mismatch: {item.memory_id}"
                )
            memory_items.append(
                {
                    "source": item.source.value,
                    "created_session": item.created_session,
                    "text": item.text,
                    "stale": annotation["stale"],
                    "conflicts_with_current_state": annotation[
                        "conflicts_with_current_state"
                    ],
                    "private_sensitivity": annotation["private_sensitivity"],
                    "item_utility": annotation["item_utility"],
                }
            )
        rows.append(
            {
                "item_id": str(selection["item_id"]),
                "current_user_text": state.current_user_text,
                "recent_dialogue_json": canonical_json(
                    [turn.model_dump(mode="json") for turn in state.current_session_history]
                ),
                "current_session_summary": state.current_session_summary,
                "authorized_user_context": str(context["authorized_user_context"]),
                "source_inventory_json": canonical_json(_inventory_for_packet(state)),
                "source_memory_items_json": canonical_json(memory_items),
                "candidate_semantic_family": state.semantic_family,
                "candidate_regime": str(context["regime"]),
                "candidate_needed_memory_sources_json": canonical_json(
                    context["needed_memory_sources"]
                ),
                "semantic_family_match": "",
                "regime_match": "",
                "needed_memory_sources_match": "",
                "memory_item_utility_match": "",
                "annotator_id": "",
                "notes": "",
            }
        )
    return rows


def write_semantic_packet(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PACKET_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def prepare_semantic_audit(
    *,
    states: Sequence[PMV2State],
    evaluator_contexts: EvaluatorContextIndex,
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    backend_by_card: Mapping[str, MemoryBackendRecord],
    out_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = semantic_audit_settings(config)
    out_dir = Path(out_dir)
    packet_path = out_dir / "semantic_sanity_packet.csv"
    manual_path = out_dir / "semantic_sanity_manual.md"
    plan_path = out_dir / "semantic_sanity_plan.json"
    targets = (packet_path, manual_path, plan_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "semantic audit outputs already exist; preserve them or use --overwrite: "
            + str(existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_balanced_semantic_audit_states(
        states, evaluator_contexts, settings
    )
    packet_rows = semantic_packet_rows(selected, backend_by_card)
    manual_path.write_text(semantic_audit_manual(settings), encoding="utf-8")
    write_semantic_packet(packet_path, packet_rows)
    plan_selections = []
    for selection, packet_row in zip(selected, packet_rows):
        state: PMV2State = selection["state"]
        context = selection["context"]
        protected = {name: packet_row[name] for name in PROTECTED_PACKET_FIELDS}
        plan_selections.append(
            {
                "item_id": selection["item_id"],
                "state_id": state.state_id,
                "card_id": state.card_id,
                "user_id": state.user_id,
                "split": state.split.value,
                "regime": context["regime"],
                "semantic_family": state.semantic_family,
                "needed_memory_sources": context["needed_memory_sources"],
                "evaluator_context_id": context["evaluator_context_id"],
                "protected_packet_row_sha256": sha256_text(
                    canonical_json(protected)
                ),
            }
        )
    source_families_by_split = {
        split.value: sorted(
            {state.semantic_family for state in states if state.split is split}
        )
        for split in AUDITED_SPLITS
    }
    selected_coverage = {
        split.value: {
            "regime_counts": {
                regime.value: sum(
                    1
                    for row in plan_selections
                    if row["split"] == split.value
                    and row["regime"] == regime.value
                )
                for regime in ResourceNeedRegime
            },
            "semantic_family_counts": {
                family: sum(
                    1
                    for row in plan_selections
                    if row["split"] == split.value
                    and row["semantic_family"] == family
                )
                for family in source_families_by_split[split.value]
            },
        }
        for split in AUDITED_SPLITS
    }
    plan = {
        "version": SEMANTIC_AUDIT_PLAN_VERSION,
        "protocol": SEMANTIC_AUDIT_PROTOCOL,
        "selection_algorithm": (
            "deterministic_greedy_set_cover_split_regime_and_family_v2"
        ),
        "semantic_sanity_config": settings,
        "input_bindings": {
            "pm_v2_config_sha256": sha256_file(config_path),
            "states_sha256": sha256_file(states_path),
            "backend_sha256": sha256_file(backend_path),
            "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
            "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        },
        "manual": {
            "version": SEMANTIC_AUDIT_MANUAL_VERSION,
            "sha256": sha256_file(manual_path),
        },
        "packet_sha256": sha256_file(packet_path),
        "expected_item_count": len(plan_selections),
        "expected_cell_count": len(AUDITED_SPLITS) * len(ResourceNeedRegime),
        "expected_split_family_cell_count": sum(
            len(families) for families in source_families_by_split.values()
        ),
        "source_semantic_families_by_split": source_families_by_split,
        "selected_coverage": selected_coverage,
        "selected_items": plan_selections,
    }
    plan["plan_sha256"] = sha256_text(canonical_json(plan))
    write_json(plan_path, plan)
    return {
        "status": "READY_FOR_INDEPENDENT_HUMAN_ANNOTATION",
        "protocol": SEMANTIC_AUDIT_PROTOCOL,
        "packet": str(packet_path),
        "manual": str(manual_path),
        "plan": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "item_count": len(plan_selections),
        "split_regime_cells": plan["expected_cell_count"],
        "api_calls": 0,
    }


def _read_packet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fields, rows


def _validate_plan(
    *,
    plan_path: Path,
    packet_path: Path,
    manual_path: Path,
    config_path: Path,
    states_path: Path,
    backend_path: Path,
    evaluator_contexts_path: Path,
    evaluator_contexts: EvaluatorContextIndex,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    plan = read_json(plan_path)
    without_self = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha256_text(canonical_json(without_self)):
        raise RuntimeError("semantic audit plan self-hash mismatch")
    if plan.get("version") != SEMANTIC_AUDIT_PLAN_VERSION:
        raise RuntimeError("semantic audit plan version mismatch")
    if plan.get("protocol") != SEMANTIC_AUDIT_PROTOCOL:
        raise RuntimeError("semantic audit plan protocol mismatch")
    if plan.get("semantic_sanity_config") != dict(settings):
        raise RuntimeError("semantic audit plan settings differ from frozen YAML")
    bindings = plan.get("input_bindings") or {}
    expected_bindings = {
        "pm_v2_config_sha256": sha256_file(config_path),
        "states_sha256": sha256_file(states_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
    }
    if bindings != expected_bindings:
        raise RuntimeError("semantic audit plan input lineage mismatch")
    if plan.get("packet_sha256") != sha256_file(packet_path):
        raise RuntimeError("semantic audit packet hash mismatch")
    manual = plan.get("manual") or {}
    if manual != {
        "version": SEMANTIC_AUDIT_MANUAL_VERSION,
        "sha256": sha256_file(manual_path),
    }:
        raise RuntimeError("semantic audit manual contract mismatch")
    states = load_states(states_path)
    expected_selection = select_balanced_semantic_audit_states(
        states, evaluator_contexts, settings
    )
    expected_count = len(expected_selection)
    if (
        int(plan.get("expected_cell_count", -1))
        != len(AUDITED_SPLITS) * len(ResourceNeedRegime)
        or int(plan.get("expected_item_count", -1)) != expected_count
        or len(plan.get("selected_items") or []) != expected_count
    ):
        raise RuntimeError("semantic audit plan is not the frozen set-cover")
    expected_identity = [
        (
            row["item_id"],
            row["state"].state_id,
            row["state"].split.value,
            str(row["context"]["regime"]),
            row["state"].semantic_family,
        )
        for row in expected_selection
    ]
    observed_identity = [
        (
            str(row["item_id"]),
            str(row["state_id"]),
            str(row["split"]),
            str(row["regime"]),
            str(row["semantic_family"]),
        )
        for row in plan["selected_items"]
    ]
    if observed_identity != expected_identity:
        raise RuntimeError("semantic audit plan differs from deterministic set-cover")
    cells = defaultdict(int)
    family_cells = defaultdict(int)
    for row in plan["selected_items"]:
        cells[(str(row["split"]), str(row["regime"]))] += 1
        family_cells[(str(row["split"]), str(row["semantic_family"]))] += 1
    minimum_cells = {
        (split.value, regime.value): int(settings["states_per_regime_per_split"])
        for split in AUDITED_SPLITS
        for regime in ResourceNeedRegime
    }
    if any(cells[key] < minimum for key, minimum in minimum_cells.items()):
        raise RuntimeError("semantic audit plan split/regime coverage mismatch")
    source_families = {
        split.value: sorted(
            {state.semantic_family for state in states if state.split is split}
        )
        for split in AUDITED_SPLITS
    }
    expected_family_cells = sum(len(values) for values in source_families.values())
    if (
        plan.get("source_semantic_families_by_split") != source_families
        or int(plan.get("expected_split_family_cell_count", -1))
        != expected_family_cells
    ):
        raise RuntimeError("semantic audit plan family universe mismatch")
    family_minimum = int(settings["minimum_states_per_family_per_split"])
    if any(
        family_cells[(split, family)] < family_minimum
        for split, families in source_families.items()
        for family in families
    ):
        raise RuntimeError("semantic audit plan split/family coverage mismatch")
    expected_coverage = {
        split.value: {
            "regime_counts": {
                regime.value: cells[(split.value, regime.value)]
                for regime in ResourceNeedRegime
            },
            "semantic_family_counts": {
                family: family_cells[(split.value, family)]
                for family in source_families[split.value]
            },
        }
        for split in AUDITED_SPLITS
    }
    if plan.get("selected_coverage") != expected_coverage:
        raise RuntimeError("semantic audit plan coverage summary mismatch")
    return plan


def _binary(value: str, *, item_id: str, field: str) -> int:
    stripped = str(value).strip()
    if stripped not in {"0", "1"}:
        raise ValueError(f"{item_id}: {field} must be exactly 0 or 1")
    return int(stripped)


def analyze_semantic_audit(
    *,
    completed_paths: Sequence[str | Path],
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    evaluator_contexts: EvaluatorContextIndex,
    packet_path: str | Path,
    manual_path: str | Path,
    plan_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not completed_paths:
        raise ValueError("semantic audit analysis requires completed annotation CSVs")
    settings = semantic_audit_settings(config)
    config_path = Path(config_path)
    states_path = Path(states_path)
    backend_path = Path(backend_path)
    evaluator_contexts_path = Path(evaluator_contexts_path)
    packet_path = Path(packet_path)
    manual_path = Path(manual_path)
    plan_path = Path(plan_path)
    report_path = Path(report_path)
    attestation_path = Path(attestation_path)
    if (report_path.exists() or attestation_path.exists()) and not overwrite:
        raise FileExistsError(
            "semantic audit report/attestation exists; preserve it or use --overwrite"
        )
    plan = _validate_plan(
        plan_path=plan_path,
        packet_path=packet_path,
        manual_path=manual_path,
        config_path=config_path,
        states_path=states_path,
        backend_path=backend_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        settings=settings,
    )
    packet_fields, packet_rows = _read_packet(packet_path)
    if packet_fields != list(PACKET_FIELDS):
        raise RuntimeError("semantic audit packet has an unexpected schema")
    packet_by_item = {row["item_id"]: row for row in packet_rows}
    if len(packet_by_item) != len(packet_rows):
        raise RuntimeError("semantic audit packet contains duplicate item IDs")
    plan_by_item = {str(row["item_id"]): row for row in plan["selected_items"]}
    if set(packet_by_item) != set(plan_by_item):
        raise RuntimeError("semantic audit packet and plan item sets differ")
    for item_id, packet_row in packet_by_item.items():
        protected = {name: packet_row[name] for name in PROTECTED_PACKET_FIELDS}
        if plan_by_item[item_id]["protected_packet_row_sha256"] != sha256_text(
            canonical_json(protected)
        ):
            raise RuntimeError(f"semantic audit protected row hash mismatch: {item_id}")

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    annotators: set[str] = set()
    for completed_path in (Path(path) for path in completed_paths):
        fields, rows = _read_packet(completed_path)
        if fields != list(PACKET_FIELDS):
            raise RuntimeError(
                f"completed semantic audit has an unexpected schema: {completed_path}"
            )
        for row in rows:
            item_id = str(row.get("item_id") or "")
            if item_id not in packet_by_item:
                raise RuntimeError(f"completed audit has unexpected item_id {item_id!r}")
            for name in PROTECTED_PACKET_FIELDS:
                if row.get(name) != packet_by_item[item_id][name]:
                    raise RuntimeError(
                        f"completed audit changed protected cell {item_id}/{name}"
                    )
            annotator_id = str(row.get("annotator_id") or "").strip()
            if not annotator_id:
                raise ValueError(f"{item_id}: missing annotator_id")
            key = (item_id, annotator_id)
            if key in seen:
                raise RuntimeError(
                    f"duplicate semantic annotation for {item_id}/{annotator_id}"
                )
            seen.add(key)
            annotators.add(annotator_id)
            annotations.append(
                {
                    "item_id": item_id,
                    "annotator_id": annotator_id,
                    **{
                        field: _binary(row[field], item_id=item_id, field=field)
                        for field in AUDIT_FIELDS
                    },
                }
            )

    minimum_annotators = int(settings["minimum_annotators"])
    if len(annotators) < minimum_annotators:
        raise RuntimeError(
            f"semantic audit has {len(annotators)} annotators; requires "
            f"{minimum_annotators}"
        )
    annotations_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        annotations_by_item[row["item_id"]].append(row)
    missing_or_underannotated = {
        item_id: len(annotations_by_item[item_id])
        for item_id in packet_by_item
        if len(annotations_by_item[item_id]) < minimum_annotators
    }
    if missing_or_underannotated:
        raise RuntimeError(
            "semantic audit items lack independent annotators: "
            + str(missing_or_underannotated)
        )

    affirmative_by_field: dict[str, float] = {}
    agreement_by_field: dict[str, float] = {}
    agreement_pair_counts: dict[str, int] = {}
    for field in AUDIT_FIELDS:
        values = [int(row[field]) for row in annotations]
        affirmative_by_field[field] = sum(values) / len(values)
        pair_equal: list[int] = []
        for item_rows in annotations_by_item.values():
            for left, right in itertools.combinations(item_rows, 2):
                pair_equal.append(int(left[field] == right[field]))
        if not pair_equal:
            raise RuntimeError(f"semantic audit has no annotation pair for {field}")
        agreement_by_field[field] = sum(pair_equal) / len(pair_equal)
        agreement_pair_counts[field] = len(pair_equal)

    grouped_affirmative: dict[str, dict[str, float]] = {
        "regime": {},
        "split": {},
        "semantic_family": {},
    }
    for grouping, plan_field in (
        ("regime", "regime"),
        ("split", "split"),
        ("semantic_family", "semantic_family"),
    ):
        categories = sorted({str(row[plan_field]) for row in plan["selected_items"]})
        for category in categories:
            values = [
                int(annotation[field])
                for annotation in annotations
                if str(plan_by_item[annotation["item_id"]][plan_field]) == category
                for field in AUDIT_FIELDS
            ]
            if not values:
                raise RuntimeError(f"semantic audit has empty {grouping}={category}")
            grouped_affirmative[grouping][category] = sum(values) / len(values)

    field_match_checks = {
        field: rate >= float(settings["minimum_affirmative_rate_per_field"])
        for field, rate in affirmative_by_field.items()
    }
    field_agreement_checks = {
        field: rate >= float(settings["minimum_pairwise_agreement_per_field"])
        for field, rate in agreement_by_field.items()
    }
    regime_checks = {
        name: rate >= float(settings["minimum_affirmative_rate_per_regime"])
        for name, rate in grouped_affirmative["regime"].items()
    }
    split_checks = {
        name: rate >= float(settings["minimum_affirmative_rate_per_split"])
        for name, rate in grouped_affirmative["split"].items()
    }
    family_checks = {
        name: rate >= float(settings["minimum_affirmative_rate_per_family"])
        for name, rate in grouped_affirmative["semantic_family"].items()
    }
    checks = {
        "minimum_annotators_per_item": not missing_or_underannotated,
        "minimum_unique_annotators": len(annotators) >= minimum_annotators,
        "affirmative_rate_per_field": field_match_checks,
        "pairwise_agreement_per_field": field_agreement_checks,
        "affirmative_rate_per_regime": regime_checks,
        "affirmative_rate_per_split": split_checks,
        "affirmative_rate_per_semantic_family": family_checks,
    }
    passed = (
        checks["minimum_annotators_per_item"]
        and checks["minimum_unique_annotators"]
        and all(field_match_checks.values())
        and all(field_agreement_checks.values())
        and all(regime_checks.values())
        and all(split_checks.values())
        and all(family_checks.values())
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": SEMANTIC_AUDIT_PROTOCOL,
        "semantic_sanity_config": settings,
        "input_bindings": {
            "pm_v2_config_sha256": sha256_file(config_path),
            "states_sha256": sha256_file(states_path),
            "backend_sha256": sha256_file(backend_path),
            "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
            "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
            "packet_sha256": sha256_file(packet_path),
            "manual_sha256": sha256_file(manual_path),
            "plan_file_sha256": sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "completed_sha256": {
                str(Path(path).resolve()): sha256_file(path)
                for path in completed_paths
            },
        },
        "sample": {
            "item_count": len(packet_by_item),
            "split_regime_cell_count": len(AUDITED_SPLITS)
            * len(ResourceNeedRegime),
            "split_semantic_family_cell_count": int(
                plan["expected_split_family_cell_count"]
            ),
            "annotations": len(annotations),
            "unique_annotators": len(annotators),
            "minimum_annotations_per_item": min(
                len(rows) for rows in annotations_by_item.values()
            ),
        },
        "metrics": {
            "affirmative_rate_per_field": affirmative_by_field,
            "pairwise_agreement_per_field": agreement_by_field,
            "agreement_pair_count_per_field": agreement_pair_counts,
            "affirmative_rate_per_regime": grouped_affirmative["regime"],
            "affirmative_rate_per_split": grouped_affirmative["split"],
            "affirmative_rate_per_semantic_family": grouped_affirmative[
                "semantic_family"
            ],
        },
        "gate": {"status": "PASS" if passed else "FAIL", "checks": checks},
        "api_calls": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    attestation_inputs: dict[str, str | Path] = {
        "pm_v2_config": config_path,
        "states": states_path,
        "backend": backend_path,
        "evaluator_contexts": evaluator_contexts_path,
        "packet": packet_path,
        "manual": manual_path,
        "plan": plan_path,
    }
    for index, path in enumerate(completed_paths):
        attestation_inputs[f"completed_{index:03d}"] = path
    create_artifact_attestation(
        attestation_path,
        stage=SEMANTIC_AUDIT_ATTESTATION_STAGE,
        inputs=attestation_inputs,
        outputs={"report": (report_path, False)},
        parameters={
            "protocol": SEMANTIC_AUDIT_PROTOCOL,
            "semantic_sanity_config": settings,
            "plan_sha256": plan["plan_sha256"],
            "gate_status": report["status"],
        },
        expected={
            "item_count": len(packet_by_item),
            "minimum_annotators": minimum_annotators,
            "minimum_expected_annotations": len(packet_by_item)
            * minimum_annotators,
        },
    )
    return report


def require_semantic_sanity_pass(
    *,
    report_path: str | Path,
    attestation_path: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
) -> dict[str, Any]:
    settings = semantic_audit_settings(config)
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=SEMANTIC_AUDIT_ATTESTATION_STAGE,
        required_output_paths={"report": report_path},
    )
    report = read_json(report_path)
    if report.get("status") != "PASS" or (report.get("gate") or {}).get(
        "status"
    ) != "PASS":
        raise RuntimeError("PM-v2 semantic-sanity audit did not PASS")
    if report.get("protocol") != SEMANTIC_AUDIT_PROTOCOL:
        raise RuntimeError("PM-v2 semantic-sanity audit protocol mismatch")
    if report.get("semantic_sanity_config") != settings:
        raise RuntimeError("PM-v2 semantic-sanity thresholds differ from frozen YAML")
    bindings = report.get("input_bindings") or {}
    expected = {
        "pm_v2_config_sha256": sha256_file(config_path),
        "states_sha256": sha256_file(states_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
    }
    mismatches = {
        name: {"expected": value, "observed": bindings.get(name)}
        for name, value in expected.items()
        if bindings.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            "PM-v2 semantic-sanity audit lineage mismatch: " + str(mismatches)
        )
    attestation = read_json(attestation_path)
    parameters = attestation.get("parameters") or {}
    if (
        parameters.get("protocol") != SEMANTIC_AUDIT_PROTOCOL
        or parameters.get("semantic_sanity_config") != settings
        or parameters.get("gate_status") != "PASS"
        or parameters.get("plan_sha256") != bindings.get("plan_sha256")
    ):
        raise RuntimeError("PM-v2 semantic-sanity attestation contract mismatch")
    return {
        "status": "PASS",
        "protocol": SEMANTIC_AUDIT_PROTOCOL,
        "report_sha256": sha256_file(report_path),
        "attestation_sha256": verification["attestation_sha256"],
        "plan_sha256": bindings["plan_sha256"],
        "item_count": int((report.get("sample") or {})["item_count"]),
    }
