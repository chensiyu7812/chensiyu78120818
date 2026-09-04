#!/usr/bin/env python3
"""Build 64 RS contrasts on the same repaired backend as MP/MS/ME."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import importlib.util
from pathlib import Path
from typing import Any

from metacom_pm.contracts import (
    MemorySource,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transport-repaired-rs-contrast-blueprint-v1"
SPLIT_QUOTAS = {"train": 4, "calibration": 2, "internal_test": 2}
TARGET_QUANTILES = {
    "train": (0.10, 0.35, 0.65, 0.90),
    "calibration": (0.25, 0.75),
    "internal_test": (0.25, 0.75),
}
MAX_GROUPS_PER_USER_BY_SPLIT = {
    "train": 6,
    "calibration": 6,
    "internal_test": 4,
}


def _load_memory_blueprint_module():
    path = (
        ROOT
        / "scripts/v1_5/"
        "24bq_build_transport_repaired_memory_contrast_blueprint_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("_memory_blueprint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repaired blueprint helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory_subsets() -> list[frozenset[MemorySource]]:
    values = tuple(MemorySource)
    return [
        frozenset(group)
        for size in range(len(values) + 1)
        for group in combinations(values, size)
    ]


def build_blueprint(
    *,
    backend_dir: Path,
    memory_blueprint_dir: Path,
    strategy_cards_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend_report = read_json(backend_dir / "report.json")
    if backend_report.get("status") != (
        "READY_FOR_OUTCOME_BLIND_192_MEMORY_CONTRAST_BLUEPRINT"
    ):
        raise RuntimeError("transport-repaired backend is not qualified")
    memory_report = read_json(
        memory_blueprint_dir / "blueprint_report.json"
    )
    memory_rows = [
        dict(row)
        for row in iter_jsonl(
            memory_blueprint_dir / "memory_contrast_blueprint.jsonl"
        )
    ]
    if (
        memory_report.get("status")
        != "READY_192_MEMORY_PAIRS_ZERO_API_RETAIN_64_RS"
        or len(memory_rows) != 192
    ):
        raise RuntimeError("repaired memory blueprint is not frozen")

    module = _load_memory_blueprint_module()
    states = [
        dict(row) for row in iter_jsonl(backend_dir / "pm_v2_states.jsonl")
    ]
    retrieval = {
        (str(row["state_id"]), str(row["source"])): dict(row)
        for row in iter_jsonl(backend_dir / "retrieval_audit.jsonl")
    }
    cards = [dict(row) for row in iter_jsonl(strategy_cards_path)]
    candidates = {
        str(state["state_id"]): module._strategy_candidate(state, cards)
        for state in states
    }
    used_states = {str(row["state_id"]) for row in memory_rows}
    user_counts: Counter[tuple[str, str]] = Counter(
        (str(row["user_id"]), str(row["split"])) for row in memory_rows
    )

    requests = [
        {
            "sources": sources,
            "split": split,
            "ordinal": ordinal,
            "target_quantile": TARGET_QUANTILES[split][ordinal],
        }
        for sources in _memory_subsets()
        for split, count in SPLIT_QUOTAS.items()
        for ordinal in range(count)
    ]
    selected: list[dict[str, Any]] = []
    for request in requests:
        sources = frozenset(request["sources"])
        split = str(request["split"])
        available = [
            state
            for state in states
            if str(state["split"]) == split
            and str(state["state_id"]) not in used_states
            and candidates[str(state["state_id"])] is not None
            and user_counts[(str(state["user_id"]), split)]
            < MAX_GROUPS_PER_USER_BY_SPLIT[split]
            and all(
                retrieval[
                    (str(state["state_id"]), source.value)
                ]["selected_count"]
                > 0
                for source in sources
            )
        ]
        if not available:
            raise RuntimeError(f"no candidate for RS slot {request}")
        available.sort(
            key=lambda state: (
                float(
                    candidates[str(state["state_id"])][
                        "retrieval_score"
                    ]
                ),
                stable_hex(PROTOCOL, str(state["state_id"]), n=24),
            )
        )
        index = int(
            round(
                float(request["target_quantile"])
                * (len(available) - 1)
            )
        )
        state = available[index]
        state_id = str(state["state_id"])
        user_id = str(state["user_id"])
        used_states.add(state_id)
        user_counts[(user_id, split)] += 1
        control = canonical_action_id(sources, StrategyMode.R0)
        treatment = canonical_action_id(sources, StrategyMode.RS)
        memory_ids = [
            memory_id
            for source in sorted(sources, key=lambda value: value.value)
            for memory_id in retrieval[
                (state_id, source.value)
            ]["selected_memory_ids"]
        ]
        selected.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": "transport_rs_slot_"
                + stable_hex(
                    PROTOCOL,
                    control,
                    split,
                    str(request["ordinal"]),
                    n=24,
                ),
                "component": "RS",
                "split": split,
                "background_action": control,
                "control_action": control,
                "treatment_action": treatment,
                "state_id": state_id,
                "card_id": str(state["card_id"]),
                "user_id": user_id,
                "semantic_family_diagnostic_only": str(
                    state["semantic_family"]
                ),
                "session_index": int(state["session_index"]),
                "selection_target_quantile": float(
                    request["target_quantile"]
                ),
                "component_candidate_observation": {
                    "candidate_present": True,
                    "strategy_family": candidates[state_id][
                        "strategy_family"
                    ],
                    "execution_profile": candidates[state_id][
                        "execution_profile"
                    ],
                    "retrieval_score": candidates[state_id][
                        "retrieval_score"
                    ],
                },
                "selected_memory_ids_by_action_generation_only": {
                    control: memory_ids,
                    treatment: memory_ids,
                },
                "current_strategy_candidate": candidates[state_id],
                "control_generation_status": (
                    "GENERATE_ON_TRANSPORT_REPAIRED_BACKEND"
                ),
                "treatment_generation_status": (
                    "GENERATE_ON_TRANSPORT_REPAIRED_BACKEND"
                ),
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
                "material_risk": "UNKNOWN_UNLESS_COMPONENT_ON_WINS",
            }
        )

    by_split = Counter(row["split"] for row in selected)
    by_background = Counter(row["background_action"] for row in selected)
    report = {
        "protocol": PROTOCOL,
        "status": "READY_64_RS_PAIRS_ON_TRANSPORT_REPAIRED_STACK",
        "api_calls_made": 0,
        "human_labels_read": False,
        "effect_labels_created": False,
        "rs_contrasts": len(selected),
        "new_response_calls_required": 2 * len(selected),
        "distinct_states": len({row["state_id"] for row in selected}),
        "overlap_with_memory_contrast_states": len(
            {row["state_id"] for row in selected}
            & {row["state_id"] for row in memory_rows}
        ),
        "split_counts": dict(sorted(by_split.items())),
        "background_counts": dict(sorted(by_background.items())),
        "checks": {
            "exactly_64_rs": len(selected) == 64,
            "eight_per_memory_background": set(
                by_background.values()
            )
            == {8},
            "split_32_16_16": by_split
            == Counter(
                {"train": 32, "calibration": 16, "internal_test": 16}
            ),
            "no_state_overlap_with_memory_pairs": not (
                {row["state_id"] for row in selected}
                & {row["state_id"] for row in memory_rows}
            ),
            "one_bit_rs_only": all(
                parse_action_id(row["control_action"])[0]
                == parse_action_id(row["treatment_action"])[0]
                and parse_action_id(row["control_action"])[1]
                is StrategyMode.R0
                and parse_action_id(row["treatment_action"])[1]
                is StrategyMode.RS
                for row in selected
            ),
        },
        "inputs": {
            "backend_report": str(
                (backend_dir / "report.json").relative_to(ROOT)
            ),
            "backend_report_sha256": sha256_file(
                backend_dir / "report.json"
            ),
            "memory_blueprint": str(
                (
                    memory_blueprint_dir
                    / "memory_contrast_blueprint.jsonl"
                ).relative_to(ROOT)
            ),
            "memory_blueprint_sha256": sha256_file(
                memory_blueprint_dir / "memory_contrast_blueprint.jsonl"
            ),
            "strategy_cards": str(
                strategy_cards_path.relative_to(ROOT)
            ),
            "strategy_cards_sha256": sha256_file(strategy_cards_path),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "BLOCKED_BY_RS_BLUEPRINT_CHECK"
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
    )
    parser.add_argument(
        "--memory-blueprint-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1",
    )
    args = parser.parse_args()
    report, rows = build_blueprint(
        backend_dir=args.backend_dir,
        memory_blueprint_dir=args.memory_blueprint_dir,
        strategy_cards_path=args.strategy_cards,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rs_contrast_blueprint.jsonl"
    write_jsonl(rows_path, rows)
    report["outputs"] = {rows_path.name: sha256_file(rows_path)}
    write_json(args.out_dir / "blueprint_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "rs_contrasts": report["rs_contrasts"],
            "new_response_calls": report["new_response_calls_required"],
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
