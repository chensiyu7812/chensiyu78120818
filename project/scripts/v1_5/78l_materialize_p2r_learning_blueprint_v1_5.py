#!/usr/bin/env python3
"""Materialize the outcome-blind P2R learning blueprint.

This command intentionally has no API mode and no label/outcome inputs.  It
joins the Worker's longitudinal catalog to independently-authored current
state specifications, runs the frozen candidate selectors and contribution
slots, and emits candidate snapshots for later paired ON/OFF generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import iter_jsonl, read_json, write_json, write_jsonl
from metacom_pm.v1_5_v5_3_p2r_learning_blueprint import (
    P2RBlueprintRow,
    P2RCurrentStateSpec,
    P2RInteractionBlueprintRow,
    assemble_interaction_row,
    audit_blueprint_rows,
    materialize_memory_state,
    materialize_rs_state,
    normalize_worker_catalog,
)
from metacom_pm.v1_5_v5_3_release_bindings import build_static_release_bindings
from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder, CachedTextEncoder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
DEFAULT_STATE_SPECS = (
    ROOT / "outputs/pm_v1_5_v5_3_p2r_current_state_specs_v1/state_specs.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_v1"
STRATEGY_BANK = ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl"


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    value = read_json(path)
    if isinstance(value, list):
        return [dict(row) for row in value]
    for key in ("catalog", "catalog_items", "items", "rows"):
        rows = value.get(key) if isinstance(value, dict) else None
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    return []


def load_worker_catalog(catalog_dir: Path) -> list[Any]:
    if not catalog_dir.is_dir():
        raise FileNotFoundError(
            f"Worker longitudinal catalog is not available yet: {catalog_dir}"
        )
    preferred = [
        catalog_dir / "catalog.jsonl",
        catalog_dir / "catalog_items.jsonl",
        catalog_dir / "longitudinal_catalog.jsonl",
    ]
    files = [path for path in preferred if path.is_file()]
    if not files:
        files = sorted(catalog_dir.glob("*.jsonl")) + sorted(catalog_dir.glob("*.json"))
    raw: list[dict[str, Any]] = []
    for path in files:
        for row in _json_rows(path):
            keys = set(row)
            if {"memory_id", "candidate_id", "item_id"} & keys and {"component", "source"} & keys:
                raw.append(row)
    if not raw:
        raise ValueError(f"no catalog item rows found in {catalog_dir}")
    return normalize_worker_catalog(raw)


def load_cards() -> list[StrategyCard]:
    return [StrategyCard.model_validate(row) for row in iter_jsonl(STRATEGY_BANK)]


def validate_longitudinal_depth(catalog_dir: Path, catalog: list[Any]) -> dict[str, int]:
    users_path = catalog_dir / "users.json"
    if not users_path.is_file():
        raise FileNotFoundError("Worker catalog must include users.json session counts")
    users = read_json(users_path)
    declared = {str(row["user_id"]): int(row["n_sessions"]) for row in users}
    catalog_users = {item.catalog_user_id for item in catalog}
    if set(declared) != catalog_users:
        raise ValueError("users.json and catalog owner sets differ")
    bad = {user: count for user, count in declared.items() if not 13 <= count <= 33}
    if bad:
        raise ValueError(
            "catalog must preserve EvoEmo-like 13-33-session depth per user; "
            f"violations={bad}"
        )
    return declared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--state-specs", type=Path, default=DEFAULT_STATE_SPECS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-lexical-ms-debug",
        action="store_true",
        help="debug only; formal P2R must use the frozen BGE-M3 MS binding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.state_specs.is_file():
        raise FileNotFoundError(
            "Leader current-state specs are not materialized yet: "
            f"{args.state_specs}"
        )
    release = build_static_release_bindings(ROOT)
    catalog = load_worker_catalog(args.catalog_dir)
    depth = validate_longitudinal_depth(args.catalog_dir, catalog)
    states = [P2RCurrentStateSpec.model_validate(row) for row in iter_jsonl(args.state_specs)]
    if not states:
        raise ValueError("state spec file is empty")
    encoder = None
    if any(state.component == "MS" for state in states) and not args.allow_lexical_ms_debug:
        encoder = CachedTextEncoder(BgeM3Encoder())
    cards = load_cards()
    rows: list[P2RBlueprintRow] = []
    for state in states:
        row = (
            materialize_rs_state(state=state, cards=cards)
            if state.component == "RS"
            else materialize_memory_state(
                state=state,
                catalog=catalog,
                ms_semantic_encoder=encoder,
            )
        )
        rows.append(row)

    interaction_members: dict[str, dict[str, P2RBlueprintRow]] = defaultdict(dict)
    interaction_variants: dict[str, str] = {}
    for row in rows:
        key = row.state.interaction_key
        if key is None:
            continue
        interaction_members[key][row.state.component] = row
        interaction_variants[key] = row.state.interaction_variant or "default"
    interactions: list[P2RInteractionBlueprintRow] = [
        assemble_interaction_row(
            component_rows=members,
            variant=interaction_variants[key],
        )
        for key, members in sorted(interaction_members.items())
    ]
    audit = audit_blueprint_rows(
        rows,
        require_final_condition_crossing=True,
        require_assigned_splits=False,
    )
    audit["interaction_count"] = len(interactions)
    audit["interaction_component_sets"] = sorted(
        "+".join(sorted(row.components)) for row in interactions
    )
    audit["catalog_user_session_depth"] = depth
    audit["release_identity"] = release.release_identity
    audit["status"] = (
        "DRAFT_STATIC_PASS_AWAITING_GROUP_SPLIT_FREEZE"
        if audit["static_structure_pass"]
        else "DRAFT_STATIC_FAIL"
    )
    audit["ready_for_formal_paired_generation"] = False
    audit["formal_generation_blocker"] = (
        "independent_worker_review_and_group_split_freeze_pending"
    )
    audit["formal_ms_retriever"] = (
        "LEXICAL_DEBUG_NOT_FORMAL"
        if args.allow_lexical_ms_debug
        else release.ms_retriever.method
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out_dir / "component_rows.jsonl",
        [row.model_dump(mode="json") for row in rows],
    )
    write_jsonl(
        args.out_dir / "interaction_rows.jsonl",
        [row.model_dump(mode="json") for row in interactions],
    )
    write_json(args.out_dir / "report.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
