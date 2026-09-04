#!/usr/bin/env python3
"""Prepare the blinded, no-API human semantic-sanity packet for PM-v2 states."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_semantic_audit import (
    load_semantic_audit_backend,
    prepare_semantic_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic 3-split x 9-regime blinded semantic audit; "
            "this command never calls an API."
        )
    )
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_semantic_sanity",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    if config.get("version") != "pm-v2.2":
        raise RuntimeError("semantic-sanity audit requires PM-v2.2 config")
    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    backend_by_card = load_semantic_audit_backend(args.backend, states)
    result = prepare_semantic_audit(
        states=states,
        evaluator_contexts=evaluator_contexts,
        config=config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        backend_by_card=backend_by_card,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
