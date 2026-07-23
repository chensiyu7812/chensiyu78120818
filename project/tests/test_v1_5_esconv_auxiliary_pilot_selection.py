"""Tests for scripts/v1_5/13a_select_esconv_auxiliary_pilot_v1_5.py.

The pilot-selection rule must be fully observable and outcome-free: 12
distinct train dialogues, one state each, stratified 4 early/4 mid/4 late by
the state's own position in its dialogue's turn sequence. Never touches
audit_only.jsonl or any gold/judge/outcome field.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.io import write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "v1_5" / "13a_select_esconv_auxiliary_pilot_v1_5.py"
    spec = importlib.util.spec_from_file_location(
        "v1_5_esconv_auxiliary_pilot_selection_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(tmp_path: Path, *, n_dialogues: int, states_per_dialogue: int):
    seeds = [
        {"dialogue_id": f"dlg_{i:02d}", "selection_ordinal": i}
        for i in range(n_dialogues)
    ]
    seeds_path = tmp_path / "seeds.jsonl"
    write_jsonl(seeds_path, seeds)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    runtime_rows = []
    backend_rows = []
    pmv2_rows = []
    for i in range(n_dialogues):
        for t in range(states_per_dialogue):
            turn_index = 3 + t * 2
            state_id = f"state_{i:02d}_{t:02d}"
            card_id = f"card_{i:02d}_{t:02d}"
            runtime_rows.append(
                {
                    "state_id": state_id,
                    "card_id": card_id,
                    "provenance": {
                        "dialogue_id": f"dlg_{i:02d}",
                        "turn_index": turn_index,
                    },
                }
            )
            backend_rows.append({"card_id": card_id, "items": []})
            pmv2_rows.append({"state_id": state_id, "dummy": True})
    write_jsonl(train_dir / "runtime_states.jsonl", runtime_rows)
    write_jsonl(train_dir / "memory_backend.jsonl", backend_rows)
    write_jsonl(train_dir / "pm_v2_states.jsonl", pmv2_rows)
    return seeds_path, train_dir


def test_selects_12_distinct_dialogues_stratified_4_4_4(tmp_path):
    module = _load_module()
    seeds_path, train_dir = _write_fixture(tmp_path, n_dialogues=24, states_per_dialogue=9)
    result = module.select_pilot_states(
        selected_seed_sources_path=seeds_path, train_split_dir=train_dir
    )
    manifest = result["manifest"]
    assert len(manifest) == 12
    assert len({row["dialogue_id"] for row in manifest}) == 12
    strata = [row["stratum"] for row in manifest]
    assert strata.count("early") == 4
    assert strata.count("mid") == 4
    assert strata.count("late") == 4
    # Early picks the smallest turn_index in its dialogue; late the largest.
    for row in manifest:
        if row["stratum"] == "early":
            assert row["turn_index"] == 3
        if row["stratum"] == "late":
            assert row["turn_index"] == 3 + (9 - 1) * 2


def test_dialogue_offset_selects_a_disjoint_deterministic_window(tmp_path):
    module = _load_module()
    seeds_path, train_dir = _write_fixture(tmp_path, n_dialogues=24, states_per_dialogue=9)
    first = module.select_pilot_states(
        selected_seed_sources_path=seeds_path,
        train_split_dir=train_dir,
        dialogue_offset=0,
    )
    second = module.select_pilot_states(
        selected_seed_sources_path=seeds_path,
        train_split_dir=train_dir,
        dialogue_offset=12,
    )
    first_dialogues = {row["dialogue_id"] for row in first["manifest"]}
    second_dialogues = {row["dialogue_id"] for row in second["manifest"]}
    assert len(first_dialogues) == 12
    assert len(second_dialogues) == 12
    assert first_dialogues.isdisjoint(second_dialogues)
    # Same offset, same fixture -> byte-identical selection (deterministic).
    repeat = module.select_pilot_states(
        selected_seed_sources_path=seeds_path,
        train_split_dir=train_dir,
        dialogue_offset=12,
    )
    assert repeat["manifest"] == second["manifest"]


def test_dialogue_offset_out_of_range_is_rejected(tmp_path):
    module = _load_module()
    seeds_path, train_dir = _write_fixture(tmp_path, n_dialogues=24, states_per_dialogue=9)
    with pytest.raises(ValueError, match="dialogue_offset"):
        module.select_pilot_states(
            selected_seed_sources_path=seeds_path,
            train_split_dir=train_dir,
            dialogue_offset=13,
        )
    with pytest.raises(ValueError, match="dialogue_offset"):
        module.select_pilot_states(
            selected_seed_sources_path=seeds_path,
            train_split_dir=train_dir,
            dialogue_offset=-1,
        )


def test_rejects_wrong_train_dialogue_count(tmp_path):
    module = _load_module()
    seeds_path, train_dir = _write_fixture(tmp_path, n_dialogues=10, states_per_dialogue=5)
    with pytest.raises(RuntimeError, match="expected exactly 24"):
        module.select_pilot_states(
            selected_seed_sources_path=seeds_path, train_split_dir=train_dir
        )


def test_never_references_gold_or_audit_fields():
    source = (
        ROOT / "scripts" / "v1_5" / "13a_select_esconv_auxiliary_pilot_v1_5.py"
    ).read_text(encoding="utf-8")
    _, _, code_after_docstring = source.partition('"""')
    _, _, code = code_after_docstring.partition('"""')
    assert "audit_only" not in code
    assert "gold_response" not in code
    assert "gold_strategy" not in code
    assert "judge" not in code.lower()
    assert "outcome" not in code.lower()
