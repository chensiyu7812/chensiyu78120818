from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from metacom_pm.contracts import (
    ALL_ACTION_IDS,
    DialogueTurn,
    MemoryBackendRecord,
    MemoryItem,
    MemorySource,
)
from metacom_pm.io import read_json, sha256_text, write_jsonl
from metacom_pm.pm_v2_contracts import (
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
    ResourceNeedRegime,
)
from metacom_pm.pm_v2_data import (
    build_evaluator_context_index,
    evaluator_context_payload_sha256,
    load_evaluator_context_index,
    load_states,
    state_to_v1_runtime,
)
from metacom_pm.pm_v2_semantic_audit import (
    AUDIT_FIELDS,
    PACKET_FIELDS,
    PROTECTED_PACKET_FIELDS,
    analyze_semantic_audit,
    load_semantic_audit_backend,
    prepare_semantic_audit,
    require_semantic_sanity_pass,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _needed_sources(regime: ResourceNeedRegime) -> list[str]:
    return {
        ResourceNeedRegime.CONTEXT_ONLY: [],
        ResourceNeedRegime.PROFILE_NEEDED: ["MP"],
        ResourceNeedRegime.SUMMARY_NEEDED: ["MS"],
        ResourceNeedRegime.EVENT_NEEDED: ["ME"],
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: ["MP", "ME"],
        ResourceNeedRegime.MEMORY_HARMFUL: [],
        ResourceNeedRegime.STRATEGY_HELPFUL: [],
        ResourceNeedRegime.STRATEGY_HARMFUL: [],
        ResourceNeedRegime.AMBIGUOUS: [],
    }[regime]


def _semantic_fixture(tmp_path: Path) -> dict:
    states = []
    contexts = []
    backend_records = []
    for split in (
        PMV2Split.TRAIN,
        PMV2Split.CALIBRATION,
        PMV2Split.INTERNAL_TEST,
    ):
        for regime in ResourceNeedRegime:
          for replicate in range(2 if split is PMV2Split.TRAIN else 1):
            base_suffix = f"{split.value}_{regime.value}"
            suffix = f"{base_suffix}_rep{replicate}"
            state_id = f"state_{suffix}"
            card_id = f"card_{suffix}"
            evaluator_id = f"eval_{suffix}"
            inventory = {
                source: ObservableSourceSummary(
                    available=True,
                    count=1,
                    min_age_sessions=1,
                    median_age_sessions=1.0,
                    max_age_sessions=1,
                    estimated_tokens=12,
                    query_similarity_mean=0.2,
                    catalog_embedding=[0.0] * 64,
                )
                for source in MemorySource
            }
            state = PMV2State(
                state_id=state_id,
                card_id=card_id,
                user_id=(
                    f"user_train_rep{replicate}"
                    if split is PMV2Split.TRAIN
                    else f"user_{suffix}"
                ),
                split=split,
                semantic_family=f"family_{base_suffix}",
                surface_form_id=f"surface_{suffix}",
                current_user_text=f"I need help with a unique {suffix} situation.",
                current_session_history=[
                    DialogueTurn(role="user", content=f"Earlier context for {suffix}.")
                ],
                current_session_summary=f"Current-session summary for {suffix}.",
                session_index=4,
                inventory=inventory,
                strategy_catalog_count=3,
                strategy_estimated_tokens=120,
                allowed_actions=list(ALL_ACTION_IDS),
                provenance={
                    "backend_record_id": card_id,
                    "evaluator_context_id": evaluator_id,
                },
            )
            memory_items = [
                MemoryItem(
                    memory_id="mem_" + sha256_text(f"{suffix}|{source.value}")[:24],
                    source=source,
                    created_session=2,
                    text=f"Synthetic {source.value} evidence for {suffix}.",
                )
                for source in MemorySource
            ]
            context = {
                "evaluator_context_id": evaluator_id,
                "state_id": state_id,
                "card_id": card_id,
                "regime": regime.value,
                "needed_memory_sources": _needed_sources(regime),
                "authorized_user_context": (
                    f"Authorized longitudinal context for {suffix}."
                ),
                "coverage_rationale": f"Hidden rationale for {suffix}.",
                "memory_annotations": [
                    {
                        "memory_id": item.memory_id,
                        "source": item.source.value,
                        "created_session": item.created_session,
                        "stale": regime is ResourceNeedRegime.MEMORY_HARMFUL,
                        "conflicts_with_current_state": False,
                        "private_sensitivity": "ordinary",
                        "item_utility": (
                            "helpful"
                            if item.source.value in _needed_sources(regime)
                            else "irrelevant"
                        ),
                    }
                    for item in memory_items
                ],
            }
            context["context_payload_sha256"] = evaluator_context_payload_sha256(
                context
            )
            states.append(state)
            contexts.append(context)
            backend_records.append(
                MemoryBackendRecord(card_id=card_id, items=memory_items)
            )

    states_path = tmp_path / "pm_v2_states.jsonl"
    backend_path = tmp_path / "memory_backend.jsonl"
    contexts_path = tmp_path / "evaluator_contexts.jsonl"
    write_jsonl(states_path, [state.model_dump(mode="json") for state in states])
    write_jsonl(
        backend_path,
        [record.model_dump(mode="json") for record in backend_records],
    )
    write_jsonl(contexts_path, contexts)
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "pm_v2.yaml").read_text(encoding="utf-8")
    )
    config["development_judging"]["compatibility_pilot"][
        "states_per_regime"
    ] = 1
    observability = config["development_judging"]["compatibility_pilot"][
        "deployable_feature_observability"
    ]
    observability["cv_folds"] = 2
    observability["minimum_train_states"] = 18
    observability["minimum_regime_macro_f1"] = 0.0
    observability["minimum_needed_sources_macro_f1"] = 0.0
    observability["minimum_memory_need_macro_f1"] = 0.0
    observability["minimum_strategy_direction_macro_f1"] = 0.0
    observability["minimum_memory_direction_macro_f1"] = 0.0
    config_path = tmp_path / "pm_v2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "states": states,
        "contexts": contexts,
        "states_path": states_path,
        "backend_path": backend_path,
        "contexts_path": contexts_path,
        "config": config,
        "config_path": config_path,
    }


def _completed_packet(packet_path: Path, out_path: Path, annotator: str) -> None:
    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in AUDIT_FIELDS:
            row[field] = "1"
        row["annotator_id"] = annotator
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PACKET_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_and_pass(tmp_path: Path) -> dict:
    fixture = _semantic_fixture(tmp_path)
    states = load_states(fixture["states_path"])
    contexts = load_evaluator_context_index(
        fixture["contexts_path"], states=states, require_exact=True
    )
    backend_by_card = load_semantic_audit_backend(fixture["backend_path"], states)
    out_dir = tmp_path / "semantic"
    result = prepare_semantic_audit(
        states=states,
        evaluator_contexts=contexts,
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        backend_by_card=backend_by_card,
        out_dir=out_dir,
    )
    first = tmp_path / "annotator_a.csv"
    second = tmp_path / "annotator_b.csv"
    _completed_packet(Path(result["packet"]), first, "annotator_a")
    _completed_packet(Path(result["packet"]), second, "annotator_b")
    report_path = out_dir / "semantic_sanity_report.json"
    attestation_path = out_dir / "artifact_attestation.json"
    report = analyze_semantic_audit(
        completed_paths=[first, second],
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        evaluator_contexts=contexts,
        packet_path=result["packet"],
        manual_path=result["manual"],
        plan_path=result["plan"],
        report_path=report_path,
        attestation_path=attestation_path,
    )
    fixture.update(
        {
            "contexts_index": contexts,
            "semantic_result": result,
            "report": report,
            "report_path": report_path,
            "attestation_path": attestation_path,
            "completed_paths": [first, second],
        }
    )
    return fixture


def test_semantic_packet_is_balanced_deterministic_and_blinded(tmp_path: Path) -> None:
    fixture = _semantic_fixture(tmp_path)
    states = load_states(fixture["states_path"])
    contexts = load_evaluator_context_index(
        fixture["contexts_path"], states=states, require_exact=True
    )
    backend_by_card = load_semantic_audit_backend(fixture["backend_path"], states)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_semantic_audit(
        states=states,
        evaluator_contexts=contexts,
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        backend_by_card=backend_by_card,
        out_dir=first_dir,
    )
    second = prepare_semantic_audit(
        states=states,
        evaluator_contexts=contexts,
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        backend_by_card=backend_by_card,
        out_dir=second_dir,
    )
    assert first["item_count"] == 27
    assert first["split_regime_cells"] == 27
    assert first["plan_sha256"] == second["plan_sha256"]
    plan = read_json(first["plan"])
    assert len(
        {(row["split"], row["regime"]) for row in plan["selected_items"]}
    ) == 27
    with Path(first["packet"]).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PACKET_FIELDS)
        packet_rows = list(reader)
    assert len(packet_rows) == 27
    assert not ({"state_id", "user_id", "split", "coverage_rationale"} & set(PACKET_FIELDS))
    packet_text = Path(first["packet"]).read_text(encoding="utf-8")
    assert "Hidden rationale" not in packet_text
    assert all(row["item_id"].startswith("sem_") for row in packet_rows)
    assert all(row[field] == "" for row in packet_rows for field in AUDIT_FIELDS)
    assert "API" in Path(first["manual"]).read_text(encoding="utf-8")


def test_semantic_set_cover_includes_every_family_when_families_exceed_regimes(
    tmp_path: Path,
) -> None:
    fixture = _semantic_fixture(tmp_path)
    states = list(fixture["states"])
    contexts = list(fixture["contexts"])
    backend_records = [
        MemoryBackendRecord.model_validate(row)
        for row in (
            json.loads(line)
            for line in fixture["backend_path"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    train_bases = [state for state in states if state.split is PMV2Split.TRAIN]
    context_by_state = {row["state_id"]: row for row in contexts}
    backend_by_card = {row.card_id: row for row in backend_records}
    for index, base in enumerate(train_bases[:5]):
        suffix = f"train_extra_family_{index}"
        state = base.model_copy(
            update={
                "state_id": f"state_{suffix}",
                "card_id": f"card_{suffix}",
                "user_id": f"user_{suffix}",
                "semantic_family": f"family_extra_{index}",
                "surface_form_id": f"surface_{suffix}",
                "provenance": {
                    "backend_record_id": f"card_{suffix}",
                    "evaluator_context_id": f"eval_{suffix}",
                },
            }
        )
        base_backend = backend_by_card[base.card_id]
        new_items = []
        id_map = {}
        for item in base_backend.items:
            new_id = "mem_" + sha256_text(f"{suffix}|{item.source.value}")[:24]
            id_map[item.memory_id] = new_id
            new_items.append(item.model_copy(update={"memory_id": new_id}))
        context = json.loads(json.dumps(context_by_state[base.state_id]))
        context.update(
            {
                "evaluator_context_id": f"eval_{suffix}",
                "state_id": state.state_id,
                "card_id": state.card_id,
            }
        )
        for annotation in context["memory_annotations"]:
            annotation["memory_id"] = id_map[annotation["memory_id"]]
        context["context_payload_sha256"] = evaluator_context_payload_sha256(
            context
        )
        states.append(state)
        contexts.append(context)
        backend_records.append(
            MemoryBackendRecord(card_id=state.card_id, items=new_items)
        )
    write_jsonl(
        fixture["states_path"],
        [state.model_dump(mode="json") for state in states],
    )
    write_jsonl(fixture["contexts_path"], contexts)
    write_jsonl(
        fixture["backend_path"],
        [record.model_dump(mode="json") for record in backend_records],
    )
    loaded_states = load_states(fixture["states_path"])
    context_index = load_evaluator_context_index(
        fixture["contexts_path"], states=loaded_states, require_exact=True
    )
    loaded_backend = load_semantic_audit_backend(
        fixture["backend_path"], loaded_states
    )
    result = prepare_semantic_audit(
        states=loaded_states,
        evaluator_contexts=context_index,
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        backend_by_card=loaded_backend,
        out_dir=tmp_path / "set_cover",
    )
    plan = read_json(result["plan"])
    split_counts = {
        split.value: sum(
            row["split"] == split.value for row in plan["selected_items"]
        )
        for split in (PMV2Split.TRAIN, PMV2Split.CALIBRATION, PMV2Split.INTERNAL_TEST)
    }
    assert split_counts == {"train": 14, "calibration": 9, "internal_test": 9}
    assert result["item_count"] == 32
    assert plan["expected_split_family_cell_count"] == 32
    for split, families in plan["source_semantic_families_by_split"].items():
        observed = {
            row["semantic_family"]
            for row in plan["selected_items"]
            if row["split"] == split
        }
        assert observed == set(families)
        assert all(
            count >= 1
            for count in plan["selected_coverage"][split][
                "semantic_family_counts"
            ].values()
        )


def test_semantic_analysis_requires_two_annotators_and_passes_all_yes(
    tmp_path: Path,
) -> None:
    fixture = _semantic_fixture(tmp_path)
    states = load_states(fixture["states_path"])
    contexts = load_evaluator_context_index(
        fixture["contexts_path"], states=states, require_exact=True
    )
    backend_by_card = load_semantic_audit_backend(fixture["backend_path"], states)
    out_dir = tmp_path / "semantic"
    result = prepare_semantic_audit(
        states=states,
        evaluator_contexts=contexts,
        config=fixture["config"],
        config_path=fixture["config_path"],
        states_path=fixture["states_path"],
        backend_path=fixture["backend_path"],
        evaluator_contexts_path=fixture["contexts_path"],
        backend_by_card=backend_by_card,
        out_dir=out_dir,
    )
    one = tmp_path / "one.csv"
    _completed_packet(Path(result["packet"]), one, "only_one")
    with pytest.raises(RuntimeError, match="requires 2"):
        analyze_semantic_audit(
            completed_paths=[one],
            config=fixture["config"],
            config_path=fixture["config_path"],
            states_path=fixture["states_path"],
            backend_path=fixture["backend_path"],
            evaluator_contexts_path=fixture["contexts_path"],
            evaluator_contexts=contexts,
            packet_path=result["packet"],
            manual_path=result["manual"],
            plan_path=result["plan"],
            report_path=out_dir / "report.json",
            attestation_path=out_dir / "attestation.json",
        )

    passed = _prepare_and_pass(tmp_path / "passed")
    assert passed["report"]["status"] == "PASS"
    assert passed["report"]["sample"]["unique_annotators"] == 2
    assert set(passed["report"]["metrics"]["affirmative_rate_per_field"]) == set(
        AUDIT_FIELDS
    )
    binding = require_semantic_sanity_pass(
        report_path=passed["report_path"],
        attestation_path=passed["attestation_path"],
        config=passed["config"],
        config_path=passed["config_path"],
        states_path=passed["states_path"],
        backend_path=passed["backend_path"],
        evaluator_contexts_path=passed["contexts_path"],
    )
    assert binding["status"] == "PASS"
    assert binding["item_count"] == 27

    with pytest.raises(RuntimeError, match="duplicate semantic annotation"):
        analyze_semantic_audit(
            completed_paths=[
                passed["completed_paths"][0],
                passed["completed_paths"][0],
            ],
            config=passed["config"],
            config_path=passed["config_path"],
            states_path=passed["states_path"],
            backend_path=passed["backend_path"],
            evaluator_contexts_path=passed["contexts_path"],
            evaluator_contexts=passed["contexts_index"],
            packet_path=passed["semantic_result"]["packet"],
            manual_path=passed["semantic_result"]["manual"],
            plan_path=passed["semantic_result"]["plan"],
            report_path=tmp_path / "duplicate_report.json",
            attestation_path=tmp_path / "duplicate_attestation.json",
        )

    tampered_completed = tmp_path / "tampered_completed.csv"
    with passed["completed_paths"][0].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        tampered_rows = list(csv.DictReader(handle))
    tampered_rows[0]["current_user_text"] += " tampered"
    with tampered_completed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PACKET_FIELDS))
        writer.writeheader()
        writer.writerows(tampered_rows)
    with pytest.raises(RuntimeError, match="changed protected cell"):
        analyze_semantic_audit(
            completed_paths=[tampered_completed, passed["completed_paths"][1]],
            config=passed["config"],
            config_path=passed["config_path"],
            states_path=passed["states_path"],
            backend_path=passed["backend_path"],
            evaluator_contexts_path=passed["contexts_path"],
            evaluator_contexts=passed["contexts_index"],
            packet_path=passed["semantic_result"]["packet"],
            manual_path=passed["semantic_result"]["manual"],
            plan_path=passed["semantic_result"]["plan"],
            report_path=tmp_path / "tampered_report.json",
            attestation_path=tmp_path / "tampered_attestation.json",
        )


def test_semantic_gate_rejects_tampering_and_script31_requires_pass(
    tmp_path: Path,
) -> None:
    fixture = _prepare_and_pass(tmp_path)
    changed_states = tmp_path / "changed_states.jsonl"
    changed_states.write_text(
        fixture["states_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lineage mismatch"):
        require_semantic_sanity_pass(
            report_path=fixture["report_path"],
            attestation_path=fixture["attestation_path"],
            config=fixture["config"],
            config_path=fixture["config_path"],
            states_path=changed_states,
            backend_path=fixture["backend_path"],
            evaluator_contexts_path=fixture["contexts_path"],
        )

    runtime = tmp_path / "runtime_states.jsonl"
    backend = fixture["backend_path"]
    strategies = tmp_path / "strategy_cards.jsonl"
    write_jsonl(
        runtime,
        [state_to_v1_runtime(state).model_dump(mode="json") for state in fixture["states"]],
    )
    strategies.write_text("{}\n", encoding="utf-8")
    pilot_plan = tmp_path / "pilot_plan.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/31_plan_pm_v2_development_pilot.py"),
        "--pm-v2-config",
        str(fixture["config_path"]),
        "--states",
        str(fixture["states_path"]),
        "--runtime",
        str(runtime),
        "--backend",
        str(backend),
        "--evaluator-contexts",
        str(fixture["contexts_path"]),
        "--strategy-bank",
        str(strategies),
        "--semantic-sanity-report",
        str(fixture["report_path"]),
        "--semantic-sanity-attestation",
        str(fixture["attestation_path"]),
        "--out",
        str(pilot_plan),
    ]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    saved = json.loads(pilot_plan.read_text(encoding="utf-8"))
    assert saved["status"] == "READY"
    assert saved["semantic_sanity"]["status"] == "PASS"

    missing = subprocess.run(
        [
            *command[: command.index("--semantic-sanity-report")],
            "--semantic-sanity-report",
            str(tmp_path / "missing_report.json"),
            "--semantic-sanity-attestation",
            str(tmp_path / "missing_attestation.json"),
            "--out",
            str(tmp_path / "must_not_exist.json"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "missing attestation" in missing.stderr
    assert not (tmp_path / "must_not_exist.json").exists()

    report_payload = read_json(fixture["report_path"])
    report_payload["sample"]["item_count"] = 999
    fixture["report_path"].write_text(
        json.dumps(report_payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="attestation failed"):
        require_semantic_sanity_pass(
            report_path=fixture["report_path"],
            attestation_path=fixture["attestation_path"],
            config=fixture["config"],
            config_path=fixture["config_path"],
            states_path=fixture["states_path"],
            backend_path=fixture["backend_path"],
            evaluator_contexts_path=fixture["contexts_path"],
        )


def test_script06_contains_preplan_semantic_gate() -> None:
    source = (PROJECT_ROOT / "scripts" / "06_run_action_sweep.py").read_text(
        encoding="utf-8"
    )
    assert "require_semantic_sanity_pass(" in source
    assert source.index("require_semantic_sanity_pass(") < source.index(
        "plan_action_sweep("
    )
    assert "semantic_sanity" in source
