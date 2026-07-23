from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.evoemo import evoemo_chronology_audit, normalize_evoemo_chronology
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.paid_run_release import (
    PAID_RUN_RELEASE_PROTOCOL,
    require_output_directory_not_previously_consumed,
    require_paid_run_release,
    resolve_first_unconsumed_output_directory,
)
from metacom_pm.pm_v1_5_algorithm_selection import (
    _select_one_standard_error_candidate,
)
from metacom_pm.freeze import _is_current_release_python as freeze_includes_python
from metacom_pm.release import _is_current_release_python as release_includes_python
from metacom_pm.v1_5_actual_corpus_review import (
    ACTUAL_CITATION_POLICY,
    ACTUAL_CORPUS_CONTROL_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_STAGE,
    ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS,
    ACTUAL_DETERMINISTIC_FIELDS,
    ACTUAL_PANEL_POLICY,
    ACTUAL_SEMANTIC_FIELDS,
    _surface_fallback_report,
    aggregate_actual_corpus_gate,
    build_actual_corpus_controls,
    evaluate_actual_deterministic_payload,
    require_actual_corpus_semantic_review_pass,
)
from metacom_pm.v1_5_automated_semantic_review import (
    V1_5_REVIEW_STRATEGY_CARD_IDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_scan_and_study_freeze_cover_every_active_v1_5_script() -> None:
    scripts = [
        *sorted((ROOT / "scripts" / "v1_5").glob("*.py")),
        *sorted((ROOT / "scripts").glob("v1_5_*.py")),
    ]
    assert scripts
    assert all(release_includes_python(ROOT, path) for path in scripts)
    assert all(freeze_includes_python(ROOT, path) for path in scripts)


def test_every_v1_5_paid_entrypoint_calls_the_central_release_gate() -> None:
    candidates = [
        *sorted((ROOT / "scripts" / "v1_5").glob("*.py")),
        *sorted((ROOT / "scripts").glob("v1_5_*.py")),
    ]
    paid_entrypoints = []
    for path in candidates:
        source = path.read_text(encoding="utf-8")
        if 'add_argument("--run"' in source:
            paid_entrypoints.append(path)
            assert "require_paid_run_release" in source, path
    assert paid_entrypoints


def test_formal_chain_cannot_be_authorized_by_consumed_v4_review() -> None:
    paths = [
        ROOT / "scripts" / "v1_5" / "20_generate_pm_v2_development_data_v1_5.py",
        ROOT / "scripts" / "v1_5" / "06_run_action_sweep_v1_5.py",
        ROOT / "scripts" / "v1_5" / "21_judge_pm_v2_action_sweep_v1_5.py",
    ]
    sources = [path.read_text(encoding="utf-8") for path in paths]
    assert all("require_automated_semantic_review_pass" not in source for source in sources)
    assert "formal generation refuses the consumed V4 semantic-review artifacts" in sources[0]
    assert "pm-v1.5-actual-468-structured-qa-v3" in sources[0]
    assert "pm-v1.5-full-sweep-gate-v3" in sources[1]
    assert "pm-v1.5-full-sweep-gate-v3" in sources[2]


def test_v1_5_evoemo_runner_verifies_its_live_response_mechanism_against_the_freeze() -> None:
    source = (
        ROOT / "scripts" / "v1_5" / "24_run_pm_v2_evoemo_v1_5.py"
    ).read_text(encoding="utf-8")
    assert "build_response_mechanism_contract" in source
    assert "require_matching_response_mechanism_contract" in source
    # The live contract must be checked against the frozen one, not rebuilt
    # from the frozen one's own values (which would make the check a no-op).
    assert 'contract.get("response_mechanism_contract")' in source


def test_formal_generation_separates_transport_retry_from_content_repair() -> None:
    script = (
        ROOT / "scripts" / "v1_5" / "20_generate_pm_v2_development_data_v1_5.py"
    ).read_text(encoding="utf-8")
    config = read_json(ROOT / "outputs" / "pm_v1_5_paid_run_release.json")
    pm_config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    execution = pm_config["formal_generation_execution"]
    assert execution["transport_retry"] == {
        "protocol": "pm-v1.5-bounded-retry-v4-independent-transport-format",
        "maximum_physical_attempts_per_content_attempt": 3,
        "transport_backoff_seconds": [10, 30],
        "retryable_classes": [
            "http_5xx",
            "network_timeout",
            "rate_limited_429",
            "request_timeout_408",
        ],
    }
    assert execution["budget_limits"] == {
        "max_api_calls": 2808,
        "max_estimated_usd": 3.0,
        "max_input_tokens_per_call": 12000,
    }
    assert "execute_with_bounded_retry(" in script
    assert "max_provider_output_attempts=1" in script
    assert "retry_class=\"content_lint_failure\"" in script
    assert "generation_transport_retry_summary" in script
    assert 'ROOT / "src" / "metacom_pm" / "bounded_retry.py"' in script
    assert 'ROOT / "src" / "metacom_pm" / "paid_run_release.py"' in script
    assert config["paid_execution_authorized"] is False


def test_central_paid_release_is_fail_closed_and_identity_bound(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "pm_v1_5.yaml"
    config_path.parent.mkdir(parents=True)
    manifest_path = tmp_path / "outputs" / "paid_release.json"
    blocked = {
        "release_revision": "pm-v1.5_1",
        "execution_release": {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "PAID_RUN_BLOCKED",
            "release_revision": "pm-v1.5_1",
            "approval_manifest": "outputs/paid_release.json",
        },
    }
    write_json(config_path, blocked)
    assert require_paid_run_release(
        blocked,
        config_path=config_path,
        stage="development_data_generation",
        run=False,
        run_identity=None,
    )["status"] == "DRY_RUN_ALLOWED"
    with pytest.raises(RuntimeError, match="centrally blocked"):
        require_paid_run_release(
            blocked,
            config_path=config_path,
            stage="development_data_generation",
            run=True,
            run_identity="fresh-cost-hash",
        )

    released = {
        **blocked,
        "execution_release": {
            **blocked["execution_release"],
            "status": "PAID_RUN_RELEASED",
        },
    }
    write_json(config_path, released)
    manifest_path.parent.mkdir(parents=True)
    write_json(
        manifest_path,
        {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "APPROVED",
            "release_revision": "pm-v1.5_1",
            "config_sha256": sha256_file(config_path),
            "stage_approvals": {
                "development_data_generation": "fresh-cost-hash"
            },
        },
    )
    assert require_paid_run_release(
        released,
        config_path=config_path,
        stage="development_data_generation",
        run=True,
        run_identity="fresh-cost-hash",
    )["status"] == "PAID_RUN_RELEASED"
    with pytest.raises(RuntimeError, match="stale"):
        require_paid_run_release(
            released,
            config_path=config_path,
            stage="development_data_generation",
            run=True,
            run_identity="different-hash",
        )
    consumed_manifest = read_json(manifest_path)
    consumed_manifest["stage_consumptions"] = {
        "development_data_generation": {
            "status": "CONSUMED_PASS",
            "approval_identity": "fresh-cost-hash",
        }
    }
    write_json(manifest_path, consumed_manifest)
    with pytest.raises(RuntimeError, match="already been consumed"):
        require_paid_run_release(
            released,
            config_path=config_path,
            stage="development_data_generation",
            run=True,
            run_identity="fresh-cost-hash",
        )
    # Consumption is identity-scoped, not a permanent ban on a scientific
    # stage. A code/config repair may proceed only with a newly reviewed dry
    # run identity, while the spent identity remains irrevocably blocked.
    consumed_manifest = read_json(manifest_path)
    consumed_manifest["stage_approvals"] = {
        "development_data_generation": "post-repair-cost-hash"
    }
    write_json(manifest_path, consumed_manifest)
    assert require_paid_run_release(
        released,
        config_path=config_path,
        stage="development_data_generation",
        run=True,
        run_identity="post-repair-cost-hash",
    )["status"] == "PAID_RUN_RELEASED"

    # Rotating a consumed stage into history must not make its old identity
    # reusable after the manifest is reopened for a later fresh paid stage.
    rotated_manifest = read_json(manifest_path)
    rotated_manifest["stage_consumptions_history"] = list(
        rotated_manifest["stage_consumptions"].values()
    )
    rotated_manifest["stage_consumptions"] = {}
    rotated_manifest["stage_approvals"] = {
        "development_data_generation": "fresh-cost-hash"
    }
    write_json(manifest_path, rotated_manifest)
    with pytest.raises(RuntimeError, match="already been consumed"):
        require_paid_run_release(
            released,
            config_path=config_path,
            stage="development_data_generation",
            run=True,
            run_identity="fresh-cost-hash",
        )

    rotated_manifest["stage_approvals"] = {
        "development_data_generation": "second-post-repair-cost-hash"
    }
    write_json(manifest_path, rotated_manifest)
    assert require_paid_run_release(
        released,
        config_path=config_path,
        stage="development_data_generation",
        run=True,
        run_identity="second-post-repair-cost-hash",
    )["status"] == "PAID_RUN_RELEASED"


def test_output_directory_previously_consumed_is_permanently_protected(
    tmp_path: Path,
) -> None:
    """Guards against the exact accident that destroyed the first
    ESConv-auxiliary generation pilot's real artifacts: a later dry-run for
    the same split reusing the same default output directory and deleting
    the earlier real run's files. Once a directory is recorded as a
    stage_consumptions (or historical) output_directory, no future script
    invocation -- dry-run or paid -- may target it again, regardless of
    whether its local ledger currently looks empty (exactly the state left
    behind by an external deletion)."""

    config_path = tmp_path / "configs" / "pm_v1_5.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "release_revision": "pm-v1.5_1",
        "execution_release": {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "PAID_RUN_RELEASED",
            "release_revision": "pm-v1.5_1",
            "approval_manifest": "outputs/paid_release.json",
        },
    }
    write_json(config_path, config)
    manifest_path = tmp_path / "outputs" / "paid_release.json"
    manifest_path.parent.mkdir(parents=True)

    protected_dir = tmp_path / "outputs" / "esconv_auxiliary_generation_v1_5_train"
    write_json(
        manifest_path,
        {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "CONSUMED_PASS",
            "release_revision": "pm-v1.5_1",
            "config_sha256": sha256_file(config_path),
            "stage_consumptions": {
                "esconv_auxiliary_generation_train": {
                    "status": "CONSUMED_PASS",
                    "output_directory": "outputs/esconv_auxiliary_generation_v1_5_train",
                }
            },
        },
    )
    # Directory need not even exist on disk (that is exactly the dangerous
    # post-deletion state) for the guard to fire.
    assert not protected_dir.exists()
    with pytest.raises(RuntimeError, match="permanently protected"):
        require_output_directory_not_previously_consumed(
            protected_dir, config=config, config_path=config_path
        )

    # A never-recorded directory is unaffected.
    require_output_directory_not_previously_consumed(
        tmp_path / "outputs" / "esconv_auxiliary_generation_v1_5_pilot_train",
        config=config,
        config_path=config_path,
    )

    # Historical (superseded) consumption records protect their directory
    # too, not just the current stage_consumptions entry.
    manifest = read_json(manifest_path)
    manifest["stage_consumptions"] = {}
    manifest["prior_stage_attempts_history"] = [
        {
            "status": "CONSUMED_PASS",
            "output_directory": "outputs/esconv_auxiliary_generation_v1_5_train",
        }
    ]
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="permanently protected"):
        require_output_directory_not_previously_consumed(
            protected_dir, config=config, config_path=config_path
        )

    # No manifest on disk yet -- nothing to protect against.
    manifest_path.unlink()
    require_output_directory_not_previously_consumed(
        protected_dir, config=config, config_path=config_path
    )


def test_resolve_first_unconsumed_output_directory_falls_back_to_a_retry_sibling(
    tmp_path: Path,
) -> None:
    """A legitimate second attempt at the same scope/split (e.g. a fresh
    full-scale run after a first one failed) must not require the operator
    to manually pick a new --out-root: the base name is tried first (so
    every already-registered directory keeps working unchanged), and only
    a consumed base name falls back to a __retry2, __retry3, ... sibling."""

    config_path = tmp_path / "configs" / "pm_v1_5.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "release_revision": "pm-v1.5_1",
        "execution_release": {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "PAID_RUN_RELEASED",
            "release_revision": "pm-v1.5_1",
            "approval_manifest": "outputs/paid_release.json",
        },
    }
    write_json(config_path, config)
    manifest_path = tmp_path / "outputs" / "paid_release.json"
    manifest_path.parent.mkdir(parents=True)

    base_dir = tmp_path / "outputs" / "esconv_auxiliary_generation_v1_5_full_train"

    # No manifest yet: the base name itself is returned, unchanged.
    resolved = resolve_first_unconsumed_output_directory(
        base_dir, config=config, config_path=config_path
    )
    assert resolved == base_dir

    # Base name already consumed by a first attempt -> falls back to
    # __retry2, without needing a manually chosen --out-root.
    write_json(
        manifest_path,
        {
            "protocol": PAID_RUN_RELEASE_PROTOCOL,
            "status": "CONSUMED_FAIL",
            "release_revision": "pm-v1.5_1",
            "config_sha256": sha256_file(config_path),
            "stage_consumptions": {
                "esconv_auxiliary_generation_full_train": {
                    "status": "CONSUMED_FAIL",
                    "output_directory": (
                        "outputs/esconv_auxiliary_generation_v1_5_full_train"
                    ),
                }
            },
        },
    )
    resolved = resolve_first_unconsumed_output_directory(
        base_dir, config=config, config_path=config_path
    )
    assert resolved == base_dir.with_name(f"{base_dir.name}__retry2")

    # __retry2 also consumed -> falls back further, to __retry3.
    manifest = read_json(manifest_path)
    manifest["stage_consumptions"]["esconv_auxiliary_generation_full_train_retry2"] = {
        "status": "CONSUMED_FAIL",
        "output_directory": (
            "outputs/esconv_auxiliary_generation_v1_5_full_train__retry2"
        ),
    }
    write_json(manifest_path, manifest)
    resolved = resolve_first_unconsumed_output_directory(
        base_dir, config=config, config_path=config_path
    )
    assert resolved == base_dir.with_name(f"{base_dir.name}__retry3")

    # Exhausting every candidate up to the cap fails closed, not silently.
    manifest = read_json(manifest_path)
    for attempt in range(3, 6):
        suffix = "" if attempt == 1 else f"__retry{attempt}"
        manifest["stage_consumptions"][f"probe_{attempt}"] = {
            "status": "CONSUMED_FAIL",
            "output_directory": (
                f"outputs/esconv_auxiliary_generation_v1_5_full_train{suffix}"
            ),
        }
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="already permanently protected"):
        resolve_first_unconsumed_output_directory(
            base_dir, config=config, config_path=config_path, max_attempts=5
        )


def test_source_disjoint_strategy_bank_keeps_all_strategy_families() -> None:
    selected_path = (
        ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl"
    )
    bank_path = ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    audit = read_json(
        ROOT / "data" / "strategy" / "strategy_bank_audit_v1_5.json"
    )
    selected = [str(row["dialogue_id"]) for row in iter_jsonl(selected_path)]
    cards = [dict(row) for row in iter_jsonl(bank_path)]
    source_ids = {str(row["source_dialogue_id"]) for row in cards}
    families = {str(row["strategy_label"]) for row in cards}

    assert len(selected) == len(set(selected)) == 52
    assert set(selected).isdisjoint(source_ids)
    assert len(cards) == 11_590
    assert len(source_ids) == 823
    assert len(families) == 8
    exclusion = audit["development_seed_exclusion"]
    assert exclusion["excluded_source_ids"] == sorted(selected)
    assert exclusion["bank_source_intersection"] == []
    assert audit["strategy_family_card_counts"]
    assert all(int(count) > 0 for count in audit["strategy_family_card_counts"].values())

    by_id = {str(row["strategy_id"]): row for row in cards}
    review_cards = [by_id[value] for value in V1_5_REVIEW_STRATEGY_CARD_IDS.values()]
    assert len(review_cards) == len(V1_5_REVIEW_STRATEGY_CARD_IDS) == 11
    assert len({row["source_dialogue_id"] for row in review_cards}) == 11
    assert {row["source_dialogue_id"] for row in review_cards}.isdisjoint(selected)
    role_labels = {
        "gentle_question": "Question",
        "open_restatement": "Restatement or Paraphrasing",
        "stress_reflection": "Reflection of feelings",
        "exam_reflection": "Reflection of feelings",
        "one_problem_suggestion": "Providing Suggestions",
        "social_connection_suggestion": "Providing Suggestions",
        "low_pressure_connection": "Providing Suggestions",
        "intrusive_long_plan": "Providing Suggestions",
        "video_call_self_disclosure": "Self-disclosure",
        "support_affirmation": "Affirmation and Reassurance",
        "job_information": "Information",
    }
    assert {
        role: by_id[card_id]["strategy_label"]
        for role, card_id in V1_5_REVIEW_STRATEGY_CARD_IDS.items()
    } == role_labels


def test_evoemo_chronology_is_timestamp_normalized_and_refs_fail_closed() -> None:
    raw = [
        {
            "id": "u1",
            "dialog_history": [
                {"id": "later", "timestamp": "2024-02-01"},
                {"id": "earlier", "timestamp": "2024-01-01"},
            ],
            "subsequent_topics": [
                {"idx": 0, "related_sessions": ["earlier", "later"]}
            ],
        }
    ]
    normalized, report = normalize_evoemo_chronology(raw)
    assert [row["id"] for row in normalized[0]["dialog_history"]] == [
        "earlier",
        "later",
    ]
    assert report["status"] == "PASS_WITH_FROZEN_TIMESTAMP_NORMALIZATION"
    assert report["reordered_user_count"] == 1

    raw[0]["subsequent_topics"][0]["related_sessions"] = ["missing"]
    with pytest.raises(RuntimeError, match="reference missing"):
        normalize_evoemo_chronology(raw)

    corpus = evoemo_chronology_audit(ROOT / "data" / "external" / "evo_emo.json")
    assert corpus["users"] == 18
    assert corpus["sessions"] == 401
    assert corpus["topics"] == 34
    assert corpus["reordered_user_count"] == 6
    assert corpus["related_session_references_checked"] == 149


def test_actual_corpus_fallback_gate_is_split_specific_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = [
        SimpleNamespace(
            user_id="train_user",
            cases=[object(), object()],
            provenance={"surface_selection": {"fallback_cases": ["one"]}},
        ),
        SimpleNamespace(
            user_id="internal_user",
            cases=[object()],
            provenance={"surface_selection": {"fallback_cases": []}},
        ),
    ]
    monkeypatch.setattr(
        "metacom_pm.v1_5_actual_corpus_review.load_bundles", lambda _: bundles
    )
    split_by_user = {
        "train_user": "train",
        "internal_user": "internal_test",
    }
    passed = _surface_fallback_report(
        bundles_path="unused.jsonl",
        split_by_user=split_by_user,
        maximum_fallback_rate_by_split={
            "train": 0.5,
            "calibration": 0.0,
            "internal_test": 0.0,
        },
    )
    assert passed["status"] == "PASS"
    failed = _surface_fallback_report(
        bundles_path="unused.jsonl",
        split_by_user=split_by_user,
        maximum_fallback_rate_by_split={
            "train": 0.49,
            "calibration": 0.0,
            "internal_test": 0.0,
        },
    )
    assert failed["status"] == "FAIL"


def test_actual_control_matrix_is_atomic_semantic_only() -> None:
    items = []
    for index in range(24):
        regime = "strategy_helpful" if index % 2 == 0 else "strategy_harmful"
        payload = {
            "semantic_family": f"family_{index % 3}",
            "regime": regime,
            "needed_memory_sources": ["MP"] if index % 2 == 0 else [],
            "advice_readiness": "explore_first" if index % 2 == 0 else "listen_only",
            "split": "train",
            "session_index": 4,
            "history": [
                {"role": "user", "content": f"earlier concern {index}"},
                {"role": "assistant", "content": "I hear you."},
            ],
            "current_user_text": f"current concern {index}",
            "session_summary": f"grounded summary {index}",
            "authorized_user_context": f"authorized context {index}",
            "coverage_rationale": "The visible conversation supports the case.",
            "memory_evidence": [
                {
                    "memory_id": f"mem_{index:012x}",
                    "source": "MP",
                    "utility": "helpful" if index % 2 == 0 else "irrelevant",
                    "created_session": 2,
                    "age": 2,
                    "stale": False,
                    "conflict": False,
                    "text": f"preference {index}",
                }
            ],
            "strategy_resource_candidate": "use" if index % 2 == 0 else "skip",
            "strategy_cards": [
                {
                    "strategy_id": f"strat_{index:012x}",
                    "strategy_label": "Reflection",
                    "guidance_text": "Reflect the feeling gently.",
                    "example_response": "That sounds difficult.",
                    "source_dialogue_id": f"esconv_{index:04d}",
                }
            ],
        }
        items.append({"item_id": f"state_{index}", "payload": payload})
    controls = build_actual_corpus_controls(
        items,
        control_seed=20260716,
        required_control_fields=ACTUAL_SEMANTIC_FIELDS,
        controls_per_field=2,
    )
    assert len(controls) == 4
    assert {
        field: sum(row["rating_field"] == field for row in controls)
        for field in ACTUAL_SEMANTIC_FIELDS
    } == {field: 2 for field in ACTUAL_SEMANTIC_FIELDS}
    assert all(row["semantic_packet"]["expected_verdict"] == "not_supported" for row in controls)
    assert all(
        "expected_verdict" not in canonical_json(row["semantic_packet"]["messages"])
        for row in controls
    )
    assert all("deliberately_unrelated_control" not in row["case_text"] for row in controls)


def test_actual_deterministic_gate_owns_structure_and_arithmetic() -> None:
    payload = {
        "session_index": 4,
        "history": [
            {"role": "user", "content": "I have been unsure lately."},
            {"role": "assistant", "content": "What feels most uncertain?"},
        ],
        "current_user_text": "I mainly need to talk it through.",
        "session_summary": "The user is uncertain and wants to talk.",
        "authorized_user_context": "The user wants reflective support.",
        "memory_evidence": [
            {
                "memory_id": "mem_000000000001",
                "source": "MP",
                "created_session": 2,
                "age": 2,
            }
        ],
    }
    passed = evaluate_actual_deterministic_payload(item_id="s1", payload=payload)
    assert passed["status"] == "PASS"
    broken = dict(payload)
    broken["history"] = [
        {"role": "user", "content": payload["current_user_text"]}
    ]
    broken["memory_evidence"] = [
        {
            "memory_id": "mem_000000000001",
            "source": "MP",
            "created_session": 2,
            "age": 3,
        }
    ]
    failed = evaluate_actual_deterministic_payload(item_id="s1", payload=broken)
    assert failed["status"] == "FAIL"
    assert failed["fields"]["dialogue_temporal_order_match"] is False
    assert failed["fields"]["memory_age_design_match"] is False


def test_actual_panel_retains_disagreement_but_blocks_unanimous_rejection() -> None:
    families = ["judge_a", "judge_b"]
    states = []
    real_results = {}
    for index in range(468):
        packets = []
        for field in ACTUAL_SEMANTIC_FIELDS:
            packet_id = f"s{index}::{field}"
            packets.append(
                {
                    "item_id": packet_id,
                    "case_item_id": f"s{index}",
                    "field": field,
                }
            )
            real_results[packet_id] = {
                "judge_a": {"verdict": "supported", "citation_valid": True},
                "judge_b": {"verdict": "supported", "citation_valid": True},
            }
        states.append({"item_id": f"s{index}", "semantic_packets": packets})
    disagreement_id = "s0::context_grounding_match"
    real_results[disagreement_id]["judge_b"]["verdict"] = "not_supported"
    controls = []
    control_results = {}
    for field in ACTUAL_SEMANTIC_FIELDS:
        for replica in (1, 2):
            packet_id = f"control_{field}_{replica}::{field}"
            packet = {
                "item_id": packet_id,
                "case_item_id": "source",
                "field": field,
            }
            controls.append(
                {
                    "item_id": packet_id,
                    "case_item_id": "source",
                    "corrupted_field": field,
                    "rating_field": field,
                    "override": {},
                    "case_text": field,
                    "semantic_packet": packet,
                }
            )
            control_results[packet_id] = {
                "judge_a": {"verdict": "not_supported", "citation_valid": True},
                "judge_b": {"verdict": "not_supported", "citation_valid": True},
            }
    gate = aggregate_actual_corpus_gate(
        state_items=states,
        real_case_results=real_results,
        control_results=control_results,
        controls=controls,
        judge_family_names=families,
        corpus_audit={"deterministic_gate": {"status": "PASS"}},
    )
    assert gate["status"] == "PASS"
    assert [row["packet_id"] for row in gate["real_case_disagreements"]] == [
        disagreement_id
    ]
    real_results[disagreement_id]["judge_a"]["verdict"] = "not_supported"
    failed = aggregate_actual_corpus_gate(
        state_items=states,
        real_case_results=real_results,
        control_results=control_results,
        controls=controls,
        judge_family_names=families,
        corpus_audit={"deterministic_gate": {"status": "PASS"}},
    )
    assert failed["status"] == "FAIL"
    assert failed["real_case_rejections"][0]["packet_id"] == disagreement_id


def test_actual_468_gate_is_attested_and_binds_state_corpus(
    tmp_path: Path,
) -> None:
    states = tmp_path / "states.jsonl"
    states.write_text('{"state_id":"s1"}\n', encoding="utf-8")
    evaluator = tmp_path / "evaluator.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategy_bank = tmp_path / "strategy.jsonl"
    config = tmp_path / "pm.yaml"
    experiment = tmp_path / "experiment.yaml"
    endpoint_names = ["judge_a", "judge_b"]
    descriptors = [
        {
            "alias": "judge_a",
            "family": "family_a",
            "model": "model-a",
            "base_url": "https://a.example.invalid",
            "transport": "openai_chat_completions",
        },
        {
            "alias": "judge_b",
            "family": "family_b",
            "model": "model-b",
            "base_url": "https://b.example.invalid",
            "transport": "openai_chat_completions",
        },
    ]
    for path, text in (
        (evaluator, '{"state_id":"s1"}\n'),
        (backend, '{"card_id":"c1"}\n'),
        (strategy_bank, '{"strategy_id":"x"}\n'),
        (
            config,
            "version: pm-v1.5\nactual_corpus_semantic_audit:\n"
            "  judge_endpoints: [judge_a, judge_b]\n",
        ),
    ):
        path.write_text(text, encoding="utf-8")
    experiment_text = (
        "endpoints:\n"
        "  judge_a:\n"
        "    base_url: https://a.example.invalid\n"
        "    model: model-a\n"
        "    api_key_env: A_KEY\n"
        "    family: family_a\n"
        "  judge_b:\n"
        "    base_url: https://b.example.invalid\n"
        "    model: model-b\n"
        "    api_key_env: B_KEY\n"
        "    family: family_b\n"
    )
    experiment.write_text(experiment_text, encoding="utf-8")
    control_manifest = [
        {
            "item_id": str(index),
            "case_item_id": "case",
            "corrupted_field": ACTUAL_SEMANTIC_FIELDS[index // 2],
            "rating_field": ACTUAL_SEMANTIC_FIELDS[index // 2],
            "override": {},
            "case_text_sha256": f"{index:064x}",
        }
        for index in range(4)
    ]
    matrix_sha256 = sha256_text(canonical_json(control_manifest))
    report_path = tmp_path / "gate_report.json"
    write_json(
        report_path,
        {
            "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
            "status": "PASS",
            "n_real_cases": 468,
            "n_real_semantic_packets": 936,
            "control_protocol": ACTUAL_CORPUS_CONTROL_PROTOCOL,
            "panel_policy": ACTUAL_PANEL_POLICY,
            "citation_policy": ACTUAL_CITATION_POLICY,
            "deterministic_fields": list(ACTUAL_DETERMINISTIC_FIELDS),
            "semantic_fields": list(ACTUAL_SEMANTIC_FIELDS),
            "derived_or_construction_fields": list(
                ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS
            ),
            "required_control_fields": list(ACTUAL_SEMANTIC_FIELDS),
            "controls_per_field": 2,
            "n_controls": 4,
            "control_field_counts": {
                field: 2 for field in ACTUAL_SEMANTIC_FIELDS
            },
            "control_contract_errors": [],
            "control_matrix_sha256": matrix_sha256,
            "control_manifest": control_manifest,
            "control_catches": [{"item_id": str(index)} for index in range(4)],
            "control_disagreements": [],
            "control_misses": [],
            "real_case_rejections": [],
            "deterministic_gate": {"status": "PASS"},
            "judge_families": endpoint_names,
            "judge_endpoint_descriptors": descriptors,
            "corpus_audit": {
                "fallback_gate": {"status": "PASS"},
                "control_matrix_sha256": matrix_sha256,
                "input_hashes": {
                    "states": sha256_file(states),
                    "evaluator_contexts": sha256_file(evaluator),
                    "backend": sha256_file(backend),
                    "strategy_bank": sha256_file(strategy_bank),
                },
            },
        },
    )
    real_judgments = tmp_path / "real.json"
    control_judgments = tmp_path / "control.json"
    controls_path = tmp_path / "controls.json"
    ledger = tmp_path / "ledger.jsonl"
    write_json(real_judgments, {})
    write_json(control_judgments, {})
    write_json(controls_path, control_manifest)
    ledger.write_text("{}\n", encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation,
        stage=ACTUAL_CORPUS_REVIEW_STAGE,
        inputs={
            "experiment_config": experiment,
            "states": states,
            "evaluator_contexts": evaluator,
            "memory_backend": backend,
            "strategy_bank": strategy_bank,
            "pm_v1_5_config": config,
        },
        outputs={
            "real_case_judgments": (real_judgments, False),
            "control_judgments": (control_judgments, False),
            "controls": (controls_path, False),
            "gate_report": (report_path, False),
            "physical_attempt_ledger": (ledger, True),
        },
        parameters={
            "judge_endpoint_descriptors": descriptors,
            "panel_policy": ACTUAL_PANEL_POLICY,
            "citation_policy": ACTUAL_CITATION_POLICY,
        },
    )
    assert require_actual_corpus_semantic_review_pass(
        report_path,
        attestation,
        expected_experiment_config_path=experiment,
        expected_states_path=states,
        expected_evaluator_contexts_path=evaluator,
        expected_backend_path=backend,
        expected_strategy_bank_path=strategy_bank,
        expected_pm_config_path=config,
    )["status"] == "PASS"
    experiment.write_text(
        experiment_text.replace("model-a", "model-a-v2"), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash mismatch|does not bind"):
        require_actual_corpus_semantic_review_pass(
            report_path,
            attestation,
            expected_experiment_config_path=experiment,
            expected_states_path=states,
            expected_evaluator_contexts_path=evaluator,
            expected_backend_path=backend,
            expected_strategy_bank_path=strategy_bank,
            expected_pm_config_path=config,
        )
    experiment.write_text(experiment_text, encoding="utf-8")
    states.write_text('{"state_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch|did not PASS"):
        require_actual_corpus_semantic_review_pass(
            report_path,
            attestation,
            expected_experiment_config_path=experiment,
            expected_states_path=states,
            expected_evaluator_contexts_path=evaluator,
            expected_backend_path=backend,
            expected_strategy_bank_path=strategy_bank,
            expected_pm_config_path=config,
        )


def test_one_standard_error_rule_prefers_simpler_near_best_candidate() -> None:
    common = {
        "mean_quality": 0.8,
        "mean_risk": 0.1,
        "mean_observed_input_tokens": 100.0,
        "action_distribution_instability": 0.1,
    }
    raw_best = {
        **common,
        "algorithm": "complex",
        "priority": 0,
        "simplicity_rank": 2,
        "mean_realized_utility": 0.80,
        "mean_realized_utility_standard_error": 0.05,
    }
    simpler = {
        **common,
        "algorithm": "simple",
        "priority": 1,
        "simplicity_rank": 0,
        "mean_realized_utility": 0.77,
        "mean_realized_utility_standard_error": 0.01,
    }
    below_band = {
        **common,
        "algorithm": "too_weak",
        "priority": 2,
        "simplicity_rank": 0,
        "mean_realized_utility": 0.74,
        "mean_realized_utility_standard_error": 0.01,
    }
    best, selected, floor, candidates = _select_one_standard_error_candidate(
        [raw_best, simpler, below_band]
    )
    assert best["algorithm"] == "complex"
    assert floor == pytest.approx(0.75)
    assert {row["algorithm"] for row in candidates} == {"complex", "simple"}
    assert selected["algorithm"] == "simple"
