#!/usr/bin/env python3
"""Zero-API OOD audit: does merging ESConv-auxiliary into training fix
ESConv-test's OOD fallback, and does it touch EvoEmo's (it should not)?

Background: this project's synthetic longitudinal training corpus has
literally zero variance in memory-source counts (always exactly 2 per
source) and a narrow session_index range (20-44). ESConv-test states are
structurally memory-less (no cross-session memory ever available) and were
found severely out-of-distribution under PMV2FeatureBuilder at 100% before
any ESConv-auxiliary training exposure. EvoEmo states have real, available
memory but at scales (7-109 items) and ages (age/session_index ratios up to
1.0, vs training's <=0.55) never seen in training -- a *different* kind of
distribution shift than ESConv's. Merging ESConv-auxiliary states (which are
also memory-less) into training was hypothesized -- and is independently
confirmed here -- to fix ESConv-test's OOD rate without doing anything for
EvoEmo's, since the two domains fail for different reasons.

Never reads ESConv/EvoEmo outcome, judge, or gold fields -- only observable
PMV2State construction (states, not labels), so this audit cannot be
influenced by, and does not need, any real generation or judging.

EvoEmo coverage is intentionally reported at two honesty levels:
  - "formal_turn_states": states built via the real production path
    (make_evo_runtime_state -> runtime_to_pmv2_state) from *real* fixed
    seeker tracks, restricted to the frozen formal evaluation turns
    (external_evaluation.evoemo.turn_indices, e.g. [3, 8]). Only as many
    scenarios as currently have a complete real track are covered -- this
    audit reports that coverage honestly (e.g. 5/102 tracks) rather than
    silently extrapolating from a partial sample.
  - "single_turn_preflight_states": one canned turn-1 state per (user,
    topic) scenario (34 total), mirroring the legacy
    evoemo._external_ood_preflight's sampling shape but using the real
    PMV2FeatureBuilder. Broader scenario coverage, shallower per-scenario
    fidelity -- explicitly labeled non-formal.

This script only ever fits/calibrates PMV2FeatureBuilder and calls
ood_report/metadata_ood_breakdown -- it never trains or freezes a PM
checkpoint and never authorizes any paid stage.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource
from metacom_pm.evoemo import (
    NEUTRAL_INITIAL_GREETING,
    _fixed_context_before_turn,
    _load_fixed_tracks,
    build_evo_memory,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.io import iter_jsonl, sha256_file, sha256_text, canonical_json, write_json
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v2_contracts import PMV2Split, PMV2State
from metacom_pm.pm_v2_data import load_states, runtime_to_pmv2_state
from metacom_pm.pm_v2_features import PMV2FeatureBuilder

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PROTOCOL = "pm-v1.5-merged-ood-audit-v1"

OOD_CALIBRATION_KWARGS = dict(
    semantic_false_positive_quantile=0.99,
    metadata_false_positive_quantile=0.99,
    maximum_joint_in_distribution_fallback_rate=0.05,
    minimum_semantic_challenge_detection_rate=0.80,
    minimum_metadata_challenge_detection_rate=0.95,
)


def _dimension_category(name: str) -> str:
    if name == "session_index":
        return "age_staleness_drift"
    if name.startswith("strategy."):
        return "strategy_drift"
    if any(
        name.endswith(suffix)
        for suffix in ("age_ratio", "age_span_ratio")
    ):
        return "age_staleness_drift"
    if name.endswith("count_ratio") or name.endswith("expected_tokens_frac"):
        return "catalog_tail_drift"
    return "other_metadata"


def _domain_report(
    builder: PMV2FeatureBuilder, states: list[PMV2State], *, domain_label: str
) -> dict[str, Any]:
    if not states:
        return {
            "domain": domain_label,
            "n_states": 0,
            "n_severe": 0,
            "severe_rate": None,
        }
    n_severe_semantic = 0
    n_severe_metadata = 0
    n_severe_either = 0
    dimension_totals: dict[str, float] = collections.defaultdict(float)
    dimension_severe_counts: dict[str, int] = collections.defaultdict(int)
    for state in states:
        report = builder.ood_report(state)
        breakdown = builder.metadata_ood_breakdown(state)
        n_severe_semantic += bool(report["severe_semantic_ood"])
        n_severe_metadata += bool(report["severe_metadata_ood"])
        n_severe_either += bool(
            report["severe_semantic_ood"] or report["severe_metadata_ood"]
        )
        for name, value in breakdown.items():
            dimension_totals[name] += value
            if value > 0.0:
                dimension_severe_counts[name] += 1
    n = len(states)
    mean_dimension = {
        name: dimension_totals[name] / n for name in dimension_totals
    }
    category_totals: dict[str, float] = collections.defaultdict(float)
    for name, mean_value in mean_dimension.items():
        category_totals[_dimension_category(name)] += mean_value
    top_dimensions = sorted(
        mean_dimension.items(), key=lambda kv: -kv[1]
    )[:10]
    return {
        "domain": domain_label,
        "n_states": n,
        "n_severe_semantic_ood": n_severe_semantic,
        "n_severe_metadata_ood": n_severe_metadata,
        "n_severe_either": n_severe_either,
        "severe_rate": n_severe_either / n,
        "severe_semantic_rate": n_severe_semantic / n,
        "severe_metadata_rate": n_severe_metadata / n,
        # Sum (not mean) of each category's constituent dimension means --
        # metadata_ood_score is the mean over ALL dimensions, so
        # sum(category totals) / len(all dimensions) reconstructs it exactly.
        # A category with more constituent dimensions naturally sums higher;
        # this shows each category's share of the aggregate, not "how bad a
        # typical dimension in it is" (see top_metadata_dimensions below for
        # that).
        "metadata_ood_contribution_sum_by_category": dict(
            sorted(category_totals.items(), key=lambda kv: -kv[1])
        ),
        "top_metadata_dimensions_by_mean_contribution": [
            {"dimension": name, "mean_contribution": value}
            for name, value in top_dimensions
        ],
    }


def _build_evoemo_states(
    *,
    evoemo_path: Path,
    encoder: FrozenTransformerSemanticEncoder,
) -> tuple[list[PMV2State], dict[str, Any]]:
    """Single-turn-preflight states: one canned turn-1 state per (user,
    topic) scenario. Broader coverage, non-formal -- see module docstring."""

    users = load_evoemo(evoemo_path)
    states: list[PMV2State] = []
    for user in users:
        items, _ = build_evo_memory(user)
        for topic in user.get("subsequent_topics") or []:
            runtime_state = make_evo_runtime_state(
                user,
                topic,
                [{"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}],
                "I want to talk about what has been happening.",
                items,
                1,
                "ood_preflight",
                track_id="ood_preflight",
                semantic_encoder=encoder,
            )
            pm_state = runtime_to_pmv2_state(
                runtime_state, split=PMV2Split.EXTERNAL_TEST, semantic_encoder=encoder
            )
            states.append(pm_state)
    total_scenarios = sum(
        len(user.get("subsequent_topics") or []) for user in users
    )
    return states, {
        "n_users": len(users),
        "n_scenarios": total_scenarios,
        "n_states": len(states),
    }


def _build_evoemo_formal_turn_states(
    *,
    evoemo_path: Path,
    fixed_tracks_path: Path,
    turn_indices: list[int],
    condition: str,
    encoder: FrozenTransformerSemanticEncoder,
) -> tuple[list[PMV2State], dict[str, Any]]:
    """Real production-path states (make_evo_runtime_state ->
    runtime_to_pmv2_state) restricted to the frozen formal evaluation turns,
    built only from real, complete fixed seeker tracks -- coverage is
    whatever currently exists on disk, reported honestly, never padded."""

    users = load_evoemo(evoemo_path)
    users_by_id = {str(user["id"]): user for user in users}
    total_scenarios = sum(
        len(user.get("subsequent_topics") or []) for user in users
    )
    tracks = _load_fixed_tracks(fixed_tracks_path) if fixed_tracks_path.is_file() else {}
    memory_cache: dict[str, list] = {}
    states: list[PMV2State] = []
    covered_scenarios: set[tuple[str, int]] = set()
    for (user_id, topic_index, seed, simulator_id), track in sorted(tracks.items()):
        user = users_by_id.get(user_id)
        if user is None:
            continue
        topic = next(
            (t for t in user.get("subsequent_topics") or [] if int(t["idx"]) == topic_index),
            None,
        )
        if topic is None:
            continue
        if user_id not in memory_cache:
            items, _ = build_evo_memory(user)
            memory_cache[user_id] = items
        items = memory_cache[user_id]
        for turn_index in turn_indices:
            if turn_index > len(track.get("seeker_turns") or []):
                continue
            seeker_message = track["seeker_turns"][turn_index - 1]
            state_context = _fixed_context_before_turn(track, turn_index)
            runtime_state = make_evo_runtime_state(
                user,
                topic,
                state_context,
                seeker_message,
                items,
                turn_index,
                condition,
                track_id=str(track["track_id"]),
                fixed_open_loop=True,
                semantic_encoder=encoder,
            )
            pm_state = runtime_to_pmv2_state(
                runtime_state, split=PMV2Split.EXTERNAL_TEST, semantic_encoder=encoder
            )
            states.append(pm_state)
        covered_scenarios.add((user_id, topic_index))
    return states, {
        "n_total_scenarios": total_scenarios,
        "n_scenarios_with_at_least_one_complete_track": len(covered_scenarios),
        "n_complete_tracks_found": len(tracks),
        "turn_indices": turn_indices,
        "n_formal_turn_states": len(states),
        "expected_formal_turn_states_at_full_coverage": (
            total_scenarios * len(turn_indices)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--synthetic-states",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"
        / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--esconv-auxiliary-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5",
    )
    parser.add_argument(
        "--esconv-test-states",
        type=Path,
        default=ROOT / "data" / "esconv_test_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--evoemo-path", type=Path, default=ROOT / "data" / "external" / "evo_emo.json"
    )
    parser.add_argument(
        "--fixed-tracks-path",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v1_5_v2_candidate"
        / "fixed_seeker_tracks.jsonl",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_merged_ood_audit.json",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    require_semantic_runtime_contract(config, encoder)
    external_evaluation = dict(config.get("external_evaluation") or {})
    turn_indices = list(external_evaluation.get("turn_indices") or [3, 8])
    condition = "ood_preflight"

    synth = load_states(args.synthetic_states)
    synth_train = [s for s in synth if s.split == PMV2Split.TRAIN]
    synth_cal = [s for s in synth if s.split == PMV2Split.CALIBRATION]
    aux_train = load_states(args.esconv_auxiliary_dir / "train" / "pm_v2_states.jsonl")
    aux_cal = load_states(args.esconv_auxiliary_dir / "calibration" / "pm_v2_states.jsonl")
    esconv_test_states = load_states(args.esconv_test_states)

    baseline = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(synth_train)
    baseline_cal_report = baseline.calibrate_ood(synth_cal, **OOD_CALIBRATION_KWARGS)

    merged = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(
        synth_train + aux_train
    )
    merged_cal_report = merged.calibrate_ood(synth_cal + aux_cal, **OOD_CALIBRATION_KWARGS)

    evoemo_single_turn_states, evoemo_single_turn_meta = _build_evoemo_states(
        evoemo_path=args.evoemo_path, encoder=encoder
    )
    evoemo_formal_states, evoemo_formal_meta = _build_evoemo_formal_turn_states(
        evoemo_path=args.evoemo_path,
        fixed_tracks_path=args.fixed_tracks_path,
        turn_indices=turn_indices,
        condition=condition,
        encoder=encoder,
    )

    report = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "input_hashes": {
            "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
            "synthetic_states_sha256": sha256_file(args.synthetic_states),
            "esconv_auxiliary_train_sha256": sha256_file(
                args.esconv_auxiliary_dir / "train" / "pm_v2_states.jsonl"
            ),
            "esconv_auxiliary_calibration_sha256": sha256_file(
                args.esconv_auxiliary_dir / "calibration" / "pm_v2_states.jsonl"
            ),
            "esconv_test_states_sha256": sha256_file(args.esconv_test_states),
            "evoemo_sha256": sha256_file(args.evoemo_path),
            "fixed_tracks_sha256": (
                sha256_file(args.fixed_tracks_path)
                if args.fixed_tracks_path.is_file()
                else None
            ),
        },
        "ood_calibration_settings": OOD_CALIBRATION_KWARGS,
        "baseline_synthetic_only": {
            "n_train": len(synth_train),
            "n_calibration": len(synth_cal),
            "calibration_report": baseline_cal_report,
            "esconv_test": _domain_report(
                baseline, esconv_test_states, domain_label="esconv_test"
            ),
            "evoemo_single_turn_preflight": {
                **_domain_report(
                    baseline,
                    evoemo_single_turn_states,
                    domain_label="evoemo_single_turn_preflight",
                ),
                "coverage": evoemo_single_turn_meta,
                "formal_result": False,
            },
        },
        "merged_synthetic_plus_esconv_auxiliary": {
            "n_train": len(synth_train) + len(aux_train),
            "n_calibration": len(synth_cal) + len(aux_cal),
            "calibration_report": merged_cal_report,
            "esconv_test": _domain_report(
                merged, esconv_test_states, domain_label="esconv_test"
            ),
            "evoemo_single_turn_preflight": {
                **_domain_report(
                    merged,
                    evoemo_single_turn_states,
                    domain_label="evoemo_single_turn_preflight",
                ),
                "coverage": evoemo_single_turn_meta,
                "formal_result": False,
            },
            "evoemo_formal_turn_states": {
                **_domain_report(
                    merged, evoemo_formal_states, domain_label="evoemo_formal_turn_states"
                ),
                "coverage": evoemo_formal_meta,
                "formal_result": (
                    evoemo_formal_meta["n_scenarios_with_at_least_one_complete_track"]
                    == evoemo_formal_meta["n_total_scenarios"]
                ),
                "note": (
                    "formal_result is only True once every scenario has a "
                    "complete fixed seeker track; partial coverage is "
                    "reported honestly, not extrapolated."
                ),
            },
        },
    }
    report["conclusion"] = {
        "esconv_test_severe_rate_baseline": report["baseline_synthetic_only"][
            "esconv_test"
        ]["severe_rate"],
        "esconv_test_severe_rate_merged": report["merged_synthetic_plus_esconv_auxiliary"][
            "esconv_test"
        ]["severe_rate"],
        "evoemo_single_turn_severe_rate_baseline": report["baseline_synthetic_only"][
            "evoemo_single_turn_preflight"
        ]["severe_rate"],
        "evoemo_single_turn_severe_rate_merged": report[
            "merged_synthetic_plus_esconv_auxiliary"
        ]["evoemo_single_turn_preflight"]["severe_rate"],
        "evoemo_formal_turn_severe_rate_merged": report[
            "merged_synthetic_plus_esconv_auxiliary"
        ]["evoemo_formal_turn_states"]["severe_rate"],
        "evoemo_formal_turn_coverage_note": (
            f"{evoemo_formal_meta['n_scenarios_with_at_least_one_complete_track']}/"
            f"{evoemo_formal_meta['n_total_scenarios']} scenarios covered by real "
            "fixed seeker tracks; NOT the full formal 68-state audit until all "
            "scenarios have complete tracks"
        ),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_path, report)
    print(canonical_json(report["conclusion"]))
    print(f"full report written to {args.out_path}")


if __name__ == "__main__":
    main()
