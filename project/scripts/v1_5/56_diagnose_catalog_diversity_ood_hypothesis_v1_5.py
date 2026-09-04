#!/usr/bin/env python3
"""Zero-API hypothesis test: does the training corpus's near-zero variance
in memory-catalog size/age explain EvoEmo's 100% severe metadata OOD rate
(see outputs/pm_v1_5_merged_ood_audit_v8_19_2_v3_production_shape.json)?

Background: a direct check found every one of the 468 synthetic training
cards has EXACTLY 2/2/2 items across MP/MS/ME (zero variance), and session
spans of only 3-11. EvoEmo's real per-user counts are MP=7 (fixed), MS
13-33, ME 37-109 -- organically wider, and PMV2FeatureBuilder's own math
explains why this matters: count_ratio = min(count, top_k)/top_k saturates
at 1.0 once count reaches MEMORY_TOP_K (MP=2, MS=2, ME=3), so training's
fixed count=2 already saturates MP/MS identically to EvoEmo (irrelevant),
but leaves ME's count_ratio stuck at 2/3=0.667 forever, while EvoEmo always
saturates it to 1.0 -- matching the OOD report's ME.count_ratio finding
exactly. age_ratio = min(age/session_index, 1) has the same shape for age:
training's narrow session spans keep every age_ratio moderate, while
EvoEmo's much longer real histories push it toward 1.0.

This script does NOT touch, regenerate, or modify the frozen memory_backend
or PMV2State artifacts. It builds an in-memory PERTURBED COPY of the
existing, already-validated synth_train/synth_cal PMV2State objects --
inventory count/age fields only, resampled from EvoEmo's REAL empirical
per-source distribution (computed directly from data/external/evo_emo.json,
not invented) -- refits PMV2FeatureBuilder on the perturbed copy, and reruns
the identical ood_report/calibrate_ood machinery script 41 uses against the
SAME real EvoEmo formal-turn states. This tests one specific, falsifiable
hypothesis (structural range coverage) in isolation; it is not a proposal
to ship this perturbed data as real training data (see the module's
companion findings doc for why: content/gold-binding is untouched by this
perturbation and would need real re-verification before any of this could
be a real training artifact).
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Any

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource, StrategyCard
from metacom_pm.evoemo import (
    NEUTRAL_INITIAL_GREETING,
    _fixed_context_before_turn,
    _load_fixed_tracks,
    build_evo_memory,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_centroid,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v1_5_step0 import build_strategy_family_catalog
from metacom_pm.pm_v2_contracts import ObservableSourceSummary, PMV2Split, PMV2State
from metacom_pm.pm_v2_data import load_states, runtime_to_pmv2_state
from metacom_pm.pm_v2_features import PMV2FeatureBuilder

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-catalog-diversity-ood-hypothesis-v1"

OOD_CALIBRATION_KWARGS = dict(
    semantic_false_positive_quantile=0.99,
    metadata_false_positive_quantile=0.99,
    maximum_joint_in_distribution_fallback_rate=0.05,
    minimum_semantic_challenge_detection_rate=0.80,
    minimum_metadata_challenge_detection_rate=0.95,
)


def _real_evoemo_count_distribution(evoemo_path: Path) -> dict[MemorySource, list[int]]:
    """Real per-user item counts per source, straight from the same data
    used everywhere else this session -- not invented reference numbers."""

    users = load_evoemo(evoemo_path)
    counts: dict[MemorySource, list[int]] = {s: [] for s in MemorySource}
    for user in users:
        items, _ = build_evo_memory(user)
        for source in MemorySource:
            counts[source].append(sum(1 for it in items if it.source is source))
    return counts


def _perturb_inventory(
    state: PMV2State, *, count_pool: dict[MemorySource, list[int]], rng: random.Random
) -> PMV2State:
    """Resample this state's count/age metadata from EvoEmo's real
    empirical distribution, holding everything else (semantic similarity,
    representation validity, token estimate shape) fixed. Deterministic
    given rng seeded from state_id."""

    new_inventory: dict[MemorySource, ObservableSourceSummary] = {}
    for source, cat in state.inventory.items():
        if not cat.available or source not in count_pool or not count_pool[source]:
            new_inventory[source] = cat
            continue
        new_count = rng.choice(count_pool[source])
        # Real EvoEmo age ratios span close to the full [0, session_index]
        # range (this session's own OOD report: up to 1.0 vs training's
        # <=0.55). Sample a plausible triple directly in ratio space, which
        # is exactly the space PMV2FeatureBuilder's age_ratio() operates in.
        session_index = max(int(state.session_index), 1)
        ratios = sorted(rng.uniform(0.05, 1.0) for _ in range(3))
        min_age = max(0, round(ratios[0] * session_index))
        median_age = max(min_age, round(ratios[1] * session_index))
        max_age = max(median_age, round(ratios[2] * session_index))
        mean_item_tokens = (cat.estimated_tokens / cat.count) if cat.count else 40.0
        new_inventory[source] = cat.model_copy(
            update={
                "count": new_count,
                "min_age_sessions": min_age,
                "median_age_sessions": float(median_age),
                "max_age_sessions": max_age,
                "estimated_tokens": int(round(mean_item_tokens * new_count)),
            }
        )
    # BUG FOUND AND FIXED (2026-08-06, same run): PMV2State.step0_observation
    # is precomputed and cached on every loaded state, and PMV2FeatureBuilder.
    # _step0() returns that cache directly when present -- it never
    # recomputes from state.inventory. The first version of this script only
    # perturbed inventory, so the perturbation was silently inert: the fitted
    # metadata_ood_threshold came back byte-identical between "baseline" and
    # "diversified" (0.00340909091009091 both times), which is what exposed
    # this. Clearing step0_observation forces _step0() down its fallback
    # path, which calls build_step0_observation() fresh from the (now
    # perturbed) inventory -- confirmed this is the only place count/age
    # metadata actually reaches the fitted feature vector.
    return state.model_copy(update={"inventory": new_inventory, "step0_observation": None})


def _diversify(
    states: list[PMV2State], *, count_pool: dict[MemorySource, list[int]]
) -> list[PMV2State]:
    out = []
    for state in states:
        rng = random.Random(f"{PROTOCOL}|{state.state_id}")
        out.append(_perturb_inventory(state, count_pool=count_pool, rng=rng))
    return out


def _domain_report(
    builder: PMV2FeatureBuilder, states: list[PMV2State], *, domain_label: str
) -> dict[str, Any]:
    if not states:
        return {"domain": domain_label, "n_states": 0, "n_severe": 0, "severe_rate": None}
    n_severe_semantic = n_severe_metadata = n_severe_either = 0
    for state in states:
        report = builder.ood_report(state)
        n_severe_semantic += bool(report["severe_semantic_ood"])
        n_severe_metadata += bool(report["severe_metadata_ood"])
        n_severe_either += bool(report["severe_semantic_ood"] or report["severe_metadata_ood"])
    n = len(states)
    return {
        "domain": domain_label,
        "n_states": n,
        "n_severe_semantic_ood": n_severe_semantic,
        "n_severe_metadata_ood": n_severe_metadata,
        "n_severe_either": n_severe_either,
        "severe_rate": n_severe_either / n,
        "severe_semantic_rate": n_severe_semantic / n,
        "severe_metadata_rate": n_severe_metadata / n,
    }


def _build_evoemo_formal_turn_states(
    *, evoemo_path: Path, fixed_tracks_path: Path, turn_indices: list[int], condition: str,
    encoder: FrozenTransformerSemanticEncoder, strategy_catalog_count: int,
    strategy_estimated_tokens: int, strategy_family_catalog,
) -> tuple[list[PMV2State], dict[str, Any]]:
    """Copied from 41_diagnose_merged_ood_audit_v1_5.py (same real
    production path, same frozen formal turns) -- not modified."""

    users = load_evoemo(evoemo_path)
    users_by_id = {str(user["id"]): user for user in users}
    total_scenarios = sum(len(user.get("subsequent_topics") or []) for user in users)
    tracks = _load_fixed_tracks(fixed_tracks_path) if fixed_tracks_path.is_file() else {}
    memory_cache: dict[str, list] = {}
    states: list[PMV2State] = []
    covered_scenarios: set[tuple[str, int]] = set()
    for (user_id, topic_index, seed, simulator_id), track in sorted(tracks.items()):
        user = users_by_id.get(user_id)
        if user is None:
            continue
        topic = next(
            (t for t in user.get("subsequent_topics") or [] if int(t["idx"]) == topic_index), None
        )
        if topic is None:
            continue
        if user_id not in memory_cache:
            items, _ = build_evo_memory(user)
            memory_cache[user_id] = items
        items = memory_cache[user_id]
        source_centroids = {
            source: semantic_centroid(encoder, [item.text for item in items if item.source is source])
            for source in MemorySource
        }
        for turn_index in turn_indices:
            if turn_index > len(track.get("seeker_turns") or []):
                continue
            seeker_message = track["seeker_turns"][turn_index - 1]
            state_context = _fixed_context_before_turn(track, turn_index)
            runtime_state = make_evo_runtime_state(
                user, topic, state_context, seeker_message, items, turn_index, condition,
                track_id=str(track["track_id"]), fixed_open_loop=True,
                semantic_encoder=encoder, semantic_source_centroids=source_centroids,
            )
            pm_state = runtime_to_pmv2_state(
                runtime_state, split=PMV2Split.EXTERNAL_TEST,
                strategy_catalog_count=strategy_catalog_count,
                strategy_estimated_tokens=strategy_estimated_tokens,
                strategy_family_catalog=strategy_family_catalog, semantic_encoder=encoder,
            )
            states.append(pm_state)
        covered_scenarios.add((user_id, topic_index))
    return states, {
        "n_total_scenarios": total_scenarios,
        "n_scenarios_with_at_least_one_complete_track": len(covered_scenarios),
        "n_complete_tracks_found": len(tracks),
        "n_formal_turn_states": len(states),
    }


def main() -> None:
    pm_v1_5_config = ROOT / "configs" / "pm_v1_5.yaml"
    synthetic_states = (
        ROOT / "data" / "pm_v1_5_formal_v8_18_duplicate_repair_candidate" / "pm_v2_states.jsonl"
    )
    evoemo_path = ROOT / "data" / "external" / "evo_emo.json"
    strategy_bank = ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    fixed_tracks_path = (
        ROOT / "outputs" / "evoemo_fixed_tracks_v1_5_v2_candidate" / "fixed_seeker_tracks.jsonl"
    )
    out_path = ROOT / "outputs" / "pm_v1_5_catalog_diversity_ood_hypothesis_v1.json"

    config = load_config(pm_v1_5_config)
    encoder = FrozenTransformerSemanticEncoder.load(semantic_encoder_spec_from_config(config))
    require_semantic_runtime_contract(config, encoder)
    external_evaluation = dict(config.get("external_evaluation") or {})
    turn_indices = list(external_evaluation.get("turn_indices") or [3, 8])
    strategy_estimated_tokens = int(external_evaluation.get("strategy_action_tokens") or 260)
    strategy_cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank)]
    strategy_family_catalog = build_strategy_family_catalog(
        strategy_cards, require_all_families=True, semantic_encoder=encoder
    )

    print("loading synthetic train/calibration states...")
    synth = load_states(synthetic_states)
    synth_train = [s for s in synth if s.split == PMV2Split.TRAIN]
    synth_cal = [s for s in synth if s.split == PMV2Split.CALIBRATION]
    print(f"  n_train={len(synth_train)} n_cal={len(synth_cal)}")

    print("building real EvoEmo formal-turn states (slow: semantic encoding)...")
    evoemo_states, evoemo_meta = _build_evoemo_formal_turn_states(
        evoemo_path=evoemo_path, fixed_tracks_path=fixed_tracks_path, turn_indices=turn_indices,
        condition="ood_preflight", encoder=encoder, strategy_catalog_count=len(strategy_cards),
        strategy_estimated_tokens=strategy_estimated_tokens,
        strategy_family_catalog=strategy_family_catalog,
    )
    print(f"  n_evoemo_formal_states={len(evoemo_states)} coverage={evoemo_meta}")

    print("computing real EvoEmo per-source count distribution...")
    count_pool = _real_evoemo_count_distribution(evoemo_path)
    for source, counts in count_pool.items():
        print(f"  {source.value}: {sorted(counts)}")

    print("fitting BASELINE (unperturbed) feature builder...")
    baseline = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(synth_train)
    baseline_cal = baseline.calibrate_ood(synth_cal, **OOD_CALIBRATION_KWARGS)
    baseline_evoemo = _domain_report(baseline, evoemo_states, domain_label="evoemo_formal_turn_states")

    print("diversifying synthetic train/cal inventory metadata from real EvoEmo counts/ages...")
    diversified_train = _diversify(synth_train, count_pool=count_pool)
    diversified_cal = _diversify(synth_cal, count_pool=count_pool)

    print("fitting DIVERSIFIED feature builder...")
    diversified = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(diversified_train)
    diversified_cal_report = diversified.calibrate_ood(diversified_cal, **OOD_CALIBRATION_KWARGS)
    diversified_evoemo = _domain_report(
        diversified, evoemo_states, domain_label="evoemo_formal_turn_states"
    )

    report = {
        "protocol": PROTOCOL,
        "note": (
            "Diagnostic hypothesis test only. inventory count/age fields on synth_train/"
            "synth_cal are perturbed in-memory from EvoEmo's real per-source count "
            "distribution; no frozen artifact is modified; content/gold-binding is NOT "
            "reverified under the new counts -- this measures only whether closing the "
            "structural range gap changes the OOD flag, not whether the perturbed states "
            "would be valid real training data."
        ),
        "input_hashes": {
            "synthetic_states_sha256": sha256_file(synthetic_states),
            "evoemo_sha256": sha256_file(evoemo_path),
        },
        "evoemo_coverage": evoemo_meta,
        "real_evoemo_count_distribution": {
            s.value: sorted(c) for s, c in count_pool.items()
        },
        "baseline": {
            "calibration_report": baseline_cal,
            "evoemo_formal_turn_states": baseline_evoemo,
        },
        "diversified": {
            "calibration_report": diversified_cal_report,
            "evoemo_formal_turn_states": diversified_evoemo,
        },
        "conclusion": {
            "baseline_evoemo_severe_metadata_rate": baseline_evoemo["severe_metadata_rate"],
            "diversified_evoemo_severe_metadata_rate": diversified_evoemo["severe_metadata_rate"],
            "hypothesis_supported": (
                diversified_evoemo["severe_metadata_rate"]
                < baseline_evoemo["severe_metadata_rate"]
            ),
        },
    }
    write_json(out_path, report)
    print("\n" + canonical_json(report["conclusion"]))
    print(f"full report written to {out_path}")


if __name__ == "__main__":
    main()
