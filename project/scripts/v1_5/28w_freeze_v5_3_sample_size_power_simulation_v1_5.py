#!/usr/bin/env python3
"""Zero-API V5.3 sample-size power simulation (plan section 4.2).

V5.2's confirmation stage used 64 counterfactual groups (256 comparisons);
its own cluster-bootstrap 95% CIs were wide enough that the always-off/
fixed-high-eligible non-inferiority question was not settled (e.g.
fixed_high: point estimate NetWin=+0.031, 95% CI [-0.109, +0.180] --
the CI includes values below the -0.05 margin). The V5.3 plan requires
freezing sample size FROM this real observed variance, before generating
any new V5.3 data, rather than picking a size "by convenience."

Method: derive each comparison's real cluster design effect (DEFF) from
outputs/pm_v1_5_v5_2_confirmation_final_report_v1/confirmation_analysis.json
(64 clusters, 256 comparisons, percentile cluster bootstrap, 50000
replicates) by comparing the reported cluster-bootstrap CI half-width to the
naive iid multinomial half-width for the same win/loss/tie counts. Then,
holding DEFF fixed (a conservative assumption: new same-stack V5.3 data
collected the same way should not have a *worse* clustering structure),
project the minimum total N (and equivalently minimum group count, assuming
4 comparisons/seeds per group as in V5.2) needed for 80% power to clear the
-0.05 non-inferiority margin, assuming a true NetWin of 0.0 (the more
conservative, worst-realistic planning assumption -- not the observed
point estimate, which would be optimistic).

This is planning-only. It does not read, use, or condition on any V5.3
outcome (there is none yet); "outcome_used_to_choose_sample_or_thresholds"
is frozen false.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from metacom_pm.io import read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"

SOURCE = ROOT / "outputs/pm_v1_5_v5_2_confirmation_final_report_v1/confirmation_analysis.json"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_sample_size_power_simulation_v1"

Z_95 = 1.959963984540054
Z_TARGET_POWER = 0.8416212335729143  # z for 80% one-sided power
NONINFERIORITY_MARGIN = -0.05
ROWS_PER_GROUP_V5_2 = 4  # 256 comparisons / 64 groups in the V5.2 confirmation design


def iid_se(p_win: float, p_loss: float, d: float, n: int) -> float:
    return math.sqrt(max((p_win + p_loss - d * d) / n, 0.0))


def required_n_for_power(*, p_win: float, p_loss: float, deff: float, true_d: float, power_z: float = Z_TARGET_POWER) -> int:
    """Smallest N (searched, not closed-form, because SE depends on N through
    the multinomial proportions held fixed at the pilot's observed rates)
    such that a two-sided 95% CI's lower bound clears NONINFERIORITY_MARGIN
    at probability >= 80%, assuming the true NetWin is ``true_d``."""

    n = ROWS_PER_GROUP_V5_2
    while n < 200_000:
        se = math.sqrt(deff) * iid_se(p_win, p_loss, true_d, n)
        if se == 0:
            return n
        # power = P(true_d - Z_95*se > margin) under D_hat ~ Normal(true_d, se^2)... solved below
        z_power = (true_d - NONINFERIORITY_MARGIN - Z_95 * se) / se
        if z_power >= power_z:
            return n
        n += ROWS_PER_GROUP_V5_2
    raise RuntimeError("required N did not converge under 200,000 comparisons")


def run(out_dir: Path) -> dict[str, Any]:
    source = read_json(SOURCE)
    if source.get("status") not in {"PASS", "FAIL", "COMPLETE"} and "learned_comparisons" not in source:
        raise RuntimeError("unexpected confirmation_analysis.json shape")
    comparisons = source["learned_comparisons"]
    bootstrap = source["bootstrap"]
    if bootstrap.get("groups") != 64:
        raise RuntimeError("expected 64 confirmation groups; source data drifted")

    per_comparator: dict[str, Any] = {}
    for name, row in comparisons.items():
        nw, nl, nt = row["quality_better"], row["quality_worse"], row["quality_tie_or_same_arm"]
        n = nw + nl + nt
        p_win, p_loss = nw / n, nl / n
        d_observed = row["quality_mean_difference"]
        lo, hi = row["quality_cluster_bootstrap_95_ci"]
        se_cluster = (hi - lo) / (2 * Z_95)
        se_iid = iid_se(p_win, p_loss, d_observed, n)
        deff = (se_cluster / se_iid) ** 2 if se_iid > 0 else float("nan")
        per_comparator[name] = {
            "n": n, "quality_better": nw, "quality_worse": nl, "quality_tie": nt,
            "p_win": round(p_win, 6), "p_loss": round(p_loss, 6),
            "netwin_observed": d_observed,
            "cluster_bootstrap_95_ci": [lo, hi],
            "se_cluster_observed": round(se_cluster, 6),
            "se_iid_naive": round(se_iid, 6),
            "design_effect": round(deff, 4),
        }

    max_deff = max(row["design_effect"] for row in per_comparator.values())
    planning_p_win = min(row["p_win"] for row in per_comparator.values())
    planning_p_loss = max(row["p_loss"] for row in per_comparator.values())

    required_n = {}
    for name, row in per_comparator.items():
        n_req = required_n_for_power(
            p_win=row["p_win"], p_loss=row["p_loss"], deff=row["design_effect"], true_d=0.0,
        )
        required_n[name] = {"design_effect_used": row["design_effect"], "required_n": n_req, "required_groups": n_req // ROWS_PER_GROUP_V5_2}

    conservative_n = required_n_for_power(p_win=planning_p_win, p_loss=planning_p_loss, deff=max_deff, true_d=0.0)

    result = {
        "protocol": "pm-v1.5-v5.3-sample-size-power-simulation-v1",
        "status": "FROZEN_ZERO_API_PLANNING_ONLY",
        "method": "design_effect_from_real_v5.2_confirmation_cluster_bootstrap_then_project_n_for_80pct_power",
        "source_confirmation_analysis": {
            "path": "outputs/pm_v1_5_v5_2_confirmation_final_report_v1/confirmation_analysis.json",
            "sha256": sha256_file(SOURCE),
            "groups": bootstrap["groups"], "replicates": bootstrap["replicates"],
        },
        "assumptions": {
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "target_power": 0.80,
            "planning_true_netwin": 0.0,
            "rows_per_group_held_constant_at_v5_2_design": ROWS_PER_GROUP_V5_2,
            "design_effect_held_fixed_from_pilot": True,
        },
        "per_comparator_from_real_v5_2_data": per_comparator,
        "required_n_per_comparator": required_n,
        "conservative_recommendation": {
            "design_effect_used": round(max_deff, 4),
            "required_total_comparisons": conservative_n,
            "required_groups": conservative_n // ROWS_PER_GROUP_V5_2,
            "note": "uses the worst observed p_win/p_loss/DEFF combination across the three real comparators as a conservative single planning N",
        },
        "engineering_floor_from_plan_section_4_2": {
            "effect_fit_min_groups_per_component": 128,
            "fresh_confirmation_min_groups_per_component": 64,
            "sealed_internal_min_groups_per_component": 64,
            "rule": "use the LARGER of the engineering floor and the power-derived N",
        },
        "outcome_used_to_choose_sample_or_thresholds": False,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "power_simulation_result.json", result)
    return result


def main() -> None:
    import sys

    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal run requires {FORMAL_PYTHON}; got {sys.executable}")
    print(json.dumps(run(OUT_DIR), indent=2))


if __name__ == "__main__":
    main()
