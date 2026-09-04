#!/usr/bin/env python3
"""Unblind and summarize one completed independent R0/RS confirmation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-rag-v4-direct-effect-independent-analysis-v3"
PREFERENCES = {"A", "B", "tie", "uncertain"}
RISK_VERDICTS = {"yes", "no", "uncertain"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _two_sided_sign_test_p(first: int, second: int) -> float | None:
    n = first + second
    if n == 0:
        return None
    tail = min(first, second)
    probability = 2.0 * sum(
        math.comb(n, value) * (0.5**n)
        for value in range(tail + 1)
    )
    return min(1.0, probability)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4_independent_blind",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4_independent_blind"
        / "independent_blind_annotations.jsonl",
    )
    parser.add_argument(
        "--prior-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4_execution"
        / "codex_blind_diagnostic_annotations.jsonl",
    )
    parser.add_argument(
        "--selected-states",
        type=Path,
        default=None,
        help=(
            "Optional selected_states.jsonl used only to report transparent "
            "strategy-family breakdowns."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4_independent_blind"
        / "independent_confirmation_report.json",
    )
    args = parser.parse_args()

    packet = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "human_blind_packet.jsonl")
    }
    key = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "private_blinding_key.jsonl")
    }
    annotations = {
        str(row["blind_item_id"]): row for row in _rows(args.annotations)
    }
    if set(packet) != set(key) or set(packet) != set(annotations):
        raise RuntimeError("packet/key/annotation blind id sets differ")

    quality = Counter()
    risk_by_arm: dict[str, Counter[str]] = {
        "R0": Counter(),
        "RS": Counter(),
    }
    risk_response_by_arm = Counter()
    risk_ids_by_pair_arm: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"R0": [], "RS": []}
    )
    resolved_by_pair: dict[str, str] = {}
    validation_errors: list[str] = []
    for blind_id in sorted(packet):
        annotation = annotations[blind_id]
        private = key[blind_id]
        preference = annotation.get("quality_preference")
        if preference not in PREFERENCES:
            validation_errors.append(f"{blind_id}: invalid quality preference")
            continue
        if preference in {"tie", "uncertain"}:
            resolved = str(preference)
        else:
            resolved = str(
                private[
                    "response_a_arm"
                    if preference == "A"
                    else "response_b_arm"
                ]
            )
        quality[resolved] += 1
        pair_id = str(private["pair_id"])
        resolved_by_pair[pair_id] = resolved

        for label, field in (("A", "risk_a"), ("B", "risk_b")):
            arm = str(
                private[
                    "response_a_arm" if label == "A" else "response_b_arm"
                ]
            )
            findings = annotation.get(field)
            if not isinstance(findings, dict):
                validation_errors.append(f"{blind_id}: missing {field}")
                continue
            for risk_id, finding in findings.items():
                if not isinstance(finding, dict):
                    validation_errors.append(
                        f"{blind_id}: malformed {field}/{risk_id}"
                    )
                    continue
                verdict = finding.get("judgment")
                if verdict not in RISK_VERDICTS:
                    validation_errors.append(
                        f"{blind_id}: invalid {field}/{risk_id}"
                    )
                    continue
                risk_by_arm[arm][f"{risk_id}:{verdict}"] += 1
                if verdict == "yes" and not str(
                    finding.get("evidence", "")
                ).strip():
                    validation_errors.append(
                        f"{blind_id}: yes without evidence for {field}/{risk_id}"
                    )
                if verdict == "yes":
                    risk_ids_by_pair_arm[pair_id][arm].append(str(risk_id))

    if validation_errors:
        raise RuntimeError("; ".join(validation_errors[:20]))

    for pair_id in resolved_by_pair:
        for arm in ("R0", "RS"):
            if risk_ids_by_pair_arm[pair_id][arm]:
                risk_response_by_arm[arm] += 1

    # The PM target is deliberately not a weighted scalar. A response with a
    # material risk is inadmissible; among admissible responses an RS quality
    # win may open the component; ties and R0 wins close it.
    risk_first_by_pair: dict[str, str] = {}
    risk_first_reason_by_pair: dict[str, str] = {}
    for pair_id, resolved in resolved_by_pair.items():
        r0_risky = bool(risk_ids_by_pair_arm[pair_id]["R0"])
        rs_risky = bool(risk_ids_by_pair_arm[pair_id]["RS"])
        if rs_risky:
            decision = "R0"
            reason = "RS_material_risk"
        elif r0_risky:
            decision = "RS"
            reason = "R0_risky_RS_admissible"
        elif resolved == "RS":
            decision = "RS"
            reason = "both_admissible_RS_material_quality_win"
        else:
            decision = "R0"
            reason = "no_material_RS_benefit"
        risk_first_by_pair[pair_id] = decision
        risk_first_reason_by_pair[pair_id] = reason

    family_by_pair: dict[str, str] = {}
    if args.selected_states is not None:
        family_by_pair = {
            str(row["pair_id"]): str(row["selected_strategy_family"])
            for row in _rows(args.selected_states)
        }
        missing_family = sorted(set(resolved_by_pair) - set(family_by_pair))
        if missing_family:
            raise RuntimeError(
                "selected-states missing annotated pairs: "
                + ", ".join(missing_family[:10])
            )
    family_quality: dict[str, Counter[str]] = defaultdict(Counter)
    family_risk_first: dict[str, Counter[str]] = defaultdict(Counter)
    pair_decisions: list[dict[str, Any]] = []
    for pair_id in sorted(resolved_by_pair):
        family = family_by_pair.get(pair_id)
        if family is not None:
            family_quality[family][resolved_by_pair[pair_id]] += 1
            family_risk_first[family][risk_first_by_pair[pair_id]] += 1
        pair_decisions.append(
            {
                "pair_id": pair_id,
                "strategy_family": family,
                "quality_result": resolved_by_pair[pair_id],
                "r0_material_risks": sorted(
                    risk_ids_by_pair_arm[pair_id]["R0"]
                ),
                "rs_material_risks": sorted(
                    risk_ids_by_pair_arm[pair_id]["RS"]
                ),
                "risk_first_action": risk_first_by_pair[pair_id],
                "risk_first_reason": risk_first_reason_by_pair[pair_id],
            }
        )
    r0_only_risk = sum(
        bool(risk_ids_by_pair_arm[pair_id]["R0"])
        and not risk_ids_by_pair_arm[pair_id]["RS"]
        for pair_id in resolved_by_pair
    )
    rs_only_risk = sum(
        bool(risk_ids_by_pair_arm[pair_id]["RS"])
        and not risk_ids_by_pair_arm[pair_id]["R0"]
        for pair_id in resolved_by_pair
    )
    both_risky = sum(
        bool(risk_ids_by_pair_arm[pair_id]["R0"])
        and bool(risk_ids_by_pair_arm[pair_id]["RS"])
        for pair_id in resolved_by_pair
    )
    neither_risky = len(resolved_by_pair) - (
        r0_only_risk + rs_only_risk + both_risky
    )
    risk_first_counts = Counter(risk_first_by_pair.values())
    minimum_train_groups_per_class = 8

    prior_rows = (
        _rows(args.prior_diagnostic)
        if args.prior_diagnostic.is_file()
        else []
    )
    prior_resolved: dict[str, str] = {}
    if prior_rows:
        key_by_pair = {str(row["pair_id"]): row for row in key.values()}
        for row in prior_rows:
            pair_id = str(row["pair_id"])
            preference = str(row["preference"])
            private = key_by_pair[pair_id]
            if preference in {"tie", "uncertain"}:
                resolved = preference
            else:
                resolved = str(
                    private[
                        "response_a_arm"
                        if preference == "A"
                        else "response_b_arm"
                    ]
                )
            prior_resolved[pair_id] = resolved
    shared = sorted(set(resolved_by_pair) & set(prior_resolved))
    agreement = (
        sum(resolved_by_pair[pair] == prior_resolved[pair] for pair in shared)
        / len(shared)
        if shared
        else None
    )
    disagreements = [
        {
            "pair_id": pair,
            "independent": resolved_by_pair[pair],
            "prior_codex_diagnostic": prior_resolved[pair],
        }
        for pair in shared
        if resolved_by_pair[pair] != prior_resolved[pair]
    ]

    pair_count = len(packet)
    report = {
        "protocol": PROTOCOL,
        "status": "INDEPENDENT_BLIND_CONFIRMATION_COMPLETE",
        "pair_count": pair_count,
        "quality_counts": dict(sorted(quality.items())),
        "rs_win_rate_all_pairs": quality["RS"] / pair_count,
        "r0_win_rate_all_pairs": quality["R0"] / pair_count,
        "tie_rate": quality["tie"] / pair_count,
        "uncertain_rate": quality["uncertain"] / pair_count,
        "non_tie_rs_win_rate": (
            quality["RS"] / (quality["RS"] + quality["R0"])
            if quality["RS"] + quality["R0"]
            else None
        ),
        "quality_non_tie_exact_sign_test_p": _two_sided_sign_test_p(
            quality["RS"], quality["R0"]
        ),
        "material_risk_counts_by_arm": {
            arm: {
                risk: count
                for risk, count in sorted(values.items())
                if risk.endswith(":yes")
            }
            for arm, values in risk_by_arm.items()
        },
        "material_risk_response_counts_by_arm": {
            arm: int(risk_response_by_arm[arm]) for arm in ("R0", "RS")
        },
        "material_risk_pairing": {
            "R0_only": r0_only_risk,
            "RS_only": rs_only_risk,
            "both": both_risky,
            "neither": neither_risky,
        },
        "material_risk_discordant_exact_sign_test_p": (
            _two_sided_sign_test_p(r0_only_risk, rs_only_risk)
        ),
        "risk_first_action_counts": dict(
            sorted(risk_first_counts.items())
        ),
        "risk_first_reason_counts": dict(
            sorted(Counter(risk_first_reason_by_pair.values()).items())
        ),
        "quality_counts_by_strategy_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_quality.items())
        },
        "risk_first_action_counts_by_strategy_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_risk_first.items())
        },
        "pair_decisions": pair_decisions,
        "qualification": {
            "nondegenerate_action_check_passed": (
                risk_first_counts["RS"] > 0
                and risk_first_counts["R0"] > 0
            ),
            "general_RS_quality_advantage_demonstrated": False,
            "general_RS_risk_advantage_demonstrated": False,
            "family_level_effect_claim_authorized": False,
            "minimum_train_groups_per_class": (
                minimum_train_groups_per_class
            ),
            "positive_RS_on_groups": risk_first_counts["RS"],
            "nonpositive_R0_groups": risk_first_counts["R0"],
            "training_data_ready": (
                risk_first_counts["RS"]
                >= minimum_train_groups_per_class
                and risk_first_counts["R0"]
                >= minimum_train_groups_per_class
            ),
        },
        "prior_diagnostic_exact_agreement": agreement,
        "prior_diagnostic_shared_pairs": len(shared),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "human_gold": False,
        "interpretation": (
            "This is one independent blind confirmation. It becomes human "
            "evidence only when the annotator is an identified human; "
            "otherwise it remains an independent model/coder diagnostic."
        ),
        "lineage": {
            "packet_sha256": sha256_file(
                args.blind_dir / "human_blind_packet.jsonl"
            ),
            "private_key_sha256": sha256_file(
                args.blind_dir / "private_blinding_key.jsonl"
            ),
            "annotations_sha256": sha256_file(args.annotations),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
