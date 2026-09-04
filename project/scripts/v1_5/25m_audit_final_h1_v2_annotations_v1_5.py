#!/usr/bin/env python3
"""Audit H1-v2 human annotations before gold freeze or model fitting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1-v2-annotation-data-quality-audit-v1"
REVIEW_PROTOCOL = "pm-v1.5-final-candidate-gold-human-review-v2"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")
ALLOWED_DECISIONS = {"on", "off", "abstain"}
ALLOWED_SUBTYPES = {
    "MP": {"CANDIDATE_ABSENT", "MP_PREFERENCE", "MP_PROFILE"},
    "MS": {"CANDIDATE_ABSENT", "MS_SESSION"},
    "ME": {
        "CANDIDATE_ABSENT",
        "ME_REUSABLE_OUTCOME",
        "ME_CONTEXT_EVENT",
        "ME_UNRESOLVED_EVENT",
    },
    "RS": {"CANDIDATE_ABSENT", "RS_ATOMIC_MOVE"},
}
ME_CUE = "a tentative option based on what helped before is welcome."


def _derived(decision: dict[str, Any]) -> str:
    gate_1 = str(decision.get("applicability_safe", "")).lower()
    gate_2 = str(decision.get("incremental_over_r0", "")).lower()
    if "no" in {gate_1, gate_2}:
        return "off"
    if gate_1 == gate_2 == "yes":
        return "on"
    return "abstain"


def _component(item: dict[str, Any], component: str) -> dict[str, Any]:
    return next(row for row in item["components"] if row["component"] == component)


def _state_signature(item: dict[str, Any]) -> str:
    """Exclude the decorative history prefix that previously masked duplicates."""
    payload = {
        "current_user_text": item["current_user_text"],
        "candidate_texts": {
            component: _component(item, component)["candidate_text"]
            for component in COMPONENTS
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _feature_key(features: dict[str, Any]) -> str:
    return json.dumps(features, ensure_ascii=False, sort_keys=True)


def _feature_ceiling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["feature_key"]].append(row["label"])
    positives = sum(row["label"] for row in rows)
    negatives = len(rows) - positives
    accuracy_correct = sum(max(sum(values), len(values) - sum(values)) for values in groups.values())
    true_positive = 0
    true_negative = 0
    for values in groups.values():
        positive = sum(values)
        negative = len(values) - positive
        positive_gain = positive / positives if positives else 0.0
        negative_gain = negative / negatives if negatives else 0.0
        if positive_gain >= negative_gain:
            true_positive += positive
        else:
            true_negative += negative
    mixed = [values for values in groups.values() if 0 < sum(values) < len(values)]
    return {
        "candidate_present_rows": len(rows),
        "distinct_feature_vectors": len(groups),
        "mixed_label_vectors": len(mixed),
        "rows_in_mixed_label_vectors": sum(len(values) for values in mixed),
        "oracle_accuracy_ceiling": accuracy_correct / len(rows),
        "oracle_balanced_accuracy_ceiling": 0.5
        * (
            true_positive / positives if positives else 0.0
        )
        + 0.5 * (true_negative / negatives if negatives else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_candidate/human_review_packet.json",
    )
    parser.add_argument(
        "--overlap-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1_v2_candidate/independent_overlap_packet.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10/model_feature_rows.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument("--secondary-overlap-annotations", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_annotation_audit_v1",
    )
    args = parser.parse_args()

    annotations = [dict(row) for row in iter_jsonl(args.annotations)]
    packet = read_json(args.packet)
    overlap_packet = read_json(args.overlap_packet)
    bindings = {row["blind_state_id"]: dict(row) for row in iter_jsonl(args.binding)}
    features = {
        (row["state_id"], row["component"]): dict(row)
        for row in iter_jsonl(args.features)
    }
    strategy_cards = {
        row["card_id"]: dict(row) for row in iter_jsonl(args.strategy_cards)
    }
    items = {row["blind_state_id"]: row for row in packet["items"]}
    primary = {row["blind_state_id"]: row for row in annotations}
    failures: list[str] = []
    warnings: list[str] = []

    if len(annotations) != 256 or len(primary) != 256:
        failures.append("primary_annotations_not_256_unique_states")
    if set(primary) != set(items):
        failures.append("primary_annotation_ids_do_not_match_packet")
    if any(row.get("protocol") != REVIEW_PROTOCOL for row in annotations):
        failures.append("wrong_primary_review_protocol")
    if len({row.get("annotator_id") for row in annotations}) != 1:
        failures.append("primary_annotator_id_not_unique")

    schema_errors: list[dict[str, str]] = []
    distributions: dict[str, Any] = {}
    subtype_distributions: dict[str, Any] = {}
    feature_rows_by_component: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in COMPONENTS:
        split_counts: dict[str, Any] = {}
        overall: Counter[str] = Counter()
        for split in SPLITS:
            counts: Counter[str] = Counter()
            for blind_id, item in items.items():
                if bindings[blind_id]["split"] != split:
                    continue
                surface = _component(item, component)
                decision = primary[blind_id]["component_decisions"][component]
                actual = str(decision.get("derived_decision", "")).lower()
                expected = _derived(decision)
                if actual not in ALLOWED_DECISIONS or actual != expected:
                    schema_errors.append(
                        {"blind_state_id": blind_id, "component": component, "error": "derived_decision_mismatch"}
                    )
                subtype = str(decision.get("adjudicated_subtype", ""))
                if subtype not in ALLOWED_SUBTYPES[component]:
                    schema_errors.append(
                        {"blind_state_id": blind_id, "component": component, "error": "invalid_subtype"}
                    )
                if not decision.get("evidence_codes"):
                    schema_errors.append(
                        {"blind_state_id": blind_id, "component": component, "error": "missing_evidence_code"}
                    )
                if not surface["candidate_present"]:
                    if not (
                        actual == "off"
                        and subtype == "CANDIDATE_ABSENT"
                        and decision.get("evidence_codes") == ["CANDIDATE_ABSENT"]
                    ):
                        schema_errors.append(
                            {"blind_state_id": blind_id, "component": component, "error": "absent_candidate_contract_mismatch"}
                        )
                    continue
                counts[actual] += 1
                overall[actual] += 1
                state_id = bindings[blind_id]["state_id"]
                feature_key = _feature_key(features[(state_id, component)]["model_features"])
                feature_rows_by_component[component].append(
                    {
                        "blind_state_id": blind_id,
                        "split": split,
                        "feature_key": feature_key,
                        "label": int(actual == "on"),
                    }
                )
            total = sum(counts.values())
            majority_share = max(counts.values(), default=0) / total if total else 1.0
            split_counts[split] = {
                "counts": dict(counts),
                "candidate_present": total,
                "majority_constant_accuracy": majority_share,
            }
            if majority_share >= 0.70:
                failures.append(f"{component}_{split}_human_label_constant_baseline_ge_0_70")
        total = sum(overall.values())
        distributions[component] = {
            "overall": {
                "counts": dict(overall),
                "candidate_present": total,
                "majority_constant_accuracy": max(overall.values(), default=0) / total if total else 1.0,
            },
            "by_split": split_counts,
        }
        subtype_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for blind_id, item in items.items():
            surface = _component(item, component)
            if not surface["candidate_present"]:
                continue
            decision = primary[blind_id]["component_decisions"][component]
            subtype_counts[str(decision["adjudicated_subtype"])][
                str(decision["derived_decision"])
            ] += 1
        subtype_distributions[component] = {
            subtype: dict(counts) for subtype, counts in sorted(subtype_counts.items())
        }
    if schema_errors:
        failures.append("primary_annotation_schema_or_contract_errors")

    signature_groups: defaultdict[str, list[str]] = defaultdict(list)
    for blind_id, item in items.items():
        signature_groups[_state_signature(item)].append(blind_id)
    duplicates = [group for group in signature_groups.values() if len(group) > 1]
    duplicate_rows = []
    for group in duplicates:
        splits = sorted({bindings[blind_id]["split"] for blind_id in group})
        duplicate_rows.append(
            {
                "blind_state_ids": group,
                "splits": splits,
                "cross_split": len(splits) > 1,
            }
        )
    cross_split_duplicates = [row for row in duplicate_rows if row["cross_split"]]
    if cross_split_duplicates:
        failures.append("meaningful_state_signature_cross_split_duplicates")

    feature_audit = {
        component: _feature_ceiling(feature_rows_by_component[component])
        for component in COMPONENTS
    }
    if feature_audit["MS"]["oracle_balanced_accuracy_ceiling"] < 0.80:
        warnings.append("MS_existing_transparent_feature_vectors_have_low_empirical_separation")
    if subtype_distributions["MP"].get("MP_PREFERENCE", {}).get("on", 0) == 0:
        failures.append("MP_PREFERENCE_has_zero_human_ON_examples")

    rs_family_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for blind_id, item in items.items():
        if not _component(item, "RS")["candidate_present"]:
            continue
        card_id = bindings[blind_id]["candidate_bindings"]["RS"]["candidate_id"]
        family = str(strategy_cards[card_id]["strategy_family"])
        label = primary[blind_id]["component_decisions"]["RS"]["derived_decision"]
        rs_family_counts[family][label] += 1
    rs_family_rows = {
        family: dict(counts) for family, counts in sorted(rs_family_counts.items())
    }
    rs_positive = sum(counts.get("on", 0) for counts in rs_family_counts.values())
    rs_negative = sum(counts.get("off", 0) for counts in rs_family_counts.values())
    rs_family_tp = 0
    rs_family_tn = 0
    for counts in rs_family_counts.values():
        positive = counts.get("on", 0)
        negative = counts.get("off", 0)
        if positive / rs_positive >= negative / rs_negative:
            rs_family_tp += positive
        else:
            rs_family_tn += negative
    rs_family_only_ba = 0.5 * (
        rs_family_tp / rs_positive + rs_family_tn / rs_negative
    )
    if rs_family_only_ba >= 0.65:
        failures.append("RS_strategy_family_only_shortcut_ge_0_65")

    me_cue_counts: Counter[str] = Counter()
    for blind_id, item in items.items():
        if not _component(item, "ME")["candidate_present"]:
            continue
        cue = ME_CUE in item["current_user_text"].lower()
        label = primary[blind_id]["component_decisions"]["ME"]["derived_decision"] == "on"
        me_cue_counts[f"cue_{str(cue).lower()}__on_{str(label).lower()}"] += 1
    cue_true = me_cue_counts["cue_true__on_true"] + me_cue_counts["cue_true__on_false"]
    cue_false = me_cue_counts["cue_false__on_true"] + me_cue_counts["cue_false__on_false"]
    me_cue_accuracy = (
        me_cue_counts["cue_true__on_true"] + me_cue_counts["cue_false__on_false"]
    ) / (cue_true + cue_false)
    me_cue_sensitivity = me_cue_counts["cue_true__on_true"] / (
        me_cue_counts["cue_true__on_true"] + me_cue_counts["cue_false__on_true"]
    )
    me_cue_specificity = me_cue_counts["cue_false__on_false"] / (
        me_cue_counts["cue_false__on_false"] + me_cue_counts["cue_true__on_false"]
    )
    me_cue_balanced_accuracy = 0.5 * (me_cue_sensitivity + me_cue_specificity)
    if me_cue_balanced_accuracy >= 0.65:
        failures.append("ME_current_message_single_cue_shortcut_ge_0_65")

    overlap_ids = {row["blind_state_id"] for row in overlap_packet["items"]}
    primary_overlap: dict[str, Any] = {}
    for component in COMPONENTS:
        counts = Counter(
            primary[blind_id]["component_decisions"][component]["derived_decision"]
            for blind_id in overlap_ids
            if _component(items[blind_id], component)["candidate_present"]
        )
        primary_overlap[component] = {
            "present": sum(counts.values()),
            "counts": dict(counts),
            "absent": sum(
                not _component(items[blind_id], component)["candidate_present"]
                for blind_id in overlap_ids
            ),
        }

    iaa: dict[str, Any] = {"status": "ROW_LEVEL_SECONDARY_ANNOTATIONS_NOT_PROVIDED"}
    if args.secondary_overlap_annotations:
        secondary_rows = [dict(row) for row in iter_jsonl(args.secondary_overlap_annotations)]
        secondary = {row["blind_state_id"]: row for row in secondary_rows}
        if set(secondary) != overlap_ids:
            failures.append("secondary_overlap_ids_do_not_match_overlap_packet")
        else:
            pairs = []
            per_component: dict[str, Any] = {}
            for component in COMPONENTS:
                component_pairs = []
                subtype_matches = []
                for blind_id in sorted(overlap_ids):
                    if not _component(items[blind_id], component)["candidate_present"]:
                        continue
                    a = primary[blind_id]["component_decisions"][component]
                    b = secondary[blind_id]["component_decisions"][component]
                    component_pairs.append((a["derived_decision"], b["derived_decision"]))
                    subtype_matches.append(a["adjudicated_subtype"] == b["adjudicated_subtype"])
                pairs.extend(component_pairs)
                agree = sum(a == b for a, b in component_pairs)
                per_component[component] = {
                    "n": len(component_pairs),
                    "decision_raw_agreement": agree / len(component_pairs),
                    "subtype_raw_agreement": sum(subtype_matches) / len(subtype_matches),
                }
            labels = ("on", "off", "abstain")
            n = len(pairs)
            observed = sum(a == b for a, b in pairs) / n
            expected = sum(
                sum(a == label for a, _ in pairs)
                / n
                * sum(b == label for _, b in pairs)
                / n
                for label in labels
            )
            kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
            iaa = {
                "status": "COMPUTED",
                "n_present_candidate_pairs": n,
                "decision_raw_agreement": observed,
                "cohen_kappa": kappa,
                "per_component": per_component,
            }
    else:
        failures.append("row_level_secondary_overlap_missing_iaa_not_computable")

    report = {
        "protocol": PROTOCOL,
        "status": "HOLD_DO_NOT_TRAIN" if failures else "PASS_READY_TO_FREEZE_GOLD",
        "decision": "PRIMARY_LABELS_ARE_VALID_DEVELOPMENT_EVIDENCE_BUT_CURRENT_H1_V2_IS_NOT_CONFIRMATORY_GOLD",
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "primary_contract": {
            "rows": len(annotations),
            "unique_blind_state_ids": len(primary),
            "packet_id_match": set(primary) == set(items),
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors,
            "annotator_ids": sorted({str(row.get("annotator_id")) for row in annotations}),
        },
        "human_label_distribution": distributions,
        "candidate_subtype_distribution": subtype_distributions,
        "meaningful_state_duplicate_audit": {
            "signature_definition": "current_user_text + four exact candidate texts; decorative visible-dialogue prefix excluded",
            "duplicate_groups": len(duplicate_rows),
            "affected_states": sum(len(row["blind_state_ids"]) for row in duplicate_rows),
            "cross_split_duplicate_groups": len(cross_split_duplicates),
            "cross_split_affected_states": sum(len(row["blind_state_ids"]) for row in cross_split_duplicates),
            "groups": duplicate_rows,
        },
        "single_cue_shortcut_audit": {
            "ME_cue": ME_CUE,
            "counts": dict(me_cue_counts),
            "cue_only_accuracy": me_cue_accuracy,
            "cue_only_sensitivity": me_cue_sensitivity,
            "cue_only_specificity": me_cue_specificity,
            "cue_only_balanced_accuracy": me_cue_balanced_accuracy,
            "frozen_single_feature_gate": "balanced_accuracy_below_0.65",
        },
        "RS_strategy_family_shortcut_audit": {
            "counts": rs_family_rows,
            "family_only_oracle_balanced_accuracy": rs_family_only_ba,
            "note": "Current Question/Restatement/Suggestion/Reflection families are not counterbalanced across ON/OFF.",
        },
        "transparent_feature_collision_audit": feature_audit,
        "primary_overlap_distribution": primary_overlap,
        "inter_annotator_agreement": iaa,
        "salvage_boundary": {
            "retain": "Primary labels may be retained as development evidence; FIT may seed feature/codebook diagnosis after removing malformed/duplicate-linked cases.",
            "do_not_use": "Current FRESH_CONFIRMATION and SEALED_INTERNAL_TEST may not qualify or select a model because meaningful duplicate states cross split and ME has a visible cue shortcut.",
            "minimum_next_data_action": "Repair the static gate and transparent factors using development evidence, then construct genuinely new counterbalanced confirmation and sealed states. Do not relabel all 256 by default.",
        },
        "provenance": {
            "annotations": str(args.annotations),
            "packet": str(args.packet),
            "overlap_packet": str(args.overlap_packet),
            "binding": str(args.binding),
            "features": str(args.features),
            "strategy_cards": str(args.strategy_cards),
            "secondary_overlap_annotations": str(args.secondary_overlap_annotations)
            if args.secondary_overlap_annotations
            else None,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "audit_report.json", report)
    write_jsonl(args.out_dir / "primary_annotations.jsonl", annotations)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "failures": len(report["failures"]),
            "cross_split_duplicate_groups": len(cross_split_duplicates),
            "ME_cue_only_accuracy": me_cue_accuracy,
            "IAA": iaa["status"],
        }
    )


if __name__ == "__main__":
    main()
