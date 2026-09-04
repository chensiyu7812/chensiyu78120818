#!/usr/bin/env python3
"""Build the frozen balanced allocation for the PM V1.5 H2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-reverse-designed-256-state-blueprint-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = (("train", 40, 10), ("calibration", 12, 3), ("internal_test", 12, 3))
SEED_TEXT = f"{PROTOCOL}:allocation"


def action_bits(action_index: int) -> dict[str, int]:
    return {
        component: (action_index >> bit_index) & 1
        for bit_index, component in enumerate(COMPONENTS)
    }


def action_id(action_index: int) -> str:
    bits = action_bits(action_index)
    memory = "".join(component for component in ("MP", "MS", "ME") if bits[component])
    return f"{memory or 'M0'}+{'RS' if bits['RS'] else 'R0'}"


def _stable_permutation(values: list[int], namespace: str) -> list[int]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{SEED_TEXT}:{namespace}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    user_offset = 0
    for split, user_count, repetitions_per_action in SPLITS:
        actions = [
            action
            for action in range(16)
            for _ in range(repetitions_per_action)
        ]
        actions = _stable_permutation(actions, f"{split}:actions")
        if len(actions) != user_count * 4:
            raise AssertionError("split allocation does not fill four states per user")
        users = _stable_permutation(
            list(range(user_offset, user_offset + user_count)),
            f"{split}:users",
        )
        for position, action in enumerate(actions):
            user_number = users[position // 4]
            local_state = position % 4
            bits = action_bits(action)
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "split": split,
                    "synthetic_user_id": f"pm15_user_{user_number:03d}",
                    "state_id": (
                        f"pm15_state_{split}_{user_number:03d}_{local_state}"
                    ),
                    "action_index": action,
                    "action_id": action_id(action),
                    "resource_need_bits": bits,
                    "blueprint_status": "ALLOCATED_CONTENT_NOT_YET_GENERATED",
                    "routing_label_source": (
                        "pre_action_blueprint_then_H2_human_review"
                    ),
                    "response_winner_used_as_label": False,
                }
            )
        user_offset += user_count
    return rows


def audit(rows: list[dict[str, object]]) -> dict[str, object]:
    split_report: dict[str, object] = {}
    for split, user_count, repetitions_per_action in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        action_counts = Counter(str(row["action_id"]) for row in selected)
        user_counts = Counter(str(row["synthetic_user_id"]) for row in selected)
        component_counts = {
            component: Counter(
                int(row["resource_need_bits"][component])  # type: ignore[index]
                for row in selected
            )
            for component in COMPONENTS
        }
        split_report[split] = {
            "states": len(selected),
            "users": len(user_counts),
            "states_per_user": sorted(set(user_counts.values())),
            "action_count_values": sorted(set(action_counts.values())),
            "expected_per_action": repetitions_per_action,
            "component_on_off": {
                component: {
                    "off": counts[0],
                    "on": counts[1],
                }
                for component, counts in component_counts.items()
            },
            "passed": (
                len(selected) == user_count * 4
                and len(user_counts) == user_count
                and set(user_counts.values()) == {4}
                and len(action_counts) == 16
                and set(action_counts.values()) == {repetitions_per_action}
                and all(counts[0] == counts[1] for counts in component_counts.values())
            ),
        }
    users_by_split = {
        split: {
            str(row["synthetic_user_id"])
            for row in rows
            if row["split"] == split
        }
        for split, _, _ in SPLITS
    }
    disjoint = all(
        not users_by_split[left] & users_by_split[right]
        for index, (left, _, _) in enumerate(SPLITS)
        for right, _, _ in SPLITS[index + 1 :]
    )
    return {
        "protocol": PROTOCOL,
        "status": (
            "BALANCED_BLUEPRINT_ALLOCATION_FROZEN"
            if disjoint and all(
                bool(report["passed"]) for report in split_report.values()  # type: ignore[union-attr]
            )
            else "BLUEPRINT_ALLOCATION_INVALID"
        ),
        "total_states": len(rows),
        "total_users": len(
            {str(row["synthetic_user_id"]) for row in rows}
        ),
        "split_users_disjoint": disjoint,
        "splits": split_report,
        "content_generation_started": False,
        "human_review_started": False,
        "next": (
            "Generate matched natural dialogue/resource content under each "
            "frozen blueprint, run pre-H2 leakage and shortcut audits, then "
            "render the single complete H2 packet."
        ),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_reverse_designed_blueprint_v1",
    )
    args = parser.parse_args()
    rows = build_rows()
    report = audit(rows)
    if report["status"] != "BALANCED_BLUEPRINT_ALLOCATION_FROZEN":
        raise RuntimeError(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "state_blueprints.jsonl", rows)
    write_json(args.out_dir / "allocation_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
