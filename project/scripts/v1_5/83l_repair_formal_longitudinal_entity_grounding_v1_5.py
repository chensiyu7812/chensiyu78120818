#!/usr/bin/env python3
"""Repair formal V5.3 relationship/entity temporal grounding.

This migration is intentionally narrow.  Apart from three documented,
meaning-preserving cross-user template repairs, it does not rewrite dialogue,
memory-source spans, typed candidates, or ME action/result spans.  It only:

* binds a named relationship at the first user turn that actually names it;
* replaces seven never-grounded proper names with the role label the user used;
* removes entity references before that binding point;
* repairs one known event that was attached to the wrong session; and
* emits a minimal relationship schema without future-trajectory notes.
* removes four cross-user 8-gram collisions by synchronously replacing the
  source turn and every exact literal-span copy of the same phrase.

The script validates every migrated user with the strengthened ingestion gate
before writing the tracked canonical intake.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIRS = (
    ROOT / "outputs/pm_v1_5_v5_3_formal_longitudinal_user_validation_v1/canonical_users",
    ROOT / "outputs/pm_v1_5_v5_3_formal_gpt_u001_u005_validation_v1/canonical_users",
    ROOT / "outputs/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/canonical_users",
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/canonical_users"
)
DEFAULT_CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_3_complete_training_data_generation_v1.json"
VALIDATOR_PATH = ROOT / "scripts/v1_5/82l_validate_formal_longitudinal_user_v1_5.py"

# These proper names never occur in user-authored dialogue.  The replacements
# are exact, stable role labels that do occur there; no new content is authored.
ROLE_LABEL_OVERRIDES = {
    ("p2r_formal_gpt_u002", "entity_chidi"): "Dad",
    ("p2r_formal_gpt_u003", "entity_keiko"): "Mum",
    ("p2r_formal_gpt_u003", "entity_masaru"): "Dad",
    ("p2r_formal_gpt_u004", "entity_huda"): "Mum",
    ("p2r_formal_claude_u001", "entity_03"): "father",
    ("p2r_formal_claude_u002", "entity_02"): "mother",
    ("p2r_formal_claude_u004", "entity_03"): "mother",
}

# Exact replacements are versioned here rather than edited by hand in the
# generated files.  They remove author-template overlap while preserving the
# same observable goal/preference.  Recursive application keeps source turns,
# literal spans, candidate_text and preference history in sync.
SURFACE_REPLACEMENTS = {
    "p2r_formal_gpt_u001": (
        (
            "My goal is to finish the certificate without using every day off as study recovery.",
            "I want to complete the certificate while still leaving some days off for actual rest.",
        ),
    ),
    "p2r_formal_gpt_u004": (
        (
            "My goal is to finish the certificate without treating every warehouse emergency as more important than class.",
            "I want to complete the certificate without letting each warehouse emergency outrank class.",
        ),
        (
            "but I haven't decided what I will do when Mélanie asks me to stay on a class night.",
            "and I still need to decide how to respond when Mélanie asks me to stay on a class night.",
        ),
        (
            "give me the direct answer first and explain after",
            "start with a direct answer, then give the explanation",
        ),
    ),
}


def _load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("formal_longitudinal_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import validator from {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _candidate_rows(user: dict[str, Any]):
    for session in user.get("sessions", []):
        for candidate in session.get("typed_candidates", []):
            yield session, candidate


def _replace_exact_strings(value: Any, old: str, new: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        replaced: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            replaced_item, item_count = _replace_exact_strings(item, old, new)
            replaced[key] = replaced_item
            count += item_count
        return replaced, count
    if isinstance(value, list):
        replaced_list = []
        count = 0
        for item in value:
            replaced_item, item_count = _replace_exact_strings(item, old, new)
            replaced_list.append(replaced_item)
            count += item_count
        return replaced_list, count
    if isinstance(value, str) and old in value:
        return value.replace(old, new), value.count(old)
    return value, 0


def _self_name_tokens(user: dict[str, Any], validator: Any) -> set[str]:
    tokens: set[str] = set()
    for profile in user.get("profile_history", []):
        if profile.get("field_type") == "name" and profile.get("item_role", "base") == "base":
            tokens |= validator._grounding_tokens(str(profile.get("field_value") or ""))
    return tokens


def _repair_tomasz_motorcycle_event(user: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    if user.get("user_id") != "p2r_formal_claude_u003":
        return
    event = next(row for row in user["events"] if row.get("event_id") == "u003_evt_14")
    before = copy.deepcopy(event)
    event["session_index"] = 24
    event["entity_ids"] = ["self", "entity_08"]
    event["description"] = (
        "The motorcycle came out of the mill's old fire station when it was cleared "
        "in the nineties."
    )
    actions.append(
        {
            "kind": "event_session_and_description_grounding",
            "event_id": event["event_id"],
            "before": before,
            "after": copy.deepcopy(event),
        }
    )


def _repair_user(raw: dict[str, Any], validator: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    user, normalization_actions = validator._normalize_user_schema(raw)
    actions: list[dict[str, Any]] = [
        {"kind": "lossless_normalization", "actions": normalization_actions}
    ] if normalization_actions else []
    user_id = str(user["user_id"])
    for old, new in SURFACE_REPLACEMENTS.get(user_id, ()):
        user, occurrence_count = _replace_exact_strings(user, old, new)
        if occurrence_count < 2:
            raise ValueError(
                f"{user_id}: template repair must update a source turn and at least one "
                f"stored copy, observed {occurrence_count} occurrence(s)"
            )
        actions.append(
            {
                "kind": "cross_user_template_surface_repair",
                "old": old,
                "new": new,
                "occurrence_count": occurrence_count,
            }
        )
    sessions = list(user["sessions"])
    session_text = {
        int(session["session_index"]): "\n".join(
            str(turn.get("text") or "")
            for turn in session.get("dialogue", [])
            if turn.get("role") == "user"
        )
        for session in sessions
    }
    self_tokens = _self_name_tokens(user, validator)
    repaired_relationships: list[dict[str, Any]] = []

    for raw_relationship in user.get("relationships", []):
        relationship = dict(raw_relationship)
        entity_id = str(relationship["entity_id"])
        old_name = str(relationship.get("name") or "")
        new_name = ROLE_LABEL_OVERRIDES.get((user_id, entity_id), old_name)
        declared_tokens = validator._grounding_tokens(new_name) - self_tokens
        visible_sessions = sorted(
            index
            for index, text in session_text.items()
            if declared_tokens & validator._grounding_tokens(text)
        )
        if not declared_tokens or not visible_sessions:
            raise ValueError(
                f"{user_id}/{entity_id}: declared name or role {new_name!r} is never grounded"
            )
        new_valid_from = visible_sessions[0]
        old_valid_from = relationship.get("valid_from_session")
        repaired = {
            "entity_id": entity_id,
            "name": new_name,
            "relationship": str(relationship.get("relationship") or relationship.get("relation") or ""),
            "valid_from_session": new_valid_from,
            "valid_until_session": relationship.get("valid_until_session"),
        }
        repaired_relationships.append(repaired)
        actions.append(
            {
                "kind": "relationship_grounding",
                "entity_id": entity_id,
                "old_name": old_name,
                "new_name": new_name,
                "old_valid_from_session": old_valid_from,
                "new_valid_from_session": new_valid_from,
                "visible_name_or_role_sessions": visible_sessions,
                "removed_noncanonical_fields": sorted(
                    set(relationship)
                    - {"entity_id", "name", "relationship", "valid_from_session", "valid_until_session"}
                ),
            }
        )

        # A session may retain later entity references, but it may not know the
        # entity before the name/role is visible.  Explicit visible mentions are
        # always represented in the structured session metadata.
        for session in sessions:
            index = int(session["session_index"])
            refs = [str(value) for value in session.get("entity_ids", [])]
            if index < new_valid_from:
                refs = [value for value in refs if value != entity_id]
            if index in visible_sessions and entity_id not in refs:
                refs.append(entity_id)
            session["entity_ids"] = _unique(refs)

        for event in user.get("events", []):
            if int(event.get("session_index") or 0) < new_valid_from:
                event["entity_ids"] = [
                    value for value in event.get("entity_ids", []) if value != entity_id
                ]
        for source_session, candidate in _candidate_rows(user):
            if int(source_session["session_index"]) < new_valid_from:
                candidate["entity_ids"] = [
                    value for value in candidate.get("entity_ids", []) if value != entity_id
                ]

    user["relationships"] = repaired_relationships
    user["sessions"] = sessions
    _repair_tomasz_motorcycle_event(user, actions)

    # Rebuild event cross-references deterministically after the targeted move.
    events_by_session: dict[int, list[str]] = {}
    for event in user.get("events", []):
        events_by_session.setdefault(int(event["session_index"]), []).append(str(event["event_id"]))
    for session in user["sessions"]:
        session["event_ids"] = events_by_session.get(int(session["session_index"]), [])

    return user, actions


def _input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no input users found")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, dest="inputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    validator = _load_validator()
    contract = _read(args.contract)
    input_paths = args.inputs or list(DEFAULT_INPUT_DIRS)
    files = _input_files(input_paths)
    users: dict[str, dict[str, Any]] = {}
    source_by_user: dict[str, Path] = {}
    actions_by_user: dict[str, list[dict[str, Any]]] = {}
    validation_by_user: dict[str, dict[str, Any]] = {}

    for path in files:
        raw = _read(path)
        user_id = str(raw.get("user_id") or "")
        if not user_id or user_id in users:
            raise ValueError(f"missing or duplicate user_id {user_id!r} from {path}")
        repaired, actions = _repair_user(raw, validator)
        result = validator._validate_user(repaired, contract)
        hard = [row for row in result["issues"] if row["severity"] == "hard"]
        if hard:
            raise ValueError(f"{user_id}: repaired user still has hard issues: {hard}")
        users[user_id] = repaired
        source_by_user[user_id] = path
        actions_by_user[user_id] = actions
        validation_by_user[user_id] = result

    expected_ids = {
        "p2r_formal_gpt_u000",
        *(f"p2r_formal_gpt_u{index:03d}" for index in range(1, 6)),
        *(f"p2r_formal_claude_u{index:03d}" for index in range(0, 5)),
    }
    if set(users) != expected_ids:
        raise ValueError(f"expected the frozen 11-user intake, got {sorted(users)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.output_dir.parent / "pre_repair_audit_sources"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for user_id, user in sorted(users.items()):
        # Preserve the exact machine-accepted pre-repair input in a directory
        # that runtime loaders never scan.  This makes the migration fully
        # reproducible even though the original outputs/ intake is gitignored.
        write_json(archive_dir / f"{user_id}.json", _read(source_by_user[user_id]))
        write_json(args.output_dir / f"{user_id}.json", user)

    manifest = {
        "protocol": "pm-v1.5-v5.3-formal-entity-and-template-grounding-repair-v2",
        "status": "MACHINE_PASS_BATCH_AND_SEMANTIC_REVIEW_PENDING",
        "output_is_formal_tracked_intake": True,
        "users": len(users),
        "repaired_users": sum(
            any(
                action.get("kind") in {"relationship_grounding", "event_session_and_description_grounding"}
                and (
                    action.get("old_name") != action.get("new_name")
                    or action.get("old_valid_from_session") != action.get("new_valid_from_session")
                    or action.get("kind") == "event_session_and_description_grounding"
                )
                for action in actions
            )
            for actions in actions_by_user.values()
        ),
        "contract": str(args.contract.relative_to(ROOT)),
        "contract_sha256": _sha256(args.contract),
        "validator": str(VALIDATOR_PATH.relative_to(ROOT)),
        "validator_sha256": _sha256(VALIDATOR_PATH),
        "source_inputs": [
            {
                "user_id": user_id,
                "original_ignored_path": str(source_by_user[user_id].relative_to(ROOT)),
                "archived_path": str(
                    (archive_dir / f"{user_id}.json").relative_to(ROOT)
                ),
                "sha256": _sha256(source_by_user[user_id]),
            }
            for user_id in sorted(users)
        ],
        "actions_by_user": actions_by_user,
        "validation_summary": {
            user_id: {
                "status": result["status"],
                "hard_issue_count": result["hard_issue_count"],
                "typed_span_core_valid": result["me_compiler"]["typed_exact_span_core_valid"],
                "legacy_core_valid": result["me_compiler"]["legacy_regex_core_valid"],
                "legacy_invalid_pass": result["me_compiler"]["legacy_regex_intended_invalid_pass"],
            }
            for user_id, result in sorted(validation_by_user.items())
        },
    }
    write_json(args.output_dir.parent / "repair_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "users": manifest["users"],
                "repaired_users": manifest["repaired_users"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
