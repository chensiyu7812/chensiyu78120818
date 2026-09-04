#!/usr/bin/env python3
"""Batch integrity, duplicate, and external-overlap audit for repaired intake."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.evoemo import load_evoemo
from metacom_pm.io import write_json
from metacom_pm.v1_5_v5_3_external_leakage_audit import (
    compare_text_surface_overlap,
    external_overlap_surfaces,
    normalized_ngrams,
    normalized_tokens,
)


ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1"
USERS_DIR = INTAKE / "canonical_users"
MANIFEST = INTAKE / "repair_manifest.json"
VALIDATOR = ROOT / "scripts/v1_5/82l_validate_formal_longitudinal_user_v1_5.py"
EVOEMO = ROOT / "data/external/evo_emo.json"
ESCONV = ROOT / "data/external/ESConv.json"
OUT = INTAKE / "batch_audit_report.json"


def _load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("formal_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("validator import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, (*path, str(index)))
    elif isinstance(value, str) and value.strip():
        yield path, value


def _formal_surfaces(users: list[dict[str, Any]]) -> list[dict[str, str]]:
    surfaces: list[dict[str, str]] = []
    for user in users:
        uid = str(user["user_id"])
        for session in user["sessions"]:
            sid = int(session["session_index"])
            for turn in session["dialogue"]:
                surfaces.append(
                    {
                        "surface_id": f"{uid}:s{sid}:turn:{turn['turn_id']}",
                        "owner_id": uid,
                        "category": f"dialogue_{turn['role']}",
                        "text": str(turn["text"]),
                    }
                )
            surfaces.append(
                {
                    "surface_id": f"{uid}:s{sid}:summary",
                    "owner_id": uid,
                    "category": "session_summary",
                    "text": str(session["summary"]),
                }
            )
            for candidate in session.get("typed_candidates", []):
                surfaces.append(
                    {
                        "surface_id": f"{uid}:candidate:{candidate['candidate_id']}",
                        "owner_id": uid,
                        "category": str(candidate["subtype"]),
                        "text": str(candidate["literal_source_span"]),
                    }
                )
        for event in user["events"]:
            surfaces.append(
                {
                    "surface_id": f"{uid}:event:{event['event_id']}",
                    "owner_id": uid,
                    "category": "event_description",
                    "text": str(event["description"]),
                }
            )
        for item in [*user["profile_history"], *user["response_preference_history"]]:
            text = str(item.get("literal_source_span") or item.get("preference_text") or "")
            if text:
                surfaces.append(
                    {
                        "surface_id": f"{uid}:item:{item['item_id']}",
                        "owner_id": uid,
                        "category": str(item["subtype"]),
                        "text": text,
                    }
                )
    return surfaces


def _cross_user_overlap(surfaces: list[dict[str, str]]) -> dict[str, Any]:
    eligible = [row for row in surfaces if len(normalized_tokens(row["text"])) >= 8]
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    ngrams: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        exact[" ".join(normalized_tokens(row["text"]))].append(row)
        for ngram in normalized_ngrams(row["text"], n=8):
            ngrams[ngram].append(row)

    exact_groups = []
    for normalized, rows in exact.items():
        if len({row["owner_id"] for row in rows}) > 1:
            exact_groups.append(
                {
                    "text_sha256": _sha(normalized),
                    "surface_ids": sorted(row["surface_id"] for row in rows),
                    "owners": sorted({row["owner_id"] for row in rows}),
                }
            )
    ngram_groups = []
    for ngram, rows in ngrams.items():
        owners = {row["owner_id"] for row in rows}
        if len(owners) > 1:
            ngram_groups.append(
                {
                    "ngram_sha256": _sha(" ".join(ngram)),
                    "surface_ids": sorted({row["surface_id"] for row in rows}),
                    "owners": sorted(owners),
                }
            )
    return {
        "eligible_surface_count": len(eligible),
        "cross_user_exact_groups": len(exact_groups),
        "cross_user_normalized_8gram_groups": len(ngram_groups),
        "exact_groups": exact_groups,
        "normalized_8gram_groups": ngram_groups,
        "raw_text_in_report": False,
    }


def _protected_view(user: dict[str, Any]) -> dict[str, Any]:
    """Fields the entity-grounding migration is not authorized to rewrite."""
    return {
        "protocol": user["protocol"],
        "user_id": user["user_id"],
        "content_author": user["content_author"],
        "primary_superdomain": user["primary_superdomain"],
        "topic_threads": user["topic_threads"],
        "profile_history": user["profile_history"],
        "response_preference_history": user["response_preference_history"],
        "sessions": [
            {
                key: value
                for key, value in session.items()
                if key not in {"entity_ids", "event_ids", "typed_candidates"}
            }
            for session in user["sessions"]
        ],
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "entity_ids"
            }
            for session in user["sessions"]
            for candidate in session.get("typed_candidates", [])
        ],
    }


def _source_fidelity(
    users: list[dict[str, Any]], manifest: dict[str, Any], validator: Any
) -> dict[str, Any]:
    sources = {
        row["user_id"]: ROOT / row["archived_path"]
        for row in manifest["source_inputs"]
    }
    drifts = []
    for repaired in users:
        uid = repaired["user_id"]
        source, _ = validator._normalize_user_schema(_read(sources[uid]))
        for action in manifest["actions_by_user"].get(uid, []):
            if action.get("kind") == "cross_user_template_surface_repair":
                source = _replace_manifest_string(source, str(action["old"]), str(action["new"]))
        if _protected_view(source) != _protected_view(repaired):
            drifts.append(uid)
    return {
        "users_compared": len(users),
        "protected_dialogue_candidate_profile_or_preference_drift_users": drifts,
        "protected_surface_drift_count": len(drifts),
        "authorized_mutable_fields": [
            "three versioned source-turn/literal-span template repairs",
            "relationships",
            "session.entity_ids",
            "session.event_ids",
            "event.entity_ids",
            "candidate.entity_ids",
            "p2r_formal_claude_u003/u003_evt_14 session and grounded description",
        ],
    }


def _replace_manifest_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_manifest_string(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_manifest_string(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _external_surfaces() -> list[dict[str, str]]:
    evoemo = external_overlap_surfaces(load_evoemo(EVOEMO))
    esconv_raw = _read(ESCONV)
    esconv = [
        {
            "surface_id": "esconv:" + ":".join(path),
            "category": "external_esconv_text",
            "text": text,
        }
        for path, text in _iter_strings(esconv_raw)
        if len(normalized_tokens(text)) >= 3
    ]
    return [*evoemo, *esconv]


def main() -> None:
    validator = _load_validator()
    manifest = _read(MANIFEST)
    users = [_read(path) for path in sorted(USERS_DIR.glob("*.json"))]
    surfaces = _formal_surfaces(users)
    internal = [
        {"surface_id": row["surface_id"], "category": row["category"], "text": row["text"]}
        for row in surfaces
        if len(normalized_tokens(row["text"])) >= 3
    ]
    external = _external_surfaces()
    external_overlap = compare_text_surface_overlap(
        internal_surfaces=internal,
        external_surfaces=external,
        ngram_size=8,
    )
    cross_user = _cross_user_overlap(surfaces)
    source_fidelity = _source_fidelity(users, manifest, validator)
    relationship_future_fields = sorted(
        {
            key
            for user in users
            for relationship in user["relationships"]
            for key in relationship
            if key.startswith("status_after_") or key in {"note", "first_mentioned_session", "relation"}
        }
    )
    failures = []
    if source_fidelity["protected_surface_drift_count"]:
        failures.append("protected_source_surface_drift")
    if cross_user["cross_user_exact_groups"] or cross_user["cross_user_normalized_8gram_groups"]:
        failures.append("cross_user_duplicate_or_8gram_overlap")
    if external_overlap["exact_collision_count"] or external_overlap["normalized_ngram_collision_count"]:
        failures.append("external_text_overlap")
    if relationship_future_fields:
        failures.append("relationship_future_or_alias_fields_remain")

    report = {
        "protocol": "pm-v1.5-v5.3-formal-longitudinal-repaired-batch-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "users": len(users),
        "formal_surfaces": len(surfaces),
        "source_fidelity": source_fidelity,
        "cross_user_overlap": cross_user,
        "external_overlap": {
            key: external_overlap[key]
            for key in (
                "ngram_size",
                "internal_surface_count",
                "external_surface_count",
                "internal_surfaces_with_exact_overlap",
                "internal_surfaces_with_normalized_ngram_overlap",
                "exact_collision_count",
                "normalized_ngram_collision_count",
                "collisions_by_external_category",
                "raw_external_text_in_report",
            )
        },
        "external_surface_note": (
            "EvoEmo sessions and embedded ES-MemEval questions/answers come from the shared "
            "released artifact; ESConv is scanned separately."
        ),
        "relationship_future_or_alias_fields": relationship_future_fields,
        "failures": failures,
        "api_calls": 0,
        "training_label_or_outcome_read": False,
    }
    write_json(OUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
