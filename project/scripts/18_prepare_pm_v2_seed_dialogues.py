#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    sha256_file,
    sha256_text,
    write_json,
)


TRAIN_SPLITS = {"train", "training"}
KNOWN_SPLITS = TRAIN_SPLITS | {
    "validation",
    "valid",
    "val",
    "test",
    "testing",
    "dev",
    "development",
}
TEST_OR_VALIDATION_SPLITS = {
    "validation",
    "valid",
    "val",
    "test",
    "testing",
    "dev",
    "development",
}
SEED_EXTRACTION_CONTRACT_VERSION = "pm-v2-seed-extraction-v1-strict-lineage"


def seed_extraction_code_manifest() -> dict[str, Any]:
    """Hash the extraction implementation and its local hashing/IO dependency."""

    project_root = Path(__file__).resolve().parents[1]
    files = {
        "scripts/18_prepare_pm_v2_seed_dialogues.py": Path(__file__).resolve(),
        "src/metacom_pm/io.py": project_root / "src/metacom_pm/io.py",
    }
    return {
        "contract_version": SEED_EXTRACTION_CONTRACT_VERSION,
        "files": [
            {"path": logical_path, "sha256": sha256_file(path)}
            for logical_path, path in sorted(files.items())
        ],
    }


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}")
                yield row
        return
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("data", "dialogues", "conversations", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported JSON structure in {path}")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"non-object record in {path}")
        yield row


def get_id(row: dict[str, Any]) -> str:
    for key in ("dialogue_id", "dialog_id", "conversation_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return "row_" + sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=False))[:20]


def get_explicit_id(row: dict[str, Any]) -> str | None:
    for key in ("dialogue_id", "dialog_id", "conversation_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def get_split(row: dict[str, Any]) -> str | None:
    for key in ("split", "dataset_split", "partition", "set"):
        if row.get(key) is not None:
            return str(row[key]).strip().lower()
    return None


def get_dialogue_text(row: dict[str, Any]) -> str:
    for key in ("dialogue_text", "text", "content"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    turns = (
        row.get("dialogue")
        or row.get("dialog")
        or row.get("turns")
        or row.get("messages")
    )
    if isinstance(turns, list):
        lines = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or turn.get("speaker") or turn.get("participant") or "unknown"
            content = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
            if str(content).strip():
                lines.append(f"{role}: {str(content).strip()}")
        if lines:
            return "\n".join(lines)
    raise ValueError(f"record {get_id(row)} does not contain dialogue text")


def read_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.add(line.strip())
    return values


def read_split_manifest(path: Path) -> list[dict[str, Any]]:
    expected_fields = {"index", "dialogue_id", "split", "excluded_for_evoemo_overlap"}
    by_index: dict[int, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(iter_records(path), 1):
        if set(raw) != expected_fields:
            raise RuntimeError(
                f"split manifest row {line_number} schema mismatch: "
                f"missing={sorted(expected_fields - set(raw))}, "
                f"extra={sorted(set(raw) - expected_fields)}"
            )
        index = raw["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"split manifest row {line_number} has invalid index")
        if index in by_index:
            raise RuntimeError(f"split manifest has duplicate index {index}")
        dialogue_id = raw["dialogue_id"]
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise RuntimeError(
                f"split manifest row {line_number} has invalid dialogue_id"
            )
        if dialogue_id in seen_ids:
            raise RuntimeError(
                f"split manifest has duplicate dialogue_id {dialogue_id}"
            )
        split = str(raw["split"]).strip().lower()
        if split not in KNOWN_SPLITS:
            raise RuntimeError(
                f"split manifest row {line_number} has unknown split {split!r}"
            )
        excluded = raw["excluded_for_evoemo_overlap"]
        if not isinstance(excluded, bool):
            raise RuntimeError(
                f"split manifest row {line_number} has non-boolean overlap exclusion"
            )
        normalized = {
            "index": index,
            "dialogue_id": dialogue_id,
            "split": split,
            "excluded_for_evoemo_overlap": excluded,
        }
        by_index[index] = normalized
        seen_ids.add(dialogue_id)
    expected_indexes = list(range(len(by_index)))
    if sorted(by_index) != expected_indexes:
        raise RuntimeError(
            "split manifest indexes must be unique and continuous from zero: "
            f"observed_head={sorted(by_index)[:20]}, expected_rows={len(by_index)}"
        )
    if not by_index:
        raise RuntimeError("split manifest is empty")
    return [by_index[index] for index in expected_indexes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--train-ids",
        type=Path,
        help="optional explicit train dialogue IDs; required when source rows lack split",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "strict index-aligned split/overlap manifest; cannot be combined with "
            "--train-ids"
        ),
    )
    parser.add_argument("--excluded-ids", type=Path)
    parser.add_argument("--minimum-seeds", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.split_manifest is not None and args.train_ids is not None:
        raise RuntimeError(
            "--split-manifest is authoritative; --train-ids cannot override it"
        )
    train_ids = read_id_file(args.train_ids)
    excluded_ids = read_id_file(args.excluded_ids)
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    observed_ids: set[str] = set()
    seen_text: set[str] = set()
    source_hashes = {str(path): sha256_file(path) for path in args.inputs}
    source_rows = [
        (path, row) for path in args.inputs for row in iter_records(path)
    ]
    source_manifest = [
        {
            "path": str(path.resolve()),
            "sha256": source_hashes[str(path)],
        }
        for path in args.inputs
    ]
    split_manifest_rows = (
        read_split_manifest(args.split_manifest)
        if args.split_manifest is not None
        else None
    )
    split_manifest_sha256 = (
        sha256_file(args.split_manifest) if args.split_manifest is not None else None
    )
    if split_manifest_rows is not None and len(split_manifest_rows) != len(source_rows):
        raise RuntimeError(
            "split manifest/source row coverage mismatch: "
            f"manifest_rows={len(split_manifest_rows)}, source_rows={len(source_rows)}"
        )
    eligible_manifest_train_ids = (
        sorted(
            row["dialogue_id"]
            for row in split_manifest_rows
            if row["split"] in TRAIN_SPLITS
            and not row["excluded_for_evoemo_overlap"]
        )
        if split_manifest_rows is not None
        else []
    )
    if split_manifest_rows is not None:
        train_manifest = {
            "mode": "index_joined_split_manifest",
            "path": str(args.split_manifest.resolve()),
            "file_sha256": split_manifest_sha256,
            "eligible_train_id_count": len(eligible_manifest_train_ids),
            "eligible_train_ids_sha256": sha256_text(
                canonical_json(eligible_manifest_train_ids)
            ),
        }
    else:
        train_manifest = {
            "mode": (
                "explicit_ids"
                if args.train_ids is not None
                else "declared_train_split"
            ),
            "path": str(args.train_ids.resolve()) if args.train_ids is not None else None,
            "file_sha256": (
                sha256_file(args.train_ids) if args.train_ids is not None else None
            ),
            "ids": sorted(train_ids),
            "ids_sha256": sha256_text(canonical_json(sorted(train_ids))),
        }
    exclusion_manifest = {
        "path": str(args.excluded_ids.resolve()) if args.excluded_ids is not None else None,
        "file_sha256": (
            sha256_file(args.excluded_ids) if args.excluded_ids is not None else None
        ),
        "ids": sorted(excluded_ids),
        "ids_sha256": sha256_text(canonical_json(sorted(excluded_ids))),
    }
    code_manifest = seed_extraction_code_manifest()
    code_manifest_sha256 = sha256_text(canonical_json(code_manifest))
    split_join_rows = []
    if split_manifest_rows is not None:
        for index, ((path, source_row), manifest_row) in enumerate(
            zip(source_rows, split_manifest_rows, strict=True)
        ):
            split_join_rows.append(
                {
                    "index": index,
                    "source_path": str(path.resolve()),
                    "source_sha256": source_hashes[str(path)],
                    "source_row_sha256": sha256_text(canonical_json(source_row)),
                    "dialogue_id": manifest_row["dialogue_id"],
                    "split": manifest_row["split"],
                    "excluded_for_evoemo_overlap": manifest_row[
                        "excluded_for_evoemo_overlap"
                    ],
                }
            )
    lineage_manifest_hashes = {
        "code_manifest_sha256": code_manifest_sha256,
        "source_manifest_sha256": sha256_text(canonical_json(source_manifest)),
        "train_manifest_sha256": sha256_text(canonical_json(train_manifest)),
        "exclusion_manifest_sha256": sha256_text(
            canonical_json(exclusion_manifest)
        ),
    }
    if split_manifest_rows is not None:
        lineage_manifest_hashes["split_join_manifest_sha256"] = sha256_text(
            canonical_json(split_join_rows)
        )
    split_counts: dict[str, int] = {}
    overlap_excluded_by_split: dict[str, int] = {}
    source_declared_id_checks = 0
    for source_index, (path, row) in enumerate(source_rows):
        manifest_row = (
            split_manifest_rows[source_index]
            if split_manifest_rows is not None
            else None
        )
        if manifest_row is not None:
            dialogue_id = str(manifest_row["dialogue_id"])
            split = str(manifest_row["split"])
            explicit_source_id = get_explicit_id(row)
            if explicit_source_id is not None:
                source_declared_id_checks += 1
                if explicit_source_id != dialogue_id:
                    raise RuntimeError(
                        f"source/manifest dialogue ID mismatch at index {source_index}: "
                        f"source={explicit_source_id!r}, manifest={dialogue_id!r}"
                    )
            source_split = get_split(row)
            if source_split is not None and source_split != split:
                raise RuntimeError(
                    f"source/manifest split mismatch at index {source_index}: "
                    f"source={source_split!r}, manifest={split!r}"
                )
            overlap_excluded = bool(
                manifest_row["excluded_for_evoemo_overlap"]
            )
            if overlap_excluded:
                overlap_excluded_by_split[split] = (
                    overlap_excluded_by_split.get(split, 0) + 1
                )
            is_train = split in TRAIN_SPLITS and not overlap_excluded
        else:
            dialogue_id = get_id(row)
            split = get_split(row)
            is_train = (
                dialogue_id in train_ids
                if args.train_ids is not None
                else split in TRAIN_SPLITS
            )
            if is_train and split is not None and split not in TRAIN_SPLITS:
                raise RuntimeError(
                    f"explicit train ID {dialogue_id} has declared non-train source "
                    f"split {split!r}; refusing cross-split seed contamination"
                )
        observed_ids.add(dialogue_id)
        split_counts[split or "missing"] = split_counts.get(split or "missing", 0) + 1
        if not is_train:
            continue
        if dialogue_id in excluded_ids:
            continue
        if dialogue_id in seen_ids:
            raise RuntimeError(f"duplicate train dialogue id: {dialogue_id}")
        dialogue_text = get_dialogue_text(row)
        normalized = " ".join(dialogue_text.lower().split())
        if normalized in seen_text:
            continue
        seen_ids.add(dialogue_id)
        seen_text.add(normalized)
        output_row = {
            "dialogue_id": dialogue_id,
            "dialogue_text": dialogue_text,
            "source_path": str(path),
            "source_sha256": source_hashes[str(path)],
            "source_split": split or "explicit_train_id",
            "seed_text_sha256": sha256_text(dialogue_text),
            "seed_extraction_contract_version": SEED_EXTRACTION_CONTRACT_VERSION,
            **lineage_manifest_hashes,
        }
        if manifest_row is not None:
            output_row.update(
                {
                    "split_manifest_index": source_index,
                    "split_manifest_sha256": split_manifest_sha256,
                    "excluded_for_evoemo_overlap": False,
                }
            )
        selected.append(output_row)
    if (
        args.train_ids is None
        and split_manifest_rows is None
        and split_counts.get("missing", 0)
    ):
        raise RuntimeError(
            "source contains rows without split metadata. Supply --train-ids; "
            "PM-v2 refuses to infer train membership."
        )
    missing_train_ids = sorted(train_ids - observed_ids) if args.train_ids is not None else []
    if missing_train_ids:
        raise RuntimeError(
            "explicit train manifest contains IDs absent from the supplied source files: "
            f"{missing_train_ids[:20]}"
        )
    if len(selected) < args.minimum_seeds:
        raise RuntimeError(
            f"only {len(selected)} unique train seeds selected; required {args.minimum_seeds}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
    args.out.write_text("", encoding="utf-8")
    for row in selected:
        append_jsonl(args.out, row)
    output_split_counts: dict[str, int] = {}
    for row in selected:
        output_split = str(row["source_split"])
        output_split_counts[output_split] = output_split_counts.get(output_split, 0) + 1
    test_or_validation_rows = sum(
        count
        for split, count in output_split_counts.items()
        if split in TEST_OR_VALIDATION_SPLITS
    )
    declared_non_train_rows = sum(
        count
        for split, count in output_split_counts.items()
        if split != "explicit_train_id" and split not in TRAIN_SPLITS
    )
    report = {
        "status": "COMPLETE",
        "seed_extraction_contract_version": SEED_EXTRACTION_CONTRACT_VERSION,
        "code_manifest": code_manifest,
        "code_manifest_sha256": code_manifest_sha256,
        "inputs": [str(path) for path in args.inputs],
        "input_sha256": source_hashes,
        "source_manifest": source_manifest,
        "source_row_count": len(source_rows),
        "train_manifest": train_manifest,
        "exclusion_manifest": exclusion_manifest,
        "lineage_manifest_hashes": lineage_manifest_hashes,
        "split_manifest": (
            {
                "path": str(args.split_manifest.resolve()),
                "sha256": split_manifest_sha256,
                "row_count": len(split_manifest_rows),
                "continuous_index_start": 0,
                "continuous_index_end": len(split_manifest_rows) - 1,
            }
            if split_manifest_rows is not None
            else None
        ),
        "split_manifest_join": {
            "status": "PASS" if split_manifest_rows is not None else "NOT_USED",
            "source_rows": len(source_rows),
            "manifest_rows": (
                len(split_manifest_rows) if split_manifest_rows is not None else 0
            ),
            "covered_source_rows": (
                len(split_join_rows) if split_manifest_rows is not None else 0
            ),
            "source_declared_id_checks": source_declared_id_checks,
            "join_manifest_sha256": lineage_manifest_hashes.get(
                "split_join_manifest_sha256"
            ),
        },
        "split_counts": split_counts,
        "output_split_counts": output_split_counts,
        "explicit_train_id_count": len(train_ids),
        "excluded_id_count": len(excluded_ids),
        "evoemo_overlap_excluded_by_split": overlap_excluded_by_split,
        "evoemo_overlap_excluded_total": sum(overlap_excluded_by_split.values()),
        "evoemo_overlap_excluded_train": sum(
            count
            for split, count in overlap_excluded_by_split.items()
            if split in TRAIN_SPLITS
        ),
        "selected_unique_train_seeds": len(selected),
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
        "test_or_validation_rows_in_output": test_or_validation_rows,
        "declared_non_train_rows_in_output": declared_non_train_rows,
    }
    write_json(args.out.with_suffix(args.out.suffix + ".audit.json"), report)
    print(report)


if __name__ == "__main__":
    main()
