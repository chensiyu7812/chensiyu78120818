from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Iterator, Mapping
from datetime import datetime, timezone


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hex(*parts: object, n: int = 16) -> str:
    return sha256_text("\x1f".join(map(str, parts)))[:n]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_keys(path: str | Path, key_fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    p = Path(path)
    if not p.exists():
        return set()
    return {
        tuple(row.get(field) for field in key_fields)
        for row in iter_jsonl(p)
    }


def index_jsonl_unique(
    path: str | Path, key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index JSONL rows and fail on duplicate or missing logical keys.

    A dictionary comprehension silently overwrites duplicate rows, allowing a
    partial rerun or stale record to masquerade as a complete experiment.
    Confirmatory loaders use this strict helper instead.
    """
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    for line_no, row in enumerate(iter_jsonl(p), 1):
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None for value in key):
            raise ValueError(
                f"{p}:{line_no}: missing unique key field(s) {key_fields}"
            )
        if key in result:
            raise ValueError(f"{p}:{line_no}: duplicate logical key {key}")
        result[key] = row
    return result


def unique_jsonl_keys(
    path: str | Path, key_fields: tuple[str, ...]
) -> set[tuple[Any, ...]]:
    return set(index_jsonl_unique(path, key_fields))


def build_manifest(
    root: str | Path,
    exclude: set[str] | None = None,
    *,
    tracked_only: bool = False,
) -> dict[str, Any]:
    """Build a content manifest without accidentally publishing local artifacts.

    Release callers should set ``tracked_only=True``.  This keeps ignored paid
    call logs, private reviewer mappings, generated checkpoints, and other
    vault-owned files out of the manifest even when they exist below ``root``.
    """

    root = Path(root).resolve()
    exclude = exclude or set()
    entries = []
    if tracked_only:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        )
        candidates = [
            root / raw.decode("utf-8")
            for raw in result.stdout.split(b"\0")
            if raw
        ]
        missing = [path for path in candidates if not path.is_file()]
        if missing:
            raise RuntimeError(
                "tracked release files are missing from the worktree: "
                + str([str(path.relative_to(root)) for path in missing[:10]])
            )
    else:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    for path in sorted(candidates):
        rel = path.relative_to(root).as_posix()
        if rel in exclude or any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        entries.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "created_at": utc_now(),
        "root": root.name,
        "files": entries,
        "manifest_sha256": sha256_text(canonical_json(entries)),
    }


def ensure_run_manifest(
    path: str | Path,
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create or verify an immutable run manifest before resuming outputs."""
    path = Path(path)
    normalized = dict(metadata)
    normalized["manifest_sha256"] = sha256_text(canonical_json(metadata))
    if overwrite and path.exists():
        path.unlink()
    if path.exists():
        existing = read_json(path)
        if existing.get("manifest_sha256") != normalized["manifest_sha256"]:
            raise RuntimeError(
                f"run manifest mismatch at {path}; use a new output directory "
                "or explicit overwrite after preserving the old run"
            )
        return existing
    write_json(path, normalized)
    return normalized
