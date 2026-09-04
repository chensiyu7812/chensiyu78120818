from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import re

from .io import canonical_json, sha256_file, sha256_text, utc_now, write_json


def _relative(root: Path, path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"frozen path must be inside release root: {resolved}") from exc


def _hash_paths(root: Path, paths: Sequence[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[_relative(root, path)] = sha256_file(path)
    return dict(sorted(result.items()))


def _is_current_release_python(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "src" and len(parts) >= 2 and parts[1] == "metacom_pm":
        return True
    if parts[0] == "scripts":
        return bool(
            (len(parts) >= 2 and parts[1] == "v1_5")
            or path.name.startswith("v1_5_")
            or re.match(r"^(?:\d{2}[a-z]?_|99_).+\.py$", path.name)
        )
    return False


def _code_paths(root: Path) -> list[Path]:
    paths = [
        *[
            path for path in sorted(root.rglob("*.py"))
            if _is_current_release_python(root, path)
        ],
        root / "pyproject.toml",
        root / "docs" / "STUDY_PROTOCOL_CN.md",
        root / "docs" / "MEASUREMENT_PROTOCOL_V2.md",
        root / "docs" / "MEASUREMENT_PROTOCOL_V3.md",
    ]
    return [path for path in paths if path.is_file()]


def create_study_freeze(
    *,
    release_root: str | Path,
    config_path: str | Path,
    checkpoint_paths: Sequence[str | Path],
    data_paths: Sequence[str | Path],
    prompt_files: Sequence[str | Path],
    out_path: str | Path,
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(release_root).resolve()
    config = Path(config_path).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    frozen = {
        "status": "FROZEN",
        "created_at": utc_now(),
        "release_root_name": root.name,
        "config_path": _relative(root, config),
        "config_sha256": sha256_file(config),
        "checkpoint_hashes": _hash_paths(root, checkpoint_paths),
        "data_hashes": _hash_paths(root, data_paths),
        "prompt_hashes": _hash_paths(root, prompt_files),
        "code_hashes": _hash_paths(root, _code_paths(root)),
        "notes": dict(notes or {}),
    }
    frozen["freeze_sha256"] = sha256_text(canonical_json(frozen))
    write_json(out_path, frozen)
    return frozen


def verify_study_freeze(
    freeze_path: str | Path,
    *,
    release_root: str | Path | None = None,
) -> dict[str, Any]:
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    root = Path(release_root).resolve() if release_root is not None else freeze_path.parent.parent.resolve()
    errors: list[str] = []

    expected_self = freeze.get("freeze_sha256")
    without_self = {k: v for k, v in freeze.items() if k != "freeze_sha256"}
    actual_self = sha256_text(canonical_json(without_self))
    if expected_self != actual_self:
        errors.append("freeze record hash mismatch")

    config = root / str(freeze.get("config_path", ""))
    if not config.is_file() or sha256_file(config) != freeze.get("config_sha256"):
        errors.append("config hash mismatch")

    for section in ("checkpoint_hashes", "data_hashes", "prompt_hashes", "code_hashes"):
        values = freeze.get(section)
        if not isinstance(values, dict) or not values:
            errors.append(f"missing or empty {section}")
            continue
        for rel, expected in values.items():
            path = root / rel
            if not path.is_file():
                errors.append(f"missing frozen file: {rel}")
            elif sha256_file(path) != expected:
                errors.append(f"hash mismatch: {rel}")

    return {
        "ok": not errors,
        "errors": errors,
        "freeze_sha256": expected_self,
        "release_root": str(root),
    }


def require_study_freeze(
    freeze_path: str | Path,
    *,
    release_root: str | Path | None = None,
    config_path: str | Path | None = None,
    required_files: Sequence[str | Path] = (),
    allow_unfrozen_debug: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the exact requested inputs are frozen and unchanged.

    ``release_root`` is normally inferred from ``outputs/study_freeze.json``.
    ``config_path`` and ``required_files`` close the loophole where a valid
    freeze exists but an evaluation script is pointed at a different model,
    selection file, data file, or strategy bank.
    """
    path = Path(freeze_path).resolve()
    if allow_unfrozen_debug:
        return {"ok": True, "debug_unfrozen": True, "freeze_path": str(path)}
    if not path.is_file():
        raise RuntimeError(
            f"study freeze is required for confirmatory evaluation: {path}. "
            "Run scripts/20_freeze_study.py after validation-only tuning."
        )
    root = (
        Path(release_root).resolve()
        if release_root is not None
        else path.parent.parent.resolve()
    )
    result = verify_study_freeze(path, release_root=root)
    errors = list(result["errors"])
    freeze = json.loads(path.read_text(encoding="utf-8"))

    if config_path is not None:
        requested_config = Path(config_path).resolve()
        frozen_config = (root / str(freeze.get("config_path", ""))).resolve()
        if requested_config != frozen_config:
            errors.append(
                f"requested config is not the frozen config: "
                f"{requested_config} != {frozen_config}"
            )
        elif not requested_config.is_file():
            errors.append(f"requested config missing: {requested_config}")
        elif sha256_file(requested_config) != freeze.get("config_sha256"):
            errors.append("requested config hash mismatch")

    frozen_paths: dict[str, str] = {}
    for section in (
        "checkpoint_hashes", "data_hashes", "prompt_hashes", "code_hashes"
    ):
        values = freeze.get(section) or {}
        if isinstance(values, dict):
            frozen_paths.update(values)
    for value in required_files:
        requested = Path(value).resolve()
        try:
            rel = requested.relative_to(root).as_posix()
        except ValueError:
            errors.append(f"requested file is outside release root: {requested}")
            continue
        expected = frozen_paths.get(rel)
        if expected is None:
            errors.append(f"requested file was not frozen: {rel}")
        elif not requested.is_file():
            errors.append(f"requested frozen file missing: {rel}")
        elif sha256_file(requested) != expected:
            errors.append(f"requested frozen file hash mismatch: {rel}")

    if errors:
        raise RuntimeError(
            "Confirmatory evaluation blocked by study-freeze gate:\n- "
            + "\n- ".join(errors)
        )
    return {
        **result,
        "ok": True,
        "freeze_path": str(path),
        "required_files": [str(Path(x).resolve()) for x in required_files],
    }
