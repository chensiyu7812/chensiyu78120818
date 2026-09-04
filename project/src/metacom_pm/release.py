from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import re
import subprocess
import sys

from .audit import audit_synthetic_release
from .config import confirmatory_model_independence
from .evoemo import load_evoemo
from .contracts import StrategyCard
from .io import iter_jsonl
from .io import build_manifest, read_json, sha256_file, utc_now, write_json
from .freeze import verify_study_freeze


FORBIDDEN_SOURCE_PATTERNS = {
    "hardcoded_home": re.compile(r"/home/(?!runner/work)"),
    "legacy_gold_proxy": re.compile(r"gold_memory_action|need_type_proxy|selected_evidence_text"),
    "test_tuning_marker": re.compile(r"claim_ready"),
}


def _is_current_release_python(root: Path, path: Path) -> bool:
    """Return True for files that are part of the current MetaCom-PM release.

    The user's working tree also contains older V1/V31/V32 recovery scripts.
    They are useful legacy artifacts, but they intentionally contain words such
    as ``gold`` and ``claim_ready`` for audit/reproduction.  Release preflight
    must fail closed for the current package while not treating archived helper
    scripts as active confirmatory code.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "src" and len(parts) >= 2 and parts[1] == "metacom_pm":
        return True
    if parts[0] == "tests":
        return True
    if parts[0] == "scripts":
        name = path.name
        return bool(re.match(r"^(?:\d{2}[a-z]?_|99_).+\.py$", name))
    return False


def _scan_python(root: Path) -> dict[str, Any]:
    syntax_errors: list[str] = []
    findings: dict[str, list[str]] = {key: [] for key in FORBIDDEN_SOURCE_PATTERNS}
    files = [
        path for path in sorted(root.rglob("*.py"))
        if _is_current_release_python(root, path)
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(root))
        try:
            ast.parse(text, filename=rel)
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}:{exc.lineno}:{exc.msg}")
        if path.name != "release.py":
            for name, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
                if pattern.search(text):
                    findings[name].append(rel)
    return {"n_python_files": len(files), "syntax_errors": syntax_errors, "findings": findings}


def run_release_preflight(
    root: str | Path,
    out_path: str | Path,
    *,
    run_tests: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    checks: dict[str, dict[str, Any]] = {}
    scan = _scan_python(root)
    checks["python_syntax"] = {"passed": not scan["syntax_errors"], "details": scan}
    prohibited = {
        key: value for key, value in scan["findings"].items() if value
    }
    checks["static_leakage_scan"] = {"passed": not prohibited, "details": prohibited}

    runtime = root / "data/synthetic/runtime_states.jsonl"
    backend = root / "data/synthetic/memory_backend.jsonl"
    try:
        synthetic = audit_synthetic_release(
            runtime, backend, root / "data/synthetic/audit_report.json"
        )
        checks["synthetic_data_audit"] = {"passed": synthetic["status"] == "PASS", "details": synthetic}
    except Exception as exc:
        checks["synthetic_data_audit"] = {"passed": False, "details": str(exc)}

    graph_path = root / "data/synthetic/pair_graph_audit.json"
    try:
        graph = read_json(graph_path)
        checks["pair_graph_audit"] = {"passed": bool(graph.get("ok")), "details": {
            "n_states": graph.get("n_states"),
            "n_training_pairs": graph.get("n_training_pairs"),
            "errors": graph.get("errors"),
        }}
    except Exception as exc:
        checks["pair_graph_audit"] = {"passed": False, "details": str(exc)}

    evo_path = root / "data/external/evo_emo.json"
    try:
        users = load_evoemo(evo_path)
        checks["evoemo_frozen_structure"] = {"passed": True, "details": {
            "users": len(users),
            "sessions": sum(len(x.get("dialog_history") or []) for x in users),
            "scenarios": sum(len(x.get("subsequent_topics") or []) for x in users),
            "sha256": sha256_file(evo_path),
        }}
    except Exception as exc:
        checks["evoemo_frozen_structure"] = {"passed": False, "details": str(exc)}

    strategy_path = root / "data/strategy/strategy_cards.jsonl"
    strategy_audit_path = root / "data/strategy/strategy_bank_audit.json"
    try:
        strategy_cards = [StrategyCard.model_validate(x) for x in iter_jsonl(strategy_path)]
        strategy_audit = read_json(strategy_audit_path)
        prebuilt = strategy_audit.get("status") == "PREBUILT_PILOT_BANK"
        checks["strategy_bank_valid"] = {
            "passed": bool(strategy_cards),
            "details": {
                "n_cards": len(strategy_cards),
                "pilot_only_prebuilt": prebuilt,
                "audit": strategy_audit,
            },
        }
    except Exception as exc:
        prebuilt = True
        checks["strategy_bank_valid"] = {"passed": False, "details": str(exc)}

    config_path = root / "configs/experiment.yaml"
    config_schema_errors: list[str] = []
    config_fill_warnings: list[str] = []
    config_value: dict[str, Any] | None = None
    config_filled = False
    if not config_path.is_file():
        config_schema_errors.append(f"missing configuration: {config_path}")
    else:
        try:
            import yaml
            config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            endpoints = (config_value or {}).get("endpoints") or {}
            for endpoint_name in ("generator", "seeker", "training_judge", "final_judge"):
                raw = endpoints.get(endpoint_name) or {}
                if not isinstance(raw, dict):
                    config_schema_errors.append(f"{endpoint_name} must be a mapping")
                    continue
                for key in ("base_url", "model", "api_key_env"):
                    value = str(raw.get(key, "")).strip()
                    if not value:
                        config_schema_errors.append(f"{endpoint_name}.{key} is empty")
                    elif key == "model" and value.startswith("YOUR_"):
                        config_fill_warnings.append(f"{endpoint_name}.model is still a placeholder")
            protocol = (config_value or {}).get("protocol") or {}
            if not protocol.get("robustness_seeds"):
                config_schema_errors.append("protocol.robustness_seeds is empty")
            config_filled = not config_schema_errors and not config_fill_warnings
        except Exception as exc:
            config_schema_errors.append(f"invalid configuration: {exc}")
    checks["configuration_present"] = {
        "passed": not config_schema_errors,
        "details": {
            "path": str(config_path),
            "schema_errors": config_schema_errors,
            "fill_warnings": config_fill_warnings,
            "filled_for_api": config_filled,
        },
    }
    if config_value is not None:
        model_independence = confirmatory_model_independence(config_value)
    else:
        model_independence = {
            "ok": False,
            "errors": ["configuration could not be parsed"],
            "families": {},
        }
    checks["model_family_independence"] = {
        "passed": bool(model_independence.get("ok")),
        "details": model_independence,
    }
    docs = [
        root / "README_CN.md",
        root / "docs/RUNBOOK_CN.md",
        root / "docs/STUDY_PROTOCOL_CN.md",
        root / "docs/MEASUREMENT_PROTOCOL_V2.md",
        root / "docs/MEASUREMENT_PROTOCOL_V3.md",
    ]
    checks["documentation_present"] = {"passed": all(x.is_file() for x in docs), "details": [str(x) for x in docs]}

    if run_tests:
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(root / "src")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
        )
        checks["pytest"] = {"passed": proc.returncode == 0, "details": {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-2000:],
        }}

    # Build the manifest last and exclude the output itself to make it stable.
    manifest = build_manifest(
        root,
        exclude={Path(out_path).name, "release_manifest.json"},
        tracked_only=True,
    )
    write_json(root / "release_manifest.json", manifest)
    passed = all(value["passed"] for value in checks.values())
    esconv_path = root / "data/external/ESConv.json"
    evoemo_path = root / "data/external/evo_emo.json"
    strategy_audit_path = root / "data/strategy/strategy_bank_audit.json"
    split_manifest_path = root / "data/strategy/esconv_split_manifest.jsonl"
    esconv_test_paths = [
        root / "data/esconv_test/runtime_states.jsonl",
        root / "data/esconv_test/memory_backend.jsonl",
        root / "data/esconv_test/audit_only.jsonl",
    ]
    confirmatory_data_ok = False
    try:
        strategy_audit_value = read_json(strategy_audit_path)
        confirmatory_data_ok = bool(
            esconv_path.is_file()
            and evoemo_path.is_file()
            and split_manifest_path.is_file()
            and all(path.is_file() for path in esconv_test_paths)
            and strategy_audit_value.get("status") != "PREBUILT_PILOT_BANK"
            and strategy_audit_value.get("esconv_sha256") == sha256_file(esconv_path)
            and strategy_audit_value.get("evoemo_sha256") == sha256_file(evoemo_path)
            and isinstance(strategy_audit_value.get("overlaps"), dict)
            and int(strategy_audit_value.get("n_strategy_cards", 0)) > 0
        )
    except Exception:
        confirmatory_data_ok = False

    freeze_path = root / "outputs/study_freeze.json"
    freeze_result = {"ok": False, "errors": ["study freeze not created"]}
    if freeze_path.is_file():
        try:
            freeze_result = verify_study_freeze(freeze_path, release_root=root)
        except Exception as exc:
            freeze_result = {"ok": False, "errors": [str(exc)]}

    confirmatory_checks = {
        "official_esconv_and_evoemo_present": bool(esconv_path.is_file() and evoemo_path.is_file()),
        "strategy_bank_rebuilt_and_overlap_audited": bool(not prebuilt and confirmatory_data_ok),
        "esconv_test_runtime_present": bool(split_manifest_path.is_file() and all(path.is_file() for path in esconv_test_paths)),
        "configuration_filled": bool(config_filled),
        "model_family_independence": bool(model_independence.get("ok")),
        "study_freeze_valid": bool(freeze_result.get("ok")),
    }
    confirmatory_ready = bool(passed and all(confirmatory_checks.values()))
    status = "CONFIRMATORY_READY" if confirmatory_ready else ("API_PILOT_READY" if passed else "BLOCKED")
    report = {
        "status": status,
        "confirmatory_ready": confirmatory_ready,
        "created_at": utc_now(),
        "checks": checks,
        "confirmatory_checks": confirmatory_checks,
        "freeze_verification": freeze_result,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_files": len(manifest["files"]),
        "scope": (
            "API_PILOT_READY permits only generator/judge calibration. "
            "CONFIRMATORY_READY additionally requires a locally installed or pinned official ESConv file, "
            "an overlap-audited train-only strategy bank, ESConv test runtime, filled model configuration, "
            "and a valid study freeze. The real judge pilot gate must still pass before full judging/training."
        ),
    }
    write_json(out_path, report)
    return report
