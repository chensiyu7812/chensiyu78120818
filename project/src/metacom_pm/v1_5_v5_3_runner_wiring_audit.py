"""Static preflight for the still-missing V5.3 formal response runner.

This is intentionally a conservative symbol-level audit.  Finding all
required calls in one script is necessary, not sufficient, for a correct
runner.  Finding none prevents the project from mistaking separate pilots
and modules for an integrated formal execution entrypoint.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from .io import sha256_file


REQUIRED_CALLS = {
    "discover_final_typed_memory_candidates": (
        "metacom_pm.v1_5_candidate_discovery",
        "discover_final_typed_memory_candidates",
    ),
    "rs_shared_candidate_top1": (
        "metacom_pm.v1_5_v5_3_candidate_layer_responsibility",
        "rs_shared_candidate_top1",
    ),
    "build_response_baseline_plan": (
        "metacom_pm.v1_5_v5_3_response_baselines",
        "build_response_baseline_plan",
    ),
    "build_typed_response_program": (
        "metacom_pm.v1_5_v5_3_typed_response_program",
        "build_typed_response_program",
    ),
    "execute_typed_response": (
        "metacom_pm.v1_5_v5_3_typed_response_program",
        "execute_typed_response",
    ),
    "StagewiseAccountabilityLedger": (
        "metacom_pm.v1_5_v5_3_accountability_runner",
        "StagewiseAccountabilityLedger",
    ),
}


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def script_call_coverage(path: str | Path) -> set[str]:
    script = Path(path)
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    imported_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        for alias in node.names:
            for label, (module, symbol) in REQUIRED_CALLS.items():
                if node.module == module and alias.name == symbol:
                    imported_aliases[alias.asname or alias.name] = label
    calls = _called_names(tree)
    return {label for alias, label in imported_aliases.items() if alias in calls}


def audit_formal_runner_wiring(
    *, project_root: str | Path, scripts: Iterable[str | Path] | None = None
) -> dict:
    root = Path(project_root).resolve()
    paths = (
        sorted((root / "scripts/v1_5").glob("*.py"))
        if scripts is None
        else [Path(path) for path in scripts]
    )
    coverage_by_script: dict[str, list[str]] = {}
    complete: list[str] = []
    required = set(REQUIRED_CALLS)
    for path in paths:
        coverage = script_call_coverage(path)
        if coverage:
            relative = str(path.resolve().relative_to(root))
            coverage_by_script[relative] = sorted(coverage)
            if coverage == required:
                complete.append(relative)
    bindings = {
        label: {
            "module": module,
            "symbol": symbol,
            "implementation_path": (
                "src/" + module.replace(".", "/") + ".py"
            ),
        }
        for label, (module, symbol) in REQUIRED_CALLS.items()
    }
    for binding in bindings.values():
        implementation = root / binding["implementation_path"]
        binding["implementation_sha256"] = sha256_file(implementation)
    return {
        "protocol": "pm-v1.5-v5.3-formal-runner-static-wiring-audit-v1",
        "status": (
            "STATIC_REQUIRED_CALLS_PRESENT_IN_ONE_ENTRYPOINT_NEEDS_RUNTIME_AUDIT"
            if complete
            else "FORMAL_RUNNER_ENTRYPOINT_NOT_YET_IMPLEMENTED"
        ),
        "audit_scope": "static imported-and-called symbol coverage; necessary not sufficient",
        "required_calls": bindings,
        "coverage_by_existing_script": coverage_by_script,
        "complete_candidate_entrypoints": complete,
        "formal_runner_ready": False,
        "reason_formal_runner_ready_is_false": (
            "no complete entrypoint exists"
            if not complete
            else "static symbol coverage still requires identity, lineage, and dry-run validation"
        ),
        "api_calls": 0,
        "generated_response_or_quality_risk_outcome_read": False,
    }
