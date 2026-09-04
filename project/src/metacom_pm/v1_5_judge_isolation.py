"""Fail-closed development/final judge-role isolation for PM-v1.5."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .api import Endpoint, endpoint_transport
from .config import endpoint_from_config


JUDGE_ROLE_ISOLATION_PROTOCOL = "pm-v1.5-judge-role-isolation-v1"


def _normalized(value: str | None) -> str:
    return str(value or "").strip().casefold()


def _endpoint_descriptor(name: str, endpoint: Endpoint) -> dict[str, str]:
    return {
        "name": name,
        "family": str(endpoint.family or ""),
        "model": endpoint.model,
        "base_url": endpoint.base_url.rstrip("/"),
        "transport": endpoint_transport(endpoint),
    }


def _strings(value: Any, *, context: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{context} must be a string or list of strings")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{context} contains an empty or non-string endpoint")
    return [str(item).strip() for item in values]


def final_judge_endpoint_names(pm_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every endpoint assigned a final-evaluation judging role."""

    external = pm_config.get("external_evaluation")
    if not isinstance(external, Mapping):
        raise ValueError("PM-v1.5 config lacks external_evaluation")
    batched = external.get("batched_judging")
    forced_swap = external.get("forced_swap")
    if not isinstance(batched, Mapping) or not isinstance(forced_swap, Mapping):
        raise ValueError("PM-v1.5 external judge configuration is incomplete")
    risk_audit = batched.get("risk_audit")
    if not isinstance(risk_audit, Mapping):
        raise ValueError("PM-v1.5 batched risk-audit configuration is incomplete")

    names: list[str] = []
    for context, value in (
        (
            "external_evaluation.external_judge_endpoints",
            external.get("external_judge_endpoints"),
        ),
        (
            "external_evaluation.batched_judging.primary_quality_endpoint",
            batched.get("primary_quality_endpoint"),
        ),
        (
            "external_evaluation.batched_judging.sensitivity_quality_endpoint",
            batched.get("sensitivity_quality_endpoint"),
        ),
        (
            "external_evaluation.batched_judging.risk_audit.judge_endpoints",
            risk_audit.get("judge_endpoints"),
        ),
        (
            "external_evaluation.forced_swap.judge_endpoints",
            forced_swap.get("judge_endpoints"),
        ),
    ):
        names.extend(_strings(value, context=context))
    return tuple(dict.fromkeys(names))


def _overlap(
    development: Iterable[dict[str, str]],
    final: Iterable[dict[str, str]],
    key: str,
) -> list[str]:
    development_values = {_normalized(row[key]) for row in development}
    final_values = {_normalized(row[key]) for row in final}
    return sorted(value for value in development_values & final_values if value)


def judge_role_isolation_report(
    experiment_config: Mapping[str, Any],
    pm_config: Mapping[str, Any],
    *,
    development_endpoint_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Check aliases and resolved identities across development/final roles."""

    if pm_config.get("version") != "pm-v1.5":
        raise ValueError("judge-role isolation requires a PM-v1.5 config")
    if development_endpoint_names is None:
        development = pm_config.get("development_judging")
        if not isinstance(development, Mapping):
            raise ValueError("PM-v1.5 config lacks development_judging")
        development_endpoint_names = _strings(
            development.get("judge_endpoints"),
            context="development_judging.judge_endpoints",
        )
    else:
        override_value = (
            development_endpoint_names
            if isinstance(development_endpoint_names, str)
            else list(development_endpoint_names)
        )
        development_endpoint_names = _strings(
            override_value,
            context="development judge endpoint override",
        )

    development_names = tuple(dict.fromkeys(development_endpoint_names))
    final_names = final_judge_endpoint_names(pm_config)
    development_rows = [
        _endpoint_descriptor(name, endpoint_from_config(experiment_config, name))
        for name in development_names
    ]
    final_rows = [
        _endpoint_descriptor(name, endpoint_from_config(experiment_config, name))
        for name in final_names
    ]

    endpoint_name_overlap = sorted(set(development_names) & set(final_names))
    family_overlap = _overlap(development_rows, final_rows, "family")
    model_overlap = _overlap(development_rows, final_rows, "model")
    development_routes = {
        (
            _normalized(row["base_url"]),
            _normalized(row["model"]),
            _normalized(row["transport"]),
        )
        for row in development_rows
    }
    final_routes = {
        (
            _normalized(row["base_url"]),
            _normalized(row["model"]),
            _normalized(row["transport"]),
        )
        for row in final_rows
    }
    route_overlap = [
        {"base_url": base_url, "model": model, "transport": transport}
        for base_url, model, transport in sorted(development_routes & final_routes)
    ]

    errors: list[str] = []
    if len(development_rows) < 2:
        errors.append("development judging requires at least two endpoints")
    if len(final_rows) < 2:
        errors.append("final judging requires at least two endpoints")
    for role, rows in (("development", development_rows), ("final", final_rows)):
        families = [_normalized(row["family"]) for row in rows]
        if any(not family for family in families):
            errors.append(f"{role} endpoints require declared model families")
        elif len(set(families)) != len(families):
            errors.append(f"{role} endpoints must use distinct model families")
    if endpoint_name_overlap:
        errors.append("development and final endpoint aliases overlap")
    if family_overlap:
        errors.append("development and final declared model families overlap")
    if model_overlap:
        errors.append("development and final model identifiers overlap")
    if route_overlap:
        errors.append("development and final resolved model routes overlap")

    return {
        "protocol": JUDGE_ROLE_ISOLATION_PROTOCOL,
        "status": "PASS" if not errors else "FAIL",
        "development_endpoints": development_rows,
        "final_endpoints": final_rows,
        "overlap": {
            "endpoint_names": endpoint_name_overlap,
            "declared_families": family_overlap,
            "model_identifiers": model_overlap,
            "resolved_routes": route_overlap,
        },
        "errors": errors,
    }


def require_judge_role_isolation(
    experiment_config: Mapping[str, Any],
    pm_config: Mapping[str, Any],
    *,
    development_endpoint_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return an auditable report or stop before any development judge call."""

    report = judge_role_isolation_report(
        experiment_config,
        pm_config,
        development_endpoint_names=development_endpoint_names,
    )
    if report["status"] != "PASS":
        raise RuntimeError(
            "PM-v1.5 development/final judge-role isolation failed: "
            + "; ".join(report["errors"])
        )
    return report
