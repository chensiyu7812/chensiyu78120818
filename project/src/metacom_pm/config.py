from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os
import yaml

from .api import Endpoint


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML configuration key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.load(
        Path(path).read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def resolve_path(config_path: str | Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = Path(config_path).resolve().parent.parent / path
    return path.resolve()


def endpoint_from_config(config: Mapping[str, Any], name: str) -> Endpoint:
    endpoints = config.get("endpoints")
    if not isinstance(endpoints, Mapping) or name not in endpoints:
        raise KeyError(f"missing endpoint config: {name}")
    raw = endpoints[name]
    if not isinstance(raw, Mapping):
        raise TypeError(f"endpoint {name} must be a mapping")
    required = {"base_url", "model", "api_key_env"}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"endpoint {name} missing keys {sorted(missing)}")
    return Endpoint(
        base_url=str(raw["base_url"]),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        timeout_seconds=float(raw.get("timeout_seconds", 180.0)),
        family=(str(raw["family"]) if raw.get("family") else None),
        transport=str(raw.get("transport") or "auto"),
        supports_strict_json_schema=bool(
            raw.get("supports_strict_json_schema", True)
        ),
        thinking_mode=str(raw.get("thinking_mode") or "provider_default"),
        gemini_thinking_budget=(
            int(raw["gemini_thinking_budget"])
            if raw.get("gemini_thinking_budget") is not None
            else None
        ),
    )


def _endpoint_family(config: Mapping[str, Any], name: str) -> str | None:
    endpoints = config.get("endpoints")
    if not isinstance(endpoints, Mapping) or name not in endpoints:
        return None
    raw = endpoints[name]
    if not isinstance(raw, Mapping):
        return None
    return str(raw.get("family") or raw.get("model") or name)


def confirmatory_model_independence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check that confirmatory generation/evaluation avoids self-play loops.

    The gate is deliberately conservative: supporter generator, training judge,
    final judge, and at least two seeker simulators should not collapse to the
    same model family.  It returns diagnostics instead of raising so CLI scripts
    can render user-friendly errors.
    """
    errors: list[str] = []
    gen = _endpoint_family(config, "generator")
    train_judge = _endpoint_family(config, "training_judge")
    final_judge = _endpoint_family(config, "final_judge")
    if gen and train_judge and gen == train_judge:
        errors.append("generator and training_judge use the same model family")
    if gen and final_judge and gen == final_judge:
        errors.append("generator and final_judge use the same model family")
    if train_judge and final_judge and train_judge == final_judge:
        errors.append("training_judge and final_judge must be independent families")

    protocol = config.get("protocol") if isinstance(config.get("protocol"), Mapping) else {}
    seeker_names = list(protocol.get("confirmatory_seeker_endpoints") or [])
    if not seeker_names:
        seeker_names = [name for name in ("seeker", "seeker_alt") if _endpoint_family(config, name)]
    seeker_families = [_endpoint_family(config, name) for name in seeker_names]
    seeker_families = [x for x in seeker_families if x]
    if len(seeker_families) < 2:
        errors.append("confirmatory evaluation requires at least two seeker simulator endpoints")
    if len(set(seeker_families)) != len(seeker_families):
        errors.append("seeker simulators must use distinct model families")
    if gen and gen in set(seeker_families):
        errors.append("supporter generator and seeker simulator use the same model family")
    return {"ok": not errors, "errors": errors, "families": {
        "generator": gen,
        "training_judge": train_judge,
        "final_judge": final_judge,
        "seekers": seeker_families,
    }}
