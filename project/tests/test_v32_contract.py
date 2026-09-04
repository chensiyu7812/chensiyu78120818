from __future__ import annotations

import pytest

from metacom_pm.v32_contract import (
    V32ContractError,
    canonical_action,
    parse_action,
    required_sources,
    require_exact_keys,
    strict_bool,
    strict_int_012,
    strict_source_list,
)


def test_combo_actions_parse_explicitly() -> None:
    assert parse_action("MPE+RS").memory_sources == frozenset({"MP", "ME"})
    assert parse_action("MSE+R0").memory_sources == frozenset({"MS", "ME"})
    assert parse_action("MPMSME+RS").memory_sources == frozenset({"MP", "MS", "ME"})
    assert required_sources("MSE") == ("MS", "ME")
    assert canonical_action(["MP", "ME"], "RS") == "MPE+RS"


def test_unknown_action_fails_closed() -> None:
    with pytest.raises(V32ContractError):
        parse_action("MEP+RS")


def test_strict_schema_helpers() -> None:
    require_exact_keys({"a": 1}, ["a"])
    with pytest.raises(V32ContractError):
        require_exact_keys({"a": 1, "b": 2}, ["a"])
    assert strict_int_012(2) == 2
    with pytest.raises(V32ContractError):
        strict_int_012("2")
    with pytest.raises(V32ContractError):
        strict_int_012(True)
    assert strict_bool(False) is False
    with pytest.raises(V32ContractError):
        strict_bool(0)
    assert strict_source_list(["MP", "ME"]) == ["MP", "ME"]
    with pytest.raises(V32ContractError):
        strict_source_list(["ME", "ME"])
