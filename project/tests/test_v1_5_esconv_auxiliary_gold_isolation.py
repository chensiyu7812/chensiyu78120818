"""Reverse tests: ESConv gold response/strategy text must never reach the
generator, judges, or PM feature extraction -- only the evaluator-only
audit_only.jsonl file may hold it.

Mirrors the established pattern in
tests/test_v1_5_esconv_generation_runner.py::test_esconv_generation_never_imports_audit_only.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every real consumer that builds a generator or judge prompt, or extracts PM
# features, from an ESConv-derived state. If a new one is added, it belongs
# in this list too -- that is the point of a reverse test.
REAL_CONSUMER_MODULES = (
    "src/metacom_pm/prompts.py",
    "src/metacom_pm/pm_v2_judging.py",
    "src/metacom_pm/pm_v2_features.py",
    "src/metacom_pm/pm_v2_model.py",
    "src/metacom_pm/esconv_v1_5.py",
    "scripts/v1_5/13b_run_esconv_auxiliary_generation_v1_5.py",
    "scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py",
)

FORBIDDEN_TOKENS = ("gold_response", "gold_strategy")


def _code_without_module_docstring(source: str) -> str:
    """Strip a leading module docstring, which may name these fields as
    documentation (e.g. describing the audit_only boundary), before checking
    that the actual code never references them. Matches the established
    pattern in test_v1_5_esconv_generation_runner.py."""

    _, _, code_after_docstring = source.partition('"""')
    _, _, code = code_after_docstring.partition('"""')
    return code


def test_real_consumers_never_reference_gold_fields_outside_the_audit_builder() -> None:
    for relative_path in REAL_CONSUMER_MODULES:
        source = _code_without_module_docstring(
            (ROOT / relative_path).read_text(encoding="utf-8")
        )
        for token in FORBIDDEN_TOKENS:
            occurrences = source.count(token)
            if relative_path == "src/metacom_pm/esconv_v1_5.py":
                # The builder itself is the ONLY place allowed to read these
                # fields, and only to route them into the audit-only bucket.
                # Every occurrence must be inside an "audit_only"/audit_rows
                # dict literal, never assigned onto a runtime/PM-state row.
                assert occurrences > 0, (
                    f"expected the audit builder to reference {token!r} at "
                    "least once (to isolate it), found none"
                )
                continue
            assert occurrences == 0, (
                f"{relative_path} must never reference {token!r}; found "
                f"{occurrences} occurrence(s)"
            )


def test_esconv_auxiliary_builder_only_writes_gold_fields_to_audit_only() -> None:
    source = (ROOT / "src" / "metacom_pm" / "esconv_v1_5.py").read_text(
        encoding="utf-8"
    )
    # Find the auxiliary builder's function body and confirm gold_response /
    # gold_strategy appear only inside the dict literal appended to
    # bucket["audit_only"], never inside the runtime/pm_v2_states dict
    # literals built earlier in the same loop iteration.
    start = source.index("def build_esconv_v1_5_auxiliary_training_artifacts")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    runtime_dict_start = body.index("runtime = RuntimeState(")
    runtime_dict_end = body.index("pm_state = runtime_to_pmv2_state(")
    runtime_literal = body[runtime_dict_start:runtime_dict_end]
    assert "gold_response" not in runtime_literal
    assert "gold_strategy" not in runtime_literal
    audit_dict_start = body.index('bucket["audit_only"].append(')
    audit_literal = body[audit_dict_start:]
    assert "gold_response" in audit_literal
    assert "gold_strategy" in audit_literal
