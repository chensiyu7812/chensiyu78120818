"""Reverse isolation: no real consumer may reference the diagnostic-only
Hybrid retriever (src/metacom_pm/hybrid_retrieval.py).

Part 3 of the approved PM-v1.5 hybrid-retrieval plan requires
HybridMemoryRetriever/HybridStrategyRetriever to remain a strictly isolated,
report-only diagnostic until a separate, explicitly-approved adoption
decision migrates real consumers. This test scans the source of every real
consumer module/script known to import MemoryRetriever/StrategyRetriever (or
otherwise participate in training, sweep, ESConv, EvoEmo generation, fixed
baselines, freezing, or paid execution) and fails if any of them mention the
Hybrid classes or the hybrid_retrieval module at all -- a plain substring
check, not an attribute probe, so it also catches a bare `import
metacom_pm.hybrid_retrieval` that never even uses the names.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

FORBIDDEN_NEEDLES = (
    "HybridMemoryRetriever",
    "HybridStrategyRetriever",
    "hybrid_retrieval",
)

REAL_CONSUMER_MODULES = (
    "metacom_pm.variance",
    "metacom_pm.esconv",
    "metacom_pm.sweep",
    "metacom_pm.pm_v22_reference_baselines",
    "metacom_pm.v1_5_actual_corpus_review",
    "metacom_pm.evoemo",
    "metacom_pm.pm_v1_5_required_hit",
    "metacom_pm.pm_v2_evoemo",
    "metacom_pm.selection",
)

ROOT = Path(__file__).resolve().parents[1]

REAL_CONSUMER_SCRIPTS = (
    ROOT / "scripts" / "11c_analyze_safety_first_selection.py",
    ROOT / "scripts" / "11d_analyze_coverage_guardrail.py",
)


@pytest.mark.parametrize("module_name", REAL_CONSUMER_MODULES)
def test_real_consumer_module_never_mentions_hybrid_retriever(module_name: str):
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    for needle in FORBIDDEN_NEEDLES:
        assert needle not in source, (
            f"{module_name} must never reference {needle!r} -- Hybrid "
            "retrieval is diagnostic-only until a separately-approved "
            "adoption decision"
        )


@pytest.mark.parametrize("script_path", REAL_CONSUMER_SCRIPTS)
def test_real_consumer_script_never_mentions_hybrid_retriever(script_path: Path):
    assert script_path.is_file(), f"expected real consumer script at {script_path}"
    source = script_path.read_text(encoding="utf-8")
    for needle in FORBIDDEN_NEEDLES:
        assert needle not in source, (
            f"{script_path} must never reference {needle!r} -- Hybrid "
            "retrieval is diagnostic-only until a separately-approved "
            "adoption decision"
        )


def test_hybrid_retrieval_module_itself_is_importable_and_isolated():
    # Sanity: the module exists and exports the two classes this test
    # guards against leaking into real consumers.
    module = importlib.import_module("metacom_pm.hybrid_retrieval")
    assert hasattr(module, "HybridMemoryRetriever")
    assert hasattr(module, "HybridStrategyRetriever")
