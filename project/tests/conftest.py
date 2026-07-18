from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.contracts import (
    DialogueTurn,
    MemoryBackendRecord,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyCard,
)

ZERO = [0.0] * 64

# These modules intentionally exercise generated/private review artifacts that
# are excluded from the clean review repository. They remain collected and are
# run only in the private artifact vault/release environment. Marking is done at
# collection time so clean CI cannot misrepresent missing local files as code
# failures, while the tests remain visible and auditable.
PRIVATE_ARTIFACT_TEST_MODULES = frozenset(
    {
        "test_pm_v2_generation_review_v8.py",
        "test_pm_v2_generation_review_v9.py",
        "test_v1_5_external_eval_wiring.py",
        "test_v9_generation_contract.py",
    }
)


def pytest_collection_modifyitems(config, items) -> None:
    marker = pytest.mark.requires_private_artifacts
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in PRIVATE_ARTIFACT_TEST_MODULES:
            item.add_marker(marker)


@pytest.fixture
def tiny_state():
    return RuntimeState(
        state_id="state_0123456789abcdef",
        card_id="card_0123456789abcdef",
        user_id="u1",
        split="development",
        semantic_family="test",
        current_user_text="I feel unsettled today.",
        current_session_history=[
            DialogueTurn(role="user", content="It has been a hard week.")
        ],
        current_session_summary="The user has been under pressure.",
        session_index=4,
        inventory={
            MemorySource.MP: SourceCatalog(
                available=True,
                count=1,
                min_age_sessions=3,
                max_age_sessions=3,
                estimated_tokens=10,
                catalog_fingerprint=ZERO,
            ),
            MemorySource.MS: SourceCatalog(
                available=False,
                count=0,
                estimated_tokens=0,
                catalog_fingerprint=ZERO,
            ),
            MemorySource.ME: SourceCatalog(
                available=True,
                count=1,
                min_age_sessions=1,
                max_age_sessions=1,
                estimated_tokens=12,
                catalog_fingerprint=ZERO,
            ),
        },
        allowed_actions=[
            "M0+R0",
            "M0+RS",
            "MP+R0",
            "MP+RS",
            "ME+R0",
            "ME+RS",
            "MPE+R0",
            "MPE+RS",
        ],
    )


@pytest.fixture
def tiny_memories():
    return [
        MemoryItem(
            memory_id="mem_0123456789abcdef",
            source=MemorySource.MP,
            created_session=1,
            text="The user prefers gentle questions.",
        ),
        MemoryItem(
            memory_id="mem_fedcba9876543210",
            source=MemorySource.ME,
            created_session=3,
            text="The user recently changed teams at work.",
        ),
    ]


@pytest.fixture
def tiny_strategy():
    return StrategyCard(
        strategy_id="strat_0123456789abcdef",
        strategy_label="Reflection of feelings",
        retrieval_text="The user feels uncertain after a change.",
        guidance_text="Reflect the feeling before offering advice.",
        example_response="It sounds like this change has left you feeling unsteady.",
        source_dialogue_id="esconv_train_1",
        source_turn_index=3,
    )
