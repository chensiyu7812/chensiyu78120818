import json
from pathlib import Path

from metacom_pm.v1_5_typed_resource_adapter import (
    CompiledResourceBundle,
    response_guard_errors,
)


ROOT = Path(__file__).resolve().parents[1]


def test_guard_boundary_contract_forbids_lexical_semantic_hard_gates():
    contract = json.loads(
        (ROOT / "data/pm_v1_5_contracts/v5_guard_and_human_outcome_boundary_v1.json").read_text()
    )
    mechanical = contract["three_layers"]["mechanical_experiment_validity"]
    assert mechanical["natural_language_regex_allowed"] is False
    assert contract["single_full_human_panel"]["additional_small_packets_allowed"] is False
    assert contract["post_v2_freeze_rule"].endswith("within this paper.")


def test_free_words_do_not_become_machine_provenance_verdicts():
    empty = CompiledResourceBundle(requested_action_id="M0+R0", directives=())
    for response in (
        "Tell me more.",
        "Take a pause before returning.",
        "You referred to an earlier attempt in your current message.",
        "ME is the word I emphasized.",
    ):
        errors = response_guard_errors(response=response, bundle=empty)
        assert "UNAUTHORIZED_PAST_ATTRIBUTION" not in errors


def test_opaque_internal_id_remains_machine_detectable():
    empty = CompiledResourceBundle(requested_action_id="M0+R0", directives=())
    errors = response_guard_errors(
        response="The internal item was mem_deadbeef1234.", bundle=empty
    )
    assert errors == ("INTERNAL_LABEL_OR_ID_LEAK",)
