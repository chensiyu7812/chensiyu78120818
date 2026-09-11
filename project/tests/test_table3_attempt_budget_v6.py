import importlib.util
from copy import deepcopy
from pathlib import Path

from metacom_pm.io import canonical_json

spec = importlib.util.spec_from_file_location('attempt_budget_v6', Path(__file__).resolve().parents[1] / 'scripts/30_run_table3_attempt_budget_v6.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FIXTURE_PARENT_CONFIG = {
    'protocol': 'fixture-protocol',
    'execution': {'maximum_attempts_per_job': 2, 'input_reservation_multiplier': 1.25,
                  'input_reservation_extra_tokens': 64, 'batch_max_input_tokens': 1000000},
    'judge': {'model': 'fixture-model', 'temperature': 0.0, 'seed': 1},
    'analysis': {'bootstrap_resamples': 10000},
    'pilot_gate': {'max_reference_overall_mae': 0.75},
    'budget': {'execution_reservation_limit_usd': 9.0},
}


def test_only_the_attempt_limit_changes():
    config = module.build_config(FIXTURE_PARENT_CONFIG)
    assert config['execution']['maximum_attempts_per_job'] == module.NEW_MAXIMUM_ATTEMPTS_PER_JOB
    assert config['execution']['maximum_attempts_per_job'] != FIXTURE_PARENT_CONFIG['execution']['maximum_attempts_per_job']
    # Every other execution field, and every other top-level section, is untouched.
    restored_execution = deepcopy(config['execution'])
    restored_execution['maximum_attempts_per_job'] = FIXTURE_PARENT_CONFIG['execution']['maximum_attempts_per_job']
    assert canonical_json(restored_execution) == canonical_json(FIXTURE_PARENT_CONFIG['execution'])
    for key in ('judge', 'analysis', 'pilot_gate', 'budget'):
        assert canonical_json(config[key]) == canonical_json(FIXTURE_PARENT_CONFIG[key])
    assert config['protocol'] != FIXTURE_PARENT_CONFIG['protocol']
    assert 'attempt_budget_amendment' in config
    assert 'attempt_budget_amendment' not in FIXTURE_PARENT_CONFIG


def test_build_config_does_not_mutate_its_input():
    before = canonical_json(FIXTURE_PARENT_CONFIG)
    module.build_config(FIXTURE_PARENT_CONFIG)
    assert canonical_json(FIXTURE_PARENT_CONFIG) == before


def test_validity_logic_is_reused_unchanged_not_redefined():
    # The v6 entry point must not redefine any scoring/validity code itself; it only
    # widens the frozen retry budget and drives the unmodified identity_v5 runner.
    # Both guarantees are checked directly: (1) no such symbols exist in this module's
    # own namespace, so nothing here could silently diverge from identity_v5's rules;
    # (2) the runner it drives is the exact class object identity_v5 defines.
    own_names = set(vars(module))
    assert not own_names & {'IdentityRunner', 'record_response', 'parse_response', 'Runner'}
    assert module.identity.IdentityRunner.__module__ == 'identity_v5'
