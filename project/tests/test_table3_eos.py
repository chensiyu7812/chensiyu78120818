import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import pytest

from metacom_pm.table3_execution import Ledger, StopRun
from metacom_pm.io import canonical_json, sha256_text

spec = importlib.util.spec_from_file_location('eos_runner', Path(__file__).resolve().parents[1] / 'scripts/27_run_table3_eos.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_changed_request_gets_new_id_and_unchanged_is_reused():
    job = {'id': 'old', 'identity': 'main:u:quality:all:0', 'body': {'messages': ['new']}}
    same = {'request_hash': sha256_text(canonical_json(job['body']))}
    assert module.amended_job(job, same)['id'] == 'old'
    assert module.amended_job(job, None)['id'] == 'old'
    different = {'request_hash': sha256_text(canonical_json({'messages': ['old']}))}
    amended = module.amended_job(job, different)
    assert amended['id'] != 'old'
    assert amended == module.amended_job(job, different)
    assert job['id'] == 'old'


def test_import_keeps_prior_fees_and_unresolved_reservations(tmp_path):
    original = Ledger(tmp_path / 'parent.sqlite')
    original.bind('parent')
    jobs = [{'id': name, 'phase': 'main', 'body': {'value': name}, 'reserve_nano': 1_000_000_000} for name in ('paid', 'queued')]
    attempts = original.reserve_many(jobs, 2)
    original.update(attempts[0][0], status='done', actual_nano=400_000_000)
    original.update(attempts[1][0], status='queued')
    target = tmp_path / 'child.sqlite'
    module.import_ledger(target, original.path, 'new', 'parent')
    child = Ledger(target)
    assert child.rows() == original.rows()
    assert child.summary()['charged_upper_usd'] == .4
    assert child.summary()['charged_plus_reserved_usd'] == 1.4
    original.bind('parent')
    child.bind('new')
    module.import_ledger(target, original.path, 'new', 'parent')
    with pytest.raises(StopRun):
        module.import_ledger(target, original.path, 'wrong', 'parent')


def test_observations_exclude_superseded_scores(tmp_path):
    ledger = Ledger(tmp_path / 'ledger.sqlite')
    jobs = []
    for identifier, score in [('superseded', 0), ('active', 3)]:
        job = {'id': identifier, 'phase': 'main', 'body': {'version': identifier}, 'reserve_nano': 0,
               'stage': 'risk', 'unit_id': 'u', 'arm': 'pm_on'}
        aid = ledger.reserve_many([job], 2)[0][0]
        ledger.update(aid, status='done', actual_nano=0, parsed_json=json.dumps({'selected_evidence_misuse': score}))
        jobs.append(job)
    runner = module.EOSRunner.__new__(module.EOSRunner)
    runner.ledger = ledger
    runner.config = {'judge': {}}
    runner.jobs = lambda *a, **kw: [jobs[1]]
    obs = runner.observations('main')
    assert len(obs) == 1
    assert obs[('u', 'pm_on', 'risk', 0)]['selected_evidence_misuse'] == 3


def test_budget_includes_old_spend_and_all_new_phases(tmp_path):
    runner = module.EOSRunner.__new__(module.EOSRunner)
    runner.run_dir = tmp_path
    runner.config = {'judge': {}, 'pilot': {'reference_judge': {}}}
    runner.ledger = SimpleNamespace(summary=lambda: {'charged_plus_reserved_usd': 8.5}, successful=lambda _: None)
    runner.jobs = lambda phase, *a, **kw: [{'id': phase, 'phase': phase, 'reserve_nano': 200_000_000}]
    with pytest.raises(StopRun, match='preflight'):
        runner.preflight()
