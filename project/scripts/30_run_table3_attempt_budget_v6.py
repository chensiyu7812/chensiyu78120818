"""Raise the per-job retry budget only, after confirmed provider-side response
variance stalled one risk-stage audit call at the frozen 2-attempt cap."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, StopRun, verify_freeze

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/table3_under10/frozen_identity_v5'
FROZEN = ROOT / 'outputs/table3_under10/frozen_identity_v6'
ASSETS = ROOT / 'outputs/table3_under10/identity_amendment_v6'
RUN = ROOT / 'outputs/table3_under10/run_identity_v6'
NEW_MAXIMUM_ATTEMPTS_PER_JOB = 5

spec = importlib.util.spec_from_file_location('identity_v5', ROOT / 'scripts/29_run_table3_identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def build_config(parent_config):
    """Only execution.maximum_attempts_per_job changes; every other key is preserved."""
    config = deepcopy(parent_config)
    config['protocol'] += '-attempt-budget-v6'
    config['execution'] = deepcopy(config['execution'])
    config['execution']['maximum_attempts_per_job'] = NEW_MAXIMUM_ATTEMPTS_PER_JOB
    config['attempt_budget_amendment'] = {
        'reason': 'One risk-stage audit job exhausted the frozen 2-attempt cap with two byte-identical '
                  'non-compliant (omission_severity!=0) responses at temperature=0 with a fixed seed. An '
                  'out-of-band diagnostic call (identical body, not entered in any ledger, not a scoring '
                  'attempt) returned a fully compliant response for the same input, showing the two official '
                  'failures were not deterministic.',
        'change': f'Raise execution.maximum_attempts_per_job from 2 to {NEW_MAXIMUM_ATTEMPTS_PER_JOB}, '
                  'uniformly for the whole run. No other execution field changes.',
        'unchanged': 'Every validity/parse rule in parse_response and record_response (including the '
                     'selected-only omission_severity==0 requirement), model, temperature, seed, prompts, '
                     'schema, pilot gates, bootstrap settings and the 9 USD budget ceiling are byte-identical '
                     'to identity_v5. A response is only ever accepted if it independently passes every '
                     'existing check.',
        'no_selection': 'This is a general retry-budget parameter, not a rule targeted at this job\'s content; '
                         'it applies identically to any job that might exhaust the old cap. No response is ever '
                         'chosen for its score.',
        'authorization': "User approved 'expand the retry budget (recommended)' after reviewing the root cause "
                          'and diagnostic evidence, 2026-09-10.',
        'incident_record': str((ASSETS / 'incident.json').relative_to(ROOT)),
    }
    return config


def initialize(manifest):
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / 'ledger.sqlite'
    if not target.exists():
        with sqlite3.connect(f'file:{ASSETS / "parent_ledger.sqlite"}?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
            old = dict(dst.execute('SELECT key,value FROM meta'))
            if old['freeze'] != manifest['parent_freeze_sha256']:
                raise StopRun('Attempt-budget parent ledger mismatch')
            dst.execute("INSERT INTO meta VALUES ('attempt_budget_v6_parent_meta',?)", (canonical_json(old),))
            dst.execute("UPDATE meta SET value=? WHERE key='freeze'", (manifest['freeze_sha256'],))
    Ledger(target).bind(manifest['freeze_sha256'])
    for source in (ASSETS / 'parent_runtime').iterdir():
        dest = RUN / source.name
        if source.is_dir():
            dest.mkdir(exist_ok=True)
            for child in source.iterdir():
                child_dest = dest / child.name
                if child_dest.exists():
                    if sha256_file(child_dest) != sha256_file(child):
                        raise StopRun('Copied parent evidence changed')
                else:
                    shutil.copyfile(child, child_dest)
        elif dest.exists():
            if sha256_file(dest) != sha256_file(source):
                raise StopRun('Copied parent evidence changed')
        else:
            shutil.copyfile(source, dest)


def freeze():
    if FROZEN.exists():
        raise StopRun('Attempt-budget freeze already exists')
    parent = verify_freeze(ROOT, PARENT)
    config = build_config(read_json(PARENT / 'config.json'))
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', read_json(PARENT / 'pilot_units.json'))
    files = dict(parent['files'])
    extra = [PARENT / 'manifest.json', FROZEN / 'config.json', FROZEN / 'pilot_units.json',
             ROOT / 'scripts/30_run_table3_attempt_budget_v6.py', ROOT / 'tests/test_table3_attempt_budget_v6.py',
             ROOT / 'docs/TABLE3_ATTEMPT_BUDGET_V6_CN.md', ASSETS / 'parent_ledger.sqlite', ASSETS / 'incident.json']
    extra.extend(sorted(p for p in (ASSETS / 'parent_runtime').rglob('*') if p.is_file()))
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    manifest = {'status': 'FROZEN_ATTEMPT_BUDGET_REPAIR', 'frozen_at': utc_now(),
                'parent_freeze_sha256': parent['freeze_sha256'], 'files': files, 'packages': parent['packages'],
                'budget_usd': 9, 'planned_states': 204, 'arms': 6}
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(FROZEN / 'manifest.json', manifest)
    print(canonical_json({'status': 'FROZEN', 'freeze_sha256': manifest['freeze_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 'run'])
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
        return
    manifest = verify_freeze(ROOT, FROZEN)
    if args.command == 'verify':
        print(canonical_json({'status': 'PASS', 'freeze_sha256': manifest['freeze_sha256']}))
        return
    initialize(manifest)
    runner = identity.IdentityRunner(ROOT, FROZEN, RUN)
    try:
        runner.run()
    except Exception as exc:
        runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
