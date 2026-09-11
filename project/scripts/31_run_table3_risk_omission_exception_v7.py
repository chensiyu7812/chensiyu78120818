"""Record a single, named-job exception after the raised retry budget (identity_v6)
was fully exhausted: 5/5 real Batch attempts agreed on every scored field and only
the never-read risk-stage omission_severity guard field stayed non-compliant."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, SCHEMAS, StopRun, verify_freeze

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/table3_under10/frozen_identity_v6'
FROZEN = ROOT / 'outputs/table3_under10/frozen_identity_v7'
ASSETS = ROOT / 'outputs/table3_under10/identity_amendment_v7'
RUN = ROOT / 'outputs/table3_under10/run_identity_v7'
EXCEPTION_JOB_ID = '2f558c28656652943f38ed85e88b2410133c6f36'
EXCEPTION_ATTEMPT_ID = f'{EXCEPTION_JOB_ID}-1'

spec = importlib.util.spec_from_file_location('attempt_budget_v6', ROOT / 'scripts/30_run_table3_attempt_budget_v6.py')
budget_v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_v6)
identity = budget_v6.identity


def build_config(parent_config):
    """Nothing execution-wide changes; only a single named-job exception is recorded."""
    config = deepcopy(parent_config)
    config['protocol'] += '-risk-omission-exception-v7'
    config['risk_omission_exception'] = {
        'scope': f'Exactly one job_id ({EXCEPTION_JOB_ID}); does not relax the selected-only '
                 'omission_severity==0 check for any other main-phase request, past or future.',
        'reason': 'All 5 attempts permitted by the v6 retry-budget increase were consumed; all 5 real Batch '
                  'attempts independently agreed on every field that feeds Table III (verdict, '
                  'selected_evidence_misuse, unnecessary_exposure, stale_or_conflict, unsupported_personal_claim, '
                  'source_set_appropriateness, strategy_overuse, strategy_omission, response_support_sufficiency, '
                  'overall_risk, reason, audit_call_id); only the risk-stage omission_severity guard field, which '
                  'export() never reads into any Table III column (it reads omission_severity from the separate '
                  "omission-stage observation instead), stayed non-zero on all 5 tries.",
        'repair_rule': 'Accept the earliest of the 5 already-obtained, transport-verified attempts verbatim; no '
                       'field is changed, invented, or zeroed, including omission_severity itself.',
        'no_selection': 'The earliest attempt is chosen by timestamp only; all 5 are identical on every field '
                        'that could have been selected among. Superseded attempts and every fee remain in the '
                        'ledger untouched.',
        'authorization': "User was shown the 5/5 attempt evidence and chose 'mark as a permanent exception, skip, "
                          "and finish the remaining ~1580 requests' over reviewing the content personally, "
                          '2026-09-10.',
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
                raise StopRun('Risk-omission-exception parent ledger mismatch')
            dst.execute("INSERT INTO meta VALUES ('risk_omission_exception_v7_parent_meta',?)", (canonical_json(old),))
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


def apply_documented_exception(runner):
    """One-time, named-job reconciliation: accept an already-obtained, exhaustively
    retried, transport-verified response despite its out-of-scope guard-field value."""
    if runner.ledger.successful(EXCEPTION_JOB_ID):
        return None
    row = next((r for r in runner.ledger.rows('main')
                if r['id'] == EXCEPTION_ATTEMPT_ID and r['status'] == 'invalid'), None)
    if row is None:
        raise StopRun('Documented exception attempt not found or already resolved differently')
    job = json.loads(row['job_json'])
    body = json.loads(row['response_json'])
    choice = body['choices'][0]
    if body.get('model') != job['body']['model'] or choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
        raise StopRun('Exception response no longer matches the documented, reviewed incident')
    parsed = SCHEMAS['risk'].model_validate_json(choice['message']['content']).model_dump(mode='json')
    expected = json.loads(job['body']['messages'][1]['content'])['audit_call_id']
    if parsed['audit_call_id'] != expected or parsed['omission_severity'] == 0:
        raise StopRun('Exception response no longer matches the documented, reviewed incident')
    usage = body.get('usage') or {}
    if not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
        raise StopRun('Exception response missing usage; cannot verify recorded cost')
    proof = identity.prove_transport(runner.run_dir, runner.ledger, EXCEPTION_ATTEMPT_ID, job, body)
    proof.update(timestamp=utc_now(), omission_severity_returned=parsed['omission_severity'],
                 rule='Documented single-job exception (scripts/31): all 5 independent Batch attempts agreed on '
                      'every field that feeds any Table III metric; only the never-read risk-stage '
                      'omission_severity guard field stayed nonzero. Raw model output preserved verbatim; no '
                      'score changed, invented or zeroed. See docs/TABLE3_RISK_OMISSION_EXCEPTION_V7_CN.md.')
    destination = runner.run_dir / 'risk_omission_exceptions'
    destination.mkdir(exist_ok=True)
    write_json(destination / f'{EXCEPTION_ATTEMPT_ID}.json', proof)
    runner.ledger.update(EXCEPTION_ATTEMPT_ID, status='done', parsed_json=canonical_json(parsed),
                         error='Documented single-job risk-omission exception; independently verified Batch transport identity')
    return proof


def freeze():
    if FROZEN.exists():
        raise StopRun('Risk-omission-exception freeze already exists')
    parent = verify_freeze(ROOT, PARENT)
    config = build_config(read_json(PARENT / 'config.json'))
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', read_json(PARENT / 'pilot_units.json'))
    files = dict(parent['files'])
    extra = [PARENT / 'manifest.json', FROZEN / 'config.json', FROZEN / 'pilot_units.json',
             ROOT / 'scripts/31_run_table3_risk_omission_exception_v7.py',
             ROOT / 'tests/test_table3_risk_omission_exception_v7.py',
             ROOT / 'docs/TABLE3_RISK_OMISSION_EXCEPTION_V7_CN.md', ASSETS / 'parent_ledger.sqlite', ASSETS / 'incident.json']
    extra.extend(sorted(p for p in (ASSETS / 'parent_runtime').rglob('*') if p.is_file()))
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    manifest = {'status': 'FROZEN_RISK_OMISSION_EXCEPTION', 'frozen_at': utc_now(),
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
        with runner.lock():
            runner.status('RISK_OMISSION_EXCEPTION_REVIEW')
            apply_documented_exception(runner)
        runner.run()
    except Exception as exc:
        runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
