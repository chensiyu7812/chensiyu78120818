"""Freeze, verify, start/resume, or inspect the budgeted Table III experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, Runner, StopRun, load_keys, verify_freeze


def freeze(root, target):
    if target.exists():
        raise StopRun('Freeze already exists; verify/resume it instead of overwriting')
    source = root / 'outputs/table3_under10/plan'
    budget_manifest = read_json(source / 'manifest.json')
    for name, expected in budget_manifest['artifact_hashes'].items():
        if sha256_file(source / name) != expected:
            raise StopRun('Approved budget artifact changed')
    original = root / budget_manifest['source_plan']
    base = read_json(original / 'manifest.json')
    if sha256_file(original / 'manifest.json') != budget_manifest['source_manifest_file_sha256']:
        raise StopRun('Original plan manifest changed')
    for category, directory in [('artifact_hashes', original), ('code_hashes', root), ('baseline_hashes', root)]:
        for name, expected in base[category].items():
            if sha256_file(directory / name) != expected:
                raise StopRun(f'Approved input changed: {name}')
    config = deepcopy(read_json(source / 'config.json'))
    config['source_plan'] = str(original.relative_to(root))
    config['execution'] = {
        'maximum_attempts_per_job': 2, 'input_reservation_multiplier': 1.25,
        'input_reservation_extra_tokens': 64, 'batch_max_input_tokens': 1_000_000,
        'batch_poll_seconds': 45, 'simultaneous_batches': 1,
        'generation_order': 'frozen unit order, then frozen arm order',
        'generation_reuse': 'Pilot replies reused in main; identical request hash required.',
        'unknown_usage': 'Keep reservation and stop; never implicitly resubmit.',
        'model_or_schema_fallback': False,
    }
    config['pilot_gate'] = {
        'complete_valid_observations_required': True,
        'max_order_mean_abs_diff': 0.50, 'max_position_mean_shift': 0.40,
        'order_gate_provenance': 'v1 evo_response_v4.run_response_v4 defaults; applied to both judges.',
        'max_reference_overall_mae': 0.75, 'minimum_reference_binary_agreement': 0.80,
        'reference_gate_provenance': 'New operational screening thresholds fixed before any outputs; not established equivalence criteria.',
        'reference_quality_comparison_order': 0,
        'binary_misuse_and_omission_threshold': 2,
        'failure_action': 'Stop before full scoring; preserve all pilot scores; no outcome-directed parameter changes.',
    }
    config['analysis'].update({
        'estimand': 'Mean over all 204 fixed states; equal state weights.',
        'bootstrap': 'Sample clusters with replacement; include all states in each sampled cluster; weighted ratio preserves state-mean estimand.',
        'confidence': 0.95,
        'primary_metrics': ['overall', 'misuse_clear', 'omission_clear'],
        'misuse_clear': 'selected_evidence_misuse >= 2', 'misuse_any': 'selected_evidence_misuse >= 1',
        'omission_clear': 'independent omission_severity >= 2',
        'planned_contrasts': ['ON minus OFF for each policy', 'PM minus Rule and PM minus Fixed at each filter setting'],
        'missing_observations': 'No final table until all 204 x 6 x 3 valid main observations are present.',
        'inference_scope': 'Descriptive paired 95% intervals; no adjusted simultaneous coverage or guaranteed superiority.',
    })
    config['budget']['enforcement_status'] = 'IMPLEMENTED_SQLITE_RESERVATIONS'
    target.mkdir(parents=True)
    write_json(target / 'config.json', config)
    write_json(target / 'pilot_units.json', read_json(source / 'pilot_units.json'))
    paths = set((root / 'src/metacom_pm').glob('*.py'))
    paths.update((root / 'tests').glob('test_table3*.py'))
    paths.update((root / 'scripts').glob('*table3*.py'))
    paths.update(root / p for p in base['baseline_hashes'])
    paths.update(p for p in original.iterdir() if p.is_file())
    paths.update(p for p in source.iterdir() if p.is_file())
    paths.update([target / 'config.json', target / 'pilot_units.json', root / 'configs/table3_requirements.txt',
                  root / 'docs/TABLE3_FROZEN_EXECUTION_CN.md'])
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    manifest = {'protocol': config['protocol'], 'frozen_at': utc_now(), 'status': 'FROZEN_BEFORE_API',
                'user_authorization': '2026-09-10: freeze the approved under-10-USD plan and start running; NVIDIA free.',
                'files': {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}, 'packages': packages,
                'source_manifest_sha256': base['manifest_sha256'],
                'budget_usd': 9, 'planned_states': 204, 'arms': 6,
                'drift_policy': 'Execution refuses changed hashes. Amendments require a new manifest and explicit provenance; never edit this freeze in place.'}
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(target / 'manifest.json', manifest)
    print(canonical_json({'freeze_sha256': manifest['freeze_sha256'], 'files': len(paths), 'path': str(target)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','verify','status','run','export'])
    parser.add_argument('--freeze', type=Path, default=Path('outputs/table3_under10/frozen_v1'))
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/table3_under10/run_v1'))
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen, run_dir = root / args.freeze, root / args.run_dir
    if args.command == 'freeze':
        freeze(root, frozen)
        return
    manifest = verify_freeze(root, frozen)
    if args.command == 'verify':
        print(canonical_json({'status': 'PASS', 'freeze_sha256': manifest['freeze_sha256']}))
        return
    if args.command == 'status':
        print(canonical_json({'status': read_json(run_dir / 'status.json') if (run_dir / 'status.json').exists() else 'NOT_STARTED',
                              'ledger': Ledger(run_dir / 'ledger.sqlite').summary()}))
        return
    load_keys(args.env_file)
    runner = Runner(root, frozen, run_dir)
    try:
        if args.command == 'export':
            with runner.lock():
                runner.export()
        else:
            runner.run()
    except StopRun as exc:
        runner.status('STOPPED', reason=str(exc))
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    try:
        main()
    except StopRun as exc:
        print('STOPPED: ' + str(exc), file=sys.stderr)
        sys.exit(2)
