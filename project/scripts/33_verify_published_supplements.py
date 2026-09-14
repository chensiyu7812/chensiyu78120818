"""Verify the September 14 PM supplements from a fresh checkout; no API/deps."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'project'
MANIFEST = PROJECT / 'outputs/supplements_20260914_publication/manifest.json'

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def check(base, rel, expected):
    key = PurePosixPath(rel)
    assert not key.is_absolute() and '..' not in key.parts, rel
    path = base / rel
    assert path.resolve().is_relative_to(base.resolve()), rel
    assert path.is_file(), f'Missing {path}; restore the earlier Table III archive for source dependencies.'
    assert sha(path) == expected, f'Hash mismatch: {rel}'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-source-dependencies', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest['files']) == manifest['n_files']
    for rel, info in manifest['files'].items():
        check(ROOT, rel, info['sha256'])
        assert (ROOT / rel).stat().st_size == info['bytes'], rel
    referenced = set()
    if args.check_source_dependencies:
        names = [
            'outputs/reviewer_supplement_20260914/frozen/manifest.json',
            'outputs/reviewer_supplement_20260914/frozen_v2/manifest.json',
            'outputs/reviewer_supplement_20260914/offline/source_manifest.json',
            'outputs/reviewer_supplement_20260914/results/analysis_manifest.json',
            'outputs/reviewer_supplement_20260914/DELIVERY_MANIFEST.json',
            'outputs/targeted_checks_20260914/freeze.json',
            'outputs/targeted_checks_20260914/analysis_manifest.json',
        ]
        for rel in names:
            data = json.loads((PROJECT / rel).read_text())
            for path, expected in data.get('sources', data.get('files', {})).items():
                check(PROJECT, path, expected)
                referenced.add(path)
    result = json.loads((PROJECT / 'outputs/targeted_checks_20260914/verification.json').read_text())
    assert result['status'] == 'PASS' and result['latency_requests'] == 720
    assert result['reference_responses'] == {'matched': 432}
    assert result['shuffle']['cards'] == 1728
    print(json.dumps({'status':'PASS','published_files':len(manifest['files']),
        'verified_source_dependencies':len(referenced),'latency_requests':720,
        'development_cards':1728,'api_calls':0}))

if __name__ == '__main__':
    main()
