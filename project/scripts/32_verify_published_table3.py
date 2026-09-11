"""Verify/restore the published Table III evidence without API calls or dependencies."""
import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'project/outputs/table3_under10'
PACKAGE = BASE / 'published_archive'
RUN = BASE / 'run_identity_v7'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / 'archive_manifest.json').read_text())
    archive = tempfile.TemporaryFile()
    archive_hash = hashlib.sha256()
    for part in manifest['parts']:
        path = PACKAGE / part['file']
        assert sha(path) == part['sha256'] and path.stat().st_size == part['bytes'], path.name
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                archive.write(block)
                archive_hash.update(block)
    assert archive_hash.hexdigest() == manifest['archive_sha256'], 'Archive hash mismatch'
    archive.seek(0)
    expected = {r['path']: r for r in manifest['files']}
    assert len(expected) == len(manifest['files']), 'Duplicate manifest path'
    restored, present, archived = 0, 0, 0
    # Validate every archive member and every existing destination before writes.
    with tarfile.open(fileobj=archive, mode='r:gz') as tar:
        seen = set()
        for member in tar:
            name = member.name
            path = PurePosixPath(name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert name in expected and name not in seen, name
            seen.add(name)
            target = ROOT / name
            assert target.resolve().is_relative_to(ROOT.resolve()), name
            data = tar.extractfile(member).read()
            assert len(data) == expected[name]['bytes'], name
            assert hashlib.sha256(data).hexdigest() == expected[name]['sha256'], name
            if target.exists():
                assert sha(target) == expected[name]['sha256'], f'Existing file differs: {name}'
                present += 1
            else:
                archived += 1
        assert seen == set(expected), 'Archive member set differs'
    if args.restore:
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:gz') as tar:
            for member in tar:
                target = ROOT / member.name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(tar.extractfile(member).read())
                assert sha(target) == expected[member.name]['sha256']
                restored += 1
    archive.close()
    result = json.loads((RUN / 'result_manifest.json').read_text())
    for name, digest in result['hashes'].items():
        assert sha(RUN / name) == digest, name
    qa = json.loads((RUN / 'all_results/QA_MANIFEST.json').read_text())
    for name, digest in qa['output_sha256'].items():
        assert sha(RUN / 'all_results' / name) == digest, name
    freeze = json.loads((BASE / 'frozen_identity_v7/manifest.json').read_text())
    assert freeze['freeze_sha256'] == result['freeze_sha256'] == qa['freeze_sha256']
    missing = []
    for name, digest in freeze['files'].items():
        target = ROOT / 'project' / name
        if target.exists():
            assert sha(target) == digest, name
        else:
            record = expected.get('project/' + name)
            assert record and record['sha256'] == digest, name
            missing.append(name)
    print(json.dumps({'status': 'PASS', 'archived_files': len(expected),
        'matching_present_files': present, 'archive_only_files': archived,
        'restored_files': restored, 'frozen_files_available_after_restore': len(missing),
        'main_calls': 2652, 'responses': 1224, 'api_calls': 0}, ensure_ascii=False))

if __name__ == '__main__':
    main()
