from pathlib import Path
import subprocess

from metacom_pm.audit import audit_synthetic_release
from metacom_pm.evoemo import load_evoemo,build_evo_memory
from metacom_pm.io import build_manifest
ROOT=Path(__file__).resolve().parents[1]

def test_packaged_synthetic_audit_passes(tmp_path):
    report=audit_synthetic_release(ROOT/'data/synthetic/runtime_states.jsonl',ROOT/'data/synthetic/memory_backend.jsonl',tmp_path/'audit.json')
    assert report['status']=='PASS'
    assert report['counts']['states']==192
    assert report['counts']['runtime_cards']==1728

def test_evoemo_is_frozen_18_users_401_sessions_34_scenarios():
    users=load_evoemo(ROOT/'data/external/evo_emo.json')
    assert len(users)==18
    assert sum(len(x.get('dialog_history') or []) for x in users)==401
    assert sum(len(x.get('subsequent_topics') or []) for x in users)==34

def test_evo_memory_ignores_gold_event_and_observation_annotations():
    user=load_evoemo(ROOT/'data/external/evo_emo.json')[0]
    items,_=build_evo_memory(user)
    combined=' '.join(x.text for x in items)
    # All items are derived from basic_info, session summaries, or seeker turns.
    assert items
    for session in user.get('dialog_history') or []:
        for obs in session.get('observation') or []:
            content=str(obs.get('content') or '').strip()
            if content and content not in ' '.join(str(t.get('content') or '') for t in session.get('dialogue') or []):
                assert content not in combined


def test_release_manifest_can_exclude_untracked_and_ignored_artifacts(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("private/\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("public\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("local\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / "raw_api_calls.jsonl").write_text("sensitive\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.txt"], cwd=tmp_path, check=True
    )

    manifest = build_manifest(tmp_path, tracked_only=True)
    assert [row["path"] for row in manifest["files"]] == [
        ".gitignore",
        "tracked.txt",
    ]
