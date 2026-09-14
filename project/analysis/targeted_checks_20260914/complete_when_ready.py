"""Finish the authorized local analysis after the frozen GPU run completes."""
import os
from pathlib import Path
import subprocess
import sys
import time
from common import OUT,ROOT,read,write

AUDIT=Path('/home/tokkio/audits/pm-v1-targeted-completion-20260914')

def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    freeze=read(OUT/'freeze.json')['freeze_sha256']
    write(AUDIT/'status.json',dict(stage='WAITING_FOR_FROZEN_RUN',freeze_sha256=freeze))
    while True:
        status=read(OUT/'latency/status.json')
        assert status['freeze_sha256']==freeze
        if status['stage']=='COMPLETE':break
        if status['stage']=='FAILED':
            write(AUDIT/'status.json',dict(stage='LATENCY_FAILED_REQUIRES_REVIEW',run_status=status));return 2
        try:os.kill(status['pid'],0)
        except ProcessLookupError:
            write(AUDIT/'status.json',dict(stage='LATENCY_PROCESS_STOPPED_REQUIRES_REVIEW',run_status=status));return 2
        time.sleep(30)
    write(AUDIT/'status.json',dict(stage='ANALYZING',freeze_sha256=freeze))
    r=subprocess.run([sys.executable,'analysis/targeted_checks_20260914/analyze_results.py'],cwd=ROOT)
    write(AUDIT/'status.json',dict(stage='COMPLETE' if r.returncode==0 else 'ANALYSIS_FAILED_REQUIRES_REVIEW',exit_code=r.returncode,report=str(OUT/'README_CN.md'),freeze_sha256=freeze))
    return r.returncode

if __name__=='__main__':raise SystemExit(main())
