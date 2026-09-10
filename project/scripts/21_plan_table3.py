#!/usr/bin/env python3
"""Materialize the complete v1 Table III candidate/arm plan without inference APIs."""
import argparse
import json
from pathlib import Path
from metacom_pm.table3 import write_plan

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/table3_rerun.json')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/table3_rerun_v1/plan')
    args = parser.parse_args()
    print(json.dumps(write_plan(ROOT,args.config,args.out),ensure_ascii=False,indent=2))
