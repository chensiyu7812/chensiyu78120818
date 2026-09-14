from pathlib import Path
import csv
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/targeted_checks_20260914'
SEED = 20260915

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def h(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):d.update(chunk)
    return d.hexdigest()

def write(path, x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def csv_write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def cluster_ci(values, labels, seed, n=10000):
    values=np.asarray(values,dtype=float)
    cats=sorted(set(labels)); groups=[np.flatnonzero(np.array(labels)==c) for c in cats]
    sums=np.array([values[g].sum(axis=0) for g in groups]);counts=np.array([len(g) for g in groups])
    draws=np.random.default_rng(seed).integers(0,len(cats),size=(n,len(cats)))
    boots=sums[draws].sum(axis=1)/counts[draws].sum(axis=1).reshape((-1,)+(1,)*(values.ndim-1))
    return np.quantile(boots,[.025,.975],axis=0)

def verify():
    manifest=read(OUT/'freeze.json')
    assert manifest['freeze_sha256']==h({k:v for k,v in manifest.items() if k!='freeze_sha256'})
    for path, expected in manifest['files'].items():
        assert sha(ROOT/path)==expected,path
    return manifest
