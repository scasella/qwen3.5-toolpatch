from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
BASE = Path('/Users/scasella/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots') / BASE_REVISION
QWEN6 = Path('/Users/scasella/Downloads/qwen-6')  # read-only: recorded runs, captures and the g23v3 edit of the preceding campaign

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def objhash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

def records(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

class Ledger:
    def __init__(self, path: Path): self.path = path
    def add(self, event: str, **data: Any) -> None:
        old = records(self.path)
        row = dict(event=event, time=time.time(), sequence=len(old), previous=objhash(old[-1]) if old else None, **data)
        append(self.path, row)
    def verify(self) -> bool:
        rows = records(self.path)
        return all(r['sequence']==i and r['previous']==(objhash(rows[i-1]) if i else None) for i,r in enumerate(rows))
