#!/usr/bin/env python
"""Copy the BFCL run's raw outputs into the release repo.

  python collect.py --work ~/Downloads/bfcl-toolpatch/work --out <repo>/experiments/transfer_bfcl

* score/: BFCL's CSV tables and the per-category score JSON (first line = accuracy,
  then one record per failed item) for the six registry entries.
* results/<registry>/BFCL_v4_<category>_result.jsonl: BFCL's per-item records.  The
  `inference_log` field (the fully formatted prompt, regenerable from the dataset and
  the chat template with the handler) is dropped to keep the files small; everything
  else — `result` (canonical JSON tool calls or the text), `raw_response` (verbatim
  generation), token counts, latency, finish reason — is kept verbatim.
* Any file over 5 MB is gzipped.
"""
import argparse
import gzip
import json
import pathlib
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gzip-over-mb", type=float, default=5.0)
a = ap.parse_args()
work, out = pathlib.Path(a.work), pathlib.Path(a.out)

# scores
for p in sorted((work / "score").rglob("*")):
    if p.is_file() and p.suffix in (".csv", ".json"):
        rel = p.relative_to(work / "score")
        if p.suffix == ".json":
            rel = rel.parent.parent / rel.name  # drop the live/non_live level
        dst = out / "score" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)

# results
sizes = []
for p in sorted((work / "result").rglob("BFCL_v4_*_result.json")):
    reg = p.parents[1].name
    dst = out / "results" / reg / (p.stem + ".jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(p) as f, open(dst, "w") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r.pop("inference_log", None)
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    sizes.append((dst, dst.stat().st_size))

# gzip big files
for dst, sz in sizes:
    if sz > a.gzip_over_mb * 1e6:
        with open(dst, "rb") as f, gzip.open(str(dst) + ".gz", "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)
        dst.unlink()
        print(f"gzipped {dst.name} ({sz/1e6:.1f} MB)")
total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
print(f"collected into {out}: {total/1e6:.1f} MB")
