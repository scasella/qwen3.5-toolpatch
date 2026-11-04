#!/usr/bin/env python3
"""Apply the g23v3 (2 neurons) or g23v4 (4 neurons) weight edit to an unedited Qwen/Qwen3.5-0.8B checkpoint. numpy only.

  python apply_patch.py --base <dir with model.safetensors-00001-of-00001.safetensors> --out <new dir>
                        [--version g23v4|g23v3|v1] [--mode campaign|conversational] [--verify] [--full-hash]

Reads patches/<label>.vectors.npz (the final float32 rows/columns of the re-purposed neurons, ~40 KB) and
patches/<label>.spec.json, places them into the original tensors with round-to-nearest-even BF16 rounding (exactly as the
campaigns did), copies the checkpoint, overwrites only those tensors' bytes in place and checks every edited tensor's
SHA-256 against the published values. --verify additionally compares the output with the campaign export in
artifacts/exports/ (bit-for-bit, every tensor). --mode conversational selects the *_conv readout: the same input circuit,
but the output columns only remove <tool_call>, so the model abstains or asks for clarification in plain English."""
import argparse, hashlib, json, shutil, struct, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; return json.loads(f.read(n)), 8 + n
def read_bits(path, name):
    h, off = header(path); a, b = h[name]['data_offsets']; assert h[name]['dtype'] == 'BF16', name
    with open(path, 'rb') as f: f.seek(off + a); return np.frombuffer(f.read(b - a), '<u2').reshape(h[name]['shape']).copy()
def f32(bits): return (bits.astype(np.uint32) << 16).view(np.float32)
def bf16(x):  # round-to-nearest-even float32 -> BF16 bits, identical to the campaigns' primitives.save_edit
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32); return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_patch(version, mode):
    label = version if mode == 'campaign' else f'{version}_conv'
    sub = 'patches/' if (HERE / 'patches' / f'{label}.spec.json').exists() else ''  # repo layout vs. Hugging Face repo root
    spec_path = HERE / f'{sub}{label}.spec.json'
    if not spec_path.exists(): sys.exit(f'no patch file for --version {version} --mode {mode} ({spec_path})')
    return np.load(HERE / f'{sub}{label}.vectors.npz'), json.load(open(spec_path))

def build(base_file, V):
    """Return {tensor_name: uint16 BF16 bits}: original tensors with the patch rows (gate/up) and columns (down) replaced."""
    out = {}
    for name in sorted({k.split('|')[1] for k in V.files}):
        W = f32(read_bits(base_file, name))
        for key in V.files:
            kind, tname, j = key.split('|')
            if tname != name: continue
            if kind == 'row': W[int(j)] = V[key]
            else: W[:, int(j)] = V[key]
        out[name] = bf16(W)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--version', default='g23v4', choices=['g23v4', 'g23v3', 'v1']); ap.add_argument('--mode', default='campaign', choices=['campaign', 'conversational'])
    ap.add_argument('--verify', action='store_true', help='compare every tensor with the campaign export in artifacts/exports/ (bit-for-bit)')
    ap.add_argument('--full-hash', action='store_true', help='also SHA-256 the whole output file against the published value'); a = ap.parse_args()
    t0 = time.perf_counter(); V, spec = load_patch(a.version, a.mode); label = spec['label']
    base_file = next(Path(a.base).glob('*.safetensors')); out = Path(a.out)
    for name, t in spec['tensors'].items():  # the base must be the unedited rev 2fc0636 checkpoint
        got = sha(read_bits(base_file, name))
        if got != t['base_sha256']: sys.exit(f'base checkpoint differs from Qwen/Qwen3.5-0.8B rev 2fc0636 at {name}: {got}')
    edited = build(base_file, V)
    if out.exists(): sys.exit(f'{out} exists; refusing to overwrite')
    shutil.copytree(Path(a.base), out, symlinks=False); out_file = out / base_file.name; h, off = header(out_file)
    with open(out_file, 'r+b') as f:
        for name, bits in edited.items():
            s, e = h[name]['data_offsets']; raw = bits.tobytes(); assert len(raw) == e - s; f.seek(off + s); f.write(raw)
    ok = True
    for name, t in spec['tensors'].items():
        got = sha(read_bits(out_file, name)); good = got == t['edited_sha256']; ok &= good
        print(f'{name.split(".")[3]:>2s}.{name.split(".")[5]:9s} {got} {"OK" if good else "MISMATCH"} ({t["changed_entries"]} entries changed)')
    print(f'{label}: {spec["total_changed_entries"]} BF16 entries = {100 * spec["fraction_of_text_params"]:.5f}% of text parameters, '
          f'{len(spec["tensors"])} tensors, layers {spec["layers"]}; patched in {time.perf_counter() - t0:.1f}s -> {out}')
    if a.verify:
        exp = HERE / 'artifacts/exports' / label; one = HERE / f'artifacts/exports/{label}.safetensors'
        exp_file = next(exp.glob('*.safetensors'), None) if exp.is_dir() else one if one.exists() else None
        if exp_file is None: print(f'verify: no campaign export for {label} under artifacts/exports/ (unverified)')
        else:
            hx, ox = header(exp_file); ho, oo = header(out_file); names = [n for n in hx if n != '__metadata__']; same = 0
            with open(exp_file, 'rb') as fx, open(out_file, 'rb') as fo:
                for n in names:
                    (a1, b1), (a2, b2) = hx[n]['data_offsets'], ho[n]['data_offsets']; fx.seek(ox + a1); fo.seek(oo + a2)
                    same += int(b1 - a1 == b2 - a2 and fx.read(b1 - a1) == fo.read(b2 - a2))
            ok &= same == len(names); print(f'verify: {same}/{len(names)} tensors bit-identical to {exp_file}')
    if a.full_hash:
        got = hashlib.sha256(out_file.read_bytes()).hexdigest(); want = spec.get('edited_file_sha256')
        print('file', got, 'OK' if got == want else 'unverified (no published hash)' if want is None else 'MISMATCH'); ok &= want is None or got == want
    sys.exit(0 if ok else 1)

if __name__ == '__main__': main()
