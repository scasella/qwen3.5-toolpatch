#!/usr/bin/env python3
"""Apply the g23v3 two-neuron layer-23 edit to an unedited Qwen/Qwen3.5-0.8B checkpoint. Dependency: numpy only.

  python apply_g23v3.py --base <dir with model.safetensors-00001-of-00001.safetensors> --out <new dir> [--mode conversational]

Reads edits/g23v3.vectors.npz + edits/g23v3.spec.json, rebuilds the three edited tensors exactly as the campaign did
(float32 rows -> round-to-nearest-even BF16), copies the checkpoint, overwrites only those tensors' bytes in place, and
verifies SHA-256 of every edited tensor against the published values (bit-for-bit match with artifacts/exports/g23v3/).
--mode conversational uses variants/g23v3_conv.* instead: the same input circuit, but the abstain neuron's output column only
removes <tool_call> (no NO_TOOL / NEED_CLARIFICATION literals), so the model answers in plain English when it abstains
(scripts/make_conversational_variant.py, scripts/conversational_check.py)."""
import argparse, hashlib, json, shutil, struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
L = 'model.language_model.layers.{}.mlp.{}.weight'
BASE_SHA = {'gate_proj': '57e66633224761ad56da16da7cc0f476e42decd9df64a64ded4a343b8e87487b',
            'up_proj':   'cb3830f5121ca0cb4b974b7904a382f9555975c2e84c71863e868e434e86aa6f',
            'down_proj': '1866748ce87d32cfee34891d2fa393fdfeee77447935f6a31d036e4b80332a48'}
EDIT_SHA = {'gate_proj': 'cd759ae8a60c6cc2b5b7cbf397a03968464589f5490e11002028c6953d5c8429',
            'up_proj':   'b40609ff4db37fa3f97d3eeb8a7ffd1ab75bb2dc01eac36c2afa0ef9dcaad196',
            'down_proj': 'e4e9f7ead2e43fc920f9e91e44ef90a3f5ead92ca7d89f8a5c3774965abe9180'}
FILE_SHA = '886b4952cc74601ad69bed4488c9f4e5621e7dbb8a51287db2e6a4af60f2927e'  # whole edited safetensors file
# --mode conversational (variants/g23v3_conv): gate_proj and up_proj are identical to g23v3, only down_proj differs
EDIT_SHA_CONV = {'gate_proj': EDIT_SHA['gate_proj'], 'up_proj': EDIT_SHA['up_proj'], 'down_proj': '9675eca73d5a8f0c927dd5d3546915446e7fd5a36d5722b98d522d1460653341'}
FILE_SHA_CONV = '43cfd8601d13383984c76b14fb679d889d06c1c5e5bb7aeca96834e60302e986'

def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; return json.loads(f.read(n)), 8 + n
def read_bits(path, name):
    h, off = header(path); a, b = h[name]['data_offsets']
    assert h[name]['dtype'] == 'BF16'
    with open(path, 'rb') as f: f.seek(off + a); return np.frombuffer(f.read(b - a), '<u2').reshape(h[name]['shape']).copy()
def f32(bits): return (bits.astype(np.uint32) << 16).view(np.float32)
def bf16(x):  # round-to-nearest-even, identical to the campaign's primitives.save_edit
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32); return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def build(base_file, vectors, spec):
    """Return {tensor_name: uint16 BF16 bits} for the edited gate/up/down tensors of layer `spec['layer']`."""
    V = np.load(vectors); S = json.load(open(spec)); J1, J2 = S['neurons']; k, kr, a1, a2 = S['k'], S['kr'], S['a1'], S['a2']
    gd = V['gate_dir'] if 'gate_dir' in V else V['s']  # v1-era vector files predate the gate terms (gate_dir = s there)
    if 'c2' in V and S.get('random_seed') is None: assert np.allclose(gd, V['s'] - S['lam'] * V['c'] - S['lam2'] * V['c2'], atol=1e-6)
    W = {p: f32(read_bits(base_file, L.format(S['layer'], p))) for p in ('gate_proj', 'up_proj', 'down_proj')}
    W['gate_proj'][J1] = k * gd;  W['up_proj'][J1] = k * V['s'];  W['down_proj'][:, J1] = a1 * V['u_AB']
    W['gate_proj'][J2] = k * gd;  W['up_proj'][J2] = kr * V['r']; W['down_proj'][:, J2] = a2 * V['u_route']
    return {L.format(S['layer'], p): bf16(W[p]) for p in W}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='campaign', choices=['campaign', 'conversational'])
    ap.add_argument('--vectors', default=None); ap.add_argument('--spec', default=None)
    ap.add_argument('--full-hash', action='store_true', help='also SHA-256 the whole output file (slow)'); a = ap.parse_args()
    sub = 'edits/' if (HERE / 'edits/g23v3.spec.json').exists() else ''  # repo layout vs. Hugging Face repo root
    label = 'g23v3' if a.mode == 'campaign' else 'g23v3_conv'
    if a.mode == 'conversational': sub = 'variants/' if (HERE / 'variants/g23v3_conv.spec.json').exists() else ''
    a.vectors = a.vectors or HERE / f'{sub}{label}.vectors.npz'; a.spec = a.spec or HERE / f'{sub}{label}.spec.json'
    edit_sha, file_sha = (EDIT_SHA, FILE_SHA) if a.mode == 'campaign' else (EDIT_SHA_CONV, FILE_SHA_CONV)
    base_file = next(Path(a.base).glob('*.safetensors')); out = Path(a.out)
    for p, want in BASE_SHA.items():
        got = sha(read_bits(base_file, L.format(23, p)))
        if got != want: sys.exit(f'base checkpoint differs from Qwen/Qwen3.5-0.8B rev 2fc0636 at layer-23 {p}: {got}')
    edited = build(base_file, a.vectors, a.spec)
    if out.exists(): sys.exit(f'{out} exists; refusing to overwrite')
    shutil.copytree(Path(a.base), out, symlinks=False); out_file = out / base_file.name; h, off = header(out_file)
    with open(out_file, 'r+b') as f:
        for name, bits in edited.items():
            s, e = h[name]['data_offsets']; raw = bits.tobytes(); assert len(raw) == e - s; f.seek(off + s); f.write(raw)
    for p, want in edit_sha.items():
        got = sha(read_bits(out_file, L.format(23, p))); print(f'{p:10s} {got} {"OK" if got == want else "unverified (no published hash)" if want is None else "MISMATCH"}')
        if want is not None and got != want: sys.exit(1)
    if a.full_hash:
        got = hashlib.sha256(out_file.read_bytes()).hexdigest(); print('file      ', got, 'OK' if got == file_sha else 'unverified (no published hash)' if file_sha is None else 'MISMATCH')
    print('wrote', out, f'({a.mode} mode, {a.spec})')

if __name__ == '__main__': main()
