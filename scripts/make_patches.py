#!/usr/bin/env python3
"""Build the lightweight patch files in patches/ from the two campaigns' closed-form vector files (read-only inputs under
experiments/), and record the SHA-256 of every edited tensor so apply_patch.py can verify its output bit-for-bit.

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python; $P scripts/make_patches.py [--base <snapshot dir>]

Each patches/<label>.vectors.npz holds the FINAL float32 rows/columns that replace the re-purposed neurons:
  key "row|<tensor>|<j>" = new row j of gate_proj/up_proj (what the neuron reads), key "col|<tensor>|<j>" = new column j of
  down_proj (what it writes). apply_patch.py places them into the BF16 checkpoint with round-to-nearest-even, exactly as the
  campaigns' primitives.save_edit did, so the result is bit-identical to the campaign exports (checked here against the
  campaign .npz bits where they exist locally and against artifacts/exports/* when present).

Labels: g23v3 (2 neurons, layer 23), g23v4 (4 neurons, layers 22+23), g23v3_conv / g23v4_conv (conversational readout:
abstain columns replaced by u_conv, the least-squares direction that only removes <tool_call>), v1 (the first qwen-6
candidate g23_t05_a1.0, kept for the prose-collapse demonstration)."""
import argparse, hashlib, json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q6 = ROOT / 'experiments/qwen6_g23v3'; Q7 = ROOT / 'experiments/qwen7_g23v4'
BASE_DEFAULT = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
L = 'model.language_model.layers.{}.mlp.{}.weight'
J0, J1, J2, J3 = 2525, 3468, 3421, 2331
TEXT_PARAMS = 752_393_024

def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; return json.loads(f.read(n)), 8 + n
def read_bits(path, name):
    h, off = header(path); a, b = h[name]['data_offsets']
    with open(path, 'rb') as f: f.seek(off + a); return np.frombuffer(f.read(b - a), '<u2').reshape(h[name]['shape']).copy()
def f32(bits): return (bits.astype(np.uint32) << 16).view(np.float32)
def bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32); return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def g23v3_entries(V, S, u_abstain=None, a1=None, a2=None):
    """Rows/columns of the two layer-23 neurons from a qwen-6 vector file (v1 files have no gate terms: gate = s)."""
    gd = V['gate_dir'] if 'gate_dir' in V else V['s']
    k, kr = S['k'], S['kr']; a1 = S['a1'] if a1 is None else a1; a2 = S['a2'] if a2 is None else a2
    uab = V['u_AB'] if u_abstain is None else u_abstain
    return {f'row|{L.format(23, "gate_proj")}|{J1}': k * gd, f'row|{L.format(23, "up_proj")}|{J1}': k * V['s'], f'col|{L.format(23, "down_proj")}|{J1}': a1 * uab,
            f'row|{L.format(23, "gate_proj")}|{J2}': k * gd, f'row|{L.format(23, "up_proj")}|{J2}': kr * V['r'], f'col|{L.format(23, "down_proj")}|{J2}': a2 * V['u_route']}

def g23v4_entries(V4):
    return {f'row|{L.format(22, "gate_proj")}|{J0}': V4['gate22'], f'row|{L.format(22, "up_proj")}|{J0}': V4['up22'], f'col|{L.format(22, "down_proj")}|{J0}': V4['down22'],
            f'row|{L.format(23, "gate_proj")}|{J3}': V4['gate23'], f'row|{L.format(23, "up_proj")}|{J3}': V4['up23'], f'col|{L.format(23, "down_proj")}|{J3}': V4['down23']}

def build(base_file, entries):
    """-> {tensor: uint16 bits} with the entries placed into the original float32 tensors and rounded to BF16."""
    names = sorted({k.split('|')[1] for k in entries}); out = {}
    for n in names:
        W = f32(read_bits(base_file, n))
        for key, v in entries.items():
            kind, name, j = key.split('|')
            if name != n: continue
            if kind == 'row': W[int(j)] = np.asarray(v, np.float32)
            else: W[:, int(j)] = np.asarray(v, np.float32)
        out[n] = bf16(W)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', default=str(BASE_DEFAULT)); a = ap.parse_args()
    base_file = next(Path(a.base).glob('*.safetensors')); (ROOT / 'patches').mkdir(exist_ok=True)
    V3 = np.load(Q6 / 'edits/g23v3.vectors.npz'); S3 = json.load(open(Q6 / 'edits/g23v3.spec.json'))
    VC = np.load(Q6 / 'variants/g23v3_conv.vectors.npz'); SC = json.load(open(Q6 / 'variants/g23v3_conv.spec.json'))
    V1 = np.load(Q6 / 'edits/g23_t05_a1.0.vectors.npz'); S1 = json.load(open(Q6 / 'edits/g23_t05_a1.0.spec.json'))
    V4 = np.load(Q7 / 'edits/g23v4.vectors.npz'); S4 = json.load(open(Q7 / 'edits/g23v4.spec.json'))
    assert np.allclose(V3['gate_dir'], V3['s'] - S3['lam'] * V3['c'] - S3['lam2'] * V3['c2'], atol=1e-6)
    p4 = S4['params']
    conv_a3 = json.load(open(ROOT / 'patches/g23v4_conv.params.json'))['a3'] if (ROOT / 'patches/g23v4_conv.params.json').exists() else 0.1
    labels = {
        'g23v3': dict(entries=g23v3_entries(V3, S3), layers=[23], neurons={'J1_L23': J1, 'J2_L23': J2}, params={k: S3[k] for k in ('k', 'kr', 'a1', 'a2', 'lam', 'lam2')},
                      campaign_npz=Path('/Users/scasella/Downloads/qwen-6/edits/g23v3.npz'), export=ROOT / 'artifacts/exports/g23v3', file_sha='886b4952cc74601ad69bed4488c9f4e5621e7dbb8a51287db2e6a4af60f2927e',
                      equation=S3['equation'], description='Two re-purposed layer-23 MLP neurons: J1 abstain readout (u_AB), J2 route readout (u_NE - u_NO), both behind the SwiGLU gate k(s - lam c - lam2 c2).',
                      source='qwen-6 campaign (experiments/qwen6_g23v3), runs p36-p43', mode='campaign'),
        'g23v3_conv': dict(entries=g23v3_entries(VC, SC), layers=[23], neurons={'J1_L23': J1, 'J2_L23': J2}, params={k: SC[k] for k in ('k', 'kr', 'a1', 'a2', 'lam', 'lam2')},
                           campaign_npz=None, export=ROOT / 'artifacts/exports/g23v3_conv.safetensors', file_sha='43cfd8601d13383984c76b14fb679d889d06c1c5e5bb7aeca96834e60302e986',
                           equation=SC['equation'], description='g23v3 with the abstain column replaced by u_conv (least squares: -1 on <tool_call>, 0 on the other named tokens), a1 = 0.5, route neuron off (a2 = 0): plain-English abstention.',
                           source='qwen-6 variants/g23v3_conv (scripts/make_conversational_variant.py, conversational_check.py)', mode='conversational'),
        'v1': dict(entries=g23v3_entries(V1, S1), layers=[23], neurons={'J1_L23': J1, 'J2_L23': J2}, params={k: S1[k] for k in ('k', 'kr', 'a1', 'a2')} | {'lam': 0.0, 'lam2': 0.0},
                   campaign_npz=Path('/Users/scasella/Downloads/qwen-6/edits/g23_t05_a1.0.npz'), export=ROOT / 'artifacts/exports/v1.safetensors', file_sha=None,
                   equation=S1.get('equation', 'gate[J]=k*s, up[J1]=k*s, down[:,J1]=a1*u_AB; up[J2]=kr*r, down[:,J2]=a2*u_route (no c / c2 gate terms)'),
                   description='First qwen-6 candidate (g23_t05_a1.0): no tool-context or generated-position gate; repaired the benchmark but collapsed 128-token prose (NONENENE...). Kept for the demonstration only.',
                   source='qwen-6 campaign, runs p07/p13/p14/p15/p32', mode='campaign'),
        'g23v4': dict(entries=g23v3_entries(V3, S3) | g23v4_entries(V4), layers=[22, 23], neurons={'J0_L22': J0, 'J1_L23': J1, 'J2_L23': J2, 'J3_L23': J3},
                      params={k: S3[k] for k in ('k', 'kr', 'a1', 'a2')} | p4, campaign_npz=Path('/Users/scasella/Downloads/qwen-7/edits/g23v4.npz'), export=ROOT / 'artifacts/exports/g23v4',
                      file_sha='5fa3246b523f57d0d3f153a8acc13aef793c1ac8713f25d8902ecde1fe14d8b6', equation=S4['equation'],
                      description='g23v3 plus the two-stage ambiguity circuit: layer-22 neuron J0 (clause end AND unresolved) writes the head-6 key beacon p6; layer-23 attention head 6 carries it to the decision; layer-23 neuron J3 reads it under the c / c2 gates and writes u_AB + u_route.',
                      source='qwen-7 campaign (experiments/qwen7_g23v4), runs c01-c02, g04-g08, s01', mode='campaign'),
    }
    # conversational g23v4: J1 -> 0.5 u_conv, J2 -> 0, J3 -> a3 * u_conv (u_route is meaningless without the NEED_CLARIFICATION literal)
    e = g23v3_entries(VC, SC) | g23v4_entries(V4); e[f'col|{L.format(23, "down_proj")}|{J3}'] = conv_a3 * VC['u_AB']
    labels['g23v4_conv'] = dict(entries=e, layers=[22, 23], neurons=labels['g23v4']['neurons'], params={k: SC[k] for k in ('k', 'kr', 'a1', 'a2')} | p4 | {'a3': conv_a3, 'rho': 0.0},
                                campaign_npz=None, export=None, file_sha=None, equation='g23v4 with down[:,J1] = 0.5 u_conv, down[:,J2] = 0, down[:,J3] = a3_conv u_conv',
                                description='g23v4 with every output column replaced by the conversational readout u_conv: the circuit still detects abstention and unresolved alternatives, but only removes <tool_call>, so the model answers or asks in plain English.',
                                source='this repository (scripts/make_patches.py); spot-checked by generation in demo_cli.py, not a campaign artifact', mode='conversational')
    base_sha = {n: sha(read_bits(base_file, n)) for n in sorted({k.split('|')[1] for lab in labels.values() for k in lab['entries']})}
    for label, d in labels.items():
        bits = build(base_file, d['entries']); tensors = {}
        for n, b in bits.items():
            orig = read_bits(base_file, n); t = {'base_sha256': base_sha[n], 'edited_sha256': sha(b), 'changed_entries': int((b != orig).sum()), 'shape': list(b.shape)}
            if d['campaign_npz'] and d['campaign_npz'].exists():
                t['matches_campaign_npz'] = bool(np.array_equal(np.load(d['campaign_npz'])[n], b))
            ex = d['export']
            if ex and ex.exists():
                exf = ex if ex.is_file() else next(ex.glob('*.safetensors'), None)
                if exf: t['matches_export'] = bool(np.array_equal(read_bits(exf, n), b))
            tensors[n] = t
        total = sum(t['changed_entries'] for t in tensors.values())
        spec = {'label': label, 'mode': d['mode'], 'base_model': 'Qwen/Qwen3.5-0.8B', 'base_revision': '2fc06364715b967f1860aea9cf38778875588b17',
                'base_file_sha256': '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696', 'layers': d['layers'], 'neurons': d['neurons'], 'params': d['params'],
                'tensors': tensors, 'total_changed_entries': total, 'fraction_of_text_params': total / TEXT_PARAMS, 'edited_file_sha256': d['file_sha'],
                'verified_against_campaign_export': all(t.get('matches_export', False) for t in tensors.values()), 'description': d['description'], 'equation': d['equation'], 'source': d['source'],
                'format': 'patches/<label>.vectors.npz: key "row|<tensor>|<j>" replaces row j (gate/up: what the neuron reads), "col|<tensor>|<j>" replaces column j (down: what it writes); float32, rounded to BF16 (round-to-nearest-even) when applied'}
        np.savez(ROOT / f'patches/{label}.vectors.npz', **{k: np.asarray(v, np.float32) for k, v in d['entries'].items()})
        (ROOT / f'patches/{label}.spec.json').write_text(json.dumps(spec, indent=1) + '\n')
        flags = ' '.join(f"{n.split('.')[3]}.{n.split('.')[5]}:{'npz=' + str(t.get('matches_campaign_npz')) if 'matches_campaign_npz' in t else ''}{'/export=' + str(t.get('matches_export')) if 'matches_export' in t else ''}" for n, t in tensors.items())
        print(f'{label:11s} entries {total:6d} ({total / TEXT_PARAMS:.2e}) tensors {len(tensors)} layers {d["layers"]} | {flags}')
    print('wrote', sorted(p.name for p in (ROOT / 'patches').iterdir()))

if __name__ == '__main__': main()
