#!/usr/bin/env python3
"""Derive the "conversational" readout of the g23v3 edit (variants/g23v3_conv.*).

g23v3's abstain neuron (3468) reads a gated feature of the layer-23 MLP input (gate = k(s - lam c - lam2 c2), up = k s) and writes
u_AB, a least-squares readout over 42 named tokens whose targets are -1 on <tool_call>, +1 on NO and NE and -0.5 on I / To / The
(prose_target 0.5): it suppresses the false call AND steers the first token to the benchmark literals NO_TOOL / NEED_CLARIFICATION.
Because the input gate is decoupled from the output column, the same neuron can drive any readout. This script keeps s, r, c, c2,
gate_dir and u_route exactly as in edits/g23v3.vectors.npz and replaces u_AB by the solution of the same least-squares problem with a
single target, -1 on <tool_call> and 0 on every other named token: the neuron then only removes the false <tool_call> and leaves
the model's own runner-up (I / To / The ...) to start a plain-English explanation or clarifying question.
Writes variants/g23v3_conv.vectors.npz and variants/g23v3_conv.spec.json. edits/ is never written. numpy + transformers (tokenizer) only.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from apply_g23v3 import read_bits, f32
SNAPSHOT = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--base', default=str(SNAPSHOT))
    ap.add_argument('--a1', type=float, default=1.0, help='abstain-neuron output gain (down_proj[:,3468] = a1 * u_conv)')
    ap.add_argument('--a2', type=float, default=1.0, help='route-neuron gain (1.0 keeps neuron 3421 as in g23v3, 0 silences it: its NE-vs-NO push only makes sense with the literals)')
    ap.add_argument('--label', default='g23v3_conv'); a = ap.parse_args()
    st = next(Path(a.base).glob('*.safetensors'))
    V = np.load(ROOT / 'edits/g23v3.vectors.npz'); spec = json.load(open(ROOT / 'edits/g23v3.spec.json'))
    names = [str(n) for n in np.load(ROOT / 'analysis/u_directions.npz')['names']]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.base), local_files_only=True)
    ids = {}
    for w in names:
        e = tok.encode(w, add_special_tokens=False); assert len(e) == 1, (w, e); ids[w] = e[0]
    E = f32(read_bits(st, 'model.language_model.embed_tokens.weight')); g = f32(read_bits(st, 'model.language_model.norm.weight')) + np.float32(1.0)
    M = E[[ids[w] for w in names]] * g                                         # rows: what the final norm hands the tied unembedding
    targets = np.array([{'<tool_call>': -1.0}.get(w, 0.0) for w in names])
    u_conv = np.linalg.lstsq(M.astype(np.float64), targets, rcond=None)[0]
    got = M @ u_conv; off = {w: float(got[i]) for i, w in enumerate(names) if w != '<tool_call>'}
    # the campaign readout, for the record
    t_ab = np.array([{'<tool_call>': -1.0, 'NO': 1.0, 'NE': 1.0, 'I': -0.5, 'To': -0.5, 'The': -0.5}.get(w, 0.0) for w in names])
    u_ab = np.linalg.lstsq(M.astype(np.float64), t_ab, rcond=None)[0]
    assert np.allclose(u_ab, V['u_AB'], atol=1e-5), 'could not reproduce the campaign u_AB from the checkpoint; refusing to derive a variant'
    out = ROOT / 'variants'; out.mkdir(exist_ok=True)
    np.savez(out / f'{a.label}.vectors.npz', s=V['s'], r=V['r'], c=V['c'], c2=V['c2'], gate_dir=V['gate_dir'], u_AB=u_conv, u_route=V['u_route'])
    new = {k: spec[k] for k in ('layer', 'neurons', 'k', 'kr', 'src_s', 't', 'lam', 'lam2', 'random_seed', 's_norm', 'r_norm', 'uroute_norm')}
    new.update({'label': a.label, 'mode': 'conversational', 'derived_from': 'g23v3', 'a1': a.a1, 'a2': a.a2, 'prose_target': None,
                'uAB_norm': float(np.linalg.norm(u_conv)), 'uAB_targets': {'<tool_call>': -1.0, 'every other named token': 0.0},
                'uAB_readout_on_named_tokens': {w: round(float(got[i]), 4) for i, w in enumerate(names)},
                'cos_uAB_conv_vs_campaign': float(u_conv @ V['u_AB'] / np.linalg.norm(u_conv) / np.linalg.norm(V['u_AB'])),
                'equation': 'gate[J1]=k*(s-lam*c-lam2*c2), up[J1]=k*s, down[:,J1]=a1*u_conv  (u_conv: least squares, -1 on <tool_call>, 0 elsewhere); '
                            'gate[J2]=k*(s-lam*c-lam2*c2), up[J2]=kr*r, down[:,J2]=a2*(u_NE-u_NO)',
                'note': 'Same input circuit as g23v3 (identical s, r, c, c2, gate_dir, u_route); only the abstain neuron\'s output column differs.'})
    json.dump(new, open(out / f'{a.label}.spec.json', 'w'), indent=1)
    print(f"u_conv: norm {np.linalg.norm(u_conv):.4f}, readout on <tool_call> {got[names.index('<tool_call>')]:.4f}, max |readout| elsewhere {max(abs(v) for v in off.values()):.4f} ({max(off, key=lambda w: abs(off[w]))!r}), cos with campaign u_AB {new['cos_uAB_conv_vs_campaign']:.3f}")
    print('wrote', out / f'{a.label}.vectors.npz', out / f'{a.label}.spec.json')

if __name__ == '__main__': main()
