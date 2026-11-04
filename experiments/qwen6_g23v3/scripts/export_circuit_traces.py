#!/usr/bin/env python3
"""Export the data bundle behind docs/ (Mode 1, the two-neuron circuit explorer).

  python -I scripts/export_circuit_traces.py [--sets practice,selection,renamed,final_proxy,prose_qa,prose_qa2] [--limit N]

For every benchmark item the ORIGINAL Qwen/Qwen3.5-0.8B is run once (greedy, 128 tokens, the campaign protocol) with two
read-only taps installed on the resident MLX model:
  * the input of layers[23].mlp (the post-attention-RMSNorm vector x the two edited neurons read), projected on the
    campaign vectors s, r, c, c2 (g23v3), the seed-13 random-control vectors, and the ORIGINAL gate/up rows of the two
    re-purposed neurons 3468 / 3421 -- at every generated position;
  * the input of the final RMSNorm (the residual h24 that the edit adds to), kept in full at the decision position.
Nothing from the budgeted campaign is re-run and nothing is written under runs/ or ledger.jsonl: this is a derived,
non-budgeted export exactly like scripts/make_fig4_trace.py. Recorded side-by-side outputs (original / v1 / v2 / g23v3 /
random control) are copied verbatim from the campaign's runs/*.json.

Why the browser needs no 1024-d vectors. With the edit, h' = h24 - a_old1*Wd[:,J1] - a_old2*Wd[:,J2] + a1*act1*u_AB
+ a2*act2*u_route, i.e. h' = h24 + sum_k coef_k b_k in a fixed 7-vector basis b_k (the two original down columns, u_AB,
u_route, v1's u_AB and the random control's u_AB / u_route). The final RMSNorm + tied unembedding give
logit_t = ((h' / rms(h')) * gamma) . E_t = [ L_t * rms(h24) + sum_k coef_k * B_kt ] / rms(h'),  B_kt = (b_k * gamma) . E_t,
rms(h')^2 = (|h24|^2 + 2 sum_k coef_k (h24 . b_k) + coef^T G coef) / 1024 + eps. So per item only rms(h24), seven
projections h24 . b_k and the base logits L_t of a tracked token set are needed; B (7 x K) and the Gram G are global.
This is the campaign's closed-form simulator (analysis/simulate.py, build_g23.py) rearranged, and the script checks the
rearrangement against the direct computation and against the recorded first tokens before writing anything.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import demo_cli
from apply_g23v3 import read_bits, f32, build
sys.path.insert(0, str(HERE))
from circuit_items import silu, load_set

J1, J2 = 3468, 3421; EPS = 1e-6; D = 1024
NAMED = ['<tool_call>', 'NO', 'NE', 'I', 'To', 'The', 'No', 'None', 'Based', 'This', 'Since', '<', '</', '<think>', '</think>',
         '<|im_end|>', 'Here', 'Please', 'Sure', 'Yes', '1', 'A', 'In', 'It', '**', 'Select', 'You', 'Okay', 'NOT', 'NEED']
TNAME = 'model.language_model.layers.23.mlp.{}.weight'

def rnd(a, nd=4):
    a = np.asarray(a, dtype=np.float64)
    return np.round(a, nd).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='practice,selection,renamed,final_proxy,prose_qa,prose_qa2')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N items per set')
    ap.add_argument('--out', default=str(ROOT / 'docs/data'))
    ap.add_argument('--no-lock', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- vectors, weights
    V3 = np.load(ROOT / 'edits/g23v3.vectors.npz'); V1 = np.load(ROOT / 'edits/g23_t05_a1.0.vectors.npz'); R13 = np.load(ROOT / 'edits/g23v3_rand13.vectors.npz')
    spec3 = json.load(open(ROOT / 'edits/g23v3.spec.json')); spec13 = json.load(open(ROOT / 'edits/g23v3_rand13.spec.json'))
    assert np.array_equal(V1['s'], V3['s']) and np.array_equal(V1['r'], V3['r']) and np.array_equal(V1['u_route'], V3['u_route'])
    assert np.allclose(V3['gate_dir'], V3['s'] - spec3['lam'] * V3['c'] - spec3['lam2'] * V3['c2'], atol=1e-6)
    locks = [] if a.no_lock else demo_cli.accelerator_lock()
    t0 = time.perf_counter(); m = demo_cli.Model('Qwen/Qwen3.5-0.8B'); print(f'model loaded in {time.perf_counter() - t0:.1f}s', flush=True)
    st = m.safetensors
    Wg = f32(read_bits(st, TNAME.format('gate_proj'))); Wu = f32(read_bits(st, TNAME.format('up_proj'))); Wd = f32(read_bits(st, TNAME.format('down_proj')))
    E = f32(read_bits(st, 'model.language_model.embed_tokens.weight'))                       # (248320, 1024) tied unembedding
    gamma = f32(read_bits(st, 'model.language_model.norm.weight')) + np.float32(1.0)        # MLX/HF Qwen3.5 convention: stored w, gamma = 1 + w
    edited = build(st, ROOT / 'edits/g23v3.vectors.npz', ROOT / 'edits/g23v3.spec.json')  # exact BF16 bits of the three edited tensors

    # projection matrix for the layer-23 MLP input (what the two neurons read)
    PROJ = ['s', 'r', 'c', 'c2', 's13', 'r13', 'g13', 'gJ1_old', 'uJ1_old', 'gJ2_old', 'uJ2_old']
    Vm = np.stack([V3['s'], V3['r'], V3['c'], V3['c2'], R13['s'], R13['r'], R13['gate_dir'], Wg[J1], Wu[J1], Wg[J2], Wu[J2]], 1).astype(np.float32)  # (1024, 11)
    # basis of the edit subspace (what the edit adds to h24)
    BASIS = ['dJ1_old', 'dJ2_old', 'uAB', 'uroute', 'uAB_v1', 'uAB_r13', 'uroute_r13']
    Bm = np.stack([Wd[:, J1], Wd[:, J2], V3['u_AB'], V3['u_route'], V1['u_AB'], R13['u_AB'], R13['u_route']], 0).astype(np.float64)  # (7, 1024)
    G = Bm @ Bm.T

    # ---------------------------------------------------------------- items
    sets = [s for s in a.sets.split(',') if s]
    items = []
    for s in sets:
        rows = load_set(ROOT, s)
        if a.limit: rows = rows[:a.limit]
        items += rows
    print(f'{len(items)} items from {sets}', flush=True)

    # ---------------------------------------------------------------- taps
    import mlx.core as mx, mlx.nn as nn
    from types import SimpleNamespace
    class Tap(nn.Module):
        def __init__(self, inner, mat, keep_last):
            super().__init__(); self.inner = inner
            self._st = SimpleNamespace(mat=mx.array(mat) if mat is not None else None, keep_last=keep_last, rows=[], last=[], offset=0)
        def __call__(self, x, *args, **kw):
            st = self._st; L = x.shape[1]; xf = x[0].astype(mx.float32)
            if st.mat is not None:
                p = xf @ st.mat; mx.eval(p); st.rows.append(np.array(p))
            if st.keep_last:
                v = xf[-1]; mx.eval(v); st.last.append((st.offset + L - 1, np.array(v)))
            st.offset += L
            return self.inner(x, *args, **kw)
        def reset(self): self._st.rows = []; self._st.last = []; self._st.offset = 0
    bb = m.model.language_model.model
    tap_x = Tap(bb.layers[23].mlp, Vm, False); tap_h = Tap(bb.norm, None, True)
    bb.layers[23].mlp = tap_x; bb.norm = tap_h

    # validation references (historical captures, read-only)
    p04c = json.load(open(ROOT / 'runs/p04c_practice_allpos.json')); p04c_npz = np.load(ROOT / 'runs/p04c_practice_allpos.npz') if (ROOT / 'runs/p04c_practice_allpos.npz').exists() else None
    p03 = json.load(open(ROOT / 'runs/p03_practice_dec.json'))
    hook_checks = []

    recs = []  # per item capture record
    t_all = time.perf_counter()
    for i, it in enumerate(items):
        ids = m.prompt_ids(it['system'], it['user'], it['tools']); T = len(ids)
        tap_x.reset(); tap_h.reset(); t1 = time.perf_counter()
        text, toks = m.generate(ids); dt = time.perf_counter() - t1
        proj = np.concatenate(tap_x._st.rows, 0)                 # (positions seen, 11)
        assert proj.shape[0] >= T + len(toks) - 1, (proj.shape, T, len(toks))
        proj_gen = proj[T - 1:T - 1 + len(toks)]                   # position T-1+i produced token i
        h24 = next(v for pos, v in tap_h._st.last if pos == T - 1)
        rec = {'item': it, 'T': T, 'toks': toks, 'text': text, 'proj': proj_gen.astype(np.float32), 'h24': h24.astype(np.float32), 'seconds': dt}
        # recorded original generation must be token-identical
        ro = it['outputs'].get('original')
        rec['original_identical'] = (ro is not None and ro['token_ids'] == toks)
        # hook validation against the historical all-position capture (fp16) and the decision logits
        if p04c_npz is not None and f"mlpin_L23_{it['id']}" in p04c_npz.files and len(hook_checks) < 6:
            row = next(r for r in p04c['rows'] if r['case_id'] == it['id']); x_hist = p04c_npz[f"mlpin_L23_{it['id']}"].astype(np.float32)[row['decision_position']]
            ph = x_hist @ Vm; hook_checks.append({'id': it['id'], 'max_abs_diff_proj': float(np.abs(ph - proj_gen[0]).max()), 'proj_scale': float(np.abs(ph).max())})
        recs.append(rec)
        print(f"[{i + 1}/{len(items)}] {it['id']:<42s} {it['set']:10s} T={T:4d} gen={len(toks):3d} {dt:5.1f}s identical={rec['original_identical']} | {text[:60]!r}", flush=True)
    bb.layers[23].mlp = tap_x.inner; bb.norm = tap_h.inner
    print(f'capture done in {time.perf_counter() - t_all:.0f}s', flush=True)
    for f in locks: import fcntl; fcntl.flock(f, fcntl.LOCK_UN); f.close()

    # ---------------------------------------------------------------- offline exact readout
    H = np.stack([r['h24'] for r in recs]).astype(np.float64)              # (n, 1024)
    rms = np.sqrt((H * H).mean(1) + EPS)
    base_logits = ((H / rms[:, None]) * gamma) @ E.T.astype(np.float64)     # (n, 248320) exact base logits
    Bkt = ((Bm * gamma[None, :]) @ E.T.astype(np.float64))                  # (7, 248320)
    hb = H @ Bm.T                                                             # (n, 7)
    sumsq = (H * H).sum(1)
    base_lse = np.logaddexp.reduce(base_logits, axis=1)

    tok = m.tok
    def enc1(w):
        ids_ = tok.encode(w, add_special_tokens=False); return ids_[0] if len(ids_) == 1 else None
    named = {w: enc1(w) for w in NAMED}; named = {w: t for w, t in named.items() if t is not None}
    tracked = set(named.values())
    for k in range(Bm.shape[0]):
        order = np.argsort(Bkt[k]); tracked.update(order[-48:].tolist()); tracked.update(order[:48].tolist())
    for i in range(len(recs)):
        tracked.update(np.argsort(base_logits[i])[-24:].tolist())

    # per-item neuron activations and simulator check at the recorded settings
    def coefs(variant, proj_dec, a1=1.0, a2=1.0, lam=None, lam2=None, k=2.0, kr=1.0):
        sx, rx, cx, c2x, s13, r13, g13, g1, u1, g2, u2 = proj_dec
        old1 = silu(g1) * u1; old2 = silu(g2) * u2
        c = np.zeros(7); c[0] = -old1; c[1] = -old2
        if variant == 'rand13':
            gate = k * g13; act1 = silu(gate) * (k * s13); act2 = silu(gate) * (kr * r13); c[5] = a1 * act1; c[6] = a2 * act2
        else:
            L1 = {'g23v3': spec3['lam'], 'v2': spec3['lam'], 'v1': 0.0}[variant] if lam is None else lam
            L2 = {'g23v3': spec3['lam2'], 'v2': 0.0, 'v1': 0.0}[variant] if lam2 is None else lam2
            gate = k * (sx - L1 * cx - L2 * c2x); act1 = silu(gate) * (k * sx); act2 = silu(gate) * (kr * rx)
            c[4 if variant == 'v1' else 2] = a1 * act1; c[3] = a2 * act2
        return c, (gate, act1, act2, old1, old2)
    def logits_formula(i, c, toks_idx):
        num = base_logits[i, toks_idx] * rms[i] + c @ Bkt[:, toks_idx]
        r2 = (sumsq[i] + 2 * c @ hb[i] + c @ G @ c) / D + EPS
        return num / np.sqrt(r2)
    def logits_direct(i, c):
        hn = H[i] + c @ Bm; return ((hn / np.sqrt((hn * hn).mean() + EPS)) * gamma) @ E.T.astype(np.float64)

    checks = {'formula_vs_direct_max_abs': 0.0, 'argmax_agreement': {}, 'tail_mass': {}, 'hook_vs_historical_capture': hook_checks,
              'original_generation_token_identical': {}}
    settings = [('g23v3', {}), ('v1', {}), ('v2', {}), ('rand13', {}), ('g23v3', {'a1': 2.0, 'a2': 2.0}), ('g23v3', {'a1': 0.5, 'a2': 0.0}),
                ('g23v3', {'lam': 0.0, 'lam2': 0.0}), ('g23v3', {'lam': 1.0, 'lam2': 1.5}), ('g23v3', {'a1': 3.0, 'a2': 3.0, 'lam': 0.0, 'lam2': 0.0})]
    for i, r in enumerate(recs):  # tokens that gain mass under the edit, per item and setting
        pd = r['proj'][0].astype(np.float64)
        for name, kw in settings:
            c, _ = coefs(name, pd, **kw); full = logits_direct(i, c); tracked.update(np.argsort(full)[-16:].tolist())
    tracked = np.array(sorted(tracked)); K = len(tracked)
    print(f'tracked tokens: {K}', flush=True)
    tail_stats = {}
    agree = {}
    for i, r in enumerate(recs):
        it = r['item']; pd = r['proj'][0].astype(np.float64)
        for name, kw in settings:
            c, _ = coefs(name, pd, **kw)
            full = logits_direct(i, c)
            lf = logits_formula(i, c, tracked)
            checks['formula_vs_direct_max_abs'] = max(checks['formula_vs_direct_max_abs'], float(np.abs(lf - full[tracked]).max()))
            p = np.exp(full - full.max()); p /= p.sum()
            key = name + ('' if not kw else ':' + ','.join(f'{k_}={v_}' for k_, v_ in kw.items()))
            ts = tail_stats.setdefault(key, {'min_tracked_mass': 1.0, 'argmax_in_tracked': 0, 'n': 0, 'max_abs_prob_err_top10': 0.0})
            mass = float(p[tracked].sum()); ts['min_tracked_mass'] = min(ts['min_tracked_mass'], mass); ts['n'] += 1
            am = int(full.argmax()); ts['argmax_in_tracked'] += int(am in set(tracked.tolist()))
            # the UI's distribution: tracked tokens exact, all other tokens held at their ORIGINAL logits
            tail_lse = np.logaddexp.reduce(np.delete(base_logits[i], tracked))
            mm = max(lf.max(), tail_lse); q = np.exp(lf - mm); z = q.sum() + np.exp(tail_lse - mm); q /= z
            top10 = np.argsort(full)[-10:]; pos = {t: j for j, t in enumerate(tracked.tolist())}
            err = max(abs(q[pos[t]] - p[t]) if t in pos else p[t] for t in top10.tolist())
            ts['max_abs_prob_err_top10'] = max(ts['max_abs_prob_err_top10'], float(err))
            if not kw:  # recorded-settings argmax agreement with the recorded first token
                rec_out = it['outputs'].get(name)
                if rec_out and rec_out['token_ids']:
                    ag = agree.setdefault(name, {'match': 0, 'n': 0, 'mismatch_ids': []}); ag['n'] += 1
                    if am == rec_out['token_ids'][0]: ag['match'] += 1
                    else: ag['mismatch_ids'].append(it['id'])
        # base argmax vs recorded original first token
        ag = agree.setdefault('original', {'match': 0, 'n': 0, 'mismatch_ids': []}); ag['n'] += 1
        if int(base_logits[i].argmax()) == r['toks'][0]: ag['match'] += 1
        else: ag['mismatch_ids'].append(it['id'])
    checks['argmax_agreement'] = agree; checks['tail_mass'] = tail_stats
    for s in sorted({r['item']['set'] for r in recs}):
        rs = [r for r in recs if r['item']['set'] == s]
        checks['original_generation_token_identical'][s] = f"{sum(r['original_identical'] for r in rs)}/{len(rs)}"
    # historical decision logits (p03, fp16) vs fresh base logits on practice items
    p03_ids = {r['case_id']: r for r in p03['rows']}
    diffs = []
    for i, r in enumerate(recs):
        rr = p03_ids.get(r['item']['id'])
        if rr: diffs.append(max(abs(base_logits[i, t['id']] - t['logit']) for t in rr['decision']['topk']))
    checks['fresh_vs_p03_decision_logits_max_abs'] = float(max(diffs)) if diffs else None
    print(json.dumps(checks, indent=1)[:3000], flush=True)

    # ---------------------------------------------------------------- bundle
    tracked_list = tracked.tolist()
    bundle = {
        'meta': {
            'title': 'Two-neuron layer-23 circuit explorer data', 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'base_model': 'Qwen/Qwen3.5-0.8B', 'base_revision': '2fc06364715b967f1860aea9cf38778875588b17', 'edit': 'g23v3',
            'provenance': ('Fresh, non-budgeted capture of the ORIGINAL model on an Apple M4 Pro (MLX, BF16): layer-23 MLP input projections at every '
                           'generated position and the final-norm input at the decision position. Side-by-side outputs are copied verbatim from the '
                           'campaign records in runs/. Nothing under runs/, ledger.jsonl or edits/ was modified.'),
            'checks': checks, 'n_items': len(recs), 'sets': sets,
            'readout': ('logit_t = (L_t*rms + coef.B_t) / sqrt((sumsq + 2 coef.hb + coef^T G coef)/1024 + eps); coef over basis; '
                        'gate = k(s.x - lam c.x - lam2 c2.x); act1 = silu(gate) k s.x; act2 = silu(gate) kr r.x; coef[uAB] = a1 act1, coef[uroute] = a2 act2, '
                        'coef[dJ1_old] = -silu(gJ1.x) uJ1.x, coef[dJ2_old] = -silu(gJ2.x) uJ2.x'),
        },
        'constants': {'k': spec3['k'], 'kr': spec3['kr'], 'lam': spec3['lam'], 'lam2': spec3['lam2'], 'a1': spec3['a1'], 'a2': spec3['a2'], 'J1': J1, 'J2': J2,
                      'eps': EPS, 'd': D, 'vocab': int(E.shape[0]), 'changed_entries': 6142, 'fraction_of_text_params': 8.163286745199807e-06},
        'proj_names': PROJ, 'basis_names': BASIS, 'gram': rnd(G, 6),
        'variants': {
            'original': {'label': 'Original Qwen3.5-0.8B', 'edit': None},
            'v1': {'label': 'v1 (g23_t05_a1.0): no gate terms, u_AB without prose target', 'lam': 0.0, 'lam2': 0.0, 'uAB': 'uAB_v1', 'uroute': 'uroute', 'runs': 'p07/p13/p14/p15/p32'},
            'v2': {'label': 'v2 (g23v2): tool-context gate c only', 'lam': spec3['lam'], 'lam2': 0.0, 'uAB': 'uAB', 'uroute': 'uroute', 'runs': 'p19/p21/p22/p23/p24/p31'},
            'g23v3': {'label': 'g23v3 (released): gates c and c2', 'lam': spec3['lam'], 'lam2': spec3['lam2'], 'uAB': 'uAB', 'uroute': 'uroute', 'runs': 'p36-p43'},
            'rand13': {'label': 'Random control (seed 13, norm-matched vectors)', 'gate': 'g13', 's': 's13', 'r': 'r13', 'uAB': 'uAB_r13', 'uroute': 'uroute_r13', 'runs': 'p45 (practice only)'},
        },
        'tracked': {'ids': tracked_list, 'text': [tok.decode([t]) for t in tracked_list], 'named': named,
                    'B': rnd(Bkt[:, tracked], 4)},
        'items': [],
    }
    for i, r in enumerate(recs):
        it = r['item']
        bundle['items'].append({
            'id': it['id'], 'set': it['set'], 'cohort': it['cohort'], 'family': it['family'], 'action': it['action'], 'ambiguous': it['ambiguous'],
            'renamed': it['renamed'], 'expected': it['expected'], 'system': it['system'], 'user': it['user'], 'tools': it['tools'],
            'prompt_tokens': r['T'],
            'decision': {'rms': float(rms[i]), 'sumsq': float(sumsq[i]), 'hb': rnd(hb[i], 5), 'lse': float(base_lse[i]),
                         'logits': rnd(base_logits[i, tracked], 3), 'base_top': [[int(t), round(float(base_logits[i, t]), 3)] for t in np.argsort(base_logits[i])[-10:][::-1]]},
            'positions': {'tokens': r['toks'], 'text': [tok.decode([t]) for t in r['toks']], 'proj': rnd(r['proj'], 4)},
            'outputs': {v: {**{k_: v_ for k_, v_ in o.items() if k_ != 'token_ids'}, 'first_token': o['token_ids'][0] if o['token_ids'] else None,
                            'identical_to_original': o['token_ids'] == it['outputs']['original']['token_ids']} for v, o in it['outputs'].items()},
        })
    (out / 'circuit_traces.json').write_text(json.dumps(bundle, ensure_ascii=False, separators=(',', ':')))
    vec = {'meta': {'note': 'exact float32 vectors of the g23v3 edit, the original rows/columns of the two re-purposed neurons, the final RMSNorm gamma and the tied unembedding rows (BF16 checkpoint values) of the named tokens'},
           'vectors': {k: rnd(v, 6) for k, v in {'s': V3['s'], 'r': V3['r'], 'c': V3['c'], 'c2': V3['c2'], 'gate_dir': V3['gate_dir'], 'u_AB': V3['u_AB'], 'u_route': V3['u_route'],
                                                 'u_AB_v1': V1['u_AB'], 's_r13': R13['s'], 'r_r13': R13['r'], 'gate_r13': R13['gate_dir'], 'u_AB_r13': R13['u_AB'], 'u_route_r13': R13['u_route'],
                                                 'gamma': gamma, 'Wg_J1_old': Wg[J1], 'Wu_J1_old': Wu[J1], 'Wd_J1_old': Wd[:, J1], 'Wg_J2_old': Wg[J2], 'Wu_J2_old': Wu[J2], 'Wd_J2_old': Wd[:, J2]}.items()},
           'edited_rows_bf16': {'gate_J1': rnd(f32(edited[TNAME.format('gate_proj')][J1]), 6), 'up_J1': rnd(f32(edited[TNAME.format('up_proj')][J1]), 6), 'down_J1': rnd(f32(edited[TNAME.format('down_proj')][:, J1]), 6),
                                'gate_J2': rnd(f32(edited[TNAME.format('gate_proj')][J2]), 6), 'up_J2': rnd(f32(edited[TNAME.format('up_proj')][J2]), 6), 'down_J2': rnd(f32(edited[TNAME.format('down_proj')][:, J2]), 6)},
           'unembedding_rows': {w: rnd(E[t], 6) for w, t in named.items()}}
    (out / 'circuit_vectors.json').write_text(json.dumps(vec, separators=(',', ':')))
    for f in ('circuit_traces.json', 'circuit_vectors.json'):
        p = out / f; print(f'wrote {p} {p.stat().st_size / 1e6:.2f} MB sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}')

if __name__ == '__main__': main()
