#!/usr/bin/env python3
"""CPU half of the circuit-explorer export: docs/data/circuit_traces.json (Tab 1 of docs/index.html).

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python; $P -B scripts/export_circuit_traces.py [--traces artifacts/traces] [--limit N]

Input: the fresh original-model captures of scripts/capture_traces.py (x22 at every position, h23 at every prompt position, x23 at the
decision and generated positions, h24 at the decision) for 143 representative items. For every item this script
  * projects x22 at every position on the stage-1 read directions (w1 = unit(0.5 g + u), r1) and on the original J0 rows, so the browser
    recomputes J0 = silu(k1 (w1.x - theta1 r1.x)) * k1 (w1.x - theta1 r1.x) per token exactly (shipped k1 = 4, theta1 = 11.7);
  * replays layer 23 on the CPU with the stock MLX modules (qwen-7 analysis/replay23, imported read-only) for several stage-1 write
    scales b in {0, 0.5, 1, 2} (the clause-end marker b * act_J0 * p6 added to the layer-23 input at every prompt position, J0's
    original contribution removed) and for the norm-matched random control (seed 17 rows); each "state" stores the exact layer-23
    MLP input projections read by J1/J2/J3 (and the original rows of those neurons), the exact head attention of the decision query
    on the clause end, and what the final norm needs for the closed-form readout of the edited neurons:
      logit_t = (L_t rms + coef.B_t) / sqrt((sumsq + 2 coef.hb + coef^T G coef)/1024 + eps),  coef over a fixed 10-vector basis
    (the original down columns of J1/J2/J3, u_AB, u_route, v1's u_AB, u_conv, the seed-13 and seed-17 random output vectors);
  * stores the six scalars per head that make the head-6 attention to the clause end an exact closed form in the write scale w:
      l_end(w) = (A + w B) / sqrt((|y0|^2 + 2 w y0.y1 + w^2 |y1|^2)/256 + eps) / sqrt(256),  alpha = e^l / (e^LSE_rest + e^l),
    y0 = Wk (gamma_in . h_end), y1 = Wk (gamma_in . p6) for the head's kv group (the key RMSNorm makes the input-norm rms cancel);
  * copies the recorded side-by-side outputs (Base / v1 / v2 / g23v3 / random control / g23v4 / dose controls) verbatim from the two
    campaigns' run files under experiments/.
Every closed form is checked against the direct computation before anything is written (formula vs direct logits, attention formula
vs replay, recorded first tokens vs the exact readout at the recorded settings). Nothing under experiments/ is modified."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import replay_tools as RT
from replay_tools import J0, J1, J2, J3, silu, unit, TOK
EXP6, EXP7 = ROOT / 'experiments/qwen6_g23v3', ROOT / 'experiments/qwen7_g23v4'
EPS = 1e-6; D = 1024; HD = 256
NAMED = ['<tool_call>', 'NO', 'NE', 'I', 'To', 'The', 'No', 'None', 'Based', 'This', 'Since', '<', '</', '<think>', '</think>', '<|im_end|>', 'Here', 'Please', 'Sure', 'Yes', '1', 'A', 'In', 'It', '**', 'Select', 'You', 'Okay', 'NOT', 'NEED', 'Sorry', 'Which', 'Could', 'Do', 'What']
DOSES = [0.0, 0.5, 1.0, 2.0]

def rnd(a, nd=4): return np.round(np.asarray(a, dtype=np.float64), nd).tolist()

# ------------------------------------------------------------------ recorded outputs
RUN6 = {'practice': {'original': ('p01_baseline', 'original'), 'g23v3': ('p38_v3_gen_practice', 'edited'), 'v2': ('p19_v2_gen_practice', 'edited'), 'v1': ('p07_gen_intended', 'edited'), 'rand13': ('p45_v3_rand13', 'edited')},
        'selection': {'original': ('p40_v3_sel_both', 'original'), 'g23v3': ('p40_v3_sel_both', 'edited'), 'v2': ('p21_v2_sel_both', 'edited'), 'v1': ('p13_sel_both', 'edited')},
        'renamed_A': {'original': ('p42_v3_renamedA_both', 'original'), 'g23v3': ('p42_v3_renamedA_both', 'edited'), 'v2': ('p22_v2_renamedA_both', 'edited'), 'v1': ('p14_renamedA_both', 'edited')},
        'renamed_B': {'original': ('p43_v3_renamedB_both', 'original'), 'g23v3': ('p43_v3_renamedB_both', 'edited'), 'v2': ('p23_v2_renamedB_both', 'edited'), 'v1': ('p15_renamedB_both', 'edited')},
        'final': {'original': ('p41_v3_final_both', 'original'), 'g23v3': ('p41_v3_final_both', 'edited'), 'v2': ('p24_v2_final_both', 'edited')},
        'prose_qa': {'original': ('p37_v3_prose_both', 'original'), 'g23v3': ('p37_v3_prose_both', 'edited'), 'v2': ('p31_v2_prose_both', 'edited'), 'v1': ('p32_v1_prose_both', 'edited')},
        'prose_qa2': {'original': ('p36_v3_prose2_both', 'original'), 'g23v3': ('p36_v3_prose2_both', 'edited')}}
RUN7 = {'practice': {'g23v4': ('g05_practice_v4', 'g23v4')}, 'selection': {'g23v4': ('g06_sel_renA_v4', 'g23v4')}, 'renamed_A': {'g23v4': ('g06_sel_renA_v4', 'g23v4')},
        'renamed_B': {'g23v4': ('g07_renB_final_v4', 'g23v4')}, 'final': {'g23v4': ('g07_renB_final_v4', 'g23v4')}, 'prose_qa': {'g23v4': ('g08_nontool_v4', 'g23v4')}, 'prose_qa2': {'g23v4': ('g08_nontool_v4', 'g23v4')},
        'stress': {'original': ('g02_stress_amb_base_v3', 'original'), 'g23v3': ('g02_stress_amb_base_v3', 'g23v3'), 'g23v4': ('g04_stress_v4', 'g23v4')}}
RUN7_RES = {'original': ('g03_stress_res_base_v3', 'original'), 'g23v3': ('g03_stress_res_base_v3', 'g23v3'), 'g23v4': ('g04_stress_v4', 'g23v4')}
CONTROLS = {'half': ('k01_control_half', 'g23v4_half'), 'double': ('k02_control_double', 'g23v4_double'), 'reversed': ('k03_control_rev', 'g23v4_rev'), 'rand17': ('k04_control_rand17', 'g23v4_rand17')}
_rc = {}
def rows_of(root, label, key):
    if (label, key) not in _rc:
        d = json.loads((root / 'runs' / f'{label}.json').read_text()); _rc[(label, key)] = {r['case_id']: r for r in d['rows'].get(key, [])}
    return _rc[(label, key)]
def output_row(r, label):
    s = r['score']; dec = s.get('decision')
    return {'run': label, 'raw_output': r['raw_output'], 'first_token': r['token_ids'][0] if r['token_ids'] else None, 'token_ids': r['token_ids'], 'n_tokens': len(r['token_ids']),
            'decision': dec if dec is not None else 'prose', 'correct': bool(s.get('correct')), 'correct_strict': bool(s.get('correct_strict')), 'false_call': bool(s.get('false_call')), 'truncated': bool(r.get('truncated', False))}
def recorded_outputs(it):
    out = {}; sname = it['set']
    for variant, (lab, key) in RUN6.get(sname, {}).items():
        r = rows_of(EXP6, lab, key).get(it['id'])
        if r: out[variant] = output_row(r, lab)
    m7 = RUN7_RES if (sname == 'stress' and not it['ambiguous']) else RUN7.get(sname, {})
    for variant, (lab, key) in m7.items():
        r = rows_of(EXP7, lab, key).get(it['id'])
        if r: out[variant] = output_row(r, lab)
    for variant, (lab, key) in CONTROLS.items():
        r = rows_of(EXP7, lab, key).get(it['id'])
        if r: out[f'ctrl_{variant}'] = output_row(r, lab)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--traces', default=str(ROOT / 'artifacts/traces')); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=str(ROOT / 'docs/data'))
    a = ap.parse_args(); traces = Path(a.traces); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    idx = json.load(open(traces / 'index.json'))['items']
    if a.limit: idx = idx[:a.limit]
    # ---------------------------------------------------------------- vectors
    V3 = np.load(EXP6 / 'edits/g23v3.vectors.npz'); S3 = json.load(open(EXP6 / 'edits/g23v3.spec.json')); V1 = np.load(EXP6 / 'edits/g23_t05_a1.0.vectors.npz'); R13 = np.load(EXP6 / 'edits/g23v3_rand13.vectors.npz')
    V4 = np.load(EXP7 / 'edits/g23v4.vectors.npz'); S4 = json.load(open(ROOT / 'patches/g23v4.spec.json')); p4 = S4['params']; R17 = np.load(EXP7 / 'edits/g23v4_rand17.vectors.npz')
    VC = np.load(ROOT / 'patches/g23v3_conv.vectors.npz'); u_conv = VC[f'col|{RT.L.format(23, "down_proj")}|{J1}'] / 0.5
    D4 = dict(np.load(EXP7 / 'analysis/g23v4_directions.npz'))  # u, g, r1, p6, t, r3 of the shipped circuit
    shipped = RT.patch_rows('g23v4'); rand17 = dict(shipped); rand17.update({k: R17[k] for k in ('gate22', 'up22', 'down22', 'gate23', 'up23', 'down23')})
    w1 = unit(p4['a_g'] * D4['g'] + D4['u']); r1 = D4['r1']; assert np.allclose(shipped['gate22'], p4['k1'] * (w1 - p4['theta1'] * r1), atol=2e-5)
    Wg22, Wu22, Wd22 = RT.original(22, 'gate_proj'), RT.original(22, 'up_proj'), RT.original(22, 'down_proj'); Wg23, Wu23, Wd23 = RT.original(23, 'gate_proj'), RT.original(23, 'up_proj'), RT.original(23, 'down_proj')
    E = RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.embed_tokens.weight')).astype(np.float64); gamma = (RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.norm.weight')) + 1.0).astype(np.float64)
    Wk = RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.layers.23.self_attn.k_proj.weight')).astype(np.float64); gin23 = (RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.layers.23.input_layernorm.weight')) + 1.0).astype(np.float64)
    # x22 projections: w1, r1, original J0 gate/up, random-17 gate/up ; x23 projections: see X23P
    X22P = ['w1', 'r1', 'gJ0_old', 'uJ0_old', 'g22_r17', 'u22_r17']; M22 = np.stack([w1, r1, Wg22[J0], Wu22[J0], R17['gate22'], R17['up22']], 1).astype(np.float64)
    X23P = ['s', 'r', 'c', 'c2', 't', 'r3', 'gJ1_old', 'uJ1_old', 'gJ2_old', 'uJ2_old', 'gJ3_old', 'uJ3_old', 's13', 'r13', 'g13', 'g23_r17', 'u23_r17']
    M23 = np.stack([V3['s'], V3['r'], V3['c'], V3['c2'], V4['t'], V4['r23'] * 0 + D4['r3'], Wg23[J1], Wu23[J1], Wg23[J2], Wu23[J2], Wg23[J3], Wu23[J3], R13['s'], R13['r'], R13['gate_dir'], R17['gate23'], R17['up23']], 1).astype(np.float64)
    assert np.allclose(shipped['gate23'], p4['k3'] * (V4['t'] - p4['lam'] * V3['c'] - p4['lam2'] * V3['c2'] - p4['theta3'] * D4['r3']), atol=2e-5)
    BASIS = ['dJ1_old', 'dJ2_old', 'dJ3_old', 'uAB', 'uroute', 'uAB_v1', 'uconv', 'uAB_r13', 'uroute_r13', 'd23_r17']
    Bm = np.stack([Wd23[:, J1], Wd23[:, J2], Wd23[:, J3], V3['u_AB'], V3['u_route'], V1['u_AB'], u_conv, R13['u_AB'], R13['u_route'], R17['down23']], 0).astype(np.float64); G = Bm @ Bm.T
    Bkt_full = (Bm * gamma[None, :]) @ E.T  # (10, vocab)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(RT.BASE), local_files_only=True)
    enc1 = lambda w: (lambda ids: ids[0] if len(ids) == 1 else None)(tok.encode(w, add_special_tokens=False))
    named = {w: t for w, t in ((w, enc1(w)) for w in NAMED) if t is not None}
    rp = RT.Replayer(); W_orig = {'mlp.gate_proj.weight': Wg23.copy(), 'mlp.up_proj.weight': Wu23.copy(), 'mlp.down_proj.weight': Wd23.copy()}; rp.weights(W_orig)
    # ---------------------------------------------------------------- per item
    recs = []; checks = {'attention_formula_vs_replay_max_abs': 0.0, 'formula_vs_direct_max_abs': 0.0, 'argmax_agreement': {}, 'tail': {}, 'original_generation_token_identical': {}}
    def unroped_key_parts(h_end, kv):
        y0 = Wk[kv * HD:(kv + 1) * HD] @ (gin23 * h_end); y1 = Wk[kv * HD:(kv + 1) * HD] @ (gin23 * D4['p6']); return y0, y1
    for n, it in enumerate(idx):
        z = np.load(traces / f"{it['id']}.npz"); x22 = z['x22'].astype(np.float32); h23 = z['h23'].astype(np.float32); x23g = z['x23_gen'].astype(np.float32); T = it['T']; G_ = it['G']
        roles = it['roles']; dec = T - 1
        act0, z0, delta = RT.stage1(shipped, x22[:T], 1.0); act0_all, _, _ = RT.stage1(shipped, x22, 1.0)
        end = roles.get('end', int(np.argmax(act0)))
        p22 = (x22.astype(np.float64) @ M22)                                     # (T+G-1, 6)
        # attention closed-form scalars per head (decision query, end key): A, B, |y0|^2, y0.y1, |y1|^2, LSE_rest
        r_nat = rp.run(h23, dec, want_qk=True); q = r_nat['rows']['q'][:, dec, :].astype(np.float64); k = r_nat['rows']['k'].astype(np.float64)  # q (8,256) k (2,T,256)
        scale = 1 / np.sqrt(HD); heads = []
        for h in range(8):
            kv = h // 4; lg = scale * (k[kv] @ q[h]); lse_rest = np.logaddexp.reduce(np.delete(lg, end)); y0, y1 = unroped_key_parts(h23[end].astype(np.float64), kv)
            rms0 = np.sqrt((y0 @ y0) / HD + EPS); A = (q[h] @ k[kv, end]) * rms0  # k[kv,end] = RoPE(gk*y0)/rms(y0)
            heads.append({'A': float(A), 'y0sq': float(y0 @ y0), 'y01': float(y0 @ y1), 'y1sq': float(y1 @ y1), 'lse_rest': float(lse_rest), 'B': None, 'alpha_nat': float(r_nat['rows'][dec][h, end])})
        # B from one marked run (w0 = 100): k'(w0) = RoPE(gk*(y0+w0 y1))/rms(y0+w0 y1)
        w0 = 100.0; h2 = h23.copy(); h2[end] += w0 * D4['p6'].astype(np.float32); r_m = rp.run(h2, dec, want_qk=True); km = r_m['rows']['k'].astype(np.float64)
        for h in range(8):
            kv = h // 4; y0, y1 = unroped_key_parts(h23[end].astype(np.float64), kv); rmsw = np.sqrt(((y0 + w0 * y1) @ (y0 + w0 * y1)) / HD + EPS)
            heads[h]['B'] = float(((q[h] @ km[kv, end]) * rmsw - heads[h]['A']) / w0)
        def alpha_formula(h, w):
            H = heads[h]; l = scale * (H['A'] + w * H['B']) / np.sqrt((H['y0sq'] + 2 * w * H['y01'] + w * w * H['y1sq']) / HD + EPS); return float(np.exp(l - np.logaddexp(H['lse_rest'], l)))
        # states: replay with the ORIGINAL layer-23 MLP (edited neurons added in closed form) for each dose and the random control
        states = {}
        for name, R, b in [(f'b{d:g}', shipped, d) for d in DOSES] + [('rand17', rand17, 1.0)]:
            if b == 0 and name != 'rand17': h23e = h23 - (silu(x22[:T] @ Wg22[J0]) * (x22[:T] @ Wu22[J0]))[:, None].astype(np.float32) * Wd22[:, J0][None, :]
            else: _, _, dl = RT.stage1(R, x22[:T], b); h23e = h23 + dl
            wr = (b * RT.stage1(R, x22[:T], 1.0)[0]) if name != 'b0' else np.zeros(T)
            e = rp.run(h23e, dec); x23 = e['x23'][dec].astype(np.float64); h24 = e['h24'][dec].astype(np.float64)
            rms = np.sqrt((h24 @ h24) / D + EPS); L = ((h24 / rms) * gamma) @ E.T; hb = h24 @ Bm.T
            st = {'write_norm_end': float(wr[end] * np.linalg.norm(R['down22'])), 'x23': rnd(x23 @ M23, 5), 'rms': float(rms), 'sumsq': float(h24 @ h24), 'hb': rnd(hb, 5), 'lse': float(np.logaddexp.reduce(L)),
                  'alpha': rnd([e['rows'][dec][h, end] for h in range(8)], 5), 'attn_argmax6': int(np.argmax(e['rows'][dec][6])), '_L': L, '_x23': x23, '_h24': h24}
            if name != 'rand17':
                af = alpha_formula(6, float(wr[end])); checks['attention_formula_vs_replay_max_abs'] = max(checks['attention_formula_vs_replay_max_abs'], abs(af - st['alpha'][6]))
                st['alpha6_formula'] = af
            states[name] = st
        if n == 0: print('first item ok', flush=True)
        # x22 projections at the decision+generated positions (J0 silence during generation) and x23 projections at generated positions
        p23g = x23g.astype(np.float64) @ M23
        outs = recorded_outputs(it); ro = outs.get('original'); same = ro is not None and ro['token_ids'] == z['gen_ids'].tolist()
        recs.append({'it': it, 'T': T, 'G': G_, 'end': int(end), 'p22': p22, 'p23g': p23g, 'heads': heads, 'states': states, 'outs': outs, 'gen_ids': z['gen_ids'].tolist(), 'prompt_ids': z['prompt_ids'].tolist(), 'identical': same})
        if (n + 1) % 10 == 0 or n == 0: print(f"[{n + 1}/{len(idx)}] {it['id'][-40:]:40s} end={end} J0(end)={act0[end]:.1f} alpha6 b1={states['b1']['alpha'][6]:.3f} (formula {states['b1'].get('alpha6_formula', 0):.3f}) identical={same} {time.perf_counter() - t0:.0f}s", flush=True)
    rp.restore()
    # ---------------------------------------------------------------- tracked tokens and the exact readout
    def coef_for(rec, state, variant, a1=1.0, a2=1.0, lam=None, lam2=None, a3=None, rho=None, k=2.0, kr=1.0, k3=2.0):
        """coef over BASIS for a variant at a state (closed form on the stored x23 projections)."""
        x = dict(zip(X23P, rec['states'][state]['x23'])); c = np.zeros(len(BASIS))
        c[0] = -silu(x['gJ1_old']) * x['uJ1_old']; c[1] = -silu(x['gJ2_old']) * x['uJ2_old']; c[2] = -silu(x['gJ3_old']) * x['uJ3_old']
        if variant == 'original': return c
        L1 = {'g23v4': p4['lam'], 'g23v3': S3['lam'], 'v2': S3['lam'], 'v1': 0.0, 'conv3': S3['lam'], 'conv4': p4['lam']}.get(variant, S3['lam']) if lam is None else lam
        L2 = {'g23v4': p4['lam2'], 'g23v3': S3['lam2'], 'v2': 0.0, 'v1': 0.0, 'conv3': S3['lam2'], 'conv4': p4['lam2']}.get(variant, S3['lam2']) if lam2 is None else lam2
        if variant in ('rand13', 'rand17'):
            if variant == 'rand13': gate = k * x['g13']; act1 = silu(gate) * (k * x['s13']); act2 = silu(gate) * (kr * x['r13']); c[7] = a1 * act1; c[8] = a2 * act2; return c
            gate = k * (x['s'] - L1 * x['c'] - L2 * x['c2']); act1 = silu(gate) * (k * x['s']); act2 = silu(gate) * (kr * x['r']); c[3] = a1 * act1; c[4] = a2 * act2
            act3 = silu(x['g23_r17']) * x['u23_r17']; c[9] = act3; return c
        gate = k * (x['s'] - L1 * x['c'] - L2 * x['c2']); act1 = silu(gate) * (k * x['s']); act2 = silu(gate) * (kr * x['r'])
        if variant == 'v1': c[5] = a1 * act1; c[4] = a2 * act2
        elif variant in ('conv3', 'conv4'): c[6] = (0.5 if a1 == 1.0 else a1) * act1
        else: c[3] = a1 * act1; c[4] = a2 * act2
        if variant in ('g23v4', 'conv4'):
            A3 = p4['a3'] if a3 is None else a3; RHO = p4['rho'] if rho is None else rho
            g3 = k3 * (x['t'] - L1 * x['c'] - L2 * x['c2'] - p4['theta3'] * x['r3']); act3 = silu(g3) * (k3 * (x['t'] - p4['theta3'] * x['r3']))
            if variant == 'conv4': c[6] += A3 * act3
            else: c[3] += A3 * act3; c[4] += A3 * RHO * act3
        return c
    def logits_direct(rec, state, c): hn = rec['states'][state]['_h24'] + c @ Bm; return ((hn / np.sqrt((hn @ hn) / D + EPS)) * gamma) @ E.T
    tracked = set(named.values())
    for kk in range(Bm.shape[0]):
        order = np.argsort(Bkt_full[kk]); tracked.update(order[-48:].tolist()); tracked.update(order[:48].tolist())
    settings = [('original', 'b0'), ('g23v3', 'b0'), ('v1', 'b0'), ('v2', 'b0'), ('rand13', 'b0'), ('g23v4', 'b1'), ('g23v4', 'b0.5'), ('g23v4', 'b2'), ('rand17', 'rand17'), ('conv3', 'b0'), ('conv4', 'b1')]
    for rec in recs:
        for st in rec['states'].values(): tracked.update(np.argsort(st['_L'])[-24:].tolist())
        for v, s in settings: tracked.update(np.argsort(logits_direct(rec, s, coef_for(rec, s, v)))[-16:].tolist())
    tracked = np.array(sorted(tracked)); pos = {t: j for j, t in enumerate(tracked.tolist())}; print(f'tracked tokens: {len(tracked)}', flush=True)
    def logits_formula(rec, state, c):
        st = rec['states'][state]; num = st['_L'][tracked] * st['rms'] + c @ Bkt_full[:, tracked]; r2 = (st['sumsq'] + 2 * c @ np.array(st['hb']) + c @ G @ c) / D + EPS; return num / np.sqrt(r2)
    agree = {}; tails = {}
    for rec in recs:
        it = rec['it']
        for v, s in settings:
            c = coef_for(rec, s, v); full = logits_direct(rec, s, c); lf = logits_formula(rec, s, c)
            checks['formula_vs_direct_max_abs'] = max(checks['formula_vs_direct_max_abs'], float(np.abs(lf - full[tracked]).max()))
            p = np.exp(full - full.max()); p /= p.sum(); tail_lse = np.logaddexp.reduce(np.delete(rec['states'][s]['_L'], tracked)); mm = max(lf.max(), tail_lse); q = np.exp(lf - mm); q /= q.sum() + np.exp(tail_lse - mm)
            top10 = np.argsort(full)[-10:]; err = max(abs(q[pos[t]] - p[t]) if t in pos else p[t] for t in top10.tolist())
            ts = tails.setdefault(f'{v}@{s}', {'n': 0, 'max_abs_prob_err_top10': 0.0, 'min_tracked_mass': 1.0}); ts['n'] += 1; ts['max_abs_prob_err_top10'] = max(ts['max_abs_prob_err_top10'], float(err)); ts['min_tracked_mass'] = min(ts['min_tracked_mass'], float(p[tracked].sum()))
            rec_key = {'g23v4': 'g23v4', 'rand17': 'ctrl_rand17'}.get(v, v); am = int(full.argmax())
            if (v, s) in (('original', 'b0'), ('g23v3', 'b0'), ('v1', 'b0'), ('v2', 'b0'), ('rand13', 'b0'), ('g23v4', 'b1'), ('rand17', 'rand17')) and rec['outs'].get(rec_key) and rec['outs'][rec_key]['first_token'] is not None:
                ag = agree.setdefault(rec_key, {'match': 0, 'n': 0, 'mismatch_ids': []}); ag['n'] += 1
                if am == rec['outs'][rec_key]['first_token']: ag['match'] += 1
                else: ag['mismatch_ids'].append(it['id'])
        ag = agree.setdefault('original_fresh', {'match': 0, 'n': 0, 'mismatch_ids': []}); ag['n'] += 1
        if int(rec['states']['b0']['_L'].argmax()) == rec['gen_ids'][0]: ag['match'] += 1
        else: ag['mismatch_ids'].append(it['id'])
    checks['argmax_agreement'] = agree; checks['tail'] = tails
    for s in sorted({r['it']['set'] for r in recs}):
        rs = [r for r in recs if r['it']['set'] == s]; checks['original_generation_token_identical'][s] = f"{sum(r['identical'] for r in rs)}/{len(rs)}"
    print(json.dumps(checks, indent=1)[:4000], flush=True)
    # ---------------------------------------------------------------- bundle
    tl = tracked.tolist()
    bundle = {'meta': {'title': 'Four-neuron circuit explorer data (g23v3 + g23v4)', 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'base_model': 'Qwen/Qwen3.5-0.8B',
                       'base_revision': '2fc06364715b967f1860aea9cf38778875588b17', 'edits': ['g23v3', 'g23v4'], 'n_items': len(recs), 'checks': checks, 'doses': DOSES,
                       'provenance': 'Fresh non-budgeted capture of the ORIGINAL model (scripts/capture_traces.py) + exact CPU replay of layer 23 per stage-1 dose (this script). Recorded outputs copied verbatim from the campaign run files under experiments/.',
                       'readout': 'logit_t = (L_t*rms + coef.B_t)/sqrt((sumsq + 2 coef.hb + coef^T G coef)/1024 + eps) over the basis; J0: z = k1 (w1.x22 - theta1 r1.x22), act = silu(z) z; J1/J2: gate = k(s - lam c - lam2 c2), act1 = silu(gate) k s, act2 = silu(gate) kr r; '
                                  'J3: g3 = k3 (t - lam c - lam2 c2 - theta3 r3), act3 = silu(g3) k3 (t - theta3 r3); coef[uAB] = a1 act1 + a3 act3, coef[uroute] = a2 act2 + a3 rho act3, coef[dJ*_old] = -silu(gJ*.x) uJ*.x; '
                                  'alpha_h(w) = exp(l)/(exp(lse_rest)+exp(l)), l = (A + w B)/sqrt((y0sq + 2 w y01 + w^2 y1sq)/256 + eps)/16 with w = b * act_J0(end) * |p6|'},
              'constants': {'g23v3': {k: S3[k] for k in ('k', 'kr', 'a1', 'a2', 'lam', 'lam2')}, 'g23v4': p4, 'J0': J0, 'J1': J1, 'J2': J2, 'J3': J3, 'eps': EPS, 'd': D, 'head_dim': HD, 'vocab': int(E.shape[0]),
                            'p6_norm': float(np.linalg.norm(shipped['down22'])), 'changed_entries': {'g23v3': 6142, 'g23v4': 12281}, 'conv_a1': 0.5, 'conv_a3': json.load(open(ROOT / 'patches/g23v4_conv.params.json'))['a3']},
              'x22_proj_names': X22P, 'x23_proj_names': X23P, 'basis_names': BASIS, 'gram': rnd(G, 6),
              'variants': {'original': {'label': 'Base Qwen3.5-0.8B'}, 'v1': {'label': 'v1 (g23_t05_a1.0): no gates, prose collapses'}, 'v2': {'label': 'v2 (g23v2): tool-context gate c only'},
                           'g23v3': {'label': 'g23v3 (2 neurons, layer 23)'}, 'g23v4': {'label': 'g23v4 (4 neurons, layers 22+23)'}, 'rand13': {'label': 'Random control on J1/J2 (seed 13, norm-matched)'},
                           'rand17': {'label': 'Random control on J0/J3 (seed 17, norm-matched; g23v3 rows kept)'}, 'conv3': {'label': 'g23v3 conversational readout'}, 'conv4': {'label': 'g23v4 conversational readout'}},
              'tracked': {'ids': tl, 'text': [tok.decode([t]) for t in tl], 'named': named, 'B': rnd(Bkt_full[:, tracked], 4)}, 'items': []}
    for rec in recs:
        it = rec['it']; T = rec['T']
        states = {name: {k: v for k, v in st.items() if not k.startswith('_')} | {'logits': rnd(st['_L'][tracked], 3), 'base_top': [[int(t), round(float(st['_L'][t]), 3)] for t in np.argsort(st['_L'])[-10:][::-1]]} for name, st in rec['states'].items()}
        bundle['items'].append({'id': it['id'], 'set': it['set'], 'cohort': it['cohort'], 'family': it['family'], 'action': it['action'], 'ambiguous': it['ambiguous'], 'renamed': it['renamed'], 'expected': it['expected'],
                                'system': it['system'], 'user': it['user'], 'tools': it['tools'], 'T': T, 'end': rec['end'], 'roles': it['roles'],
                                'prompt': {'ids': rec['prompt_ids'], 'text': [tok.decode([t]) for t in rec['prompt_ids']], 'x22': rnd(rec['p22'][:T], 4)},
                                'generated': {'ids': rec['gen_ids'], 'text': [tok.decode([t]) for t in rec['gen_ids']], 'x22': rnd(rec['p22'][T - 1:], 4), 'x23': rnd(rec['p23g'], 4)},
                                'heads': rec['heads'], 'states': states, 'outputs': rec['outs'], 'fresh_original_identical': rec['identical']})
    p = out / 'circuit_traces.json'; p.write_text(json.dumps(bundle, ensure_ascii=False, separators=(',', ':')))
    print(f'wrote {p} {p.stat().st_size / 1e6:.2f} MB sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]} in {time.perf_counter() - t0:.0f}s')

if __name__ == '__main__': main()
