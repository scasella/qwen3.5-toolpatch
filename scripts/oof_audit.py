#!/usr/bin/env python3
"""Strict out-of-fold audit of the g23v4 circuit, CPU replay only (0 forward passes of the budgeted kind).

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python; $P -B scripts/oof_audit.py [--theta1 9.6]

The shipped g23v4 directions were fit on all 192 captured triple items (D1 discovery families + D2 renamed twins + D3 selection/final
families with unseen phrasings) and tested on the 64-item Ambiguity Stress Suite. This audit refits EVERY direction of the circuit on
D1+D2 only -- u, g (stage-1 conjunction), r1 (constant-projection bias), p6 (head-6 key beacon), t (delivered feature), r3 (layer-23 bias)
-- with the stage-1 threshold theta1 set midway in the D1+D2->D3 transfer gap reported in the qwen-7 notes (ambiguous min 10.2 vs harmful
max 9.0) and the gain k1 chosen by a D1+D2-only rule (match the shipped minimum stage-1 write of 58 at the D1+D2 ambiguous minimum), and
then evaluates it once on the held-out D3 items (28 ambiguous, 18 call, 18 resolved) and on the 64 stress-suite items captured by
scripts/capture_traces.py. Two things are reported separately: (a) direction-level separation of the D1+D2-fit stage-1 gate on the
held-out positions, (b) the end-to-end replayed decision token. The shipped circuit is replayed on the same items for comparison, and
the shipped replay is compared with the real edited model's teacher-forced logits (pass c01) as the simulator check.
Writes experiments/qwen7_g23v4/analysis/oof_audit.{json,md} and docs/data/oof_audit.json. Reads the campaigns read-only."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import replay_tools as RT
from replay_tools import ROOT, QWEN6, QWEN7, J0, J1, J2, J3, TOK, silu, unit
EXP6, EXP7 = ROOT / 'experiments/qwen6_g23v3', ROOT / 'experiments/qwen7_g23v4'

def lda_dir(Xa, Xb, shrink=0.5):
    mu_a = Xa.mean(0); mu_b = Xb.mean(0); R = np.concatenate([Xa - mu_a, Xb - mu_b]); S = R.T @ R / len(R)
    S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0]); return unit(np.linalg.solve(S, mu_a - mu_b))
def const_dir(X, shrink=0.3):
    mu = X.mean(0); R = X - mu; C = R.T @ R / len(R); C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0]); r = np.linalg.solve(C, mu); return r / (mu @ r)

def prompt_tokens(tok, cap, cid):
    """Token strings of the rendered prompt of a triple item (for the '.'-token mask of the clause-end gate negatives)."""
    import fp2
    from benchmark.render import prompt as render
    it = fp2.items_for(cap.set, [cid])[0]; p = render(tok, it['input'], 'standard', False); ids = tok.encode(p, add_special_tokens=False)
    return ids, [tok.decode([t]) for t in ids]

def fit_stage1(caps, tok, rng):
    """u (unresolved) and g (user fact-clause end) shrinkage-LDA directions at the layer-22 MLP input, as g23v4_build.fit_directions, on the given captures only."""
    ends = []; var = []; negs = []
    for c in caps:
        for i in c.ids:
            X = c.full(i, 'mlpin22'); ro = c.roles(i); ends.append(X[ro['end']]); var.append(c.meta[i]['variant'])
            ids, toks = prompt_tokens(tok, c, i); ustart = ids.index(tok.encode('<|im_start|>', add_special_tokens=False)[0], 1)
            idx = np.arange(len(X)); dot = np.array([('.' in t) for t in toks])
            ok = (idx <= ustart) | ((idx > ustart) & (idx <= ro['imend']) & ~dot) | (idx > ro['imend']); ok[ro['end']] = False
            negs.append(X[rng.choice(idx[ok], 60, replace=False)])
    ends = np.stack(ends).astype(np.float64); var = np.array(var); negs = np.concatenate(negs).astype(np.float64)
    return lda_dir(ends[var == 'ambiguous'], ends[var != 'ambiguous'], 0.5), lda_dir(ends, negs, 0.5), ends, var

def harmful_positions(caps):
    """Every layer-22 MLP-input position of non-ambiguous items of the given triple captures, plus qwen-6 generated captures (calls, NO_TOOL,
    explicit-missing continuations; ambiguous practice items excluded) and the qwen-7 g01 capture (prose and short non-tool answers)."""
    groups = {}
    for c in caps:
        for i in c.ids:
            if c.meta[i]['variant'] != 'ambiguous': groups.setdefault(('triple', c.meta[i]['variant']), []).append(c.full(i, 'mlpin22'))
    for lab in ('p04c_practice_allpos', 'p05_teacher_allpos'):
        js = json.loads((QWEN6 / 'runs' / f'{lab}.json').read_text()); z = np.load(QWEN6 / 'runs' / f'{lab}.npz')
        for r in js['rows']:
            if r['action'] == 'clarify': continue   # as in the campaign: a marker at an explicit-missing clause end only strengthens the gold NEED_CLARIFICATION
            groups.setdefault(('q6', r['action']), []).append(z[f"mlpin_L22_{r['case_id']}"].astype(np.float32))
    g01 = RT.load_capture('g01_genpos')
    for i in g01.ids: groups.setdefault(('g01', 'prose' if i.startswith('prose') else 'reg2'), []).append(g01.full(i, 'mlpin22'))
    return groups

def generated23(g01):
    gen = []
    for lab in ('p04c_practice_allpos', 'p05_teacher_allpos', 'p17_prose_capture', 'p35_prose_capture_nontool', 'p16_regression_capture'):
        js = json.loads((QWEN6 / 'runs' / f'{lab}.json').read_text()); z = np.load(QWEN6 / 'runs' / f'{lab}.npz')
        for r in js['rows']: X = z[f"mlpin_L23_{r['case_id']}"].astype(np.float32); gen.append(X[r['decision_position']:])
    for i in g01.ids: gen.append(g01.full(i, 'mlpin23')[g01.rows[i]['decision_position']:])
    return np.concatenate(gen).astype(np.float64)

def fit_stage2(caps, rp, p6, V3):
    """t = unit mean delivered change of the layer-23 MLP input at the decision when the clause end carries +100 p6; r3 = constant-projection direction."""
    dx = []; nat = []; dl = []
    for c in caps:
        for i in c.ids:
            h23 = c.full(i, 'h23'); ro = c.roles(i); b = rp.run(h23, ro['dec']); nat.append(b['x23'][ro['dec']])
            if c.meta[i]['variant'] == 'ambiguous':
                h2 = h23.copy(); h2[ro['end']] += 100 * p6; e = rp.run(h2, ro['dec']); dx.append(e['x23'][ro['dec']] - b['x23'][ro['dec']]); dl.append(e['x23'][ro['dec']])
    t = unit(np.stack(dx).mean(0)); nat = np.stack(nat).astype(np.float64); dl = np.stack(dl).astype(np.float64)
    gen = generated23(RT.load_capture('g01_genpos')); tg = t - 0.3 * V3['c'] - 0.5 * V3['c2']; s = gen @ tg
    r3 = const_dir(np.concatenate([nat, dl, gen[np.argsort(s)[-3000:]]]), 0.3)
    return t, r3, nat, dl, gen

def fit_beacon(caps, rp):
    Wk = RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.layers.23.self_attn.k_proj.weight'))
    gin = RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.layers.23.input_layernorm.weight')) + 1.0
    gk = RT.f32(RT.read_bits(RT.BASE_FILE, 'model.language_model.layers.23.self_attn.k_norm.weight')) + 1.0
    Q = []
    for c in caps:
        for i in c.ids[:48]:
            r = rp.run(c.full(i, 'h23'), c.roles(i)['dec'], want_qk=True); Q.append(r['rows']['q'][:, c.roles(i)['dec'], :])
    Q = np.stack(Q); kt = np.zeros(256); kt[64:] = (gk * Q[:, 6, :].mean(0))[64:]
    for h2 in (4, 5, 7):
        o = (Q[:, h2, :].mean(0) * gk).copy(); o[:64] = 0; kt -= (kt @ o) / (o @ o) * o
    return unit(np.linalg.pinv(Wk[256:512] * gin[None, :]) @ kt)

def evaluate(rp, R_test, R_ref, items, tag):
    """items: list of (id, variant, x22 (T,1024), h23 (T,1024), roles). Replays g23v3-only (R_ref) and the test circuit; returns per-item rows."""
    W_ref = RT.layer23_tensors(R_ref); W_test = RT.layer23_tensors(R_test); rows = []
    for (cid, var, x22, h23, ro) in items:
        act, z, delta = RT.stage1(R_test, x22[:len(h23)]); h23e = h23 + delta
        rp.weights(W_ref); b = rp.run(h23, ro['dec']); lb = rp.logits(b['h24'][ro['dec']])
        rp.weights(W_test); e = rp.run(h23e, ro['dec']); le = rp.logits(e['h24'][ro['dec']]); x23 = e['x23'][ro['dec']]
        ja = RT.j_acts(R_test, x23[None]); end = ro.get('end')
        other = np.delete(act, end) if end is not None else act
        rows.append({'id': cid, 'var': var, 'set': tag, 'ref_argmax': int(lb.argmax()), 'test_argmax': int(le.argmax()), 'NE_call': float(le[TOK['NE']] - le[TOK['call']]),
                     'NE_NO': float(le[TOK['NE']] - le[TOK['NO']]), 'NE_other': float(le[TOK['NE']] - np.delete(le, TOK['NE']).max()),
                     'logits': {k: float(le[v]) for k, v in TOK.items()}, 'ref_logits': {k: float(lb[v]) for k, v in TOK.items()},
                     'z_end': float(z[end]) if end is not None else None, 'act_end': float(act[end]) if end is not None else None, 'act_max_other': float(other.max()),
                     'z_max_other': float(np.delete(z, end).max() if end is not None else z.max()), 'alpha6_end': float(e['rows'][ro['dec']][6, end]) if end is not None else None,
                     'j3_act': float(ja['J3'][0]), 'j1_act': float(ja['J1'][0]), 'max_dlogit_top20': float(np.abs(le[np.argsort(lb)[::-1][:20]] - lb[np.argsort(lb)[::-1][:20]]).max())})
    return rows

def summarize(rows):
    amb = [r for r in rows if r['var'] == 'ambiguous']; non = [r for r in rows if r['var'] != 'ambiguous']
    out = {'n_ambiguous': len(amb), 'n_non_ambiguous': len(non),
           'ambiguous_to_NE': sum(r['test_argmax'] == TOK['NE'] for r in amb),
           'ambiguous_NE_minus_call_min': min((r['NE_call'] for r in amb), default=None), 'ambiguous_NE_minus_runnerup_min': min((r['NE_other'] for r in amb), default=None),
           'non_ambiguous_argmax_kept': sum(r['test_argmax'] == r['ref_argmax'] for r in non), 'non_ambiguous_max_abs_dlogit_top20': max((r['max_dlogit_top20'] for r in non), default=None),
           'stage1_act_at_ambiguous_ends': [min(r['act_end'] for r in amb), max(r['act_end'] for r in amb)] if amb else None,
           'stage1_z_at_ambiguous_ends': [min(r['z_end'] for r in amb), max(r['z_end'] for r in amb)] if amb else None,
           'stage1_act_max_non_ambiguous_positions': max((max(r['act_end'] or 0.0, r['act_max_other']) for r in non), default=None),
           'stage1_z_max_non_ambiguous_positions': max((max(r['z_end'] if r['z_end'] is not None else -99, r['z_max_other']) for r in non), default=None),
           'stage1_act_max_ambiguous_other_positions': max((r['act_max_other'] for r in amb), default=None),
           'alpha6_at_end_ambiguous': [min(r['alpha6_end'] for r in amb), max(r['alpha6_end'] for r in amb)] if amb else None,
           'alpha6_at_end_non_ambiguous_max': max((r['alpha6_end'] for r in non if r['alpha6_end'] is not None), default=None),
           'j3_act_ambiguous': [min(r['j3_act'] for r in amb), max(r['j3_act'] for r in amb)] if amb else None, 'j3_act_non_ambiguous_max': max((r['j3_act'] for r in non), default=None),
           'misses': [(r['id'], r['test_argmax'], round(r['NE_other'], 2), round(r['act_end'] or 0, 2)) for r in amb if r['test_argmax'] != TOK['NE']]}
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--theta1', type=float, default=9.6); ap.add_argument('--traces', default=str(ROOT / 'artifacts/traces')); a = ap.parse_args()
    t0 = time.perf_counter(); rng = np.random.default_rng(1)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(RT.BASE), local_files_only=True)
    D1, D2, D3 = (RT.load_capture(l) for l in ('d01b_triples_D1', 'd02_triples_D2', 'd03_triples_D3'))
    V3 = np.load(EXP6 / 'edits/g23v3.vectors.npz'); S4 = json.load(open(ROOT / 'patches/g23v4.spec.json')); p4 = S4['params']
    shipped = RT.patch_rows('g23v4'); ref = dict(shipped); ref.update({'gate22': RT.original(22, 'gate_proj')[J0], 'up22': RT.original(22, 'up_proj')[J0], 'down22': RT.original(22, 'down_proj')[:, J0],
                                                                   'gate23': RT.original(23, 'gate_proj')[J3], 'up23': RT.original(23, 'up_proj')[J3], 'down23': RT.original(23, 'down_proj')[:, J3]})
    # ---------------- stage 1 on D1+D2 only
    u, g, ends12, var12 = fit_stage1([D1, D2], tok, rng); w1 = unit(p4['a_g'] * g + u)
    harm = harmful_positions([D1, D2]); allharm = np.concatenate([X for v in harm.values() for X in v]).astype(np.float64)
    pos12 = ends12[var12 == 'ambiguous']; sc = allharm @ w1; r1 = const_dir(np.concatenate([pos12, allharm[np.argsort(sc)[-3000:]]]), 0.1)
    w1g = w1 - a.theta1 * r1; z12 = pos12 @ w1g; zh = allharm @ w1g
    # D1+D2-only gain rule: the shipped circuit's minimum stage-1 write on captured ambiguous ends was 58 -> choose k1 so that min_{D1+D2 amb} silu(k1 z) k1 z = 58
    zmin = float(z12.min()); lo, hi = 0.5, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2; val = float(silu(mid * zmin) * mid * zmin) if zmin > 0 else -1
        lo, hi = (mid, hi) if val < 58 else (lo, mid)
    k1_rule = (lo + hi) / 2 if zmin > 0 else p4['k1']; k1 = p4['k1']
    stage1_report = {'theta1': a.theta1, 'k1': k1, 'k1_rule_value': k1_rule, 'k1_rule': 'alternative gain: min D1+D2 ambiguous-end activation = 58 (shipped minimum)', 'fit_items': {'D1': len(D1.ids), 'D2': len(D2.ids)},
                     'D12_ambiguous_w1x': [float((pos12 @ w1).min()), float((pos12 @ w1).max())], 'D12_ambiguous_z': [float(z12.min()), float(z12.max())],
                     'D12_harmful_w1x_max': float(sc.max()), 'D12_harmful_z_max': float(zh.max()), 'D12_harmful_positions': int(len(allharm)), 'r1x_on_ambiguous_ends': [float((pos12 @ r1).min()), float((pos12 @ r1).max())]}
    print('stage-1 (fit D1+D2):', json.dumps(stage1_report), flush=True)
    rp = RT.Replayer()
    p6 = fit_beacon([D1, D2], rp); t, r3, nat12, dl12, gen23 = fit_stage2([D1, D2], rp, p6, V3)
    tg = t - p4['lam'] * V3['c'] - p4['lam2'] * V3['c2'] - p4['theta3'] * r3
    def oof_rows(kk):
        o = dict(shipped); o.update({'gate22': (kk * w1g).astype(np.float32), 'up22': (kk * w1g).astype(np.float32), 'down22': (p4['b'] * p6).astype(np.float32),
                                    'gate23': (p4['k3'] * tg).astype(np.float32), 'up23': (p4['k3'] * (t - p4['theta3'] * r3)).astype(np.float32), 'down23': (p4['a3'] * (V3['u_AB'] + p4['rho'] * V3['u_route'])).astype(np.float32)})
        return o
    oof = oof_rows(k1); oof_rule = oof_rows(k1_rule)
    stage2_report = {'cos_p6_oof_vs_shipped': float(abs(p6 @ unit(shipped['down22'].astype(np.float64)))), 'cos_t_oof_vs_shipped_t': None,
                     'J3_z_delivered_D12': [float((dl12 @ tg).min()), float((dl12 @ tg).max())], 'J3_z_natural_decisions_D12_max': float((nat12 @ tg).max()), 'J3_z_generated_max': float((gen23 @ tg).max())}
    V4 = np.load(EXP7 / 'edits/g23v4.vectors.npz'); stage2_report['cos_t_oof_vs_shipped_t'] = float(abs(t @ unit(V4['t'].astype(np.float64))))
    stage2_report['cos_u_oof_vs_shipped'] = float(abs(u @ unit(V4['u']))); stage2_report['cos_g_oof_vs_shipped'] = float(abs(g @ unit(V4['g'])))
    print('stage-2 (fit D1+D2):', json.dumps(stage2_report), flush=True)
    # ---------------- held-out evaluation: D3 (qwen-7 capture) and the stress suite (fresh capture of the original model)
    d3_items = [(i, D3.meta[i]['variant'], D3.full(i, 'mlpin22'), D3.full(i, 'h23'), D3.roles(i)) for i in D3.ids]
    idx = json.load(open(Path(a.traces) / 'index.json'))['items']; stress_items = []
    for it in idx:
        if it['set'] != 'stress': continue
        z = np.load(Path(a.traces) / f"{it['id']}.npz"); stress_items.append((it['id'], 'ambiguous' if it['ambiguous'] else 'resolved', z['x22'].astype(np.float32), z['h23'].astype(np.float32), it['roles']))
    results = {}
    for tag, items in (('D3', d3_items), ('stress', stress_items)):
        for name, R in (('oof_D1D2', oof), ('oof_D1D2_k1rule', oof_rule), ('shipped', shipped)):
            rows = evaluate(rp, R, ref, items, tag); results[f'{name}:{tag}'] = {'summary': summarize(rows), 'rows': rows}
            print(f'{name:9s} on {tag:6s}:', json.dumps(results[f"{name}:{tag}"]["summary"]), flush=True)
    # direction-level separation on the held-out positions (stage-1 gate pre-activation z, before the gain)
    def z_stats(items, R):
        za = []; zh = []
        for (cid, var, x22, h23, ro) in items:
            z = (x22[:len(h23)] @ R['gate22']) / (R['gate22'] @ R['gate22']) ** 0.5 * np.linalg.norm(w1g)  # rescale to the w1g scale
            if var == 'ambiguous': za.append(float(z[ro['end']])); zh.append(float(np.delete(z, ro['end']).max()))
            else: zh.append(float(z.max()))
        return {'ambiguous_end_min': min(za), 'ambiguous_end_max': max(za), 'other_positions_max': max(zh), 'margin': min(za) - max(zh), 'sign_correct_ambiguous': sum(v > 0 for v in za), 'sign_correct_other_items': sum(v < 0 for v in zh), 'n_amb': len(za), 'n_other': len(zh)}
    results['direction_level'] = {'oof_D1D2:D3': z_stats(d3_items, oof), 'oof_D1D2:stress': z_stats(stress_items, oof)}
    print('direction level:', json.dumps(results['direction_level']), flush=True)
    # ---------------- simulator check: shipped replay on D1 vs the real edited model (pass c01, teacher-forced)
    c01 = RT.load_capture('c01_calib_D1'); W4 = RT.layer23_tensors(shipped); rp.weights(W4); agree = 0; errs = []; scatter = []
    for i in D1.ids:
        h23 = D1.full(i, 'h23'); ro = D1.roles(i); act, z, delta = RT.stage1(shipped, D1.full(i, 'mlpin22')); e = rp.run(h23 + delta, ro['dec']); le = rp.logits(e['h24'][ro['dec']]); real = c01.logits(i)
        agree += int(le.argmax() == real.argmax()); err = max(abs(float(le[v] - real[v])) for v in TOK.values()); errs.append(err)
        scatter.append({'id': i, 'var': D1.meta[i]['variant'], 'sim': {k: float(le[v]) for k, v in TOK.items()}, 'real': {k: float(real[v]) for k, v in TOK.items()},
                        'alpha6_sim': float(e['rows'][ro['dec']][6, ro['end']]), 'alpha6_real': float(c01.rows_attn(i)[5, 0][6, ro['end']]),
                        'j0_sim': float(act[ro['end']]), 'j0_real': float(c01.z[f'mlp_{i}'][c01.pos_index(i, 'end'), 0][J0]),
                        'j3_sim': float(RT.j_acts(shipped, e['x23'][ro['dec']][None])['J3'][0]), 'j3_real': float(c01.z[f'mlp_{i}'][c01.pos_index(i, 'dec'), 1][J3])})
    rp.restore()
    results['simulator_vs_real_D1'] = {'argmax_agreement': f'{agree}/{len(D1.ids)}', 'abs_logit_error_NE_NO_call_median': float(np.median(errs)), 'max': float(max(errs)), 'rows': scatter}
    print('simulator vs real (D1, shipped):', results['simulator_vs_real_D1']['argmax_agreement'], 'median err', round(float(np.median(errs)), 3), 'max', round(float(max(errs)), 3), flush=True)
    out = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage1_fit_D1D2': stage1_report, 'stage2_fit_D1D2': stage2_report, 'results': results, 'seconds': time.perf_counter() - t0}
    (EXP7 / 'analysis/oof_audit.json').write_text(json.dumps(out, indent=1)); (ROOT / 'docs/data').mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in out.items() if k != 'results'}; slim['results'] = {k: ({'summary': v['summary']} if 'summary' in v else {kk: vv for kk, vv in v.items() if kk != 'rows'}) for k, v in results.items()}
    slim['results']['direction_level'] = results['direction_level']; (ROOT / 'docs/data/oof_audit.json').write_text(json.dumps(slim, indent=1))
    # markdown table
    s = lambda k: results[k]['summary']
    md = ['| circuit | fit items | held-out set | ambiguous -> NEED_CLARIFICATION (replay argmax) | non-ambiguous argmax kept | stage-1 act: ambiguous ends / max elsewhere | alpha6 at clause end (ambiguous) | J3 act: ambiguous / max elsewhere |', '|---|---|---|---|---|---|---|---|']
    for name, fit in (('oof_D1D2', f'D1+D2 (128), k1 = {k1}'), ('oof_D1D2_k1rule', f'D1+D2 (128), k1 = {k1_rule:.2f}'), ('shipped', 'D1+D2+D3 (192), k1 = 4')):
        for tag in ('D3', 'stress'):
            q = s(f'{name}:{tag}')
            md.append(f"| {name} | {fit} | {tag} | {q['ambiguous_to_NE']}/{q['n_ambiguous']} | {q['non_ambiguous_argmax_kept']}/{q['n_non_ambiguous']} | {q['stage1_act_at_ambiguous_ends'][0]:.1f}..{q['stage1_act_at_ambiguous_ends'][1]:.1f} / {q['stage1_act_max_non_ambiguous_positions']:.3f} | {q['alpha6_at_end_ambiguous'][0]:.3f}..{q['alpha6_at_end_ambiguous'][1]:.3f} | {q['j3_act_ambiguous'][0]:.1f}..{q['j3_act_ambiguous'][1]:.1f} / {q['j3_act_non_ambiguous_max']:.4f} |")
    dl = results['direction_level']
    md += ['', f"Stage-1 gate fit on D1+D2 only (theta1 = {a.theta1}; gain-independent): pre-activation z at held-out ambiguous clause ends vs every other held-out position:",
           *[f"- {k}: ambiguous ends [{v['ambiguous_end_min']:.2f}, {v['ambiguous_end_max']:.2f}] ({v['sign_correct_ambiguous']}/{v['n_amb']} above threshold), other positions max {v['other_positions_max']:.2f} ({v['sign_correct_other_items']}/{v['n_other']} items fully below), margin {v['margin']:+.2f}" for k, v in dl.items()],
           '', f"Simulator check (shipped g23v4 replay vs real edited model, pass c01, 64 D1 items): argmax agreement {results['simulator_vs_real_D1']['argmax_agreement']}, |logit error| on NE/NO/<tool_call> median {results['simulator_vs_real_D1']['abs_logit_error_NE_NO_call_median']:.3f}, max {results['simulator_vs_real_D1']['max']:.3f}."]
    (EXP7 / 'analysis/oof_audit.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md)); print(f'done in {time.perf_counter() - t0:.0f}s')

if __name__ == '__main__': main()
