#!/usr/bin/env python3
"""Publication figures for the unified g23v3 / g23v4 release, computed from the campaigns' recorded data (read-only inputs under
experiments/ and the derived captures under artifacts/traces). Writes SVG + PNG into docs/figures/.

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python; $P -B scripts/make_figures.py

fig1_progression_overview      Base -> g23v3 -> g23v4 on the 192 tool items (per class), the 64-item stress suite, non-tool cohorts; reversed /
                               random controls on the 64-item control set            (analysis/final_tables.json, analysis/controls.json)
fig2_positional_bottleneck_dprime  leave-one-family-out d' per layer at the decision token (qwen-6 practice capture p03: NO_TOOL, explicit-missing,
                               ambiguous vs calls; qwen-7 192-item triples: ambiguous vs call/resolved, raw and shrinkage-LDA) and at the user's
                               clause tokens (qwen-7 diag2)                           (runs/p03_practice_dec.npz, analysis/diag2_*.json)
fig3_four_neuron_circuit_diagram   the two-act circuit as a wiring diagram with the real activation distributions of J1 (fresh original-model
                               capture, all cohorts), J0 at the clause end, head-6 attention and J3 at the decision (pass c01 / c02, real g23v4)
fig4_dose_and_controls         dose-response on the 64-item control set and the CPU replay vs real-model logits scatter (pass c01)
                                                                                        (analysis/controls.json, analysis/oof_audit.json)"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
EXP6, EXP7 = ROOT / 'experiments/qwen6_g23v3', ROOT / 'experiments/qwen7_g23v4'; Q6 = Path('/Users/scasella/Downloads/qwen-6'); Q7 = Path('/Users/scasella/Downloads/qwen-7')
OUT = ROOT / 'docs/figures'; OUT.mkdir(parents=True, exist_ok=True)
C = {'base': '#2a78d6', 'g23v3': '#eb6834', 'g23v4': '#1baf7a', 'ctrl': '#8a8984', 'ink': '#0b0b0b', 'muted': '#52514e', 'grid': '#e6e5e1', 'surface': '#fcfcfb'}
plt.rcParams.update({'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9.5, 'axes.edgecolor': C['muted'], 'axes.labelcolor': C['ink'],
                     'xtick.color': C['muted'], 'ytick.color': C['muted'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': C['grid'],
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'], 'savefig.facecolor': C['surface']})
J0, J1, J2, J3 = 2525, 3468, 3421, 2331

def save(fig, name):
    for ext in ('svg', 'png'): fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig); print('wrote', OUT / f'{name}.svg')

def bars(ax, groups, series, colors, ylim=(0, 105), fmt='{:.0f}%', width=0.26, label_fmt=None):
    """Grouped thin bars with rounded tops (direct labels on every bar)."""
    x = np.arange(len(groups)); n = len(series)
    for k, (name, vals, labels) in enumerate(series):
        xs = x + (k - (n - 1) / 2) * width
        ax.bar(xs, vals, width - 0.03, color=colors[k], label=name, zorder=3, linewidth=0)
        for xx, v, lab in zip(xs, vals, labels): ax.text(xx, v + 1.5, lab, ha='center', va='bottom', fontsize=7.2, color=C['ink'], rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.set_ylim(*ylim); ax.grid(axis='x', visible=False)

def fig1():
    T = json.load(open(EXP7 / 'analysis/final_tables.json')); K = json.load(open(EXP7 / 'analysis/controls.json'))
    tool = ['Practice (64)', 'Selection (32)', 'Renamed A (32)', 'Renamed B (32)', 'Final-split proxy (32)']; models = ['Base', 'g23v3', 'g23v4']
    def cls(model, c): return sum(T[s][model]['summary']['per_class'][c]['correct'] for s in tool), sum(T[s][model]['summary']['per_class'][c]['n'] for s in tool)
    def tot(model): return sum(T[s][model]['summary']['lenient'] for s in tool), sum(T[s][model]['summary']['n'] for s in tool)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={'width_ratios': [5, 2.4, 3.1], 'wspace': 0.22})
    groups = ['All 192\ntool items', 'Valid calls\n(72)', 'NO_TOOL\n(60)', 'Explicit\nmissing (30)', 'Ambiguous\n"either X or Y" (30)']
    series = []
    for m in models:
        vals = [tot(m)] + [cls(m, c) for c in ('valid_call', 'no_tool', 'need_clarification', 'ambiguous')]
        series.append((m if m != 'Base' else 'Base Qwen3.5-0.8B', [100 * a / b for a, b in vals], [f'{a}' for a, b in vals]))
    bars(axes[0], groups, series, [C['base'], C['g23v3'], C['g23v4']]); axes[0].set_ylabel('correct (lenient scoring), %'); axes[0].set_title('A  The 192 benchmark tool items (5 cohorts, same prompts, greedy)', loc='left', fontsize=10)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.22), ncol=3, fontsize=8.5)
    st = ['Stress: ambiguous (32)', 'Stress: resolved two-value calls (32)']
    series = [(m, [100 * T[s][m]['summary']['lenient'] / 32 for s in st], [f"{T[s][m]['summary']['lenient']}" for s in st]) for m in models]
    bars(axes[1], ['Unseen ambiguous\nphrasings (32)', 'Resolved two-value\ncalls (32)'], series, [C['base'], C['g23v3'], C['g23v4']]); axes[1].set_title('B  Ambiguity Stress Suite (held out)', loc='left', fontsize=10)
    # controls on the 64-item control set: ambiguous->NE, resolved calls, practice calls, NO_TOOL, explicit-missing
    labs = ['Intended (g23v4)', 'Reversed (a3=-0.1)', 'Random norm-matched (seed 17)']
    def cell(lab, i): s = K[lab]['cells'][i]; a, b = s.split()[0].split('/'); return int(a), int(b)
    groups = ['Ambig.\n-> NE\n(32)', 'Resolved\ncalls\n(16)', 'Practice\ncalls\n(8)', 'NO_\nTOOL\n(4)', 'Explicit\nmissing\n(4)']
    series = [(l.split(' (')[0], [100 * cell(l, i)[0] / cell(l, i)[1] for i in range(5)], [f'{cell(l, i)[0]}' for i in range(5)]) for l in labs]
    bars(axes[2], groups, series, [C['g23v4'], '#c3c2b7', C['ctrl']]); axes[2].set_title('C  Controls on the same 6 tensors (64-item control set)', loc='left', fontsize=10)
    for b in axes[2].patches[len(groups):]: b.set_hatch('////'); b.set_edgecolor(C['surface']); b.set_linewidth(0)
    axes[2].legend(loc='upper left', bbox_to_anchor=(-0.05, -0.26), ncol=3, fontsize=7.6, handlelength=1.6, columnspacing=1.0)
    for ax in axes[1:]: ax.set_ylabel('')
    fig.text(0.0, -0.3, 'Base = unedited Qwen/Qwen3.5-0.8B; g23v3 = 2 re-purposed layer-23 neurons (6,142 BF16 entries); g23v4 = g23v3 + layer-22 J0 -> head 6 -> layer-23 J3 (12,281 entries, 0.0016% of weights). '
             'Non-tool cohorts (not shown): short regression 32/32 token-identical, 128-token prose 32/32 intact for both edits. Panel B: all 32 resolved-call streams are token-identical across the three models. Reversed flips J3\'s output sign; Random replaces every new vector by a norm-matched random one.', fontsize=7.4, color=C['muted'], wrap=True)
    save(fig, 'fig1_progression_overview')

def lofo_dprime(X, pos, fam):
    """Leave-one-family-out diff-of-means probe -> (d', accuracy)."""
    X = np.asarray(X, np.float64); proj = np.zeros(len(X)); pred = np.zeros(len(X), bool)
    for f in np.unique(fam):
        tr = fam != f; te = fam == f
        if pos[tr].sum() == 0 or (~pos[tr]).sum() == 0: continue
        d = X[tr & pos].mean(0) - X[tr & ~pos].mean(0); d /= np.linalg.norm(d); p = X @ d; thr = (p[tr & pos].mean() + p[tr & ~pos].mean()) / 2
        proj[te] = p[te]; pred[te] = p[te] > thr
    a, b = proj[pos], proj[~pos]; s = np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / s), float((pred == pos).mean())

def fig2():
    js = json.load(open(Q6 / 'runs/p03_practice_dec.json')); z = np.load(Q6 / 'runs/p03_practice_dec.npz'); H = z['h_dec'].astype(np.float32)  # (64, 25, 1024)
    cases = {json.loads(l)['id']: json.loads(l) for l in (EXP6 / 'data/practice.jsonl').read_text().splitlines() if l.strip()}
    ids = [r['case_id'] for r in js['rows']]; act = np.array([cases[i]['spec']['action'] for i in ids]); amb = np.array([bool(cases[i]['spec'].get('ambiguous')) for i in ids]); fam = np.array([cases[i]['family'] for i in ids])
    layers = np.arange(25); curves = {}
    for name, posmask, negmask in (('NO_TOOL vs calls (20 vs 24)', act == 'answer', act == 'call'), ('explicit-missing vs calls (10 vs 24)', (act == 'clarify') & ~amb, act == 'call'), ('ambiguous "either X or Y" vs calls (10 vs 24)', amb, act == 'call')):
        keep = posmask | negmask; curves[name] = [np.nan] + [lofo_dprime(H[keep, l], posmask[keep], fam[keep])[0] for l in layers[1:]]  # layer 0 = identical embeddings of the decision token
    D2 = json.load(open(EXP7 / 'analysis/diag2_d01b_triples_D1_d02_triples_D2_d03_triples_D3.json'))
    dec = D2['decision']; dl = sorted(int(k) for k in dec if k.isdigit()); roles = D2['roles']
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True, gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.12})
    ax = axes[0]
    for (name, ys), col in zip(curves.items(), [C['base'], C['g23v3'], C['g23v4']]): ax.plot(layers, ys, color=col, lw=2, marker='o', ms=3.5, label=name)
    ax.plot(dl, [dec[str(l)]['raw']['dprime'] for l in dl], color=C['g23v4'], lw=1.6, ls='--', marker='s', ms=3.5, label='ambiguous vs call+resolved, 192 triples, raw (84 vs 108)')
    ax.plot(dl, [dec[str(l)]['lda05']['dprime'] for l in dl], color=C['g23v4'], lw=1.6, ls=':', marker='^', ms=3.5, label='ambiguous vs call+resolved, 192 triples, shrinkage-LDA')
    ax.axhline(3, color=C['muted'], lw=0.8, ls='-.'); ax.text(0.3, 3.3, "d' = 3", color=C['muted'], fontsize=8)
    ax.set_xlabel('residual-stream layer boundary (0 = embeddings, 24 = final norm input)'); ax.set_ylabel("leave-one-family-out d'"); ax.set_title('A  At the decision token (\\n\\n after </think>)', loc='left', fontsize=10)
    ax.legend(fontsize=7.6, loc='upper left'); ax.set_xticks(range(0, 25, 2))
    ax = axes[1]; rl = sorted({int(k.split(':')[1]) for k in roles})
    for role, lab, col, mk in (('end', "clause-end '.' (84 vs 108)", C['g23v4'], 'o'), ('conn', 'connector "or" (84 vs 54)', C['g23v3'], 's'), ('tail', 'indecision tail (66 vs 17)', C['base'], '^'), ('v2', 'second value (84 vs 54)', '#6250d6', 'D')):
        ks = [l for l in rl if f'{role}:{l}' in roles]
        ax.plot(ks, [roles[f'{role}:{l}']['lda05']['dprime'] for l in ks], color=col, lw=2, marker=mk, ms=3.5, label=f'{lab}, shrinkage-LDA')
        ax.plot(ks, [roles[f'{role}:{l}']['raw']['dprime'] for l in ks], color=col, lw=1.2, ls='--', marker=mk, ms=2.5, alpha=0.7, label=f'{lab}, raw')
    ax.axhline(3, color=C['muted'], lw=0.8, ls='-.'); ax.set_xlabel('layer boundary'); ax.set_title("B  At the user's clause tokens (same 192 items)", loc='left', fontsize=10)
    ax.legend(fontsize=7.0, loc='upper left', bbox_to_anchor=(1.01, 1.0)); ax.set_xticks(range(10, 24, 2))
    ax.text(10.2, 0.6, 'natural attention from the decision query to the clause-end token: 0.000-0.003 at layer 23,\n0.015-0.062 at layer 15 (heads 2 and 6 of layer 23 attend 97% / 87% to themselves)', fontsize=7.4, color=C['muted'])
    fig.text(0.0, -0.08, "Panel A: qwen-6 practice capture p03 (64 items, original model; LOFO = leave-one-family-out over the 18 families) and the qwen-7 192-item triples at the decision. "
             "Panel B: the same 192 triples probed at the clause tokens (qwen-7 analysis/diag2). The 'unresolved alternatives' feature is linearly clean where the two values are written and absent where the decision is made.", fontsize=7.4, color=C['muted'], wrap=True)
    save(fig, 'fig2_positional_bottleneck_dprime')

def fig3():
    sys.path.insert(0, str(HERE)); import replay_tools as RT
    R = RT.patch_rows('g23v4'); idx = json.load(open(ROOT / 'artifacts/traces/index.json'))['items']
    # J1 activation at the decision on the ORIGINAL model's x23 (what g23v3 reads), per cohort
    j1 = {}
    for it in idx:
        z = np.load(ROOT / 'artifacts/traces' / f"{it['id']}.npz"); x = z['x23_gen'][0].astype(np.float64)
        j1.setdefault(it['cohort'], []).append(float(RT.silu(x @ R['gateJ1']) * (x @ R['upJ1'])))
    # J0 / head 6 / J3 from the real g23v4 model (c01 on D1, c02 on the stress ambiguous items)
    c01 = RT.load_capture('c01_calib_D1'); c02 = RT.load_capture('c02_stress_calib')
    keys = ['ambiguous (D1)', 'call / resolved (D1)', 'stress ambiguous hit', 'stress ambiguous miss']; j0 = {k: [] for k in keys}; a6 = {k: [] for k in keys}; j3 = {k: [] for k in keys}
    for c, tag in ((c01, None), (c02, 'stress')):
        for i in c.ids:
            ro = c.roles(i); v = c.meta[i]['variant']; e = c.pos_index(i, 'end'); d = c.pos_index(i, 'dec')
            act0 = float(c.z[f'mlp_{i}'][e, 0][J0]); al = float(c.rows_attn(i)[5, 0][6, ro['end']]); act3 = float(c.z[f'mlp_{i}'][d, 1][J3])
            if tag is None: k = 'ambiguous (D1)' if v == 'ambiguous' else 'call / resolved (D1)'
            else: k = 'stress ambiguous hit' if c.rows[i]['decision']['argmax'] == 3811 else 'stress ambiguous miss'
            j0[k].append(act0); a6[k].append(al); j3[k].append(act3)
    fig = plt.figure(figsize=(14.5, 8.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=0.22, wspace=0.36)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')
    def box(x, y, w, h, title, body, col, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc='white', ec=col, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=fs + 0.8, fontweight='bold', color=col, zorder=3)
        ax.text(x + w / 2, y + (h - 3.2) / 2, body, ha='center', va='center', fontsize=fs - 0.9, color=C['ink'], zorder=3, linespacing=1.3)
    def arrow(x0, y0, x1, y1, col, text='', ty=1.2, ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, color=col, lw=1.6, ls=ls, zorder=4))
        if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha='center', fontsize=7.4, color=col)
    ax.text(1, 33, 'Act I  (g23v3, layer 23, decision position)', fontsize=10.5, fontweight='bold', color=C['g23v3'])
    ax.text(1, 16.2, 'Act II  (g23v4 adds a route from the clause end to the decision)', fontsize=10.5, fontweight='bold', color=C['g23v4'])
    box(1, 19.5, 20, 11, 'x23 at the decision', 'layer-23 MLP input\nprojections: s (abstain), r (route),\nc (tool context), c2 (generated position)', C['muted'])
    box(26, 19.5, 22, 11, 'J1 = 3468  (abstain)', 'gate = 2(s - 0.3c - 0.5c2)\nup = 2s,  down = u_AB\n-> removes <tool_call>, adds NO + NE', C['g23v3'])
    box(52, 19.5, 22, 11, 'J2 = 3421  (route)', 'gate = 2(s - 0.3c - 0.5c2)\nup = r,  down = u_NE - u_NO\n-> NEED_CLARIFICATION over NO_TOOL', C['g23v3'])
    box(78, 19.5, 21, 11, 'final norm + unembedding', 'NO_TOOL / NEED_CLARIFICATION\nfirst token; calls and 128-token\nprose untouched (gates shut)', C['muted'])
    arrow(21.4, 25, 25.6, 25, C['g23v3']); arrow(48.4, 25, 51.6, 25, C['g23v3']); arrow(74.4, 25, 77.6, 25, C['g23v3'])
    box(1, 2.5, 22, 12, 'J0 = 2525  (layer 22 MLP)', 'reads x22 at every prompt token\ngate = up = 4(unit(0.5g + u) - 11.7 r1)\n"clause end AND unresolved"\nact 58..410 on the clause-end "."\n<= 0.016 on 141k other positions', C['g23v4'])
    box(28, 2.5, 22, 12, 'beacon p6 in the residual', 'down[:,J0] = p6 = pinv(Wk gamma) k_target\nunrotated RoPE dims 64:256, orthogonal to\nheads 4/5/7 of the same GQA group\nnorm 58..410 vs residual ~9.6', C['g23v4'])
    box(55, 2.5, 20, 12, 'layer-23 attention head 6', 'decision query attends to the\nmarked clause end: alpha 0.0001 -> 0.99\ncopies its value (unchanged weights)\nzero spill on heads 4/5/7', C['g23v4'])
    box(80, 2.5, 19, 12, 'J3 = 2331  (layer 23 MLP)', 'gate = 2(t - 0.3c - 0.5c2 - 2 r3)\nup = 2(t - 2 r3),  down = 0.1(u_AB + u_route)\nact 59..109 ambiguous, <= 0.0003 elsewhere', C['g23v4'])
    arrow(23.4, 8.5, 27.6, 8.5, C['g23v4']); arrow(50.4, 8.5, 54.6, 8.5, C['g23v4']); arrow(75.4, 8.5, 79.6, 8.5, C['g23v4'])
    arrow(89.5, 14.9, 89.5, 19.1, C['g23v4'])
    ax.text(50, 0.0, 'Edited: layers.22.mlp.{gate,up,down}_proj rows/column J0; layers.23.mlp.{gate,up,down}_proj rows/columns J1, J2, J3. 6 tensors, 12,281 BF16 entries. No attention weight changed.', ha='center', fontsize=7.6, color=C['muted'])
    # activation panels
    def strip(ax, data, title, ylabel, log=False, order=None):
        order = order or list(data); rng = np.random.default_rng(0)
        for k, name in enumerate(order):
            v = np.array(data[name]); col = [C['g23v3'], C['muted'], C['g23v4'], C['ctrl']][k % 4] if 'D1' not in name else (C['g23v4'] if 'ambiguous' in name else C['muted'])
            if 'stress' in name: col = C['g23v4'] if 'hit' in name else '#e34948'
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v if not log else np.maximum(v, 1e-4), s=12, color=col, alpha=0.75, linewidths=0, zorder=3)
            ax.text(k, (ax.get_ylim()[1] if not log else 1), '', fontsize=7)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(['ambiguous\n(D1)', 'call / res.\n(D1)', 'stress amb.\nhit', 'stress amb.\nmiss'], fontsize=7.2)
        if log: ax.set_yscale('log')
        ax.set_title(title, loc='left', fontsize=8.6); ax.set_ylabel(ylabel); ax.grid(axis='x', visible=False)
    order1 = ['no_tool', 'need_clarification', 'ambiguous', 'valid_call', 'stress_ambiguous', 'stress_resolved', 'prose']
    j1o = {k: j1[k] for k in order1 if k in j1}
    ax = fig.add_subplot(gs[1, 0]); rng = np.random.default_rng(0)
    for k, name in enumerate(j1o):
        v = np.array(j1o[name]); col = C['g23v3'] if name in ('no_tool', 'need_clarification') else C['muted']
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=col, alpha=0.75, linewidths=0, zorder=3)
    ax.set_xticks(range(len(j1o))); ax.set_xticklabels(['NO_\nTOOL', 'expl.\nmiss.', 'amb.', 'valid\ncall', 'stress\namb.', 'stress\nres.', 'prose'], fontsize=7.0)
    ax.set_title('J1 act at the decision (original x23, 143 items)', loc='left', fontsize=8.6); ax.set_ylabel('silu(gate) x up'); ax.grid(axis='x', visible=False)
    ax.axhline(0, color=C['muted'], lw=0.6)
    ax = fig.add_subplot(gs[1, 1]); strip(ax, j0, 'J0 act at the clause-end "." (real g23v4)', 'activation (log)', log=True)
    ax = fig.add_subplot(gs[1, 2]); strip(ax, a6, 'head 6: decision -> clause end (real g23v4)', 'alpha (log)', log=True)
    ax = fig.add_subplot(gs[1, 3]); strip(ax, j3, 'J3 act at the decision (real g23v4)', 'activation (log)', log=True)
    fig.text(0.0, 0.0, 'Bottom row: dots are items. J1 from a fresh original-model capture (scripts/capture_traces.py) at the decision position of 143 items across every cohort; J0, head 6 and J3 from the real edited model\'s teacher-forced '
             'passes c01 (64 D1 items: 28 ambiguous, 18 calls, 18 resolved) and c02 (32 stress-suite ambiguous items: 27 hits, 5 misses whose stage-1 gate stayed shut). Values below 1e-4 are drawn at 1e-4.', fontsize=7.4, color=C['muted'], wrap=True)
    save(fig, 'fig3_four_neuron_circuit_diagram')

def fig4():
    K = json.load(open(EXP7 / 'analysis/controls.json')); A = json.load(open(EXP7 / 'analysis/oof_audit.json'))
    order = ['Zero (g23v3)', 'Half (a3=0.05, b=0.5)', 'Intended (g23v4)', 'Double (a3=0.2, b=2)', 'Reversed (a3=-0.1)', 'Random norm-matched (seed 17)']
    def cell(lab, i): s = K[lab]['cells'][i]; a, b = s.split()[0].split('/'); return 100 * int(a) / int(b)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), gridspec_kw={'width_ratios': [2.6, 2.2, 2.2], 'wspace': 0.28})
    ax = axes[0]; x = np.arange(len(order))
    for i, (lab, col, mk) in enumerate((('ambiguous -> NEED_CLARIFICATION (32)', C['g23v4'], 'o'), ('resolved two-value calls correct (16)', C['base'], 's'), ('practice calls correct (8)', C['g23v3'], '^'), ('NO_TOOL (4)', '#6250d6', 'D'), ('explicit-missing (4)', '#e34948', 'v'))):
        ax.plot(x, [cell(o, i) for o in order], color=col, lw=1.8, marker=mk, ms=5, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(['Zero\n(= g23v3)', 'Half\na3 .05, b .5', 'Intended\na3 .1, b 1', 'Double\na3 .2, b 2', 'Reversed\na3 -.1', 'Random\nnorm-matched'], fontsize=7.6); ax.set_ylabel('% correct'); ax.set_ylim(-3, 105)
    ax.set_title('A  Dose-response and negative controls\n(64-item control set, real generation)', loc='left', fontsize=9.4); ax.legend(fontsize=7.2, loc='lower left'); ax.grid(axis='x', visible=False)
    rows = A['results']['simulator_vs_real_D1']['rows']
    ax = axes[1]
    for tok, col, mk in (('NE', C['g23v4'], 'o'), ('NO', C['g23v3'], 's'), ('call', C['base'], '^')):
        xs = [r['sim'][tok] for r in rows]; ys = [r['real'][tok] for r in rows]; ax.scatter(xs, ys, s=14, color=col, marker=mk, alpha=0.8, linewidths=0, label={'NE': 'NE (-> NEED_CLARIFICATION)', 'NO': 'NO (-> NO_TOOL)', 'call': '<tool_call>'}[tok], zorder=3)
    lo = min(min(r['sim'][t] for r in rows for t in ('NE', 'NO', 'call')), min(r['real'][t] for r in rows for t in ('NE', 'NO', 'call'))) - 2; hi = max(max(r['sim'][t] for r in rows for t in ('NE', 'NO', 'call')), max(r['real'][t] for r in rows for t in ('NE', 'NO', 'call'))) + 2
    ax.plot([lo, hi], [lo, hi], color=C['muted'], lw=0.8, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel('CPU replay prediction (logit)'); ax.set_ylabel('real g23v4 model (teacher-forced, pass c01)')
    s = A['results']['simulator_vs_real_D1']; ax.set_title(f"B  CPU replay vs real model, decision logits\n(argmax {s['argmax_agreement']}, |err| median {s['abs_logit_error_NE_NO_call_median']:.2f}, max {s['max']:.2f})", loc='left', fontsize=9.4); ax.legend(fontsize=7.4, loc='lower right')
    ax = axes[2]
    for key, lab, col in (('alpha6', 'head-6 attention to the clause end', C['g23v4']), ('j0', 'J0 activation at the clause end / 400', C['g23v3']), ('j3', 'J3 activation at the decision / 100', C['base'])):
        sc = {'alpha6': 1.0, 'j0': 400.0, 'j3': 100.0}[key]; xs = [r[f'{key}_sim'] / sc for r in rows]; ys = [r[f'{key}_real'] / sc for r in rows]
        ax.scatter(xs, ys, s=14, color=col, alpha=0.8, linewidths=0, label=lab, zorder=3)
    ax.plot([0, 1.05], [0, 1.05], color=C['muted'], lw=0.8, ls='--'); ax.set_xlim(-0.03, 1.08); ax.set_ylim(-0.03, 1.08); ax.set_xlabel('CPU replay prediction'); ax.set_ylabel('real g23v4 model'); ax.set_title('C  Mechanism quantities, replay vs real\n(64 D1 items)', loc='left', fontsize=9.4); ax.legend(fontsize=7.4, loc='lower right')
    fig.text(0.0, -0.1, 'Panel A: analysis/controls.md of the qwen-7 campaign (passes k01-k04 on data/control.ids). Panels B-C: the exact layer-23 CPU replay (scripts/oof_audit.py, shipped g23v4 rows) against the real edited model\'s teacher-forced capture c01 on the 64 D1 items; '
             'non-ambiguous items cluster near zero in panel C by construction (gate shut, attention 0.0001).', fontsize=7.4, color=C['muted'], wrap=True)
    save(fig, 'fig4_dose_and_controls')

if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4']
    for w in which: {'1': fig1, '2': fig2, '3': fig3, '4': fig4}[w]()
