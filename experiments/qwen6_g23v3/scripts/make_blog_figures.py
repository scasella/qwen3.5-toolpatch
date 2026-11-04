#!/usr/bin/env python3
"""Publication figures for the blog post, computed directly from the recorded campaign data.

Reads (never writes): runs/*.json, runs/*.npz, data/practice.jsonl, edits/g23v3.vectors.npz, docs/figures/fig4_trace.json.
Writes: docs/figures/fig{1,2,3,4}_*.{svg,png} and docs/figures/figure_data.json (the numbers behind every panel).

  /Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python scripts/make_blog_figures.py

Dependencies: numpy, matplotlib. No campaign module is imported, so the script cannot touch the ledger or the edits."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / 'runs'; OUT = ROOT / 'docs/figures'; OUT.mkdir(parents=True, exist_ok=True)
C = dict(orig='#9aa0a6', v3='#0f766e', v1='#d97706', v2='#7c3aed', rand='#b45309', rev='#6b7280', call='#4b5563', amb='#0f766e',
         nontool='#ea580c', gen='#7c3aed', grid='#e5e7eb', text='#111827')
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': '#374151', 'axes.labelcolor': C['text'],
                     'xtick.color': C['text'], 'ytick.color': C['text'], 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
DATA = {}

def run(label): return json.loads((RUNS / f'{label}.json').read_text())
def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight'); fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight'); plt.close(fig); print('wrote', name)
def dprime(pos, neg): return float((pos.mean() - neg.mean()) / (np.sqrt((pos.var() + neg.var()) / 2) + 1e-9))
def unit(v): return v / np.linalg.norm(v)

# ----------------------------------------------------------------------------------------------------------------------- figure 1
def fig1():
    def summary(label, which): return run(label)[which]
    def correct(label, which, cls=None):
        s = summary(label, which); return s['per_class'][cls]['correct'] if cls else s['correct']
    sets = [  # name, n, (original run, key), (edited run, key)
        ('Practice\n(64)', 'p01_baseline', 'original', 'p38_v3_gen_practice', 'edited'),
        ('Selection\n(32)', 'p40_v3_sel_both', 'original', 'p40_v3_sel_both', 'edited'),
        ('Renamed A\n(32)', 'p42_v3_renamedA_both', 'original', 'p42_v3_renamedA_both', 'edited'),
        ('Renamed B\n(32)', 'p43_v3_renamedB_both', 'original', 'p43_v3_renamedB_both', 'edited'),
        ('Final-split\nfamilies (32)', 'p41_v3_final_both', 'original', 'p41_v3_final_both', 'edited')]
    rows = []
    for name, ro, ko, re_, ke in sets:
        rows.append((name, summary(ro, ko)['n'], correct(ro, ko), correct(re_, ke)))
    calls_o = sum(correct(ro, ko, 'call') for _, ro, ko, _, _ in sets); calls_e = sum(correct(re_, ke, 'call') for _, _, _, re_, ke in sets)
    calls_n = sum(summary(ro, ko)['per_class']['call']['n'] for _, ro, ko, _, _ in sets)
    rows.append(('Valid calls\npreserved (72)', calls_n, calls_o, calls_e))
    prose_o = correct('p36_v3_prose2_both', 'original') + correct('p37_v3_prose_both', 'original'); prose_e = correct('p36_v3_prose2_both', 'edited') + correct('p37_v3_prose_both', 'edited')
    rows.append(('Prose QA,\nno tools (32)', 32, prose_o, prose_e))
    DATA['fig1_main'] = [dict(set=n.replace('\n', ' '), n=k, original=o, g23v3=e) for n, k, o, e in rows]
    # per-class totals over the 192 tool-context items
    cls = {}
    for c in ('call', 'answer', 'clarify'):
        cls[c] = dict(n=sum(summary(ro, ko)['per_class'][c]['n'] for _, ro, ko, _, _ in sets), original=sum(correct(ro, ko, c) for _, ro, ko, _, _ in sets),
                      g23v3=sum(correct(re_, ke, c) for _, _, _, re_, ke in sets))
    DATA['fig1_classes'] = cls
    controls = [('g23v3\n(final)', 'p38_v3_gen_practice', C['v3']), ('random\nseed 7', 'p12_gen_rand7', C['rand']),
                ('random\nseed 11', 'p28_v2_rand11', C['rand']), ('random\nseed 13', 'p45_v3_rand13', C['rand']),
                ('reversed\n(v1)', 'p11_gen_rev', C['rev']), ('reversed\n(v2)', 'p27_v2_rev', C['rev'])]
    base = correct('p01_baseline', 'original')
    ctl = [(n, base, correct(l, 'edited'), col) for n, l, col in controls]
    DATA['fig1_controls'] = [dict(control=n.replace('\n', ' '), original=o, edited=e) for n, o, e, _ in ctl]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.22))
    x = np.arange(len(rows)); w = 0.38
    po = [100 * o / n for _, n, o, _ in rows]; pe = [100 * e / n for _, n, _, e in rows]
    b1 = ax.bar(x - w / 2, po, w, color=C['orig'], label='Unedited Qwen3.5-0.8B'); b2 = ax.bar(x + w / 2, pe, w, color=C['v3'], label='g23v3 (two-neuron edit)')
    for i, (name, n, o, e) in enumerate(rows):
        ax.text(x[i] - w / 2, po[i] + 1.5, f'{o}/{n}', ha='center', va='bottom', fontsize=8, color='#4b5563', rotation=90 if po[i] > 90 else 0)
        ax.text(x[i] + w / 2, pe[i] + 1.5, f'{e}/{n}', ha='center', va='bottom', fontsize=8, color=C['v3'], fontweight='bold', rotation=90 if pe[i] > 90 else 0)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5); ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel('correct (lenient scoring), %')
    ax.yaxis.grid(True, color=C['grid']); ax.set_axisbelow(True); ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_title('A. Paired evaluation, identical prompts, greedy decoding', loc='left', fontsize=11, fontweight='bold')
    ax.axvline(4.5, color='#d1d5db', lw=1, ls='--'); ax.text(4.55, 118, 'must not change →', fontsize=8, color='#6b7280')
    # right: controls on the same two neurons (practice, 64)
    xc = np.arange(len(ctl)); ax2.axhline(100 * base / 64, color=C['orig'], lw=1.5, ls='--'); ax2.text(len(ctl) - 0.45, 100 * base / 64 - 2.5, f'unedited baseline {base}/64', ha='right', va='top', fontsize=8.5, color='#4b5563')
    bars = ax2.bar(xc, [100 * e / 64 for _, _, e, _ in ctl], 0.6, color=[col for *_, col in ctl])
    for i, (n, o, e, col) in enumerate(ctl): ax2.text(xc[i], 100 * e / 64 + 1.5, f'{e}/64', ha='center', va='bottom', fontsize=8.5, color=col, fontweight='bold')
    ax2.set_xticks(xc); ax2.set_xticklabels([c[0] for c in ctl], fontsize=8.5); ax2.set_ylim(0, 122); ax2.set_yticks(range(0, 101, 20)); ax2.yaxis.grid(True, color=C['grid']); ax2.set_axisbelow(True)
    ax2.set_title('B. Same two neurons, other directions (practice, 64)', loc='left', fontsize=11, fontweight='bold'); ax2.set_ylabel('correct (lenient), %')
    ax2.text(2, 112, 'random = norm-matched Gaussian gate/up/down directions (v1, v2, v3 norms)', ha='center', fontsize=8, color='#6b7280')
    save(fig, 'fig1_headline_results')

# ----------------------------------------------------------------------------------------------------------------------- figure 2
def fig2():
    t = np.load(RUNS / 'p02_teacher_dec.npz'); p = np.load(RUNS / 'p03_practice_dec.npz')
    tact = np.array(['call' if '-call' in str(c) else 'answer' if '-answer' in str(c) else 'clarify' for c in t['case_ids']])
    spec = {json.loads(l)['id']: json.loads(l)['spec'] for l in (ROOT / 'data/practice.jsonl').read_text().splitlines() if l}
    pact = np.array([spec[str(c)]['action'] for c in p['case_ids']]); pamb = np.array([bool(spec[str(c)]['ambiguous']) for c in p['case_ids']])
    layers = list(range(1, 25)); A, Bc, Camb, Amlp = [], [], [], []
    for l in layers:
        H, Hp = t['h_dec'][:, l], p['h_dec'][:, l]
        d = unit(H[tact == 'answer'].mean(0) - H[tact == 'call'].mean(0)); f = Hp @ d; A.append(dprime(f[pact == 'answer'], f[pact == 'call']))
        d = unit(H[tact == 'clarify'].mean(0) - H[tact == 'answer'].mean(0)); f = Hp @ d; Bc.append(dprime(f[(pact == 'clarify') & ~pamb], f[pact == 'answer']))
        # ambiguous vs complete call: no teacher items exist, so leave-one-out difference of means within practice
        idx = np.where((pact == 'call') | pamb)[0]; y = pamb[idx]; X = Hp[idx]; proj = np.zeros(len(idx))
        for i in range(len(idx)):
            m = np.ones(len(idx), bool); m[i] = False; dd = unit(X[m][y[m]].mean(0) - X[m][~y[m]].mean(0)); proj[i] = X[i] @ dd
        Camb.append(dprime(proj[y], proj[~y]))
        if l <= 23:  # MLP activation space of block l (teacher-selected top-64 neurons by |d'|, diff-of-means direction on them)
            Am, Amp = t['mlp_dec'][:, l], p['mlp_dec'][:, l]; pos, neg = Am[tact == 'answer'], Am[tact == 'call']
            dn = (pos.mean(0) - neg.mean(0)) / (np.sqrt((pos.var(0) + neg.var(0)) / 2) + 1e-9); cols = np.argsort(np.abs(dn))[::-1][:64]
            v = unit(pos[:, cols].mean(0) - neg[:, cols].mean(0)); f = Amp[:, cols] @ v; Amlp.append(dprime(f[pact == 'answer'], f[pact == 'call']))
    DATA['fig2'] = dict(layers=layers, explain_answer_vs_call=A, explain_mlp_top64=Amlp, route_clarify_vs_answer=Bc, ambiguous_vs_call_loo=Camb)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(22.5, 23.5, color='#fef3c7', zorder=0); ax.text(23, 14.6, 'edit\nreads\nhere', ha='center', va='top', fontsize=8, color='#92400e')
    ax.plot(layers, A, '-o', color=C['v3'], lw=2, ms=4, label="A  s: explanation request vs. complete call (residual stream)")
    ax.plot(layers[:23], Amlp, ':', color=C['v3'], lw=1.5, label="A' same contrast in MLP activation space (top-64 neurons)")
    ax.plot(layers, Bc, '-s', color=C['v2'], lw=2, ms=4, label="B  r: explicit-missing (clarify) vs. explanation")
    ax.plot(layers, Camb, '-^', color=C['nontool'], lw=2, ms=4, label="C  ambiguous 'either X or Y' vs. complete call (leave-one-out)")
    ax.axhline(3, color='#9ca3af', lw=1, ls='--'); ax.text(24.4, 3.2, "d' = 3", fontsize=8, color='#6b7280', ha='right')
    ax.set_xlabel('residual stream after layer L  (L = 24 is the input to the final norm)'); ax.set_ylabel("separability d' on practice items")
    ax.set_xticks(range(1, 25)); ax.set_xlim(0.5, 24.5); ax.set_ylim(0, 15); ax.yaxis.grid(True, color=C['grid']); ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, fontsize=8.8)
    ax.set_title("Where the backbone represents each decision: teacher-derived directions, scored per item on practice", loc='left', fontsize=11, fontweight='bold')
    save(fig, 'fig2_layer_separability')

# ----------------------------------------------------------------------------------------------------------------------- figure 3
def fig3():
    V = np.load(ROOT / 'edits/g23v3.vectors.npz'); s, c, c2 = V['s'], V['c'], V['c2']; lam, lam2, k = 0.3, 0.5, 2.0
    spec = {json.loads(l)['id']: json.loads(l)['spec'] for l in (ROOT / 'data/practice.jsonl').read_text().splitlines() if l}
    def rows(lab):
        js = run(lab); arr = np.load(RUNS / f'{lab}.npz'); return [(r, arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32)) for r in js['rows']]
    G = {g: [] for g in ('abstain', 'ambiguous', 'call', 'nontool_dec', 'gen_literal', 'gen_incall', 'gen_prose_tool', 'gen_prose_nontool')}
    for lab in ('p04c_practice_allpos', 'p05_teacher_allpos'):
        for r, x in rows(lab):
            d, a = r['decision_position'], r['action']
            if a == 'clarify' and spec.get(r['case_id'], {}).get('ambiguous'): G['ambiguous'].append(x[d])
            elif a == 'call': G['call'].append(x[d])
            else: G['abstain'].append(x[d])
            (G['gen_incall'] if a == 'call' else G['gen_literal']).extend(x[d + 1:])
    for r, x in rows('p16_regression_capture'): G['nontool_dec'].append(x[r['decision_position']])
    for r, x in rows('p17_prose_capture'): G['gen_prose_tool'].extend(x[r['decision_position'] + 1:])
    for r, x in rows('p35_prose_capture_nontool'): G['gen_prose_nontool'].extend(x[r['decision_position'] + 1:])
    P = {}
    for g, xs in G.items():
        X = np.stack(xs); fs, fc, fc2 = X @ s, X @ c, X @ c2; pre = k * (fs - lam * fc - lam2 * fc2); act = pre / (1 + np.exp(-pre)) * (k * fs)
        P[g] = dict(fs=fs, close=lam * fc + lam2 * fc2, act=act, fc=fc, fc2=fc2)
        DATA.setdefault('fig3', {})[g] = dict(n=int(len(X)), s_x=[float(fs.min()), float(fs.max())], c_x=[float(fc.min()), float(fc.max())],
                                             c2_x=[float(fc2.min()), float(fc2.max())], c2_x_p1=float(np.percentile(fc2, 1)), act_absmax=float(np.abs(act).max()),
                                             n_act_above_1=int((np.abs(act) > 1).sum()))
    fig = plt.figure(figsize=(14, 6.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.18)
    # ---- schematic
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    def box(x, y, w, h, text, fc='#f3f4f6', ec='#9ca3af', fs=8.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec=ec, lw=1.2)); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=C['text'], fontweight='bold' if bold else 'normal')
    def arrow(x0, y0, x1, y1, col='#374151'): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11, color=col, lw=1.2))
    box(2.6, 8.7, 4.8, 0.9, 'x = layer-23 MLP input (post-attention RMSNorm, d = 1024)', fc='#eef2ff', ec='#6366f1', bold=True)
    box(0.3, 6.2, 3.2, 1.5, 'gate (both neurons)\nk·(s − 0.3c − 0.5c₂)·x\n"tool-context decision\nwith s·x > 0 ?"', fc='#fef3c7', ec='#d97706')
    box(3.9, 6.2, 2.7, 1.5, 'up, J1 = 3468\nk·(s·x)\n"how strongly\nan abstention?"', fc='#d1fae5', ec=C['v3'])
    box(7.0, 6.2, 2.7, 1.5, 'up, J2 = 3421\nk_r·(r·x)\n"clarify (+) or\nexplain (−)?"', fc='#ede9fe', ec=C['v2'])
    for xx in (1.9, 5.25, 8.35): arrow(5.0, 8.7, xx, 7.75)
    box(1.0, 4.4, 1.8, 0.75, 'SiLU', fc='#fff7ed', ec='#d97706'); arrow(1.9, 6.2, 1.9, 5.2)
    ax.add_patch(plt.Circle((5.25, 4.78), 0.33, fc='white', ec='#374151', lw=1.2)); ax.text(5.25, 4.78, '×', ha='center', va='center', fontsize=13)
    ax.add_patch(plt.Circle((8.35, 4.78), 0.33, fc='white', ec='#374151', lw=1.2)); ax.text(8.35, 4.78, '×', ha='center', va='center', fontsize=13)
    arrow(5.25, 6.2, 5.25, 5.15); arrow(8.35, 6.2, 8.35, 5.15); arrow(2.8, 4.78, 4.9, 4.78); arrow(5.6, 4.78, 8.0, 4.78, col='#9ca3af'); ax.text(6.8, 4.95, 'same gate', fontsize=7.5, color='#6b7280', ha='center')
    box(3.5, 2.3, 3.45, 1.4, 'down[:, J1] = u_AB\n+1 NO, +1 NE, −1 <tool_call>\n−0.5 I/To/The, 0 on 39 others', fc='#d1fae5', ec=C['v3'], fs=7.8)
    box(7.05, 2.3, 2.9, 1.4, 'down[:, J2] =\nu_NE − u_NO\n+1 NE, −1 NO, 0 on 39', fc='#ede9fe', ec=C['v2'], fs=7.8)
    arrow(5.25, 4.45, 5.25, 3.75); arrow(8.35, 4.45, 8.35, 3.75)
    box(2.4, 0.3, 6.4, 0.9, 'residual  →  final RMSNorm  →  tied unembedding  →  logits', fc='#eef2ff', ec='#6366f1', bold=True)
    arrow(5.25, 2.3, 5.4, 1.25); arrow(8.35, 2.3, 7.2, 1.25)
    ax.text(0.15, 3.0, 'act_J1 ≈ k²(s·x)²\nwhen the gate is open,\n≈ 0 otherwise\n\nact_J2 = SiLU(gate)·k_r·(r·x)\nsigned route, same gate', fontsize=8, color='#374151', va='center')
    ax.set_title('A. The two re-purposed neurons as a gated quadratic readout', loc='left', fontsize=11, fontweight='bold')
    # ---- scatter
    ax2 = fig.add_subplot(gs[1]); rng = np.random.default_rng(0)
    def sub(g, n): 
        d = P[g]; i = rng.choice(len(d['fs']), size=min(n, len(d['fs'])), replace=False); return d['fs'][i], d['close'][i]
    for g, n, col, mk, lab, al in (('gen_incall', 600, C['gen'], '.', 'generated positions inside tool calls', .35), ('gen_literal', 280, '#a78bfa', '.', 'generated positions after NO_TOOL / NEED_CLARIFICATION', .5),
                                  ('gen_prose_tool', 400, '#c4b5fd', '.', 'generated prose, tool context', .5), ('gen_prose_nontool', 600, '#8b5cf6', '.', 'generated prose, no tools (model\'s own answers)', .35)):
        fs_, cl_ = sub(g, n); ax2.scatter(fs_, cl_, s=9, c=col, marker=mk, alpha=al, label=lab, linewidths=0)
    ax2.scatter(P['nontool_dec']['fs'], P['nontool_dec']['close'], s=26, c=C['nontool'], marker='D', label='first token, prompt without tools (n=32)', zorder=3)
    ax2.scatter(P['call']['fs'], P['call']['close'], s=26, c=C['call'], marker='s', label='decision token, valid tool call (n=44)', zorder=4)
    ax2.scatter(P['ambiguous']['fs'], P['ambiguous']['close'], s=34, facecolors='none', edgecolors=C['amb'], marker='o', label='decision token, ambiguous "either X or Y" (n=10)', zorder=4, linewidths=1.3)
    ax2.scatter(P['abstain']['fs'], P['abstain']['close'], s=26, c=C['v3'], marker='o', label='decision token, should abstain (NO_TOOL / NEED_CLARIFICATION, n=70)', zorder=5)
    xs = np.linspace(-7, 7.5, 2); ax2.plot(xs, xs, color='#111827', lw=1.2, ls='--'); ax2.fill_between(xs, -3, xs, color='#d1fae5', alpha=.35, zorder=0)
    ax2.text(5.2, 1.6, 'gate OPEN\nact ≈ k²(s·x)²', fontsize=9, color=C['v3'], fontweight='bold', ha='center'); ax2.text(-6.6, 15.6, 'gate CLOSED (SiLU ≈ 0)', fontsize=9, color='#374151', fontweight='bold')
    ax2.axvline(0, color='#9ca3af', lw=.8); ax2.set_xlim(-7, 7.5); ax2.set_ylim(-3, 17)
    ax2.set_xlabel('s·x   (abstain feature read at the layer-23 MLP input)'); ax2.set_ylabel('0.3·c·x + 0.5·c₂·x   (gate-closing terms)')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.6, frameon=False, markerscale=1.6); ax2.yaxis.grid(True, color=C['grid']); ax2.set_axisbelow(True)
    ax2.set_title('B. Gate pre-activation = 2·(s·x − 0.3c·x − 0.5c₂·x): open below the dashed line', loc='left', fontsize=11, fontweight='bold')
    save(fig, 'fig3_two_neuron_circuit')

# ----------------------------------------------------------------------------------------------------------------------- figure 4
def fig4():
    T = json.loads((OUT / 'fig4_trace.json').read_text()); E = T['prompts']['demo-case-6']; V = E['variants']
    frag = re.compile(r'^(NO|NE|NONE|NENE|N|E|O|_TOOL|ED|_CL|AR|IFICATION|YES|EE)[,.\s]*$')
    order = [('Base', 'original', C['orig']), ('v1  k·s gate, no context terms', 'v1', C['v1']), ('v2  k·(s − 0.3c) gate', 'v2', C['v2']), ('g23v3  k·(s − 0.3c − 0.5c₂) gate', 'g23v3', C['v3'])]
    W = 128; fig = plt.figure(figsize=(13, 10.6)); ax = fig.add_subplot(111); fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02); ax.set_xlim(0, W); ax.axis('off'); y = 0; rowinfo = []
    FS = (0.96 * 13 * 72 / W) / 0.602 * 0.97  # monospace advance ≈ 0.602 em: one data unit per character
    ax.text(0, y, f"user: \u201c{E['user']}\u201d   system: \u201c{T['system']}\u201d   greedy, {T['max_tokens']} tokens, no tools", fontsize=9.5, color='#374151', va='top', family='monospace'); y -= 2.2
    for title, key, col in order:
        toks = V[key]['tokens']; nbad = 0; ident = '' if key == 'original' else ('   (token-identical to Base)' if V[key]['token_ids'] == V['original']['token_ids'] else '   (differs from Base)')
        ax.text(0, y, title + ident, fontsize=10.5, fontweight='bold', color=col, va='top'); y -= 1.55
        x = 0; line_y = y
        for tk in toks:
            disp = tk.replace('\n', '⏎'); bad = bool(frag.match(tk.strip())) and tk.strip() != ''
            if bad: nbad += 1
            if x + len(disp) > W: x = 0; line_y -= 1.25
            ax.text(x, line_y, disp, fontsize=FS, family='monospace', va='top', color='#991b1b' if bad else '#111827',
                    bbox=dict(boxstyle='square,pad=0.05', fc='#fee2e2', ec='none') if bad else None)
            x += len(disp)
        rowinfo.append((title, len(toks), nbad)); y = line_y - 2.6
        ax.text(W, line_y - 1.3, f'{len(toks)} tokens, {nbad} literal-fragment tokens', fontsize=8.5, color=col, ha='right', va='top')
    ax.set_ylim(y, 1)
    DATA['fig4'] = dict(prompt=E['user'], rows=[dict(variant=t, tokens=n, literal_fragment_tokens=b) for t, n, b in rowinfo],
                        base_equals_g23v3=V['original']['token_ids'] == V['g23v3']['token_ids'])
    fig.savefig(OUT / 'fig4_v1_v2_v3_ablation.svg'); fig.savefig(OUT / 'fig4_v1_v2_v3_ablation.png', dpi=200); plt.close(fig); print('wrote fig4_v1_v2_v3_ablation')

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    (OUT / 'figure_data.json').write_text(json.dumps(DATA, indent=1)); print('wrote figure_data.json')
