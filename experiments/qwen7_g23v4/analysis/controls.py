"""Dose-response / random-control table on data/control.ids: zero (g23v3, from g02/g03/p38), half, intended (g23v4 from g04/g05), double, reversed, random (k01-k04)."""
import json
from runtime.common import ROOT, QWEN6
from build_data import load_cases
ids=(ROOT/'data/control.ids').read_text().split()
def rows(lab,w,root=ROOT): return {r['case_id']:r for r in json.load(open(root/'runs'/f'{lab}.json'))['rows'][w]}
zero={}; zero.update(rows('g02_stress_amb_base_v3','g23v3')); zero.update(rows('g03_stress_res_base_v3','g23v3')); zero.update(rows('p38_v3_gen_practice','edited',QWEN6))
intended={}; intended.update(rows('g04_stress_v4','g23v4')); intended.update(rows('g05_practice_v4','g23v4'))
base={}; base.update(rows('g02_stress_amb_base_v3','original')); base.update(rows('g03_stress_res_base_v3','original')); base.update(rows('p01_baseline','original',QWEN6))
models=[('Zero (g23v3)',zero),('Half (a3=0.05, b=0.5)',rows('k01_control_half','g23v4_half')),('Intended (g23v4)',intended),('Double (a3=0.2, b=2)',rows('k02_control_double','g23v4_double')),
        ('Reversed (a3=-0.1)',rows('k03_control_rev','g23v4_rev')),('Random norm-matched (seed 17)',rows('k04_control_rand17','g23v4_rand17'))]
cls={}
for c in load_cases('stress')+load_cases('practice'):
    cls[c.id]='ambiguous' if (c.spec.action=='clarify' and c.spec.ambiguous) else 'resolved' if c.id.startswith('stress') else {'call':'call','answer':'no_tool','clarify':'explicit_missing'}[c.spec.action]
groups=[('ambiguous',32),('resolved',16),('call',8),('no_tool',4),('explicit_missing',4)]
out=['| edit | ambiguous -> NEED_CLARIFICATION (32) | resolved calls correct (16) | practice calls correct (8) | NO_TOOL (4) | explicit-missing (4) | total lenient (64) | vs g23v3: w->r / r->w / token-identical | vs Base: r->w |','|---|---|---|---|---|---|---|---|---|']
J={}
for name,R in models:
    cells=[]
    for g,n in groups:
        sub=[i for i in ids if cls[i]==g]; k=sum(R[i]['score']['correct'] for i in sub); dec={}
        for i in sub: d=str(R[i]['score']['decision']); dec[d]=dec.get(d,0)+1
        cells.append(f"{k}/{n} {dec}")
    tot=sum(R[i]['score']['correct'] for i in ids)
    w2r=sum((not zero[i]['score']['correct']) and R[i]['score']['correct'] for i in ids); r2w=sum(zero[i]['score']['correct'] and not R[i]['score']['correct'] for i in ids)
    ident=sum(zero[i]['token_ids']==R[i]['token_ids'] for i in ids); br2w=sum(base[i]['score']['correct'] and not R[i]['score']['correct'] for i in ids)
    out.append(f"| {name} | {' | '.join(cells)} | {tot}/64 | {w2r} / {r2w} / {ident} | {br2w} |")
    J[name]={'total':tot,'w2r_vs_g23v3':w2r,'r2w_vs_g23v3':r2w,'identical_vs_g23v3':ident,'r2w_vs_base':br2w,'cells':cells}
(ROOT/'analysis/controls.md').write_text('\n'.join(out)+'\n'); json.dump(J,open(ROOT/'analysis/controls.json','w'),indent=1); print('\n'.join(out))
