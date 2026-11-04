"""Realizing a threshold without a bias term: find the direction r whose projection r.x is most nearly constant over the critical positions (ambiguous clause ends
and the top-scoring harmful positions), r = C^-1 mu / (mu^T C^-1 mu) with shrinkage; then gate = w - theta * r gives w.x - theta*(r.x) ~ w.x - theta.
Reports the achievable margins for stage 1 (layer-22 input) and for J3 (layer-23 input)."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, unit, ROOT
from analysis.stage1e import harmful_sets
import g23v4_build as B
from runtime.common import QWEN6

def const_dir(X,shrink=0.3):
    mu=X.mean(0); R=X-mu; C=R.T@R/len(R); C=(1-shrink)*C+shrink*np.trace(C)/C.shape[0]*np.eye(C.shape[0]); r=np.linalg.solve(C,mu); r/=mu@r; return r

def main():
    D=B.fit_directions(); pos,posfam,harm=harmful_sets(); harm={k:v for k,v in harm.items() if not (k[0]=='q6' and k[1]=='clarify')}
    allharm=np.concatenate([X for v in harm.values() for X in v]).astype(np.float64)
    for a_g in (0.5,0.75):
        w1=unit(a_g*D['g']+D['u']); s=allharm@w1; top=allharm[np.argsort(s)[-3000:]]
        crit=np.concatenate([pos,top])
        for shrink in (0.1,0.3,0.6):
            r=const_dir(crit,shrink); rx_pos=pos@r; rx_top=top@r; rx_all=allharm@r
            print(f"a_g={a_g} shrink={shrink}: r.x on amb ends [{rx_pos.min():.3f},{rx_pos.max():.3f}] top-harmful [{rx_top.min():.3f},{rx_top.max():.3f}] all-harmful [{rx_all.min():.3f},{rx_all.max():.3f}] (|r|={np.linalg.norm(r):.2f})")
            for theta in (10.0,11.0,11.7,12.5):
                w=w1-theta*r; za=pos@w; zh=allharm@w
                print(f"     theta={theta}: amb ends z [{za.min():.2f},{za.max():.2f}] | harmful max {zh.max():.2f} (99.99th {np.percentile(zh,99.99):.2f}) | margin {za.min()-zh.max():.2f} | |w|={np.linalg.norm(w):.2f}")
    # J3: critical = decision x23 of all items (natural) and delivered values on ambiguous; harmful = natural decisions + generated positions
    J=dict(np.load(ROOT/'analysis/g23v4_directions.npz'))
    caps=[Capture(l) for l in B.LABELS]; from analysis.replay23 import Replay
    rp=Replay(); nat=[];dl=[]
    for c in caps:
        for i in c.ids[:40]:
            h23=c.full(i,'h23'); ro=c.roles(i); b=rp.run(h23,want_x23=True); nat.append(b['x23'][ro['dec']])
            if c.meta[i]['variant']=='ambiguous':
                h2=h23.copy(); h2[ro['end']]+=100*D['p6']; e=rp.run(h2,want_x23=True); dl.append(e['x23'][ro['dec']])
    nat=np.stack(nat); dl=np.stack(dl)
    gen=[]
    for lab in ('p04c_practice_allpos','p05_teacher_allpos','p17_prose_capture','p35_prose_capture_nontool','p16_regression_capture'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r_ in js['rows']: X=z[f"mlpin_L23_{r_['case_id']}"].astype(np.float32); gen.append(X[r_['decision_position']:])
    g01=Capture('g01_genpos')
    for i in g01.ids: gen.append(g01.full(i,'mlpin23')[g01.rows[i]['decision_position']:])
    gen=np.concatenate(gen).astype(np.float64)
    V3=np.load(ROOT/'edits/g23v3.vectors.npz'); tg=D['t']-0.3*V3['c']-0.5*V3['c2']
    s=gen@tg; topg=gen[np.argsort(s)[-3000:]]; crit=np.concatenate([nat,dl,topg]); r3=const_dir(crit,0.3)
    print(f"\nJ3: r3.x on natural decisions [{(nat@r3).min():.3f},{(nat@r3).max():.3f}] delivered [{(dl@r3).min():.3f},{(dl@r3).max():.3f}] top generated [{(topg@r3).min():.3f},{(topg@r3).max():.3f}]")
    for theta in (0.0,1.0,2.0,3.0):
        w=tg-theta*r3; print(f"   theta={theta}: delivered z [{(dl@w).min():.2f},{(dl@w).max():.2f}] | natural decisions max {(nat@w).max():.2f} | generated max {(gen@w).max():.2f} (99.9th {np.percentile(gen@w,99.9):.2f})")
    np.savez(ROOT/'analysis/bias_dirs.npz',r3=r3)
if __name__=='__main__': main()
