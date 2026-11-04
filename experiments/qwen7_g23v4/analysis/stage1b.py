"""Stage-1 direction trained as 'ambiguous clause-end vs everything else' (LDA with other prompt positions subsampled as negatives), LOFO by family;
leak statistics split by item class (ambiguous / resolved / call) and by position type (own clause tokens vs elsewhere), plus qwen-6 generated positions by kind."""
from __future__ import annotations
import sys, json
import numpy as np
from analysis.cap import Capture, dprime, unit, ROOT
from runtime.common import QWEN6
rng=np.random.default_rng(0)

def lda_dir(Xa,Xb,shrink):
    mu_a=Xa.mean(0); mu_b=Xb.mean(0); R=np.concatenate([Xa-mu_a,Xb-mu_b]); S=R.T@R/len(R); S=(1-shrink)*S+shrink*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    return unit(np.linalg.solve(S,mu_a-mu_b))

def main():
    labels=['d01b_triples_D1','d02_triples_D2','d03_triples_D3']; caps=[Capture(l) for l in labels]
    items=[]   # (cid, cap, variant, family, X (T,1024), end, clause_lo, clause_hi)
    for c in caps:
        for i in c.ids:
            ro=c.roles(i); X=c.full(i,'mlpin22'); lo=ro.get('v1c',ro['v1']); lo=min(lo,ro['v1']); hi=ro['end']
            items.append((i,c,c.meta[i]['variant'],c.meta[i]['family'],X,ro['end'],lo,hi))
    fams=np.array([it[3] for it in items]); var=np.array([it[2] for it in items])
    # negatives pool: ends of call/resolved + 40 random other positions per item (excluding the own clause span of ambiguous items)
    def negatives(sel):
        out=[]
        for it in sel:
            i,c,v,f,X,e,lo,hi=it
            if v!='ambiguous': out.append(X[e][None])
            idx=np.arange(len(X)); ok=(idx<lo)|(idx>hi) if v=='ambiguous' else idx!=e
            pick=rng.choice(idx[ok],40,replace=False); out.append(X[pick])
        return np.concatenate(out).astype(np.float64)
    def positives(sel): return np.stack([it[4][it[5]] for it in sel if it[2]=='ambiguous']).astype(np.float64)
    gen={}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); d=r['decision_position']
            kind=('AMBIG' if any(k in r['case_id'] for k in ()) else r['action'])
            gen.setdefault(('dec',r['action']),[]).append(X[d]); gen.setdefault(('gen',r['action']),[]).append(X[d+1:]); gen.setdefault(('prompt',r['action']),[]).append(X[:d])
    gen={k:np.concatenate([x[None] if x.ndim==1 else x for x in v]) for k,v in gen.items()}
    for shrink in (0.3,0.6):
        print(f'\n=== LDA(amb clause-end vs {{other ends + random prompt positions}}), shrink {shrink}')
        proj_end=np.zeros(len(items)); leak_own=np.full(len(items),-99.0); leak_else=np.full(len(items),-99.0)
        for f in np.unique(fams):
            tr=[it for it,ff in zip(items,fams) if ff!=f]; te=[k for k,ff in enumerate(fams) if ff==f]
            d=lda_dir(positives(tr),negatives(tr),shrink)
            for k in te:
                i,c,v,fam,X,e,lo,hi=items[k]; p=X@d; proj_end[k]=p[e]
                own=np.zeros(len(X),bool); own[lo:hi+1]=True
                leak_own[k]=p[own&(np.arange(len(X))!=e)].max(); leak_else[k]=p[~own].max()
        amb=var=='ambiguous'
        print(f" clause-end: amb [{proj_end[amb].min():.2f},{proj_end[amb].max():.2f}] (5th pct {np.percentile(proj_end[amb],5):.2f}) | call/res ends max {proj_end[~amb].max():.2f} | d' {dprime(proj_end[amb],proj_end[~amb]):.2f}")
        print(f" other own-clause tokens max: amb {leak_own[amb].max():.2f}  res {leak_own[var=='resolved'].max():.2f}  call {leak_own[var=='call'].max():.2f}")
        print(f" elsewhere in prompt max: amb {leak_else[amb].max():.2f}  res {leak_else[var=='resolved'].max():.2f}  call {leak_else[var=='call'].max():.2f}  (99th pct over items {np.percentile(leak_else,99):.2f})")
        d=lda_dir(positives(items),negatives(items),shrink)
        for k,v in sorted(gen.items()): 
            p=v@d; print(f" qwen-6 {k[0]:6s} {k[1]:8s}: n={len(p):5d} max {p.max():.2f} 99.9th {np.percentile(p,99.9):.2f}")
        np.save(ROOT/f'analysis/stage1_lda{shrink}.npy',d)
if __name__=='__main__': main()
