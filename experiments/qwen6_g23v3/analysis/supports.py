"""Candidate supports for feature A (explanation) and feature B (explicit missing value). Readout v on support C is the teacher
difference of class means, orthogonalized against the mean call activation on C (so f(call) ~ 0 on average); per-case f reported."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.lens import load_run, classes, practice_classes, dprime
from build_data import load_cases

def ortho(v, m):
    m=np.asarray(m,np.float64); v=np.asarray(v,np.float64)
    return v-(v@m)/(m@m+1e-12)*m

def main():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,_=practice_classes(pa['case_ids'])
    cases={c.id:c for c in load_cases('practice')}
    amb=np.array([cases[str(c)].spec.ambiguous for c in pa['case_ids']])
    notprov=(pact=='clarify')&~amb; ambig=(pact=='clarify')&amb
    out={}
    for feat,pos in (('A_explain','answer'),('B_missing','clarify')):
        print(f'\n===== feature {feat}: v = diff of means (teacher, {pos} - call) on top-k |d\'| columns, orthogonalized to call mean on support =====')
        print('per layer/k: f on practice: pos-class min/mean | call min/max | other-class max | NOTPROV min | AMBIG max | prefix maxabs ; teacher: pos min | call max')
        for l in range(10,22):
            A=ta['mlp_dec'][:,l]; Ap=pa['mlp_dec'][:,l]; Pre=np.concatenate([ta['mlp_pre'][:,l],pa['mlp_pre'][:,l]])
            d=dprime(A[tact==pos],A[tact=='call']); order=np.argsort(np.abs(d))[::-1]
            line=f'L{l:2d}'
            for k in (64,128,256,512):
                cols=order[:k]
                m_call=np.concatenate([A[tact=='call'][:,cols],Ap[pact=='call'][:,cols]]).mean(0)
                v=A[tact==pos][:,cols].mean(0)-A[tact=='call'][:,cols].mean(0); v=ortho(v,m_call); v/=np.linalg.norm(v)
                ft=A[:,cols]@v; fp=Ap[:,cols]@v; fpre=Pre[:,cols]@v
                if feat=='A_explain':
                    s=f' | k{k}: ans {fp[pact=="answer"].min():5.2f}/{fp[pact=="answer"].mean():4.2f} call {fp[pact=="call"].min():5.2f}/{fp[pact=="call"].max():4.2f} cla max {fp[pact=="clarify"].max():4.2f} pre {np.abs(fpre).max():4.2f} ; T ans {ft[tact=="answer"].min():4.2f} call {ft[tact=="call"].max():4.2f}'
                else:
                    s=f' | k{k}: NOTPROV {fp[notprov].min():5.2f}/{fp[notprov].mean():4.2f} AMBIG {fp[ambig].min():5.2f}/{fp[ambig].max():4.2f} call {fp[pact=="call"].min():5.2f}/{fp[pact=="call"].max():4.2f} ans max {fp[pact=="answer"].max():4.2f} pre {np.abs(fpre).max():4.2f} ; T cla {ft[tact=="clarify"].min():4.2f} call {ft[tact=="call"].max():4.2f}'
                line+=s
                out[f'{feat}_L{l}_k{k}']={'cols':[int(c) for c in cols],'v':[float(x) for x in v]}
            print(line)
    json.dump(out,open('analysis/support_candidates.json','w'))
if __name__=='__main__': main()
