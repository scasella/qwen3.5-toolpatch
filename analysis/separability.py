"""How separable are call / answer / clarify at the decision position, per layer, in residual and MLP-activation space,
using teacher-derived difference-of-means directions evaluated per case on practice (cross-set)."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.lens import load_run, classes, practice_classes, dprime

def gapstats(f, act, pos_cls, neg_cls):
    pos=f[np.isin(act,pos_cls)]; neg=f[np.isin(act,neg_cls)]
    spread=np.sqrt((pos.var()+neg.var())/2)+1e-9
    return {'gap':float(pos.min()-neg.max()),'dprime':float((pos.mean()-neg.mean())/spread),'pos_min':float(pos.min()),'neg_max':float(neg.max()),'n_overlap_pos':int((pos<=neg.max()).sum()),'n_overlap_neg':int((neg>=pos.min()).sum())}

def main():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,popq=practice_classes(pa['case_ids'])
    print('=== residual stream h_l: diff-of-means (teacher) -> per-case projections; [teacher gap/dprime | practice gap/dprime | prefix max] ===')
    for l in range(8,25):
        H=ta['h_dec'][:,l]; Hp=pa['h_dec'][:,l]
        for name,pos,neg in (('noncall-vs-call',['answer','clarify'],['call']),('clarify-vs-answer',['clarify'],['answer'])):
            d=H[np.isin(tact,pos)].mean(0)-H[np.isin(tact,neg)].mean(0); d/=np.linalg.norm(d)
            ft=H@d; fp=Hp@d; gt=gapstats(ft,tact,pos,neg); gp=gapstats(fp,pact,pos,neg)
            pre=np.concatenate([ta['h_pre'][:,l]@d,pa['h_pre'][:,l]@d]) if l<25 and 'h_pre' in ta else None
            print(f'L{l:2d} {name:18s} teacher gap {gt["gap"]:7.2f} d\'={gt["dprime"]:5.2f} | practice gap {gp["gap"]:7.2f} d\'={gp["dprime"]:5.2f} overlap {gp["n_overlap_pos"]}/{gp["n_overlap_neg"]} | prefix max {pre.max() if pre is not None else float("nan"):7.2f} (call mean {fp[pact=="call"].mean():6.2f}, noncall mean {fp[pact!="call"].mean():6.2f})')
    print('\n=== MLP activations: top-k columns by |d\'| (teacher), v = diff of means on support; practice per-case gap ===')
    for l in range(10,24):
        A=ta['mlp_dec'][:,l]; Ap=pa['mlp_dec'][:,l]
        line=f'L{l:2d}'
        for name,pos,neg in (('NC',['answer','clarify'],['call']),('CA',['clarify'],['answer'])):
            d=dprime(A[np.isin(tact,pos)],A[np.isin(tact,neg)]); order=np.argsort(np.abs(d))[::-1]
            parts=[]
            for k in (64,256,1024,3584):
                cols=order[:k]; v=A[np.isin(tact,pos)][:,cols].mean(0)-A[np.isin(tact,neg)][:,cols].mean(0); v/=np.linalg.norm(v)
                fp=Ap[:,cols]@v; gp=gapstats(fp,pact,pos,neg)
                parts.append(f'k{k}:{gp["dprime"]:4.1f}/{gp["n_overlap_pos"]+gp["n_overlap_neg"]:2d}')
            line+=f' | {name} '+' '.join(parts)
        print(line)
    print('\n(format: dprime / number of overlapping practice cases)')
if __name__=='__main__': main()
