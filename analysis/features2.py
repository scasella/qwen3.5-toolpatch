"""Separate features: explanation (answer vs call) and missing (clarify vs call), per layer; residual + MLP top-k; prefix quiescence; cross-set."""
from __future__ import annotations
import numpy as np
from analysis.lens import load_run, classes, practice_classes, dprime
from analysis.separability import gapstats

def main():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,popq=practice_classes(pa['case_ids'])
    for name,pos in (('EXPLAIN answer-vs-call',['answer']),('MISSING clarify-vs-call',['clarify'])):
        print(f'\n=== {name}: residual diff-of-means (teacher) -> practice per-case; prefix = max over call prefix positions (teacher+practice) ===')
        for l in range(6,25):
            H=ta['h_dec'][:,l]; Hp=pa['h_dec'][:,l]
            d=H[np.isin(tact,pos)].mean(0)-H[tact=='call'].mean(0); d/=np.linalg.norm(d)
            ft=H@d; fp=Hp@d; gt=gapstats(ft,tact,pos,['call']); gp=gapstats(fp,pact,pos,['call'])
            other=[c for c in ('answer','clarify') if c not in pos][0]
            pre=np.concatenate([ta['h_pre'][:,l]@d,pa['h_pre'][:,l]@d])
            # center on call mean for interpretability
            c0=fp[pact=='call'].mean()
            print(f'L{l:2d} teacher d\'={gt["dprime"]:5.2f} ov={gt["n_overlap_pos"]+gt["n_overlap_neg"]:2d} | practice d\'={gp["dprime"]:5.2f} ov={gp["n_overlap_pos"]}/{gp["n_overlap_neg"]} | pos min-callmean {fp[np.isin(pact,pos)].min()-c0:6.2f} call max-callmean {fp[pact=="call"].max()-c0:6.2f} other-class mean {fp[pact==other].mean()-c0:6.2f} | prefix max-callmean {pre.max()-c0:6.2f}')
        print(f'--- {name}: MLP top-k by |d\'| (teacher), v=diff of means on support -> practice d\'/overlap; prefix maxabs relative to pos mean ---')
        for l in range(8,24):
            A=ta['mlp_dec'][:,l]; Ap=pa['mlp_dec'][:,l]; Apre=np.concatenate([ta['mlp_pre'][:,l],pa['mlp_pre'][:,l]])
            d=dprime(A[np.isin(tact,pos)],A[tact=='call']); order=np.argsort(np.abs(d))[::-1]
            parts=[]
            for k in (32,64,128,256,512):
                cols=order[:k]; v=A[np.isin(tact,pos)][:,cols].mean(0)-A[tact=='call'][:,cols].mean(0); v/=np.linalg.norm(v)
                fp=Ap[:,cols]@v; gp=gapstats(fp,pact,pos,['call']); c0=fp[pact=='call'].mean(); posmean=fp[np.isin(pact,pos)].mean()-c0
                pre=np.abs(Apre[:,cols]@v-c0).max()
                parts.append(f'k{k}:{gp["dprime"]:4.1f}/{gp["n_overlap_pos"]+gp["n_overlap_neg"]:2d} pre{pre/posmean:4.2f}')
            print(f'L{l:2d} '+' | '.join(parts))
if __name__=='__main__': main()
