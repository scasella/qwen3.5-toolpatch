"""Direct logit attribution of MLP neurons (per layer) to the call-NO margin at the decision position, by class."""
from __future__ import annotations
import numpy as np
from analysis.lens import load_run, classes, practice_classes, embed, gamma_final, rms, TOK
import primitives as P

def main():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,_=classes(ta['case_ids']); pact,_=practice_classes(pa['case_ids'])
    E=embed(); g=gamma_final(); u=g*(E[TOK['call']]-E[TOK['NO']])
    print('=== per-layer MLP direct contribution to (call - NO) logit margin, mean by class [teacher | practice] ===')
    tops={}
    for l in range(24):
        W=P.load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight'); p=W.T@u
        rows=[]
        for arr,act,nm in ((ta,tact,'T'),(pa,pact,'P')):
            D=arr['mlp_dec'][:,l]*p[None,:]/rms(arr['h_dec'][:,24])[:,None]   # (n,3584) per-neuron contributions
            tot=D.sum(1); rows.append((nm,{c:float(tot[act==c].mean()) for c in ('call','answer','clarify')},D,act))
        # neurons that push call on NON-call cases but not on calls: mean D on noncall minus mean D on call (practice), with call max
        D=rows[1][2]; act=rows[1][3]
        drive=D[act!='call'].mean(0)-D[act=='call'].mean(0)
        order=np.argsort(drive)[::-1][:5]
        print(f'L{l:2d} T call {rows[0][1]["call"]:6.2f} ans {rows[0][1]["answer"]:6.2f} cla {rows[0][1]["clarify"]:6.2f} | P call {rows[1][1]["call"]:6.2f} ans {rows[1][1]["answer"]:6.2f} cla {rows[1][1]["clarify"]:6.2f} | top noncall-selective call-pushers: '+', '.join(f'{int(i)}:{drive[i]:+.2f}(call {D[act=="call",i].mean():+.2f})' for i in order))
if __name__=='__main__': main()
