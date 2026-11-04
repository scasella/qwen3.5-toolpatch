"""J3 read feature at the layer-23 MLP input: t = unit(mean delta x23) delivered by head 6 when the clause end carries the marker p6 (write dominates).
Reports t.x23 (natural) at the decision of every triple item, the delivered value on ambiguous items, natural t.x23 at qwen-6 decision/generated positions and at
g01 prose/non-tool positions; also the g23v3 gate terms c, c2 at those positions for reference. Saves p6, t."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, unit, dprime, ROOT
from analysis.replay23 import Replay
from runtime.common import QWEN6
import primitives as P
L='model.language_model.layers.23.{}.weight'

def main():
    caps=[Capture(l) for l in ('d01b_triples_D1','d02_triples_D2','d03_triples_D3')]
    rp=Replay(); Wk=P.load_original(L.format('self_attn.k_proj')); gin=P.load_original('model.language_model.layers.23.input_layernorm.weight')+1.0; gk=P.load_original(L.format('self_attn.k_norm'))+1.0
    T=np.load(ROOT/'analysis/transport_heads.npz'); Q=T['Q']; h=6
    qbar=Q[:,h,:].mean(0); kt=np.zeros(256); kt[64:]=(gk*qbar)[64:]
    for h2 in (4,5,7):
        o=(Q[:,h2,:].mean(0)*gk).copy(); o[:64]=0; kt-=(kt@o)/(o@o)*o
    WkG=Wk[256:512]*gin[None,:]; p=unit(np.linalg.pinv(WkG)@kt)
    V3=np.load(ROOT/'edits/g23v3.vectors.npz'); c=V3['c']; c2=V3['c2']; s=V3['s']
    base_x=[];ed_x=[];var=[];ids=[];alpha=[]
    for cap in caps:
        for i in cap.ids:
            h23=cap.full(i,'h23'); ro=cap.roles(i); b=rp.run(h23,query_positions=(ro['dec'],),want_x23=True); base_x.append(b['x23'][ro['dec']]); var.append(cap.meta[i]['variant']); ids.append(i)
            if cap.meta[i]['variant']=='ambiguous':
                h2=h23.copy(); h2[ro['end']]=100*p; e=rp.run(h2,query_positions=(ro['dec'],),want_x23=True); ed_x.append(e['x23'][ro['dec']]); alpha.append(e['rows'][ro['dec']][h,ro['end']])
    base_x=np.stack(base_x); ed_x=np.stack(ed_x); var=np.array(var); amb=var=='ambiguous'
    dx=ed_x-base_x[amb]; t=unit(dx.mean(0))
    print(f'alpha(h6 at marker) mean {np.mean(alpha):.3f} min {np.min(alpha):.3f}; |dx23| mean {np.linalg.norm(dx,axis=1).mean():.2f}')
    fb=base_x@t; fe=ed_x@t
    print(f"t.x23 natural at decision: amb [{fb[amb].min():.2f},{fb[amb].max():.2f}] call [{fb[var=='call'].min():.2f},{fb[var=='call'].max():.2f}] res [{fb[var=='resolved'].min():.2f},{fb[var=='resolved'].max():.2f}] | delivered (amb with marker) [{fe.min():.2f},{fe.max():.2f}]")
    print(f"s.x23 (g23v3 abstain) at decision: amb [{(base_x@s)[amb].min():.2f},{(base_x@s)[amb].max():.2f}] -> with marker [{(ed_x@s).min():.2f},{(ed_x@s).max():.2f}]")
    print(f"c.x23: amb nat [{(base_x@c)[amb].min():.2f},{(base_x@c)[amb].max():.2f}] marker [{(ed_x@c).min():.2f},{(ed_x@c).max():.2f}] | c2.x23: nat [{(base_x@c2)[amb].min():.2f},{(base_x@c2)[amb].max():.2f}] marker [{(ed_x@c2).min():.2f},{(ed_x@c2).max():.2f}]")
    # natural t.x23 elsewhere
    rows={}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos','p17_prose_capture','p35_prose_capture_nontool','p16_regression_capture'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L23_{r['case_id']}"].astype(np.float32); d=r['decision_position']
            rows.setdefault(f'{lab}:dec:{r["action"]}',[]).append(X[d]@t); rows.setdefault(f'{lab}:gen:{r["action"]}',[]).append(X[d+1:]@t)
    g01=Capture('g01_genpos')
    for i in g01.ids:
        X=g01.full(i,'mlpin23'); d=g01.rows[i]['decision_position']; kind='prose' if i.startswith('prose') else 'reg2'
        rows.setdefault(f'g01:dec:{kind}',[]).append(X[d]@t); rows.setdefault(f'g01:gen:{kind}',[]).append(X[d+1:]@t)
    print('\nnatural t.x23 elsewhere (max / 99.9th):')
    for k in sorted(rows):
        v=np.concatenate([np.atleast_1d(x) for x in rows[k]]); print(f'  {k:45s} n={len(v):5d} max {v.max():6.2f} 99.9th {np.percentile(v,99.9):6.2f} min {v.min():6.2f}')
    np.savez(ROOT/'analysis/j3_feature.npz',p6=p,t=t,kt=kt,base_x=base_x,ed_x=ed_x,var=var,ids=np.array(ids))
if __name__=='__main__': main()
