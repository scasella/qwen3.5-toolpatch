"""Layer-23 transport analysis for the K-alignment design: for each head h, (1) consistency of its decision query across items, (2) natural attention logits,
(3) the marker direction p_k = (Wk gamma)^+ k_target with k_target = gamma_k * mean decision query (unrotated dims), (4) replay with the clause-end residual replaced by
B*p (B large: the write dominates) -> attention mass on the marker for the decision query of every head, payload norm, change of the MLP input x23 at the decision."""
from __future__ import annotations
import sys, json
import numpy as np
from analysis.cap import Capture, unit, dprime, ROOT
from analysis.replay23 import Replay
import primitives as P
L='model.language_model.layers.23.{}.weight'

def main():
    caps=[Capture(l) for l in ('d01b_triples_D1','d03_triples_D3')]
    rp=Replay()
    Wk=P.load_original(L.format('self_attn.k_proj')); Wv=P.load_original(L.format('self_attn.v_proj')); Wo=P.load_original(L.format('self_attn.o_proj')); Wq=P.load_original(L.format('self_attn.q_proj'))
    gin=P.load_original('model.language_model.layers.23.input_layernorm.weight')+1.0; gq=P.load_original(L.format('self_attn.q_norm'))+1.0; gk=P.load_original(L.format('self_attn.k_norm'))+1.0
    # (1),(2): natural decision queries / logits
    Q=[];Klog=[];meta=[]
    sel=[(c,i) for c in caps for i in c.ids]
    for c,i in sel[:96]:
        h23=c.full(i,'h23'); ro=c.roles(i); r=rp.run(h23,query_positions=(ro['dec'],),want_x23=True,want_qk=True)
        q=r['rows']['q'][:,ro['dec'],:]; k=r['rows']['k']; T=len(h23)
        Q.append(q); row=r['rows'][ro['dec']]
        lg=np.stack([(q[h]@np.repeat(k,4,axis=0)[h].T)/16 for h in range(8)])   # (8,T) logits
        Klog.append({'max':lg.max(1),'end':lg[:,ro['end']],'argmax':lg.argmax(1),'alpha_end':row[:,ro['end']],'T':T,'dec':ro['dec'],'gate_mean':r['gate'][ro['dec']].mean(1)})
        meta.append((i,c.meta[i]['variant']))
    Q=np.stack(Q)   # (n,8,256) post-norm post-rope decision queries
    print('decision-query consistency per head (mean cos to the mean query; unrotated dims 64:):')
    for h in range(8):
        qu=Q[:,h,64:]; m=unit(qu.mean(0)); cos=(qu@m)/np.linalg.norm(qu,axis=1)
        print(f"  h{h}: cos mean {cos.mean():.3f} min {cos.min():.3f} | natural logit max {np.mean([k['max'][h] for k in Klog]):.2f} (range {np.min([k['max'][h] for k in Klog]):.2f}..{np.max([k['max'][h] for k in Klog]):.2f}), logit at end {np.mean([k['end'][h] for k in Klog]):.2f} | alpha_end {np.mean([k['alpha_end'][h] for k in Klog]):.4f} | gate mean {np.mean([k['gate_mean'][h] for k in Klog]):.2f}")
    # (3) marker direction per head via the existing k_proj
    print('\nattention-logit cap per head: ||(gamma_k * qhat_h)[64:]|| (a marker key can reach this logit at most; natural max logits above)')
    for h in range(8):
        caps_=np.linalg.norm((Q[:,h,:]*gk[None,:])[:,64:],axis=1); print(f'  h{h}: cap mean {caps_.mean():.2f} min {caps_.min():.2f} | natural max mean {np.mean([k["max"][h] for k in Klog]):.2f} -> margin {caps_.mean()-np.mean([k["max"][h] for k in Klog]):.2f}')
    print('\nK-alignment marker per head: p_k = pinv(Wk_g gamma) k_target, k_target = gamma_k * mean q (dims 64:), zero on rotated dims')
    res={}
    for h in range(8):
        WkG=Wk[(h//4)*256:(h//4+1)*256]*gin[None,:]; pinv=np.linalg.pinv(WkG)   # kv group of head h
        qbar=Q[:,h,:].mean(0); kt=np.zeros(256); kt[64:]=(gk*qbar)[64:]
        # orthogonalize against the other heads of the same kv group (weighted by gamma_k) to limit spillover
        others=[Q[:,h2,:].mean(0)*gk for h2 in range(8) if h2//4==h//4 and h2!=h]
        kt_o=kt.copy()
        for o in others:
            o=o.copy(); o[:64]=0; kt_o-=(kt_o@o)/(o@o)*o
        p=pinv@kt_o; p=unit(p)
        # replay on ambiguous items with the clause-end residual replaced by 100*p (write dominates); measure attention and x23 change
        alphas=[];spill=[];dx=[];payload=[]
        for (c,i),m in zip(sel[:96],meta):
            if m[1]!='ambiguous': continue
            h23=c.full(i,'h23'); ro=c.roles(i); base=rp.run(h23,query_positions=(ro['dec'],),want_x23=True)
            h2=h23.copy(); h2[ro['end']]=100*p; ed=rp.run(h2,query_positions=(ro['dec'],),want_x23=True)
            row=ed['rows'][ro['dec']]; alphas.append(row[h,ro['end']]); spill.append([row[hh,ro['end']] for hh in range(8) if hh!=h])
            dx.append(ed['x23'][ro['dec']]-base['x23'][ro['dec']]); payload.append(ed['attn_out'][ro['dec']]-base['attn_out'][ro['dec']])
        dx=np.stack(dx); payload=np.stack(payload); t=unit(dx.mean(0)); cons=(dx@t)/np.linalg.norm(dx,axis=1)
        res[h]={'alpha_end':float(np.mean(alphas)),'alpha_min':float(np.min(alphas)),'spill_max':float(np.max(spill)),'payload_norm':float(np.linalg.norm(payload,axis=1).mean()),'dx23_norm':float(np.linalg.norm(dx,axis=1).mean()),'dx23_consistency':float(cos:=cons.mean()),'t':t}
        print(f"  h{h}: alpha at marker {np.mean(alphas):.3f} (min {np.min(alphas):.3f}) spill max (other heads) {np.max(spill):.3f} | payload |d attn_out| {np.linalg.norm(payload,axis=1).mean():.2f} | |d x23| {np.linalg.norm(dx,axis=1).mean():.2f} consistency(cos to mean) {cons.mean():.3f} min {cons.min():.3f}")
    np.savez(ROOT/'analysis/transport_heads.npz',**{f't{h}':res[h]['t'] for h in res},Q=Q)
    json.dump({h:{k:v for k,v in r.items() if k!='t'} for h,r in res.items()},open(ROOT/'analysis/transport.json','w'),indent=1)
if __name__=='__main__': main()
