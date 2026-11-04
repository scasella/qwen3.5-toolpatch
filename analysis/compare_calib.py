"""Compare an edited teacher-forced pass with the CPU simulation of the same edit; summarize continuation safety."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.simulate import load_all, simulate, EPS, TOK
from analysis.lens import load_run, embed, gamma_final
from build_g23 import J1, J2

def main(label, edit_label):
    spec=json.load(open(f'edits/{edit_label}.spec.json')); V=np.load(f'edits/{edit_label}.vectors.npz')
    D,W=load_all(); E=embed(); g=gamma_final()
    lg,fs,fr,a1,a2=simulate(D,W,V['s'],V['r'],spec['k'],spec['kr'],spec['a1'],spec['a2'],J1,J2,V['u_AB'],V['u_route'],E,g,'dec')
    js,arr=load_run(label); ids=list(D['ids'])
    errs=[];agree=0;rows=[]
    tok_watch={'<tool_call>':TOK['call'],'NO':TOK['NO'],'NE':TOK['NE']}
    for r in js['rows']:
        i=ids.index(r['case_id']); sim=lg[i]; meas=r['decision']['watch']
        e=[meas[w]-sim[t] for w,t in tok_watch.items()]; errs.append(max(abs(x) for x in e))
        sim_arg=int(sim.argmax()); agree+=int(sim_arg==r['decision']['argmax'])
        rows.append((r['case_id'],r['action'],r['decision']['topk'][0]['text'],round(meas['NO']-meas['<tool_call>'],1),round(meas['NE']-meas['NO'],1),[c['rank'] for c in r['continuation_tokens']],[round(c['logprob'],2) for c in r['continuation_tokens']]))
    print(f'decision argmax agreement sim vs measured: {agree}/{len(rows)}; max |logit err| over NO/NE/call: {max(errs):.2f} (median {np.median(errs):.2f})')
    from collections import Counter
    for cls in ('call','answer','clarify'):
        rr=[x for x in rows if x[1]==cls]; print(f'{cls}: argmax counts',dict(Counter(x[2] for x in rr)),'| NO-call range',(min(x[3] for x in rr),max(x[3] for x in rr)),'| NE-NO range',(min(x[4] for x in rr),max(x[4] for x in rr)))
    bad=[(x[0],x[1],x[5],x[6]) for x in rows if any(k!=1 for k in x[5])]
    print('continuation tokens with rank != 1:',len(bad)); 
    for b in bad[:20]: print('  ',b)
    # continuation watch margins: min over continuation positions of (gold logit - max(NO,NE logits)) using watch dict
    worst=[]
    for r in js['rows']:
        for c in r['continuation_tokens']:
            w=c['watch']; gold_logit=None
            # gold logit not stored directly; use top3 and rank: if rank==1 the gold is argmax; margin vs NO/NE from watch requires gold logit -> approximate with logprob? skip
            worst.append((r['case_id'],c['text'],c['rank'],round(c['logprob'],3),{k:round(v,1) for k,v in w.items()},c['top3']))
    worst.sort(key=lambda t:(t[2]==1,t[3]))
    print('lowest-confidence continuation positions:'); 
    for t in worst[:12]: print('  ',t)
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
