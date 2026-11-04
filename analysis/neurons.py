"""Localization statistics from teacher-forced captures (CPU only)."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.lens import *

def summarize_decisions(js):
    rows=js['rows']; out={}
    for cls in ('call','answer','clarify'):
        rr=[r for r in rows if r['action']==cls]
        if not rr: continue
        g=np.array([[r['gaps']['call_minus_NO'],r['gaps']['call_minus_NE'],r['gaps']['NE_minus_NO'],r['gaps']['call_minus_argmax_other']] for r in rr])
        out[cls]={'n':len(rr),'mean_call-NO':g[:,0].mean(),'min':g[:,0].min(),'max':g[:,0].max(),'mean_call-NE':g[:,1].mean(),'mean_NE-NO':g[:,2].mean(),'min_NE-NO':g[:,2].min(),'max_NE-NO':g[:,2].max(),
                  'argmax_counts':{},'gold_first_rank':[r['decision']['gold']['rank'] for r in rr]}
        from collections import Counter
        out[cls]['argmax_counts']=dict(Counter(r['decision']['topk'][0]['text'] for r in rr))
    return out

def continuation_summary(js):
    out={}
    for cls in ('answer','clarify','call'):
        rr=[r for r in js['rows'] if r['action']==cls]
        if not rr: continue
        L=max(len(r['continuation_tokens']) for r in rr)
        per=[]
        for j in range(min(L,8)):
            lps=[r['continuation_tokens'][j]['logprob'] for r in rr if len(r['continuation_tokens'])>j]
            ranks=[r['continuation_tokens'][j]['rank'] for r in rr if len(r['continuation_tokens'])>j]
            per.append({'j':j,'text':rr[0]['continuation_tokens'][j]['text'],'mean_lp':float(np.mean(lps)),'min_lp':float(np.min(lps)),'max_rank':int(max(ranks)),'n_rank1':int(sum(r==1 for r in ranks)),'n':len(lps)})
        out[cls]=per
    return out

def layer_lens(arr,acts):
    """virtual margins per layer per class using final norm + unembedding on residual h_l."""
    toks=[TOK['call'],TOK['NO'],TOK['NE']]
    res={}
    for cls in ('call','answer','clarify'):
        idx=np.where(acts==cls)[0]
        if len(idx)==0: continue
        m=[]
        for l in range(25):
            v=lens(arr['h_dec'][idx,l],toks)  # (n,3)
            m.append([float((v[:,0]-v[:,1]).mean()),float((v[:,0]-v[:,2]).mean()),float((v[:,2]-v[:,1]).mean())])
        res[cls]=np.array(m)
    return res

def neuron_stats(arr,acts,ren):
    """per (layer,neuron): d' noncall-vs-call, clarify-vs-answer, call quiescence, orig/renamed agreement."""
    A=arr['mlp_dec']  # (n,24,3584)
    call=acts=='call'; non=~call; cla=acts=='clarify'; ans=acts=='answer'
    d_non=np.stack([dprime(A[non,l],A[call,l]) for l in range(24)])           # +: higher on non-call
    d_cla=np.stack([dprime(A[cla,l],A[ans,l]) for l in range(24)])            # +: higher on clarify
    mean_non=A[non].mean(0); mean_call=A[call].mean(0); mean_cla=A[cla].mean(0); mean_ans=A[ans].mean(0)
    maxabs_call=np.abs(A[call]).max(0); min_non=A[non].min(0); max_non=A[non].max(0)
    # name invariance: correlation across the (group,class) grid between original and renamed
    stats={'d_non':d_non,'d_cla':d_cla,'mean_non':mean_non,'mean_call':mean_call,'mean_cla':mean_cla,'mean_ans':mean_ans,'maxabs_call':maxabs_call,'min_non':min_non,'max_non':max_non}
    if ren is not None and ren.any():
        o=A[~ren]; r=A[ren]
        if o.shape==r.shape:
            oc=o-o.mean(0); rc=r-r.mean(0)
            corr=(oc*rc).sum(0)/np.sqrt((oc*oc).sum(0)*(rc*rc).sum(0)+1e-12)
            stats['name_corr']=corr
    return stats

def dla(arr,acts,layer_range=range(24)):
    """direct logit attribution of each neuron (through final norm scale of its own case) to call-NO and NE-NO logit differences."""
    E=embed(); g=gamma_final()
    u_callNO=g*(E[TOK['call']]-E[TOK['NO']]); u_NENO=g*(E[TOK['NE']]-E[TOK['NO']]); u_callNE=g*(E[TOK['call']]-E[TOK['NE']])
    r=rms(arr['h_dec'][:,24])  # (n,)
    out={}
    for l in layer_range:
        W=P.load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight')  # (1024,3584)
        p1=W.T@u_callNO; p2=W.T@u_NENO; p3=W.T@u_callNE   # (3584,)
        A=arr['mlp_dec'][:,l]  # (n,3584)
        out[l]={'callNO':A*p1[None,:]/r[:,None],'NENO':A*p2[None,:]/r[:,None],'callNE':A*p3[None,:]/r[:,None]}
    return out

if __name__=='__main__':
    label=sys.argv[1] if len(sys.argv)>1 else 'p02_teacher_dec'
    js,arr=load_run(label)
    print('lens check:',json.dumps(verify_lens(js,arr,3)))
    acts,ren=classes(arr['case_ids']) if label.startswith('p02') else practice_classes(arr['case_ids'])
    print('decisions:',json.dumps(summarize_decisions(js),default=float,indent=None))
    print('continuations:',json.dumps(continuation_summary(js),indent=None))
    ll=layer_lens(arr,acts)
    print('layer lens (call-NO, call-NE, NE-NO) per class:')
    for l in range(25): print(l, {c:[round(x,2) for x in ll[c][l]] for c in ll})
