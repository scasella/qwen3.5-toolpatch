"""Column screening for a selective abstention feature on MLP down_proj inputs.

For each layer: per-neuron teacher statistics (matched triples, orig+renamed) and practice statistics; candidates must
(a) separate non-call from call at the decision position on BOTH sets, (b) be quiescent per case on every call (decision and '=' prefix),
(c) agree between original and renamed twins. v = normalized difference of class means on the support; f(x)=v.a_C(x) reported per case."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.lens import load_run, classes, practice_classes, dprime

def stats(A, non, call, cla=None, ans=None):
    out={'mean_non':A[non].mean(0),'mean_call':A[call].mean(0),'sd_non':A[non].std(0),'sd_call':A[call].std(0),
         'maxabs_call':np.abs(A[call]).max(0),'d_non':dprime(A[non],A[call])}
    if cla is not None: out['d_cla']=dprime(A[cla],A[ans]); out['mean_cla']=A[cla].mean(0); out['mean_ans']=A[ans].mean(0)
    return out

def load_all():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,popq=practice_classes(pa['case_ids'])
    return (tj,ta,tact,tren),(pj,pa,pact,popq)

def screen(layer, T, Pr, k=64, min_sep=1.0):
    tj,ta,tact,tren=T; pj,pa,pact,_=Pr
    At=ta['mlp_dec'][:,layer]; Ap=pa['mlp_dec'][:,layer]
    st=stats(At,tact!='call',tact=='call',tact=='clarify',tact=='answer'); sp=stats(Ap,pact!='call',pact=='call',pact=='clarify',pact=='answer')
    pre_t=ta['mlp_pre'][:,layer] if 'mlp_pre' in ta else np.zeros((0,At.shape[1])); pre_p=pa['mlp_pre'][:,layer] if 'mlp_pre' in pa else np.zeros((0,Ap.shape[1]))
    maxabs_pre=np.abs(np.concatenate([pre_t,pre_p],0)).max(0)
    # name invariance: correlation between original and renamed activation vectors over (group x class)
    o=At[~tren]; r=At[tren]; oc=o-o.mean(0); rc=r-r.mean(0); name_corr=(oc*rc).sum(0)/np.sqrt((oc*oc).sum(0)*(rc*rc).sum(0)+1e-12)
    # signed separation consistent across sets
    sign=np.sign(st['d_non']); consistent=(np.sign(sp['d_non'])==sign)
    sep=np.minimum(np.abs(st['d_non']),np.abs(sp['d_non']))*consistent
    # quiescence: activation magnitude on calls (decision + prefix) relative to non-call mean magnitude
    quiet=np.maximum.reduce([st['maxabs_call'],sp['maxabs_call'],maxabs_pre])
    drive=np.minimum(np.abs(st['mean_non']),np.abs(sp['mean_non']))
    ratio=drive/(quiet+1e-6)
    score=sep*np.minimum(ratio,3.0)*(name_corr>0.5)
    order=np.argsort(score)[::-1][:k]
    cols=[int(i) for i in order if score[i]>0 and sep[i]>=min_sep]
    return {'layer':layer,'cols':cols,'score':score,'sep':sep,'ratio':ratio,'quiet':quiet,'drive':drive,'name_corr':name_corr,'st':st,'sp':sp,'sign':sign}

def feature(cols, v, A): return A[:,cols]@v

def evaluate(layer, cols, v, T, Pr):
    tj,ta,tact,tren=T; pj,pa,pact,_=Pr
    ft=feature(cols,v,ta['mlp_dec'][:,layer]); fp=feature(cols,v,pa['mlp_dec'][:,layer])
    res={}
    for name,f,act in (('teacher',ft,tact),('practice',fp,pact)):
        res[name]={'call_max':float(f[act=='call'].max()),'call_min':float(f[act=='call'].min()),'call_mean':float(f[act=='call'].mean()),
                   'answer_min':float(f[act=='answer'].min()),'answer_mean':float(f[act=='answer'].mean()),'clarify_min':float(f[act=='clarify'].min()),'clarify_mean':float(f[act=='clarify'].mean()),
                   'noncall_min':float(f[act!='call'].min()),'gap_noncallmin_minus_callmax':float(f[act!='call'].min()-f[act=='call'].max())}
    pre=[]
    for a in (ta,pa):
        if 'mlp_pre' in a: pre.append(feature(cols,v,a['mlp_pre'][:,layer]))
    if pre: pre=np.concatenate(pre); res['prefix']={'max':float(pre.max()),'min':float(pre.min()),'maxabs':float(np.abs(pre).max())}
    return res,ft,fp

if __name__=='__main__':
    T,Pr=load_all()
    summary=[]
    for layer in range(24):
        s=screen(layer,T,Pr)
        cols=s['cols']
        if len(cols)<4: summary.append({'layer':layer,'n':len(cols)}); continue
        v=(s['st']['mean_non'][cols]-s['st']['mean_call'][cols]); v/=np.linalg.norm(v)
        res,_,_=evaluate(layer,cols,v,T,Pr)
        summary.append({'layer':layer,'n':len(cols),'teacher_gap':round(res['teacher']['gap_noncallmin_minus_callmax'],3),'practice_gap':round(res['practice']['gap_noncallmin_minus_callmax'],3),
                        'practice_call_max':round(res['practice']['call_max'],3),'practice_noncall_min':round(res['practice']['noncall_min'],3),'practice_noncall_mean':round((res['practice']['answer_mean']+res['practice']['clarify_mean'])/2,3),
                        'prefix_maxabs':round(res.get('prefix',{}).get('maxabs',float('nan')),3),'top_scores':[round(float(x),2) for x in np.sort(s['score'])[::-1][:5]]})
    for row in summary: print(row)
