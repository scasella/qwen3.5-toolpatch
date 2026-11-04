"""(a) Transfer: fit u (end-only LDA) and g (clause-end gate) on D1+D2 only; evaluate the combination on D3 ambiguous ends and on harmful positions
(NOTPROV/clarify items excluded: a marker there pushes toward the gold NEED_CLARIFICATION). (b) Per-group worst positions with token text. (c) Constant dims of the
layer-22/23 MLP inputs (an exact bias source for thresholds)."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, unit, ROOT
from runtime.common import QWEN6, BASE
from analysis.stage1 import lda_dir
from analysis.stage1e import harmful_sets
rng=np.random.default_rng(3)

def fit_dirs(caps):
    ends=[];var=[];negs=[]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    from benchmark.render import prompt as render
    import fp2
    for c in caps:
        for i in c.ids:
            X=c.full(i,'mlpin22'); ro=c.roles(i); ends.append(X[ro['end']]); var.append(c.meta[i]['variant'])
            it=fp2.items_for(c.set,[i])[0]; p=render(tok,it['input'],'standard',False); ids=tok.encode(p,add_special_tokens=False); toks=[tok.decode([t]) for t in ids]
            ustart=ids.index(tok.encode('<|im_start|>',add_special_tokens=False)[0],1); idx=np.arange(len(X)); dot=np.array([('.' in t) for t in toks])
            ok=(idx<=ustart)|((idx>ustart)&(idx<=ro['imend'])&~dot)|(idx>ro['imend']); ok[ro['end']]=False
            negs.append(X[rng.choice(idx[ok],60,replace=False)])
    ends=np.stack(ends).astype(np.float64); var=np.array(var); negs=np.concatenate(negs).astype(np.float64)
    u=lda_dir(ends[var=='ambiguous'],ends[var!='ambiguous'],0.5); g=lda_dir(ends,negs,0.5); return u,g

def main():
    pos,posfam,harm=harmful_sets()
    harm={k:v for k,v in harm.items() if not (k[0]=='q6' and k[1]=='clarify')}
    capsA=[Capture(l) for l in ('d01b_triples_D1','d02_triples_D2')]; capD3=Capture('d03_triples_D3')
    uA,gA=fit_dirs(capsA); D=np.load(ROOT/'analysis/stage1_dirs.npz'); uF,gF=D['u'],D['g']
    d3_amb=np.stack([capD3.full(i,'mlpin22')[capD3.roles(i)['end']] for i in capD3.ids if capD3.meta[i]['variant']=='ambiguous']).astype(np.float64)
    d12_amb=np.stack([c.full(i,'mlpin22')[c.roles(i)['end']] for c in capsA for i in c.ids if c.meta[i]['variant']=='ambiguous']).astype(np.float64)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    for tag,(u,g) in (('fit D1+D2',(uA,gA)),('fit D1+D2+D3',(uF,gF))):
        for a in (0.4,0.5,0.6,0.75):
            d=unit(a*g+u); worst=[]
            for k,v in harm.items():
                for X in v: worst.append((float((X@d).max()),k,int(np.argmax(X@d))))
            worst.sort(reverse=True); hm=worst[0][0]
            print(f"{tag:14s} a={a}: amb ends D1+D2 min {(d12_amb@d).min():.2f} | D3 min {(d3_amb@d).min():.2f} 5th {np.percentile(d3_amb@d,5):.2f} | harmful max {hm:.2f} (margin vs D3 {(d3_amb@d).min()-hm:.2f}) | worst {[(round(m,2),k[0],k[1],k[2]) for m,k,_ in worst[:5]]}")
    # constant dims
    for name in ('mlpin22','mlpin23'):
        X=np.concatenate([capD3.full(i,name) for i in capD3.ids[:20]]).astype(np.float64); mu=X.mean(0); sd=X.std(0); ratio=np.abs(mu)/(sd+1e-9); o=np.argsort(ratio)[::-1][:6]
        print(f'{name}: most constant dims (idx, mean, sd, |mean|/sd):',[(int(d),round(float(mu[d]),2),round(float(sd[d]),3),round(float(ratio[d]),1)) for d in o],'| mean |x|',round(float(np.linalg.norm(X,axis=1).mean()),1))
if __name__=='__main__': main()
