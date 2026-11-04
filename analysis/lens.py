"""CPU analysis utilities: logit lens through final RMSNorm + tied unembedding, per-neuron statistics, direct logit attribution.
No model forwards; operates on saved captures from fp.py teacher passes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import primitives as P
ROOT=Path(__file__).resolve().parents[1]
EPS=1e-6
TOK={'call':248058,'NO':8725,'NE':3811,'I':40,'To':1206}
_cache={}

def load_run(label):
    js=json.loads((ROOT/'runs'/f'{label}.json').read_text())
    with np.load(ROOT/'runs'/f'{label}.npz',allow_pickle=False) as z: arr={n:z[n] for n in z.files}
    return js,arr

def embed():
    if 'E' not in _cache: _cache['E']=P.load_original('model.language_model.embed_tokens.weight')
    return _cache['E']

def gamma_final():
    if 'g' not in _cache: _cache['g']=P.load_original('model.language_model.norm.weight')+1.0  # MLX convention: stored weight + 1
    return _cache['g']

def final_norm(h):
    """h: (...,1024) pre-final-norm residual -> normalized, scaled vector."""
    h=np.asarray(h,dtype=np.float32); r=np.sqrt((h*h).mean(-1,keepdims=True)+EPS)
    return h/r*gamma_final()

def rms(h): h=np.asarray(h,dtype=np.float32); return np.sqrt((h*h).mean(-1)+EPS)

def lens(h,tokens=None):
    """virtual logits of residual h through final norm + unembedding; tokens None = full vocab."""
    x=final_norm(h); E=embed()
    return x@E.T if tokens is None else x@E[np.asarray(tokens)].T

def classes(case_ids):
    """teacher/practice ids -> action labels + renamed flags."""
    acts=[];ren=[]
    for cid in case_ids:
        cid=str(cid); ren.append('renamed' in cid)
        acts.append('call' if '-call' in cid else 'answer' if '-answer' in cid else 'clarify' if '-clarify' in cid else None)
    return np.array(acts),np.array(ren)

def practice_classes(case_ids):
    from build_data import load_cases
    m={c.id:c for c in load_cases('practice')}
    return np.array([m[str(c)].spec.action for c in case_ids]),np.array([m[str(c)].opaque for c in case_ids])

def dprime(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    s=np.sqrt((a.var(0)+b.var(0))/2+1e-12); return (a.mean(0)-b.mean(0))/s

def verify_lens(js,arr,n=5):
    """reconstruct decision logits from h_dec[:,24] and compare with saved logits (fp16)."""
    out=[]
    for i in range(min(n,len(js['rows']))):
        h=arr['h_dec'][i,24]; toks=list(TOK.values()); pred=lens(h,toks)
        saved=arr['logits_dec'][i].astype(np.float32)[toks]
        out.append({'case':js['rows'][i]['case_id'],'pred':[round(float(x),2) for x in pred],'saved':[round(float(x),2) for x in saved],'max_abs_err':float(np.abs(pred-saved).max())})
    return out
