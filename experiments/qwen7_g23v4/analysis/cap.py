"""Load an fp2 teacher capture (runs/<label>.json + .npz) with its data/<set>.meta.json into per-item records. CPU only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FULL_ATTN=(3,7,11,15,19,23)
TOK={'call':248058,'NO':8725,'NE':3811}

class Capture:
    def __init__(self,label,set_name=None):
        self.label=label; self.js=json.loads((ROOT/'runs'/f'{label}.json').read_text()); set_name=set_name or self.js['set']; self.set=set_name
        mp=ROOT/f'data/{set_name}.meta.json'; self.meta=json.loads(mp.read_text()) if mp.exists() else {}
        self.z=np.load(ROOT/'runs'/f'{label}.npz',allow_pickle=False)
        self.rows={r['case_id']:r for r in self.js['rows']}; self.ids=[r['case_id'] for r in self.js['rows']]
        self.variant=np.array([self.meta.get(i,{}).get('variant','-') for i in self.ids]); self.family=np.array([self.meta.get(i,{}).get('family','-') for i in self.ids])
        self.phrasing=np.array([self.meta.get(i,{}).get('phrasing','-') for i in self.ids]); self.existing=np.array([bool(self.meta.get(i,{}).get('existing')) for i in self.ids])
    def roles(self,cid): return self.rows[cid]['roles']
    def pos_index(self,cid,role):
        p=self.rows[cid]['roles'].get(role)
        if p is None: return None
        pl=self.z[f'pos_{cid}']; return int(np.where(pl==p)[0][0])
    def h(self,cid,role,layer):
        i=self.pos_index(cid,role); return None if i is None else self.z[f'h_{cid}'][i,layer].astype(np.float32)
    def attn(self,cid,role,layer):
        i=self.pos_index(cid,role); return None if i is None else self.z[f'attn_{cid}'][i,FULL_ATTN.index(layer)].astype(np.float32)
    def mlp23(self,cid,role):
        i=self.pos_index(cid,role); return None if i is None else self.z[f'mlp_{cid}'][i,0].astype(np.float32)
    def full(self,cid,name): return self.z[f'{name}_{cid}'].astype(np.float32)          # mlpin22 / h23 / mlpin23 / mlpin18 / h19 : (T,1024)
    def rows_attn(self,cid): return self.z[f'rows_{cid}'].astype(np.float32)             # (6,3,8,T)
    def logits(self,cid): return self.z[f'logits_{cid}'].astype(np.float32)
    def stack(self,role,layer,ids=None):
        ids=self.ids if ids is None else ids; out=[]; keep=[]
        for c in ids:
            v=self.h(c,role,layer)
            if v is not None: out.append(v); keep.append(c)
        return np.stack(out),keep

def unit(v): return v/np.linalg.norm(v)
def dprime(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64); s=np.sqrt((a.var()+b.var())/2+1e-12); return float((a.mean()-b.mean())/s)

def lofo(X,pos,fam,center=None):
    """Leave-one-family-out diff-of-means probe. X (n,d), pos bool (positive class), fam labels. Returns pooled held-out projections, d', threshold accuracy, gap."""
    X=np.asarray(X,np.float64); pos=np.asarray(pos); fam=np.asarray(fam); proj=np.zeros(len(X)); pred=np.zeros(len(X),bool)
    for f in np.unique(fam):
        tr=fam!=f; te=fam==f
        if pos[tr].sum()==0 or (~pos[tr]).sum()==0: continue
        d=unit(X[tr&pos].mean(0)-X[tr&~pos].mean(0)); p=X@d; thr=(p[tr&pos].mean()+p[tr&~pos].mean())/2
        proj[te]=p[te]; pred[te]=p[te]>thr
    a=proj[pos]; b=proj[~pos]
    return {'dprime':dprime(a,b),'acc':float((pred==pos).mean()),'gap':float(a.min()-b.max()),'n_pos':int(pos.sum()),'n_neg':int((~pos).sum()),'proj':proj}

def family_center(X,fam,variant):
    """Subtract, per family, the mean of that family's (call,resolved,ambiguous) triple (extra ambiguous items excluded from the mean, but centered)."""
    X=np.asarray(X,np.float64).copy()
    for f in np.unique(fam):
        m=fam==f; trip=m&np.isin(variant,['call','resolved','ambiguous'])
        X[m]-=X[trip].mean(0)
    return X
