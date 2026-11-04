"""Pooled probes (several captures), family-subspace masking (Hypothesis A proper), shrinkage-LDA ceiling, cross-set transfer, and the
layer-15 transport gain for a hypothetical stage-1 write at the clause roles. Usage: python -m analysis.diag2 d01b_triples_D1 d02_triples_D2 [d03_triples_D3]"""
from __future__ import annotations
import sys, json
import numpy as np
from analysis.cap import Capture, lofo, family_center, dprime, unit, FULL_ATTN, ROOT
import primitives as P

def gather(caps,role,layer,which='h'):
    X=[];var=[];fam=[];cid=[];src=[]
    for c in caps:
        for i in c.ids:
            if which=='h': v=c.h(i,role,layer)
            else:
                p=c.roles(i).get(role); v=None if p is None else c.full(i,which)[p]
            if v is None: continue
            X.append(v); var.append(c.meta[i]['variant']); fam.append(c.meta[i]['family']); cid.append(i); src.append(c.label)
    return np.stack(X).astype(np.float64),np.array(var),np.array(fam),np.array(cid),np.array(src)

def lofo_masked(X,pos,var,fam,k):
    """Project out the top-k principal directions of the training families' {call,resolved} means before the diff-of-means probe (fixed linear map; held-out family excluded)."""
    proj=np.zeros(len(X)); pred=np.zeros(len(X),bool)
    for f in np.unique(fam):
        tr=fam!=f; te=fam==f
        fams=[g for g in np.unique(fam[tr])]
        M=np.stack([X[tr&(fam==g)&(var!='ambiguous')].mean(0) for g in fams]); mu=M.mean(0); U,S,Vt=np.linalg.svd(M-mu,full_matrices=False)
        B=Vt[:k]                                   # family subspace (k x d)
        Pm=lambda Z:Z-(Z@B.T)@B
        Xt=Pm(X-mu)
        d=unit(Xt[tr&pos].mean(0)-Xt[tr&~pos].mean(0)); p=Xt@d; thr=(p[tr&pos].mean()+p[tr&~pos].mean())/2
        proj[te]=p[te]; pred[te]=p[te]>thr
    a=proj[pos]; b=proj[~pos]; return {'dprime':dprime(a,b),'acc':float((pred==pos).mean()),'gap':float(a.min()-b.max())}

def lofo_lda(X,pos,fam,shrink=0.5):
    proj=np.zeros(len(X)); pred=np.zeros(len(X),bool)
    for f in np.unique(fam):
        tr=fam!=f; te=fam==f
        Xa=X[tr&pos]; Xb=X[tr&~pos]; mu_a=Xa.mean(0); mu_b=Xb.mean(0)
        R=np.concatenate([Xa-mu_a,Xb-mu_b]); S=R.T@R/len(R); S=(1-shrink)*S+shrink*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
        w=np.linalg.solve(S,mu_a-mu_b); w/=np.linalg.norm(w); p=X@w; thr=(p[tr&pos].mean()+p[tr&~pos].mean())/2
        proj[te]=p[te]; pred[te]=p[te]>thr
    a=proj[pos]; b=proj[~pos]; return {'dprime':dprime(a,b),'acc':float((pred==pos).mean()),'gap':float(a.min()-b.max())}

def main():
    labels=sys.argv[1:]; caps=[Capture(l) for l in labels]; out={'labels':labels}
    n=sum(len(c.ids) for c in caps); print(f'=== pooled {labels}: {n} items')
    print('\n=== decision position: raw LOFO | family-subspace masked (k=4,8,16) | shrinkage LDA (0.5, 0.2) | ambiguous vs {call,resolved}')
    print(' layer | raw d\' acc gap | mask4 | mask8 | mask16 | lda.5 d\' acc gap | lda.2 d\' acc gap')
    res={}
    for l in [12,14,16,18,20,22,23,'x23','x22']:
        if l=='x23': X,var,fam,cid,src=gather(caps,'dec',23,'mlpin23')
        elif l=='x22': X,var,fam,cid,src=gather(caps,'dec',22,'mlpin22')
        else: X,var,fam,cid,src=gather(caps,'dec',l)
        pos=var=='ambiguous'; r=lofo(X,pos,fam); m4=lofo_masked(X,pos,var,fam,4); m8=lofo_masked(X,pos,var,fam,8); m16=lofo_masked(X,pos,var,fam,16); l5=lofo_lda(X,pos,fam,0.5); l2=lofo_lda(X,pos,fam,0.2)
        res[str(l)]={'raw':{k:v for k,v in r.items() if k!='proj'},'mask4':m4,'mask8':m8,'mask16':m16,'lda05':l5,'lda02':l2}
        print(f" {str(l):>5} | {r['dprime']:5.2f} {r['acc']:.2f} {r['gap']:6.2f} | {m4['dprime']:5.2f} | {m8['dprime']:5.2f} | {m16['dprime']:5.2f} | {l5['dprime']:5.2f} {l5['acc']:.2f} {l5['gap']:6.2f} | {l2['dprime']:5.2f} {l2['acc']:.2f} {l2['gap']:6.2f}")
    out['decision']=res
    print('\n=== clause-end token (.) and connector: same probes, ambiguous vs {call,resolved} at end; ambiguous vs resolved at conn/v2')
    res2={}
    for role,neg in (('end',None),('conn','resolved'),('v2','resolved'),('tail',None)):
        for l in [10,12,14,15,16,18,20,22,23]:
            X,var,fam,cid,src=gather(caps,role,l)
            if neg: keep=np.isin(var,['ambiguous',neg]); X,var,fam=X[keep],var[keep],fam[keep]
            pos=var=='ambiguous'
            if role=='tail' and (~pos).sum()<5: continue
            r=lofo(X,pos,fam); l5=lofo_lda(X,pos,fam,0.5)
            res2[f'{role}:{l}']={'raw':{k:v for k,v in r.items() if k!='proj'},'lda05':l5}
            print(f" {role:>4} L{l:2d} | raw d' {r['dprime']:5.2f} acc {r['acc']:.2f} gap {r['gap']:6.2f} | lda.5 d' {l5['dprime']:5.2f} acc {l5['acc']:.2f} gap {l5['gap']:6.2f}  (n {int(pos.sum())}/{int((~pos).sum())})")
    out['roles']=res2
    # ---------------------------------------------------------------- cross-set transfer: fit on first capture, test raw on the others
    if len(caps)>1:
        print('\n=== transfer: diff-of-means direction fit on',labels[0],'(all its families), projected on the other captures (raw, no refit); d\' and gap, decision x23 and end token L15/L23')
        tr=caps[:1]; te=caps[1:]
        for role,l,which in (('dec',23,'mlpin23'),('dec',22,'h'),('end',15,'h'),('end',23,'h'),('dec',16,'h')):
            Xa,va,fa,_,_=gather(tr,role,l,which if which!='h' else 'h'); Xb,vb,fb,_,_=gather(te,role,l,which if which!='h' else 'h')
            d=unit(Xa[va=='ambiguous'].mean(0)-Xa[va!='ambiguous'].mean(0)); pa=Xa@d; pb=Xb@d
            thr=(pa[va=='ambiguous'].mean()+pa[va!='ambiguous'].mean())/2
            print(f" {role} L{l} {which}: train d' {dprime(pa[va=='ambiguous'],pa[va!='ambiguous']):.2f} | test d' {dprime(pb[vb=='ambiguous'],pb[vb!='ambiguous']):.2f} acc {((pb>thr)==(vb=='ambiguous')).mean():.2f} gap {pb[vb=='ambiguous'].min()-pb[vb!='ambiguous'].max():.2f} | test amb vs call {dprime(pb[vb=='ambiguous'],pb[vb=='call']):.2f} amb vs res {dprime(pb[vb=='ambiguous'],pb[vb=='resolved']):.2f}")
    # ---------------------------------------------------------------- transport gain through layer-15 heads: delivered residual per unit write at a clause role
    print('\n=== layer-15 transport: for a unit write delta at a clause role position (layer-14 output), delivered = alpha_h * sig(g_h) * Wo_h Wv_kv diag(gamma15) delta / rms15 ; report top singular gain of M_h and mean alpha per role by class')
    L=15; Wo=P.load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); Wv=P.load_original(f'model.language_model.layers.{L}.self_attn.v_proj.weight'); g15=P.load_original(f'model.language_model.layers.{L}.input_layernorm.weight')+1.0
    gains={}
    for h in range(8):
        M=Wo[:,h*256:(h+1)*256]@Wv[(h//4)*256:(h//4+1)*256,:]*g15[None,:]
        s=np.linalg.svd(M,compute_uv=False); gains[h]={'sigma_max':float(s[0]),'sigma_median':float(np.median(s)),'rank_eff':int((s>s[0]*0.1).sum())}
    print(' OV gains per head (sigma_max / median):',{h:(round(v['sigma_max'],3),round(v['sigma_median'],3)) for h,v in gains.items()})
    rms15={}
    for c in caps:
        for i in c.ids:
            for role in ('end','conn','v2','tail','imend'):
                v=c.h(i,role,15)
                if v is not None: rms15.setdefault(role,[]).append(float(np.sqrt((v*v).mean())))
    print(' residual RMS at layer-15 input by role:',{k:round(float(np.mean(v)),4) for k,v in rms15.items()})
    alpha={}
    for c in caps:
        for i in c.ids:
            rows=c.rows_attn(i)[FULL_ATTN.index(15),0]; ro=c.roles(i); v=c.meta[i]['variant']
            for role in ('end','conn','v2','tail','imend'):
                p=ro.get(role)
                if p is None: continue
                for h in range(8): alpha.setdefault((role,h,v),[]).append(float(rows[h,p]))
    for role in ('end','conn','v2','tail'):
        print(f' alpha L15 at {role:4s}: '+' '.join(f"h{h}:{np.mean(alpha.get((role,h,'ambiguous'),[0])):.3f}/{np.mean(alpha.get((role,h,'resolved'),[0])):.3f}/{np.mean(alpha.get((role,h,'call'),[0])):.3f}" for h in range(8)))
    out['transport']={'gains':gains,'rms15':{k:float(np.mean(v)) for k,v in rms15.items()}}
    json.dump(out,open(ROOT/'analysis'/f'diag2_{"_".join(labels)}.json','w'),indent=1,default=float)
if __name__=='__main__': main()
