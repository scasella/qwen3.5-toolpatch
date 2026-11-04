"""Exact CPU simulation of a two-neuron final-layer (23) MLP edit on captured decision/prefix positions.
n1 (abstain): gate=k*s, up=k*s, down=a1*u_AB ; n2 (route): gate=k*s, up=kr*r, down=a2*(u_NE-u_NO).
Neurons j1,j2 are re-purposed (their original rows/columns replaced); their original contribution is removed in the simulation."""
from __future__ import annotations
import json, sys, itertools
import numpy as np
import primitives as P
from analysis.lens import load_run, classes, practice_classes, embed, gamma_final, EPS, TOK
from analysis.l23 import mlp_input, silu
from build_data import load_cases
L=23
NAMES=lambda s:f'model.language_model.layers.{L}.{s}.weight'

def load_all():
    Wo=P.load_original(NAMES('self_attn.o_proj')); gpost=P.load_original(f'model.language_model.layers.{L}.post_attention_layernorm.weight')+1.0
    Wd=P.load_original(NAMES('mlp.down_proj')); Wg=P.load_original(NAMES('mlp.gate_proj')); Wu=P.load_original(NAMES('mlp.up_proj'))
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,_=practice_classes(pa['case_ids'])
    cases={c.id:c for c in load_cases('practice')}; amb=np.array([cases[str(c)].spec.ambiguous for c in pa['case_ids']])
    xt,_=mlp_input(ta['h_dec'][:,23],ta['attn_dec'][:,5],Wo,gpost); xp,_=mlp_input(pa['h_dec'][:,23],pa['attn_dec'][:,5],Wo,gpost)
    xtp,_=mlp_input(ta['h_pre'][:,23],ta['attn_pre'][:,5],Wo,gpost); xpp,_=mlp_input(pa['h_pre'][:,23],pa['attn_pre'][:,5],Wo,gpost)
    D=dict(x=np.concatenate([xt,xp]),h24=np.concatenate([ta['h_dec'][:,24],pa['h_dec'][:,24]]),act=np.concatenate([tact,pact]),
           src=np.array(['T']*len(tact)+['P']*len(pact)),ids=np.concatenate([ta['case_ids'],pa['case_ids']]),
           sub=np.array([('AMBIG' if (a=='clarify' and i>=len(tact) and amb[i-len(tact)]) else ('NOTPROV' if a=='clarify' else a)) for i,a in enumerate(np.concatenate([tact,pact]))]),
           xpre=np.concatenate([xtp,xpp]),h24pre=np.concatenate([ta['h_pre'][:,24],pa['h_pre'][:,24]]),
           pre_gold=np.array([r['prefix']['gold']['id'] for r in tj['rows']+pj['rows'] if 'prefix' in r]),
           gold_first=np.array([r['decision']['gold']['id'] for r in tj['rows']+pj['rows']]))
    D['teacher_h']={l:ta['h_dec'][:,l] for l in range(25)}; D['tact']=tact
    return D,dict(Wo=Wo,gpost=gpost,Wd=Wd,Wg=Wg,Wu=Wu)

def unit(v): return v/np.linalg.norm(v)
def ortho_to(v,ref): return v-(v@ref)/(ref@ref)*ref

def directions(D,src_s=14,src_r=14):
    x=D['x']; act=D['act']; sub=D['sub']; tact=D['tact']; nT=len(tact)
    H=D['teacher_h'][src_s]; s=unit(H[tact!='call'].mean(0)-H[tact=='call'].mean(0))
    Hr=D['teacher_h'][src_r]; r=unit(Hr[tact=='clarify'].mean(0)-Hr[tact=='answer'].mean(0))
    # place the s boundary at the midpoint between the call mean and the abstain mean (practice+teacher, excluding AMBIG), by orthogonalizing against that reference
    callm=x[act=='call'].mean(0); absm=x[(act!='call')&(sub!='AMBIG')].mean(0)
    s=unit(ortho_to(s,(callm+absm)/2))
    ansm=x[act=='answer'].mean(0); npm=x[sub=='NOTPROV'].mean(0)
    r=unit(ortho_to(r,(ansm+npm)/2))
    return s,r

def simulate(D,W,s,r,k,kr,a1,a2,j1,j2,u_AB,u_route,E,g,which='dec'):
    x=D['x'] if which=='dec' else D['xpre']; h=D['h24'] if which=='dec' else D['h24pre']
    fs=x@s; fr=x@r
    a_1=silu(k*fs)*(k*fs); a_2=silu(k*fs)*(kr*fr)
    # original contributions of replaced neurons
    orig=np.zeros_like(h)
    for j in (j1,j2):
        aj=silu(x@W['Wg'][j])*(x@W['Wu'][j]); orig+=aj[:,None]*W['Wd'][:,j][None,:]
    hn=h-orig+a1*a_1[:,None]*u_AB[None,:]+a2*a_2[:,None]*u_route[None,:]
    xn=hn/np.sqrt((hn*hn).mean(-1,keepdims=True)+EPS)*g
    return xn@E.T,fs,fr,a_1,a_2

def report(D,logits,fs,fr,tag,E,gold_first=None):
    act=D['act']; sub=D['sub']; src=D['src']; out={}
    am=logits.argmax(1)
    for cls in ('call','answer','NOTPROV','AMBIG'):
        for sr in ('T','P'):
            m=(sub==cls)&(src==sr)
            if not m.any(): continue
            want={'call':TOK['call'],'answer':TOK['NO'],'NOTPROV':TOK['NE'],'AMBIG':TOK['NE']}[cls]
            ok=int((am[m]==want).sum()); n=int(m.sum())
            mNO=logits[m,TOK['NO']]-logits[m,TOK['call']]; mNE=logits[m,TOK['NE']]-logits[m,TOK['call']]; mR=logits[m,TOK['NE']]-logits[m,TOK['NO']]
            out[f'{cls}/{sr}']=f'{ok}/{n} NO-call[{mNO.min():.1f},{mNO.max():.1f}] NE-call[{mNE.min():.1f},{mNE.max():.1f}] NE-NO[{mR.min():.1f},{mR.max():.1f}]'
    return out

if __name__=='__main__':
    D,W=load_all(); E=embed(); g=gamma_final()
    U=np.load('analysis/u_directions.npz'); u_AB=U['uAB']; u_route=U['uB']-U['uA']
    j1,j2=2089,3468
    # baseline check
    x=D['x']; h=D['h24']; base=(h/np.sqrt((h*h).mean(-1,keepdims=True)+EPS)*g)@E.T
    print('baseline:',report(D,base,None,None,'base',E))
    for src_s in (14,16,18):
        s,r=directions(D,src_s,14); fs=x@s; fr=x@r
        print(f'\n### s from L{src_s}: fs call[T] [{fs[(D["act"]=="call")&(D["src"]=="T")].min():.2f},{fs[(D["act"]=="call")&(D["src"]=="T")].max():.2f}] call[P] [{fs[(D["act"]=="call")&(D["src"]=="P")].min():.2f},{fs[(D["act"]=="call")&(D["src"]=="P")].max():.2f}] answer[P] [{fs[(D["act"]=="answer")&(D["src"]=="P")].min():.2f},{fs[(D["act"]=="answer")&(D["src"]=="P")].max():.2f}] NOTPROV [{fs[D["sub"]=="NOTPROV"].min():.2f},{fs[D["sub"]=="NOTPROV"].max():.2f}] cla[T] [{fs[(D["act"]=="clarify")&(D["src"]=="T")].min():.2f},{fs[(D["act"]=="clarify")&(D["src"]=="T")].max():.2f}] ans[T] [{fs[(D["act"]=="answer")&(D["src"]=="T")].min():.2f},{fs[(D["act"]=="answer")&(D["src"]=="T")].max():.2f}] | fr answer[P] [{fr[(D["act"]=="answer")&(D["src"]=="P")].min():.2f},{fr[(D["act"]=="answer")&(D["src"]=="P")].max():.2f}] NOTPROV [{fr[D["sub"]=="NOTPROV"].min():.2f},{fr[D["sub"]=="NOTPROV"].max():.2f}] cla[T] [{fr[(D["act"]=="clarify")&(D["src"]=="T")].min():.2f},{fr[(D["act"]=="clarify")&(D["src"]=="T")].max():.2f}] ans[T] [{fr[(D["act"]=="answer")&(D["src"]=="T")].min():.2f},{fr[(D["act"]=="answer")&(D["src"]=="T")].max():.2f}]')
        for k in (1.0,2.0):
            for a1 in (0.5,1.0,2.0):
                for a2 in (0.0,0.5,1.0):
                    lg,fs,fr,a_1,a_2=simulate(D,W,s,r,k,1.0,a1,a2,j1,j2,u_AB,u_route,E,g,'dec')
                    rep=report(D,lg,fs,fr,'',E)
                    lgp,_,_,ap1,ap2=simulate(D,W,s,r,k,1.0,a1,a2,j1,j2,u_AB,u_route,E,g,'pre')
                    pre_ok=int((lgp.argmax(1)==D['pre_gold']).sum())
                    print(f'k={k} a1={a1} a2={a2} | prefix gold argmax {pre_ok}/{len(D["pre_gold"])} max|a1|pre {np.abs(ap1).max():.2f} | '+' | '.join(f'{kk}: {v}' for kk,v in rep.items()))
