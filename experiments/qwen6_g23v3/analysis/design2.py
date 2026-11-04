"""Design G2: two re-purposed layer-23 neurons with a shared decision-position gate.
n1: gate = w (position selector), up = s (signed abstain feature), down = a1*u_AB
n2: gate = w,                      up = r (signed route feature),   down = a2*(u_NE-u_NO)
Exact CPU simulation at decision/prefix positions; leakage at continuation positions from all-position captures."""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.simulate import load_all, directions, silu, report, EPS, TOK, embed, gamma_final
from analysis.lens import load_run
import primitives as P

def position_selector():
    """w = unit(mean x at decision positions - mean x at continuation positions), from teacher+practice all-position captures."""
    dec=[];cont=[];prompt_max=[];sets={}
    for lab in ('p05_teacher_allpos','p04c_practice_allpos'):
        js,arr=load_run(lab); xs=[]
        for r in js['rows']:
            x=arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32); d=r['decision_position']
            dec.append(x[d]); cont.append(x[d+1:]); xs.append((x,d,r))
        sets[lab]=xs
    dec=np.stack(dec); cont=np.concatenate(cont)
    w=dec.mean(0)-cont.mean(0); w/=np.linalg.norm(w)
    stats={}
    for lab,xs in sets.items():
        pd=[];pc=[];pp=[]
        for x,d,r in xs:
            f=x@w; pd.append(f[d]); pc.append(f[d+1:]); pp.append(f[:d])
        pc=np.concatenate(pc); pp=np.concatenate(pp); pd=np.array(pd)
        stats[lab]={'dec_min':float(pd.min()),'dec_max':float(pd.max()),'cont_max':float(pc.max()),'cont_min':float(pc.min()),'cont_p99':float(np.percentile(pc,99)),'prompt_max':float(pp.max()),'prompt_min':float(pp.min())}
    return w,stats,sets

def simulate_g2(D,W,w,s,r,a1,a2,j1,j2,u_AB,u_route,E,g,which='dec',nu=1.0,kappa=1.0,kr=1.0):
    x=D['x'] if which=='dec' else D['xpre']; h=D['h24'] if which=='dec' else D['h24pre']
    gate=silu(nu*(x@w)); fs=x@s; fr=x@r
    act1=gate*(kappa*fs); act2=gate*(kr*fr)
    orig=np.zeros_like(h)
    for j in (j1,j2):
        aj=silu(x@W['Wg'][j])*(x@W['Wu'][j]); orig+=aj[:,None]*W['Wd'][:,j][None,:]
    hn=h-orig+a1*act1[:,None]*u_AB[None,:]+a2*act2[:,None]*u_route[None,:]
    xn=hn/np.sqrt((hn*hn).mean(-1,keepdims=True)+EPS)*g
    return xn@E.T,fs,fr,act1,act2,gate

if __name__=='__main__':
    D,W=load_all(); E=embed(); g=gamma_final()
    U=np.load('analysis/u_directions.npz'); u_AB=U['uAB']; u_route=U['uB']-U['uA']
    w,stats,sets=position_selector(); print('position selector w stats:',json.dumps(stats,indent=None)); np.save('analysis/w_position.npy',w)
    j1,j2=3468,3421
    for src_s in (16,18):
        s,r=directions(D,src_s,14)
        print(f'\n### G2 with s from L{src_s}, r from L14, w gate; neurons {j1},{j2}')
        for a1 in (0.1,0.2,0.3,0.5):
            for a2 in (0.0,0.1,0.2,0.4):
                lg,fs,fr,act1,act2,gate=simulate_g2(D,W,w,s,r,a1,a2,j1,j2,u_AB,u_route,E,g,'dec')
                rep=report(D,lg,fs,fr,'',E)
                lgp,_,_,ap1,ap2,gp=simulate_g2(D,W,w,s,r,a1,a2,j1,j2,u_AB,u_route,E,g,'pre')
                pre_ok=int((lgp.argmax(1)==D['pre_gold']).sum())
                am=lg.argmax(1); callm=D['act']=='call'
                marg=lg[:,TOK['call']]-np.where(np.arange(lg.shape[1])[None,:]==TOK['call'],-1e9,lg).max(1)
                print(f'a1={a1} a2={a2} | pre {pre_ok}/44 gate_pre[{gp.min():.2f},{gp.max():.2f}] | call min margin {marg[callm].min():.2f} | '+' | '.join(f'{k}: {v.split(" ")[0]}' for k,v in rep.items()))
        # detail for one config
        a1,a2=0.3,0.2
        lg,fs,fr,act1,act2,gate=simulate_g2(D,W,w,s,r,a1,a2,j1,j2,u_AB,u_route,E,g,'dec')
        print(f'  detail a1={a1} a2={a2}: gate at decision [{gate.min():.2f},{gate.max():.2f}] ; act1 answer/P [{act1[(D["sub"]=="answer")&(D["src"]=="P")].min():.1f},{act1[(D["sub"]=="answer")&(D["src"]=="P")].max():.1f}] NOTPROV [{act1[D["sub"]=="NOTPROV"].min():.1f},{act1[D["sub"]=="NOTPROV"].max():.1f}] call [{act1[D["act"]=="call"].min():.1f},{act1[D["act"]=="call"].max():.1f}] AMBIG [{act1[D["sub"]=="AMBIG"].min():.1f},{act1[D["sub"]=="AMBIG"].max():.1f}] ; act2 answer/P [{act2[(D["sub"]=="answer")&(D["src"]=="P")].min():.1f},{act2[(D["sub"]=="answer")&(D["src"]=="P")].max():.1f}] NOTPROV [{act2[D["sub"]=="NOTPROV"].min():.1f},{act2[D["sub"]=="NOTPROV"].max():.1f}] call [{act2[D["act"]=="call"].min():.1f},{act2[D["act"]=="call"].max():.1f}]')
        for k,v in report(D,lg,fs,fr,'',E).items(): print('   ',k,v)
        # continuation leakage with this config
        print('  continuation leakage (act1, act2 at continuation positions vs decision):')
        for lab,xs in sets.items():
            worst=[]
            for x,d,rr in xs:
                gt=silu(x@w); A1=gt*(x@s); A2=gt*(x@r)
                worst.append((rr['case_id'],rr['action'],float(A1[d]),float(np.abs(A1[d+1:]).max()),float(A2[d]),float(np.abs(A2[d+1:]).max()),float(np.abs(A1[:d]).max())))
            mx1=max(wv[3] for wv in worst); mx2=max(wv[5] for wv in worst)
            print(f'   {lab}: max |act1| over continuation positions {mx1:.2f} (decision act1 range [{min(wv[2] for wv in worst):.1f},{max(wv[2] for wv in worst):.1f}]); max |act2| cont {mx2:.2f}; max|act1| prompt {max(wv[6] for wv in worst):.1f}')
            print('     worst cont act1 cases:',sorted(worst,key=lambda t:-t[3])[:3])
