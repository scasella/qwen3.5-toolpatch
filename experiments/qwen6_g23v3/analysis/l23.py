"""Final-layer (23) readout design from existing captures.
MLP input x = post_attention_layernorm(h_23 + o_proj(attn_in_23)) reconstructed on CPU; verified against h_24."""
from __future__ import annotations
import json, sys
import numpy as np
import primitives as P
from analysis.lens import load_run, classes, practice_classes, dprime, EPS
from build_data import load_cases

def silu(z): return z/(1+np.exp(-z))

def mlp_input(h23, attn_in, Wo, gpost):
    mid=h23+attn_in@Wo.T
    r=np.sqrt((mid*mid).mean(-1,keepdims=True)+EPS)
    return mid/r*gpost, mid

def main():
    L=23
    Wo=P.load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')      # (1024,2048)
    gpost=P.load_original(f'model.language_model.layers.{L}.post_attention_layernorm.weight')+1.0
    Wd=P.load_original(f'model.language_model.layers.{L}.mlp.down_proj.weight')           # (1024,3584)
    Wg=P.load_original(f'model.language_model.layers.{L}.mlp.gate_proj.weight'); Wu=P.load_original(f'model.language_model.layers.{L}.mlp.up_proj.weight')
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,_=practice_classes(pa['case_ids'])
    cases={c.id:c for c in load_cases('practice')}; amb=np.array([cases[str(c)].spec.ambiguous for c in pa['case_ids']])
    xt,midt=mlp_input(ta['h_dec'][:,23],ta['attn_dec'][:,5],Wo,gpost); xp,midp=mlp_input(pa['h_dec'][:,23],pa['attn_dec'][:,5],Wo,gpost)
    # verification: h24 == mid + Wd @ mlp_act ; and mlp_act == silu(Wg x)*(Wu x)
    rec=midt+ta['mlp_dec'][:,23]@Wd.T; err=np.abs(rec-ta['h_dec'][:,24]).max(); print('verify h24 = mid + down(mlp_act): max abs err',err, 'h24 scale',np.abs(ta['h_dec'][:,24]).mean())
    act=silu(xt@Wg.T)*(xt@Wu.T); err2=np.abs(act-ta['mlp_dec'][:,23]).max(); print('verify mlp_act = silu(gate x)*(up x): max abs err',err2,'act scale',np.abs(ta['mlp_dec'][:,23]).mean())
    # prefix inputs
    xtp,_=mlp_input(ta['h_pre'][:,23],ta['attn_pre'][:,5],Wo,gpost); xpp,_=mlp_input(pa['h_pre'][:,23],pa['attn_pre'][:,5],Wo,gpost)
    xpre=np.concatenate([xtp,xpp])
    print('x stats: ||x|| mean',np.linalg.norm(xt,axis=1).mean(), 'cos between decision inputs of two random cases', float(xt[0]@xt[7]/np.linalg.norm(xt[0])/np.linalg.norm(xt[7])))
    out={}
    for feat,pos in (('A','answer'),('B','clarify')):
        xc=np.concatenate([xt[tact=='call'],xp[pact=='call']]); m_call=xc.mean(0)
        d=xt[tact==pos].mean(0)-xt[tact=='call'].mean(0)
        d_o=d-(d@m_call)/(m_call@m_call)*m_call; d_o/=np.linalg.norm(d_o); d_n=d/np.linalg.norm(d)
        for nm,dd in (('raw',d_n),('ortho_callmean',d_o)):
            ft=xt@dd; fp=xp@dd; fpre=xpre@dd
            print(f'\n=== feature {feat} ({pos} vs call), direction {nm} ===')
            for setname,f,a in (('teacher',ft,tact),('practice',fp,pact)):
                s=f'  {setname}: call [{f[a=="call"].min():6.2f},{f[a=="call"].max():6.2f}] mean {f[a=="call"].mean():6.2f} | answer [{f[a=="answer"].min():6.2f},{f[a=="answer"].max():6.2f}] | clarify [{f[a=="clarify"].min():6.2f},{f[a=="clarify"].max():6.2f}]'
                if setname=='practice': s+=f' | NOTPROV [{f[(a=="clarify")&~amb].min():6.2f},{f[(a=="clarify")&~amb].max():6.2f}] AMBIG [{f[(a=="clarify")&amb].min():6.2f},{f[(a=="clarify")&amb].max():6.2f}]'
                print(s)
            print(f'  prefix (calls): [{fpre.min():6.2f},{fpre.max():6.2f}]')
            out[f'{feat}_{nm}']=dd.tolist()
    # near-dead neurons at layer 23 from decision+prefix activations
    A=np.concatenate([ta['mlp_dec'][:,23],pa['mlp_dec'][:,23],ta['mlp_pre'][:,23],pa['mlp_pre'][:,23]])
    mx=np.abs(A).max(0); coln=np.linalg.norm(Wd,axis=0); gn=np.linalg.norm(Wg,axis=1); un=np.linalg.norm(Wu,axis=1)
    order=np.argsort(mx)[:20]
    print('\nquietest layer-23 neurons (max|act| over decision+prefix captures):',[(int(i),round(float(mx[i]),4),'|down|=%.3f'%coln[i]) for i in order])
    print('down_proj column norm stats: median',np.median(coln),'min',coln.min(),'| gate row norm median',np.median(gn),'| up row norm median',np.median(un))
    json.dump(out,open('analysis/l23_directions.json','w'))
if __name__=='__main__': main()
