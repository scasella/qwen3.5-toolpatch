"""g23v4 = g23v3 + a two-stage ambiguity circuit. Closed form only (difference of means, shrinkage LDA, pseudo-inverse, orthogonalization); no gradients.

Stage 1 (layer 22, MLP neuron J0=2525, re-purposed; its original rows/column are replaced):
    gate[J0] = k1 * w1g,  up[J0] = k1 * w1g  (quadratic),  down[:,J0] = b * p6
    w1g = unit(a_g*g + u) - theta1*r1 : u = shrinkage-LDA direction 'ambiguous clause end vs call/resolved clause end' at the layer-22 MLP input,
          g = shrinkage-LDA 'user fact-clause end vs other positions', r1 = constant-projection direction (r1.x ~ 1 on clause ends and on the hardest
          negative positions; C^-1 mu / mu^T C^-1 mu with shrinkage) so that w1g.x ~ w1.x - theta1 (a threshold without a bias term)
    p6  = unit( pinv(Wk_kv1 * gamma_in23) k_target ), k_target = gamma_k * mean decision query of layer-23 head 6 (unrotated dims 64:256), orthogonalized against the
          mean decision queries of heads 4/5/7 (same kv group): a residual marker at the clause end whose layer-23 key is aligned with head 6's decision query.
Transport: unchanged layer-23 attention; head 6 at the decision position attends to the marked clause end (alpha ~ 1) and copies its value.
Stage 2 (layer 23, MLP neuron J3=2331):
    gate[J3] = k3 * (t - lam*c - lam2*c2 - theta3*r3),  up[J3] = k3 * (t - theta3*r3) (quad) or k3u * r3 (lin),  down[:,J3] = a3 * (u_AB + rho*u_route)
    r3 = constant-projection direction at the layer-23 MLP input (decision positions, delivered decisions, hardest generated positions)
    t = unit mean change of the layer-23 MLP input at the decision when the marker is present (replay), c/c2/u_AB/u_route from g23v3.
J1/J2 rows of g23v3 are kept verbatim. Edited tensors: layers.22.mlp.{gate,up,down}_proj, layers.23.mlp.{gate,up,down}_proj (6 tensors, 2 layers)."""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
from analysis.cap import Capture, unit, ROOT
from analysis.stage1 import lda_dir
from analysis.replay23 import Replay
import primitives as P
from runtime.common import QWEN6, BASE
L22='model.language_model.layers.22.{}.weight'; L23='model.language_model.layers.23.{}.weight'
J0=2525; J3=2331; J1=3468; J2=3421
LABELS=('d01b_triples_D1','d02_triples_D2','d03_triples_D3')
CACHE=ROOT/'analysis/g23v4_directions.npz'

def silu(z): z=np.asarray(z,np.float64); return z/(1+np.exp(-z))

def fit_directions(force=False):
    if CACHE.exists() and not force: return dict(np.load(CACHE))
    caps=[Capture(l) for l in LABELS]; rng=np.random.default_rng(1)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    from benchmark.render import prompt as render
    import fp2
    ends=[];var=[];negs=[]
    for c in caps:
        for i in c.ids:
            X=c.full(i,'mlpin22'); ro=c.roles(i); ends.append(X[ro['end']]); var.append(c.meta[i]['variant'])
            it=fp2.items_for(c.set,[i])[0]; p=render(tok,it['input'],'standard',False); ids=tok.encode(p,add_special_tokens=False); toks=[tok.decode([t]) for t in ids]
            ustart=ids.index(tok.encode('<|im_start|>',add_special_tokens=False)[0],1); idx=np.arange(len(X)); dot=np.array([('.' in t) for t in toks])
            ok=(idx<=ustart)|((idx>ustart)&(idx<=ro['imend'])&~dot)|(idx>ro['imend']); ok[ro['end']]=False
            negs.append(X[rng.choice(idx[ok],60,replace=False)])
    ends=np.stack(ends).astype(np.float64); var=np.array(var); negs=np.concatenate(negs).astype(np.float64)
    u=lda_dir(ends[var=='ambiguous'],ends[var!='ambiguous'],0.5); g=lda_dir(ends,negs,0.5); r22=unit(ends.mean(0))
    # marker p6 and the delivered feature t (replay of layer 23)
    rp=Replay(); Wk=P.load_original(L23.format('self_attn.k_proj')); gin=P.load_original('model.language_model.layers.23.input_layernorm.weight')+1.0; gk=P.load_original(L23.format('self_attn.k_norm'))+1.0
    Q=[]
    for c in caps:
        for i in c.ids[:48]:
            r=rp.run(c.full(i,'h23'),want_x23=True,want_qk=True); Q.append(r['rows']['q'][:,c.roles(i)['dec'],:])
    Q=np.stack(Q); h=6; kt=np.zeros(256); kt[64:]=(gk*Q[:,h,:].mean(0))[64:]
    for h2 in (4,5,7):
        o=(Q[:,h2,:].mean(0)*gk).copy(); o[:64]=0; kt-=(kt@o)/(o@o)*o
    p6=unit(np.linalg.pinv(Wk[256:512]*gin[None,:])@kt)
    dx=[];xdec=[]
    for c in caps:
        for i in c.ids:
            h23=c.full(i,'h23'); ro=c.roles(i); b=rp.run(h23,want_x23=True); xdec.append(b['x23'][ro['dec']])
            if c.meta[i]['variant']=='ambiguous':
                h2=h23.copy(); h2[ro['end']]+=100*p6; e=rp.run(h2,want_x23=True); dx.append(e['x23'][ro['dec']]-b['x23'][ro['dec']])
    t=unit(np.stack(dx).mean(0)); r23=unit(np.stack(xdec).mean(0))
    # constant-projection bias directions (analysis/bias.py): r1 over {ambiguous clause ends, top-3000 harmful positions under w1}, r3 over {natural decision x23,
    # delivered x23, top-3000 generated positions under t - lam c - lam2 c2}; r.x ~ 1 on all of them, so w - theta*r realizes 'w.x - theta'
    from analysis.bias import const_dir
    from analysis.stage1e import harmful_sets
    pos,posfam,harm=harmful_sets(); harm={k:v for k,v in harm.items() if not (k[0]=='q6' and k[1]=='clarify')}
    allharm=np.concatenate([X for v in harm.values() for X in v]).astype(np.float64)
    w1=unit(0.5*g+u); sc=allharm@w1; r1=const_dir(np.concatenate([pos,allharm[np.argsort(sc)[-3000:]]]),0.1)
    V3=np.load(ROOT/'edits/g23v3.vectors.npz'); tg=t-0.3*V3['c']-0.5*V3['c2']
    nat=np.stack(xdec); dl=nat[:0]
    dlv=[]
    for c in caps:
        for i in c.ids:
            if c.meta[i]['variant']=='ambiguous':
                h23=c.full(i,'h23'); ro=c.roles(i); h2=h23.copy(); h2[ro['end']]+=100*p6; e=rp.run(h2,want_x23=True); dlv.append(e['x23'][ro['dec']])
    gen=[]
    for lab in ('p04c_practice_allpos','p05_teacher_allpos','p17_prose_capture','p35_prose_capture_nontool','p16_regression_capture'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r_ in js['rows']: X=z[f"mlpin_L23_{r_['case_id']}"].astype(np.float32); gen.append(X[r_['decision_position']:])
    g01=Capture('g01_genpos')
    for i in g01.ids: gen.append(g01.full(i,'mlpin23')[g01.rows[i]['decision_position']:])
    gen=np.concatenate(gen).astype(np.float64); sg=gen@tg; r3=const_dir(np.concatenate([nat,np.stack(dlv),gen[np.argsort(sg)[-3000:]]]),0.3)
    out=dict(u=u,g=g,r22=r22,p6=p6,t=t,r23=r23,kt=kt,qbar6=Q[:,6,:].mean(0),r1=r1,r3=r3); np.savez(CACHE,**out); return out

def rows(params,D=None):
    """-> dict with float32 rows/cols for the six edited tensors (J0 at layer 22; J1,J2,J3 at layer 23)."""
    D=D or fit_directions(); V3=np.load(ROOT/'edits/g23v3.vectors.npz'); S3=json.load(open(ROOT/'edits/g23v3.spec.json'))
    pr=params
    w1=unit(pr['a_g']*D['g']+D['u']); w1g=w1-pr['theta1']*D['r1']            # gate pre-activation ~ k1*(w1.x - theta1)
    gate22=pr['k1']*w1g; up22=pr['k1']*w1g; down22=pr['b']*D['p6']
    tg=D['t']-pr['lam']*V3['c']-pr['lam2']*V3['c2']-pr['theta3']*D['r3']     # ~ t.x - lam c.x - lam2 c2.x - theta3
    gate23=pr['k3']*tg; up23=(pr['k3']*(D['t']-pr['theta3']*D['r3'])) if pr.get('form','quad')=='quad' else (pr['k3u']*D['r3'])
    down23=pr['a3']*(V3['u_AB']+pr['rho']*V3['u_route'])
    if pr.get('random_seed') is not None:   # norm-matched random control on the same support (stage-1 and stage-2 vectors random, g23v3 kept)
        rng=np.random.default_rng(pr['random_seed']); rnd=lambda v:unit(rng.standard_normal(len(v)))*np.linalg.norm(v)
        gate22=rnd(gate22); up22=gate22.copy() if pr.get('form','quad')=='quad' else rnd(up22); down22=rnd(down22); gate23=rnd(gate23); up23=rnd(up23); down23=rnd(down23)
    return {'gate22':gate22.astype(np.float32),'up22':up22.astype(np.float32),'down22':down22.astype(np.float32),'gate23':gate23.astype(np.float32),'up23':up23.astype(np.float32),'down23':down23.astype(np.float32),
            'g23v3':{'gate':(S3['k']*V3['gate_dir']).astype(np.float32),'upJ1':(S3['k']*V3['s']).astype(np.float32),'downJ1':(S3['a1']*V3['u_AB']).astype(np.float32),'upJ2':(S3['kr']*V3['r']).astype(np.float32),'downJ2':(S3['a2']*V3['u_route']).astype(np.float32)}}

def layer_tensors(R):
    """Full float32 tensors of the six edited weights (original tensors with the neuron rows/columns replaced)."""
    Wg22=P.load_original(L22.format('mlp.gate_proj')); Wu22=P.load_original(L22.format('mlp.up_proj')); Wd22=P.load_original(L22.format('mlp.down_proj'))
    Wg23=P.load_original(L23.format('mlp.gate_proj')); Wu23=P.load_original(L23.format('mlp.up_proj')); Wd23=P.load_original(L23.format('mlp.down_proj'))
    Wg22[J0]=R['gate22']; Wu22[J0]=R['up22']; Wd22[:,J0]=R['down22']
    g3=R['g23v3']; Wg23[J1]=g3['gate']; Wu23[J1]=g3['upJ1']; Wd23[:,J1]=g3['downJ1']; Wg23[J2]=g3['gate']; Wu23[J2]=g3['upJ2']; Wd23[:,J2]=g3['downJ2']
    Wg23[J3]=R['gate23']; Wu23[J3]=R['up23']; Wd23[:,J3]=R['down23']
    return {L22.format('mlp.gate_proj'):Wg22,L22.format('mlp.up_proj'):Wu22,L22.format('mlp.down_proj'):Wd22,L23.format('mlp.gate_proj'):Wg23,L23.format('mlp.up_proj'):Wu23,L23.format('mlp.down_proj'):Wd23}

def write_edit(label,params):
    R=rows(params); T=layer_tensors(R); P.clear(); rec={}
    for name,W in T.items(): rec[name]=P.save_edit(name,W)
    path=P.snapshot(label)
    D=fit_directions()
    spec={'label':label,'layers':[22,23],'neurons':{'J0_L22':J0,'J1_L23':J1,'J2_L23':J2,'J3_L23':J3},'params':params,'receipts':rec,
          'total_changed_entries':sum(r['changed_entries'] for r in rec.values()),'fraction_of_text_params':rec[list(rec)[-1]]['fraction'],
          'equation':__doc__.split('\n',1)[1].strip()}
    json.dump(spec,open(ROOT/f'edits/{label}.spec.json','w'),indent=1)
    np.savez(ROOT/f'edits/{label}.vectors.npz',**{k:R[k] for k in ('gate22','up22','down22','gate23','up23','down23')},u=D['u'],g=D['g'],r22=D['r22'],p6=D['p6'],t=D['t'],r23=D['r23'])
    return path,spec

if __name__=='__main__':
    D=fit_directions(force='--refit' in sys.argv); print({k:v.shape for k,v in D.items()})
