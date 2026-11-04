"""Build the layer-23 two-neuron edit ("G23"): re-purpose neurons J1 (abstain) and J2 (route).
gate[J1]=k*s, up[J1]=k*s, down[:,J1]=a1*u_AB ; gate[J2]=k*s, up[J2]=kr*r, down[:,J2]=a2*(u_NE-u_NO).
s: abstain direction (teacher diff of means at h_18, orthogonalized against literal end/first continuation signatures and the
call/abstain midpoint at the layer-23 MLP input); r: route direction (teacher clarify-answer at h_14, midpoint-orthogonalized).
Usage: python build_g23.py --label NAME --a1 A --a2 B [--k 2] [--kr 1] [--random SEED] [--src 18] [--base original]"""
from __future__ import annotations
import argparse, json
import numpy as np
import primitives as P
from analysis.simulate import load_all, directions, silu, report, EPS, TOK
from analysis.lens import load_run, embed, gamma_final
from analysis.leak2 import unit, ortho
J1,J2=3468,3421; L=23

def build_directions(src=18,t=0.5):
    D,W=load_all()
    allx=[]
    for lab in ('p05_teacher_allpos','p04c_practice_allpos'):
        js,arr=load_run(lab); allx+=[(arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32),r['decision_position'],r) for r in js['rows']]
    ids=list(D['ids'])
    def kind(r):
        if r['action']=='call': return 'call'
        if r['action']=='answer': return 'answer'
        return D['sub'][ids.index(r['case_id'])] if r['case_id'] in ids else 'clarify'
    mu_calldec=np.mean([x[d] for x,d,r in allx if kind(r)=='call'],0); mu_absdec=np.mean([x[d] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)
    # boundary reference: practice answer+NOTPROV and teacher clarify (teacher weak-cue answers excluded)
    mu_ref=np.mean([x[d] for x,d,r in allx if (kind(r) in ('answer','NOTPROV') and r['case_id'].startswith('discovery')) or (kind(r)=='clarify')],0)
    lit_end=np.mean([x[-1] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)-mu_calldec
    lit_first=np.mean([x[d+1] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)-mu_calldec
    s0,r=directions(D,src,14)
    ref=mu_calldec+t*(mu_ref-mu_calldec)
    s=unit(ortho(unit(ortho(ortho(s0,lit_end),lit_first)),ref))
    return D,W,s,r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True); ap.add_argument('--a1',type=float,required=True); ap.add_argument('--a2',type=float,required=True)
    ap.add_argument('--k',type=float,default=2.0); ap.add_argument('--kr',type=float,default=1.0); ap.add_argument('--src',type=int,default=18); ap.add_argument('--random',type=int,default=None); ap.add_argument('--simulate_only',action='store_true'); ap.add_argument('--t',type=float,default=0.5); ap.add_argument('--lam',type=float,default=0.0); ap.add_argument('--prose_target',type=float,default=0.0); ap.add_argument('--lam2',type=float,default=0.0)
    a=ap.parse_args()
    D,W,s,r=build_directions(a.src,a.t); E=embed(); g=gamma_final()
    U=np.load('analysis/u_directions.npz'); u_AB=U['uAB']; u_route=U['uB']-U['uA']
    if a.prose_target:
        names=list(U['names']); tid={w:P.tok.encode(w,add_special_tokens=False)[0] for w in names}; M=E[[tid[w] for w in names]]*g
        u_AB=np.linalg.lstsq(M,np.array([{'<tool_call>':-1.0,'NO':1.0,'NE':1.0,'I':-a.prose_target,'To':-a.prose_target,'The':-a.prose_target}.get(w,0.0) for w in names]),rcond=None)[0]
    # tool-context gate term: c = unit(ortho(mean non-tool decision x - mean tool decision x, mean tool decision x)); gate = k (s - lam c)
    c=np.zeros(1024); c2=np.zeros(1024)
    from analysis.lens import load_run
    if a.lam:
        js16,arr16=load_run('p16_regression_capture'); xr=np.stack([arr16[f"mlpin_L23_{rr['case_id']}"].astype(np.float32)[rr['decision_position']] for rr in js16['rows']])
        c=unit(ortho(xr.mean(0)-D['x'].mean(0),D['x'].mean(0)))
    if a.lam2:
        # generated-position term: mean x over ALL captured generated (post-decision) positions -- literal continuations, inside tool calls,
        # prose in tool contexts (p17) and prose answers to non-tool prompts (p35) -- minus the tool-context decision mean, orthogonal to that mean
        gen=[]
        for lab in ('p05_teacher_allpos','p04c_practice_allpos','p17_prose_capture','p35_prose_capture_nontool'):
            try: js_,arr_=load_run(lab)
            except FileNotFoundError: continue
            gen+=[arr_[f"mlpin_L23_{rr['case_id']}"].astype(np.float32)[rr['decision_position']+1:] for rr in js_['rows']]
        gen=np.concatenate(gen); c2=unit(ortho(gen.mean(0)-D['x'].mean(0),D['x'].mean(0)))
    gate_dir=s-a.lam*c-a.lam2*c2
    if a.random is not None:
        rng=np.random.default_rng(a.random)
        s=unit(rng.standard_normal(1024))*np.linalg.norm(s); r=unit(rng.standard_normal(1024))*np.linalg.norm(r); gate_dir=unit(rng.standard_normal(1024))*np.linalg.norm(gate_dir)
        u_AB=unit(rng.standard_normal(1024))*np.linalg.norm(u_AB); u_route=unit(rng.standard_normal(1024))*np.linalg.norm(u_route)
    def simulate(D,W,s,r,k,kr,a1,a2,j1,j2,u_AB,u_route,E,g,which='dec'):
        x=D['x'] if which=='dec' else D['xpre']; h=D['h24'] if which=='dec' else D['h24pre']
        fs=x@s; fr=x@r; fg=x@gate_dir; a_1=silu(k*fg)*(k*fs); a_2=silu(k*fg)*(kr*fr); orig=np.zeros_like(h)
        for j in (j1,j2): orig+=(silu(x@W['Wg'][j])*(x@W['Wu'][j]))[:,None]*W['Wd'][:,j][None,:]
        hn=h-orig+a1*a_1[:,None]*u_AB[None,:]+a2*a_2[:,None]*u_route[None,:]
        return (hn/np.sqrt((hn*hn).mean(-1,keepdims=True)+EPS)*g)@E.T,fs,fr,a_1,a_2
    lg,fs,fr,a_1,a_2=simulate(D,W,s,r,a.k,a.kr,a.a1,a.a2,J1,J2,u_AB,u_route,E,g,'dec'); rep=report(D,lg,fs,fr,'',E)
    lgp,_,_,_,_=simulate(D,W,s,r,a.k,a.kr,a.a1,a.a2,J1,J2,u_AB,u_route,E,g,'pre')
    callm=D['act']=='call'; marg=lg[:,TOK['call']]-np.where(np.arange(lg.shape[1])[None,:]==TOK['call'],-1e9,lg).max(1)
    sim={'prefix_gold_argmax':int((lgp.argmax(1)==D['pre_gold']).sum()),'call_min_margin':float(marg[callm].min()),'report':rep}
    print(json.dumps(sim,indent=1))
    if a.simulate_only: return
    P.clear()
    rows={}
    gate=P.load_original(f'model.language_model.layers.{L}.mlp.gate_proj.weight'); up=P.load_original(f'model.language_model.layers.{L}.mlp.up_proj.weight'); down=P.load_original(f'model.language_model.layers.{L}.mlp.down_proj.weight')
    gate[J1]=a.k*gate_dir; up[J1]=a.k*s; down[:,J1]=a.a1*u_AB
    gate[J2]=a.k*gate_dir; up[J2]=a.kr*r; down[:,J2]=a.a2*u_route
    for nm,t in (('mlp.gate_proj',gate),('mlp.up_proj',up),('mlp.down_proj',down)):
        rows[nm]=P.save_edit(f'model.language_model.layers.{L}.{nm}.weight',t)
    path=P.snapshot(a.label)
    spec={'label':a.label,'layer':L,'neurons':[J1,J2],'k':a.k,'kr':a.kr,'a1':a.a1,'a2':a.a2,'src_s':a.src,'t':a.t,'lam':a.lam,'lam2':a.lam2,'prose_target':a.prose_target,'random_seed':a.random,
          's_norm':float(np.linalg.norm(s)),'r_norm':float(np.linalg.norm(r)),'uAB_norm':float(np.linalg.norm(u_AB)),'uroute_norm':float(np.linalg.norm(u_route)),
          'equation':'gate[J1]=k*(s-lam*c-lam2*c2), up[J1]=k*s, down[:,J1]=a1*u_AB; gate[J2]=k*(s-lam*c-lam2*c2), up[J2]=kr*r, down[:,J2]=a2*(u_NE-u_NO)','receipts':rows,'simulation':sim}
    json.dump(spec,open(f'edits/{a.label}.spec.json','w'),indent=1); np.savez(f'edits/{a.label}.vectors.npz',s=s,r=r,u_AB=u_AB,u_route=u_route,gate_dir=gate_dir,c=c,c2=c2)
    print('saved',path,json.dumps({k:(v['changed_entries'],round(v['l2'],4)) for k,v in rows.items()}),'total changed',sum(v['changed_entries'] for v in rows.values()))
if __name__=='__main__': main()

def prose_signature():
    """Mean layer-23 MLP input over prose-continuation positions of abstain prompts (p17) minus the call decision mean; and the
    decision-position mean of non-tool regression prompts (p16). Used to orthogonalize s so the readout is silent there."""
    from analysis.lens import load_run
    D,W=load_all(); ids=list(D['ids'])
    js,arr=load_run('p17_prose_capture')
    prose=np.concatenate([arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32)[r['decision_position']+1:] for r in js['rows']])
    js16,arr16=load_run('p16_regression_capture')
    reg_dec=np.stack([arr16[f"mlpin_L23_{r['case_id']}"].astype(np.float32)[r['decision_position']] for r in js16['rows']])
    reg_cont=np.concatenate([arr16[f"mlpin_L23_{r['case_id']}"].astype(np.float32)[r['decision_position']+1:] for r in js16['rows']])
    allx=[]
    for lab in ('p05_teacher_allpos','p04c_practice_allpos'):
        j,a=load_run(lab); allx+=[(a[f"mlpin_L23_{r['case_id']}"].astype(np.float32),r['decision_position'],r) for r in j['rows']]
    mu_calldec=np.mean([x[d] for x,d,r in allx if r['action']=='call'],0)
    return {'prose':prose.mean(0)-mu_calldec,'reg_dec':reg_dec.mean(0)-mu_calldec,'reg_cont':reg_cont.mean(0)-mu_calldec,'prose_x':prose,'reg_dec_x':reg_dec,'reg_cont_x':reg_cont}
