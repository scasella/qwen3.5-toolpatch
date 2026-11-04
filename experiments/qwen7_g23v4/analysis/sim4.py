"""Exact CPU simulation of a g23v4 candidate: stage-1 activations from captured layer-22 MLP inputs at every prompt position, the resulting residual writes into the
layer-23 input, an exact replay of layer 23 (attention + MLP with J1/J2/J3 rows) and the final norm/unembedding at the decision position of all 192 triple items;
g23v3-only reference logits from the same replay; stage-1 and J3 silence statistics over every captured generated position (qwen-6 p04c/p05 calls+literals, g01 prose and
short non-tool answers) and non-ambiguous prompt positions. Usage: python -m analysis.sim4 '{"a_g":0.5,...}' [--scan]"""
from __future__ import annotations
import json, sys
import numpy as np
from analysis.cap import Capture, unit, ROOT, TOK
from analysis.replay23 import Replay
from analysis.stage1e import harmful_sets
import g23v4_build as B
import primitives as P
from runtime.common import QWEN6

DEFAULT={'a_g':0.5,'theta1':11.7,'k1':4.0,'b':1.0,'lam':0.3,'lam2':0.5,'theta3':2.0,'k3':2.0,'a3':0.15,'rho':0.5,'form':'quad'}

def stage1_acts(R,X):
    z=X@R['gate22']; return B.silu(z)*(X@R['up22']),z

def j3_acts(R,X):
    z=X@R['gate23']; return B.silu(z)*(X@R['up23']),z

def main():
    params=dict(DEFAULT); 
    if len(sys.argv)>1 and sys.argv[1].startswith('{'): params.update(json.loads(sys.argv[1]))
    D=B.fit_directions(); R=B.rows(params,D)
    caps=[Capture(l) for l in B.LABELS]
    Wg22=P.load_original(B.L22.format('mlp.gate_proj'))[B.J0]; Wu22=P.load_original(B.L22.format('mlp.up_proj'))[B.J0]; Wd22c=P.load_original(B.L22.format('mlp.down_proj'))[:,B.J0]
    # ---------------- stage-1 margins over harmful positions vs ambiguous ends (z = gate pre-activation)
    pos,posfam,harm=harmful_sets(); harm={k:v for k,v in harm.items() if not (k[0]=='q6' and k[1]=='clarify')}
    if '--scan' in sys.argv:
        w1=unit(params['a_g']*D['g']+D['u'])
        for th in (10.5,11.0,11.7,12.2):
            w=w1-th*D['r1']; za=pos@w; zh=max(float((X@w).max()) for v in harm.values() for X in v)
            print(f' theta1={th}: amb end z [{za.min():.2f},{za.max():.2f}] | harmful max {zh:.2f} | margin {za.min()-zh:.2f}')
        return
    za=(pos@R['gate22'])/params['k1']; zh=[(float((X@R['gate22']).max())/params['k1'],k) for k,v in harm.items() for X in v]; zh.sort(reverse=True)
    acts_h=max(float(stage1_acts(R,X)[0].max()) for v in harm.values() for X in v)
    print(f"stage-1 z (gate/k1): amb ends [{za.min():.2f},{za.max():.2f}] | harmful max {zh[0][0]:.2f} {zh[0][1]} | activation: amb ends min {(B.silu(params['k1']*za.min())*params['k1']*za.min()):.2f} | harmful max act {acts_h:.4f} -> max write norm {acts_h*np.linalg.norm(R['down22']):.3f}")
    # generated positions (qwen-6 + g01) for stage 1
    gen22={}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); d=r['decision_position']; gen22.setdefault(r['action'],[]).append(X[d:])
    g01=Capture('g01_genpos')
    for i in g01.ids: gen22.setdefault('prose' if i.startswith('prose') else 'reg2',[]).append(g01.full(i,'mlpin22')[g01.rows[i]['decision_position']:])
    print(' stage-1 at generated positions (max act, max write norm): '+' | '.join(f"{k}: {max(float(stage1_acts(R,X)[0].max()) for X in v):.3f}/{max(float(stage1_acts(R,X)[0].max()) for X in v)*np.linalg.norm(R['down22']):.3f}" for k,v in gen22.items()))
    # ---------------- replay of layer 23 on the 192 items
    rp=Replay(); T=B.layer_tensors(R); W23={'mlp.gate_proj.weight':T[B.L23.format('mlp.gate_proj')],'mlp.up_proj.weight':T[B.L23.format('mlp.up_proj')],'mlp.down_proj.weight':T[B.L23.format('mlp.down_proj')]}
    # g23v3-only weights for the reference
    Rv3={k:v for k,v in R.items()}; T3=B.layer_tensors(Rv3); 
    Wg=P.load_original(B.L23.format('mlp.gate_proj')); Wu=P.load_original(B.L23.format('mlp.up_proj')); Wd=P.load_original(B.L23.format('mlp.down_proj'))
    g3=R['g23v3']; Wg[B.J1]=g3['gate']; Wu[B.J1]=g3['upJ1']; Wd[:,B.J1]=g3['downJ1']; Wg[B.J2]=g3['gate']; Wu[B.J2]=g3['upJ2']; Wd[:,B.J2]=g3['downJ2']
    W23v3={'mlp.gate_proj.weight':Wg,'mlp.up_proj.weight':Wu,'mlp.down_proj.weight':Wd}
    res=[]
    for c in caps:
        for i in c.ids:
            h23=c.full(i,'h23'); x22=c.full(i,'mlpin22'); ro=c.roles(i); v=c.meta[i]['variant']
            a0,z0=stage1_acts(R,x22); a_orig=B.silu(x22@Wg22)*(x22@Wu22)
            h23e=h23+a0[:,None]*R['down22'][None,:]-a_orig[:,None]*Wd22c[None,:]
            rp.set_weights(W23v3); b=rp.run(h23,query_positions=(ro['dec'],),want_x23=True); lb=rp.logits(b['h24'][ro['dec']])
            rp.set_weights(W23); e=rp.run(h23e,query_positions=(ro['dec'],),want_x23=True); le=rp.logits(e['h24'][ro['dec']])
            x23=e['x23'][ro['dec']]; aj3,zj3=j3_acts(R,x23[None]); 
            top=np.argsort(le)[::-1][:3]
            res.append({'id':i,'var':v,'phr':c.meta[i]['phrasing'],'act0_end':float(a0[ro['end']]),'write_end':float(a0[ro['end']]*np.linalg.norm(R['down22'])),'act0_max_other':float(np.delete(a0,ro['end']).max()),
                        'alpha6':float(e['rows'][ro['dec']][6,ro['end']]),'t_x23':float(x23@D['t']),'j3_act':float(aj3[0]),'j3_z':float(zj3[0]),
                        'v3_argmax':int(lb.argmax()),'v4_argmax':int(le.argmax()),'v4_top':[(int(t_),float(le[t_])) for t_ in top],
                        'NE_call':float(le[TOK['NE']]-le[TOK['call']]),'NE_NO':float(le[TOK['NE']]-le[TOK['NO']]),'NE_other':float(le[TOK['NE']]-np.delete(le,TOK['NE']).max()),
                        'logit_NE':float(le[TOK['NE']]),'logit_NO':float(le[TOK['NO']]),'logit_call':float(le[TOK['call']]),'alpha6_all':[float(e['rows'][ro['dec']][h,ro['end']]) for h in range(8)],'j3_act_pred':float(aj3[0]),'max_dlogit_top':float(np.abs(le[np.argsort(lb)[::-1][:20]]-lb[np.argsort(lb)[::-1][:20]]).max()),'v3_NE_call':float(lb[TOK['NE']]-lb[TOK['call']])})
    rp.restore()
    amb=[r for r in res if r['var']=='ambiguous']; non=[r for r in res if r['var']!='ambiguous']
    ne_ok=sum(r['v4_argmax']==TOK['NE'] for r in amb); same=sum(r['v4_argmax']==r['v3_argmax'] for r in non)
    print(f"\nREPLAY g23v4: ambiguous NE argmax {ne_ok}/{len(amb)} (NE-call min {min(r['NE_call'] for r in amb):.2f}, NE-NO min {min(r['NE_NO'] for r in amb):.2f}, NE-other min {min(r['NE_other'] for r in amb):.2f}) | alpha6 min {min(r['alpha6'] for r in amb):.3f} | write at end [{min(r['write_end'] for r in amb):.1f},{max(r['write_end'] for r in amb):.1f}] | j3 act [{min(r['j3_act'] for r in amb):.1f},{max(r['j3_act'] for r in amb):.1f}] t.x23 [{min(r['t_x23'] for r in amb):.2f},{max(r['t_x23'] for r in amb):.2f}]")
    print(f"              non-ambiguous argmax same as g23v3 {same}/{len(non)} | max |dlogit| (top-20 tokens) {max(r['max_dlogit_top'] for r in non):.3f} | stage-1 act at their ends max {max(r['act0_end'] for r in non):.4f}, elsewhere max {max(r['act0_max_other'] for r in non):.4f} | j3 act max {max(r['j3_act'] for r in non):.4f} t.x23 max {max(r['t_x23'] for r in non):.2f}")
    fails=[r for r in amb if r['v4_argmax']!=TOK['NE']]
    for r in fails[:10]: print('   FAIL',r['id'],r['phr'],'top',[(t_,round(l,2)) for t_,l in r['v4_top']],'alpha6',round(r['alpha6'],3),'write',round(r['write_end'],1),'j3',round(r['j3_act'],2),'NE-call',round(r['NE_call'],2))
    worst=sorted(amb,key=lambda r:r['NE_other'])[:5]; print('   weakest ambiguous:',[(r['id'][-24:],r['phr'],round(r['NE_other'],2),round(r['alpha6'],3),round(r['write_end'],1)) for r in worst])
    byphr={}
    for r in amb: byphr.setdefault(r['phr'],[]).append(r['v4_argmax']==TOK['NE'])
    print('   by phrasing:',{k:f'{sum(v)}/{len(v)}' for k,v in sorted(byphr.items())})
    # ---------------- J3 silence at generated positions (closed form on captured x23)
    gen23={}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos','p17_prose_capture','p35_prose_capture_nontool','p16_regression_capture'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L23_{r['case_id']}"].astype(np.float32); d=r['decision_position']; gen23.setdefault(f'{lab[:4]}:{r["action"]}',[]).append(X[d:])
    for i in g01.ids: gen23.setdefault('g01:'+('prose' if i.startswith('prose') else 'reg2'),[]).append(g01.full(i,'mlpin23')[g01.rows[i]['decision_position']:])
    print(' J3 at decision+generated positions (max act -> max write norm): '+' | '.join(f"{k}: {max(float(j3_acts(R,X)[0].max()) for X in v):.3f}->{max(float(j3_acts(R,X)[0].max()) for X in v)*np.linalg.norm(R['down23']):.3f}" for k,v in sorted(gen23.items())))
    json.dump({'params':params,'rows':res},open(ROOT/'analysis/sim4_last.json','w'),indent=1)
if __name__=='__main__': main()
