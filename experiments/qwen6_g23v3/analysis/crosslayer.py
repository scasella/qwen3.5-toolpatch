"""Read mid-layer teacher-derived residual directions (answer-call, clarify-call) at the final-layer MLP input x_23 and at h_l."""
from __future__ import annotations
import numpy as np
import primitives as P
from analysis.lens import load_run, classes, practice_classes, EPS
from analysis.l23 import mlp_input, silu
from build_data import load_cases

def main():
    Wo=P.load_original('model.language_model.layers.23.self_attn.o_proj.weight'); gpost=P.load_original('model.language_model.layers.23.post_attention_layernorm.weight')+1.0
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    tact,tren=classes(ta['case_ids']); pact,_=practice_classes(pa['case_ids'])
    cases={c.id:c for c in load_cases('practice')}; amb=np.array([cases[str(c)].spec.ambiguous for c in pa['case_ids']])
    xt,_=mlp_input(ta['h_dec'][:,23],ta['attn_dec'][:,5],Wo,gpost); xp,_=mlp_input(pa['h_dec'][:,23],pa['attn_dec'][:,5],Wo,gpost)
    xpre=np.concatenate([mlp_input(ta['h_pre'][:,23],ta['attn_pre'][:,5],Wo,gpost)[0],mlp_input(pa['h_pre'][:,23],pa['attn_pre'][:,5],Wo,gpost)[0]])
    print('gpost stats: mean',gpost.mean(),'min',gpost.min(),'max',gpost.max())
    for feat,pos in (('A explain','answer'),('B missing','clarify')):
        print(f'\n===== {feat}: direction from teacher h_src diff of means (ortho to call mean at source), read at x_23 (normalized final MLP input) =====')
        print(' src | teacher: call max / pos min / other max | practice: call [min,max] / pos [min,max] / NOTPROV min / AMBIG max / answer-or-clarify-other max | prefix [min,max]  (all relative to practice call mean)')
        for src in (10,12,13,14,15,16,18,20,23):
            H=ta['h_dec'][:,src]; d=H[tact==pos].mean(0)-H[tact=='call'].mean(0)
            mc=np.concatenate([H[tact=='call'],pa['h_dec'][:,src][pact=='call']]).mean(0)
            d=d-(d@mc)/(mc@mc)*mc; d/=np.linalg.norm(d)
            # read at x_23 (what a layer-23 neuron would see): projection of normalized input
            dx=d*gpost  # account for per-dim gamma: x = mid/rms*gpost ; reading direction d on 'mid' corresponds to d/gpost on x; here we read raw d on x for simplicity -> use d_x = d (both reported)
            for nm,dd in (('d on x23',d),):
                ft=xt@dd; fp=xp@dd; fpre=xpre@dd; c0=fp[pact=='call'].mean()
                other='clarify' if pos=='answer' else 'answer'
                print(f' L{src:2d} | T call max {ft[tact=="call"].max()-c0:6.2f} pos min {ft[tact==pos].min()-c0:6.2f} other max {ft[tact==other].max()-c0:6.2f} | P call [{fp[pact=="call"].min()-c0:6.2f},{fp[pact=="call"].max()-c0:6.2f}] pos [{fp[pact==pos].min()-c0:6.2f},{fp[pact==pos].max()-c0:6.2f}] NOTPROV min {fp[(pact=="clarify")&~amb].min()-c0:6.2f} AMBIG max {fp[(pact=="clarify")&amb].max()-c0:6.2f} other max {fp[pact==other].max()-c0:6.2f} | prefix [{fpre.min()-c0:6.2f},{fpre.max()-c0:6.2f}]')
        print(f'--- same, but read at h_src itself (for reference) ---')
        for src in (12,14,16,18):
            H=ta['h_dec'][:,src]; Hp=pa['h_dec'][:,src]; d=H[tact==pos].mean(0)-H[tact=='call'].mean(0)
            mc=np.concatenate([H[tact=='call'],Hp[pact=='call']]).mean(0); d=d-(d@mc)/(mc@mc)*mc; d/=np.linalg.norm(d)
            ft=H@d; fp=Hp@d; c0=fp[pact=='call'].mean(); pre=np.concatenate([ta['h_pre'][:,src]@d,pa['h_pre'][:,src]@d])
            other='clarify' if pos=='answer' else 'answer'
            print(f' L{src:2d} | T call max {ft[tact=="call"].max()-c0:6.3f} pos min {ft[tact==pos].min()-c0:6.3f} | P call [{fp[pact=="call"].min()-c0:6.3f},{fp[pact=="call"].max()-c0:6.3f}] pos [{fp[pact==pos].min()-c0:6.3f},{fp[pact==pos].max()-c0:6.3f}] NOTPROV min {fp[(pact=="clarify")&~amb].min()-c0:6.3f} AMBIG max {fp[(pact=="clarify")&amb].max()-c0:6.3f} other max {fp[pact==other].max()-c0:6.3f} | prefix [{pre.min()-c0:6.3f},{pre.max()-c0:6.3f}]  (rms h_src {np.sqrt((H*H).mean()):.3f})')
if __name__=='__main__': main()
