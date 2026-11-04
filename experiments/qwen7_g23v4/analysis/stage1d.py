"""Stage-1 as a conjunction at the layer-22 MLP input: u = 'unresolved' (end-only LDA) and g = 'end of a user fact clause' (LDA: tagged clause ends vs all
positions outside the user turn + user-turn tokens that are not '.'-type). Scatter of (g.x, u.x) over position classes from the triple captures, qwen-6 generated
positions (calls, literals; tool-context) and the g01 capture (prose answers, short non-tool answers). Saves both directions."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, dprime, unit, ROOT
from runtime.common import QWEN6, BASE
from analysis.stage1 import lda_dir
rng=np.random.default_rng(1)

def main():
    caps=[Capture(l) for l in ('d01b_triples_D1','d02_triples_D2','d03_triples_D3')]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    from benchmark.render import prompt as render
    import fp2
    recs=[]
    for c in caps:
        for i in c.ids:
            X=c.full(i,'mlpin22'); ro=c.roles(i); it=fp2.items_for(c.set,[i])[0]; p=render(tok,it['input'],'standard',False); ids=tok.encode(p,add_special_tokens=False)
            toks=[tok.decode([t]) for t in ids]
            ustart=ids.index(tok.encode('<|im_start|>',add_special_tokens=False)[0],1)   # second <|im_start|> = user turn
            uend=ro['imend']
            recs.append({'cid':i,'var':c.meta[i]['variant'],'fam':c.meta[i]['family'],'X':X,'end':ro['end'],'ustart':ustart,'uend':uend,'toks':toks,'roles':ro})
    # ---- directions
    ends=np.stack([r['X'][r['end']] for r in recs]).astype(np.float64); var=np.array([r['var'] for r in recs]); fam=np.array([r['fam'] for r in recs])
    u=lda_dir(ends[var=='ambiguous'],ends[var!='ambiguous'],0.5)
    def neg_for_gate(rs):
        out=[]
        for r in rs:
            idx=np.arange(len(r['X'])); dot=np.array([('.' in t) for t in r['toks']])
            ok=(idx<=r['ustart'])|((idx>r['ustart'])&(idx<=r['uend'])&~dot)|(idx>r['uend'])
            ok[r['end']]=False
            pick=rng.choice(idx[ok],60,replace=False); out.append(r['X'][pick])
        return np.concatenate(out).astype(np.float64)
    g=lda_dir(ends,neg_for_gate(recs),0.5)
    # LOFO for g
    proj_g=np.zeros(len(recs))
    for f in np.unique(fam):
        tr=[r for r in recs if r['fam']!=f]; gg=lda_dir(np.stack([r['X'][r['end']] for r in tr]).astype(np.float64),neg_for_gate(tr),0.5)
        for k,r in enumerate(recs):
            if r['fam']==f: proj_g[k]=r['X'][r['end']]@gg
    print(f"gate g (LOFO) at clause ends: [{proj_g.min():.2f},{proj_g.max():.2f}] 5th {np.percentile(proj_g,5):.2f}")
    # ---- scatter by position class
    classes={}
    def add(name,Xs): classes.setdefault(name,[]).append(np.stack([Xs@g,Xs@u],1) if Xs.ndim==2 else np.array([[Xs@g,Xs@u]]))
    for r in recs:
        X=r['X']; idx=np.arange(len(X)); dot=np.array([('.' in t) for t in r['toks']]); e=r['end']
        add(f"end:{r['var']}",X[e])
        inuser=(idx>r['ustart'])&(idx<=r['uend'])&(idx!=e)
        ro=r['roles']; lo=min(ro.get('v1c',ro['v1']),ro['v1']); own=np.zeros(len(X),bool); own[lo:e]=True
        add(f"ownclause:{r['var']}",X[own]) if own.any() else None
        add("user_dot_other",X[inuser&dot&~own]); add("user_nondot",X[inuser&~dot&~own]); add("system_tools",X[idx<=r['ustart']]); add("after_user",X[idx>r['uend']])
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); d=r['decision_position']
            add(f"q6_gen:{r['action']}",X[d+1:]); add(f"q6_dec:{r['action']}",X[d])
    g01=Capture('g01_genpos') if (ROOT/'runs/g01_genpos.json').exists() else None
    if g01:
        for i in g01.ids:
            X=g01.full(i,'mlpin22'); d=g01.rows[i]['decision_position']; kind='prose' if i.startswith('prose') else 'reg2'
            add(f"g01_gen:{kind}",X[d:]); add(f"g01_prompt:{kind}",X[:d])
    print(f"\n{'class':22s} {'n':>6s} | g.x: min   5th   max | u.x: min  95th   max | max over positions of (u.x) where g.x > -2")
    out={}
    for k in sorted(classes):
        A=np.concatenate(classes[k]); gm=A[:,0]; um=A[:,1]; sel=gm>-2
        print(f"{k:22s} {len(A):6d} | {gm.min():6.2f} {np.percentile(gm,5):6.2f} {gm.max():6.2f} | {um.min():6.2f} {np.percentile(um,95):6.2f} {um.max():6.2f} | {um[sel].max() if sel.any() else float('nan'):6.2f} (n {int(sel.sum())})")
        out[k]={'n':int(len(A)),'g':[float(gm.min()),float(gm.max())],'u':[float(um.min()),float(um.max())]}
    np.savez(ROOT/'analysis/stage1_dirs.npz',u=u,g=g); json.dump(out,open(ROOT/'analysis/stage1d.json','w'),indent=1)
    # candidate thresholds: find the best linear combination gate = a*(g.x) + (u.x) - th separating end:ambiguous from everything else; report margins
    amb=np.concatenate(classes['end:ambiguous']); rest=np.concatenate([np.concatenate(v) for k,v in classes.items() if k!='end:ambiguous'])
    best=None
    for a in (0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0):
        za=a*amb[:,0]+amb[:,1]; zr=a*rest[:,0]+rest[:,1]; m=za.min()-zr.max()
        print(f" gate = {a}*g.x + u.x : amb end min {za.min():.2f} | rest max {zr.max():.2f} | margin {m:.2f} | threshold midpoint {(za.min()+zr.max())/2:.2f}")
        if best is None or m>best[0]: best=(m,a)
    print('best a',best)
if __name__=='__main__': main()
