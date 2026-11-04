"""Stage-1 read direction at the layer-22 MLP input: fires at the clause-end token of an unresolved clause, silent everywhere else.
Candidates: diff-of-means (amb end minus {call,res} end), shrinkage LDA; evaluated LOFO on the 192 triple items and against every other prompt position
of every item (max leak), and against qwen-6's generated-position captures (tool calls, literals) at layer 22."""
from __future__ import annotations
import sys, json
import numpy as np
from analysis.cap import Capture, lofo, dprime, unit, ROOT
from runtime.common import QWEN6

def silu(z): return z/(1+np.exp(-z))

def collect(caps,layer_name='mlpin22'):
    end=[];var=[];fam=[];cid=[];other=[];other_tag=[]
    for c in caps:
        for i in c.ids:
            X=c.full(i,layer_name); ro=c.roles(i); e=ro['end']
            end.append(X[e]); var.append(c.meta[i]['variant']); fam.append(c.meta[i]['family']); cid.append(i)
            mask=np.ones(len(X),bool); mask[e]=False
            other.append(X[mask]); other_tag.append((i,np.where(mask)[0]))
    return np.stack(end).astype(np.float64),np.array(var),np.array(fam),np.array(cid),other,other_tag

def lda_dir(Xa,Xb,shrink=0.5):
    mu_a=Xa.mean(0); mu_b=Xb.mean(0); R=np.concatenate([Xa-mu_a,Xb-mu_b]); S=R.T@R/len(R); S=(1-shrink)*S+shrink*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    w=np.linalg.solve(S,mu_a-mu_b); return unit(w)

def main():
    labels=sys.argv[1:] or ['d01b_triples_D1','d02_triples_D2','d03_triples_D3']; caps=[Capture(l) for l in labels]
    end,var,fam,cid,other,other_tag=collect(caps)
    pos=var=='ambiguous'
    print(f'{len(end)} clause-end vectors ({pos.sum()} ambiguous), {sum(len(o) for o in other)} other prompt positions')
    # qwen-6 generated positions at layer 22 (practice gold continuations: calls, literals; teacher)
    gen=[]
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); gen.append(X[r['decision_position']:])   # decision + generated
    gen=np.concatenate(gen).astype(np.float64); print('qwen-6 generated+decision positions at L22:',gen.shape)
    for name,shrink in (('diffmeans',None),('lda0.5',0.5),('lda0.2',0.2),('lda0.8',0.8)):
        # LOFO evaluation on clause ends
        proj=np.zeros(len(end)); 
        for f in np.unique(fam):
            tr=fam!=f
            d=unit(end[tr&pos].mean(0)-end[tr&~pos].mean(0)) if shrink is None else lda_dir(end[tr&pos],end[tr&~pos],shrink)
            proj[fam==f]=end[fam==f]@d
        # full-data direction for the leak checks
        d=unit(end[pos].mean(0)-end[~pos].mean(0)) if shrink is None else lda_dir(end[pos],end[~pos],shrink)
        ref=(end[pos]@d).min(); a=proj[pos]; b=proj[~pos]
        leak=np.array([ (o@d).max() for o in other]); leak_gen=(gen@d)
        # where does the worst leak sit?
        worst=np.argsort(leak)[-3:]
        print(f"\n{name}: LOFO d' {dprime(a,b):.2f} amb [{a.min():.2f},{a.max():.2f}] nonamb [{b.min():.2f},{b.max():.2f}] | full-fit amb min {ref:.2f} nonamb-end max {(end[~pos]@d).max():.2f} | other prompt positions max {leak.max():.2f} (99th {np.percentile(leak,99):.2f}) | qwen-6 generated max {leak_gen.max():.2f} 99.9th {np.percentile(leak_gen,99.9):.2f}")
        for w in worst:
            i,idx=other_tag[w]; c=next(c for c in caps if i in c.rows); X=c.full(i,'mlpin22'); pj=X[idx]@d; k=idx[np.argmax(pj)]
            print(f"   worst leak {leak[w]:.2f} at {i} pos {k} (roles {[r for r,p in c.roles(i).items() if p==k]})")
    np.savez(ROOT/'analysis/stage1_dirs.npz',end=end,var=var,fam=fam,cid=cid)
if __name__=='__main__': main()
