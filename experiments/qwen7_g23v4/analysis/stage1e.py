"""Stage-1 selectivity against the HARMFUL positions only: every prompt + generated position of non-ambiguous items (triples: call/resolved; qwen-6: call/answer/
NOTPROV incl. generated continuations; g01: prose and short non-tool answers incl. prompts), versus ambiguous clause ends. Candidates: (a) 2-D combination a*g+u,
(b) hard-negative LDA fit directly on positives vs harmful positions (LOFO by family for the triple part). Saves the chosen direction(s)."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, unit, dprime, ROOT
from runtime.common import QWEN6, BASE
from analysis.stage1 import lda_dir
rng=np.random.default_rng(2)

def harmful_sets():
    caps=[Capture(l) for l in ('d01b_triples_D1','d02_triples_D2','d03_triples_D3')]
    pos=[];posfam=[];harm={};harmfam=[]
    for c in caps:
        for i in c.ids:
            X=c.full(i,'mlpin22'); ro=c.roles(i); v=c.meta[i]['variant']; f=c.meta[i]['family']
            if v=='ambiguous': pos.append(X[ro['end']]); posfam.append(f)
            else: harm.setdefault(('triple',v,f),[]).append(X)
    from build_data import load_cases
    pc={c.id:c for c in load_cases('practice')}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); kind=r['action']
            if kind=='clarify' and r['case_id'] in pc and pc[r['case_id']].spec.ambiguous: continue   # ambiguous item: harmless
            harm.setdefault(('q6',kind,r['family']),[]).append(X)
    g01=Capture('g01_genpos')
    for i in g01.ids: harm.setdefault(('g01','prose' if i.startswith('prose') else 'reg2','-'),[]).append(g01.full(i,'mlpin22'))
    return np.stack(pos).astype(np.float64),np.array(posfam),harm

def main():
    pos,posfam,harm=harmful_sets()
    D=np.load(ROOT/'analysis/stage1_dirs.npz'); u=D['u']; g=D['g']
    print(f'{len(pos)} ambiguous clause ends; harmful groups {len(harm)} with {sum(len(x) for v in harm.values() for x in v)} positions')
    def report(name,d,thr=None):
        pp=pos@d; worst=[]
        for k,v in harm.items():
            m=max(float((x@d).max()) for x in v); worst.append((m,k))
        worst.sort(reverse=True); hm=worst[0][0]
        print(f" {name:28s}: amb end min {pp.min():6.2f} 5th {np.percentile(pp,5):6.2f} max {pp.max():6.2f} | harmful max {hm:6.2f} -> margin {pp.min()-hm:5.2f} | worst groups {[ (round(m,2),k) for m,k in worst[:4]]}")
        return pp.min()-hm
    for a in (0.5,0.75,1.0,1.25):
        report(f'{a}*g+u',unit(a*g+u))
    # hard-negative LDA: positives vs a stratified sample of harmful positions (all of the triple own-clause/user tokens + 10% of system/tool + all generated)
    negs=[]
    for k,v in harm.items():
        for X in v:
            n=len(X); take=min(n,80); negs.append(X[rng.choice(n,take,replace=False)])
    negs=np.concatenate(negs).astype(np.float64); print('hard-negative sample',negs.shape)
    for shrink in (0.3,0.5,0.7):
        d=lda_dir(pos,negs,shrink); m=report(f'hardLDA shrink {shrink}',d)
        # LOFO on the triple families for the positives/negatives from triples only (qwen-6/g01 negatives kept in training)
        pp=np.zeros(len(pos))
        for f in np.unique(posfam):
            tr=posfam!=f; d_=lda_dir(pos[tr],negs,shrink); pp[~tr]=pos[~tr]@d_
        print(f"     LOFO amb end min {pp.min():.2f} 5th {np.percentile(pp,5):.2f} (full-fit harmful max as above)")
        np.save(ROOT/f'analysis/stage1_hard{shrink}.npy',d)
    # combination of hard LDA with the gate g
    d5=np.load(ROOT/'analysis/stage1_hard0.5.npy')
    for a in (0.25,0.5,1.0): report(f'{a}*g+hardLDA0.5',unit(a*g+d5))
if __name__=='__main__': main()
