"""Leak split for the end-only LDA stage-1 direction: per item class, max projection over own-clause tokens vs elsewhere; qwen-6 prompt/generated positions by kind;
token identities of the worst non-ambiguous leaks."""
from __future__ import annotations
import json, numpy as np
from analysis.cap import Capture, dprime, unit, ROOT
from runtime.common import QWEN6, BASE
from analysis.stage1 import lda_dir, collect

def main():
    caps=[Capture(l) for l in ('d01b_triples_D1','d02_triples_D2','d03_triples_D3')]
    end,var,fam,cid,other,other_tag=collect(caps); pos=var=='ambiguous'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    from benchmark.render import prompt as render
    gen={}
    for lab in ('p04c_practice_allpos','p05_teacher_allpos'):
        js=json.loads((QWEN6/'runs'/f'{lab}.json').read_text()); z=np.load(QWEN6/'runs'/f'{lab}.npz')
        for r in js['rows']:
            X=z[f"mlpin_L22_{r['case_id']}"].astype(np.float32); d=r['decision_position']; kind=r['action']
            if kind=='clarify' and lab=='p04c_practice_allpos':
                from build_data import load_cases
                c=next(c for c in load_cases('practice') if c.id==r['case_id']); kind='AMBIG' if c.spec.ambiguous else 'NOTPROV'
            gen.setdefault(('dec',kind),[]).append((r['case_id'],d,X[d][None])); gen.setdefault(('gen',kind),[]).append((r['case_id'],d+1,X[d+1:])); gen.setdefault(('prompt',kind),[]).append((r['case_id'],0,X[:d]))
    for shrink in (0.5,0.2):
        d=lda_dir(end[pos],end[~pos],shrink)
        # LOFO projections at ends
        proj=np.zeros(len(end))
        for f in np.unique(fam):
            tr=fam!=f; dd=lda_dir(end[tr&pos],end[tr&~pos],shrink); proj[fam==f]=end[fam==f]@dd
        print(f'\n=== end-only LDA shrink {shrink}: LOFO ends amb [{proj[pos].min():.2f},{proj[pos].max():.2f}] 5th {np.percentile(proj[pos],5):.2f} | nonamb ends max {proj[~pos].max():.2f}')
        worst=[]
        for v in ('ambiguous','resolved','call'):
            own=[];els=[]
            for k,(i,idx) in enumerate(other_tag):
                if var[k]!=v: continue
                c=next(c for c in caps if i in c.rows); X=c.full(i,'mlpin22'); ro=c.roles(i); lo=min(ro.get('v1c',ro['v1']),ro['v1']); hi=ro['end']; p=X@d
                m_own=np.zeros(len(X),bool); m_own[lo:hi+1]=True; m_own[hi]=False
                own.append(p[m_own].max() if m_own.any() else -99); els.append(p[~m_own&(np.arange(len(X))!=hi)].max())
                if v!='ambiguous':
                    j=int(np.argmax(np.where(~m_own&(np.arange(len(X))!=hi),p,-99))); worst.append((float(p[j]),i,j))
            print(f'  {v:9s}: own-clause other tokens max {np.max(own):.2f} | elsewhere max {np.max(els):.2f} 99th {np.percentile(els,99):.2f} median {np.median(els):.2f}')
        worst.sort(reverse=True)
        for val,i,j in worst[:8]:
            c=next(c for c in caps if i in c.rows); it=next(x for x in c.js['rows'] if x['case_id']==i)
            p=render(tok,__import__('fp2').items_for(c.set,[i])[0]['input'],'standard',False); ids=tok.encode(p,add_special_tokens=False)
            print(f'     leak {val:.2f} {i} pos {j}: ...{tok.decode(ids[max(0,j-6):j])!r} [{tok.decode([ids[j]])!r}]')
        for k,v in sorted(gen.items()):
            P_=np.concatenate([x[2] for x in v])@d; print(f'  qwen-6 {k[0]:6s} {k[1]:8s}: n={len(P_):5d} max {P_.max():.2f} 99.9th {np.percentile(P_,99.9):.2f}')
            if k[0]=='gen' and P_.max()>3:
                # which generated token?
                best=max(((float((x[2]@d).max()),x[0],x[1]+int(np.argmax(x[2]@d))) for x in v)); print('     worst generated leak',best)
        np.save(ROOT/f'analysis/stage1_endlda{shrink}.npy',d)
if __name__=='__main__': main()
