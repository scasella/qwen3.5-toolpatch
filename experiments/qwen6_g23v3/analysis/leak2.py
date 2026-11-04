"""Where inside generated continuations does the abstain readout fire? Build s orthogonal to continuation signatures and report absolute leakage."""
from __future__ import annotations
import numpy as np, json
from collections import Counter
from analysis.simulate import load_all, directions, silu
from analysis.lens import load_run
unit=lambda v:v/np.linalg.norm(v)
def ortho(v,ref): return v-(v@ref)/(ref@ref)*ref

def main():
    D,W=load_all()
    allx=[]
    for lab in ('p05_teacher_allpos','p04c_practice_allpos'):
        js,arr=load_run(lab); allx+=[(arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32),r['decision_position'],r) for r in js['rows']]
    ids=list(D['ids'])
    def kind(r):
        if r['action']=='call': return 'call'
        if r['action']=='answer': return 'answer'
        return D['sub'][ids.index(r['case_id'])] if r['case_id'] in ids else 'clarify'
    mu_calldec=np.mean([x[d] for x,d,r in allx if kind(r)=='call'],0)
    mu_absdec=np.mean([x[d] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)
    sig={}
    sig['lit_end']=np.mean([x[-1] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)-mu_calldec
    sig['lit_first']=np.mean([x[d+1] for x,d,r in allx if kind(r) in ('answer','NOTPROV')],0)-mu_calldec
    sig['call_end']=np.mean([x[-1] for x,d,r in allx if kind(r)=='call'],0)-mu_calldec
    sig['call_cont']=np.mean(np.concatenate([x[d+1:-1] for x,d,r in allx if kind(r)=='call']),0)-mu_calldec
    # token-position-specific call signatures: group continuation positions of calls by the token predicted there
    tokpos={}
    for x,d,r in allx:
        if kind(r)!='call': continue
        for j,c in enumerate(r['continuation_tokens']):
            tokpos.setdefault(c['text'],[]).append(x[d+j])   # position d+j predicts continuation token j
    for src in (16,18):
        s0,rdir=directions(D,src,14)
        chains={'s0':[], 's_lit':['lit_end','lit_first'], 's_lit_callend':['lit_end','lit_first','call_end'], 's_all':['lit_end','lit_first','call_end','call_cont']}
        for nm,chain in chains.items():
            s=s0.copy()
            for k in chain: s=ortho(s,sig[k])
            s=unit(ortho(unit(s),(mu_calldec+mu_absdec)/2))
            q=lambda f: silu(2*f)*(2*f)
            dec_call=[];dec_abs=[];dec_amb=[];leaks=[]
            for x,d,r in allx:
                f=x@s; k=kind(r)
                if k=='call': dec_call.append(f[d])
                elif k=='AMBIG': dec_amb.append(f[d])
                else: dec_abs.append((f[d],r['case_id']))
                for j in range(d+1,len(x)):
                    tok=r['continuation_tokens'][j-d-1]['text'] if j-d-1<len(r['continuation_tokens']) else '?'
                    leaks.append((q(f[j]),f[j],k,tok,r['case_id']))
            dec_call=np.array(dec_call); absv=np.array([a for a,_ in dec_abs])
            pa=np.array([a for a,c in dec_abs if c.startswith('discovery')])   # practice abstain (answer+NOTPROV)
            worst=sorted(leaks,key=lambda t:-t[0])[:6]
            print(f'L{src} {nm:14s}: dec call max {dec_call.max():5.2f} (q {q(dec_call.max()):5.2f}) | practice abstain min {pa.min():5.2f} (q {q(pa.min()):5.1f}) 25th {np.percentile(pa,25):5.2f} (q {q(np.percentile(pa,25)):5.1f}) | AMBIG max {max(dec_amb):.2f} | worst continuation leaks (q, f, kind, predicted token): '+'; '.join(f'{a:.1f}/{b:.2f}/{k}/{t!r}' for a,b,k,t,_ in worst))
        print()
if __name__=='__main__': main()
