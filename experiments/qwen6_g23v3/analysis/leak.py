"""Can the abstain direction be made silent at generated continuation positions while keeping decision-position separation?
Candidates: s0 (diff of means), s1 = s0 orthogonalized against end-of-literal and first-continuation signatures."""
from __future__ import annotations
import numpy as np
from analysis.simulate import load_all, directions, silu
from analysis.lens import load_run

def unit(v): return v/np.linalg.norm(v)
def ortho(v,ref): return v-(v@ref)/(ref@ref)*ref

def main():
    D,W=load_all()
    sets={}
    for lab in ('p05_teacher_allpos','p04c_practice_allpos'):
        js,arr=load_run(lab); sets[lab]=[(arr[f"mlpin_L23_{r['case_id']}"].astype(np.float32),r['decision_position'],r) for r in js['rows']]
    allx=[x for s in sets.values() for x in s]
    is_abst=lambda r: r['action'] in ('answer','clarify') and not (r['case_id'].startswith('discovery') and D['sub'][list(D['ids']).index(r['case_id'])]=='AMBIG')
    # position signatures
    mu_end=np.mean([x[-1] for x,d,r in allx if is_abst(r)],0)          # last position: predicting <|im_end|>
    mu_first=np.mean([x[d+1] for x,d,r in allx if is_abst(r)],0)       # first continuation position (after NO / NE)
    mu_calldec=np.mean([x[d] for x,d,r in allx if r['action']=='call'],0)
    mu_absdec=np.mean([x[d] for x,d,r in allx if is_abst(r)],0)
    mu_callcont=np.mean(np.concatenate([x[d+1:] for x,d,r in allx if r['action']=='call']),0)
    e_end=mu_end-mu_calldec; e_first=mu_first-mu_calldec; a_dec=mu_absdec-mu_calldec
    print('cos(abstain-dec diff, end-of-literal diff)=%.3f ; cos(abstain-dec diff, first-cont diff)=%.3f ; cos(end,first)=%.3f'%(unit(a_dec)@unit(e_end),unit(a_dec)@unit(e_first),unit(e_end)@unit(e_first)))
    for src in (16,18):
        s0,r=directions(D,src,14)
        variants={'s0':s0}
        s1=unit(ortho(s0,e_end)); s2=unit(ortho(ortho(s0,e_end),e_first)); s3=unit(ortho(ortho(ortho(s0,e_end),e_first),mu_callcont-mu_calldec))
        variants.update({'s1=s0-end':s1,'s2=s0-end-first':s2,'s3=s2-callcont':s3})
        for nm,s in variants.items():
            # re-center at midpoint between call mean and abstain mean at decision (so fs(call mean)=-fs(abstain mean))
            mid=(mu_calldec+mu_absdec)/2; s=unit(ortho(s,mid))
            fs_dec_call=[];fs_dec_abs=[];fs_dec_amb=[];leak=[];leak_pos=[];hard=None
            for x,d,r in allx:
                f=x@s
                if r['action']=='call':
                    fs_dec_call.append(f[d]); 
                    if r['case_id']=='teacher-06-call': hard=f[d]
                    leak_pos.append(np.abs(f[d+1:]).max())
                elif is_abst(r):
                    fs_dec_abs.append(f[d]); q=silu(2*f)*(2*f); leak.append((q[d+1:].max()/max(q[d],1e-6),f[d],f[d+1:].max(),r['case_id']))
                else: fs_dec_amb.append(f[d])
            fs_dec_call=np.array(fs_dec_call); fs_dec_abs=np.array(fs_dec_abs)
            lk=sorted(leak,key=lambda t:-t[0])
            print(f'L{src} {nm:16s}: dec call [{fs_dec_call.min():5.2f},{fs_dec_call.max():5.2f}] hard06 {hard:5.2f} | abstain [{fs_dec_abs.min():5.2f},{fs_dec_abs.max():5.2f}] (10th pct {np.percentile(fs_dec_abs,10):.2f}) | AMBIG max {max(fs_dec_amb):.2f} | quadratic leak ratio: median {np.median([t[0] for t in leak]):.2f} max {lk[0][0]:.2f} ({lk[0][3]}: fs dec {lk[0][1]:.2f}, cont max {lk[0][2]:.2f}) | calls: max fs over cont {max(leak_pos):.2f}')
if __name__=='__main__': main()
