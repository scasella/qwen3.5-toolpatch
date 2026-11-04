"""Hypotheses A/B/C on an fp2 triple capture: where is 'unresolved two candidate values' represented, and does it reach the decision position?
Usage: python -m analysis.diag d01b_triples_D1 [more labels...]  (labels pooled; LOFO folds are families)."""
from __future__ import annotations
import sys, json
import numpy as np
from analysis.cap import Capture, lofo, family_center, dprime, unit, FULL_ATTN, TOK, ROOT
import primitives as P

def load(labels):
    caps=[Capture(l) for l in labels]; return caps

def pooled(caps,role,layer,which='h'):
    X=[];cid=[];var=[];fam=[]
    for c in caps:
        for i in c.ids:
            v=c.h(i,role,layer) if which=='h' else (c.full(i,'mlpin23')[c.roles(i)[role]] if which=='x23' else c.full(i,'mlpin22')[c.roles(i)[role]])
            if v is None: continue
            X.append(v); cid.append(i); var.append(c.meta[i]['variant']); fam.append(c.meta[i]['family'])
    return np.stack(X),np.array(cid),np.array(var),np.array(fam)

def main():
    labels=sys.argv[1:]; caps=load(labels); out={'labels':labels}
    print(f'=== {labels}: {sum(len(c.ids) for c in caps)} items; variants',{v:int(sum((c.variant==v).sum() for c in caps)) for v in ('call','resolved','ambiguous')})
    # ---------------------------------------------------------------- A: decision position, per layer, raw LOFO vs family-centered
    print('\n=== Hypothesis A: decision position (\\n\\n). ambiguous vs {call,resolved}; LOFO diff-of-means; raw and family-triple-centered ===')
    print(' layer | raw d\'  acc  gap   | centered d\'  acc  gap | amb-vs-call d\' | amb-vs-res d\' | res-vs-call d\'(raw)')
    A={}
    for l in list(range(0,25))+['x23']:
        X,cid,var,fam=pooled(caps,'dec',l if l!='x23' else 23,'h' if l!='x23' else 'x23')
        pos=var=='ambiguous'
        r=lofo(X,pos,fam); Xc=family_center(X,fam,var); rc=lofo(Xc,pos,fam)
        m=var!='resolved'; r1=lofo(X[m],pos[m],fam[m]); m2=var!='call'; r2=lofo(X[m2],pos[m2],fam[m2]); m3=var!='ambiguous'; r3=lofo(X[m3],var[m3]=='resolved',fam[m3])
        A[str(l)]={'raw':{k:v for k,v in r.items() if k!='proj'},'centered':{k:v for k,v in rc.items() if k!='proj'},'amb_vs_call':r1['dprime'],'amb_vs_res':r2['dprime'],'res_vs_call':r3['dprime']}
        print(f" {str(l):>5} | {r['dprime']:5.2f} {r['acc']:.2f} {r['gap']:6.2f} | {rc['dprime']:5.2f} {rc['acc']:.2f} {rc['gap']:6.2f} | {r1['dprime']:5.2f} | {r2['dprime']:5.2f} | {r3['dprime']:5.2f}")
    out['A']=A
    # ---------------------------------------------------------------- B: clause roles per layer
    print('\n=== Hypothesis B: role positions. d\' (LOFO, raw) ambiguous vs resolved at shared roles; ambiguous vs {call,resolved} at end/imend/dec/sysamb ===')
    roles=['v1','conn','v2','tail','end','imend','dec','sysamb']
    B={}
    hdr=' layer | '+' | '.join(f'{r:>6}' for r in roles); print(hdr)
    for l in [4,8,10,12,14,15,16,17,18,19,20,21,22,23,24]:
        line=f' {l:>5} | '; B[l]={}
        for role in roles:
            X,cid,var,fam=pooled(caps,role,l)
            if role in ('end','imend','dec','sysamb'): pos=var=='ambiguous'
            else:
                keep=np.isin(var,['ambiguous','resolved']); X,cid,var,fam=X[keep],cid[keep],var[keep],fam[keep]; pos=var=='ambiguous'
            if pos.sum()<3 or (~pos).sum()<3: line+=f'{"-":>6} | '; continue
            r=lofo(X,pos,fam); B[l][role]={'dprime':r['dprime'],'acc':r['acc'],'gap':r['gap'],'n':[r['n_pos'],r['n_neg']]}
            line+=f"{r['dprime']:6.2f} | "
        print(line)
    out['B_roles']=B
    # tail present only in ambiguous (most resolved phrasings have no tail): also ambiguous tail vs resolved v2 (the last clause content token) is a cross-role comparison, skipped.
    # ---------------------------------------------------------------- B2: attention mass from the decision query
    print('\n=== Hypothesis B (routing): attention mass from the DECISION query, per full-attention layer and head, mean over items by class ===')
    print(' columns: clause span (v1c..end) | end | imend | self | pos0 | [ambiguous / resolved / call]')
    R={}
    for li,l in enumerate(FULL_ATTN):
        R[l]={}
        for h in range(8):
            acc={k:{'ambiguous':[],'resolved':[],'call':[]} for k in ('clause','end','imend','self','pos0','v2','conn')}
            for c in caps:
                for i in c.ids:
                    rows=c.rows_attn(i)[li,0,h]; ro=c.roles(i); v=c.meta[i]['variant']
                    a=ro.get('v1c',ro['v1']); b=ro['end']
                    acc['clause'][v].append(rows[a:b+1].sum()); acc['end'][v].append(rows[b]); acc['imend'][v].append(rows[ro['imend']]); acc['self'][v].append(rows[ro['dec']]); acc['pos0'][v].append(rows[0])
                    if ro.get('v2') is not None: acc['v2'][v].append(rows[ro['v2']]); acc['conn'][v].append(rows[ro['conn']])
            R[l][h]={k:{v:float(np.mean(x)) if x else None for v,x in d.items()} for k,d in acc.items()}
            s=' '.join(f"{k}:{R[l][h][k]['ambiguous'] or 0:.3f}/{R[l][h][k]['resolved'] or 0:.3f}/{R[l][h][k]['call'] or 0:.3f}" for k in ('clause','end','imend','self','pos0'))
            print(f' L{l:2d} h{h} | {s}')
    out['B_rows']=R
    # which prompt positions receive the most decision-query attention at layer 23 (top 8 per head), by token text
    # ---------------------------------------------------------------- C: per-head outputs at the decision position (layers 15,19,23)
    print('\n=== Hypothesis C: per-head gated output (o_proj input slice) at the decision position, ambiguous vs {call,resolved}, LOFO d\' ===')
    C={}
    for l in (11,15,19,23):
        line=f' L{l:2d} |'; C[l]={}
        for h in range(8):
            X=[];var=[];fam=[]
            for c in caps:
                for i in c.ids:
                    X.append(c.attn(i,'dec',l)[h*256:(h+1)*256]); var.append(c.meta[i]['variant']); fam.append(c.meta[i]['family'])
            X=np.stack(X); var=np.array(var); fam=np.array(fam); r=lofo(X,var=='ambiguous',fam); C[l][h]=r['dprime']; line+=f" h{h}:{r['dprime']:5.2f}"
        print(line)
    out['C_heads']=C
    # ---------------------------------------------------------------- near-dead neurons at layers 18/22 (candidate stage-1 hosts) from colstats
    print('\n=== near-dead neurons (max |activation| over every prompt position of all items), layers 15/19/21/22/23 ===')
    ND={}
    for l in (15,19,21,22,23):
        mx=np.max(np.concatenate([c.z[f'colstats_L{l}_max'] for c in caps]),0); order=np.argsort(mx)
        ND[l]={'quietest':[(int(i),float(mx[i])) for i in order[:12]],'n_below_0.05':int((mx<0.05).sum()),'n_below_0.1':int((mx<0.1).sum()),'median':float(np.median(mx))}
        print(f' L{l}: n(max|a|<0.05)={ND[l]["n_below_0.05"]} n(<0.1)={ND[l]["n_below_0.1"]} median {ND[l]["median"]:.2f} quietest {ND[l]["quietest"][:8]}')
    out['near_dead']=ND
    json.dump(out,open(ROOT/'analysis'/f'diag_{"_".join(labels)}.json','w'),indent=1,default=float)
if __name__=='__main__': main()
