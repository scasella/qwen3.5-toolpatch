"""Matched triples and the Ambiguity Stress Suite (deterministic, no model inference).

Per family a base call case from the campaign generator; from it, by replacing exactly one fact clause:
  (i)  call      : '<desc>: <X>.'                                   (expected call with X)
  (ii) resolved  : two values mentioned, choice unambiguous (X)     (expected call with X)
  (iii) ambiguous: two candidate values, no choice                  (expected NEED_CLARIFICATION)
Phrasing set A (derivation; discovery families only) / set B (held-out; selection+final families, renamed twins, stress suite).
Sets written to data/: triples_D1 (discovery, original names), triples_D2 (renamed twins), triples_D3 (selection+final, set B),
stress (64: 32 ambiguous + 32 resolved; discovery/selection/renamed; mostly set-B phrasings). Each set has a sidecar data/<set>.meta.json
with X, Y, phrasing id, clause text and char spans of the roles used for position tagging."""
from __future__ import annotations
import json, random, re, hashlib, sys
from dataclasses import replace
from pathlib import Path
from benchmark.contracts import Case, SemanticSpec
from benchmark.generator import generate
from benchmark.families import FAMILIES
from build_data import rename, load_cases, DATA, BASE
from benchmark.render import prompt

J=json.dumps
# ---------------------------------------------------------------- phrasings: templates over X (chosen / first candidate) and Y (other)
AMB_A={'A1':'either "{X}" or "{Y}", with no preference specified',   # the qwen-6 phrasing (both values quoted)
       'A2':'{X} or possibly {Y} — I haven\'t decided',
       'A3':'unsure whether to use {X} or {Y}',
       'A4':'still deciding between {X} and {Y}',
       'A5':'{X}, or maybe {Y}; no choice has been made yet',
       'A6':'one of {X} or {Y} (undecided which)'}
AMB_B={'B1':'{X} or {Y}, whichever — I have not picked one',
       'B2':'could be {X}, could be {Y}; it is not settled',
       'B3':'not sure yet: {X} or {Y}',
       'B4':'torn between {X} and {Y}',
       'B5':'{X} or {Y}, I keep going back and forth',
       'B6':'maybe {X}, maybe {Y}, I cannot say which'}
RES_A={'R1':'use {X} rather than {Y}',
       'R2':'previously {Y}, but today use {X}',
       'R3':'{X} (not {Y})',
       'R4':'{X} instead of {Y}',
       'R5':'I considered {Y} but settled on {X}',
       'R6':'was going to be {Y}; final answer is {X}'}
RES_B={'S1':'definitely {X}, not {Y}',
       'S2':'{X}, replacing the earlier {Y}',
       'S3':'changed from {Y} to {X}, so use {X}',
       'S4':'{Y} was a typo; the correct value is {X}',
       'S5':'{X} — ignore {Y}, which was last week\'s value',
       'S6':'after comparing {X} and {Y}, go with {X}'}
PHRASINGS={**AMB_A,**AMB_B,**RES_A,**RES_B}
SEEN=set(AMB_A)|set(RES_A)

def other_value(x, rng):
    if isinstance(x,int):
        while True:
            y=rng.randrange(4,81)
            if y!=x: return y
    if re.fullmatch(r'0\d\d',x):
        while True:
            y=f'0{rng.randrange(10,99)}'
            if y!=x: return y
    while True:
        y=f'{rng.choice(["birch","cedar","maple","spruce"])}_{rng.randrange(100,999)}'
        if y!=x: return y

def render_val(v, quoted):
    return str(v) if quoted else J(v)      # quoted phrasing (A1) wraps the bare string itself in quotes; others use JSON rendering

def clause_for(desc, pid, X, Y):
    quoted=(pid=='A1'); t=PHRASINGS[pid]
    body=t.format(X=render_val(X,quoted),Y=render_val(Y,quoted))
    clause=f'{desc}: {body}.'
    # char spans (relative to clause) of roles: first/second value mention, X, Y, connector (text between the values), tail (after second value, before '.')
    xs=('"%s"'%X if quoted else J(X)); ys=('"%s"'%Y if quoted else J(Y))
    px=clause.index(xs); py=clause.index(ys)
    first,second=((px,xs),(py,ys)) if px<py else ((py,ys),(px,xs))
    spans={'x':[px,px+len(xs)],'y':[py,py+len(ys)],'v1':[first[0],first[0]+len(first[1])],'v2':[second[0],second[0]+len(second[1])],
           'conn':[first[0]+len(first[1]),second[0]],'tail':[second[0]+len(second[1]),len(clause)-1],'end':[len(clause)-1,len(clause)]}
    return clause,spans

def variants(base:Case, field_idx:int, amb_pid:str, res_pid:str, seed:str):
    fam=next(f for f in FAMILIES if f.id==base.family); key,typ,desc=fam.fields[field_idx]
    X=base.spec.values[key]; rng=random.Random(seed); Y=other_value(X,rng)
    needle=f'{desc}: {J(X)}.'; assert base.user.count(needle)==1, (base.id,needle)
    out=[]
    cl,sp=clause_for(desc,amb_pid,X,Y); amb=replace(base,id=base.id+'-amb-'+amb_pid,user=base.user.replace(needle,cl,1),spec=replace(base.spec,action='clarify',missing=key,ambiguous=True),expected_name=None,expected_arguments={})
    cl2,sp2=clause_for(desc,res_pid,X,Y); res=replace(base,id=base.id+'-res-'+res_pid,user=base.user.replace(needle,cl2,1),spec=replace(base.spec,action='call',ambiguous=False))
    callspans={'x':[len(desc)+2,len(desc)+2+len(J(X))],'v1':[len(desc)+2,len(desc)+2+len(J(X))],'end':[len(needle)-1,len(needle)]}
    call=replace(base,id=base.id+'-call')
    for c,variant,pid,clause,spans in ((call,'call','-',needle,callspans),(res,'resolved',res_pid,cl2,sp2),(amb,'ambiguous',amb_pid,cl,sp)):
        c.validate(); out.append((c,{'variant':variant,'phrasing':pid,'phrasing_seen':(pid in SEEN) if pid!='-' else None,'field':key,'X':X,'Y':Y,'clause':clause,'spans':spans,'family':base.family,'opaque':base.opaque,'base_id':base.id}))
    return out

def base_calls(split, seed, per_family=1):
    pool=generate(split,288 if split=='discovery' else 96,seed); fams=[f.id for f in FAMILIES if f.split==split]; out=[]
    for fi,fam in enumerate(fams):
        cands=[c for c in pool if c.family==fam and c.spec.action=='call']
        want=[bool((fi+j)%2) for j in range(per_family)]
        picked=[]
        for w in want:
            c=next(c for c in cands if c.opaque==w and c not in picked); picked.append(c)
        out+=picked
    return out

def retag_renamed(c:Case, meta:dict, tok, seed):
    """rename() needs a case whose expected_name/arguments are consistent; for clarify variants expected_name is None -> rename a call twin and transplant."""
    callc=replace(c,spec=replace(c.spec,action='call',missing=None,ambiguous=False),expected_name=meta['_call_name'],expected_arguments=meta['_call_args'])
    rc=rename(callc,seed,tokenizer=tok,match_length=False)
    if c.spec.action=='clarify': rc=replace(rc,spec=c.spec,expected_name=None,expected_arguments={})
    rc=replace(rc,id=c.id+'-renamed'); rc.validate(); return rc

def write_set(name, items, tok):
    path=DATA/f'{name}.jsonl'; path.write_text(''.join(J(c.to_dict(),sort_keys=True)+'\n' for c,_ in items))
    meta={}
    for c,m in items:
        m=dict(m); m.pop('_call_name',None); m.pop('_call_args',None)
        p=prompt(tok,c.model_input(),'standard',False); m['prompt_tokens']=len(tok.encode(p,add_special_tokens=False))
        assert m['prompt_tokens']+128<=2048
        assert p.count(m['clause'])==1, (c.id, m['clause'])
        meta[c.id]=m
    (DATA/f'{name}.meta.json').write_text(J(meta,indent=1,sort_keys=True))
    counts={}
    for c,m in items: counts[m['variant']]=counts.get(m['variant'],0)+1
    return {'count':len(items),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()[:16],'variants':counts,'phrasings':sorted({m['phrasing'] for _,m in items})}

def existing_ambiguous(set_name, n):
    out=[]
    for c in load_cases(set_name):
        if c.spec.action=='clarify' and c.spec.ambiguous:
            fam=next(f for f in FAMILIES if f.id==c.family); key=c.spec.missing; desc=next(d for k,_,d in fam.fields if k==key)
            X=c.spec.values[key]; Y=str(X)+'9'; clause=f'{desc}: either "{X}" or "{Y}", with no preference specified.'; assert c.user.count(clause)==1
            _,sp=clause_for(desc,'A1',X,Y)
            out.append((c,{'variant':'ambiguous','phrasing':'A1','phrasing_seen':True,'field':key,'X':X,'Y':Y,'clause':clause,'spans':sp,'family':c.family,'opaque':c.opaque,'base_id':c.id,'existing':set_name}))
    return out[:n]

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    man={}
    # ---- D1: discovery families, set-A phrasings (derivation) + 10 existing practice ambiguous items
    ambA=list(AMB_A); resA=list(RES_A); d1=[]
    for fi,base in enumerate(base_calls('discovery',711401)):
        fam=next(f for f in FAMILIES if f.id==base.family)
        trip=variants(base,fi%len(fam.fields),ambA[fi%6],resA[(fi//2)%6],f'd1:{base.id}')
        for c,m in trip: m['_call_name']=base.expected_name; m['_call_args']=base.expected_arguments
        d1+=trip
    d1+=existing_ambiguous('practice',10)
    man['triples_D1']=write_set('triples_D1',d1,tok)
    # ---- D2: renamed twins of the 54 D1 triples + the 10 renamed twins of the existing practice ambiguous items (data/renamed.jsonl)
    d2=[]
    for c,m in d1[:54]:
        rc=retag_renamed(c,m,tok,711403); mm=dict(m); mm['renamed']=True; mm['base_id']=c.id; d2.append((rc,mm))
    ren={c.id:c for c in load_cases('renamed')}
    for c,m in existing_ambiguous('practice',10):
        rc=ren[c.id+'-renamed-611403']; mm=dict(m); mm['renamed']=True; mm['existing']='renamed'; d2.append((rc,mm))
    man['triples_D2']=write_set('triples_D2',d2,tok)
    # ---- D3: selection (6) + final (12) families, set-B phrasings (held-out diagnosis) + 5 selection + 5 final existing ambiguous items
    ambB=list(AMB_B); resB=list(RES_B); d3=[]
    bases=base_calls('selection',711404)+base_calls('final',711405)
    for fi,base in enumerate(bases):
        fam=next(f for f in FAMILIES if f.id==base.family)
        trip=variants(base,(fi+1)%len(fam.fields),ambB[fi%6],resB[(fi//3)%6],f'd3:{base.id}')
        for c,m in trip: m['_call_name']=base.expected_name; m['_call_args']=base.expected_arguments
        d3+=trip
    d3+=existing_ambiguous('selection',5)+existing_ambiguous('final_proxy',5)
    man['triples_D3']=write_set('triples_D3',d3,tok)
    # ---- Stress suite: 32 ambiguous + 32 resolved; discovery (fresh base cases, other field where possible), selection, renamed; mostly held-out phrasings,
    #      plus the task's example phrasings (A1/A2/A3, R1/R2) on unseen families or names (flagged phrasing_seen)
    disc=base_calls('discovery',711406); sel=base_calls('selection',711407,per_family=2)
    amb_items=[]; res_items=[]
    plan_amb=[('disc',i,ambB[i%6],False) for i in range(12)]+[('sel',i,ambB[(i+2)%6],False) for i in range(6)]+[('sel',6+i,['A1','A2','A3'][i%3],False) for i in range(3)] \
             +[('disc',12+i,ambB[(i+3)%6],True) for i in range(6)]+[('sel',9+i,['A2','A3','B1','B2','B6'][i%5],True) for i in range(3)]+[('sel',12-1-i,['A1','B4','B5'][i%3],True) for i in range(2)]
    plan_res=[('disc',i,resB[i%6],False) for i in range(12)]+[('sel',i,resB[(i+2)%6],False) for i in range(6)]+[('sel',6+i,['R1','R2','R4'][i%3],False) for i in range(3)] \
             +[('disc',12+i,resB[(i+3)%6],True) for i in range(6)]+[('sel',9+i,['R1','R2','S1','S2','S6'][i%5],True) for i in range(3)]+[('sel',12-1-i,['R2','S3','S5'][i%3],True) for i in range(2)]
    assert len(plan_amb)==32 and len(plan_res)==32
    def build(plan, which):
        out=[]
        for k,(src,idx,pid,ren_) in enumerate(plan):
            base=(disc if src=='disc' else sel)[idx]; fam=next(f for f in FAMILIES if f.id==base.family)
            fidx=(idx+k)%len(fam.fields)
            trip=variants(base,fidx,pid if which=='amb' else 'A1',pid if which=='res' else 'R1',f'stress:{which}:{k}:{base.id}')
            c,m=trip[2] if which=='amb' else trip[1]
            m['_call_name']=base.expected_name; m['_call_args']=base.expected_arguments; m['stress_index']=k; m['renamed']=ren_
            c=replace(c,id=f'stress-{which}-{k:02d}-'+c.id)
            if ren_: c=retag_renamed(c,m,tok,711408+k)
            out.append((c,m))
        return out
    stress=build(plan_amb,'amb')+build(plan_res,'res')
    man['stress']=write_set('stress',stress,tok)
    man['phrasings']={'A_ambiguous_derivation':AMB_A,'B_ambiguous_heldout':AMB_B,'R_resolved_derivation':RES_A,'S_resolved_heldout':RES_B}
    man['stress_breakdown']={'ambiguous':{'seen_phrasing':sum(m['phrasing_seen'] for c,m in stress[:32]),'renamed':sum(m['renamed'] for c,m in stress[:32]),'families':len({m['family'] for c,m in stress[:32]})},
                             'resolved':{'seen_phrasing':sum(m['phrasing_seen'] for c,m in stress[32:]),'renamed':sum(m['renamed'] for c,m in stress[32:]),'families':len({m['family'] for c,m in stress[32:]})}}
    (DATA/'triples_manifest.json').write_text(J(man,indent=1))
    print(J({k:v for k,v in man.items() if k!='phrasings'},indent=1))
if __name__=='__main__': main()
