from __future__ import annotations
import random, json
from .contracts import Case, SemanticSpec
from .families import FAMILIES

def generate(split: str, n: int, seed: int) -> list[Case]:
    if split not in ('discovery','selection','final'): raise ValueError(split)
    families=[f for f in FAMILIES if f.split==split]
    cases=[]
    # A quartet per family keeps siblings together. Independent naming block balances every class.
    for i in range(n):
        family=families[(i//4)%len(families)]
        r=random.Random(f'{seed}:{split}:{i}')
        cls=['call','call','answer','clarify'][i%4]
        opaque=bool(((i//4)+(i//(len(families)*4))+(i%4)//2)%2)
        op=r.randrange(2); vals={}
        for j,(key,typ,_) in enumerate(family.fields):
            vals[key]=r.randrange(4,81) if typ=='integer' else (f'0{r.randrange(10,99)}' if (i+j)%3==0 else f'{r.choice(["birch","cedar","maple","spruce"])}_{r.randrange(100,999)}')
        missing=r.choice(family.fields)[0] if cls=='clarify' else None
        ambiguous=cls=='clarify' and (i//4)%2==1
        spec=SemanticSpec(family.id,op,cls,vals,missing,ambiguous)
        # Render only after the semantic task is fixed.
        names=[f'fx_{r.randrange(10000,99999)}' for _ in range(2)] if opaque else [o[0] for o in family.operations]
        while len(set(names))!=2: names[1]=f'fx_{r.randrange(10000,99999)}'
        mapping={key:(f'p_{j+1}_{r.randrange(10,99)}' if opaque else key) for j,(key,_,_) in enumerate(family.fields)}
        properties={mapping[k]:{'type':t,'description':d} for k,t,d in family.fields}
        properties['detail' if not opaque else 'p_optional']={'type':'string','enum':['brief','full'],'default':'brief','description':'Optional output detail; omit unless requested. Default is brief.'}
        tools=[{'type':'function','function':{'name':names[j],'description':desc+'. Inert synthetic operation.', 'parameters':{'type':'object','properties':properties,'required':[mapping[k] for k,_,_ in family.fields],'additionalProperties':False}}} for j,(_,desc) in enumerate(family.operations)]
        r.shuffle(tools)
        facts=[]
        for key,typ,desc in family.fields:
            if key==missing:
                if ambiguous: facts.append(f'{desc}: either "{vals[key]}" or "{str(vals[key])+"9"}", with no preference specified.')
                else: facts.append(f'{desc}: not provided.')
            else: facts.append(f'{desc}: {json.dumps(vals[key])}.')
        r.shuffle(facts)
        operation=family.operations[op][1]
        if cls=='answer':
            request=f'I am writing documentation, not requesting an operation. Explain what it means to {operation.lower()}. Do not access or change stored data.'
        else:
            forms=[f'Please {operation.lower()}.', f'The operation I need is: {operation.lower()}.',f'Carry out this request: {operation.lower()}.']
            request=forms[(i//4)%3]
        distractor=f'Unrelated archived note: {r.choice(["birch","cedar","maple","spruce"])}_{r.randrange(100,999)}, reference 0{r.randrange(10,99)}. These are not inputs to this request.'
        pieces=[request,' '.join(facts),distractor]
        if (i//4)%2: pieces=[distractor,request,' '.join(facts)]
        c=Case(f'{split}-{seed}-{i:04}',split,family.id,family.id,seed,(i//4)%3,opaque,spec,'\n'.join(pieces),tools,names[op] if cls=='call' else None,{mapping[k]:v for k,v in vals.items()} if cls=='call' else {})
        c.validate(); cases.append(c)
    return cases

def pilot_cases(seed=7103):
    # Spread 48 discovery cases across all 18 families rather than take the first 12.
    pool=generate('discovery',144,seed)
    disc=[pool[4*f+j] for f in range(18) for j in range(4) if (f<6 or j in ((0,2) if f%2==0 else (1,3))) ]
    disc=_balance_names(disc,pool,72)
    assert len(disc)==48
    pool=generate('selection',48,seed+1)
    sel=[pool[4*f+j] for f in range(6) for j in range(4) if f<2 or j in ((0,2) if f%2==0 else (1,3))]
    sel=_balance_names(sel,pool,24)
    assert len(sel)==16
    return disc,sel

def counterfactual(case: Case) -> Case:
    """Matched field-present/absent pair; same schema, names, order, distractors."""
    from dataclasses import replace
    if case.spec.action!='call': raise ValueError('requires a call case')
    field=next(iter(case.spec.values)); value=case.spec.values[field]
    text=case.user.replace(json.dumps(value)+'.','not provided.',1)
    return replace(case,id=case.id+'-missing',user=text,spec=replace(case.spec,action='clarify',missing=field),expected_name=None,expected_arguments={})


def _balance_names(cases,pool,offset):
    counts={'call':0,'answer':0,'clarify':0}; result=[]
    for c in cases:
        cls=c.spec.action; desired=bool(counts[cls]%2); counts[cls]+=1
        if c.opaque!=desired:
            index=pool.index(c); c=pool[index+offset]
            assert c.opaque==desired
        result.append(c)
    return result
