"""Deterministic campaign fixtures (no model inference). Practice/selection/renamed/final-proxy/teacher/regression.

Families: discovery (18) for practice + teacher; selection (6); final (12) only as a local proxy for the hidden held-out test.
Prompt framing 'standard' has no format examples, so no example names can collide with any family.
"""
from __future__ import annotations
import copy, hashlib, json, random, sys
from dataclasses import replace
from pathlib import Path
from benchmark.contracts import Case, from_dict
from benchmark.generator import generate
from benchmark.render import prompt
from benchmark.regression import generate_regression
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'
BASE=Path('/Users/scasella/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')

def balanced(split, seed, counts):
    pool=generate(split,288,seed)
    families=list(dict.fromkeys(c.family for c in pool))
    chosen=[];used=set()
    for action,n in counts.items():
        offset={'call':0,'answer':2,'clarify':4}[action]
        for j in range(n):
            family=families[(j*5+offset)%len(families)]
            want_opaque=bool(j%2)
            # rotate templates/distractor order via pool order
            cands=[c for c in pool if c.family==family and c.spec.action==action and c.opaque==want_opaque and c.id not in used]
            c=cands[j%len(cands)]
            chosen.append(c);used.add(c.id)
    random.Random(seed).shuffle(chosen)
    return chosen

WORDS=['lumen','marble','parcel','cobalt','vessel','ripple','fable','lattice','velvet','orbit','mosaic','cipher','tango','plum','fern','zinc','quartz','sable','harbor','meadow']
def rename(case, seed, *, tokenizer=None, match_length=False):
    """Fresh identifiers; descriptions, order, values and roles preserved."""
    r=random.Random(f'{seed}:{case.id}')
    tools=copy.deepcopy(case.tools)
    old_fn=[t['function']['name'] for t in tools]
    old_keys=list(tools[0]['function']['parameters']['properties'])
    used=set(old_fn+old_keys)
    def new_name(old):
        old_len=len(tokenizer.encode(old,add_special_tokens=False)) if match_length else None
        for trial in range(20000):
            k=trial%4
            name=(r.choice(WORDS) if k==0 else r.choice(WORDS)+'_'+r.choice(WORDS) if k==1 else 'z_'+str(r.randrange(10000,99999)) if k==2 else 'z_'+str(r.randrange(1,9))+'_'+str(r.randrange(10,99)))
            if name not in used and (not match_length or len(tokenizer.encode(name,add_special_tokens=False))==old_len):
                used.add(name);return name
        raise RuntimeError('rename failed')
    fnmap={k:new_name(k) for k in old_fn}; argmap={k:new_name(k) for k in old_keys}
    for t in tools:
        f=t['function'];f['name']=fnmap[f['name']];s=f['parameters']
        s['properties']={argmap[k]:v for k,v in s['properties'].items()}
        s['required']=[argmap[k] for k in s['required']]
    c=replace(case,id=case.id+f'-renamed-{seed}',split='transfer',opaque=True,tools=tools,
              expected_name=fnmap.get(case.expected_name),expected_arguments={argmap[k]:v for k,v in case.expected_arguments.items()})
    c.validate();return c

def load_cases(name):
    return [from_dict(json.loads(l)) for l in (DATA/(name+'.jsonl')).read_text().splitlines() if l]

def _write(name, cases):
    DATA.mkdir(exist_ok=True); path=DATA/(name+'.jsonl')
    path.write_text(''.join(json.dumps(c.to_dict(),sort_keys=True)+'\n' for c in cases))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(cases),
            'class_counts':{a:sum(c.spec.action==a for c in cases) for a in ('call','answer','clarify')},
            'families':sorted(set(c.family for c in cases)),'opaque':sum(c.opaque for c in cases)}

def build_teacher(tok, max_groups=10, seed=611404):
    pool=generate('discovery',288,seed)
    output=[];groups=[];families=set()
    missing_variants=['unspecified','not given','not provided','not yet provided','not currently provided','not provided at all','not supplied by the user','not available in this request','not stated anywhere in this request','unknown to the requester','missing from the request','not specified in the request','not provided in this request','left blank','to be determined','not known yet','omitted here']
    size=lambda c:len(tok.encode(prompt(tok,c.model_input(),'standard',False),add_special_tokens=False))
    for original in pool:
        if original.spec.action!='call' or original.family in families:continue
        schema=original.tools[0]['function']['parameters'];keys=schema['required']
        fields=list(original.spec.values)
        string_fields=[(f,k) for f,k in zip(fields,keys) if type(original.spec.values[f]) is str]
        if not string_fields:continue
        description=next(t['function']['description'] for t in original.tools if t['function']['name']==original.expected_name).removesuffix('. Inert synthetic operation.')
        facts=[f"{schema['properties'][k]['description']}: {json.dumps(original.spec.values[f])}." for f,k in zip(fields,keys)]
        g=len(groups)
        call=replace(original,id=f'teacher-{g:02}-call',user='Perform this operation now: '+description+'.\n'+' '.join(facts))
        answer=replace(call,id=f'teacher-{g:02}-answer',user=call.user.replace('Perform this operation now:','Describe this operation now:',1),spec=replace(call.spec,action='answer'),expected_name=None,expected_arguments={})
        if size(call)!=size(answer):continue
        matched=None
        for field,key in string_fields:
            needle=f"{schema['properties'][key]['description']}: {json.dumps(original.spec.values[field])}."
            for phrase in missing_variants:
                repl=f"{schema['properties'][key]['description']}: {phrase}."
                clar=replace(call,id=f'teacher-{g:02}-clarify',user=call.user.replace(needle,repl,1),spec=replace(call.spec,action='clarify',missing=field),expected_name=None,expected_arguments={})
                if size(clar)==size(call):matched=clar;break
            if matched:break
        if not matched:continue
        triple=[call,answer,matched]; renamed=[]
        try:
            for c in triple:
                rc=rename(replace(c,id=call.id),611405+g,tokenizer=tok,match_length=True); renamed.append(replace(rc,id=c.id+'-renamed'))
        except RuntimeError: continue
        lengths=[size(c) for c in triple+renamed]
        if len(set(lengths))!=1:continue
        for c in triple+renamed:c.validate()
        output.extend(triple+renamed);families.add(call.family)
        groups.append({'group':g,'family':call.family,'ids':[c.id for c in triple+renamed],'prompt_tokens':lengths[0],'missing_field':matched.spec.missing})
        if len(groups)==max_groups:break
    meta=_write('teacher',output)
    stored=load_cases('teacher')
    for i,g in enumerate(groups):
        if len(set(size(c) for c in stored[6*i:6*i+6]))!=1:raise RuntimeError('serialized length mismatch')
    (DATA/'teacher_groups.json').write_text(json.dumps({'groups':groups,'manifest':meta},indent=2)+'\n')
    return {'groups':len(groups),'cases':len(output),'lengths':[g['prompt_tokens'] for g in groups]}

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    practice=balanced('discovery',611401,{'call':24,'answer':20,'clarify':20})
    selection=balanced('selection',611402,{'call':12,'answer':10,'clarify':10})
    final_proxy=balanced('final',611406,{'call':12,'answer':10,'clarify':10})
    renamed=[rename(c,611403) for c in practice]
    manifest={n:_write(n,cs) for n,cs in [('practice',practice),('selection',selection),('final_proxy',final_proxy),('renamed',renamed)]}
    reg=generate_regression(32,60131)
    (DATA/'regression.jsonl').write_text(''.join(json.dumps({'id':t['input'].id,'user':t['input'].user,'domain':t['domain'],'gold':t['gold']})+'\n' for t in reg))
    manifest['regression']={'count':len(reg),'sha256':hashlib.sha256((DATA/'regression.jsonl').read_bytes()).hexdigest()}
    manifest['teacher']=build_teacher(tok)
    sizes={}
    for n in ['practice','selection','final_proxy','renamed','teacher']:
        ls=[len(tok.encode(prompt(tok,c.model_input(),'standard',False),add_special_tokens=False)) for c in load_cases(n)]
        sizes[n]={'min':min(ls),'max':max(ls),'mean':sum(ls)/len(ls)}
    manifest['prompt_tokens']=sizes
    manifest['provenance']={'render_variant':'standard','thinking':False,'format_examples':'none','final_proxy_note':'local proxy generated from final-split family definitions; hidden held-out test is separate and never seen'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:(v if k in ('prompt_tokens','teacher','regression') else {kk:v[kk] for kk in ('count','class_counts','opaque')}) for k,v in manifest.items() if k!='provenance'},indent=1))
if __name__=='__main__': main()
