"""Independent strict native-serialization parser and semantic checker; never repairs outputs."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from typing import Any
from jsonschema import Draft202012Validator
from benchmark.contracts import Case
@dataclass(frozen=True)
class Parsed:
    action: str
    name: str | None
    arguments: dict[str,Any]
@dataclass(frozen=True)
class Score:
    parse_valid: bool
    action_correct: bool
    function_correct: bool
    required_present: bool
    values_types_correct: bool
    no_extra_arguments: bool
    schema_valid: bool
    exact_arguments: bool
    correct: bool
    false_call: bool
    unnecessary_clarification: bool
    error: str
    def to_dict(self): return asdict(self)

def parse(text: str, tools: list[dict]) -> Parsed:
    s=text.strip()
    if s=='NO_TOOL': return Parsed('answer',None,{})
    if s=='NEED_CLARIFICATION': return Parsed('clarify',None,{})
    m=re.fullmatch(r'<tool_call>\s*<function=([A-Za-z_][\w]*)>\s*(.*?)\s*</function>\s*</tool_call>',s,re.S)
    if not m: raise ValueError('invalid native call or literal response')
    name,body=m.groups(); args={}; cursor=0
    schema=next((t['function']['parameters'] for t in tools if t['function']['name']==name),None)
    for match in re.finditer(r'<parameter=([A-Za-z_][\w]*)>\s*([^<>]*?)\s*</parameter>',body,re.S):
        if body[cursor:match.start()].strip(): raise ValueError('unparsed parameter content')
        key,raw=match.groups()
        if key in args: raise ValueError('duplicate parameter')
        raw=raw.strip(); typ=(schema or {}).get('properties',{}).get(key,{}).get('type')
        # Native template prints bare strings; numeric types use JSON numeric syntax.
        if typ in ('integer','number','boolean','array','object'):
            try: val=json.loads(raw)
            except (json.JSONDecodeError,ValueError): raise ValueError('invalid typed value')
            if isinstance(val,float) and not __import__('math').isfinite(val): raise ValueError('nonfinite')
        else: val=raw
        args[key]=val; cursor=match.end()
    if body[cursor:].strip(): raise ValueError('unparsed parameter content')
    return Parsed('call',name,args)

def grade(case: Case, text: str) -> Score:
    try: p=parse(text,case.tools)
    except ValueError as e:
        return Score(False,False,False,False,False,False,False,False,False,'<tool_call>' in text and case.spec.action!='call',False,'syntax: '+str(e))
    action=p.action==case.spec.action
    function=p.name==case.expected_name
    required=values=extra=schema_ok=exact=True
    if p.action=='call':
        schema=next((t['function']['parameters'] for t in case.tools if t['function']['name']==p.name),None)
        required=bool(schema) and all(k in p.arguments for k in schema['required'])
        extra=bool(schema) and set(p.arguments)<=set(schema['properties'])
        schema_ok=bool(schema) and not list(Draft202012Validator(schema).iter_errors(p.arguments))
        if schema:
            types={'integer':int,'string':str,'boolean':bool,'object':dict,'array':list}
            schema_ok=schema_ok and all(type(v) is types[t] for k,v in p.arguments.items() if (t:=schema['properties'].get(k,{}).get('type')) in types)
        gold=case.expected_arguments
        values=all(k in p.arguments and type(p.arguments[k]) is type(v) and p.arguments[k]==v for k,v in gold.items())
        # Only equivalence: omitted optional default == explicit default, declared in schema.
        extras=set(p.arguments)-set(gold)
        defaults=bool(schema) and all(k in schema['properties'] and p.arguments[k]==schema['properties'][k].get('default',object()) for k in extras)
        exact=required and extra and schema_ok and values and defaults
    ok=action and function and exact
    err='correct' if ok else ('decision' if not action else 'function' if not function else 'arguments')
    return Score(True,action,function,required,values,extra,schema_ok,exact,ok,p.action=='call' and case.spec.action!='call',p.action=='clarify' and case.spec.action!='clarify',err)

def inert_execute(case: Case, text: str) -> dict:
    p=parse(text,case.tools)
    if p.action!='call': return {'action':p.action}
    schema=next(t['function']['parameters'] for t in case.tools if t['function']['name']==p.name)
    Draft202012Validator(schema).validate(p.arguments)
    # No dynamic dispatch, networking, filesystem operations, or eval of model text.
    from runtime.common import objhash
    return {'synthetic_receipt':objhash({'name':p.name,'arguments':p.arguments}), 'task_satisfied':grade(case,text).correct}
