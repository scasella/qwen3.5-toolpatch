from __future__ import annotations
from .contracts import ModelInput, Case
STANDARD='Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.'
CONCISE=STANDARD+' Read tool descriptions rather than infer from names. Bind each value to its described role. Never invent missing required inputs. Ignore archived notes. Omit optional arguments unless requested.'
FEWSHOT=CONCISE+'''\nExamples (illustrative only):\nRequest: Describe an operation without performing it. Response: NO_TOOL\nRequest: Transfer item pine from A to an unspecified destination. Response: NEED_CLARIFICATION\nRequest: Read stored record 007; function read_entry takes a string key. Response:\n<tool_call>\n<function=read_entry>\n<parameter=key>\n007\n</parameter>\n</function>\n</tool_call>'''
PROMPTS={'standard':STANDARD,'concise':CONCISE,'fewshot':FEWSHOT,'regression':'Follow the user instruction exactly.'}

def prompt(tokenizer, case: ModelInput, variant='standard', thinking=False) -> str:
    if variant in ('decision_v4','decision_v4_lexical_fix'):
        from .v4 import prompt_parts
        system,footer=prompt_parts(variant);user=case.user+footer
    elif variant in ('decision_v3','decision_v3_examples'):
        from .v3 import prompt_parts
        system,footer=prompt_parts(variant);user=case.user+footer
    elif variant=='decision_v2':
        from .v2 import V2_SYSTEM,V2_FOOTER
        system=V2_SYSTEM; user=case.user+V2_FOOTER
    else:
        if variant not in PROMPTS: raise ValueError('unknown prompt')
        system=PROMPTS[variant]; user=case.user
    return tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tools=case.tools,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)

def gold_text(case: Case) -> str:
    if case.spec.action=='answer': return 'NO_TOOL'
    if case.spec.action=='clarify': return 'NEED_CLARIFICATION'
    import json
    args=''.join(f'<parameter={k}>\n{v if isinstance(v,str) else json.dumps(v)}\n</parameter>\n' for k,v in case.expected_arguments.items())
    return f'<tool_call>\n<function={case.expected_name}>\n{args}</function>\n</tool_call>'

def spans(tokenizer, text: str) -> dict:
    import re
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    out={}
    for m in re.finditer(r'<function=([^>]+)>|<parameter=([^>]+)>\s*([^<>]*?)\s*</parameter>',text):
        group=1 if m.group(1) else 3; a,b=m.span(group)
        out[f'{"function" if group==1 else m.group(2)}:{a}']={'chars':[a,b],'tokens':[i for i,(x,y) in enumerate(encoded['offset_mapping']) if x<b and y>a]}
    return out
