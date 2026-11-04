"""Assemble Base vs g23v3 vs g23v4 across every cohort from this workspace's runs and qwen-6's recorded runs (read-only). Writes analysis/final_tables.md + json."""
import json, numpy as np
from collections import Counter
from runtime.common import ROOT, QWEN6
from build_data import load_cases

def rows_of(label, which, root):
    p=root/'runs'/f'{label}.json'
    if not p.exists(): return None
    d=json.loads(p.read_text()); return {r['case_id']:r for r in d['rows'][which]}

def summ(rows, ids, cases):
    rs=[rows[i] for i in ids if i in rows]
    out={'n':len(rs),'lenient':sum(r['score']['correct'] for r in rs),'strict':sum(r['score']['correct_strict'] for r in rs),'false_calls':sum(bool(r['score']['false_call']) for r in rs),
         'mean_tokens':round(float(np.mean([r['output_tokens'] for r in rs])),1) if rs else None,'per_class':{}}
    for cls in ('valid_call','no_tool','need_clarification','ambiguous','resolved','prose','regression'):
        sub=[r for r in rs if cases.get(r['case_id'],{}).get('cohort')==cls]
        if sub: out['per_class'][cls]={'n':len(sub),'correct':sum(r['score']['correct'] for r in sub),'decisions':dict(Counter(str(r['score']['decision']) for r in sub))}
    return out

def paired(a,b,ids):
    ids=[i for i in ids if i in a and i in b]
    return {'wrong_to_right':sum((not a[i]['score']['correct']) and b[i]['score']['correct'] for i in ids),'right_to_wrong':sum(a[i]['score']['correct'] and not b[i]['score']['correct'] for i in ids),
            'token_identical':sum(a[i]['token_ids']==b[i]['token_ids'] for i in ids),'n':len(ids),
            'right_to_wrong_ids':[i for i in ids if a[i]['score']['correct'] and not b[i]['score']['correct']],
            'decision_changes':[(i,a[i]['score']['decision'],b[i]['score']['decision']) for i in ids if a[i]['score']['decision']!=b[i]['score']['decision']]}

def cohorts():
    C={}
    for name in ('practice','selection','final_proxy','renamed','stress'):
        for c in load_cases(name):
            act=c.spec.action; amb=c.spec.ambiguous
            coh='valid_call' if act=='call' else 'no_tool' if act=='answer' else 'ambiguous' if amb else 'need_clarification'
            if name=='stress' and act=='call': coh='resolved'
            C[c.id]={'cohort':coh,'set':name}
    for name in ('regression2','prose_qa','prose_qa2'):
        for l in (ROOT/f'data/{name}.jsonl').read_text().splitlines():
            if l: r=json.loads(l); C[r['id']]={'cohort':'prose' if name.startswith('prose') else 'regression','set':name}
    return C

def main():
    C=cohorts(); md=[]; J={}
    renA=(ROOT/'data/renamed_A.ids').read_text().split(); renB=(ROOT/'data/renamed_B.ids').read_text().split()
    st=[json.loads(l)['id'] for l in (ROOT/'data/stress.jsonl').read_text().splitlines() if l]
    plan=[('Practice (64)',[c.id for c in load_cases('practice')],('p01_baseline','original',QWEN6),('p38_v3_gen_practice','edited',QWEN6),('g05_practice_v4','g23v4',ROOT)),
          ('Selection (32)',[c.id for c in load_cases('selection')],('p40_v3_sel_both','original',QWEN6),('p40_v3_sel_both','edited',QWEN6),('g06_sel_renA_v4','g23v4',ROOT)),
          ('Renamed A (32)',renA,('p42_v3_renamedA_both','original',QWEN6),('p42_v3_renamedA_both','edited',QWEN6),('g06_sel_renA_v4','g23v4',ROOT)),
          ('Renamed B (32)',renB,('p43_v3_renamedB_both','original',QWEN6),('p43_v3_renamedB_both','edited',QWEN6),('g07_renB_final_v4','g23v4',ROOT)),
          ('Final-split proxy (32)',[c.id for c in load_cases('final_proxy')],('p41_v3_final_both','original',QWEN6),('p41_v3_final_both','edited',QWEN6),('g07_renB_final_v4','g23v4',ROOT)),
          ('Short non-tool regression2 (32)',[json.loads(l)['id'] for l in (ROOT/'data/regression2.jsonl').read_text().splitlines() if l],('p39_v3_regression2_both','original',QWEN6),('p39_v3_regression2_both','edited',QWEN6),('g08_nontool_v4','g23v4',ROOT)),
          ('Prose QA 128-token (16 derivation set)',[json.loads(l)['id'] for l in (ROOT/'data/prose_qa.jsonl').read_text().splitlines() if l],('p37_v3_prose_both','original',QWEN6),('p37_v3_prose_both','edited',QWEN6),('g08_nontool_v4','g23v4',ROOT)),
          ('Prose QA 128-token (16 fresh)',[json.loads(l)['id'] for l in (ROOT/'data/prose_qa2.jsonl').read_text().splitlines() if l],('p36_v3_prose2_both','original',QWEN6),('p36_v3_prose2_both','edited',QWEN6),('g08_nontool_v4','g23v4',ROOT)),
          ('Stress: ambiguous (32)',st[:32],('g02_stress_amb_base_v3','original',ROOT),('g02_stress_amb_base_v3','g23v3',ROOT),('g04_stress_v4','g23v4',ROOT)),
          ('Stress: resolved two-value calls (32)',st[32:],('g03_stress_res_base_v3','original',ROOT),('g03_stress_res_base_v3','g23v3',ROOT),('g04_stress_v4','g23v4',ROOT))]
    md.append('| cohort | model | lenient | strict | false calls | per class (correct/n, decisions) | mean tokens | vs Base: w->r / r->w / identical | vs g23v3: w->r / r->w / identical |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for name,ids,base,v3,v4 in plan:
        B=rows_of(base[0],base[1],base[2]); V3=rows_of(v3[0],v3[1],v3[2]); V4=rows_of(v4[0],v4[1],v4[2])
        if B is None or V3 is None or V4 is None: md.append(f'| {name} | (missing runs) |'); continue
        J[name]={}
        for mname,R in (('Base',B),('g23v3',V3),('g23v4',V4)):
            s=summ(R,ids,C); pb=paired(B,R,ids) if R is not B else None; p3=paired(V3,R,ids) if R is not V3 else None
            pc=', '.join(f"{k}:{v['correct']}/{v['n']} {v['decisions']}" for k,v in s['per_class'].items())
            md.append(f"| {name} | {mname} | {s['lenient']}/{s['n']} | {s['strict']}/{s['n']} | {s['false_calls']} | {pc} | {s['mean_tokens']} | {'-' if pb is None else f'{pb['wrong_to_right']} / {pb['right_to_wrong']} / {pb['token_identical']}'} | {'-' if p3 is None else f'{p3['wrong_to_right']} / {p3['right_to_wrong']} / {p3['token_identical']}'} |")
            J[name][mname]={'summary':s,'vs_base':pb,'vs_g23v3':p3}
    (ROOT/'analysis/final_tables.md').write_text('\n'.join(md)+'\n'); json.dump(J,open(ROOT/'analysis/final_tables.json','w'),indent=1)
    print('\n'.join(md))
if __name__=='__main__': main()
