"""Assemble paired result tables from runs/*.json for results.md."""
import json, sys, os
from fp import paired, summarize
def load(label):
    p=f'runs/{label}.json'
    return json.load(open(p)) if os.path.exists(p) else None
def row(name, a, b):
    p=paired(a,b); sa=summarize(a); sb=summarize(b)
    cls=lambda s,c,k: s['per_class'].get(c,{}).get(k,'-')
    return f"| {name} | {len(a)} | {sa['correct']} -> {sb['correct']} | {sa['correct_strict']} -> {sb['correct_strict']} | {cls(sa,'call','correct')} -> {cls(sb,'call','correct')} (n={cls(sa,'call','n')}) | {cls(sa,'answer','correct')} -> {cls(sb,'answer','correct')} (n={cls(sa,'answer','n')}) | {cls(sa,'clarify','correct')} -> {cls(sb,'clarify','correct')} (n={cls(sa,'clarify','n')}) | {sa['false_calls']} -> {sb['false_calls']} | {p['lenient']['wrong_to_right']} / {p['lenient']['right_to_wrong']} | {p['token_identical']} | {sa['mean_output_tokens']:.1f} -> {sb['mean_output_tokens']:.1f} |"
HEAD="| set (edit) | n | lenient correct | strict correct | calls | NO_TOOL | NEED_CLARIFICATION | false calls | wrong->right / right->wrong | token-identical | mean new tokens |\n|---|---|---|---|---|---|---|---|---|---|---|"
def main():
    base=load('p01_baseline')['rows']['original']
    lines=[HEAD]
    for name,lab in [('practice, g23v3 (final)','p38_v3_gen_practice'),('practice, g23v3 random','p45_v3_rand13'),('practice, g23v2 (intended)','p19_v2_gen_practice'),('practice, g23v2 half','p25_v2_half'),('practice, g23v2 double','p26_v2_double'),('practice, g23v2 reversed','p27_v2_rev'),('practice, g23v2 random','p28_v2_rand11'),
                     ('practice, g23 v1 intended','p07_gen_intended'),('practice, v1 zero','p08_gen_zero'),('practice, v1 half','p09_gen_half'),('practice, v1 double','p10_gen_double'),('practice, v1 reversed','p11_gen_rev'),('practice, v1 random','p12_gen_rand7')]:
        d=load(lab)
        if d: lines.append(row(name,base,d['rows']['edited']))
    for name,lab in [('selection (both), g23v3','p40_v3_sel_both'),('renamed A (both), g23v3','p42_v3_renamedA_both'),('renamed B (both), g23v3','p43_v3_renamedB_both'),('final proxy (both), g23v3','p41_v3_final_both'),('regression2 fresh non-tool (both), g23v3','p39_v3_regression2_both'),('prose QA fresh (both), g23v3','p36_v3_prose2_both'),('prose QA derivation set (both), g23v3','p37_v3_prose_both'),('regression2 fresh non-tool (both), g23v2','p30_v2_regression2_both'),('prose QA (both), g23v2','p31_v2_prose_both'),('prose QA (both), v1','p32_v1_prose_both'),('practice concise framing (both), g23v2','p33_v2_concise_both'),('selection (both), g23v2','p21_v2_sel_both'),('renamed A (both), g23v2','p22_v2_renamedA_both'),('renamed B (both), g23v2','p23_v2_renamedB_both'),('final proxy (both), g23v2','p24_v2_final_both'),('regression non-tool (both), g23v2','p20_v2_regression_both'),
                     ('selection (both), v1','p13_sel_both'),('renamed A (both), v1','p14_renamedA_both'),('renamed B (both), v1','p15_renamedB_both'),('regression non-tool (both), v1','p18_regression_both')]:
        d=load(lab)
        if d: lines.append(row(name,d['rows']['original'],d['rows']['edited']))
    for name,lab in [('stock reload (fresh process), g23v3','p44_v3_stock'),('stock reload (fresh process), g23v2','p34_v2_stock')]:
        d=load(lab)
        if d: lines.append(row(name,d['rows']['original'],d['rows']['edited']))
    print('\n'.join(lines))
    for lab in ('p44_v3_stock','p34_v2_stock'):
        d=load(lab)
        if d: print(f'\n{lab} stock vs in-process:',d.get('stock_vs_inprocess'))
if __name__=='__main__': main()
