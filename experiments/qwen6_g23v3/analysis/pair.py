"""Paired comparison of two generation runs (same items): transitions, token identity, decision changes."""
import json, sys
from fp import paired, summarize
def rows_of(label, which=None):
    d=json.load(open(f'runs/{label}.json')); r=d['rows']; return r[which] if which else (r.get('edited') or r.get('original'))
if __name__=='__main__':
    a=rows_of(sys.argv[1],sys.argv[3] if len(sys.argv)>3 else None); b=rows_of(sys.argv[2],sys.argv[4] if len(sys.argv)>4 else None)
    ids=set(x['case_id'] for x in a)&set(x['case_id'] for x in b)
    a=[x for x in a if x['case_id'] in ids]; b=sorted([x for x in b if x['case_id'] in ids],key=lambda x:[y['case_id'] for y in a].index(x['case_id']))
    p=paired(a,b); sa=summarize(a); sb=summarize(b)
    print(json.dumps({'n':len(a),'A':{k:sa[k] for k in ('correct','correct_strict','false_calls','syntax_valid','mean_output_tokens')},'B':{k:sb[k] for k in ('correct','correct_strict','false_calls','syntax_valid','mean_output_tokens')},
                      'lenient':{k:v for k,v in p['lenient'].items() if 'ids' not in k},'strict':{k:v for k,v in p['strict'].items() if 'ids' not in k},'token_identical':p['token_identical'],'call_right_to_wrong':p['call_right_to_wrong'],'per_class':p['per_class'],
                      'right_to_wrong_ids':p['lenient']['right_to_wrong_ids'],'decision_changes':p['decision_changes']},indent=None))
