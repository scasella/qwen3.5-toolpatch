"""Compare the fresh stock-process streams (runs/s01_v4_stock) with the in-process runs: original vs g09/g02/g03, edited vs g05/g04."""
import json
from runtime.common import ROOT
s=json.load(open(ROOT/'runs/s01_v4_stock.json'))
def rows(lab,w): return {r['case_id']:r for r in json.load(open(ROOT/f'runs/{lab}.json'))['rows'][w]}
ref_o={}; ref_e={}
for lab,w in (('g09_practice_orig','original'),('g02_stress_amb_base_v3','original'),('g03_stress_res_base_v3','original')): ref_o.update(rows(lab,w))
for lab,w in (('g05_practice_v4','g23v4'),('g04_stress_v4','g23v4')): ref_e.update(rows(lab,w))
for cp,ref in (('original',ref_o),('edited',ref_e)):
    rs=s['rows'][cp]; same=[r['case_id'] for r in rs if ref[r['case_id']]['token_ids']==r['token_ids']]; diff=[r['case_id'] for r in rs if ref[r['case_id']]['token_ids']!=r['token_ids']]
    print(cp,'identical',len(same),'different',len(diff),diff[:5],'| lenient',sum(r['score']['correct'] for r in rs),'/',len(rs))
print('export:',s['export'],'| paired original->edited:',{k:v for k,v in s['paired']['lenient'].items() if 'ids' not in k},'token_identical',s['paired']['token_identical'])
