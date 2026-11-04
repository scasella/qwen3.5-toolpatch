"""Compare the real edited model's teacher-forced decision logits/attention (runs/<calib label>) with the replay prediction (analysis/sim4_last.json)."""
import json, sys, numpy as np
from analysis.cap import Capture, TOK, ROOT
lab=sys.argv[1] if len(sys.argv)>1 else 'c01_calib_D1'
c=Capture(lab); sim={r['id']:r for r in json.load(open(ROOT/'analysis/sim4_last.json'))['rows']}
J0,J3=2525,2331
agree=0; errs=[]; a6=[]; j0_amb=[]; j0_non=[]; j3_amb=[]; j3_non=[]; rows=[]
for i in c.ids:
    lg=c.logits(i); s=sim[i]; v=c.meta[i]['variant']; ro=c.roles(i)
    am=int(lg.argmax()); agree+=int(am==s['v4_argmax'])
    e={'NE':float(lg[TOK['NE']]-s['logit_NE']),'NO':float(lg[TOK['NO']]-s['logit_NO']),'call':float(lg[TOK['call']]-s['logit_call'])}; errs.append(max(abs(x) for x in e.values()))
    rows_=c.rows_attn(i)[5,0]   # layer 23, dec query: (8,T)
    mlp22=c.z[f'mlp_{i}'][c.pos_index(i,'end'),0].astype(np.float32); mlp23=c.z[f'mlp_{i}'][c.pos_index(i,'dec'),1].astype(np.float32)
    if v=='ambiguous': a6.append(float(rows_[6,ro['end']])); j0_amb.append(float(mlp22[J0])); j3_amb.append(float(mlp23[J3]))
    else: j0_non.append(float(mlp22[J0])); j3_non.append(float(mlp23[J3]))
    rows.append((i,v,am==s['v4_argmax'],c.rows[i]['decision']['topk'][0]['text'],round(max(abs(x) for x in e.values()),3),round(float(rows_[6,ro['end']]),3),round(float(mlp22[J0]),2),round(s['act0_end'],2),round(float(mlp23[J3]),2),round(s['j3_act_pred'],2)))
print(f'{lab}: argmax agreement {agree}/{len(c.ids)} | |logit err| on NE/NO/call: median {np.median(errs):.3f} max {max(errs):.3f}')
print(f' alpha6 at marker (real): min {min(a6):.3f} mean {np.mean(a6):.3f} | J0 act at end: ambiguous [{min(j0_amb):.1f},{max(j0_amb):.1f}] non-ambiguous max {max(j0_non):.4f} | J3 act at dec: ambiguous [{min(j3_amb):.1f},{max(j3_amb):.1f}] non-ambiguous max {max(j3_non):.4f}')
print(' id | variant | agree | real argmax | max err | alpha6 | J0 real | J0 sim | J3 real | J3 sim')
for r in rows:
    if (not r[2]) or r[4]>0.5 or r[1]=='ambiguous' and r[5]<0.9: print('  ',r)
print(' sample:',rows[0],rows[1],rows[2])
