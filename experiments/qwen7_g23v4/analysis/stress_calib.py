"""Per-item mechanism readout on the 32 ambiguous stress items under g23v4 (runs/c02_stress_calib): stage-1 pre-activation z and activation at the clause end,
head-6 attention on the clause end, J3 activation at the decision, decision argmax; joined with the generation outcome (g04)."""
import json, numpy as np
from analysis.cap import Capture, TOK, ROOT
import g23v4_build as B
c=Capture('c02_stress_calib'); R=B.rows(json.load(open(ROOT/'edits/g23v4.spec.json'))['params'])
g04={r['case_id']:r for r in json.load(open(ROOT/'runs/g04_stress_v4.json'))['rows']['g23v4']}
z=np.load(ROOT/'runs/c02_stress_calib.npz')
print(' item | phrasing | renamed | J0 z (gate/k1) at end | J0 act | alpha6 | J3 act | argmax | generated decision')
rows=[]
for i in c.ids:
    ro=c.roles(i); m=c.meta[i]; mlp22=c.z[f'mlp_{i}'][c.pos_index(i,'end'),0].astype(np.float32); mlp23=c.z[f'mlp_{i}'][c.pos_index(i,'dec'),1].astype(np.float32)
    a6=float(c.rows_attn(i)[5,0,6,ro['end']]); am=c.rows[i]['decision']['topk'][0]['text']
    rows.append((i,m['phrasing'],m['renamed'],float(mlp22[B.J0]),a6,float(mlp23[B.J3]),am,g04[i]['score']['decision']))
for r in sorted(rows,key=lambda r:r[3]): print(f' {r[0][:46]:46s} {r[1]} {"ren" if r[2] else "   "} | J0 act {r[3]:7.2f} | alpha6 {r[4]:.3f} | J3 {r[5]:6.2f} | {r[6]!r:8s} | {r[7]}')
byp={}
for r in rows: byp.setdefault(r[1],[]).append(r[7]=='clarify')
print('by phrasing:',{k:f'{sum(v)}/{len(v)}' for k,v in sorted(byp.items())})
