"""Write edits/g23v4.npz (+ .spec.json, .vectors.npz) and the controls on the same support: g23v4_half, g23v4_double, g23v4_rev, g23v4_rand17 (norm-matched random stage-1
and stage-2 vectors, g23v3 rows kept). Zero control = edits/g23v3.npz itself. Verifies that the J1/J2 rows are bit-identical to edits/g23v3.npz."""
import json, numpy as np
from runtime.common import ROOT
import g23v4_build as B
BASEP={'a_g':0.5,'theta1':11.7,'k1':4.0,'b':1.0,'lam':0.3,'lam2':0.5,'theta3':2.0,'k3':2.0,'a3':0.1,'rho':1.0,'form':'quad'}
VARIANTS={'g23v4':{},'g23v4_half':{'a3':0.05,'b':0.5},'g23v4_double':{'a3':0.2,'b':2.0},'g23v4_rev':{'a3':-0.1},'g23v4_rand17':{'random_seed':17}}
v3=np.load(ROOT/'edits/g23v3.npz')
for label,over in VARIANTS.items():
    params={**BASEP,**over}
    if (ROOT/f'edits/{label}.npz').exists(): print('exists',label); continue
    path,spec=B.write_edit(label,params)
    z=np.load(path); names=sorted(z.files)
    # bit-identity of the g23v3 rows/columns inside the layer-23 tensors
    ok=True
    for n in v3.files:
        a=z[n]; b=v3[n]
        if 'down_proj' in n: ok&=np.array_equal(a[:,[B.J1,B.J2]],b[:,[B.J1,B.J2]])
        else: ok&=np.array_equal(a[[B.J1,B.J2]],b[[B.J1,B.J2]])
    rec=spec['receipts']; print(label,'tensors',len(names),'layers',sorted({int(n.split('.')[3]) for n in names}),'entries',spec['total_changed_entries'],'fraction %.3e'%spec['fraction_of_text_params'],'g23v3 rows identical:',ok)
