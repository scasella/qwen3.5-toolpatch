"""Tensor-by-tensor CPU byte audit of stock MLX-LM conversion; no forwards."""
import hashlib,json
import numpy as np
from .common import ROOT,BASE,digest,write_json
from .inspect import tensor_bits

def validate_loaded(model,mx):
    from mlx.utils import tree_flatten
    from patches.store import bf16,f32
    manifest=json.loads((ROOT/'environment/architecture.json').read_text())
    for name,sha in manifest['checkpoint_hashes'].items():
        if digest(BASE/name)!=sha:raise ValueError('immutable base checkpoint hash mismatch')
    params=dict(tree_flatten(model.parameters()));expected=[t for t in manifest['tensors'] if t['runtime']]
    if set(params)!={t['runtime'] for t in expected}:raise ValueError('unexpected/missing text tensors after stock conversion')
    rows=[]
    for t in expected:
        dt,bits=tensor_bits(BASE/t['file'],t['source']);x=bits
        if t['transform']=='moveaxis(2,1)':x=np.moveaxis(x,2,1)
        elif t['transform'].startswith('MLX norm'):x=bf16(f32(x)+np.float32(1))
        value=params[t['runtime']]
        mx.eval(value)
        actual=np.array(value.view(mx.uint16) if dt=='BF16' else value)
        if not np.array_equal(x,actual):raise ValueError('conversion mismatch: '+t['source'])
        rows.append({'source':t['source'],'runtime':t['runtime'],'source_sha256':t['sha256'],'runtime_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'runtime_shape':list(value.shape),'runtime_dtype':str(value.dtype),'transform':t['transform']})
        del bits,x,actual
    write_json(ROOT/'environment/conversion_verified.json',{'verified':True,'method':'CPU byte comparison of every loaded text tensor after explicitly reproduced stock sanitizer transform; no model forwards','tensors':rows})
    return True
