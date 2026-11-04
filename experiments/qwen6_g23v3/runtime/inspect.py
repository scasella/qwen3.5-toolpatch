from __future__ import annotations
import json, struct, hashlib, platform, importlib.metadata, math
from pathlib import Path
from .common import BASE, BASE_REVISION, ROOT, digest, write_json

def header(path:Path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),n+8

def tensor_bits(path:Path,name:str):
    import numpy as np
    h,offset=header(path); m=h[name]; a,b=m['data_offsets']
    dtype={'BF16':'<u2','F16':'<u2','F32':'<f4'}[m['dtype']]
    with path.open('rb') as f: f.seek(offset+a); bits=np.frombuffer(f.read(b-a),dtype=dtype).copy().reshape(m['shape'])
    return m['dtype'],bits

def inspect_model(base:Path=BASE, output:Path|None=None) -> dict:
    cfg=json.loads((base/'config.json').read_text()); tensors=[]; total=0
    for file in sorted(base.glob('*.safetensors')):
        h,off=header(file)
        with file.open('rb') as f:
            for name,m in h.items():
                if name=='__metadata__': continue
                a,b=m['data_offsets']; f.seek(off+a); sha=hashlib.sha256(f.read(b-a)).hexdigest()
                text=name.startswith('model.language_model.')
                num=math.prod(m['shape']); total+=num if text else 0
                dest=name.replace('model.language_model.','language_model.model.') if text else None
                transform='identity'
                if text and 'conv1d.weight' in name: transform='moveaxis(2,1)'
                elif text and name.endswith(('.input_layernorm.weight','.post_attention_layernorm.weight','.q_norm.weight','.k_norm.weight','model.language_model.norm.weight')): transform='MLX norm convention: BF16 + 1.0, official bytes retained for export'
                tensors.append(dict(source=name,runtime=dest,shape=m['shape'],dtype=m['dtype'],entries=num,sha256=sha,transform=transform,editable=text and '.layers.' in name and len(m['shape'])==2,file=file.name))
    backend_path=Path(importlib.metadata.distribution('mlx-lm').locate_file('mlx_lm/models/qwen3_5.py'))
    result={'revision':BASE_REVISION,'source':str(base),'config':cfg,'text_unique_parameters':total,'max_changed_entries':int(total*.0025),
      'tensors':tensors,'checkpoint_hashes':{f.name:digest(f) for f in base.glob('*.safetensors')},
      'tokenizer_template_hashes':{n:digest(base/n) for n in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','merges.txt','vocab.json']},
      'backend_source_hash':digest(backend_path),
      'conversion':'Stock MLX-LM sanitize: key mapping, depthwise conv axis permutation, offset RMSNorm convention. Vision is omitted from text runtime, original vision bytes retained in export. No quantization, vocabulary or template changes.',
      'environment':{p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','tokenizers','numpy','safetensors','psutil']},'machine':platform.machine(),'python':platform.python_version()}
    if output: write_json(output,result)
    return result
