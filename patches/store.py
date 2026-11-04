from __future__ import annotations
import hashlib, json, re, shutil, math, os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from runtime.common import BASE, BASE_REVISION, ROOT, write_json, digest, objhash
from runtime.inspect import tensor_bits, header
BASE_FILE=next(BASE.glob('*.safetensors'))

def bhash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(bits): return (bits.astype(np.uint32)<<16).view(np.float32)
def bf16(values):
    v=np.asarray(values,dtype=np.float32)
    if not np.isfinite(v).all(): raise ValueError('nonfinite values')
    bits=v.view(np.uint32); return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)

@dataclass(frozen=True)
class PatchSpec:
    patch_id: str
    tensor: str
    columns: tuple[int,...]
    strength: float
    derivation: str
    provenance: str='ToolPatch discovery'
    def validate(self):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',self.patch_id): raise ValueError('invalid patch id')
        if not self.columns or len(self.columns)!=len(set(self.columns)): raise ValueError('unique columns required')
        if not np.isfinite(self.strength) or not self.derivation: raise ValueError('derivation/strength')

def save_patch(patch_id,arrays,derivation,provenance,rank=None):
    folder=ROOT/'patches'/patch_id
    if folder.exists(): raise FileExistsError('patches are immutable')
    manifest=json.loads((ROOT/'environment/architecture.json').read_text())
    tensors=[]
    for name,bits in arrays.items():
        dt,orig=tensor_bits(BASE_FILE,name)
        if dt!='BF16': raise ValueError('only BF16 patch format supported')
        delta=f32(bits)-f32(orig)
        tensors.append({'name':name,'shape':list(bits.shape),'dtype':'BF16','original_hash':bhash(orig),'replacement_hash':bhash(bits),'changed_entries':int(np.count_nonzero(bits!=orig)),'l2':float(np.linalg.norm(delta.astype(np.float64))),'linf':float(np.max(np.abs(delta))),'relative_frobenius':float(np.linalg.norm(delta)/np.linalg.norm(f32(orig))),'rank_before_cast':rank,'rank_after_cast':'not estimated; BF16 rounding can increase rank'})
    meta={'patch_id':patch_id,'base_revision':BASE_REVISION,'checkpoint_hashes':manifest['checkpoint_hashes'],'tensors':tensors,'derivation':derivation,'provenance':provenance,'deployment_dtype':'BF16','unique_text_parameters':manifest['text_unique_parameters'],'changed_entries':sum(t['changed_entries'] for t in tensors),'created_with_evaluator_hash':evaluator_hash()}
    validate_patch(meta,arrays)
    folder.mkdir(); np.savez(folder/'replacement.npz',**arrays); write_json(folder/'manifest.json',meta)
    return meta

def propose_patch(spec:PatchSpec,budget=None):
    spec.validate()
    if budget: budget.candidate(spec.patch_id)
    _,bits=tensor_bits(BASE_FILE,spec.tensor); w=f32(bits)
    if any(c<0 or c>=w.shape[1] for c in spec.columns): raise ValueError('column out of bounds')
    w[:,list(spec.columns)]*=(1-spec.strength)
    # This changes selected input columns only: W'[:,C]=(1-alpha) W[:,C], y=W x.
    replacement=bf16(w)
    if (ROOT/'patches'/spec.patch_id).exists():
        meta,arrays=load_patch(spec.patch_id);validate_patch(meta,arrays)
        if set(arrays)!={spec.tensor} or not np.array_equal(arrays[spec.tensor],replacement): raise ValueError('immutable patch ID collision')
        return meta
    return save_patch(spec.patch_id,{spec.tensor:replacement},spec.derivation,spec.provenance,rank=len(spec.columns))

def load_patch(patch_id):
    folder=ROOT/'patches'/patch_id
    meta=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'replacement.npz',allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    return meta,arrays

def validate_patch(meta,arrays):
    if meta['base_revision']!=BASE_REVISION: raise ValueError('wrong revision')
    architecture=json.loads((ROOT/'environment/architecture.json').read_text())
    if meta['checkpoint_hashes']!=architecture['checkpoint_hashes']: raise ValueError('wrong checkpoint hashes')
    if set(arrays)!={t['name'] for t in meta['tensors']}: raise ValueError('tensor list mismatch')
    if len(arrays)>8: raise ValueError('tensor budget')
    layers=set(); support=0
    for t in meta['tensors']:
        name=t['name']; match=re.fullmatch(r'model.language_model.layers.(\d+).(?:mlp.(?:down_proj|up_proj|gate_proj)|self_attn.(?:o_proj|q_proj|k_proj|v_proj)|linear_attn.(?:out_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b)).weight',name)
        if not match: raise ValueError('forbidden tensor')
        layers.add(int(match.group(1))); dt,orig=tensor_bits(BASE_FILE,name); a=arrays[name]
        if list(a.shape)!=t['shape'] or a.shape!=orig.shape or a.dtype!=np.uint16 or dt!='BF16': raise ValueError('shape/dtype mismatch')
        if not np.isfinite(f32(a)).all(): raise ValueError('nonfinite replacement')
        if bhash(orig)!=t['original_hash'] or bhash(a)!=t['replacement_hash']: raise ValueError('tensor hash mismatch')
        changed=int(np.count_nonzero(a!=orig)); support+=changed
        if changed!=t['changed_entries']: raise ValueError('false support declaration')
        # Reject all-entry scalar rescaling even if the matrix is small.
        if changed==a.size:
            x=f32(orig).astype(np.float64); y=f32(a).astype(np.float64); scale=(x*y).sum()/(x*x).sum()
            if np.linalg.norm(y-scale*x)<=.01*max(np.linalg.norm(y),1e-30): raise ValueError('whole tensor rescaling forbidden')
    if len(layers)>2 or support>architecture['max_changed_entries']: raise ValueError('edit support/layer budget')
    if support!=meta['changed_entries']: raise ValueError('support mismatch')
    return True

def evaluator_hash():
    return objhash({str(p.relative_to(ROOT)):digest(p) for folder in ('benchmark','eval') for p in sorted((ROOT/folder).glob('*.py'))})

def export_patch(patch_id):
    meta,arrays=load_patch(patch_id); validate_patch(meta,arrays)
    target=ROOT/'artifacts/exports'/patch_id
    if target.exists():
        old=json.loads((target/'toolpatch_export.json').read_text())
        if old['patch_manifest_hash']!=objhash(meta): raise ValueError('conflicting export')
        for n,h in old['files'].items():
            if digest(target/n)!=h: raise ValueError('export corrupted')
        return target
    target.mkdir(parents=True)
    for source in BASE.iterdir():
        if source.is_file(): shutil.copyfile(source,target/source.name)
    out=target/BASE_FILE.name; hh,offset=header(out)
    with out.open('r+b') as f:
        for name,a in arrays.items():
            start,end=hh[name]['data_offsets']; raw=np.ascontiguousarray(a).tobytes()
            if len(raw)!=end-start: raise ValueError('serialized tensor size')
            f.seek(offset+start); f.write(raw)
        f.flush(); os.fsync(f.fileno())
    for name,a in arrays.items():
        if bhash(tensor_bits(out,name)[1])!=bhash(a): raise ValueError('export mismatch')
    files={p.name:digest(p) for p in target.iterdir() if p.is_file()}
    write_json(target/'toolpatch_export.json',{'patch_manifest_hash':objhash(meta),'base_revision':BASE_REVISION,'files':files,'format':'original HF full multimodal checkpoint; only named BF16 tensors replaced'})
    return target

def rollback_export(patch_id):
    target=export_patch(patch_id); meta,arrays=load_patch(patch_id); out=target/BASE_FILE.name; hh,offset=header(out)
    # Roll back into a distinct artifact; never overwrite the edited export.
    dest=ROOT/'artifacts/exports'/(patch_id+'-reverted')
    if dest.exists(): raise FileExistsError(dest)
    shutil.copytree(target,dest); out=dest/BASE_FILE.name
    with out.open('r+b') as f:
        for name in arrays:
            _,bits=tensor_bits(BASE_FILE,name); f.seek(offset+hh[name]['data_offsets'][0]); f.write(bits.tobytes())
    if digest(out)!=digest(BASE_FILE): raise ValueError('rollback not bit exact')
    write_json(dest/'toolpatch_export.json',{'base_revision':BASE_REVISION,'rollback_of':patch_id,'rollback_exact':True,'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='toolpatch_export.json'},'format':'Original HF checkpoint restored exactly; no active patch.'})
    return dest

def published_patch():
    patch_id='published-p50c'
    if (ROOT/'patches'/patch_id).exists(): return load_patch(patch_id)[0]
    man=json.loads((ROOT/'sources/published_manifest.json').read_text()); z=np.load(ROOT/'sources/published_edit.npz',allow_pickle=False); arrays={}
    for i,t in enumerate(man['tensors']):
        _,orig=tensor_bits(BASE_FILE,t['name'])
        if bhash(orig)!=t['base_sha256']: raise ValueError('published base mismatch')
        if t['form']=='lowrank':
            w=f32(orig).astype(np.float64)
            for k in range(z[f't{i}_a'].shape[1]): w+=np.outer(z[f't{i}_a'][:,k],z[f't{i}_b'][k])
            bits=bf16(w.astype(np.float32))
        else: bits=orig.copy()
        bits.reshape(-1)[z[f't{i}_idx']]=z[f't{i}_val']
        if bhash(bits)!=t['edited_sha256']: raise ValueError('published edited hash mismatch')
        arrays[t['name']]=bits
    return save_patch(patch_id,arrays,'Published p50c repetition direction removal on selected MLP output columns; rebuilt from bit-exact diff. Not a ToolPatch discovery.','RikiyaT/lm_understand-logs exp8_math/runs/p50c/edit, HF repaired revision d1671a178705bcab7644806d23c535517f35723b',rank=1)

def freeze_patch(patch_id,run):
    evidence=json.loads((run/'gates.json').read_text())
    if not evidence.get('confirmation_allowed'): raise ValueError('confirmation gates not passed')
    if evidence.get('patch_id')!=patch_id:raise ValueError('confirmation gates belong to another patch')
    meta,arrays=load_patch(patch_id); validate_patch(meta,arrays)
    from runtime.campaign import source_hashes
    frozen={'source_hashes':source_hashes(),'patch_id':patch_id,'patch_manifest_hash':objhash(meta),'evaluator_hash':evaluator_hash(),'run':str(run),'baseline':json.loads((run/'baseline_frozen.json').read_text()),'gates':evidence,'final_seed':993821,'final_cases':1024,'regression_cases':384,'regression_seed':993822,'stock_final_cases':16}
    path=run/'frozen.json'
    if path.exists(): raise FileExistsError('freeze is immutable')
    write_json(path,frozen); return path
