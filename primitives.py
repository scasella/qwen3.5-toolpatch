"""Direct source-tensor BF16 edit API with on-disk persistence (edits/current.npz). No forward passes, no gradients.

NAMES            all source tensor names
load(name)       float32 tensor: current edit if present else original
load_original(name)
save_edit(name, tensor)  validate (scope, shape, dtype, no whole-tensor scaling, <=2 layers, <=8 tensors, <=0.25% entries) and store BF16
drop_edit(name)
edited()         tuple of edited tensor names
snapshot(label)  copy current edit set to edits/<label>.npz (immutable)
tok              tokenizer (lazy)
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from runtime.common import BASE, ROOT, Ledger
from runtime.inspect import header, tensor_bits
from patches.store import bf16, f32

SOURCE = next(BASE.glob('*.safetensors'))
HEADER, _ = header(SOURCE)
NAMES = tuple(sorted(n for n in HEADER if n != '__metadata__'))
EDITS = ROOT/'edits'; CURRENT = EDITS/'current.npz'
ALLOWED = re.compile(r'model\.language_model\.layers\.(\d+)\.(?:mlp\.(?:down_proj|up_proj|gate_proj)|self_attn\.(?:o_proj|q_proj|k_proj|v_proj)|linear_attn\.(?:out_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b))\.weight')
_tok = None

def __getattr__(name):
    global _tok
    if name != 'tok': raise AttributeError(name)
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    return _tok

def _read():
    if not CURRENT.exists(): return {}
    with np.load(CURRENT, allow_pickle=False) as z: return {n: z[n] for n in z.files}

def _write(edits):
    EDITS.mkdir(exist_ok=True)
    if edits: np.savez(CURRENT, **edits)
    elif CURRENT.exists(): CURRENT.unlink()

def load_original(name):
    dt, a = tensor_bits(SOURCE, name)
    return f32(a) if dt == 'BF16' else a.astype(np.float32)

def load(name):
    e = _read()
    return f32(e[name]) if name in e else load_original(name)

def edited(): return tuple(sorted(_read()))

def drop_edit(name):
    e = _read(); e.pop(name, None); _write(e)

def receipt(edits=None):
    edits = _read() if edits is None else edits
    rows = {}; total = 0; layers = set()
    for n, b in edits.items():
        _, orig = tensor_bits(SOURCE, n); d = f32(b).astype(np.float64) - f32(orig).astype(np.float64)
        ch = int(np.count_nonzero(b != orig)); total += ch; layers.add(int(ALLOWED.fullmatch(n).group(1)))
        rows[n] = {'changed_entries': ch, 'l2': float(np.linalg.norm(d)), 'linf': float(np.abs(d).max()) if d.size else 0.0,
                   'rel_frobenius': float(np.linalg.norm(d)/np.linalg.norm(f32(orig).astype(np.float64)))}
    arch = json.loads((ROOT/'environment/architecture.json').read_text())
    return {'tensors': rows, 'total_changed_entries': total, 'fraction_of_text_params': total/arch['text_unique_parameters'],
            'layers': sorted(layers), 'limit_entries': arch['max_changed_entries']}

def save_edit(name, tensor):
    if not ALLOWED.fullmatch(name): raise ValueError('Forbidden tensor: only decoder-layer projection weights are editable')
    dt, orig = tensor_bits(SOURCE, name)
    a = np.asarray(tensor, dtype=np.float32)
    if dt != 'BF16' or a.shape != orig.shape: raise ValueError('Source dtype/shape mismatch')
    bits = bf16(a)
    x = f32(orig).astype(np.float64); y = f32(bits).astype(np.float64)
    scale = float(np.vdot(x, y)/np.vdot(x, x))
    if not np.array_equal(bits, orig) and np.array_equal(bf16((f32(orig)*np.float32(scale)).astype(np.float32)), bits):
        raise ValueError('Whole-tensor scalar rescaling is forbidden')
    cand = {**_read(), name: bits}
    r = receipt(cand)
    if len(r['layers']) > 2 or len(cand) > 8 or r['total_changed_entries'] > r['limit_entries']:
        raise ValueError(f'Candidate support budget exceeded: {r["layers"]} layers, {len(cand)} tensors, {r["total_changed_entries"]} entries')
    _write(cand)
    out = {'name': name, 'dtype': 'BF16', **r['tensors'][name], 'total_changed_entries': r['total_changed_entries'], 'fraction': r['fraction_of_text_params']}
    Ledger(ROOT/'ledger.jsonl').add('save_edit', **out)
    return out

def snapshot(label):
    e = _read(); path = EDITS/f'{label}.npz'
    if path.exists(): raise FileExistsError(path)
    np.savez(path, **e); (EDITS/f'{label}.json').write_text(json.dumps(receipt(e), indent=2)+'\n')
    Ledger(ROOT/'ledger.jsonl').add('snapshot', label=label, **{k: v for k, v in receipt(e).items() if k != 'tensors'})
    return path

def restore(label):
    with np.load(EDITS/f'{label}.npz', allow_pickle=False) as z: e = {n: z[n] for n in z.files}
    _write(e); return edited()

def clear():
    _write({})
