"""Shared CPU-only helpers for the layer-22/23 circuit: exact replay of layer 23 from captured layer-23 inputs with the
stage-1 write added, using the qwen-7 campaign's validated replay module (imported read-only, no bytecode written).

Set QWEN7 to the qwen-7 workspace (default /Users/scasella/Downloads/qwen-7). Everything here is derived analysis: it reads
campaign captures and never writes into either campaign folder."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1'); sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
QWEN7 = Path(os.environ.get('QWEN7', '/Users/scasella/Downloads/qwen-7')); QWEN6 = Path(os.environ.get('QWEN6', '/Users/scasella/Downloads/qwen-6'))
sys.path.insert(0, str(QWEN7)); sys.path.insert(0, str(ROOT))
from apply_patch import read_bits, f32, bf16
BASE = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
BASE_FILE = next(BASE.glob('*.safetensors'))
L = 'model.language_model.layers.{}.mlp.{}.weight'
J0, J1, J2, J3 = 2525, 3468, 3421, 2331
TOK = {'call': 248058, 'NO': 8725, 'NE': 3811}

def silu(z): z = np.asarray(z, np.float64); return z / (1 + np.exp(-z))
def unit(v): return v / np.linalg.norm(v)

_orig = {}
def original(layer, proj):
    key = (layer, proj)
    if key not in _orig: _orig[key] = f32(read_bits(BASE_FILE, L.format(layer, proj)))
    return _orig[key]

def patch_rows(label=None, entries=None):
    """-> dict of final float32 rows for the four neurons from patches/<label>.vectors.npz (or an explicit entries dict)."""
    if entries is None:
        V = np.load(ROOT / 'patches' / f'{label}.vectors.npz'); entries = {k: V[k] for k in V.files}
    def get(kind, layer, proj, j):
        return np.asarray(entries.get(f'{kind}|{L.format(layer, proj)}|{j}', (original(layer, proj)[j] if kind == 'row' else original(layer, proj)[:, j])), np.float32)
    return {'gate22': get('row', 22, 'gate_proj', J0), 'up22': get('row', 22, 'up_proj', J0), 'down22': get('col', 22, 'down_proj', J0),
            'gate23': get('row', 23, 'gate_proj', J3), 'up23': get('row', 23, 'up_proj', J3), 'down23': get('col', 23, 'down_proj', J3),
            'gateJ1': get('row', 23, 'gate_proj', J1), 'upJ1': get('row', 23, 'up_proj', J1), 'downJ1': get('col', 23, 'down_proj', J1),
            'gateJ2': get('row', 23, 'gate_proj', J2), 'upJ2': get('row', 23, 'up_proj', J2), 'downJ2': get('col', 23, 'down_proj', J2)}

def layer23_tensors(R):
    """Full float32 layer-23 MLP tensors with the J1/J2/J3 rows of R placed (keys as Replay.set_weights expects)."""
    Wg = original(23, 'gate_proj').copy(); Wu = original(23, 'up_proj').copy(); Wd = original(23, 'down_proj').copy()
    Wg[J1] = R['gateJ1']; Wu[J1] = R['upJ1']; Wd[:, J1] = R['downJ1']; Wg[J2] = R['gateJ2']; Wu[J2] = R['upJ2']; Wd[:, J2] = R['downJ2']
    Wg[J3] = R['gate23']; Wu[J3] = R['up23']; Wd[:, J3] = R['down23']
    return {'mlp.gate_proj.weight': Wg, 'mlp.up_proj.weight': Wu, 'mlp.down_proj.weight': Wd}

def stage1(R, X22, dose=1.0):
    """Stage-1 activation at every position of X22 and the modified layer-23 input increment (original J0 contribution removed)."""
    z = X22 @ R['gate22']; act = silu(z) * (X22 @ R['up22'])
    a_orig = silu(X22 @ original(22, 'gate_proj')[J0]) * (X22 @ original(22, 'up_proj')[J0])
    delta = dose * act[:, None] * R['down22'][None, :] - a_orig[:, None] * original(22, 'down_proj')[:, J0][None, :]
    return act, z, delta.astype(np.float32)

def j_acts(R, X23):
    out = {}
    for name, g, u in (('J1', 'gateJ1', 'upJ1'), ('J2', 'gateJ2', 'upJ2'), ('J3', 'gate23', 'up23')):
        z = X23 @ R[g]; out[name] = silu(z) * (X23 @ R[u])
    return out

class Replayer:
    """Thin wrapper around qwen-7's analysis.replay23.Replay (CPU MLX, stock modules)."""
    def __init__(self):
        from analysis.replay23 import Replay
        self.rp = Replay(); self.cur = None
    def weights(self, W23):
        self.rp.set_weights(W23); self.cur = W23
    def restore(self): self.rp.restore(); self.cur = None
    def run(self, h23, dec, want_qk=False):
        r = self.rp.run(h23.astype(np.float32), query_positions=(dec,), want_x23=True, want_qk=want_qk)
        return r
    def logits(self, h24_row): return self.rp.logits(h24_row)

def load_capture(label):
    from analysis.cap import Capture
    return Capture(label)
