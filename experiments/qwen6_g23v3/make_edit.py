"""Analytic rank-k column updates on down_proj (or other projection) weights, saved through primitives.save_edit.

W'[:, C] = W[:, C] + sum_k alpha_k * u_k v_k^T      (u_k in R^1024 output space, v_k in R^|C| over selected input columns)
Effect on the layer output at any position: delta = sum_k alpha_k * u_k * (v_k . a_C), a_C = MLP activations on columns C.
Random control: same support C, same Frobenius norm, random u (Gaussian) and v (Gaussian), computed deterministically from a seed.
"""
from __future__ import annotations
import json
import numpy as np
import primitives as P
from patches.store import bf16, f32

def tensor_name(layer, proj='mlp.down_proj'):
    return f'model.language_model.layers.{layer}.{proj}.weight'

def build_delta(W, columns, terms):
    """terms: list of (alpha, u(1024,), v(|C|,)) -> dense delta (same shape as W, zeros off-support)."""
    D=np.zeros_like(W,dtype=np.float64)
    for alpha,u,v in terms:
        D[:,columns]+=float(alpha)*np.outer(np.asarray(u,np.float64),np.asarray(v,np.float64))
    return D

def apply(layer, columns, terms, proj='mlp.down_proj', base='original'):
    """Save W + delta (cast to BF16). base='original' builds from the pristine tensor; base='current' stacks onto the current edit."""
    name=tensor_name(layer,proj)
    W=(P.load_original(name) if base=='original' else P.load(name)).astype(np.float64)
    D=build_delta(W,columns,terms)
    Wn=(W+D).astype(np.float32)
    rec=P.save_edit(name,Wn)
    realized=f32(bf16(Wn)).astype(np.float64)-f32(bf16(W.astype(np.float32))).astype(np.float64)
    intended=np.linalg.norm(D); got=np.linalg.norm(realized)
    rec.update({'intended_l2':float(intended),'realized_l2':float(got),'realized_over_intended':float(got/intended) if intended>0 else None,
                'support_entries':int(len(columns)*W.shape[0]),'cos_realized_intended':float((realized*D).sum()/(np.linalg.norm(realized)*intended+1e-30)) if intended>0 else None})
    return rec

def random_terms(columns, terms, seed):
    """Norm-matched random control: random Gaussian u,v per term scaled so each term's Frobenius norm matches the original term."""
    r=np.random.default_rng(seed); out=[]
    for alpha,u,v in terms:
        fro=abs(alpha)*np.linalg.norm(u)*np.linalg.norm(v)
        ru=r.standard_normal(len(u)); rv=r.standard_normal(len(columns))
        ru/=np.linalg.norm(ru); rv/=np.linalg.norm(rv)
        out.append((fro,ru,rv))
    return out

def describe(columns, terms):
    return {'n_columns':len(columns),'terms':[{'alpha':float(a),'u_norm':float(np.linalg.norm(u)),'v_norm':float(np.linalg.norm(v)),'frobenius':float(abs(a)*np.linalg.norm(u)*np.linalg.norm(v))} for a,u,v in terms]}
