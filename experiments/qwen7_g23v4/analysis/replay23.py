"""Exact CPU replay of layer 23 (+ final norm + tied unembedding) from captured h23 sequences, using the stock mlx_lm modules on the CPU device.
Nothing here touches Metal or the budget. Supports: modified h23 rows (stage-1 writes), modified layer-23 tensors (edits), per-head attention rows at any query."""
from __future__ import annotations
import os, json
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import mlx.core as mx
from pathlib import Path
from runtime.common import BASE, ROOT
import primitives as P
from patches.store import f32, bf16

mx.set_default_device(mx.cpu); mx.disable_compile()   # the CPU backend would otherwise JIT-compile fused kernels with g++ (no SDK headers here)
_model=None
def model():
    global _model
    if _model is None:
        from mlx_lm import load
        _model,_=load(str(BASE),lazy=True)
        bb=_model.language_model.model
        mx.eval(bb.layers[23].parameters(),bb.norm.parameters(),bb.embed_tokens.parameters())
    return _model

class Replay:
    def __init__(self):
        m=model(); self.bb=m.language_model.model; self.layer=self.bb.layers[23]; self.norm=self.bb.norm; self.E=self.bb.embed_tokens
        self.orig={}; self._snapshot()
    def _snapshot(self):
        from mlx.utils import tree_flatten
        self.orig={k:v for k,v in tree_flatten(self.layer.parameters())}
    def set_weights(self,updates:dict):
        """updates: {param path within layer 23 (e.g. 'mlp.gate_proj.weight','self_attn.k_proj.weight'): np.float32 array (full tensor)} -> loaded as bf16 (round-to-nearest-even)."""
        pairs=[]
        for k,v in updates.items(): pairs.append((k,mx.array(bf16(np.asarray(v,np.float32))).view(mx.bfloat16)))
        self.layer.load_weights(pairs,strict=False); mx.eval(self.layer.parameters())
    def restore(self):
        self.layer.load_weights(list(self.orig.items()),strict=False); mx.eval(self.layer.parameters())
    def run(self,h23:np.ndarray,query_positions=(),want_x23=False,want_qk=False):
        """h23: (T,1024) float32 layer-23 input. Returns dict with h24 (T,1024) float32, logits at the last position (and others requested), attention rows per head
        for the query positions (via a sdpa patch), x23 (MLP input) if wanted."""
        import mlx_lm.models.qwen3_next as qn
        rows={}; orig_sdpa=qn.scaled_dot_product_attention
        def spy(q,k,v,cache,scale,mask,sinks=None):
            if want_qk: rows['q']=np.array(q[0].astype(mx.float32)); rows['k']=np.array(k[0].astype(mx.float32)); rows['v']=np.array(v[0].astype(mx.float32))
            for p in query_positions:
                qq=q[0,:,p,:].astype(mx.float32); kk=mx.repeat(k[0].astype(mx.float32)[:, :p+1,:],q.shape[1]//k.shape[1],axis=0)
                sc=(qq[:,None,:]*kk).sum(-1)*scale; rows[p]=np.array(mx.softmax(sc,axis=-1))
            return orig_sdpa(q,k,v,cache,scale,mask,sinks)
        qn.scaled_dot_product_attention=spy
        try:
            x=mx.array(h23.astype(np.float32)).astype(mx.bfloat16)[None]
            T=x.shape[1]
            from mlx_lm.models.base import create_attention_mask
            mask=create_attention_mask(x,None)
            # replicate DecoderLayer.__call__ to expose the MLP input
            L=self.layer
            r=L.self_attn(L.input_layernorm(x),mask,None); h=x+r; xm=L.post_attention_layernorm(h); out=h+L.mlp(xm)
            hn=self.norm(out); mx.eval(out,hn)
        finally: qn.scaled_dot_product_attention=orig_sdpa
        res={'h24':np.array(out[0].astype(mx.float32)),'rows':rows}
        if want_x23:
            res['x23']=np.array(xm[0].astype(mx.float32)); res['attn_out']=np.array(r[0].astype(mx.float32))
            qp=L.self_attn.q_proj(L.input_layernorm(x)); g=mx.split(qp.reshape(1,T,8,-1),2,axis=-1)[1]; res['gate']=np.array(mx.sigmoid(g[0]).astype(mx.float32))   # (T,8,256)
            res['xnorm']=np.array(L.input_layernorm(x)[0].astype(mx.float32))
        return res
    def logits(self,h24_row:np.ndarray):
        hn=self.norm(mx.array(h24_row.astype(np.float32)).astype(mx.bfloat16)[None,None]); lg=self.E.as_linear(hn)[0,0].astype(mx.float32); mx.eval(lg); return np.array(lg)

if __name__=='__main__':
    # validation: replay every D1/D2/D3 item from its captured h23, compare the decision logits and h24 with the capture
    import sys
    from analysis.cap import Capture
    rp=Replay(); errs=[]; herr=[]; am=0; n=0
    for lab in sys.argv[1:] or ['d01b_triples_D1']:
        c=Capture(lab)
        for i in c.ids[:24]:
            h23=c.full(i,'h23'); r=rp.run(h23); lg=rp.logits(r['h24'][-1]); ref=c.logits(i)
            errs.append(float(np.abs(lg-ref).max())); n+=1; am+=int(lg.argmax()==ref.argmax())
            h24ref=c.h(i,'dec',24); herr.append(float(np.abs(r['h24'][-1]-h24ref).max()/np.abs(h24ref).max()))
        print(lab,'items',n,'argmax agree',am,'max |logit err|',max(errs),'median',float(np.median(errs)),'rel h24 err max',max(herr))
