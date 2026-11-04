"""Temporary hooks wrap supported projection boundaries only, never DeltaNet recurrent state."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import mlx.core as mx
import mlx.nn as nn
@dataclass(frozen=True)
class InterventionSpec:
    layer: int
    component: str # mlp.down_proj, self_attn.o_proj, linear_attn.out_proj
    columns: tuple[int,...] = ()
    strength: float = 0.0
    position: int | None = None # None = all positions (weight-equivalent column ablation)
    direction: tuple[float,...] = ()
    mode: str = 'ablate_input'
    output_direction: tuple[float,...] = ()
    def validate(self,model):
        if self.layer not in range(len(model.layers)): raise ValueError('layer')
        if self.component not in ('mlp.down_proj','self_attn.o_proj','linear_attn.out_proj'): raise ValueError('unsupported component')
        if self.component.startswith('self_attn') and model.layers[self.layer].is_linear: raise ValueError('DeltaNet has no full attention heads')
        if self.component.startswith('linear_attn') and not model.layers[self.layer].is_linear: raise ValueError('not DeltaNet')
        obj=model.layers[self.layer]
        for attr in self.component.split('.'): obj=getattr(obj,attr)
        if any(c<0 or c>=obj.weight.shape[1] for c in self.columns): raise ValueError('column out of range')
        if self.mode not in ('ablate_input','random_sign_input','add_output','project_columns','amplify_input_direction','rank_one_columns'): raise ValueError('mode')
        if self.mode in ('random_sign_input','amplify_input_direction','rank_one_columns') and len(self.direction)!=len(self.columns): raise ValueError('input direction shape')
        if self.mode in ('add_output','project_columns') and len(self.direction)!=obj.weight.shape[0]: raise ValueError('direction shape')
        if self.mode=='rank_one_columns' and len(self.output_direction)!=obj.weight.shape[0]:raise ValueError('output direction shape')
        if len(set(self.columns))!=len(self.columns):raise ValueError('duplicate columns')
        if any(not __import__('math').isfinite(v) for v in self.direction+self.output_direction):raise ValueError('nonfinite direction')
        if not __import__('math').isfinite(self.strength): raise ValueError('strength')

class Probe(nn.Module):
    def __init__(self,inner,positions=(),sink=None,spec=None):
        super().__init__(); self.inner=inner; self._positions=positions; self._sink=sink; self._spec=spec; self._offset=0
    def __call__(self,x):
        s=self._spec
        if self._sink is not None:
            for p in self._positions:
                if self._offset<=p<self._offset+x.shape[1]:
                    v=x[0,p-self._offset,:]; mx.eval(v)
                    import numpy as np
                    self._sink.append({'position':p,'input':np.array(v.astype(mx.float32))})
        if s and s.mode in ('ablate_input','random_sign_input') and s.strength:
            mask=mx.ones(x.shape[-1],dtype=x.dtype)
            change=-s.strength if s.mode=='ablate_input' else -s.strength*mx.array(s.direction,dtype=x.dtype)
            mask=mask.at[mx.array(s.columns)].add(change)
            if s.position is None: x=x*mask
            elif self._offset<=s.position<self._offset+x.shape[1]:
                x=x.at[:,s.position-self._offset,:].multiply(mask)
        if s and s.mode=='amplify_input_direction' and s.strength:
            cols=mx.array(s.columns);v=mx.array(s.direction,dtype=mx.float32)
            delta=s.strength*(x[...,cols].astype(mx.float32)@v)[...,None]*v
            if s.position is None:x=x.at[...,cols].add(delta.astype(x.dtype))
            elif self._offset<=s.position<self._offset+x.shape[1]:
                p=s.position-self._offset;x=x.at[:,p,cols].add(delta[:,p,:].astype(x.dtype))
        out=self.inner(x)
        if s and s.mode=='rank_one_columns' and s.strength:
            cols=mx.array(s.columns);v=mx.array(s.direction,dtype=mx.float32);u=mx.array(s.output_direction,dtype=mx.float32)
            delta=s.strength*(x[...,cols].astype(mx.float32)@v)[...,None]*u
            if s.position is None:out=(out.astype(mx.float32)+delta).astype(out.dtype)
            elif self._offset<=s.position<self._offset+x.shape[1]:
                p=s.position-self._offset;out=out.at[:,p,:].add(delta[:,p,:].astype(out.dtype))
        if s and s.mode=='project_columns' and s.strength:
            cols=mx.array(s.columns)
            u=mx.array(s.direction,dtype=mx.float32)
            partial=x[...,cols].astype(mx.float32) @ self.inner.weight[:,cols].astype(mx.float32).T
            correction=s.strength*(partial @ u)[...,None]*u
            if s.position is None: out=(out.astype(mx.float32)-correction).astype(out.dtype)
            elif self._offset<=s.position<self._offset+x.shape[1]:
                p=s.position-self._offset
                out=out.at[:,p,:].add(-correction[:,p,:].astype(out.dtype))
        if s and s.mode=='add_output' and s.strength:
            delta=mx.array(s.direction,dtype=out.dtype)*s.strength
            if s.position is None: out=out+delta
            elif self._offset<=s.position<self._offset+x.shape[1]: out=out.at[:,s.position-self._offset,:].add(delta)
        self._offset+=x.shape[1]
        return out

@contextmanager
def hooks(model,specs=(),capture_specs=()):
    installed=[]; sinks={}
    keys=set()
    try:
        for spec in specs:
            spec.validate(model); key=(spec.layer,spec.component)
            if key in keys: raise ValueError('duplicate hook')
            keys.add(key); parent=model.layers[spec.layer]; attrs=spec.component.split('.')
            for a in attrs[:-1]: parent=getattr(parent,a)
            old=getattr(parent,attrs[-1]); setattr(parent,attrs[-1],Probe(old,spec=spec)); installed.append((parent,attrs[-1],old))
        for layer,component,positions in capture_specs:
            InterventionSpec(layer,component).validate(model)
            if (layer,component) in keys: raise ValueError('capture and intervention same boundary unsupported')
            parent=model.layers[layer]; attrs=component.split('.')
            for a in attrs[:-1]: parent=getattr(parent,a)
            old=getattr(parent,attrs[-1]); sink=[]; sinks[(layer,component)]=sink
            setattr(parent,attrs[-1],Probe(old,positions,sink)); installed.append((parent,attrs[-1],old))
        yield sinks
    finally:
        for parent,name,old in reversed(installed): setattr(parent,name,old)
