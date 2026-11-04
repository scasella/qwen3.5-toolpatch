from __future__ import annotations
import gc, time, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from interventions.hooks import InterventionSpec
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from .common import BASE, ROOT, objhash
from .resources import Budget, guard, memory, GIB
from benchmark.contracts import ModelInput
from benchmark.render import prompt

@dataclass(frozen=True)
class Decoding:
    prompt: str='standard'
    thinking: bool=False
    max_tokens: int=128
    context_limit: int=2048
    temperature: float=0.0
    def __post_init__(self):
        if self.max_tokens<1 or self.max_tokens>128 or not 1 <= self.context_limit <=2048 or self.temperature!=0: raise ValueError('outside frozen primary/diagnostic decoding limits')

class Runtime:
    """One resident model. All forwards require a Budget.request context."""
    def __init__(self,budget:Budget,checkpoint:Path=BASE):
        from . import resources
        if not resources._LOCK_HELD: raise RuntimeError('shared accelerator lock is required before model load')
        import mlx.core as mx
        from mlx_lm import load
        if not mx.metal.is_available(): raise RuntimeError('Metal required; CPU fallback forbidden')
        mx.set_default_device(mx.gpu); mx.set_memory_limit(12*GIB); mx.set_cache_limit(512*1024**2)
        self.mx=mx; self.budget=budget; self.path=checkpoint; self.checkpoint='base'; self.originals={}
        guard(mx); self.model,self.tokenizer=load(str(checkpoint),lazy=False); self.model.eval(); mx.eval(self.model.parameters()); guard(mx)
        if checkpoint==BASE:
            from .validation import validate_loaded
            validate_loaded(self.model,mx)
        self.initial_memory=memory(mx)
    def close(self):
        self.model=None; self.tokenizer=None; self.originals.clear(); gc.collect(); self.mx.clear_cache()
    def set_checkpoint(self,checkpoint:str):
        if checkpoint==self.checkpoint: return
        from patches.store import load_patch, validate_patch
        if self.originals:
            self.model.load_weights(list(self.originals.items()),strict=False); self.originals={}
        if checkpoint!='base':
            meta,arrays=load_patch(checkpoint)
            validate_patch(meta,arrays)
            from mlx.utils import tree_flatten
            params=dict(tree_flatten(self.model.parameters()))
            changes=[]
            for name,bits in arrays.items():
                mapped=name.replace('model.language_model.','language_model.model.')
                self.originals[mapped]=params[mapped]
                v=self.mx.array(bits).view(self.mx.bfloat16); changes.append((mapped,v))
            self.model.load_weights(changes,strict=False)
        self.mx.eval(self.model.parameters()); self.mx.clear_cache(); self.checkpoint=checkpoint
        self.budget.ledger.add('checkpoint',checkpoint=checkpoint,cache_policy='fresh caches every execution; no prefix reuse')
    def _tokens(self,case:ModelInput,decoding:Decoding):
        p=prompt(self.tokenizer,case,decoding.prompt,decoding.thinking)
        ids=self.tokenizer.encode(p,add_special_tokens=False)
        if len(ids)+decoding.max_tokens>decoding.context_limit: raise ValueError(f'context length exceeds frozen limit: {case.id}: {len(ids)}')
        return p,ids
    def generate(self,cases:Sequence[ModelInput],checkpoint='base',decoding:Decoding=Decoding(),intervention_specs=()) -> list[dict]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from contextlib import nullcontext
        self.set_checkpoint(checkpoint); out=[]
        for case in cases:
            guard(self.mx); p,ids=self._tokens(case,decoding)
            rid=self.budget.reserve(len(ids)+decoding.max_tokens,decoding.max_tokens,case.id,checkpoint+':'+decoding.prompt)
            start=time.perf_counter(); text=''; tokenids=[]; last=None
            # Never reuse KVCache, ArraysCache, recurrent state, or detokenizer state across examples.
            cache=self.model.make_cache()
            if intervention_specs:
                from interventions.hooks import hooks
                hook_context=hooks(self.model,intervention_specs)
            else: hook_context=nullcontext()
            with hook_context:
                for response in stream_generate(self.model,self.tokenizer,ids,max_tokens=decoding.max_tokens,sampler=make_sampler(0),prompt_cache=cache,prefill_step_size=256):
                    text+=response.text; tokenids.append(response.token); last=response
                    if len(tokenids)%16==0: guard(self.mx)
            elapsed=time.perf_counter()-start
            m=guard(self.mx); generated=last.generation_tokens if last else 0
            row={'case_id':case.id,'checkpoint':checkpoint,'prompt_variant':decoding.prompt,'thinking':decoding.thinking,'raw_output':text,'token_ids':tokenids,'input_tokens':len(ids),'processed_input_tokens':len(ids)+max(generated-1,0),'output_tokens':generated,'seconds':elapsed,'truncated':last.finish_reason=='length' if last else False,'prompt_hash':objhash(p),'memory':m}
            self.budget.finish(rid,row['processed_input_tokens'],generated,seconds=elapsed,memory=m)
            out.append(row); del cache,last; self.mx.clear_cache()
        return out
    def teacher_force(self,cases:Sequence[ModelInput],continuations:Sequence[str],selected_positions:Sequence[Sequence[int]],decoding=Decoding(),intervention_specs=(),alternative_token_ids:Sequence[int]=()) -> list[dict]:
        if not(len(cases)==len(continuations)==len(selected_positions)): raise ValueError('length mismatch')
        from interventions.hooks import hooks
        if any(len(p)>16 for p in selected_positions): raise ValueError('at most 16 teacher-forced positions per case')
        if len(alternative_token_ids)>8:raise ValueError('at most eight diagnostic alternatives')
        output=[]
        for case,cont,positions in zip(cases,continuations,selected_positions):
            p,_=self._tokens(case,decoding); ids=self.tokenizer.encode(p+cont,add_special_tokens=False)
            if len(ids)>decoding.context_limit or any(i<=0 or i>=len(ids) for i in positions): raise ValueError('invalid teacher-forced position')
            rid=self.budget.reserve(len(ids),0,case.id,'teacher_force:'+self.checkpoint)
            # Chunks limit transient vocabulary logits; only requested scalar log probabilities survive.
            cache=self.model.make_cache(); result=[]
            with hooks(self.model,intervention_specs):
                for offset in range(0,len(ids),64):
                    logits=self.model(self.mx.array(ids[offset:offset+64])[None,:],cache=cache)
                    for i in positions:
                        if offset<=i-1<offset+logits.shape[1]:
                            v=logits[0,i-1-offset,:].astype(self.mx.float32); lp=v[ids[i]]-self.mx.logsumexp(v); self.mx.eval(lp)
                            row={'position':i,'token':ids[i],'log_probability':lp.item()}
                            if alternative_token_ids:
                                alts=v[self.mx.array(alternative_token_ids)]-self.mx.logsumexp(v);self.mx.eval(alts)
                                row['alternatives']={str(t):float(p) for t,p in zip(alternative_token_ids,alts.tolist())}
                            result.append(row)
                    self.mx.eval([c.state for c in cache]); del logits; guard(self.mx)
            output.append({'case_id':case.id,'measurements':result})
            self.budget.finish(rid,len(ids),0,memory=memory(self.mx)); del cache; self.mx.clear_cache()
        return output
    def capture(self,cases:Sequence[ModelInput],layers:Sequence[int],positions:Sequence[Sequence[int]],components:Sequence[str],decoding=Decoding()) -> list[dict]:
        from interventions.hooks import hooks
        import numpy as np
        if len(cases)!=len(positions): raise ValueError('positions per example required')
        if len(layers)>2 or len(components)>2: raise ValueError('bounded captures only')
        outputs=[]
        for case,pp in zip(cases,positions):
            _,ids=self._tokens(case,decoding)
            if len(pp)>16 or any(p<0 or p>=len(ids) for p in pp): raise ValueError('selected positions invalid')
            rid=self.budget.reserve(len(ids),0,case.id,'capture:'+self.checkpoint)
            with hooks(self.model,capture_specs=[(l,c,pp) for l in layers for c in components]) as sinks:
                cache=self.model.make_cache()
                for offset in range(0,len(ids),256):
                    v=self.model.language_model.model(self.mx.array(ids[offset:offset+256])[None,:],cache=cache); self.mx.eval(v); guard(self.mx)
                outputs.append({'case_id':case.id,'captures':{f'{l}:{c}':[{'position':s['position'],'input':s['input'].tolist()} for s in sink] for (l,c),sink in sinks.items()}})
            self.budget.finish(rid,len(ids),0,memory=memory(self.mx)); del cache,v; self.mx.clear_cache()
        return outputs
    def intervene(self,cases:Sequence[ModelInput],intervention_spec:InterventionSpec,decoding:Decoding=Decoding()):
        specs=intervention_spec if isinstance(intervention_spec,(list,tuple)) else [intervention_spec]
        if len(specs)>2: raise ValueError('at most two temporary component interventions')
        return self.generate(cases,self.checkpoint,decoding,specs)
