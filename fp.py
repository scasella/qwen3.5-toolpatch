#!/usr/bin/env python
"""Forward-pass tool. One budgeted pass per invocation (<=64 model-example executions; 50 passes total).

  generate: fp.py generate --set practice [--ids a,b] --model original|edited|both --label L --hypothesis H --prediction P [--edit current|<snapshot>] [--prompt standard]
  teacher:  fp.py teacher  --set teacher  [--ids ...] --model original|edited --label L --hypothesis H --prediction P [--continuation gold|none|text:<literal>]
            [--capture decision,prefix] [--columns 20:1,2,3;18:7,8] (all-position capture of listed down_proj input columns) [--layers 0-23]

Evaluation protocol is fixed: official chat template with native <tools>, enable_thinking=False, greedy, max 128 new tokens, context <=2048.
"""
from __future__ import annotations
import argparse, json, os, sys, time, gc
from collections import Counter
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
from runtime.common import ROOT, BASE, Ledger, write_json, objhash
from runtime.resources import Budget, accelerator_lock, guard, memory, memory_policy, GIB
from benchmark.render import prompt as render_prompt, gold_text
from benchmark.contracts import ModelInput, Case
from benchmark.regression import check as regression_check
from eval.scorer import grade, parse
from build_data import load_cases

FULL_ATTN=(3,7,11,15,19,23)
WATCH=['<tool_call>','NO','NE','I','To','The','No','None','Based','This','Since','Select','<','<think>','Please','Sure','Here','Okay','You','It','1','\n','Yes','NOT','No_','NEED','need','no','<|im_end|>']
MAX_TOKENS=128; CONTEXT=2048

# ----------------------------------------------------------------------------- data
def items_for(set_name, ids=None):
    if set_name in ('regression','regression2','prose_qa','prose_qa2'):
        rows=[json.loads(l) for l in (ROOT/f'data/{set_name}.jsonl').read_text().splitlines() if l]
        items=[{'id':r['id'],'input':ModelInput(r['id'],r['user'],[]),'case':None,'domain':r['domain'],'gold':r['gold'],'action':'regression','family':r['domain']} for r in rows]
    else:
        items=[{'id':c.id,'input':c.model_input(),'case':c,'action':c.spec.action,'family':c.family} for c in load_cases(set_name)]
    if ids:
        want=set(ids); items=[i for i in items if i['id'] in want]
        missing=want-{i['id'] for i in items}
        if missing: raise ValueError(f'unknown ids: {sorted(missing)}')
    return items

def score(item, text):
    if item['case'] is None:
        if item['domain']=='prose':
            ok=bool(text.strip()) and not any(f in text for f in ('NO_TOOL','NEED_CLARIFICATION','<tool_call>','NENE','NONO'))
        else: ok=regression_check({'domain':item['domain'],'gold':item['gold']},text)
        return {'correct':ok,'correct_strict':ok,'parse_valid':True,'false_call':'<tool_call>' in text,'error':'correct' if ok else 'wrong','decision':None}
    case=item['case']; s=grade(case,text).to_dict()
    strict=s['correct']
    decision=None
    if s['parse_valid']:
        p=parse(text,case.tools); decision=p.action
        if p.action=='call' and set(p.arguments)!=set(case.expected_arguments): strict=False
    else:
        decision='call' if '<tool_call>' in text else 'prose'
    s['correct_strict']=bool(strict); s['decision']=decision
    return s

def summarize(rows):
    n=len(rows)
    out={'n':n,'correct':sum(r['score']['correct'] for r in rows),'correct_strict':sum(r['score']['correct_strict'] for r in rows),
         'syntax_valid':sum(r['score']['parse_valid'] for r in rows),'false_calls':sum(bool(r['score']['false_call']) for r in rows),
         'mean_output_tokens':sum(r['output_tokens'] for r in rows)/max(n,1),'truncated':sum(r['truncated'] for r in rows)}
    out['per_class']={}
    for a in sorted({r['action'] for r in rows}):
        rr=[r for r in rows if r['action']==a]
        out['per_class'][a]={'n':len(rr),'correct':sum(r['score']['correct'] for r in rr),'correct_strict':sum(r['score']['correct_strict'] for r in rr),
                             'decisions':dict(Counter(str(r['score']['decision']) for r in rr))}
    out['errors']=dict(Counter(r['score']['error'] for r in rows if not r['score']['correct']))
    return out

def paired(a,b):
    assert [r['case_id'] for r in a]==[r['case_id'] for r in b]
    def trans(key):
        w2r=[x['case_id'] for x,y in zip(a,b) if not x['score'][key] and y['score'][key]]
        r2w=[x['case_id'] for x,y in zip(a,b) if x['score'][key] and not y['score'][key]]
        return {'wrong_to_right':len(w2r),'right_to_wrong':len(r2w),'wrong_to_right_ids':w2r,'right_to_wrong_ids':r2w}
    res={'lenient':trans('correct'),'strict':trans('correct_strict'),
         'token_identical':sum(x['token_ids']==y['token_ids'] for x,y in zip(a,b)),
         'call_right_to_wrong':[x['case_id'] for x,y in zip(a,b) if x['action']=='call' and x['score']['correct'] and not y['score']['correct']],
         'decision_changes':[{'id':x['case_id'],'action':x['action'],'from':x['score']['decision'],'to':y['score']['decision']} for x,y in zip(a,b) if x['score']['decision']!=y['score']['decision']]}
    res['per_class']={}
    for cls in sorted({x['action'] for x in a}):
        aa=[x for x in a if x['action']==cls]; bb=[y for y in b if y['action']==cls]
        res['per_class'][cls]={'wrong_to_right':sum(not x['score']['correct'] and y['score']['correct'] for x,y in zip(aa,bb)),'right_to_wrong':sum(x['score']['correct'] and not y['score']['correct'] for x,y in zip(aa,bb))}
    return res

# ----------------------------------------------------------------------------- model
class Engine:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        if not mx.metal.is_available(): raise RuntimeError('Metal required')
        mx.set_default_device(mx.gpu); mx.set_memory_limit(12*GIB); mx.set_cache_limit(512*1024**2)
        self.mx=mx
        def forbidden(*a,**k): raise RuntimeError('gradients disabled')
        mx.grad=mx.value_and_grad=forbidden
        guard(mx); self.model,self.tok=load(str(BASE),lazy=False); self.model.eval(); mx.eval(self.model.parameters()); guard(mx)
        ver=ROOT/'environment/conversion_verified.json'
        if not ver.exists():
            from runtime.validation import validate_loaded
            validate_loaded(self.model,mx)
        self.originals={}; self.checkpoint='original'
        self.watch_ids={w:self.tok.encode(w,add_special_tokens=False) for w in WATCH}
        self.watch_ids={w:ids[0] for w,ids in self.watch_ids.items() if len(ids)==1}
        self.initial_memory=memory(mx)
    @property
    def backbone(self): return self.model.language_model.model
    def embed_logits(self,states):
        return self.backbone.embed_tokens.as_linear(states)
    def apply_edit(self,arrays):
        from mlx.utils import tree_flatten
        mx=self.mx; params=dict(tree_flatten(self.model.parameters())); changes=[]
        for name,bits in arrays.items():
            mapped=name.replace('model.language_model.','language_model.model.')
            if mapped not in self.originals: self.originals[mapped]=params[mapped]
            changes.append((mapped,mx.array(bits).view(mx.bfloat16)))
        self.model.load_weights(changes,strict=False); mx.eval(self.model.parameters())
        params=dict(tree_flatten(self.model.parameters()))
        for name,bits in arrays.items():
            mapped=name.replace('model.language_model.','language_model.model.')
            got=np.array(params[mapped].view(mx.uint16))
            if not np.array_equal(got,bits): raise RuntimeError('edit not realized in loaded weights: '+name)
        self.checkpoint='edited'; mx.clear_cache()
    def revert(self):
        if self.originals:
            self.model.load_weights(list(self.originals.items()),strict=False); self.mx.eval(self.model.parameters()); self.originals={}
        self.checkpoint='original'; self.mx.clear_cache()
    def prompt_ids(self,inp:ModelInput,variant='standard'):
        p=render_prompt(self.tok,inp,variant,False); ids=self.tok.encode(p,add_special_tokens=False)
        if len(ids)+MAX_TOKENS>CONTEXT: raise ValueError(f'context exceeds limit: {inp.id} {len(ids)}')
        return p,ids
    def generate(self,ids):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        cache=self.model.make_cache(); text='';toks=[];last=None
        for r in stream_generate(self.model,self.tok,ids,max_tokens=MAX_TOKENS,sampler=make_sampler(0),prompt_cache=cache,prefill_step_size=256):
            text+=r.text; toks.append(r.token); last=r
            if len(toks)%32==0: guard(self.mx)
        del cache; self.mx.clear_cache()
        return text,toks,(last.generation_tokens if last else 0),(last.finish_reason=='length' if last else False)

class Observer:
    """Factory: returns an nn.Module wrapper recording the wrapped module's input at selected positions; optional all-position capture
    of listed columns ('columns'), all-position full capture as float16 ('full'), or per-column running max-abs/sum ('stats')."""
    def __new__(cls,inner,positions,sink,key,columns=None,allsink=None,full=False,stats=False):
        import mlx.core as mx, mlx.nn as nn
        from types import SimpleNamespace
        class _Obs(nn.Module):
            def __init__(self):
                super().__init__(); self.inner=inner
                self._st=SimpleNamespace(positions=list(positions),sink=sink,key=key,columns=(list(columns) if columns is not None else None),allsink=allsink,offset=0,full=full,stats=stats)
            def __call__(self,x,*a,**k):
                st=self._st; L=x.shape[1]
                for p in st.positions:
                    q=p-st.offset
                    if 0<=q<L:
                        v=x[0,q,:].astype(mx.float32); mx.eval(v); st.sink[(st.key,p)]=np.array(v)
                if st.columns is not None:
                    v=x[0,:,mx.array(st.columns)].astype(mx.float32); mx.eval(v); st.allsink.setdefault(st.key,[]).append(np.array(v))
                if st.full:
                    v=x[0].astype(mx.float16); mx.eval(v); st.allsink.setdefault(('full',)+tuple(st.key),[]).append(np.array(v))
                if st.stats:
                    xf=x[0].astype(mx.float32); mx_=mx.abs(xf).max(axis=0); sm=xf.sum(axis=0); mx.eval(mx_,sm)
                    st.allsink.setdefault(('stats',)+tuple(st.key),[]).append((np.array(mx_),np.array(sm),L))
                st.offset+=L
                return self.inner(x,*a,**k)
        return _Obs()

def install(engine,positions,layers,columns,mlpin=(),colstats=()):
    """columns: dict layer->list[int] for all-position capture of down_proj inputs; mlpin: layers whose MLP input (post-norm residual)
    is captured at all positions (float16); colstats: layers whose down_proj input gets per-column max-abs/sum over all positions."""
    model=engine.model; bb=engine.backbone; installed=[]; sink={}; allsink={}
    def wrap(parent,attr,key,cols=None,full=False,stats=False):
        orig=getattr(parent,attr); setattr(parent,attr,Observer(orig,positions,sink,key,cols,allsink,full,stats)); installed.append((parent,attr,orig))
    for l in range(24):
        wrap(bb.layers[l],'input_layernorm',('h',l))
        if l in layers or l in columns or l in colstats: wrap(bb.layers[l].mlp,'down_proj',('mlp',l),columns.get(l),False,l in colstats)
        if l in mlpin: wrap(bb.layers[l],'mlp',('mlpin',l),None,True,False)
        if l in FULL_ATTN: wrap(bb.layers[l].self_attn,'o_proj',('attn',l))
    wrap(bb,'norm',('h',24))
    return installed,sink,allsink

def uninstall(installed):
    for parent,attr,orig in reversed(installed): setattr(parent,attr,orig)

def teacher_case(engine,ids,prompt_len,capture_prefix,layers,columns,topk=24,mlpin=(),colstats=()):
    """One sequence. Logits at positions prompt_len-1 .. len(ids)-2 (predicting every continuation token) plus decision; captures at decision (+ '=' prefix position)."""
    mx=engine.mx; n=len(ids); dec=prompt_len-1
    logit_pos=list(range(dec,n-1)) if n>prompt_len else [dec]
    cap=[dec]; prefix_pos=None
    if capture_prefix:
        # continuation starts '<tool_call>\n<function=' -> '=' is the 5th continuation token
        if ids[prompt_len:prompt_len+4]==[248058,198,27,1628]:
            prefix_pos=prompt_len+4 if ids[prompt_len+4]==28 else prompt_len+3  # '=' position, or 'function' position when '=' merged with the name
            cap.append(prefix_pos)
    installed,sink,allsink=install(engine,cap,layers,columns,mlpin,colstats)
    logits={}
    try:
        cache=engine.model.make_cache()
        for off in range(0,n,256):
            chunk=ids[off:off+256]
            state=engine.backbone(mx.array(chunk)[None,:],cache=cache)
            sel=[p for p in logit_pos if off<=p<off+len(chunk)]
            if sel:
                lg=engine.embed_logits(state[:,mx.array([p-off for p in sel]),:])[0].astype(mx.float32); mx.eval(lg)
                lgn=np.array(lg)
                for i,p in enumerate(sel): logits[p]=lgn[i]
            mx.eval(state,[c.state for c in cache]); guard(mx)
        del cache,state
    finally:
        uninstall(installed); mx.clear_cache()
    return dec,prefix_pos,logits,sink,allsink

def logit_summary(engine,lg,topk,gold=None):
    z=float(np.logaddexp.reduce(lg.astype(np.float64)))
    best=np.argpartition(lg,-topk)[-topk:]; best=best[np.argsort(lg[best])[::-1]]
    out={'logsumexp':z,'topk':[{'id':int(i),'text':engine.tok.decode([int(i)]),'logit':float(lg[i])} for i in best],
         'watch':{w:float(lg[i]) for w,i in engine.watch_ids.items()},'argmax':int(best[0])}
    if gold is not None: out['gold']={'id':int(gold),'text':engine.tok.decode([int(gold)]),'logit':float(lg[gold]),'logprob':float(lg[gold]-z),'rank':int((lg>lg[gold]).sum())+1}
    return out

# ----------------------------------------------------------------------------- passes
def run_generate(engine,budget,items,models,edit_arrays,variant,label):
    result={'rows':{}}
    for which in models:
        if which=='edited': engine.apply_edit(edit_arrays)
        else: engine.revert()
        rows=[]
        for it in items:
            guard(engine.mx); p,ids=engine.prompt_ids(it['input'],variant)
            rid=budget.reserve(len(ids)+MAX_TOKENS,MAX_TOKENS,it['id'],f'generate:{which}')
            t0=time.perf_counter(); text,toks,gen,trunc=engine.generate(ids); dt=time.perf_counter()-t0
            m=guard(engine.mx); s=score(it,text)
            row={'case_id':it['id'],'action':it['action'],'family':it['family'],'model':which,'raw_output':text,'token_ids':toks,'input_tokens':len(ids),'output_tokens':gen,'truncated':trunc,'seconds':dt,'score':s,'prompt_hash':objhash(p)}
            budget.finish(rid,len(ids)+max(gen-1,0),gen,seconds=dt,memory=m); rows.append(row)
            print(f"  {which:8s} {it['id']:<40s} {it['action']:8s} {'OK ' if s['correct'] else 'BAD'} {'strictOK' if s['correct_strict'] else 'strictBAD'} dec={s['decision']} | {text[:70]!r}",flush=True)
        result['rows'][which]=rows; result[which]=summarize(rows); print(json.dumps({which:result[which]}),flush=True)
    if models==['original','edited']: result['paired']=paired(result['rows']['original'],result['rows']['edited']); print(json.dumps({'paired':{k:v for k,v in result['paired'].items() if 'ids' not in k}}),flush=True)
    engine.revert(); return result

def run_teacher(engine,budget,items,which,edit_arrays,variant,continuation,capture,layers,columns,label,mlpin=(),colstats=()):
    if which=='edited': engine.apply_edit(edit_arrays)
    rows=[]; arrays={'h_dec':[],'mlp_dec':[],'attn_dec':[],'logits_dec':[]}; prefix_arrays={'h_pre':[],'mlp_pre':[],'attn_pre':[],'ids':[]}; allcols={}
    for it in items:
        guard(engine.mx); p,pids=engine.prompt_ids(it['input'],variant)
        if continuation=='gold': cont=gold_text(it['case']) if it['case'] is not None else ''
        elif continuation=='none': cont=''
        elif continuation.startswith('text:'): cont=continuation[5:]
        elif continuation.startswith('fromrun:'):
            _,lab,which=continuation.split(':'); src_rows=json.loads((ROOT/'runs'/f'{lab}.json').read_text())['rows'][which]
            cont=next(rr['raw_output'] for rr in src_rows if rr['case_id']==it['id'])
        else: raise ValueError(continuation)
        ids=engine.tok.encode(p+cont,add_special_tokens=False)
        if ids[:len(pids)]!=pids: raise RuntimeError('prompt/continuation tokenization boundary unstable: '+it['id'])
        if len(ids)>CONTEXT: raise ValueError('context')
        rid=budget.reserve(len(ids),0,it['id'],f'teacher:{which}')
        t0=time.perf_counter()
        dec,prefix_pos,logits,sink,allsink=teacher_case(engine,ids,len(pids),'prefix' in capture,layers,columns,24,mlpin,colstats)
        dt=time.perf_counter()-t0; m=guard(engine.mx)
        row={'case_id':it['id'],'action':it['action'],'family':it['family'],'model':which,'prompt_tokens':len(pids),'sequence_tokens':len(ids),'decision_position':dec,'continuation':cont,
             'decision':logit_summary(engine,logits[dec],24,ids[dec+1] if len(ids)>dec+1 else None),'seconds':dt}
        d=row['decision']['watch']; row['gaps']={'call_minus_NO':d['<tool_call>']-d['NO'],'call_minus_NE':d['<tool_call>']-d['NE'],'NE_minus_NO':d['NE']-d['NO'],'call_minus_argmax_other':d['<tool_call>']-max(v for w,v in d.items() if w!='<tool_call>')}
        cont_rows=[]
        for pos in range(dec,len(ids)-1):
            lg=logits[pos]; g=ids[pos+1]; z=float(np.logaddexp.reduce(lg.astype(np.float64)))
            cont_rows.append({'position':pos+1,'token':int(g),'text':engine.tok.decode([int(g)]),'logprob':float(lg[g]-z),'rank':int((lg>lg[g]).sum())+1,'argmax':int(lg.argmax()),'argmax_text':engine.tok.decode([int(lg.argmax())]),'watch':{w:float(lg[i]) for w,i in engine.watch_ids.items() if w in ('<tool_call>','NO','NE','<|im_end|>')},'top3':[engine.tok.decode([int(i)]) for i in np.argsort(lg)[::-1][:3]]})
        row['continuation_tokens']=cont_rows
        if prefix_pos is not None: row['prefix']={'position':prefix_pos,**logit_summary(engine,logits[prefix_pos],16,ids[prefix_pos+1])}
        arrays['h_dec'].append(np.stack([sink[(('h',l),dec)] for l in range(25)]))
        arrays['mlp_dec'].append(np.stack([sink[(('mlp',l),dec)] for l in layers]) if layers else np.zeros((0,3584),np.float32))
        arrays['attn_dec'].append(np.stack([sink[(('attn',l),dec)] for l in FULL_ATTN]))
        arrays['logits_dec'].append(logits[dec].astype(np.float16))
        if prefix_pos is not None:
            prefix_arrays['h_pre'].append(np.stack([sink[(('h',l),prefix_pos)] for l in range(25)]))
            prefix_arrays['mlp_pre'].append(np.stack([sink[(('mlp',l),prefix_pos)] for l in layers]) if layers else np.zeros((0,3584),np.float32))
            prefix_arrays['attn_pre'].append(np.stack([sink[(('attn',l),prefix_pos)] for l in FULL_ATTN])); prefix_arrays['ids'].append(it['id'])
        for key,chunks in allsink.items():
            if key[0]=='full': allcols[f"mlpin_L{key[2]}_{it['id']}"]=np.concatenate(chunks,0)
            elif key[0]=='stats':
                mxs=np.max(np.stack([c[0] for c in chunks]),0); sm=np.sum(np.stack([c[1] for c in chunks]),0)/sum(c[2] for c in chunks)
                allcols.setdefault(f"colstats_L{key[2]}_max",[]).append(mxs); allcols.setdefault(f"colstats_L{key[2]}_mean",[]).append(sm)
            else: allcols[f"cols_L{key[1]}_{it['id']}"]=np.concatenate(chunks,0)
        budget.finish(rid,len(ids),0,seconds=dt,memory=m); rows.append(row)
        g=row['gaps']; print(f"  {it['id']:<40s} {it['action']:8s} argmax={row['decision']['topk'][0]['text']!r:12s} call-NO={g['call_minus_NO']:6.2f} call-NE={g['call_minus_NE']:6.2f} NE-NO={g['NE_minus_NO']:6.2f} cont_lp={[round(c['logprob'],2) for c in cont_rows[:6]]}",flush=True)
    out={k:np.stack(v) for k,v in arrays.items()}
    if prefix_arrays['ids']:
        for k in ('h_pre','mlp_pre','attn_pre'): out[k]=np.stack(prefix_arrays[k])
        out['prefix_ids']=np.array(prefix_arrays['ids'])
    out.update({k:(np.stack(v) if isinstance(v,list) else v) for k,v in allcols.items()}); out['case_ids']=np.array([r['case_id'] for r in rows]); out['layers']=np.array(list(layers))
    engine.revert()
    return {'rows':rows,'model':which,'layers':list(layers),'capture':capture,'continuation':continuation},out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['generate','teacher']); ap.add_argument('--set',required=True); ap.add_argument('--ids')
    ap.add_argument('--model',default='original',choices=['original','edited','both']); ap.add_argument('--label',required=True); ap.add_argument('--hypothesis',required=True); ap.add_argument('--prediction',required=True)
    ap.add_argument('--edit',default='current'); ap.add_argument('--prompt',default='standard'); ap.add_argument('--continuation',default='gold'); ap.add_argument('--capture',default='decision,prefix')
    ap.add_argument('--layers',default='0-23'); ap.add_argument('--columns',default=''); ap.add_argument('--limit',type=int); ap.add_argument('--mlpin',default=''); ap.add_argument('--colstats',default='')
    a=ap.parse_args()
    target=ROOT/'runs'/f'{a.label}.json'
    if target.exists(): raise FileExistsError(target)
    ids=[s for s in a.ids.split(',') if s] if a.ids else None
    if a.ids and Path(a.ids).exists(): ids=[l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
    items=items_for(a.set,ids)
    if a.limit: items=items[:a.limit]
    models=['original','edited'] if a.model=='both' else [a.model]
    if len(items)*len(models)>64: raise ValueError(f'{len(items)*len(models)} executions exceed 64 per pass')
    edit_arrays={}
    if 'edited' in models:
        path=ROOT/'edits'/('current.npz' if a.edit=='current' else f'{a.edit}.npz')
        with np.load(path,allow_pickle=False) as z: edit_arrays={n:z[n] for n in z.files}
        if not edit_arrays: print('WARNING: edited model has an empty edit set (identical to original)')
    layers=[]
    for part in a.layers.split(','):
        if not part: continue
        if '-' in part: lo,hi=part.split('-'); layers+=list(range(int(lo),int(hi)+1))
        else: layers.append(int(part))
    columns={}
    if a.columns.startswith('@'): a.columns=Path(a.columns[1:]).read_text().strip()
    for part in a.columns.split(';'):
        if part: l,cs=part.split(':'); columns[int(l)]=[int(c) for c in cs.split(',')]
    ledger=Ledger(ROOT/'ledger.jsonl'); budget=Budget(ledger,requests=50,executions=64,input_tokens=50_000_000,output_tokens=2_000_000,candidates=200)
    with (ROOT/'notes.md').open('a') as f: f.write(f"\n### pass {a.label} ({a.mode}, set={a.set}, model={a.model}, n={len(items)}, edit={a.edit if 'edited' in models else '-'})\n\nHypothesis: {a.hypothesis}\n\nPrediction: {a.prediction}\n")
    with accelerator_lock(), memory_policy(allow_warning=True):
        engine=Engine(); print(json.dumps({'loaded_memory':engine.initial_memory}),flush=True)
        with budget.request(a.hypothesis,a.prediction,a.label):
            t0=time.perf_counter()
            if a.mode=='generate':
                result=run_generate(engine,budget,items,models,edit_arrays,a.prompt,a.label)
            else:
                mlpin=[int(x) for x in a.mlpin.split(',') if x]; colstats=[int(x) for x in a.colstats.split(',') if x]
                result,arrays=run_teacher(engine,budget,items,models[0],edit_arrays,a.prompt,a.continuation,a.capture.split(','),layers,columns,a.label,mlpin,colstats)
                np.savez_compressed(ROOT/'runs'/f'{a.label}.npz',**arrays)
            result.update({'label':a.label,'mode':a.mode,'set':a.set,'models':models,'edit':a.edit if 'edited' in models else None,'prompt_variant':a.prompt,'n_items':len(items),'seconds':time.perf_counter()-t0,
                           'usage':budget.usage(),'memory':{'rss_gib':memory(engine.mx)['rss_bytes']/GIB,'backend_peak_gib':engine.mx.get_peak_memory()/GIB},'edit_tensors':sorted(edit_arrays)})
            write_json(target,result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired')}),flush=True)
        engine.revert(); del engine; gc.collect()
if __name__=='__main__': main()
