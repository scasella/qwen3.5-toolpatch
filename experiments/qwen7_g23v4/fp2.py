#!/usr/bin/env python
"""Forward-pass tool, qwen-7 extension of fp.py. One budgeted pass per invocation (<=64 executions; shared 50-pass ledger with fp.py).

  generate: fp2.py generate --set S [--ids ...] --models original,g23v3,g23v4 --label L --hypothesis H --prediction P [--prompt standard]
            (each non-'original' entry is an edit label edits/<label>.npz; rows per model, paired transitions vs the first model)
  teacher:  fp2.py teacher --set S [--ids ...] --model original|<edit label> --label L --hypothesis H --prediction P [--continuation none|gold|text:..]
            [--roles] (tag role positions from data/<set>.meta.json: v1 v1c v2 v2c x y conn tail end imend dec sysamb)
            [--hfull 19,23] (residual input of those layers at every position, fp16; 24 = final-norm input)
            [--mlpin 18,22,23] (MLP input at every position, fp16) [--colstats 15,19,21,22] [--attnrows] (attention rows from the dec/imend/end queries at the six
            full-attention layers over every key) [--layers 23] (down_proj input at tagged positions)
Protocol fixed: official chat template with native <tools>, enable_thinking=False, greedy, max 128 new tokens, context <=2048."""
from __future__ import annotations
import argparse, json, os, sys, time, gc
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
from runtime.common import ROOT, BASE, Ledger, write_json, objhash
from runtime.resources import Budget, accelerator_lock, guard, memory, memory_policy, GIB
from benchmark.render import prompt as render_prompt, gold_text
from benchmark.contracts import ModelInput, from_dict
from fp import Engine, score, summarize, paired, logit_summary, MAX_TOKENS, CONTEXT, FULL_ATTN, WATCH
import fp as fp1

ROLE_QUERIES=('dec','imend','end')

def items_for(set_name, ids=None):
    if '+' in set_name:
        out=[]
        for s_ in set_name.split('+'):
            for it in items_for(s_): it['set']=s_; out.append(it)
        if ids:
            want=set(ids); out=[i for i in out if i['id'] in want]; missing=want-{i['id'] for i in out}
            if missing: raise ValueError(f'unknown ids: {sorted(missing)}')
        return out
    if set_name in ('regression','regression2','prose_qa','prose_qa2'): items=fp1.items_for(set_name,ids)
    else:
        rows=[json.loads(l) for l in (ROOT/f'data/{set_name}.jsonl').read_text().splitlines() if l]
        cases=[from_dict(r) for r in rows]
        items=[{'id':c.id,'input':c.model_input(),'case':c,'action':c.spec.action,'family':c.family} for c in cases]
        if ids:
            want=set(ids); items=[i for i in items if i['id'] in want]; missing=want-{i['id'] for i in items}
            if missing: raise ValueError(f'unknown ids: {sorted(missing)}')
    return items

def load_edit(label):
    if label=='original': return {}
    with np.load(ROOT/'edits'/f'{label}.npz',allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    if not arrays: raise ValueError('empty edit '+label)
    return arrays

# ----------------------------------------------------------------------------- role tagging
def tag_roles(tok, p, ids, meta):
    hf=tok if callable(tok) else tok._tokenizer   # mlx_lm's TokenizerWrapper is not callable; its _tokenizer is the HF fast tokenizer
    enc=hf(p,add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==ids, 'offset tokenization differs'
    off=enc['offset_mapping']
    def last_tok(a,b):
        hits=[i for i,(x,y) in enumerate(off) if x<b and y>a]
        return hits[-1] if hits else None
    roles={}
    cs=p.index(meta['clause']); assert p.count(meta['clause'])==1
    for role,(a,b) in meta['spans'].items():
        seg=meta['clause'][a:b]
        if role in ('conn','tail'):
            s=seg.strip()
            if not s: continue
            a2=a+seg.index(s); b2=a2+len(s); roles[role]=last_tok(cs+a2,cs+b2)
        else:
            roles[role]=last_tok(cs+a,cs+b)
            if role in ('v1','v2','x','y') and seg.endswith('"') and len(seg)>2: roles[role+'c']=last_tok(cs+a,cs+b-1)
    ce=cs+len(meta['clause']); ie=p.index('<|im_end|>',ce); roles['imend']=last_tok(ie,ie+len('<|im_end|>'))
    sa=p.index('missing or ambiguous'); roles['sysamb']=last_tok(sa+len('missing or '),sa+len('missing or ambiguous'))
    roles['dec']=len(ids)-1
    return roles

# ----------------------------------------------------------------------------- observers
class Obs:
    def __new__(cls,inner,positions,sink,key,full=False,stats=False,allsink=None):
        import mlx.core as mx, mlx.nn as nn
        from types import SimpleNamespace
        class _O(nn.Module):
            def __init__(self):
                super().__init__(); self.inner=inner; self._st=SimpleNamespace(positions=list(positions),sink=sink,key=key,offset=0,full=full,stats=stats,allsink=allsink)
            def __call__(self,x,*a,**k):
                st=self._st; L=x.shape[1]
                for p in st.positions:
                    q=p-st.offset
                    if 0<=q<L:
                        v=x[0,q,:].astype(mx.float32); mx.eval(v); st.sink[(st.key,p)]=np.array(v)
                if st.full:
                    v=x[0].astype(mx.float16); mx.eval(v); st.allsink.setdefault(('full',)+tuple(st.key),[]).append(np.array(v))
                if st.stats:
                    xf=x[0].astype(mx.float32); m_=mx.abs(xf).max(axis=0); sm=xf.sum(axis=0); mx.eval(m_,sm)
                    st.allsink.setdefault(('stats',)+tuple(st.key),[]).append((np.array(m_),np.array(sm),L))
                st.offset+=L
                return self.inner(x,*a,**k)
        return _O()

class AttnState:
    layer=None; queries=[]; rows={}
def attn_wrap(inner,l):
    import mlx.nn as nn
    class _A(nn.Module):
        def __init__(self): super().__init__(); self.inner=inner
        def __call__(self,x,*a,**k):
            AttnState.layer=l; return self.inner(x,*a,**k)
    return _A()
def patched_sdpa(orig):
    import mlx.core as mx
    def f(queries,keys,values,cache,scale,mask,sinks=None):
        if AttnState.layer is not None and AttnState.queries:
            Lq=queries.shape[2]; S=keys.shape[2]; off=S-Lq; H=queries.shape[1]; Hk=keys.shape[1]; g=H//Hk
            for p in AttnState.queries:
                if off<=p<off+Lq:
                    q=queries[0,:,p-off,:].astype(mx.float32)                       # (H,D)
                    kk=keys[0].astype(mx.float32)[:, :p+1, :]                           # (Hk,p+1,D)
                    kk=mx.repeat(kk,g,axis=0)                                           # (H,p+1,D)
                    sc=(q[:,None,:]*kk).sum(-1)*scale                                   # (H,p+1)
                    pr=mx.softmax(sc,axis=-1); mx.eval(pr)
                    AttnState.rows[(AttnState.layer,p)]=np.array(pr)
        return orig(queries,keys,values,cache,scale,mask,sinks)
    return f

def install(engine,positions,layers,hfull,mlpin,colstats,attnrows,queries):
    bb=engine.backbone; installed=[]; sink={}; allsink={}
    def wrap(parent,attr,key,full=False,stats=False):
        orig=getattr(parent,attr); setattr(parent,attr,Obs(orig,positions,sink,key,full,stats,allsink)); installed.append((parent,attr,orig))
    for l in range(24):
        wrap(bb.layers[l],'input_layernorm',('h',l),full=(l in hfull))
        if l in layers or l in colstats: wrap(bb.layers[l].mlp,'down_proj',('mlp',l),False,l in colstats)
        if l in mlpin: wrap(bb.layers[l],'mlp',('mlpin',l),full=True)
        if l in FULL_ATTN:
            wrap(bb.layers[l].self_attn,'o_proj',('attn',l))
            if attnrows:
                orig=bb.layers[l].self_attn; setattr(bb.layers[l],'self_attn',attn_wrap(orig,l)); installed.append((bb.layers[l],'self_attn',orig))
    wrap(bb,'norm',('h',24),full=(24 in hfull))
    patched=None
    if attnrows:
        import mlx_lm.models.qwen3_next as qn
        patched=(qn,qn.scaled_dot_product_attention); qn.scaled_dot_product_attention=patched_sdpa(patched[1])
        AttnState.queries=list(queries); AttnState.rows={}; AttnState.layer=None
    return installed,sink,allsink,patched
def uninstall(installed,patched):
    for parent,attr,orig in reversed(installed): setattr(parent,attr,orig)
    if patched: patched[0].scaled_dot_product_attention=patched[1]
    AttnState.queries=[]; AttnState.layer=None

def teacher_case(engine,ids,prompt_len,tag_positions,layers,hfull,mlpin,colstats,attnrows):
    mx=engine.mx; n=len(ids); dec=prompt_len-1
    logit_pos=list(range(dec,n-1)) if n>prompt_len else [dec]
    cap=sorted(set([dec]+[p for p in tag_positions.values() if p is not None]))
    queries=[tag_positions.get(r) for r in ROLE_QUERIES if tag_positions.get(r) is not None]
    installed,sink,allsink,patched=install(engine,cap,layers,hfull,mlpin,colstats,attnrows,queries)
    logits={}
    try:
        cache=engine.model.make_cache()
        for off in range(0,n,256):
            chunk=ids[off:off+256]
            state=engine.backbone(mx.array(chunk)[None,:],cache=cache)
            sel=[p for p in logit_pos if off<=p<off+len(chunk)]
            if sel:
                lg=engine.embed_logits(state[:,mx.array([p-off for p in sel]),:])[0].astype(mx.float32); mx.eval(lg); lgn=np.array(lg)
                for i,p in enumerate(sel): logits[p]=lgn[i]
            mx.eval(state,[c.state for c in cache]); guard(mx)
        del cache,state
    finally:
        rows=dict(AttnState.rows); uninstall(installed,patched); mx.clear_cache()
    return dec,logits,sink,allsink,rows

def run_generate(engine,budget,items,models,variant):
    result={'rows':{}}
    for which in models:
        engine.revert()
        if which!='original': engine.apply_edit(load_edit(which))
        rows=[]
        for it in items:
            guard(engine.mx); p,ids=engine.prompt_ids(it['input'],variant)
            rid=budget.reserve(len(ids)+MAX_TOKENS,MAX_TOKENS,it['id'],f'generate:{which}')
            t0=time.perf_counter(); text,toks,gen,trunc=engine.generate(ids); dt=time.perf_counter()-t0
            m=guard(engine.mx); s=score(it,text)
            row={'case_id':it['id'],'action':it['action'],'family':it['family'],'model':which,'raw_output':text,'token_ids':toks,'input_tokens':len(ids),'output_tokens':gen,'truncated':trunc,'seconds':dt,'score':s,'prompt_hash':objhash(p)}
            budget.finish(rid,len(ids)+max(gen-1,0),gen,seconds=dt,memory=m); rows.append(row)
            print(f"  {which:8s} {it['id']:<48s} {it['action']:8s} {'OK ' if s['correct'] else 'BAD'} {'strictOK' if s['correct_strict'] else 'strictBAD'} dec={s['decision']} | {text[:70]!r}",flush=True)
        result['rows'][which]=rows; result[which]=summarize(rows); print(json.dumps({which:result[which]}),flush=True)
    if len(models)>1:
        result['paired']={}
        for which in models[1:]:
            result['paired'][f'{models[0]}->{which}']=paired(result['rows'][models[0]],result['rows'][which])
            print(json.dumps({f'paired {models[0]}->{which}':{k:v for k,v in result['paired'][f'{models[0]}->{which}'].items() if 'ids' not in k and k!='decision_changes'}}),flush=True)
    engine.revert(); return result

def run_teacher(engine,budget,items,which,variant,continuation,roles_meta,layers,hfull,mlpin,colstats,attnrows,a_set_names=()):
    engine.revert()
    if which!='original': engine.apply_edit(load_edit(which))
    rows=[]; arrays={}; per_item={}; colacc={}
    for it in items:
        guard(engine.mx); p,pids=engine.prompt_ids(it['input'],variant)
        spec=continuation
        if ',' in continuation:   # per-set specs, aligned with a '+'-joined --set
            spec=dict(zip(a_set_names,continuation.split(',')))[it['set']]
        if spec=='gold': cont=gold_text(it['case']) if it['case'] is not None else ''
        elif spec=='none': cont=''
        elif spec.startswith('text:'): cont=spec[5:]
        elif spec.startswith('fromqwen6:'):
            from runtime.common import QWEN6
            _,lab,wh=spec.split(':'); src_rows=json.loads((QWEN6/'runs'/f'{lab}.json').read_text())['rows'][wh]
            cont=next(rr['raw_output'] for rr in src_rows if rr['case_id']==it['id'])
        else: raise ValueError(spec)
        ids=engine.tok.encode(p+cont,add_special_tokens=False)
        if ids[:len(pids)]!=pids: raise RuntimeError('prompt/continuation tokenization boundary unstable: '+it['id'])
        if len(ids)>CONTEXT: raise ValueError('context')
        tags=tag_roles(engine.tok,p,pids,roles_meta[it['id']]) if roles_meta else {'dec':len(pids)-1}
        if roles_meta and '_pretagged' in roles_meta[it['id']]: assert roles_meta[it['id']]['_pretagged']==(tags,len(pids)), 'pre-tagged roles differ from engine tokenization'
        rid=budget.reserve(len(ids),0,it['id'],f'teacher:{which}')
        t0=time.perf_counter(); dec,logits,sink,allsink,attn_rows=teacher_case(engine,ids,len(pids),tags,layers,hfull,mlpin,colstats,attnrows); dt=time.perf_counter()-t0; m=guard(engine.mx)
        row={'case_id':it['id'],'action':it['action'],'family':it['family'],'model':which,'prompt_tokens':len(pids),'sequence_tokens':len(ids),'decision_position':dec,'continuation':cont,'roles':tags,
             'decision':logit_summary(engine,logits[dec],24,ids[dec+1] if len(ids)>dec+1 else None),'seconds':dt}
        d=row['decision']['watch']; row['gaps']={'call_minus_NO':d['<tool_call>']-d['NO'],'call_minus_NE':d['<tool_call>']-d['NE'],'NE_minus_NO':d['NE']-d['NO']}
        cont_rows=[]
        for pos in range(dec,len(ids)-1):
            lg=logits[pos]; g=ids[pos+1]; z=float(np.logaddexp.reduce(lg.astype(np.float64)))
            cont_rows.append({'position':pos+1,'token':int(g),'text':engine.tok.decode([int(g)]),'logprob':float(lg[g]-z),'rank':int((lg>lg[g]).sum())+1,'argmax':int(lg.argmax()),'argmax_text':engine.tok.decode([int(lg.argmax())])})
        row['continuation_tokens']=cont_rows
        cid=it['id']; poslist=sorted(set([dec]+[v for v in tags.values() if v is not None]))
        arrays[f'pos_{cid}']=np.array(poslist)
        arrays[f'h_{cid}']=np.stack([np.stack([sink[(('h',l),pp)] for l in range(25)]) for pp in poslist]).astype(np.float16)        # (P,25,1024)
        arrays[f'attn_{cid}']=np.stack([np.stack([sink[(('attn',l),pp)] for l in FULL_ATTN]) for pp in poslist]).astype(np.float16)  # (P,6,2048)
        if layers: arrays[f'mlp_{cid}']=np.stack([np.stack([sink[(('mlp',l),pp)] for l in layers]) for pp in poslist]).astype(np.float16)
        arrays[f'logits_{cid}']=logits[dec].astype(np.float16)
        for key,chunks in allsink.items():
            if key[0]=='full': arrays[f"{key[1]}{key[2]}_{cid}"]=np.concatenate(chunks,0)       # h19_<id>, mlpin22_<id> ...
            elif key[0]=='stats':
                mxs=np.max(np.stack([c[0] for c in chunks]),0); sm=np.sum(np.stack([c[1] for c in chunks]),0)/sum(c[2] for c in chunks)
                colacc.setdefault(f"colstats_L{key[2]}_max",[]).append(mxs); colacc.setdefault(f"colstats_L{key[2]}_mean",[]).append(sm)
        if attn_rows:
            qs=[tags.get(r) for r in ROLE_QUERIES]; T=len(ids)
            A=np.zeros((len(FULL_ATTN),len(ROLE_QUERIES),8,T),np.float32)
            for li,l in enumerate(FULL_ATTN):
                for qi,q in enumerate(qs):
                    if q is not None and (l,q) in attn_rows: A[li,qi,:,:q+1]=attn_rows[(l,q)]
            arrays[f'rows_{cid}']=A.astype(np.float16)
        budget.finish(rid,len(ids),0,seconds=dt,memory=m); rows.append(row)
        g=row['gaps']; print(f"  {cid:<48s} {it['action']:8s} argmax={row['decision']['topk'][0]['text']!r:12s} call-NO={g['call_minus_NO']:6.2f} call-NE={g['call_minus_NE']:6.2f} NE-NO={g['NE_minus_NO']:6.2f} roles={ {k:v for k,v in tags.items() if k in ('v1','v2','end','imend','dec')} }",flush=True)
    for k,v in colacc.items(): arrays[k]=np.stack(v)
    arrays['case_ids']=np.array([r['case_id'] for r in rows]); arrays['layers']=np.array(list(layers)); arrays['full_attn']=np.array(FULL_ATTN); arrays['role_queries']=np.array(ROLE_QUERIES)
    engine.revert()
    return {'rows':rows,'model':which,'layers':list(layers),'hfull':list(hfull),'mlpin':list(mlpin),'colstats':list(colstats),'attnrows':attnrows,'continuation':continuation},arrays

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['generate','teacher']); ap.add_argument('--set',required=True); ap.add_argument('--ids')
    ap.add_argument('--models',default='original'); ap.add_argument('--model',default='original'); ap.add_argument('--label',required=True); ap.add_argument('--hypothesis',required=True); ap.add_argument('--prediction',required=True)
    ap.add_argument('--prompt',default='standard'); ap.add_argument('--continuation',default='none'); ap.add_argument('--roles',action='store_true')
    ap.add_argument('--layers',default=''); ap.add_argument('--hfull',default=''); ap.add_argument('--mlpin',default=''); ap.add_argument('--colstats',default=''); ap.add_argument('--attnrows',action='store_true'); ap.add_argument('--limit',type=int)
    a=ap.parse_args()
    target=ROOT/'runs'/f'{a.label}.json'
    if target.exists(): raise FileExistsError(target)
    ids=[s for s in a.ids.split(',') if s] if a.ids else None
    if a.ids and Path(a.ids).exists(): ids=[l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
    items=items_for(a.set,ids)
    if a.limit: items=items[:a.limit]
    models=[m for m in a.models.split(',') if m] if a.mode=='generate' else [a.model]
    if len(items)*len(models)>64: raise ValueError(f'{len(items)*len(models)} executions exceed 64 per pass')
    for m in models: load_edit(m)   # fail early on a missing edit
    ints=lambda s:[int(x) for x in s.split(',') if x]
    layers=ints(a.layers); hfull=ints(a.hfull); mlpin=ints(a.mlpin); colstats=ints(a.colstats)
    roles_meta=json.loads((ROOT/f'data/{a.set}.meta.json').read_text()) if a.roles else None
    if roles_meta is not None:   # fail before anything is charged: tag every item with the HF tokenizer now
        from transformers import AutoTokenizer
        _tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
        for it in items:
            _p=render_prompt(_tok,it['input'],a.prompt,False); _ids=_tok.encode(_p,add_special_tokens=False); _t=tag_roles(_tok,_p,_ids,roles_meta[it['id']])
            roles_meta[it['id']]['_pretagged']=(_t,len(_ids))
    ledger=Ledger(ROOT/'ledger.jsonl'); budget=Budget(ledger,requests=50,executions=64,input_tokens=50_000_000,output_tokens=2_000_000,candidates=200)
    with (ROOT/'notes.md').open('a') as f: f.write(f"\n### pass {a.label} ({a.mode}, set={a.set}, models={models}, n={len(items)})\n\nHypothesis: {a.hypothesis}\n\nPrediction: {a.prediction}\n")
    with accelerator_lock(), memory_policy(allow_warning=True):
        engine=Engine(); print(json.dumps({'loaded_memory':engine.initial_memory}),flush=True)
        with budget.request(a.hypothesis,a.prediction,a.label):
            t0=time.perf_counter()
            if a.mode=='generate': result=run_generate(engine,budget,items,models,a.prompt)
            else:
                result,arrays=run_teacher(engine,budget,items,models[0],a.prompt,a.continuation,roles_meta,layers,hfull,mlpin,colstats,a.attnrows,a.set.split('+'))
                np.savez_compressed(ROOT/'runs'/f'{a.label}.npz',**arrays)
            result.update({'label':a.label,'mode':a.mode,'set':a.set,'models':models,'prompt_variant':a.prompt,'n_items':len(items),'seconds':time.perf_counter()-t0,
                           'usage':budget.usage(),'memory':{'rss_gib':memory(engine.mx)['rss_bytes']/GIB,'backend_peak_gib':engine.mx.get_peak_memory()/GIB}})
            write_json(target,result); print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired')}),flush=True)
        engine.revert(); del engine; gc.collect()
if __name__=='__main__': main()
