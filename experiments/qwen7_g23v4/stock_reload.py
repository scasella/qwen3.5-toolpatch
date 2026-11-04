"""Export the current/named edit into a full HF checkpoint copy (only edited BF16 tensors byte-replaced) and run fresh stock MLX-LM
generation in a subprocess that imports nothing from this campaign (runtime/stock_worker.py). One budget pass (<=64 executions)."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from runtime.common import ROOT, BASE, Ledger, write_json, digest
from runtime.inspect import header, tensor_bits
from runtime.resources import Budget, accelerator_lock, memory_policy
from benchmark.render import prompt as render_prompt
from fp import score, summarize, paired, MAX_TOKENS
from fp2 import items_for
BASE_FILE=next(BASE.glob('*.safetensors'))

def export(label):
    target=ROOT/'artifacts/exports'/label
    with np.load(ROOT/'edits'/f'{label}.npz',allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    if target.exists(): return target,arrays
    target.mkdir(parents=True)
    for src in BASE.iterdir():
        if src.is_file() or src.is_symlink(): shutil.copyfile(src,target/src.name)
    out=target/BASE_FILE.name; hh,offset=header(out)
    with out.open('r+b') as f:
        for name,a in arrays.items():
            start,end=hh[name]['data_offsets']; raw=np.ascontiguousarray(a).tobytes(); assert len(raw)==end-start
            f.seek(offset+start); f.write(raw)
        f.flush(); os.fsync(f.fileno())
    for name,a in arrays.items():
        assert np.array_equal(tensor_bits(out,name)[1],a)
    # every non-edited tensor must be byte-identical to the base
    h0,_=header(BASE_FILE)
    for name in h0:
        if name=='__metadata__' or name in arrays: continue
        assert np.array_equal(tensor_bits(out,name)[1],tensor_bits(BASE_FILE,name)[1]), name
    write_json(target/'export_manifest.json',{'edit':label,'edited_tensors':sorted(arrays),'safetensors_sha256':digest(out),'base_sha256':digest(BASE_FILE),'format':'original HF multimodal checkpoint; only named BF16 tensors replaced'})
    return target,arrays

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--edit',required=True); ap.add_argument('--set',required=True); ap.add_argument('--ids'); ap.add_argument('--label',required=True)
    ap.add_argument('--checkpoints',default='original,edited'); ap.add_argument('--compare',help='run label whose rows (original/edited) to compare token streams against'); a=ap.parse_args()
    target=ROOT/'runs'/f'{a.label}.json'
    if target.exists(): raise FileExistsError(target)
    ids=[l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()] if a.ids and Path(a.ids).exists() else ([s for s in a.ids.split(',') if s] if a.ids else None)
    items=items_for(a.set,ids); cps=a.checkpoints.split(',')
    if len(items)*len(cps)>64: raise ValueError('exceeds 64 executions')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    exported,arrays=export(a.edit)
    ledger=Ledger(ROOT/'ledger.jsonl'); budget=Budget(ledger,requests=50,executions=64,input_tokens=50_000_000,output_tokens=2_000_000,candidates=200)
    hyp=f'Fresh stock MLX-LM reload of exported edit {a.edit} reproduces in-process token streams exactly for both original and edited checkpoints.'
    with (ROOT/'notes.md').open('a') as f: f.write(f"\n### pass {a.label} (stock reload, set={a.set}, n={len(items)}, checkpoints={cps}, edit={a.edit})\n\nHypothesis: {hyp}\n")
    result={'rows':{},'export':str(exported),'edit':a.edit}
    with accelerator_lock(), memory_policy(allow_warning=True):
        from runtime import resources
        with budget.request(hyp,'token-identical streams; identical semantic scores',a.label):
            for cp in cps:
                ckpt=str(BASE) if cp=='original' else str(exported)
                job={'checkpoint':ckpt,'output':str(ROOT/'runs'/f'{a.label}_{cp}_worker.json'),'items':[]}
                rids=[]
                for it in items:
                    p=render_prompt(tok,it['input'],'regression' if it.get('set',a.set) in ('regression','regression2','prose_qa','prose_qa2') else 'standard',False); tids=tok.encode(p,add_special_tokens=False)
                    job['items'].append({'case_id':it['id'],'tokens':tids,'max_tokens':MAX_TOKENS}); rids.append((budget.reserve(len(tids)+MAX_TOKENS,MAX_TOKENS,it['id'],f'stock:{cp}'),len(tids)))
                jp=ROOT/'runs'/f'{a.label}_{cp}_job.json'; write_json(jp,job)
                env={**os.environ,'TOOLPATCH_LOCK_FD':str(resources._LOCK_FD),'TOOLPATCH_MEMORY_PRESSURE_POLICY':'warning_allowed'}
                t0=time.perf_counter()
                subprocess.run([sys.executable,'-I',str(ROOT/'runtime/stock_worker.py'),str(jp)],check=True,env=env,pass_fds=[resources._LOCK_FD],cwd=str(ROOT))
                res=json.loads(Path(job['output']).read_text())
                if res['forbidden_imports']: raise RuntimeError('worker imported campaign modules: '+str(res['forbidden_imports']))
                rows=[]
                for (rid,ntok),it,r in zip(rids,items,res['rows']):
                    budget.finish(rid,ntok+max(r['output_tokens']-1,0),r['output_tokens'],seconds=r['seconds'],memory=r['memory'])
                    rows.append({'case_id':it['id'],'action':it['action'],'family':it['family'],'model':cp,'raw_output':r['raw_output'],'token_ids':r['token_ids'],'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],'truncated':r['truncated'],'seconds':r['seconds'],'score':score(it,r['raw_output'])})
                result['rows'][cp]=rows; result[cp]=summarize(rows); result[cp]['worker_seconds']=time.perf_counter()-t0
                print(json.dumps({cp:result[cp]}),flush=True)
            if len(cps)==2: result['paired']=paired(result['rows'][cps[0]],result['rows'][cps[1]])
            if a.compare:
                ref=json.loads((ROOT/'runs'/f'{a.compare}.json').read_text())
                result['stock_vs_inprocess']={}
                for cp in cps:
                    key='original' if cp=='original' else 'edited'
                    if key in ref['rows']:
                        m={r['case_id']:r['token_ids'] for r in ref['rows'][key]}
                        same=[r['case_id'] for r in result['rows'][cp] if r['case_id'] in m and m[r['case_id']]==r['token_ids']]
                        diff=[r['case_id'] for r in result['rows'][cp] if r['case_id'] in m and m[r['case_id']]!=r['token_ids']]
                        result['stock_vs_inprocess'][cp]={'identical':len(same),'different':len(diff),'different_ids':diff}
                print(json.dumps({'stock_vs_inprocess':result['stock_vs_inprocess']}),flush=True)
            result['usage']=budget.usage(); write_json(target,result)
if __name__=='__main__': main()
