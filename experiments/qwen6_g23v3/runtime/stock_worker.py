"""Fresh stock inference subprocess. No discovery, intervention, patch or scorer imports.
Parent reserves budget and owns the accelerator lock. Not an independent experiment CLI.
"""
import os,sys,json,time,subprocess
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
import psutil

def main():
    job=json.load(open(sys.argv[1])); mx.set_default_device(mx.gpu)
    lock_fd=int(os.environ['TOOLPATCH_LOCK_FD'])
    os.fstat(lock_fd)  # Parent passes the held shared lock; direct unlocked execution fails.
    if not mx.metal.is_available(): raise RuntimeError('Metal required')
    mx.set_memory_limit(12*1024**3); mx.set_cache_limit(512*1024**2)
    model,tok=load(job['checkpoint']); model.eval()
    rows=[]; pressure_at=0.; pressure_count=0
    for item in job['items']:
        text='';tokens=[];last=None;start=time.perf_counter()
        for response in stream_generate(model,tok,item['tokens'],max_tokens=item['max_tokens'],sampler=make_sampler(0),prompt_cache=model.make_cache(),prefill_step_size=256):
            text+=response.text;tokens.append(response.token);last=response
            if time.monotonic()-pressure_at>=2:
                p=subprocess.run(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True)
                if p.returncode:raise RuntimeError('Cannot inspect kernel memory pressure')
                level=int(p.stdout);pressure_at=time.monotonic();pressure_count=pressure_count+1 if level>1 else 0
                if level>=4 or (os.environ.get('TOOLPATCH_MEMORY_PRESSURE_POLICY')!='warning_allowed' and pressure_count>=3):raise MemoryError('kernel memory pressure')
            if max(psutil.Process().memory_info().rss,mx.get_active_memory()+mx.get_cache_memory())>=12*1024**3 or psutil.virtual_memory().available<3*1024**3: raise MemoryError('resource stop')
        rows.append({'case_id':item['case_id'],'raw_output':text,'token_ids':tokens,'output_tokens':last.generation_tokens,'input_tokens':len(item['tokens']),'seconds':time.perf_counter()-start,'truncated':last.finish_reason=='length','memory':{'rss_bytes':psutil.Process().memory_info().rss,'backend_peak_bytes':mx.get_peak_memory(),'backend_active_bytes':mx.get_active_memory(),'backend_cache_bytes':mx.get_cache_memory()}})
        mx.clear_cache()
    json.dump({'rows':rows,'forbidden_imports':[x for x in sys.modules if x.startswith(('interventions','patches','eval','benchmark','runtime.v2','runtime.v3','runtime.v4','runtime.v5','runtime.campaign'))]},open(job['output'],'w'))
if __name__=='__main__': main()
