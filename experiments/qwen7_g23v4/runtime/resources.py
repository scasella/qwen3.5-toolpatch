from __future__ import annotations
import fcntl, os, json, time, subprocess
from pathlib import Path
from contextlib import contextmanager
import psutil
from .common import Ledger, records
GIB=1024**3
# Both projects must honor this file. Additional NativeALU paths are acquired if discovered.
DEFAULT_LOCK='/tmp/qwen-local-accelerator.lock'
_LOCK_HELD=False
_LOCK_FD=None

@contextmanager
def accelerator_lock(path: str = DEFAULT_LOCK):
    global _LOCK_HELD,_LOCK_FD
    # Acquire both setup-era aliases in stable order; NativeALU may honor either.
    paths=sorted(set([path,'/tmp/qwen35-mlx-accelerator.lock','/tmp/qwen-local-accelerator.lock'])) if path.startswith('/tmp/qwen') else [path]
    files=[]
    previous_held,previous_fd=_LOCK_HELD,_LOCK_FD
    try:
        deadline=time.monotonic()+float(os.environ.get('QWEN6_LOCK_WAIT_S','3600')); waited=False
        for name in paths:
            f=open(name,'a+')
            while True:
                try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.monotonic()>deadline: f.close(); raise RuntimeError(f'accelerator busy: {name}')
                    if not waited: print(f'[lock] accelerator busy ({name}); waiting...',flush=True); waited=True
                    time.sleep(1)
            if waited: print(f'[lock] acquired {name} after wait',flush=True)
            files.append(f)
            f.seek(0); f.truncate(); f.write(json.dumps({'pid':os.getpid(),'project':'ToolPatch','since':time.time()})); f.flush()
        _LOCK_FD=files[0].fileno()
        _LOCK_HELD=True
        yield
    finally:
        _LOCK_HELD,_LOCK_FD=previous_held,previous_fd
        for f in reversed(files):
            fcntl.flock(f,fcntl.LOCK_UN); f.close()

def memory(mx=None) -> dict:
    vm=psutil.virtual_memory()
    return {'rss_bytes':psutil.Process().memory_info().rss,'system_available_bytes':vm.available,
            'backend_active_bytes':int(mx.get_active_memory()) if mx else 0,
            'backend_peak_bytes':int(mx.get_peak_memory()) if mx else 0,
            'backend_cache_bytes':int(mx.get_cache_memory()) if mx else 0}

_pressure_last=0.0
_pressure_warnings=0
_pressure_level=None

@contextmanager
def memory_policy(allow_warning=False):
    if type(allow_warning) is not bool:raise ValueError('memory policy must be boolean')
    key='TOOLPATCH_MEMORY_PRESSURE_POLICY';old=os.environ.get(key)
    os.environ[key]='warning_allowed' if allow_warning else 'strict'
    try:yield
    finally:
        if old is None:os.environ.pop(key,None)
        else:os.environ[key]=old


def guard(mx=None):
    global _pressure_last,_pressure_warnings,_pressure_level
    m=memory(mx)
    allow_warning=os.environ.get('TOOLPATCH_MEMORY_PRESSURE_POLICY')=='warning_allowed'
    m['pressure_policy']='warning_allowed' if allow_warning else 'strict'
    if mx is not None and time.monotonic()-_pressure_last>=2:
        check=subprocess.run(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True)
        if check.returncode!=0: raise RuntimeError('Cannot inspect kernel memory pressure; refusing inference')
        _pressure_level=int(check.stdout.strip());_pressure_last=time.monotonic()
        _pressure_warnings=_pressure_warnings+1 if _pressure_level>1 else 0
        if _pressure_level>=4 or (not allow_warning and _pressure_warnings>=3): raise MemoryError('critical kernel memory pressure' if _pressure_level>=4 else 'sustained kernel memory pressure')
    if mx is not None: m['kernel_memory_pressure_level']=_pressure_level
    # RSS and Metal buffers overlap on unified memory. Do NOT sum them.
    if max(m['rss_bytes'],m['backend_active_bytes']+m['backend_cache_bytes'])>=12*GIB:
        raise MemoryError('12 GiB working-memory target reached (below 16 GiB hard limit)')
    if m['system_available_bytes']<1*GIB: raise MemoryError('system available memory below 1 GiB')
    return m

class Budget:
    def __init__(self, ledger:Ledger, requests=8, executions=64, input_tokens=5000000, output_tokens=20000, candidates=4):
        self.ledger=ledger; self.limits=dict(requests=requests,executions=executions,input_tokens=input_tokens,output_tokens=output_tokens,candidates=candidates)
        if requests is not None and (type(requests) is not int or requests < 1): raise ValueError('invalid request ceiling')
        if any(type(v) is not int or v < 0 for v in (executions,input_tokens,output_tokens,candidates)) or executions>64: raise ValueError('invalid budget')
        self.active=None
    def usage(self):
        rows=records(self.ledger.path)
        reservations=[r for r in rows if r['event']=='reserve']
        done={r['reservation']:r for r in rows if r['event']=='usage'}
        return {'requests':sum(r['event']=='request' for r in rows),'executions':len(reservations),
                'input_tokens':sum(done.get(r['reservation'],r)['input_tokens'] for r in reservations),
                'output_tokens':sum(done.get(r['reservation'],r)['output_tokens'] for r in reservations),
                'candidates':sum(r['event']=='candidate' for r in rows)}
    @contextmanager
    def request(self, hypothesis:str, prediction:str, label:str):
        if self.active is not None: raise RuntimeError('nested request')
        if not hypothesis.strip() or not prediction.strip(): raise ValueError('hypothesis and prediction required')
        if self.limits['requests'] is not None and self.usage()['requests']>=self.limits['requests']: raise RuntimeError('request budget exhausted')
        self.active={'label':label,'executions':0}; self.ledger.add('request',label=label,hypothesis=hypothesis,prediction=prediction)
        try: yield
        except BaseException as e:
            self.ledger.add('request_failed',label=label,error=str(e)); raise
        finally: self.active=None
    def reserve(self, input_tokens:int, output_tokens:int, case_id:str, variant:str):
        if self.active is None: raise RuntimeError('model execution requires a preregistered request')
        if self.active['executions']>=self.limits['executions']: raise RuntimeError('64 executions/request exceeded')
        if any(type(v) is not int or v < 0 for v in (input_tokens,output_tokens)): raise ValueError('invalid token reservation')
        usage=self.usage()
        if any(usage[k]+v>self.limits[k] for k,v in [('input_tokens',input_tokens),('output_tokens',output_tokens)]): raise RuntimeError('token budget exhausted')
        rid=f'{usage["requests"]}:{usage["executions"]}'
        self.active['executions']+=1
        self.ledger.add('reserve',reservation=rid,label=self.active['label'],case_id=case_id,variant=variant,input_tokens=input_tokens,output_tokens=output_tokens)
        return rid
    def finish(self,rid:str,input_tokens:int,output_tokens:int,**data):
        rows=records(self.ledger.path)
        reservations=[r for r in rows if r['event']=='reserve' and r['reservation']==rid]
        if len(reservations)!=1 or any(r['event']=='usage' and r['reservation']==rid for r in rows): raise ValueError('unknown or already finished reservation')
        reserved=reservations[0]
        if any(type(v) is not int or v<0 or v>reserved[k] for k,v in [('input_tokens',input_tokens),('output_tokens',output_tokens)]): raise ValueError('usage exceeds reservation')
        self.ledger.add('usage',reservation=rid,input_tokens=input_tokens,output_tokens=output_tokens,**data)
    def candidate(self,patch_id:str):
        if self.usage()['candidates']>=self.limits['candidates']: raise RuntimeError('candidate budget exceeded')
        self.ledger.add('candidate',patch_id=patch_id)
