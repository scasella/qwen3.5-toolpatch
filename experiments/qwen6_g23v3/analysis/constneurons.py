"""Decision-position 'constant' neurons: active on every decision case (all classes), stable, and silent at the '=' prefix position."""
from __future__ import annotations
import numpy as np
from analysis.lens import load_run, classes, practice_classes

def main():
    tj,ta=load_run('p02_teacher_dec'); pj,pa=load_run('p03_practice_dec')
    A=np.concatenate([ta['mlp_dec'],pa['mlp_dec']],0)   # (124,24,3584)
    Pre=np.concatenate([ta['mlp_pre'],pa['mlp_pre']],0)
    for l in range(8,24):
        X=A[:,l]; m=X.mean(0); s=X.std(0); lo=X.min(0); hi=X.max(0); pre=np.abs(Pre[:,l]).max(0)
        # positive constant: min over cases large relative to prefix max; also negative constant
        score_pos=lo/(pre+0.05); score_neg=-hi/(pre+0.05)
        bp=np.argsort(score_pos)[::-1][:4]; bn=np.argsort(score_neg)[::-1][:3]
        print(f'L{l:2d} pos: '+', '.join(f'{int(i)}: min {lo[i]:5.2f} mean {m[i]:5.2f} sd {s[i]:4.2f} pre {pre[i]:4.2f}' for i in bp)+' | neg: '+', '.join(f'{int(i)}: max {hi[i]:6.2f} mean {m[i]:6.2f} sd {s[i]:4.2f} pre {pre[i]:4.2f}' for i in bn))
if __name__=='__main__': main()
