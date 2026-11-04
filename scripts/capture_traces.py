#!/usr/bin/env python3
"""GPU half of the circuit-explorer data export (docs/data/circuit_traces.json) and of the out-of-fold audit.

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python; $P -B scripts/capture_traces.py [--out artifacts/traces] [--limit N]

Runs the ORIGINAL Qwen/Qwen3.5-0.8B once per representative item (greedy, 128 new tokens, the campaign protocol) with four
read-only taps on the resident MLX model and stores, per item, as float16 under <out>/<id>.npz:
  x22      layer-22 MLP input (what the stage-1 neuron J0 reads) at every prompt AND generated position
  h23      layer-23 residual input at every prompt position (the sequence the CPU replay of layer 23 consumes)
  x23_gen  layer-23 MLP input (what J1/J2/J3 read) at the decision position and every generated position
  h24_dec  final-RMSNorm input at the decision position (float32)
plus prompt token ids, generated token ids, the generated text and the role positions of the user's clause (first value,
connector, second value, indecision tail, clause-end '.', <|im_end|>, decision) where the campaign metadata defines them.
Derived, non-budgeted capture (like qwen-6's scripts/export_circuit_traces.py): nothing under experiments/ is touched. The
CPU half (scripts/export_circuit_traces.py, scripts/oof_audit.py) replays layer 23 from these files."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import demo_cli as D
EXP6, EXP7 = D.EXP6, D.EXP7

def read_jsonl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def select_items(limit=0):
    """Representative items across all cohorts (ids are stable campaign ids)."""
    items = []
    def tool_item(c, set_name, renamed=False):
        act = c['spec']['action']; amb = bool(c['spec'].get('ambiguous'))
        cohort = 'valid_call' if act == 'call' else 'no_tool' if act == 'answer' else 'ambiguous' if amb else 'need_clarification'
        if set_name == 'stress': cohort = 'stress_ambiguous' if amb else 'stress_resolved'
        expected = {'answer': 'NO_TOOL', 'clarify': 'NEED_CLARIFICATION', 'call': f"<tool_call> {c['expected_name']} {json.dumps(c['expected_arguments'])}"}[act]
        return {'id': c['id'], 'set': set_name, 'cohort': cohort, 'family': c['family'], 'action': act, 'ambiguous': amb, 'renamed': renamed,
                'expected': expected, 'system': D.SYSTEM_TOOLS, 'user': c['user'], 'tools': c['tools']}
    def pick(rows, set_name, n_call, n_answer, n_clarify, must=(), renamed=False):
        out = []; cnt = {'call': 0, 'answer': 0, 'clarify': 0}; lim = {'call': n_call, 'answer': n_answer, 'clarify': n_clarify}
        for c in rows:
            act = c['spec']['action']; amb = bool(c['spec'].get('ambiguous'))
            if amb or c['id'] in must: out.append(tool_item(c, set_name, renamed)); continue
            if cnt[act] < lim[act]: out.append(tool_item(c, set_name, renamed)); cnt[act] += 1
        return out
    items += pick(read_jsonl(EXP6 / 'data/practice.jsonl'), 'practice', 8, 6, 6, must=('discovery-611401-0000', 'discovery-611401-0226', 'discovery-611401-0131'))
    items += pick(read_jsonl(EXP6 / 'data/selection.jsonl'), 'selection', 3, 2, 2)
    ab = {cid: part for part in 'AB' for cid in (EXP6 / f'data/renamed_{part}.ids').read_text().split()}
    ren = read_jsonl(EXP6 / 'data/renamed.jsonl')
    for part in 'AB': items += pick([c for c in ren if ab[c['id']] == part], f'renamed_{part}', 2, 1, 1, renamed=True)
    items += pick(read_jsonl(EXP6 / 'data/final_proxy.jsonl'), 'final', 3, 1, 1)
    items += [tool_item(c, 'stress', renamed=c['id'].endswith('-renamed')) for c in read_jsonl(EXP7 / 'data/stress.jsonl')]
    for s in ('prose_qa', 'prose_qa2'):
        for r in read_jsonl(EXP6 / f'data/{s}.jsonl')[:3]:
            items.append({'id': r['id'], 'set': s, 'cohort': 'prose', 'family': r['domain'], 'action': 'prose', 'ambiguous': False, 'renamed': False,
                          'expected': 'fluent prose (128 tokens, no literal, no tool call)', 'system': D.SYSTEM_PLAIN, 'user': r['user'], 'tools': []})
    return items[:limit] if limit else items

def clause_meta():
    meta = {}
    for p in (EXP7 / 'data/stress.meta.json', EXP7 / 'data/triples_D1.meta.json', EXP7 / 'data/triples_D2.meta.json', EXP7 / 'data/triples_D3.meta.json'):
        for k, v in json.load(open(p)).items(): meta.setdefault(k, v)
    return meta

def tag_roles(hf_tok, prompt, ids, meta):
    """Port of the campaign's fp2.tag_roles: last token overlapping each clause span."""
    enc = hf_tok(prompt, add_special_tokens=False, return_offsets_mapping=True); assert enc['input_ids'] == ids
    off = enc['offset_mapping']
    def last_tok(a, b):
        hits = [i for i, (x, y) in enumerate(off) if x < b and y > a]; return hits[-1] if hits else None
    roles = {}; cs = prompt.index(meta['clause'])
    for role, (a, b) in meta['spans'].items():
        seg = meta['clause'][a:b]
        if role in ('conn', 'tail'):
            s = seg.strip()
            if not s: continue
            a2 = a + seg.index(s); roles[role] = last_tok(cs + a2, cs + a2 + len(s))
        else: roles[role] = last_tok(cs + a, cs + b)
    ce = cs + len(meta['clause']); ie = prompt.index('<|im_end|>', ce); roles['imend'] = last_tok(ie, ie + len('<|im_end|>'))
    roles['dec'] = len(ids) - 1
    return {k: v for k, v in roles.items() if v is not None}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'artifacts/traces')); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-lock', action='store_true'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = select_items(a.limit); meta = clause_meta(); print(f'{len(items)} items', flush=True)
    locks = [] if a.no_lock else D.accelerator_lock()
    t0 = time.perf_counter(); m = D.Model('Qwen/Qwen3.5-0.8B'); print(f'model loaded in {time.perf_counter() - t0:.1f}s', flush=True)
    import mlx.core as mx, mlx.nn as nn
    from types import SimpleNamespace
    class Tap(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner; self._st = SimpleNamespace(rows=[], offset=0)
        def __call__(self, x, *args, **kw):
            st = self._st; v = x[0].astype(mx.float16); mx.eval(v); st.rows.append((st.offset, np.array(v))); st.offset += x.shape[1]
            return self.inner(x, *args, **kw)
        def reset(self): self._st.rows = []; self._st.offset = 0
        def seq(self, lo, hi):  # rows for positions lo..hi-1
            X = np.concatenate([r for _, r in self._st.rows], 0); return X[lo:hi]
    bb = m.model.language_model.model
    taps = {'x22': Tap(bb.layers[22].mlp), 'h23': Tap(bb.layers[23].input_layernorm), 'x23': Tap(bb.layers[23].mlp), 'h24': Tap(bb.norm)}
    bb.layers[22].mlp = taps['x22']; bb.layers[23].input_layernorm = taps['h23']; bb.layers[23].mlp = taps['x23']; bb.norm = taps['h24']
    hf = m.tok._tokenizer
    index = []; tall = time.perf_counter()
    for i, it in enumerate(items):
        text_prompt = m.tok.apply_chat_template([{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': it['user']}], tools=it['tools'] or None,
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = m.tok.encode(text_prompt, add_special_tokens=False); T = len(ids)
        for t in taps.values(): t.reset()
        t1 = time.perf_counter(); text, toks = m.generate(ids); dt = time.perf_counter() - t1; G = len(toks)
        x22 = taps['x22'].seq(0, T + G - 1); h23 = taps['h23'].seq(0, T); x23g = taps['x23'].seq(T - 1, T + G - 1); h24 = taps['h24'].seq(T - 1, T)[0].astype(np.float32)
        assert x22.shape[0] == T + G - 1 and h23.shape[0] == T and x23g.shape[0] == G, (x22.shape, h23.shape, x23g.shape, T, G)
        roles = tag_roles(hf, text_prompt, ids, meta[it['id']]) if it['id'] in meta else {'dec': T - 1}
        np.savez(out / f"{it['id']}.npz", x22=x22, h23=h23, x23_gen=x23g, h24_dec=h24, prompt_ids=np.array(ids, np.int32), gen_ids=np.array(toks, np.int32))
        index.append({**it, 'T': T, 'G': G, 'text': text, 'roles': roles, 'seconds': dt, 'prompt_text_sha': __import__('hashlib').sha256(text_prompt.encode()).hexdigest()[:16]})
        print(f"[{i + 1}/{len(items)}] {it['id']:<48s} {it['set']:10s} T={T:4d} G={G:3d} {dt:4.1f}s | {text[:50]!r}", flush=True)
        (out / 'index.json').write_text(json.dumps({'items': index, 'n': len(index)}, ensure_ascii=False, indent=0))
    bb.layers[22].mlp = taps['x22'].inner; bb.layers[23].input_layernorm = taps['h23'].inner; bb.layers[23].mlp = taps['x23'].inner; bb.norm = taps['h24'].inner
    print(f'done: {len(index)} items in {time.perf_counter() - tall:.0f}s -> {out}', flush=True)
    for f in locks: import fcntl; fcntl.flock(f, fcntl.LOCK_UN); f.close()

if __name__ == '__main__': main()
