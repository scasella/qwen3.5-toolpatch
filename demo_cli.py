#!/usr/bin/env python3
"""Side-by-side demo: Base Qwen/Qwen3.5-0.8B vs g23v3 (2 neurons, layer 23) vs g23v4 (4 neurons, layers 22+23) on seven
representative prompts, plus the conversational readout (--mode conversational of apply_patch.py) on the abstention cases.

  python demo_cli.py [--base Qwen/Qwen3.5-0.8B] [--check] [--cases 1,4,5] [--no-conv | --conv-all] [--no-v1] [--no-lock]

Protocol is the campaigns': official chat template with native <tools>, enable_thinking=False, greedy decoding, max 128 new
tokens, the fixed standard system prompt. Cases 1-4 are qwen-6 practice items, 5-6 are held-out Ambiguity Stress Suite items
from qwen-7, 7 is a free-form non-tool prose question (shown with v1, the first candidate that collapsed prose). Every edit is
rebuilt in closed form from patches/*.vectors.npz by apply_patch.build and loaded into the resident model, so one model load
serves all variants. Nothing here is a budgeted campaign pass. --check compares the token streams with the recorded campaign
runs under experiments/. Requires mlx-lm on Apple Silicon; imports nothing from the campaign harnesses."""
import argparse, fcntl, json, os, sys, time
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from apply_patch import build
EXP6 = HERE / 'experiments/qwen6_g23v3'; EXP7 = HERE / 'experiments/qwen7_g23v4'

SYSTEM_TOOLS = ('Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information '
                'is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.')
SYSTEM_PLAIN = 'Follow the user instruction exactly.'
MAX_TOKENS = 128
P6, P7 = EXP6 / 'data/practice.jsonl', EXP7 / 'data/stress.jsonl'
R = {'p01': EXP6 / 'runs/p01_baseline.json', 'p38': EXP6 / 'runs/p38_v3_gen_practice.json', 'g05': EXP7 / 'runs/g05_practice_v4.json',
     'g02': EXP7 / 'runs/g02_stress_amb_base_v3.json', 'g03': EXP7 / 'runs/g03_stress_res_base_v3.json', 'g04': EXP7 / 'runs/g04_stress_v4.json'}
PRACTICE_REF = {'original': ('p01', 'original'), 'g23v3': ('p38', 'edited'), 'g23v4': ('g05', 'g23v4')}
CASES = [  # (title, items file, case id, {variant: (run, rows key)} for --check)
    ('1. Required tool call (must be preserved token-identically)',         P6, 'discovery-611401-0000', PRACTICE_REF),
    ('2. Explanation-only request -> NO_TOOL',                              P6, 'discovery-611401-0226', PRACTICE_REF),
    ('3. Required value explicitly missing -> NEED_CLARIFICATION',          P6, 'discovery-611401-0131', PRACTICE_REF),
    ('4. Unresolved "either X or Y" -> NEED_CLARIFICATION (g23v4 only)',    P6, 'discovery-611401-0239', PRACTICE_REF),
    ('5. Unseen phrasing (Stress Suite B5): "X or Y, I keep going back and forth"', P7, 'stress-amb-04-discovery-711406-0016-amb-B5',
     {'original': ('g02', 'original'), 'g23v3': ('g02', 'g23v3'), 'g23v4': ('g04', 'g23v4')}),
    ('6. Contrastive resolved two-value call: "use X rather than Y" (no false clarification)', P7, 'stress-res-18-selection-711407-0012-res-R1',
     {'original': ('g03', 'original'), 'g23v3': ('g03', 'g23v3'), 'g23v4': ('g04', 'g23v4')}),
    ('7. Free-form prose, no tools (128 tokens): v1 collapse vs intact essays', None, 'What are the main differences between bacteria and viruses?', {}),
]
LABEL = {'original': 'Base', 'g23v3': 'g23v3', 'g23v4': 'g23v4', 'g23v4_conv': 'g23v4 conv', 'g23v3_conv': 'g23v3 conv', 'v1': 'v1'}

def accelerator_lock(wait_s=3600):
    """Shared-machine etiquette: take the same advisory flocks the campaigns use; poll, never kill the holder."""
    files = []
    for name in ('/tmp/qwen-local-accelerator.lock', '/tmp/qwen35-mlx-accelerator.lock'):
        f = open(name, 'a+'); t0 = time.monotonic(); said = False
        while True:
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() - t0 > wait_s: raise RuntimeError(f'accelerator busy: {name}')
                if not said: print(f'[lock] {name} busy; waiting', flush=True); said = True
                time.sleep(1)
        files.append(f)
    return files

def load_case(path, cid):
    for line in path.read_text().splitlines():
        if line and json.loads(line)['id'] == cid: return json.loads(line)
    raise KeyError(cid)

_runs = {}
def recorded(run, key, cid):
    if run not in _runs: _runs[run] = json.loads(R[run].read_text())['rows']
    return next((r['token_ids'] for r in _runs[run].get(key, []) if r['case_id'] == cid), None)

class Model:
    def __init__(self, base):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.utils import hf_repo_to_path
        self.mx = mx; mx.set_default_device(mx.gpu)
        self.path = Path(base) if Path(base).exists() else hf_repo_to_path(base)
        self.model, self.tok = load(str(self.path)); self.model.eval(); self.originals = {}
        self.safetensors = next(self.path.glob('*.safetensors'))
    def set_edit(self, label):
        """Load the BF16 bits of patches/<label> (rebuilt from its vectors) into the live model; None restores the original."""
        from mlx.utils import tree_flatten
        mx = self.mx
        if self.originals: self.model.load_weights(list(self.originals.items()), strict=False); self.originals = {}
        if label is not None:
            arrays = build(self.safetensors, np.load(HERE / 'patches' / f'{label}.vectors.npz'))
            params = dict(tree_flatten(self.model.parameters())); changes = []
            for name, bits in arrays.items():
                mapped = name.replace('model.language_model.', 'language_model.model.')
                self.originals[mapped] = params[mapped]; changes.append((mapped, mx.array(bits).view(mx.bfloat16)))
            self.model.load_weights(changes, strict=False)
        mx.eval(self.model.parameters()); mx.clear_cache()
    def prompt_ids(self, system, user, tools):
        text = self.tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                            tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(text, add_special_tokens=False)
    def generate(self, ids):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        text, toks = '', []
        for r in stream_generate(self.model, self.tok, ids, max_tokens=MAX_TOKENS, sampler=make_sampler(0),
                                 prompt_cache=self.model.make_cache(), prefill_step_size=256):
            text += r.text; toks.append(r.token)
        self.mx.clear_cache(); return text, toks

def show(label, text, width=100):
    lines = text.rstrip('\n').split('\n') or ['']
    print(f'  {label:11s}| ' + lines[0][:width]); [print(' ' * 13 + '| ' + l[:width]) for l in lines[1:12]]
    if len(lines) > 12: print(' ' * 13 + f'| ... ({len(lines) - 12} more lines)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', default='Qwen/Qwen3.5-0.8B')
    ap.add_argument('--check', action='store_true', help='compare token streams with the recorded campaign runs'); ap.add_argument('--cases', default='1,2,3,4,5,6,7')
    ap.add_argument('--no-conv', action='store_true', help='skip the conversational readout (g23v4_conv) on cases 2-5')
    ap.add_argument('--conv-all', action='store_true', help='show g23v4_conv on every tool case'); ap.add_argument('--no-v1', action='store_true')
    ap.add_argument('--no-lock', action='store_true'); a = ap.parse_args()
    locks = [] if a.no_lock else accelerator_lock()
    t0 = time.perf_counter(); m = Model(a.base); print(f'loaded {m.path} in {time.perf_counter() - t0:.1f}s\n')
    want = {int(c) for c in a.cases.split(',')}; checks = []; same_as = {}
    for i, (title, path, cid, refs) in enumerate(CASES, 1):
        if i not in want: continue
        if path is not None:
            case = load_case(path, cid); user, tools, system = case['user'], case['tools'], SYSTEM_TOOLS
            gold = {'answer': 'NO_TOOL', 'clarify': 'NEED_CLARIFICATION', 'call': f"<tool_call> {case['expected_name']} {json.dumps(case['expected_arguments'])}"}[case['spec']['action']]
        else: user, tools, system, gold = cid, [], SYSTEM_PLAIN, 'fluent prose (judged by inspection)'
        print('=' * 112); print(title, f'[{cid}]' if path is not None else ''); print('-' * 112)
        for l in user.split('\n'): print('  user       | ' + l)
        if tools: print('  tools      | ' + '; '.join(f"{t['function']['name']}: {t['function']['description'].replace(' Inert synthetic operation.', '')}" for t in tools))
        print('  expected   | ' + gold); print()
        ids = m.prompt_ids(system, user, tools)
        variants = ['original', 'g23v3', 'g23v4']
        if i == 7 and not a.no_v1: variants = ['original', 'v1', 'g23v3', 'g23v4']
        if path is not None and not a.no_conv and (a.conv_all or i in (2, 3, 4, 5)): variants.append('g23v4_conv')
        outs = {}
        for v in variants:
            m.set_edit(None if v == 'original' else v); t1 = time.perf_counter(); text, toks = m.generate(ids); outs[v] = toks
            note = '' if v == 'original' else ('; token-identical to Base' if toks == outs['original'] else ('; token-identical to g23v3' if toks == outs.get('g23v3') else '; differs from Base'))
            show(LABEL[v], text); print(f'  {"":11s}| ({len(toks)} tokens, {time.perf_counter() - t1:.1f}s{note})')
            if a.check and v in refs:
                run, key = refs[v]; ref = recorded(run, key, cid)
                if ref is not None: ok = ref == toks; checks.append(ok); print(f'  {"":11s}| recorded campaign run {run}: {"token-identical" if ok else "DIFFERENT"}')
        print()
    m.set_edit(None)
    if a.check: print(f'check: {sum(checks)}/{len(checks)} generations token-identical to the recorded campaign runs')
    for f in locks: fcntl.flock(f, fcntl.LOCK_UN); f.close()

if __name__ == '__main__': main()
