#!/usr/bin/env python3
"""Side-by-side demo: original Qwen/Qwen3.5-0.8B vs the g23v3 two-neuron layer-23 edit on six representative prompts.

  python demo_cli.py [--base Qwen/Qwen3.5-0.8B] [--check] [--no-v1] [--cases 1,2,6]

Protocol is the campaign's: official chat template with native <tools>, enable_thinking=False, greedy decoding, max 128 new
tokens. Cases 1-5 are items from data/practice.jsonl / data/renamed.jsonl (the paper's sets); case 6 is a free-form non-tool prose
question, shown for original / v1 (first candidate, which corrupted prose) / g23v3. Edits are built in closed form from the 38 KB
vector files with apply_g23v3.build and loaded into the resident model, so one model load serves all variants. Not a budgeted
campaign pass: nothing here writes to ledger.jsonl or runs/. --check compares token streams with the recorded campaign runs.
Requires mlx-lm on Apple Silicon. Imports nothing from runtime/ or the campaign harness."""
import argparse, fcntl, json, os, sys, time
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from apply_g23v3 import build

SYSTEM_TOOLS = ('Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information '
                'is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.')
SYSTEM_PLAIN = 'Follow the user instruction exactly.'
MAX_TOKENS = 128
CASES = [  # (title, set file, case id, recorded runs for --check: (original run, edited run))
    ('1. Required call (must be preserved)',            'practice', 'discovery-611401-0000',                 ('p01_baseline', 'p38_v3_gen_practice')),
    ('2. Explanation-only request -> NO_TOOL',          'practice', 'discovery-611401-0226',                 ('p01_baseline', 'p38_v3_gen_practice')),
    ('3. Explicit missing value -> NEED_CLARIFICATION', 'practice', 'discovery-611401-0131',                 ('p01_baseline', 'p38_v3_gen_practice')),
    ('4. Renamed tools, opaque identifiers',            'renamed',  'discovery-611401-0083-renamed-611403',  ('p42_v3_renamedA_both', 'p42_v3_renamedA_both')),
    ('5. Ambiguous "either X or Y" (known limit)',      'practice', 'discovery-611401-0239',                 ('p01_baseline', 'p38_v3_gen_practice')),
    ('6. Free-form prose, no tools (128 tokens)',       None,       'What are the main differences between bacteria and viruses?', None),
]

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

def load_case(set_name, cid):
    for line in (HERE / f'data/{set_name}.jsonl').read_text().splitlines():
        if line and json.loads(line)['id'] == cid: return json.loads(line)
    raise KeyError(cid)

def recorded(run, which, cid):
    rows = json.loads((HERE / f'runs/{run}.json').read_text())['rows'].get(which, [])
    return next((r['token_ids'] for r in rows if r['case_id'] == cid), None)

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
        """Load the BF16 bits of edits/<label> (rebuilt from its vectors) into the live model; None restores the original."""
        from mlx.utils import tree_flatten
        mx = self.mx
        if self.originals: self.model.load_weights(list(self.originals.items()), strict=False); self.originals = {}
        if label is not None:
            arrays = build(self.safetensors, HERE / f'edits/{label}.vectors.npz', HERE / f'edits/{label}.spec.json')
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
    print(f'  {label:9s}| ' + lines[0][:width]); [print(' ' * 11 + '| ' + l[:width]) for l in lines[1:12]]
    if len(lines) > 12: print(' ' * 11 + f'| ... ({len(lines) - 12} more lines)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', default='Qwen/Qwen3.5-0.8B'); ap.add_argument('--edit', default='g23v3')
    ap.add_argument('--v1', default='g23_t05_a1.0', help='first candidate shown on case 6 (--no-v1 to skip)'); ap.add_argument('--no-v1', action='store_true')
    ap.add_argument('--check', action='store_true', help='compare token streams with the recorded campaign runs'); ap.add_argument('--cases', default='1,2,3,4,5,6')
    ap.add_argument('--no-lock', action='store_true'); a = ap.parse_args()
    locks = [] if a.no_lock else accelerator_lock()
    t0 = time.perf_counter(); m = Model(a.base); print(f'loaded {m.path} in {time.perf_counter() - t0:.1f}s\n')
    want = {int(c) for c in a.cases.split(',')}; checks = []
    for i, (title, set_name, cid, runs) in enumerate(CASES, 1):
        if i not in want: continue
        if set_name:
            case = load_case(set_name, cid); user, tools = case['user'], case['tools']; system = SYSTEM_TOOLS
            gold = {'answer': 'NO_TOOL', 'clarify': 'NEED_CLARIFICATION', 'call': f"<tool_call> {case['expected_name']} {case['expected_arguments']}"}[case['spec']['action']]
        else: user, tools, system, gold = cid, [], SYSTEM_PLAIN, 'fluent prose (judged by inspection)'
        print('=' * 110); print(title, f'[{cid}]' if set_name else ''); print('-' * 110)
        for l in user.split('\n'): print('  user    | ' + l)
        if tools: print('  tools   | ' + '; '.join(f"{t['function']['name']}: {t['function']['description'].replace(' Inert synthetic operation.', '')}" for t in tools))
        print('  expected| ' + gold); print()
        ids = m.prompt_ids(system, user, tools)
        variants = [('original', None), ('g23v3', a.edit)] if (i != 6 or a.no_v1) else [('original', None), ('v1', a.v1), ('g23v3', a.edit)]
        outs = {}
        for name, label in variants:
            m.set_edit(label); t1 = time.perf_counter(); text, toks = m.generate(ids); outs[name] = toks
            same = '' if name == 'original' else ('; token-identical to original' if toks == outs['original'] else '; differs from original')
            show(name, text); print(f'  {"":9s}| ({len(toks)} tokens, {time.perf_counter() - t1:.1f}s{same})')
            if a.check and runs and label in (None, 'g23v3'):
                ref = recorded(runs[0 if label is None else 1], 'original' if label is None else 'edited', cid)
                if ref is not None: ok = ref == toks; checks.append(ok); print(f'  {"":9s}| recorded campaign run: {"token-identical" if ok else "DIFFERENT"}')
        print()
    m.set_edit(None)
    if a.check: print(f'check: {sum(checks)}/{len(checks)} generations token-identical to the recorded campaign runs')
    for f in locks: fcntl.flock(f, fcntl.LOCK_UN); f.close()

if __name__ == '__main__': main()
