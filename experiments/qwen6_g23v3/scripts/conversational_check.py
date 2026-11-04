#!/usr/bin/env python3
"""Generate with the conversational readout (variants/g23v3_conv) on the practice and selection sets and classify the outputs.

  python -I scripts/conversational_check.py [--labels g23v3_conv] [--sets practice,selection] [--out docs/data/conversational_mode.json]
                                            [--system campaign|plain|none|<text>]   (labels may include "original")

Not a budgeted campaign pass (nothing is written under runs/ or ledger.jsonl); same protocol as the campaign (official chat template with
native <tools>, enable_thinking=False, greedy, 128 new tokens). For every item the recorded ORIGINAL output comes from runs/ (p01, p40)
so that required calls can be checked token-identical. Classification is deliberately mechanical: tool call / literal / prose, whether
prose contains a question mark, and whether any literal fragment (NO_TOOL, NEED_CLARIFICATION, NENE, NONO, _TOOL) leaked into prose."""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import demo_cli
from circuit_items import load_set

LEAK = ('NO_TOOL', 'NEED_CLARIFICATION', 'NENE', 'NONO', '_TOOL', '<tool_call')
SYSTEMS = {'campaign': None,  # the item's own (benchmark) system prompt, which instructs the NO_TOOL / NEED_CLARIFICATION literals
           'plain': 'You are a helpful assistant. Use the available tools when a request needs one; otherwise answer in plain English.',
           'none': ''}       # no system message at all: only the chat template's native <tools> block

def classify(text, toks, rec_original):
    t = text.strip()
    has_call = '<tool_call>' in text
    literal = t.startswith('NO_TOOL') or t.startswith('NEED_CLARIFICATION')
    kind = 'call' if has_call else 'literal' if literal else 'prose' if t else 'empty'
    return {'kind': kind, 'question': '?' in t, 'leak': (kind == 'prose' and any(f in t for f in LEAK)),
            'identical_to_original': rec_original is not None and toks == rec_original['token_ids'], 'n_tokens': len(toks), 'truncated': len(toks) >= demo_cli.MAX_TOKENS}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--labels', default='g23v3_conv'); ap.add_argument('--sets', default='practice,selection')
    ap.add_argument('--out', default=str(ROOT / 'docs/data/conversational_mode.json')); ap.add_argument('--no-lock', action='store_true')
    ap.add_argument('--system', default='campaign', help='campaign (the benchmark prompt), plain, none, or a literal system prompt'); a = ap.parse_args()
    labels = [l for l in a.labels.split(',') if l]
    system_override = SYSTEMS.get(a.system, a.system)
    items = []
    for s in a.sets.split(','): items += load_set(ROOT, s)
    locks = [] if a.no_lock else demo_cli.accelerator_lock()
    m = demo_cli.Model('Qwen/Qwen3.5-0.8B'); t_all = time.perf_counter()
    rows = {it['id']: {'id': it['id'], 'set': it['set'], 'cohort': it['cohort'], 'family': it['family'], 'user': it['user'], 'expected': it['expected'],
                       'system': (it['system'] if system_override is None else system_override),
                       'tools': [t['function']['name'] for t in it['tools']], 'outputs': {'original': {'text': it['outputs']['original']['raw_output'], **classify(it['outputs']['original']['raw_output'], it['outputs']['original']['token_ids'], it['outputs']['original']),
                                                                                              'run': it['outputs']['original']['run']}}} for it in items}
    for it in items:
        if 'g23v3' in it['outputs']: rows[it['id']]['outputs']['g23v3'] = {'text': it['outputs']['g23v3']['raw_output'], **classify(it['outputs']['g23v3']['raw_output'], it['outputs']['g23v3']['token_ids'], it['outputs']['original']), 'run': it['outputs']['g23v3']['run']}
    for label in labels:
        m.set_edit(None if label == 'original' else label)
        for i, it in enumerate(items):
            system = it['system'] if system_override is None else system_override
            ids = m.prompt_ids(system, it['user'], it['tools']); t0 = time.perf_counter(); text, toks = m.generate(ids)
            c = classify(text, toks, it['outputs']['original']); key = label if (system_override is None and label != 'original') else f'{label}@{a.system}'
            rows[it['id']]['outputs'][key] = {'text': text, **c, 'seconds': round(time.perf_counter() - t0, 2)}
            print(f"[{label}] {i + 1}/{len(items)} {it['id']:<36s} {it['cohort']:<19s} {c['kind']:7s} q={int(c['question'])} leak={int(c['leak'])} ident={int(c['identical_to_original'])} | {text[:70]!r}", flush=True)
    m.set_edit(None)
    for f in locks: import fcntl; fcntl.flock(f, fcntl.LOCK_UN); f.close()
    summary = {}
    gen_keys = [l if (system_override is None and l != 'original') else f'{l}@{a.system}' for l in labels]
    for label in dict.fromkeys(['original', 'g23v3'] + gen_keys):
        for coh in ('valid_call', 'no_tool', 'need_clarification', 'ambiguous'):
            rr = [r['outputs'][label] for r in rows.values() if r['cohort'] == coh and label in r['outputs']]
            if not rr: continue
            summary.setdefault(label, {})[coh] = {'n': len(rr), 'call': sum(x['kind'] == 'call' for x in rr), 'literal': sum(x['kind'] == 'literal' for x in rr), 'prose': sum(x['kind'] == 'prose' for x in rr),
                                                   'prose_with_question': sum(x['kind'] == 'prose' and x['question'] for x in rr), 'leak': sum(x['leak'] for x in rr),
                                                   'identical_to_original': sum(x['identical_to_original'] for x in rr), 'truncated': sum(x['truncated'] for x in rr)}
    out = {'meta': {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': round(time.perf_counter() - t_all, 1), 'sets': a.sets.split(','), 'labels': labels,
                    'system': a.system, 'system_text': system_override if system_override is not None else '(each item\'s benchmark system prompt)',
                    'recorded_rows_are_campaign_prompt': True,
                    'protocol': 'official chat template with native <tools>, enable_thinking=False, greedy, 128 new tokens, BF16 MLX; not a budgeted campaign pass',
                    'note': 'original and g23v3 rows are the recorded campaign outputs (runs/); the variant rows were generated by this script'},
           'summary': summary, 'rows': list(rows.values())}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(summary, indent=1)); print('wrote', a.out)

if __name__ == '__main__': main()
