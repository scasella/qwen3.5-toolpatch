#!/usr/bin/env python3
"""Generate the prose traces behind Figure 4 of the blog post: one non-tool question answered by the original weights and by the
v1 (g23_t05_a1.0), v2 (g23v2) and g23v3 edits, using demo_cli's resident-model machinery (same protocol as the campaign: official
chat template, enable_thinking=False, greedy, 128 new tokens). Two prompts are traced: the demo question used in the figure and
prose2-07 from data/prose_qa2.jsonl, whose original and g23v3 streams are checked against the recorded run p36.

Not a budgeted campaign pass: reads runs/, data/ and edits/*.vectors.npz, and writes only docs/figures/fig4_trace.json.

  HF_HUB_OFFLINE=1 python -I scripts/make_fig4_trace.py      (needs mlx-lm on Apple Silicon)"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import demo_cli as D

def main():
    locks = D.accelerator_lock(); m = D.Model('Qwen/Qwen3.5-0.8B')
    rec = json.loads((ROOT / 'runs/p36_v3_prose2_both.json').read_text())['rows']
    out = {'system': D.SYSTEM_PLAIN, 'max_tokens': D.MAX_TOKENS, 'decoding': 'greedy', 'prompts': {}}
    for key, user in [('prose2-07', 'Describe the difference between a virus and a bacterium.'),
                      ('demo-case-6', 'What are the main differences between bacteria and viruses?')]:
        ids = m.prompt_ids(D.SYSTEM_PLAIN, user, []); entry = {'user': user, 'variants': {}}
        for name, label in [('original', None), ('v1', 'g23_t05_a1.0'), ('v2', 'g23v2'), ('g23v3', 'g23v3')]:
            m.set_edit(label); text, toks = m.generate(ids)
            entry['variants'][name] = {'edit': label or 'none', 'text': text, 'token_ids': toks, 'tokens': [m.tok.decode([t]) for t in toks]}
            print(key, name, len(toks), repr(text[:100]))
        if key == 'prose2-07':
            for which, name in (('original', 'original'), ('edited', 'g23v3')):
                ref = next(r['token_ids'] for r in rec[which] if r['case_id'] == 'prose2-07')
                entry['variants'][name]['matches_recorded_p36'] = (ref == entry['variants'][name]['token_ids']); print(name, 'matches p36:', entry['variants'][name]['matches_recorded_p36'])
        out['prompts'][key] = entry
    m.set_edit(None)
    (ROOT / 'docs/figures/fig4_trace.json').write_text(json.dumps(out, indent=1)); print('wrote docs/figures/fig4_trace.json')
    for f in locks: f.close()

if __name__ == '__main__': main()
