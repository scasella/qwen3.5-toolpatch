#!/usr/bin/env python
"""Does the BFCL serving path reproduce the campaign? Formats six demo cases (practice +
stress suite) with the HF tokenizer exactly as the handler does, sends them to serve.py,
and compares the raw text with the recorded g23v4 campaign runs (experiments/qwen7_g23v4/runs).
Run from the repo root while serve.py serves artifacts/exports/g23v4 on port 1053.
Result on 2026-11-04: 6/6 identical, prompt and output token counts equal."""
import json, pathlib, sys
from transformers import AutoTokenizer
from openai import OpenAI
EXP6, EXP7 = pathlib.Path('experiments/qwen6_g23v3'), pathlib.Path('experiments/qwen7_g23v4')
SYS = ('Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information '
       'is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.')
cases = [(EXP6/'data/practice.jsonl','discovery-611401-0000',EXP7/'runs/g05_practice_v4.json'),
         (EXP6/'data/practice.jsonl','discovery-611401-0226',EXP7/'runs/g05_practice_v4.json'),
         (EXP6/'data/practice.jsonl','discovery-611401-0131',EXP7/'runs/g05_practice_v4.json'),
         (EXP6/'data/practice.jsonl','discovery-611401-0239',EXP7/'runs/g05_practice_v4.json'),
         (EXP7/'data/stress.jsonl','stress-amb-04-discovery-711406-0016-amb-B5',EXP7/'runs/g04_stress_v4.json'),
         (EXP7/'data/stress.jsonl','stress-res-18-selection-711407-0012-res-R1',EXP7/'runs/g04_stress_v4.json')]
tok = AutoTokenizer.from_pretrained('artifacts/exports/g23v4', local_files_only=True)
cl = OpenAI(base_url=f"http://localhost:{sys.argv[1] if len(sys.argv) > 1 else 1053}/v1", api_key='EMPTY')
ok = 0
for path, cid, run in cases:
    case = next(json.loads(l) for l in path.read_text().splitlines() if l and json.loads(l)['id'] == cid)
    rec = next(r for r in json.load(open(run))['rows']['g23v4'] if r['case_id'] == cid)
    txt = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': case['user']}],
                                  tools=case['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    r = cl.completions.create(model='x', prompt=txt, max_tokens=128, temperature=0)
    same = r.choices[0].text == rec['raw_output']; ok += same
    print(f"{cid}: {'IDENTICAL' if same else 'DIFFERENT'} | in {r.usage.prompt_tokens} vs {rec['input_tokens']} | out {r.usage.completion_tokens} vs {rec['output_tokens']}")
print(f"{ok}/{len(cases)} identical to the recorded g23v4 campaign runs")
