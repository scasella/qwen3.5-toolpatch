"""Item lists and recorded side-by-side outputs for scripts/export_circuit_traces.py (read-only access to data/ and runs/)."""
import json
import numpy as np

SYSTEM_TOOLS = ('Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information '
                'is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.')
SYSTEM_PLAIN = 'Follow the user instruction exactly.'
VARIANT_LABELS = {'original': 'Original', 'v1': 'v1', 'v2': 'v2', 'g23v3': 'g23v3', 'rand13': 'Random control (seed 13)'}
# set -> variant -> (run label, rows key); renamed is split into A/B by data/renamed_{A,B}.ids
RUN_MAP = {
    'practice':   {'original': ('p01_baseline', 'original'), 'g23v3': ('p38_v3_gen_practice', 'edited'), 'v2': ('p19_v2_gen_practice', 'edited'),
                   'v1': ('p07_gen_intended', 'edited'), 'rand13': ('p45_v3_rand13', 'edited')},
    'selection':  {'original': ('p40_v3_sel_both', 'original'), 'g23v3': ('p40_v3_sel_both', 'edited'), 'v2': ('p21_v2_sel_both', 'edited'), 'v1': ('p13_sel_both', 'edited')},
    'renamed_A':  {'original': ('p42_v3_renamedA_both', 'original'), 'g23v3': ('p42_v3_renamedA_both', 'edited'), 'v2': ('p22_v2_renamedA_both', 'edited'), 'v1': ('p14_renamedA_both', 'edited')},
    'renamed_B':  {'original': ('p43_v3_renamedB_both', 'original'), 'g23v3': ('p43_v3_renamedB_both', 'edited'), 'v2': ('p23_v2_renamedB_both', 'edited'), 'v1': ('p15_renamedB_both', 'edited')},
    'final':      {'original': ('p41_v3_final_both', 'original'), 'g23v3': ('p41_v3_final_both', 'edited'), 'v2': ('p24_v2_final_both', 'edited')},
    'prose_qa':   {'original': ('p37_v3_prose_both', 'original'), 'g23v3': ('p37_v3_prose_both', 'edited'), 'v2': ('p31_v2_prose_both', 'edited'), 'v1': ('p32_v1_prose_both', 'edited')},
    'prose_qa2':  {'original': ('p36_v3_prose2_both', 'original'), 'g23v3': ('p36_v3_prose2_both', 'edited')},
}

def silu(z):
    z = np.asarray(z, dtype=np.float64); return z / (1 + np.exp(-z))

_run_cache = {}
def _rows(root, label, key):
    if (label, key) not in _run_cache:
        d = json.loads((root / 'runs' / f'{label}.json').read_text())
        _run_cache[(label, key)] = {r['case_id']: r for r in d['rows'][key]}
    return _run_cache[(label, key)]

def _output(root, set_name, variant, cid):
    if variant not in RUN_MAP[set_name]: return None
    label, key = RUN_MAP[set_name][variant]; r = _rows(root, label, key).get(cid)
    if r is None: return None
    s = r['score']; dec = s.get('decision')
    return {'run': label, 'raw_output': r['raw_output'], 'token_ids': r['token_ids'], 'n_tokens': len(r['token_ids']),
            'decision': dec if dec is not None else 'prose', 'correct': bool(s.get('correct')), 'correct_strict': bool(s.get('correct_strict')),
            'false_call': bool(s.get('false_call')), 'truncated': bool(r.get('truncated', False))}

def load_set(root, set_name):
    """-> list of item dicts (id, set, cohort, family, action, ambiguous, renamed, expected, system, user, tools, outputs)."""
    items = []
    if set_name in ('prose_qa', 'prose_qa2'):
        for line in (root / f'data/{set_name}.jsonl').read_text().splitlines():
            if not line: continue
            r = json.loads(line)
            it = {'id': r['id'], 'set': set_name, 'cohort': 'prose', 'family': r['domain'], 'action': 'prose', 'ambiguous': False, 'renamed': False,
                  'expected': 'fluent prose (128 tokens, no literal, no tool call)', 'system': SYSTEM_PLAIN, 'user': r['user'], 'tools': []}
            it['outputs'] = {v: o for v in VARIANT_LABELS if (o := _output(root, set_name, v, r['id'])) is not None}
            items.append(it)
        return items
    ab = {}
    if set_name == 'renamed':
        for part in ('A', 'B'):
            for cid in (root / f'data/renamed_{part}.ids').read_text().split():
                ab[cid] = part
    for line in (root / f'data/{set_name}.jsonl').read_text().splitlines():
        if not line: continue
        c = json.loads(line); act = c['spec']['action']; amb = bool(c['spec'].get('ambiguous'))
        cohort = 'valid_call' if act == 'call' else 'no_tool' if act == 'answer' else 'ambiguous' if amb else 'need_clarification'
        sname = {'practice': 'practice', 'selection': 'selection', 'final_proxy': 'final'}.get(set_name) or f"renamed_{ab[c['id']]}"
        expected = {'answer': 'NO_TOOL', 'clarify': 'NEED_CLARIFICATION', 'call': f"<tool_call> {c['expected_name']} {json.dumps(c['expected_arguments'])}"}[act]
        it = {'id': c['id'], 'set': sname, 'cohort': cohort, 'family': c['family'], 'action': act, 'ambiguous': amb, 'renamed': set_name == 'renamed',
              'expected': expected, 'system': SYSTEM_TOOLS, 'user': c['user'], 'tools': c['tools']}
        it['outputs'] = {v: o for v in VARIANT_LABELS if (o := _output(root, sname, v, c['id'])) is not None}
        items.append(it)
    return items
