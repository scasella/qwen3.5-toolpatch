#!/usr/bin/env python3
"""Build the ONNX repository behind the web app's Tab 2 (live WebGPU runner) and verify it.

  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python
  $P -B scripts/export_webgpu_onnx.py [--version g23v4|g23v3] [--src onnx-community/Qwen3.5-0.8B-ONNX | <local dir>]
                                     [--out artifacts/exports/g23v4-onnx] [--variants fp16] [--validate] [--drift]
                                     [--inherit-report <build_report.json of an earlier export>] [--upload <hf repo id>]

What it does (surgical, no re-quantization, nothing re-exported from PyTorch):
  1. starts from onnx-community's Transformers.js-compatible decoder graphs of Qwen3.5-0.8B (fp16 by default; q4f16 / q4 optional);
     a local folder with the same layout (e.g. an earlier export of this script) is accepted as --src and copied with APFS clonefile;
  2. for 4-bit graphs, replaces the MatMulNBits nodes of the edited layers' MLPs (layer 22 and 23 for g23v4) by plain MatMul nodes holding
     the exact checkpoint values, so block quantization cannot smear neurons J0 / J1 / J2 / J3 (fp16 graphs already hold them in fp16);
  3. writes two decoders per variant -- `decoder_model_merged_<v>` (ORIGINAL weights) and `decoder_model_merged_<version>_<v>` (the edit,
     bit-exact BF16 values from patches/<version>.vectors.npz via apply_patch.build): the edited data file is a byte copy of the base one
     with exactly the edited entries overwritten at their offsets (12,281 for g23v4), so the two files differ in nothing else;
  4. writes `patch_manifest.json`: byte offsets of the edited tensors (keys '<layer>.<proj>') plus the edited values, so the web worker can
     turn the downloaded original into the edited model in memory (one download serves both models), with SHA-256 of both data files;
  5. copies the tokenizer / processor / config files and the embed_tokens and vision_encoder sessions;
  6. --drift: runs the ORIGINAL decoder on ORT CPU and compares the layer-23 MLP input at the decision position with the qwen-6 BF16 captures
     (cosine, gate sign); the fp16 backbone keeps the features (cos 0.99), the 4-bit backbones do not (cos 0.89), which is why fp16 is shipped;
  7. --validate: ORT CPU greedy decoding of practice and stress-suite prompts with both decoders, compared with the recorded campaign runs.
Requires: onnx, onnxruntime, huggingface_hub, numpy, transformers (tokenizer only); the base checkpoint in the HF cache."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from apply_patch import read_bits, f32, build, load_patch
EXP6, EXP7 = ROOT / 'experiments/qwen6_g23v3', ROOT / 'experiments/qwen7_g23v4'
HF_NAME = 'model.language_model.layers.{}.mlp.{}.weight'
SIDE_FILES = ['config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json']
SNAPSHOT = Path(os.environ.get('QWEN35_SNAPSHOT', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
NEURONS = {'g23v4': {22: [2525], 23: [3468, 3421, 2331]}, 'g23v3': {23: [3468, 3421]}}
PROJ = ('gate_proj', 'up_proj', 'down_proj')

def sha256_file(p, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(bufsize): h.update(chunk)
    return h.hexdigest()
def clone(src, dst):
    """Copy a (large) file; APFS clonefile on macOS (instant, no extra disk) else a normal copy."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size == Path(src).stat().st_size: return
    if sys.platform == 'darwin' and subprocess.run(['cp', '-c', str(src), str(dst)]).returncode == 0: os.chmod(dst, 0o644); return
    shutil.copyfile(src, dst); os.chmod(dst, 0o644)

def ensure_source(src, variants):
    p = Path(src)
    if p.exists(): return p
    from huggingface_hub import snapshot_download
    pats = ['*.json', '*.jinja'] + [f'onnx/{n}_{v}.onnx*' for v in variants for n in ('decoder_model_merged', 'embed_tokens', 'vision_encoder')] + ['onnx/embed_tokens_q4f16.onnx*', 'onnx/vision_encoder_fp16.onnx*']
    return Path(snapshot_download(src, allow_patterns=pats, local_dir=str(ROOT / 'artifacts/exports/onnx-src' / src.replace('/', '__'))))

def dequant_matmulnbits(q, scales, zp, K, N, block):
    nblk = q.shape[1]; lo = (q & 0x0F).astype(np.int16); hi = (q >> 4).astype(np.int16)
    vals = np.stack([lo, hi], -1).reshape(N, nblk, block)
    if zp is None: z = np.full((N, nblk), 8, np.int16)
    else: zl = (zp & 0x0F).astype(np.int16); zh = (zp >> 4).astype(np.int16); z = np.stack([zl, zh], -1).reshape(N, -1)[:, :nblk]
    w = (vals - z[:, :, None]) * scales.astype(np.float32)[:, :, None]
    return w.reshape(N, nblk * block)[:, :K].T.astype(np.float32)

def onnx_name(l, p): return f'model.layers.{l}.mlp.{p}.MatMul.weight'

def surgery(src_graph, base_w, layers, act_dtype):
    """Replace the edited layers' MLP MatMulNBits nodes by MatMul + dense initializers (no-op check for already-dense graphs)."""
    import onnx
    from onnx import numpy_helper, helper
    m = onnx.load(str(src_graph)); g = m.graph; inits = {t.name: t for t in g.initializer}; report = {}
    for l in layers:
        for proj in PROJ:
            idx, node = next((i, n) for i, n in enumerate(g.node) if n.op_type in ('MatMulNBits', 'MatMul') and f'/layers.{l}/mlp/{proj}/' in n.name)
            W_hf = base_w[(l, proj)]
            if node.op_type == 'MatMul':
                t = inits[node.input[1]]; dense = numpy_helper.to_array(t); assert dense.shape == W_hf.T.shape, (l, proj, dense.shape)
                report[f'{l}.{proj}'] = {'K': dense.shape[0], 'N': dense.shape[1], 'dense_vs_checkpoint_max_abs': float(np.abs(dense.astype(np.float32) - W_hf.T).max()), 'dtype': dense.dtype.name, 'already_dense': True, 'init_name': node.input[1]}
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}; K, N, block = attrs['K'], attrs['N'], attrs['block_size']
            q = numpy_helper.to_array(inits[node.input[1]]); sc = numpy_helper.to_array(inits[node.input[2]]); zp = numpy_helper.to_array(inits[node.input[3]]) if len(node.input) > 3 and node.input[3] else None
            assert W_hf.shape == (N, K); deq = dequant_matmulnbits(q, sc, zp, K, N, block)
            report[f'{l}.{proj}'] = {'K': K, 'N': N, 'block': block, 'dequant_vs_checkpoint_max_abs': float(np.abs(deq - W_hf.T).max()), 'dequant_vs_checkpoint_corr': float(np.corrcoef(deq.ravel(), W_hf.T.ravel())[0, 1])}
            new_name = onnx_name(l, proj); g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W_hf.T.astype(act_dtype)), new_name))
            g.node.remove(node); g.node.insert(idx, helper.make_node('MatMul', [node.input[0], new_name], list(node.output), name=f'/model/layers.{l}/mlp/{proj}/MatMul'))
            used = {x for n in g.node for x in n.input}
            for old in node.input[1:]:
                if old and old not in used: g.initializer.remove(inits[old])
    return m, report

def save_external(m, out_graph):
    import onnx
    out_graph = Path(out_graph); data_name = out_graph.name + '_data'
    for p in (out_graph, out_graph.parent / data_name):
        if p.exists(): p.unlink()
    onnx.save_model(m, str(out_graph), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024, convert_attribute=False)
    os.chmod(out_graph, 0o644); os.chmod(out_graph.parent / data_name, 0o644); return data_name

def locate(out_graph, names):
    import onnx
    m = onnx.load(str(out_graph), load_external_data=False); out = {}
    for t in m.graph.initializer:
        if t.name in names:
            ext = {e.key: e.value for e in t.external_data}
            out[t.name] = {'offset': int(ext.get('offset', 0)), 'length': int(ext['length']), 'dims': list(t.dims), 'dtype': onnx.TensorProto.DataType.Name(t.data_type).lower(), 'location': ext['location']}
    assert set(out) == set(names), (set(names) - set(out)); return out

def fp16_bits(a): return np.asarray(a, np.float32).astype(np.float16).view(np.uint16).tolist()

def patch_copy(base_graph, edited_graph, edit_w, names):
    """Edited decoder = byte copy of the base data file with the edited tensors overwritten at their offsets; graph re-pointed to it."""
    import onnx
    base_graph, edited_graph = Path(base_graph), Path(edited_graph)
    loc = locate(base_graph, list(names.values())); data_in = base_graph.parent / next(iter(loc.values()))['location']; data_out = edited_graph.parent / (edited_graph.name + '_data')
    if data_out.exists(): data_out.unlink()
    clone(data_in, data_out)
    with open(data_out, 'r+b') as f:
        for (l, proj), name in names.items():
            t = loc[name]; dt = {'float16': np.float16, 'float': np.float32}[t['dtype']]
            raw = np.ascontiguousarray(edit_w[(l, proj)].T.astype(dt)).tobytes(); assert len(raw) == t['length'], (name, len(raw), t['length'])
            f.seek(t['offset']); f.write(raw)
    m = onnx.load(str(base_graph), load_external_data=False)
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == 'location': e.value = data_out.name
    onnx.save(m, str(edited_graph)); os.chmod(edited_graph, 0o644); return data_out.name

def ort_sessions(out, variant, decoder_file):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    emb = ort.InferenceSession(str(out / f'onnx/embed_tokens_{variant}.onnx'), so, providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(str(out / 'onnx' / decoder_file), so, providers=['CPUExecutionProvider']); return emb, dec

def feed_for(dec, emb, ids):
    meta = {i.name: i for i in dec.get_inputs()}; T = len(ids); feed = {}
    e = emb.run(None, {'input_ids': np.array([ids], np.int64)})[0]
    for name, i in meta.items():
        dt = np.float16 if 'float16' in i.type else np.float32
        if name == 'inputs_embeds': feed[name] = e.astype(dt)
        elif name == 'attention_mask': feed[name] = np.ones((1, T), np.int64)
        elif name == 'position_ids': feed[name] = np.tile(np.arange(T, dtype=np.int64), (3, 1, 1))
        else: feed[name] = np.zeros([1 if isinstance(d, str) and 'batch' in d else 0 if isinstance(d, str) else d for d in i.shape], dt)
    return feed

def drift(out, variant, embv, n_per=2):
    """Layer-23 MLP input of the ORIGINAL ONNX decoder (ORT CPU) vs the qwen-6 BF16 MLX capture p04c (read-only)."""
    import onnx
    from transformers import AutoTokenizer
    Q6 = Path(os.environ.get('QWEN6', '/Users/scasella/Downloads/qwen-6')); tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    graph = out / f'onnx/decoder_model_merged_{variant}.onnx'; m = onnx.load(str(graph), load_external_data=False); want = '/model/layers.23/post_attention_layernorm/output_0'
    fp16 = variant.endswith('f16') or variant == 'fp16'
    if want not in {o.name for o in m.graph.output}: m.graph.output.append(onnx.helper.make_tensor_value_info(want, onnx.TensorProto.FLOAT16 if fp16 else onnx.TensorProto.FLOAT, None))
    tmp = graph.parent / f'_drift_{variant}.onnx'; onnx.save(m, str(tmp)); emb, dec = ort_sessions(out, embv, tmp.name); outs = [o.name for o in dec.get_outputs()]
    V = np.load(EXP6 / 'edits/g23v3.vectors.npz'); z = np.load(Q6 / 'runs/p04c_practice_allpos.npz'); rows = {r['case_id']: r for r in json.load(open(EXP6 / 'runs/p04c_practice_allpos.json'))['rows']}
    cases = [json.loads(l) for l in (EXP6 / 'data/practice.jsonl').read_text().splitlines() if l.strip()]; pick = []
    for coh in ('answer', 'clarify', 'call'): pick += [c for c in cases if c['spec']['action'] == coh][:n_per]
    SYS = 'Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.'
    res = []
    for c in pick:
        text = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': c['user']}], tools=c['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False); r = dict(zip(outs, dec.run(None, feed_for(dec, emb, ids)))); x = r[want][0, -1].astype(np.float64)
        row = rows[c['id']]; xm = z[f"mlpin_L23_{c['id']}"].astype(np.float64)[row['decision_position']]; gate, gate_m = 2 * float(x @ V['gate_dir']), 2 * float(xm @ V['gate_dir'])
        res.append({'id': c['id'], 'action': c['spec']['action'], 'cos': round(float(x @ xm / np.linalg.norm(x) / np.linalg.norm(xm)), 4), 'gate': round(gate, 2), 'gate_mlx': round(gate_m, 2)}); print('  drift', json.dumps(res[-1]), flush=True)
    tmp.unlink()
    return {'variant': variant, 'embed': embv, 'n': len(res), 'mean_cos': round(float(np.mean([r['cos'] for r in res])), 4), 'min_cos': round(float(min(r['cos'] for r in res)), 4), 'gate_sign_agrees': int(sum((r['gate'] > 0) == (r['gate_mlx'] > 0) for r in res)), 'rows': res}

def validate(out, version, variant, n_items, max_new):
    """ORT CPU greedy decoding (embed_tokens + decoder, original and edited) on practice + stress-suite prompts vs the recorded campaign runs."""
    from transformers import AutoTokenizer
    from export_circuit_traces import recorded_outputs
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    SYS = 'Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.'
    def items(path, set_name):
        out_ = []
        for l in path.read_text().splitlines():
            if not l.strip(): continue
            c = json.loads(l); act = c['spec']['action']; amb = bool(c['spec'].get('ambiguous'))
            out_.append({'id': c['id'], 'set': set_name, 'ambiguous': amb, 'cohort': 'valid_call' if act == 'call' else 'no_tool' if act == 'answer' else 'ambiguous' if amb else 'need_clarification', 'user': c['user'], 'tools': c['tools']})
        return out_
    pr = items(EXP6 / 'data/practice.jsonl', 'practice'); st = items(EXP7 / 'data/stress.jsonl', 'stress'); pick = []
    for coh in ('valid_call', 'no_tool', 'need_clarification', 'ambiguous'): pick += [it for it in pr if it['cohort'] == coh][: max(1, n_items // 4)]
    pick += [it for it in st if it['ambiguous']][:2] + [it for it in st if not it['ambiguous']][:1]
    results = {}
    for tag, fname in (('original', f'decoder_model_merged_{variant}.onnx'), (version, f'decoder_model_merged_{version}_{variant}.onnx')):
        emb, dec = ort_sessions(out, variant, fname); outs = [o.name for o in dec.get_outputs()]; meta = {i.name: i for i in dec.get_inputs()}
        def present_name(past):
            for cand in (past.replace('past_key_values.', 'present.'), past.replace('past_', 'present_'), past.replace('past_key_values.', 'present_key_values.')):
                if cand in outs: return cand
            raise KeyError(past)
        state_map = {n: present_name(n) for n in meta if n.startswith('past')}; rows = []
        for it in pick:
            text = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': it['user']}], tools=it['tools'] or None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False); T = len(ids); feed = feed_for(dec, emb, ids); gen = []; t0 = time.perf_counter(); pos = T
            for _ in range(max_new):
                res = dict(zip(outs, dec.run(None, feed))); nxt = int(res['logits'][0, -1].astype(np.float32).argmax()); gen.append(nxt)
                if nxt in (248046, 248044): break
                for name, pres in state_map.items(): feed[name] = res[pres]
                feed['inputs_embeds'] = emb.run(None, {'input_ids': np.array([[nxt]], np.int64)})[0].astype(feed['inputs_embeds'].dtype)
                feed['attention_mask'] = np.ones((1, pos + 1), np.int64); feed['position_ids'] = np.full((3, 1, 1), pos, np.int64); pos += 1
            rec = recorded_outputs(it).get(tag); k = min(len(gen), len(rec['token_ids'])) if rec else 0; same = bool(rec and gen[:k] == rec['token_ids'][:k])
            rows.append({'id': it['id'], 'cohort': it['cohort'], 'generated': tok.decode(gen), 'recorded': rec['raw_output'][:80] if rec else None, 'prefix_match': same, 'n': len(gen), 'seconds': round(time.perf_counter() - t0, 1)})
            print(f"  [{tag}] {it['id'][-36:]:<36s} {it['cohort']:<18s} match={same} {len(gen):3d} tok {rows[-1]['seconds']:5.1f}s | {tok.decode(gen)[:70]!r}", flush=True)
        results[tag] = rows
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='g23v4', choices=['g23v4', 'g23v3']); ap.add_argument('--src', default='onnx-community/Qwen3.5-0.8B-ONNX'); ap.add_argument('--out', default=None)
    ap.add_argument('--variants', default='fp16'); ap.add_argument('--browser-variant', default='fp16'); ap.add_argument('--embed-dtype', default='q4f16')
    ap.add_argument('--drift', action='store_true'); ap.add_argument('--drift-items', type=int, default=2); ap.add_argument('--inherit-report', default=None, help='earlier build_report.json whose drift/validation sections are carried over (same base decoder)')
    ap.add_argument('--validate', action='store_true'); ap.add_argument('--validate-items', type=int, default=8); ap.add_argument('--validate-tokens', type=int, default=24)
    ap.add_argument('--upload', default=None); ap.add_argument('--skip-build', action='store_true'); a = ap.parse_args()
    version = a.version; layers = sorted(NEURONS[version]); variants = [v for v in a.variants.split(',') if v]; out = Path(a.out or ROOT / f'artifacts/exports/{version}-onnx'); (out / 'onnx').mkdir(parents=True, exist_ok=True)
    report = {'source': a.src, 'version': version, 'variants': {}, 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    names = {(l, p): onnx_name(l, p) for l in layers for p in PROJ}
    if not a.skip_build:
        src = ensure_source(a.src, variants); print('source', src, flush=True); st = next(SNAPSHOT.glob('*.safetensors'))
        base_w = {(l, p): f32(read_bits(st, HF_NAME.format(l, p))) for l in layers for p in PROJ}
        V, spec = load_patch(version, 'campaign'); edited_bits = build(st, V); edit_w = {(l, p): f32(edited_bits[HF_NAME.format(l, p)]) for l in layers for p in PROJ}
        changed = sum(int(np.count_nonzero(base_w[k] != edit_w[k])) for k in base_w); assert changed == spec['total_changed_entries'], (changed, spec['total_changed_entries'])
        neurons = [f'L{l}:{j}' for l in layers for j in NEURONS[version][l]]
        manifest = {'base_model': 'onnx-community/Qwen3.5-0.8B-ONNX', 'edit': version, 'neurons': neurons, 'layers': layers, 'changed_entries': changed, 'browser_variant': a.browser_variant, 'embed_dtype': a.embed_dtype, 'variants': {}}
        for v in variants:
            act = np.float16 if v.endswith('f16') or v == 'fp16' else np.float32; t0 = time.perf_counter()
            base_graph = out / f'onnx/decoder_model_merged_{v}.onnx'; edited_graph = out / f'onnx/decoder_model_merged_{version}_{v}.onnx'
            m, rep = surgery(src / f'onnx/decoder_model_merged_{v}.onnx', base_w, layers, act)
            if all(r.get('already_dense') for r in rep.values()):
                del m
                for f in src.glob(f'onnx/decoder_model_merged_{v}.onnx*'): clone(f, out / 'onnx' / f.name)
                nm = {k: r['init_name'] for k, r in rep.items()}; names = {(l, p): nm[f'{l}.{p}'] for (l, p) in names}
                data_base = next(iter(locate(base_graph, list(names.values())).values()))['location']
            else: data_base = save_external(m, base_graph); del m
            data_edit = patch_copy(base_graph, edited_graph, edit_w, names)
            loc = locate(base_graph, list(names.values())); loc_e = locate(edited_graph, list(names.values()))
            for n in names.values(): assert loc[n]['offset'] == loc_e[n]['offset'] and loc[n]['length'] == loc_e[n]['length'], n
            tensors = {f'{l}.{p}': {k: loc[names[(l, p)]][k] for k in ('offset', 'length', 'dims', 'dtype')} for (l, p) in names}
            conv = fp16_bits if act is np.float16 else (lambda x: np.asarray(x, np.float32).tolist())
            patch = {f'{l}.{p}': {str(J): conv(edit_w[(l, p)][J] if p != 'down_proj' else edit_w[(l, p)][:, J]) for J in NEURONS[version][l]} for (l, p) in names}
            sb, se = sha256_file(out / 'onnx' / data_base), sha256_file(out / 'onnx' / data_edit)
            A = np.memmap(out / 'onnx' / data_base, dtype=np.uint8, mode='r'); B = np.memmap(out / 'onnx' / data_edit, dtype=np.uint8, mode='r'); assert A.shape == B.shape
            dif = np.flatnonzero(A != B); inside = sum(int(((dif >= t['offset']) & (dif < t['offset'] + t['length'])).sum()) for t in tensors.values()); assert inside == len(dif), (inside, len(dif)); del A, B
            manifest['variants'][v] = {'graph': f'onnx/decoder_model_merged_{v}.onnx', 'data_file': f'onnx/{data_base}', 'edited_graph': f'onnx/decoder_model_merged_{version}_{v}.onnx', 'edited_data_file': f'onnx/{data_edit}',
                                       'sha256_base': sb, 'sha256_edited': se, 'tensors': tensors, 'patch': patch, 'activation_dtype': 'float16' if act is np.float16 else 'float32', 'differing_bytes': int(len(dif))}
            report['variants'][v] = {'layout_check_base': rep, 'seconds': round(time.perf_counter() - t0, 1), 'bytes_base': (out / 'onnx' / data_base).stat().st_size, 'bytes_edited': (out / 'onnx' / data_edit).stat().st_size, 'differing_bytes': int(len(dif))}
            print(f'[{v}] built in {time.perf_counter() - t0:.0f}s; {len(dif)} differing bytes inside the {len(tensors)} edited tensors; layout check {json.dumps(rep)[:300]}', flush=True)
            for n in ('embed_tokens', 'vision_encoder'):
                for f in list(src.glob(f'onnx/{n}_{v}.onnx*')) + list(src.glob(f'onnx/{n}_{a.embed_dtype}.onnx*')): clone(f, out / 'onnx' / f.name)
        bv = manifest['variants'][manifest['browser_variant']]; manifest.update({'data_file': bv['data_file'], 'sha256_base': bv['sha256_base'], 'sha256_edited': bv['sha256_edited']})
        (out / 'patch_manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
        for f in SIDE_FILES:
            if (src / f).exists(): shutil.copy2(src / f, out / f)
        cfg = json.loads((out / 'config.json').read_text()); tj = cfg.setdefault('transformers.js_config', {}); ued = tj.setdefault('use_external_data_format', {})
        for v in variants: ued[f'decoder_model_merged_{version}_{v}.onnx'] = 1
        ued[f'decoder_model_merged_{version}'] = 1; (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        (out / 'README.md').write_text(model_card(version, manifest, report, a.inherit_report)); print('wrote', out, flush=True)
    if a.inherit_report and Path(a.inherit_report).exists():
        old = json.loads(Path(a.inherit_report).read_text())
        for k in ('drift', 'validation_g23v3'):
            if k in old: report[f'inherited_{k}'] = old[k]
        if 'validation' in old: report['inherited_validation_g23v3'] = old['validation']
        report['inherited_from'] = a.inherit_report
    if a.drift: report['drift'] = {v: drift(out, v, a.embed_dtype if v != 'q4' else 'q4', a.drift_items) for v in variants}
    if a.validate:
        v = variants[0]
        try:
            report['validation'] = validate(out, version, v, a.validate_items, a.validate_tokens)
            print('validation prefix matches:', {t: f"{sum(r['prefix_match'] for r in rows)}/{len(rows)}" for t, rows in report['validation'].items()}, flush=True)
        except Exception as e: report['validation_error'] = f'{type(e).__name__}: {e}'; print('validation failed:', e, flush=True)
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(a.upload, repo_type='model', exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.upload, repo_type='model', commit_message=f'ONNX fp16 export of the {version} edit for Transformers.js / WebGPU')
        print('uploaded to', a.upload)

def model_card(version, man, rep, inherit):
    bv = man['variants'].get(man.get('browser_variant', 'fp16')) or next(iter(man['variants'].values()))
    dr = {}
    if inherit and Path(inherit).exists(): dr = json.loads(Path(inherit).read_text()).get('drift', {})
    def drow(v, label):
        d = dr.get(v); return f"| {label} | not measured | |" if not d else "| %s | %.3f (min %.3f) | %d/%d |" % (label, d['mean_cos'], d['min_cos'], d['gate_sign_agrees'], d['n'])
    n_entries = man['changed_entries']; neurons = ', '.join(man['neurons'])
    return f"""---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-0.8B
- scasella91/Qwen3.5-0.8B-toolpatch-{version}
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, qwen3_5, webgpu, transformers.js, tool-calling, weight-editing]
---
# Qwen3.5-0.8B-toolpatch-{version} (ONNX fp16, for Transformers.js / WebGPU)

ONNX export of [scasella91/Qwen3.5-0.8B-toolpatch-{version}](https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-{version}): `Qwen/Qwen3.5-0.8B` with the
closed-form {version} edit ({n_entries:,} BF16 weights in {len(man['layers'])} decoder layer{'s' if len(man['layers']) > 1 else ''}; neurons {neurons}) that makes the model abstain
(`NO_TOOL`), ask (`NEED_CLARIFICATION`) and, for {version}, detect unresolved two-value requests instead of emitting `<tool_call>`. Story, figures and controls:
[blog post](https://scasella.github.io/qwen3.5-toolpatch/blog.html); interactive comparison (this repository powers its *Run live in browser* tab):
https://scasella.github.io/qwen3.5-toolpatch/ ; code: https://github.com/scasella/qwen3.5-toolpatch .

Built from [onnx-community/Qwen3.5-0.8B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) by `scripts/export_webgpu_onnx.py`, without re-exporting
or re-quantizing anything. Two fp16 decoders:

| file | weights |
|---|---|
| `onnx/decoder_model_merged_fp16.onnx` (+`_data`, 1.5 GB) | **original** Qwen3.5-0.8B -- byte-identical to onnx-community's fp16 export |
| `onnx/decoder_model_merged_{version}_fp16.onnx` (+`_data`) | **{version}** edit -- a byte copy of the file above with exactly the {n_entries:,} edited entries overwritten |

`patch_manifest.json` lists the byte offsets of the edited MLP tensors (keys `<layer>.<proj>`) and the edited values, so a client can download the original once
and produce {version} in memory; the web app does this and checks the SHA-256 of the result (base `{bv['sha256_base'][:16]}…`, edited `{bv['sha256_edited'][:16]}…`).
`embed_tokens_*` (q4f16 and fp16) and `vision_encoder_fp16` are copied unchanged; tokenizer, chat template and configs are the base model's.

## Why fp16 and not a 4-bit build

The edit is a gated readout of directions in the residual stream that were fitted on the BF16 model's activations. `export_webgpu_onnx.py --drift` runs the
ORIGINAL decoder of each precision on ORT (CPU) and compares the layer-23 MLP input at the decision position with the campaign's BF16 captures:

| backbone | cosine to the BF16 input, mean (min) | abstain-gate sign agrees |
|---|---|---|
{drow('fp16', 'fp16 decoder, q4f16 embeddings (shipped)')}
{drow('q4f16', 'q4f16 decoder (edited MLPs in fp16)')}
{drow('q4', 'q4 decoder (edited MLPs in fp32)')}

With a 4-bit backbone the input drifts far enough that the gates close and the edited neurons never fire -- the model keeps calling tools. fp16 keeps the
features, so that is what is shipped. (Measured for the g23v3 export; the base decoder is the same file here.)

## Use with Transformers.js (WebGPU)

```js
import {{ AutoTokenizer, Qwen3_5ForCausalLM, TextStreamer }} from "@huggingface/transformers";
const id = "scasella91/Qwen3.5-0.8B-toolpatch-{version}-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(id);
const model = await Qwen3_5ForCausalLM.from_pretrained(id, {{ device: "webgpu", dtype: {{ embed_tokens: "q4f16", decoder_model_merged: "fp16" }} }});  // ORIGINAL weights
const text = tokenizer.apply_chat_template([{{ role: "system", content: "Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text." }}, {{ role: "user", content: "Explain what the archive tool does; do not run it." }}], {{ tools, add_generation_prompt: true, tokenize: false, enable_thinking: false }});
const out = await model.generate({{ ...tokenizer(text, {{ add_special_tokens: false }}), max_new_tokens: 128, do_sample: false, streamer: new TextStreamer(tokenizer, {{ skip_prompt: true }}) }});
```
To run the **edited** decoder with the stock library, load a copy of the folder in which `decoder_model_merged_{version}_fp16.onnx(_data)` is renamed to
`decoder_model_merged_fp16.onnx(_data)`, or patch the downloaded data buffer with `patch_manifest.json` as `docs/worker.js` in the GitHub repository does.

## Caveats
The campaigns' results were measured on the BF16 MLX weights with greedy decoding; fp16 ONNX on WebGPU reproduces the mechanism but small numerical differences
can change individual outputs. Evaluated only under the campaigns' synthetic protocol; see the main model card for limitations.
"""

if __name__ == '__main__': main()
