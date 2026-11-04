#!/usr/bin/env python3
"""Build the ONNX repository behind the web app's Mode 2 (live WebGPU runner) and verify it.

  python -I scripts/export_webgpu_onnx.py --src <dir or repo id of onnx-community/Qwen3.5-0.8B-ONNX> --out artifacts/exports/g23v3-onnx
                                          [--variants q4f16,q4] [--validate] [--upload scasella91/Qwen3.5-0.8B-toolpatch-g23v3-ONNX]

What it does (surgical, no re-quantization, nothing re-exported from PyTorch):
  1. starts from onnx-community's Transformers.js-compatible decoder graphs of Qwen3.5-0.8B (fp16 by default; q4f16 / q4 optional);
  2. for 4-bit graphs, replaces the three MatMulNBits nodes of the layer-23 MLP (gate_proj, up_proj, down_proj) by plain MatMul nodes
     holding the exact checkpoint values, so block quantization cannot smear neurons 3468 / 3421 (fp16 graphs already hold them in fp16);
     every other tensor keeps onnx-community's bytes;
  3. writes two decoders per variant -- `decoder_model_merged_<v>` (ORIGINAL weights) and `decoder_model_merged_g23v3_<v>` (the released
     g23v3 edit, bit-exact BF16 values of edits/g23v3): the edited data file is a byte-copy of the base one with exactly the 6,142
     edited entries overwritten at their offsets, so the two files differ in nothing else;
  4. writes `patch_manifest.json`: byte offsets of the three tensors plus the edited values, so the web worker can turn the downloaded
     original into g23v3 in memory (one download serves both models), with SHA-256 of both data files;
  5. copies the tokenizer / processor / config files and the embed_tokens and vision_encoder sessions;
  6. --drift: runs the ORIGINAL decoder of a variant on ORT CPU and compares the layer-23 MLP input at the decision position with the
     campaign's BF16 captures (cosine, s.x, gate sign, would-be activation of neuron 3468). This is how the shipped precision was chosen:
     with a 4-bit backbone the layer-23 input drifts to cos ~0.89 and the abstain gate closes on explanation prompts, i.e. the edit
     does not transfer to q4 / q4f16; fp16 keeps it (numbers in build_report.json and the model card);
  7. --validate: ORT CPU greedy-decodes a few practice prompts with both decoders of a variant and compares the token streams with the
     recorded campaign runs (runs/p01_baseline.json and runs/p38_v3_gen_practice.json); fp16 needs an ORT build with fp16 CPU kernels.
Requires: onnx, onnxruntime, huggingface_hub (pip), numpy, transformers (tokenizer only); the base checkpoint in the HF cache.
"""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from apply_g23v3 import read_bits, f32, build

J1, J2 = 3468, 3421; L = 23
HF_NAME = 'model.language_model.layers.23.mlp.{}.weight'
SIDE_FILES = ['config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json']
SNAPSHOT = Path(os.environ.get('QWEN35_SNAPSHOT', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))

def sha256_file(p, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(bufsize): h.update(chunk)
    return h.hexdigest()

def ensure_source(src, variants):
    p = Path(src)
    if p.exists(): return p
    from huggingface_hub import snapshot_download
    pats = [f'*.json', '*.jinja'] + [f'onnx/{n}_{v}.onnx*' for v in variants for n in ('decoder_model_merged', 'embed_tokens', 'vision_encoder')]
    return Path(snapshot_download(src, allow_patterns=pats, local_dir=str(ROOT / 'artifacts/exports/onnx-src' / src.replace('/', '__'))))

def dequant_matmulnbits(q, scales, zp, K, N, block):
    """MatMulNBits (bits=4) -> dense (K, N) float; q: (N, nblk, blob) uint8, scales: (N, nblk), zp: (N, ceil(nblk/2)) uint8 or None."""
    nblk = q.shape[1]
    lo = (q & 0x0F).astype(np.int16); hi = (q >> 4).astype(np.int16)
    vals = np.stack([lo, hi], -1).reshape(N, nblk, block * 1)                       # pairs are (low nibble, high nibble) consecutive along K
    if zp is None: z = np.full((N, nblk), 8, np.int16)
    else:
        zl = (zp & 0x0F).astype(np.int16); zh = (zp >> 4).astype(np.int16); z = np.stack([zl, zh], -1).reshape(N, -1)[:, :nblk]
    w = (vals - z[:, :, None]) * scales.astype(np.float32)[:, :, None]            # (N, nblk, block)
    return w.reshape(N, nblk * block)[:, :K].T.astype(np.float32)                    # (K, N)

def surgery(src_graph, weights, act_dtype):
    """Replace the three layer-23 MLP MatMulNBits nodes by MatMul + dense initializers. weights: {'gate_proj': (3584,1024) f32 (out,in), ...}."""
    import onnx
    from onnx import numpy_helper, helper
    m = onnx.load(str(src_graph))                                # loads the external data next to the graph
    g = m.graph; inits = {t.name: t for t in g.initializer}; report = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        idx, node = next((i, n) for i, n in enumerate(g.node) if n.op_type in ('MatMulNBits', 'MatMul') and f'/layers.{L}/mlp/{proj}/' in n.name)
        if node.op_type == 'MatMul':  # already dense (fp16 / fp32 export): nothing to replace, only verify the layout
            t = inits[node.input[1]]; dense = numpy_helper.to_array(t); W_hf = weights[proj]
            assert dense.shape == W_hf.T.shape, (proj, dense.shape)
            report[proj] = {'K': dense.shape[0], 'N': dense.shape[1], 'dense_vs_checkpoint_max_abs': float(np.abs(dense.astype(np.float32) - W_hf.T).max()), 'dtype': dense.dtype.name, 'already_dense': True}
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        K, N, block = attrs['K'], attrs['N'], attrs['block_size']
        q = numpy_helper.to_array(inits[node.input[1]]); sc = numpy_helper.to_array(inits[node.input[2]])
        zp = numpy_helper.to_array(inits[node.input[3]]) if len(node.input) > 3 and node.input[3] else None
        W_hf = weights[proj]                                      # (out, in)
        assert W_hf.shape == (N, K), (proj, W_hf.shape, K, N)
        # layout check against the dequantized 4-bit weights: the ONNX tensor is W^T (K, N)
        deq = dequant_matmulnbits(q, sc, zp, K, N, block)
        err = float(np.abs(deq - W_hf.T).max()); corr = float(np.corrcoef(deq.ravel(), W_hf.T.ravel())[0, 1])
        report[proj] = {'K': K, 'N': N, 'block': block, 'dequant_vs_checkpoint_max_abs': err, 'dequant_vs_checkpoint_corr': corr, 'scale_dtype': sc.dtype.name}
        new_name = f'model.layers.{L}.mlp.{proj}.MatMul.weight'
        dense = np.ascontiguousarray(W_hf.T.astype(act_dtype))
        g.initializer.append(numpy_helper.from_array(dense, new_name))
        new_node = helper.make_node('MatMul', [node.input[0], new_name], list(node.output), name=f'/model/layers.{L}/mlp/{proj}/MatMul')
        g.node.remove(node); g.node.insert(idx, new_node)
        used = {x for n in g.node for x in n.input}
        for old in node.input[1:]:
            if old and old not in used: g.initializer.remove(inits[old])
    return m, report

def save_external(m, out_graph):
    import onnx
    out_graph = Path(out_graph); data_name = out_graph.name + '_data'
    if out_graph.exists(): out_graph.unlink()
    if (out_graph.parent / data_name).exists(): (out_graph.parent / data_name).unlink()
    onnx.save_model(m, str(out_graph), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024, convert_attribute=False)
    os.chmod(out_graph, 0o644); os.chmod(out_graph.parent / data_name, 0o644)
    # onnx.checker does not know ORT's fused contrib ops (SimplifiedLayerNormalization, MatMulNBits ...), so only a structural re-read is possible here;
    # ORT itself validates the graph in --validate and in the browser
    chk = onnx.load(str(out_graph), load_external_data=False)
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        n = next(n for n in chk.graph.node if n.name == f'/model/layers.{L}/mlp/{proj}/MatMul'); assert n.op_type == 'MatMul'
        assert any(t.name == n.input[1] for t in chk.graph.initializer), n.input[1]
    assert not any(n.op_type == 'MatMulNBits' and f'/layers.{L}/mlp/' in n.name for n in chk.graph.node)
    return data_name

def locate(out_graph, names):
    import onnx
    m = onnx.load(str(out_graph), load_external_data=False)
    out = {}
    for t in m.graph.initializer:
        if t.name in names:
            ext = {e.key: e.value for e in t.external_data}
            out[t.name] = {'offset': int(ext.get('offset', 0)), 'length': int(ext['length']), 'dims': list(t.dims), 'dtype': onnx.TensorProto.DataType.Name(t.data_type).lower(), 'location': ext['location']}
    assert set(out) == set(names), (set(names) - set(out))
    return out

def fp16_bits(a): return np.asarray(a, np.float32).astype(np.float16).view(np.uint16).tolist()

def tensor_names(): return [f'model.layers.{L}.mlp.{p}.MatMul.weight' for p in ('gate_proj', 'up_proj', 'down_proj')]

def patch_copy(base_graph, edited_graph, edit_w):
    """Write decoder_model_merged_g23v3_<v>.onnx(+_data) next to the base files: same bytes except the three layer-23 MLP tensors."""
    import onnx
    base_graph, edited_graph = Path(base_graph), Path(edited_graph)
    loc = locate(base_graph, tensor_names()); data_in = base_graph.parent / next(iter(loc.values()))['location']; data_out = edited_graph.parent / (edited_graph.name + '_data')
    shutil.copyfile(data_in, data_out); os.chmod(data_out, 0o644)
    with open(data_out, 'r+b') as f:
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            t = loc[f'model.layers.{L}.mlp.{proj}.MatMul.weight']; dt = {'float16': np.float16, 'float': np.float32}[t['dtype']]
            raw = np.ascontiguousarray(edit_w[proj].T.astype(dt)).tobytes(); assert len(raw) == t['length'], (proj, len(raw), t['length'])
            f.seek(t['offset']); f.write(raw)
    m = onnx.load(str(base_graph), load_external_data=False)
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == 'location': e.value = data_out.name
    onnx.save(m, str(edited_graph)); os.chmod(edited_graph, 0o644)
    return data_out.name

def drift(src, out, variant, embv, n_per=2):
    """Layer-23 MLP input of the ORIGINAL ONNX decoder (ORT CPU) vs the campaign's BF16 MLX captures (runs/p04c, read-only)."""
    import onnx, onnxruntime as ort
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE)); from circuit_items import load_set, silu
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    graph = out / f'onnx/decoder_model_merged_{variant}.onnx'
    m = onnx.load(str(graph), load_external_data=False); want = f'/model/layers.{L}/post_attention_layernorm/output_0'
    fp16 = variant.endswith('f16') or variant == 'fp16'
    if want not in {o.name for o in m.graph.output}: m.graph.output.append(onnx.helper.make_tensor_value_info(want, onnx.TensorProto.FLOAT16 if fp16 else onnx.TensorProto.FLOAT, None))
    tmp = graph.parent / f'_drift_{variant}.onnx'; onnx.save(m, str(tmp))
    so = ort.SessionOptions(); so.log_severity_level = 3
    emb = ort.InferenceSession(str(out / f'onnx/embed_tokens_{embv}.onnx'), so, providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(str(tmp), so, providers=['CPUExecutionProvider']); meta = {i.name: i for i in dec.get_inputs()}; outs = [o.name for o in dec.get_outputs()]
    V = np.load(ROOT / 'edits/g23v3.vectors.npz'); z = np.load(ROOT / 'runs/p04c_practice_allpos.npz'); rows = {r['case_id']: r for r in json.load(open(ROOT / 'runs/p04c_practice_allpos.json'))['rows']}
    items = load_set(ROOT, 'practice'); pick = []
    for coh in ('no_tool', 'need_clarification', 'valid_call', 'ambiguous'): pick += [i for i in items if i['cohort'] == coh][:n_per]
    res = []
    for it in pick:
        text = tok.apply_chat_template([{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': it['user']}], tools=it['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False); T = len(ids); feed = {}
        e = emb.run(None, {'input_ids': np.array([ids], np.int64)})[0]
        for name, i in meta.items():
            dt = np.float16 if 'float16' in i.type else np.float32
            if name == 'inputs_embeds': feed[name] = e.astype(dt)
            elif name == 'attention_mask': feed[name] = np.ones((1, T), np.int64)
            elif name == 'position_ids': feed[name] = np.tile(np.arange(T, dtype=np.int64), (3, 1, 1))
            else: feed[name] = np.zeros([1 if isinstance(d, str) and 'batch' in d else 0 if isinstance(d, str) else d for d in i.shape], dt)
        r = dict(zip(outs, dec.run(None, feed))); x = r[want][0, -1].astype(np.float64); lg = r['logits'][0, -1].astype(np.float64)
        row = rows[it['id']]; xm = z[f"mlpin_L23_{it['id']}"].astype(np.float64)[row['decision_position']]
        gate, gate_m = 2 * float(x @ V['gate_dir']), 2 * float(xm @ V['gate_dir'])
        res.append({'id': it['id'], 'cohort': it['cohort'], 'cos': round(float(x @ xm / np.linalg.norm(x) / np.linalg.norm(xm)), 4), 's_x': round(float(x @ V['s']), 2), 's_x_mlx': round(float(xm @ V['s']), 2),
                    'gate': round(gate, 2), 'gate_mlx': round(gate_m, 2), 'act3468': round(float(silu(gate) * 2 * (x @ V['s'])), 2), 'act3468_mlx': round(float(silu(gate_m) * 2 * (xm @ V['s'])), 2),
                    'call_minus_NO': round(float(lg[248058] - lg[8725]), 2), 'call_minus_NO_mlx': round(row['gaps']['call_minus_NO'], 2)})
        print('  drift', json.dumps(res[-1]), flush=True)
    tmp.unlink()
    summary = {'variant': variant, 'embed': embv, 'n': len(res), 'mean_cos': round(float(np.mean([r['cos'] for r in res])), 4), 'min_cos': round(float(min(r['cos'] for r in res)), 4),
               'gate_sign_agrees': int(sum((r['gate'] > 0) == (r['gate_mlx'] > 0) for r in res)), 'rows': res}
    print('drift summary', {k: v for k, v in summary.items() if k != 'rows'}, flush=True)
    return summary

def validate(out, variant, n_items, max_new):
    """ORT CPU greedy decoding with embed_tokens + decoder (original and g23v3) vs the recorded campaign runs."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE)); from circuit_items import load_set, SYSTEM_TOOLS
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    so = ort.SessionOptions(); so.log_severity_level = 3
    emb = ort.InferenceSession(str(out / f'onnx/embed_tokens_{variant}.onnx'), so, providers=['CPUExecutionProvider'])
    items = load_set(ROOT, 'practice')
    pick = []
    for coh in ('valid_call', 'no_tool', 'need_clarification', 'ambiguous'):
        pick += [it for it in items if it['cohort'] == coh][: max(1, n_items // 4)]
    results = {}
    for tag, fname in (('original', f'decoder_model_merged_{variant}.onnx'), ('g23v3', f'decoder_model_merged_g23v3_{variant}.onnx')):
        dec = ort.InferenceSession(str(out / 'onnx' / fname), so, providers=['CPUExecutionProvider'])
        meta = {i.name: i for i in dec.get_inputs()}; outs = [o.name for o in dec.get_outputs()]
        def present_name(past):  # past_key_values.3.key -> present.3.key ; past_conv.0 -> present_conv.0 ; past_recurrent.0 -> present_recurrent.0
            for cand in (past.replace('past_key_values.', 'present.'), past.replace('past_', 'present_'), past.replace('past_key_values.', 'present_key_values.')):
                if cand in outs: return cand
            raise KeyError(f'no present output for {past}; outputs: {outs[:6]}...')
        state_map = {n: present_name(n) for n in meta if n.startswith('past')}
        rows = []
        for it in pick:
            text = tok.apply_chat_template([{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': it['user']}], tools=it['tools'] or None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False); T = len(ids)
            def embed(tok_ids): return emb.run(None, {'input_ids': np.array([tok_ids], np.int64)})[0]
            feed = {}
            for name, i in meta.items():
                if name == 'inputs_embeds': feed[name] = embed(ids).astype(np.float16 if 'float16' in i.type else np.float32)
                elif name == 'attention_mask': feed[name] = np.ones((1, T), np.int64)
                elif name == 'position_ids': feed[name] = np.tile(np.arange(T, dtype=np.int64), (3, 1, 1))
                else:
                    shape = [1 if isinstance(d, str) and 'batch' in d else 0 if isinstance(d, str) else d for d in i.shape]
                    feed[name] = np.zeros(shape, np.float16 if 'float16' in i.type else np.float32)
            gen = []; t0 = time.perf_counter(); pos = T
            for step in range(max_new):
                res = dict(zip(outs, dec.run(None, feed)))
                nxt = int(res['logits'][0, -1].astype(np.float32).argmax()); gen.append(nxt)
                if nxt in (248046, 248044): break
                for name, pres in state_map.items(): feed[name] = res[pres]
                feed['inputs_embeds'] = embed([nxt]).astype(feed['inputs_embeds'].dtype)
                feed['attention_mask'] = np.ones((1, pos + 1), np.int64); feed['position_ids'] = np.full((3, 1, 1), pos, np.int64); pos += 1
            dt = time.perf_counter() - t0
            rec = it['outputs'][tag]['token_ids']; k = min(len(gen), len(rec)); same_prefix = gen[:k] == rec[:k]
            rows.append({'id': it['id'], 'cohort': it['cohort'], 'generated': tok.decode(gen), 'recorded': it['outputs'][tag]['raw_output'][:80], 'prefix_match': bool(same_prefix), 'n': len(gen), 'seconds': round(dt, 1)})
            print(f"  [{tag}] {it['id']:<28s} {it['cohort']:<20s} match={same_prefix} {len(gen):3d} tok {dt:5.1f}s | {tok.decode(gen)[:70]!r}", flush=True)
        results[tag] = rows
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='onnx-community/Qwen3.5-0.8B-ONNX'); ap.add_argument('--out', default=str(ROOT / 'artifacts/exports/g23v3-onnx'))
    ap.add_argument('--variants', default='fp16', help='comma list of fp16, q4f16, q4 (fp16 is what the web app uses; see --drift)')
    ap.add_argument('--browser-variant', default='fp16'); ap.add_argument('--embed-dtype', default='q4f16', help='embed_tokens precision the web app loads (q4f16 or fp16)')
    ap.add_argument('--drift', action='store_true', help='measure layer-23 feature drift of each variant vs the BF16 captures (ORT CPU)'); ap.add_argument('--drift-items', type=int, default=2)
    ap.add_argument('--validate', action='store_true'); ap.add_argument('--validate-variant', default=None); ap.add_argument('--validate-items', type=int, default=4); ap.add_argument('--validate-tokens', type=int, default=24)
    ap.add_argument('--upload', default=None, help='Hugging Face repo id to upload the finished folder to (needs a logged-in token)')
    ap.add_argument('--skip-build', action='store_true', help='only validate / upload an existing --out folder')
    a = ap.parse_args()
    variants = [v for v in a.variants.split(',') if v]; out = Path(a.out); (out / 'onnx').mkdir(parents=True, exist_ok=True)
    report = {'source': a.src, 'variants': {}, 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if not a.skip_build:
        src = ensure_source(a.src, variants); print('source', src, flush=True)
        st = next(SNAPSHOT.glob('*.safetensors'))
        base_w = {p: f32(read_bits(st, HF_NAME.format(p))) for p in ('gate_proj', 'up_proj', 'down_proj')}
        edited_bits = build(st, ROOT / 'edits/g23v3.vectors.npz', ROOT / 'edits/g23v3.spec.json')
        edit_w = {p: f32(edited_bits[HF_NAME.format(p)]) for p in ('gate_proj', 'up_proj', 'down_proj')}
        for p in base_w: assert np.count_nonzero(base_w[p] != edit_w[p]) in (2047, 2048), p
        manifest = {'base_model': 'onnx-community/Qwen3.5-0.8B-ONNX', 'edit': 'g23v3', 'neurons': [J1, J2], 'layer': L, 'changed_entries': 6142, 'browser_variant': a.browser_variant, 'embed_dtype': a.embed_dtype, 'variants': {}}
        for v in variants:
            act = np.float16 if v.endswith('f16') or v == 'fp16' else np.float32
            names = tensor_names(); t0 = time.perf_counter()
            base_graph = out / f'onnx/decoder_model_merged_{v}.onnx'; edited_graph = out / f'onnx/decoder_model_merged_g23v3_{v}.onnx'
            m, rep = surgery(src / f'onnx/decoder_model_merged_{v}.onnx', base_w, act)
            if all(r.get('already_dense') for r in rep.values()):  # fp16 / fp32 export: the base files are onnx-community's, byte for byte
                del m
                for f in src.glob(f'onnx/decoder_model_merged_{v}.onnx*'):
                    if not (out / 'onnx' / f.name).exists() or (out / 'onnx' / f.name).stat().st_size != f.stat().st_size: shutil.copyfile(f, out / 'onnx' / f.name)
                    os.chmod(out / 'onnx' / f.name, 0o644)
                data_base = next(iter(locate(base_graph, names).values()))['location']
            else:
                data_base = save_external(m, base_graph); del m
            data_edit = patch_copy(base_graph, edited_graph, edit_w)
            loc = locate(base_graph, names); loc_e = locate(edited_graph, names)
            for n in names: assert loc[n]['offset'] == loc_e[n]['offset'] and loc[n]['length'] == loc_e[n]['length'], n
            tensors = {p: {k: loc[f'model.layers.{L}.mlp.{p}.MatMul.weight'][k] for k in ('offset', 'length', 'dims', 'dtype')} for p in ('gate_proj', 'up_proj', 'down_proj')}
            conv = fp16_bits if act is np.float16 else (lambda x: np.asarray(x, np.float32).tolist())
            patch = {p: {str(J): conv(edit_w[p][J] if p != 'down_proj' else edit_w[p][:, J]) for J in (J1, J2)} for p in ('gate_proj', 'up_proj', 'down_proj')}
            sb, se = sha256_file(out / 'onnx' / data_base), sha256_file(out / 'onnx' / data_edit)
            # differing bytes between the two data files must lie inside the three tensors
            A = np.memmap(out / 'onnx' / data_base, dtype=np.uint8, mode='r'); B = np.memmap(out / 'onnx' / data_edit, dtype=np.uint8, mode='r'); assert A.shape == B.shape
            dif = np.flatnonzero(A != B); inside = sum(int(((dif >= t['offset']) & (dif < t['offset'] + t['length'])).sum()) for t in tensors.values()); assert inside == len(dif), (inside, len(dif)); del A, B
            manifest['variants'][v] = {'graph': f'onnx/decoder_model_merged_{v}.onnx', 'data_file': f'onnx/{data_base}', 'edited_graph': f'onnx/decoder_model_merged_g23v3_{v}.onnx', 'edited_data_file': f'onnx/{data_edit}',
                                       'sha256_base': sb, 'sha256_edited': se, 'tensors': tensors, 'patch': patch, 'activation_dtype': 'float16' if act is np.float16 else 'float32', 'differing_bytes': int(len(dif))}
            report['variants'][v] = {'layout_check_base': rep, 'seconds': round(time.perf_counter() - t0, 1), 'bytes_base': (out / 'onnx' / data_base).stat().st_size, 'bytes_edited': (out / 'onnx' / data_edit).stat().st_size, 'differing_bytes': int(len(dif))}
            print(f'[{v}] built in {time.perf_counter() - t0:.0f}s; {len(dif)} differing bytes; layout check {json.dumps(rep)}', flush=True)
            for n in ('embed_tokens', 'vision_encoder'):
                for f in list(src.glob(f'onnx/{n}_{v}.onnx*')) + list(src.glob(f'onnx/{n}_{a.embed_dtype}.onnx*')):
                    if not (out / 'onnx' / f.name).exists(): shutil.copy2(f, out / 'onnx' / f.name)
        bv = manifest['variants'][manifest['browser_variant']]
        manifest.update({'data_file': bv['data_file'], 'sha256_base': bv['sha256_base'], 'sha256_edited': bv['sha256_edited']})
        (out / 'patch_manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
        for f in SIDE_FILES:
            if (src / f).exists(): shutil.copy2(src / f, out / f)
        cfg = json.loads((out / 'config.json').read_text()); tj = cfg.setdefault('transformers.js_config', {}); ued = tj.setdefault('use_external_data_format', {})
        for v in variants: ued[f'decoder_model_merged_g23v3_{v}.onnx'] = 1
        ued['decoder_model_merged_g23v3'] = 1
        (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        (out / 'README.md').write_text(model_card(manifest, report))
        print('wrote', out, flush=True)
    if a.drift:
        src = ensure_source(a.src, variants) if not a.skip_build else None
        report['drift'] = {v: drift(src or out, out, v, a.embed_dtype if v != 'q4' else 'q4', a.drift_items) for v in variants}
    if a.validate:
        v = a.validate_variant or variants[0]
        try:
            report['validation'] = validate(out, v, a.validate_items, a.validate_tokens)
            ok = {t: sum(r['prefix_match'] for r in rows) for t, rows in report['validation'].items()}; print('validation prefix matches:', ok, flush=True)
        except Exception as e: report['validation_error'] = f'{type(e).__name__}: {e}'; print('validation failed:', e, flush=True)
    if (out / 'build_report.json').exists():
        old_rep = json.loads((out / 'build_report.json').read_text())
        for k, v in old_rep.items():  # keep earlier measurements and notes unless this run produced new ones
            if k not in report or (k == 'variants' and not report['variants']): report[k] = v
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(a.upload, repo_type='model', exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.upload, repo_type='model', commit_message='ONNX (q4f16/q4) export of the g23v3 two-neuron layer-23 edit for Transformers.js / WebGPU')
        print('uploaded to', a.upload)

def model_card(man, rep):
    bv = man['variants'].get(man.get('browser_variant', 'fp16')) or next(iter(man['variants'].values()))
    dr = rep.get('drift', {})
    def drow(v, label):
        d = dr.get(v)
        if not d: return f"| {label} | not measured | | |"
        acts = ', '.join('%.1f vs %.1f' % (r['act3468'], r['act3468_mlx']) for r in d['rows'] if r['cohort'] == 'no_tool')
        return "| %s | %.3f (min %.3f) | %d/%d | %s |" % (label, d['mean_cos'], d['min_cos'], d['gate_sign_agrees'], d['n'], acts)
    return f"""---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-0.8B
- scasella91/Qwen3.5-0.8B-toolpatch-g23v3
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, qwen3_5, webgpu, transformers.js, tool-calling, weight-editing]
---
# Qwen3.5-0.8B-toolpatch-g23v3 (ONNX fp16, for Transformers.js / WebGPU)

ONNX export of [scasella91/Qwen3.5-0.8B-toolpatch-g23v3](https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v3): `Qwen/Qwen3.5-0.8B` with a
closed-form two-neuron edit of the layer-23 MLP (6,142 BF16 weights, 0.00082 % of the text model) that stops the model from emitting `<tool_call>` on
requests that should not be calls. Story, figures and controls: [blog post](https://scasella.github.io/qwen3.5-toolpatch/blog.html); interactive
comparison (this repository powers its *Run live in browser* tab): https://scasella.github.io/qwen3.5-toolpatch/ ; code: https://github.com/scasella/qwen3.5-toolpatch .

Built from [onnx-community/Qwen3.5-0.8B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) by `scripts/export_webgpu_onnx.py`, without
re-exporting or re-quantizing anything. Two fp16 decoders:

| file | weights |
|---|---|
| `onnx/decoder_model_merged_fp16.onnx` (+`_data`, 1.5 GB) | **original** Qwen3.5-0.8B -- byte-identical to onnx-community's fp16 export |
| `onnx/decoder_model_merged_g23v3_fp16.onnx` (+`_data`) | **g23v3** edit -- a byte copy of the file above with exactly the 6,142 edited entries overwritten |

`patch_manifest.json` lists the byte offsets of the three layer-23 MLP tensors and the edited values, so a client can download the original once
and produce g23v3 in memory; the web app does this and checks the SHA-256 of the result (base `{bv['sha256_base'][:16]}…`, edited `{bv['sha256_edited'][:16]}…`).
`embed_tokens_*` (q4f16 and fp16) and `vision_encoder_fp16` are copied unchanged; tokenizer, chat template and configs are the base model's.

## Why fp16 and not a 4-bit build

The edit is a gated readout of directions in the layer-23 residual stream that were fitted on the BF16 model's activations. `export_webgpu_onnx.py --drift`
runs the ORIGINAL decoder of each precision on ORT (CPU) and compares the layer-23 MLP input at the decision position with the campaign's BF16 captures
on eight practice prompts (two per class):

| backbone | cosine to BF16 input, mean (min) | abstain-gate sign agrees | neuron 3468 on the two explanation prompts (ONNX vs BF16) |
|---|---|---|---|
{drow('fp16', 'fp16 decoder, q4f16 embeddings (shipped)')}
{drow('q4f16', 'q4f16 decoder (layer-23 MLP in fp16)')}
{drow('q4', 'q4 decoder (layer-23 MLP in fp32)')}

With a 4-bit backbone the input has drifted far enough that the gate closes on explanation prompts and the edited neurons never fire -- the model keeps
calling tools. fp16 keeps the features, so that is what is shipped (one 1.5 GB download; the two 4-bit variants can be rebuilt with `--variants q4f16,q4`
but are not published because the published edit does not behave as the edit there).

## Use with Transformers.js (WebGPU)

```js
import {{ AutoTokenizer, Qwen3_5ForCausalLM, TextStreamer }} from "@huggingface/transformers";
const id = "scasella91/Qwen3.5-0.8B-toolpatch-g23v3-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(id);
// text-only: loads embed_tokens + decoder_model_merged (ORIGINAL weights)
const model = await Qwen3_5ForCausalLM.from_pretrained(id, {{ device: "webgpu", dtype: {{ embed_tokens: "q4f16", decoder_model_merged: "fp16" }} }});
const text = tokenizer.apply_chat_template([{{ role: "system", content: "Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text." }}, {{ role: "user", content: "Explain what the archive tool does; do not run it." }}], {{ tools, add_generation_prompt: true, tokenize: false, enable_thinking: false }});
const out = await model.generate({{ ...tokenizer(text, {{ add_special_tokens: false }}), max_new_tokens: 128, do_sample: false, streamer: new TextStreamer(tokenizer, {{ skip_prompt: true }}) }});
```
To run the **edited** decoder with the stock library, load a copy of the folder in which `decoder_model_merged_g23v3_fp16.onnx(_data)` is renamed to
`decoder_model_merged_fp16.onnx(_data)`, or patch the downloaded data buffer with `patch_manifest.json` as `docs/worker.js` in the GitHub repository does.

## Caveats
The campaign's results (token-identical calls, 0 right-to-wrong regressions, 87 wrong-to-right) were measured on the BF16 MLX weights with greedy decoding;
fp16 ONNX on WebGPU reproduces the mechanism but small numerical differences can change individual outputs. Evaluated only under the campaign's synthetic
protocol; see the main model card for limitations.
"""

if __name__ == '__main__': main()
