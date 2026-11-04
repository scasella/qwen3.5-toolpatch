// Mode 2 worker: real Qwen3.5-0.8B generation in the browser with Transformers.js + ONNX Runtime Web (WebGPU).
// One download serves both models: the "edited" model is made in memory by writing the 6,142 edited values (from
// patch_manifest.json) into the layer-23 MLP tensors of the downloaded ORIGINAL data file before the session is created.
// The repository's patch_manifest.json names the precision to load (fp16: with a 4-bit backbone the layer-23 input drifts
// far enough that the abstain gate closes, so the edit does not transfer to q4f16; see scripts/export_webgpu_onnx.py --drift).
// Everything the UI needs arrives as postMessage events: log, progress, loaded, first_logits, token, done, all_done, error.
import { env, AutoTokenizer, Qwen3_5ForCausalLM, TextStreamer, LogitsProcessor, InterruptableStoppingCriteria }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.4.0-next.0/dist/transformers.min.js';  // self-contained bundle (ORT included); the .web.js builds import onnxruntime-web by bare specifier

const post = (m) => self.postMessage(m);
const log = (text) => post({ type: 'log', text });
const CACHE_NAME = 'qwen-toolpatch-models-v2';  // v1 used URL fragments as chunk keys, which the Cache API strips
const W = { modelId: null, baseUrl: null, tokenizer: null, manifest: null, patchMode: 'original', models: {}, stopper: null, keepBoth: true, f16: true, hashes: {} };

// ---------------------------------------------------------------- file loading: own Cache-API store + in-memory layer-23 patch
env.allowLocalModels = false; env.allowRemoteModels = true; env.useBrowserCache = false; env.useCustomCache = false;
const cachePromise = (typeof caches !== 'undefined') ? caches.delete('qwen-toolpatch-models-v1').catch(() => null).then(() => caches.open(CACHE_NAME)).catch(() => null) : Promise.resolve(null);
const nativeFetch = self.fetch.bind(self);
env.fetch = async (url, init) => {
  const u = String(url);
  if (!W.baseUrl || !u.startsWith(W.baseUrl) || (init && init.method && init.method !== 'GET')) return nativeFetch(url, init);
  const file = u.slice(W.baseUrl.length);
  const resp = await cachedFetch(u, file);
  if (resp.status !== 200) return resp;
  if (W.patchMode === 'edited' && W.manifest && file === (W.manifest.variants?.[W.manifest.browser_variant || 'fp16']?.data_file || W.manifest.data_file)) {
    const buf = await readAll(resp, file, 'patching layer 23');
    const MV = W.manifest.variants[W.manifest.browser_variant || 'fp16'];
    const h0 = await sha256(buf); W.hashes.base = h0;
    if (MV.sha256_base && h0 !== MV.sha256_base) log(`WARNING: base data file hash ${h0.slice(0, 12)} differs from manifest ${MV.sha256_base.slice(0, 12)}`);
    applyPatch(buf, W.manifest);
    const h1 = await sha256(buf); W.hashes.edited = h1;
    post({ type: 'progress', variant: 'edited', file, loaded: buf.byteLength, total: buf.byteLength, status: h1 === MV.sha256_edited ? 'patched · SHA-256 matches the published edited file' : 'patched' });
    return new Response(buf, { status: 200, headers: { 'content-length': String(buf.byteLength), 'content-type': 'application/octet-stream' } });
  }
  return resp;
};
// Large files are stored in the Cache API as 64 MB chunks (url.__chunkN) plus an index entry (url.__index): a single 1.5 GB entry fails in Chromium.
// (Suffixes, not fragments: the Cache API ignores URL fragments when keying.)
const CHUNK = 64 * 1024 * 1024;
async function cacheGet(cache, u, file) {
  const idx = await cache.match(u + '.__index'); if (!idx) return null;
  const { chunks, total, type } = await idx.json();
  const out = new Uint8Array(total); let o = 0;
  for (let i = 0; i < chunks; i++) {
    const part = await cache.match(`${u}.__chunk${i}`); if (!part) return null;
    const b = new Uint8Array(await part.arrayBuffer()); out.set(b, o); o += b.byteLength;
    post({ type: 'progress', variant: W.patchMode, file, loaded: o, total, status: 'from cache' });
  }
  return new Response(out.buffer, { status: 200, headers: { 'content-length': String(total), 'content-type': type } });
}
async function cachePut(cache, u, buf, type) {
  const chunks = Math.ceil(buf.byteLength / CHUNK);
  for (let i = 0; i < chunks; i++) await cache.put(`${u}.__chunk${i}`, new Response(buf.slice(i * CHUNK, Math.min(buf.byteLength, (i + 1) * CHUNK)), { status: 200 }));
  await cache.put(u + '.__index', new Response(JSON.stringify({ chunks, total: buf.byteLength, type }), { status: 200, headers: { 'content-type': 'application/json' } }));
}
async function cachedFetch(u, file) {
  const cache = await cachePromise;
  if (cache) { try { const hit = await cacheGet(cache, u, file); if (hit) return hit; } catch (e) { log(`cache read failed for ${file}: ${e.message || e}`); } }
  const r = await nativeFetch(u);
  if (!r.ok) return r;
  const buf = await readAll(r, file, 'downloading');
  const type = r.headers.get('content-type') || 'application/octet-stream';
  if (cache && buf.byteLength > 0) { try { await cachePut(cache, u, buf, type); } catch (e) { log(`cache store failed for ${file}: ${e.message || e} (continuing without cache)`); } }
  return new Response(buf, { status: 200, headers: { 'content-length': String(buf.byteLength), 'content-type': type } });
}
async function readAll(resp, file, status) {
  const total = +resp.headers.get('content-length') || 0;
  if (!resp.body) { const b = await resp.arrayBuffer(); post({ type: 'progress', variant: W.patchMode, file, loaded: b.byteLength, total: b.byteLength, status }); return b; }
  const reader = resp.body.getReader(); const chunks = []; let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); loaded += value.byteLength;
    const now = performance.now(); if (now - lastPost > 120) { lastPost = now; post({ type: 'progress', variant: W.patchMode, file, loaded, total, status }); }
  }
  const out = new Uint8Array(loaded); let o = 0; for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  post({ type: 'progress', variant: W.patchMode, file, loaded, total: total || loaded, status: status === 'downloading' ? 'downloaded' : status });
  return out.buffer;
}
async function sha256(buf) { const d = await crypto.subtle.digest('SHA-256', buf); return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join(''); }
function applyPatch(buf, man) {
  // ONNX stores MatMul weights transposed w.r.t. the checkpoint: gate/up are [1024 (in), 3584 (out)], down is [3584 (in), 1024 (out)].
  const dv = new DataView(buf); const P = man.variants[man.browser_variant || 'fp16']; let n = 0;
  for (const [name, t] of Object.entries(P.tensors)) {
    const [rows, cols] = t.dims; const bytes = t.dtype === 'float16' ? 2 : 4;
    for (const [j, vals] of Object.entries(P.patch[name])) {
      const J = +j;
      for (let i = 0; i < vals.length; i++) {
        const idx = name === 'down_proj' ? J * cols + i : i * cols + J;   // down: row J; gate/up: column J
        const off = t.offset + idx * bytes;
        if (bytes === 2) dv.setUint16(off, vals[i], true); else dv.setFloat32(off, vals[i], true);
        n++;
      }
    }
  }
  log(`wrote ${n} values into the layer-23 MLP tensors (neurons ${man.neurons.join(', ')})`);
}

// ---------------------------------------------------------------- model management
function setSource(source) {
  let host, template, modelId;
  if (/^https?:\/\//i.test(source)) { const s = source.replace(/\/+$/, ''); const k = s.lastIndexOf('/'); host = s.slice(0, k + 1); template = '{model}/'; modelId = s.slice(k + 1); }
  else { host = 'https://huggingface.co/'; template = '{model}/resolve/{revision}/'; modelId = source.replace(/^\/+|\/+$/g, ''); }
  env.remoteHost = host; env.remotePathTemplate = template; W.modelId = modelId;
  W.baseUrl = host + template.replace('{model}', modelId).replace('{revision}', 'main');
}
async function loadVariant(variant) {
  if (W.models[variant]) return W.models[variant];
  W.patchMode = variant; const t0 = performance.now();
  if (!W.manifest) {
    const r = await env.fetch(W.baseUrl + 'patch_manifest.json'); if (!r.ok) throw new Error(`patch_manifest.json not found at ${W.baseUrl} (${r.status})`);
    W.manifest = await r.json(); log(`manifest: ${W.manifest.data_file}, edited neurons ${W.manifest.neurons.join('/')}, ${W.manifest.changed_entries} entries`);
  }
  if (!W.tokenizer) W.tokenizer = await AutoTokenizer.from_pretrained(W.modelId);
  const dec = W.manifest.browser_variant || 'fp16', embd = W.manifest.embed_dtype || 'q4f16';
  if (!W.loggedDtype) { W.loggedDtype = true; log(`decoder ${dec} (layer-23 MLP in ${W.manifest.variants?.[dec]?.activation_dtype || 'fp16'}), embeddings ${embd}, device webgpu`); }
  const model = await Qwen3_5ForCausalLM.from_pretrained(W.modelId, {
    device: 'webgpu', dtype: { embed_tokens: embd, decoder_model_merged: dec },
    progress_callback: (p) => { if (p.status === 'progress' || p.status === 'done' || p.status === 'initiate') post({ type: 'progress', variant, file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0, status: p.status === 'done' ? 'ready' : p.status }); },
  });
  W.models[variant] = model;
  const verified = variant === 'edited' ? (W.hashes.edited ? W.hashes.edited === W.manifest.variants[W.manifest.browser_variant || 'fp16'].sha256_edited : undefined) : undefined;
  post({ type: 'loaded', variant, ms: performance.now() - t0, verified });
  return model;
}
async function unloadVariant(variant) { const m = W.models[variant]; if (!m) return; delete W.models[variant]; try { await m.dispose(); } catch (e) { log(`dispose(${variant}): ${e.message || e}`); } }

class FirstStepLogits extends LogitsProcessor {
  constructor() { super(); this.first = null; }
  _call(input_ids, logits) {
    if (this.first === null) { const d = logits.data; const n = logits.dims.at(-1); this.first = Float32Array.from(d.subarray ? d.subarray(0, n) : Array.prototype.slice.call(d, 0, n)); }
    return logits;
  }
}
const NAMED = ['<tool_call>', 'NO', 'NE', 'I', 'To', 'The'];
async function generateOne(variant, system, user, tools, maxNewTokens) {
  const model = await loadVariant(variant); const tok = W.tokenizer;
  const messages = [{ role: 'system', content: system }, { role: 'user', content: user }];
  const opts = { add_generation_prompt: true, tokenize: false, enable_thinking: false }; if (tools && tools.length) opts.tools = tools;
  const text = tok.apply_chat_template(messages, opts);
  const inputs = tok(text, { add_special_tokens: false });
  const nPrompt = inputs.input_ids.dims.at(-1);
  log(`${variant}: prompt ${nPrompt} tokens`);
  const fl = new FirstStepLogits(); const stopper = new InterruptableStoppingCriteria(); W.stopper = stopper;
  let n = 0, tFirst = 0; const t0 = performance.now();
  const streamer = new TextStreamer(tok, {
    skip_prompt: true, skip_special_tokens: false,
    token_callback_function: () => { n++; if (!tFirst) { tFirst = performance.now(); emitFirstLogits(variant, fl, tok); } },
    callback_function: (txt) => post({ type: 'token', variant, text: txt, n, elapsed: performance.now() - t0 }),
  });
  const out = await model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false, streamer, stopping_criteria: stopper, logits_processor: [fl] });
  const tEnd = performance.now();
  const gen = out.slice(null, [nPrompt, null]); const ids = gen.tolist()[0].map(Number); const full = tok.decode(ids, { skip_special_tokens: false });
  post({ type: 'done', variant, nTokens: ids.length, prefillMs: (tFirst || tEnd) - t0, genMs: tEnd - (tFirst || t0), text: full, stopped: stopper.interrupted === true });
}
function emitFirstLogits(variant, fl, tok) {
  if (!fl.first) return; const L = fl.first; const named = {};
  for (const w of NAMED) { const ids = tok.encode(w, { add_special_tokens: false }); if (ids.length === 1) named[w] = L[ids[0]]; }
  const idx = [...L.keys()]; idx.sort((a, b) => L[b] - L[a]);
  post({ type: 'first_logits', variant, named, top: idx.slice(0, 6).map((i) => [tok.decode([i]), L[i]]) });
}

// ---------------------------------------------------------------- message loop
let busy = false;
self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'load') {
      if (busy) return log('busy; try again when the current job finishes');
      busy = true; W.keepBoth = m.keepBoth; W.f16 = m.f16; setSource(m.source);
      for (const v of Object.keys(W.models)) await unloadVariant(v);
      log(`base URL ${W.baseUrl}`);
      await loadVariant('original');
      if (W.keepBoth) await loadVariant('edited'); else log('sequential mode: the edited model is built after the original has generated');
      busy = false;
    } else if (m.type === 'generate') {
      if (busy) return log('busy');
      busy = true; W.keepBoth = m.keepBoth;
      for (const v of ['original', 'edited']) {
        if (!W.keepBoth) { for (const o of Object.keys(W.models)) if (o !== v) await unloadVariant(o); }
        await generateOne(v, m.system, m.user, m.tools, m.maxNewTokens);
        if (W.stopper?.interrupted) break;
      }
      busy = false; post({ type: 'all_done' });
    } else if (m.type === 'stop') { W.stopper?.interrupt(); }
  } catch (e) { busy = false; console.error(e); post({ type: 'error', text: (e && e.message) ? e.message : String(e) }); }
};
log('worker ready');
