// Two-Neuron Circuit Explorer — Mode 1 (exact closed-form readout of the g23v3 edit) and Mode 2 (WebGPU runner UI).
// Mode 1 math: the edit changes the final residual h24 only inside a fixed 7-vector basis b_k; with coef_k the weight on b_k,
//   logit_t = (L_t*rms + sum_k coef_k B_kt) / sqrt((sumsq + 2 sum_k coef_k hb_k + coef^T G coef)/d + eps)
// where L_t are the ORIGINAL model's logits at the decision position, B_kt = (b_k*gamma).E_t, hb_k = h24.b_k, G = Gram(b).
// This is identical to running the edited layer-23 MLP + final RMSNorm + unembedding (checked to 1e-13 in the export script).

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : (Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const silu = (z) => z / (1 + Math.exp(-z));

const VARIANTS = ['original', 'v1', 'v2', 'g23v3', 'rand13'];
const VLABEL = { original: 'Original', v1: 'v1 (first candidate)', v2: 'v2', g23v3: 'g23v3 (released)', rand13: 'Random control' };
const COHORTS = [['valid_call', 'Valid call'], ['no_tool', 'Explanation → NO_TOOL'], ['need_clarification', 'Missing value → NEED_CLARIFICATION'], ['ambiguous', 'Ambiguous "either X or Y"'], ['prose', 'Non-tool prose QA (128 tokens)']];
const SETS = [['practice', 'Practice (64)'], ['selection', 'Selection, unseen families (32)'], ['renamed_A', 'Renamed A (32)'], ['renamed_B', 'Renamed B (32)'], ['final', 'Final-split proxy (32)'], ['prose_qa', 'Prose QA (16)'], ['prose_qa2', 'Prose QA 2 (16)']];
const DECISION_NAME = { call: '<tool_call> → tool call', answer: 'NO_TOOL', clarify: 'NEED_CLARIFICATION', prose: 'prose', null: '—' };

const S = { data: null, items: [], filtered: [], cur: null, variant: 'g23v3', sl: { a1: 1, a2: 1, lam: 0.3, lam2: 0.5 }, ids: {}, tidx: null, pos: {} };

// ------------------------------------------------------------------ boot
init().catch((e) => { $('#loading').textContent = 'Failed to load data: ' + e; console.error(e); });
async function init() {
  const r = await fetch('data/circuit_traces.json'); if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  S.data = await r.json(); S.items = S.data.items;
  const C = S.data.constants; S.sl = { a1: C.a1, a2: C.a2, lam: C.lam, lam2: C.lam2 };
  S.ids = S.data.tracked.named; S.tidx = new Map(S.data.tracked.ids.map((t, i) => [t, i]));
  buildFilters(); buildVariants(); buildSliders(); renderProvenance(); setupTabs();
  const q = new URLSearchParams(location.search).get('item');
  applyFilters();
  selectItem(S.items.find((it) => it.id === q) || S.items.find((it) => it.id === 'discovery-611401-0226') || S.items[0]);
  $('#loading').hidden = true; $('#explorer').hidden = false;
  if (location.hash === '#live') switchTab('live');
  setupLive();
}

// ------------------------------------------------------------------ filters / list
function buildFilters() {
  const fc = $('#filter-cohort'), fs = $('#filter-set');
  for (const [k, lab] of COHORTS) fc.insertAdjacentHTML('beforeend', `<label><input type="checkbox" data-cohort="${k}" checked> ${esc(lab)}</label>`);
  for (const [k, lab] of SETS) fs.insertAdjacentHTML('beforeend', `<label><input type="checkbox" data-set="${k}" checked> ${esc(lab)}</label>`);
  fc.addEventListener('change', applyFilters); fs.addEventListener('change', applyFilters);
  $('#search').addEventListener('input', applyFilters);
}
function applyFilters() {
  const coh = new Set($$('#filter-cohort input:checked').map((i) => i.dataset.cohort));
  const sets = new Set($$('#filter-set input:checked').map((i) => i.dataset.set));
  const q = $('#search').value.trim().toLowerCase();
  S.filtered = S.items.filter((it) => coh.has(it.cohort) && sets.has(it.set) && (!q || (it.id + ' ' + it.family + ' ' + it.user).toLowerCase().includes(q)));
  const ul = $('#itemlist'); ul.innerHTML = '';
  for (const it of S.filtered) {
    const o = it.outputs, d0 = o.original?.decision, d1 = o.g23v3?.decision;
    const li = document.createElement('li'); li.role = 'option'; li.dataset.id = it.id; li.className = S.cur && S.cur.id === it.id ? 'sel' : '';
    li.innerHTML = `<span class="id" title="${esc(it.id)}">${esc(it.id)}</span><span class="badge ${it.cohort}">${esc(cohortShort(it))}</span>
      <span class="fam">${esc(it.family)} · ${esc(it.set)} · ${esc(shortDec(d0))} → <b>${esc(shortDec(d1))}</b>${o.g23v3 ? (o.g23v3.correct ? ' ✓' : ' ✗') : ''}</span>`;
    li.onclick = () => selectItem(it); ul.appendChild(li);
  }
  $('#count').textContent = `${S.filtered.length} of ${S.items.length} prompts`;
}
const cohortShort = (it) => ({ valid_call: 'call', no_tool: 'NO_TOOL', need_clarification: 'NEED_CLAR', ambiguous: 'ambiguous', prose: 'prose' })[it.cohort];
const shortDec = (d) => ({ call: 'call', answer: 'NO_TOOL', clarify: 'NEED_CLAR', prose: 'prose' })[d] ?? '—';

// ------------------------------------------------------------------ item
function selectItem(it) {
  if (!it) return; S.cur = it;
  $$('#itemlist li').forEach((li) => li.classList.toggle('sel', li.dataset.id === it.id));
  const u = new URL(location); u.searchParams.set('item', it.id); history.replaceState(null, '', u);
  $('#item-title').textContent = it.id;
  $('#item-badges').innerHTML = `<span class="badge ${it.cohort}">${esc(COHORTS.find((c) => c[0] === it.cohort)[1])}</span><span class="badge">${esc(it.set)}</span><span class="badge">${esc(it.family)}</span>${it.renamed ? '<span class="badge">renamed identifiers</span>' : ''}<span class="badge">${it.prompt_tokens} prompt tokens</span>`;
  $('#p-system').textContent = it.system; $('#p-user').textContent = it.user; $('#p-expected').textContent = it.expected;
  $('#p-tools').textContent = it.tools.length ? JSON.stringify(it.tools, null, 1) : '(no tools in this prompt)';
  $('#p-tools-summary').textContent = it.tools.length ? it.tools.map((t) => t.function.name).join(', ') + ` (${it.tools.length} functions, click to expand)` : 'no tools';
  renderOutputs(it); recompute();
}
function renderOutputs(it) {
  const box = $('#outputs'); box.innerHTML = '';
  for (const v of VARIANTS) {
    const o = it.outputs[v]; const col = document.createElement('div'); col.className = 'out-col ' + v + (o ? '' : ' missing');
    if (!o) { col.innerHTML = `<h3>${esc(VLABEL[v])}</h3><pre>not run on this item in the campaign</pre><div class="meta">${v === 'rand13' ? 'random controls were run on the practice set only' : v === 'v1' ? 'v1 was dropped before the final-split pass' : 'no recorded pass'}</div>`; }
    else {
      const ok = o.correct ? 'ok' : 'bad';
      col.innerHTML = `<h3>${esc(VLABEL[v])}<span class="badge ${o.decision}">${esc(DECISION_NAME[o.decision] ?? o.decision)}</span></h3><pre>${esc(o.raw_output)}</pre>
        <div class="meta"><span class="badge ${ok}">${it.cohort === 'prose' ? (o.correct ? 'intact prose' : 'corrupted') : (o.correct ? 'correct' : 'wrong')}${o.correct_strict && it.cohort !== 'prose' ? ' · strict' : ''}</span><span>${o.n_tokens} tok${o.truncated ? ' (cut at 128)' : ''}</span>${v !== 'original' ? `<span>${o.identical_to_original ? 'token-identical to original' : 'differs from original'}</span>` : ''}<span title="campaign pass">${esc(o.run)}</span></div>`;
    }
    box.appendChild(col);
  }
}

// ------------------------------------------------------------------ variants & sliders
function buildVariants() {
  const box = $('#variants');
  for (const v of VARIANTS) { const b = document.createElement('button'); b.textContent = VLABEL[v]; b.dataset.v = v; b.title = S.data.variants[v].label; b.onclick = () => setVariant(v); box.appendChild(b); }
  setVariant('g23v3', false);
}
function setVariant(v, re = true) {
  S.variant = v; $$('#variants button').forEach((b) => b.classList.toggle('active', b.dataset.v === v));
  const C = S.data.constants, V = S.data.variants[v];
  if (v === 'original') S.sl = { ...S.sl, a1: 0, a2: 0 };
  else if (v === 'rand13') S.sl = { ...S.sl, a1: 1, a2: 1 };
  else S.sl = { a1: C.a1, a2: C.a2, lam: V.lam, lam2: V.lam2 };
  const gateFree = v === 'original' || v === 'rand13';
  for (const k of ['lam', 'lam2']) $('#sl-' + k).disabled = gateFree;
  for (const k of ['a1', 'a2']) $('#sl-' + k).disabled = v === 'original';
  syncSliders(); if (re) recompute();
}
function buildSliders() {
  for (const k of ['a1', 'a2', 'lam', 'lam2']) $('#sl-' + k).addEventListener('input', (e) => { S.sl[k] = +e.target.value; $('#o-' + k).value = fmt(S.sl[k], 2); if (S.variant === 'original' && (k === 'a1' || k === 'a2')) {} recompute(); });
  syncSliders();
}
function syncSliders() { for (const k of ['a1', 'a2', 'lam', 'lam2']) { $('#sl-' + k).value = S.sl[k]; $('#o-' + k).value = fmt(S.sl[k], 2); } }

// ------------------------------------------------------------------ the math
function neuron(it, j, variant = S.variant, sl = S.sl) {
  const C = S.data.constants, k = C.k, kr = C.kr;
  const [sx, rx, cx, c2x, s13, r13, g13, g1, u1, g2, u2] = it.positions.proj[j];
  const old1 = silu(g1) * u1, old2 = silu(g2) * u2;
  const coef = [0, 0, 0, 0, 0, 0, 0];
  if (variant === 'original') return { gate: NaN, act1: 0, act2: 0, old1, old2, comps: { s: NaN, c: NaN, c2: NaN }, coef, sx, rx, cx, c2x };
  coef[0] = -old1; coef[1] = -old2;
  if (variant === 'rand13') {
    const gate = k * g13, act1 = silu(gate) * k * s13, act2 = silu(gate) * kr * r13;
    coef[5] = sl.a1 * act1; coef[6] = sl.a2 * act2;
    return { gate, act1, act2, old1, old2, comps: { s: k * g13, c: 0, c2: 0 }, coef, sx: s13, rx: r13, cx: 0, c2x: 0 };
  }
  const gs = k * sx, gc = -k * sl.lam * cx, gc2 = -k * sl.lam2 * c2x, gate = gs + gc + gc2;
  const act1 = silu(gate) * k * sx, act2 = silu(gate) * kr * rx;
  coef[variant === 'v1' ? 4 : 2] = sl.a1 * act1; coef[3] = sl.a2 * act2;
  return { gate, act1, act2, old1, old2, comps: { s: gs, c: gc, c2: gc2 }, coef, sx, rx, cx, c2x };
}
function logitsFor(it, coef) {
  const d = it.decision, B = S.data.tracked.B, G = S.data.gram, C = S.data.constants, K = d.logits.length;
  let r2 = d.sumsq;
  for (let k = 0; k < 7; k++) { r2 += 2 * coef[k] * d.hb[k]; for (let l = 0; l < 7; l++) r2 += coef[k] * G[k][l] * coef[l]; }
  const rn = Math.sqrt(r2 / C.d + C.eps), out = new Float64Array(K);
  for (let t = 0; t < K; t++) { let num = d.logits[t] * d.rms; for (let k = 0; k < 7; k++) if (coef[k] !== 0) num += coef[k] * B[k][t]; out[t] = num / rn; }
  return { logits: out, rms: rn };
}
function softmaxWithTail(it, logits) {
  // tail = all non-tracked tokens, held at their ORIGINAL logits (exact when nothing is edited; measured error bound shown in the UI)
  const d = it.decision; let m = -Infinity; for (const v of logits) if (v > m) m = v;
  let sTr = 0; for (let t = 0; t < d.logits.length; t++) sTr += Math.exp(d.logits[t] - d.lse);
  const tailLse = sTr < 1 ? d.lse + Math.log(1 - sTr) : -Infinity;
  const mm = Math.max(m, tailLse); let z = Math.exp(tailLse - mm); const e = new Float64Array(logits.length);
  for (let t = 0; t < logits.length; t++) { e[t] = Math.exp(logits[t] - mm); z += e[t]; }
  for (let t = 0; t < logits.length; t++) e[t] /= z;
  return { p: e, tail: Math.exp(tailLse - mm) / z };
}

// ------------------------------------------------------------------ render live circuit
function recompute() {
  const it = S.cur; if (!it) return;
  const n = neuron(it, 0); const base = logitsFor(it, [0, 0, 0, 0, 0, 0, 0]); const ed = logitsFor(it, n.coef);
  const sm = softmaxWithTail(it, ed.logits), smb = softmaxWithTail(it, base.logits);
  const order = [...ed.logits.keys()].sort((a, b) => ed.logits[b] - ed.logits[a]);
  const am = S.data.tracked.ids[order[0]], amText = S.data.tracked.text[order[0]];
  renderSchematic(it, n, ed, base, am, amText);
  renderReadouts(it, n, ed, base, am, amText, sm.p[order[0]]);
  renderTokens(it, ed, base, sm, smb, order, am);
  renderTrace(it);
}
function decisionOf(tokId, text) { const N = S.ids; if (tokId === N['<tool_call>']) return ['call', '<tool_call> → tool call']; if (tokId === N['NO']) return ['answer', 'NO → NO_TOOL']; if (tokId === N['NE']) return ['clarify', 'NE → NEED_CLARIFICATION']; return ['prose', `“${text}” → prose`]; }
function renderReadouts(it, n, ed, base, am, amText, pam) {
  const C = S.data.constants, [dc, dl] = decisionOf(am, amText);
  const idx = (w) => S.tidx.get(S.ids[w]);
  const g = (arr, w) => idx(w) === undefined ? NaN : arr[idx(w)];
  const cls = (v) => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
  const ro = [
    ['decision', `first token`, `<span class="badge ${dc}">${esc(dl)}</span>`, `p = ${fmt(pam, 3)} · margin over runner-up ${fmt(ed.logits[[...ed.logits.keys()].sort((a, b) => ed.logits[b] - ed.logits[a])[0]] - ed.logits[[...ed.logits.keys()].sort((a, b) => ed.logits[b] - ed.logits[a])[1]], 2)} logits`],
    ['', 'neuron 3468 (abstain) act', fmt(n.act1, 3), `original neuron here: ${fmt(n.old1, 3)}`],
    ['', 'neuron 3421 (route) act', fmt(n.act2, 3), `original neuron here: ${fmt(n.old2, 3)}`],
    ['', 'gate pre-activation', fmt(n.gate, 3), S.variant === 'rand13' ? 'k·g₁₃·x (random direction)' : S.variant === 'original' ? 'edit off' : `k·s·x ${fmt(n.comps.s, 2)} − λ term ${fmt(-n.comps.c, 2)} − λ₂ term ${fmt(-n.comps.c2, 2)}`],
    ['', 's·x / r·x', `${fmt(n.sx, 2)} / ${fmt(n.rx, 2)}`, 'abstain and route features at the decision position'],
    ['', 'c·x / c₂·x', `${fmt(n.cx, 2)} / ${fmt(n.c2x, 2)}`, 'tool-context and generated-position gate features'],
    ['', 'rms(h′) vs rms(h₂₄)', `${fmt(ed.rms, 3)} / ${fmt(base.rms, 3)}`, 'final RMSNorm denominator'],
    ['', '<tool_call> − NO', `${fmt(g(ed.logits, '<tool_call>') - g(ed.logits, 'NO'), 2)}`, `original: ${fmt(g(base.logits, '<tool_call>') - g(base.logits, 'NO'), 2)}`],
    ['', 'NE − NO', `${fmt(g(ed.logits, 'NE') - g(ed.logits, 'NO'), 2)}`, `original: ${fmt(g(base.logits, 'NE') - g(base.logits, 'NO'), 2)}`],
  ];
  $('#readouts').innerHTML = ro.map(([c, k, v, d]) => `<div class="ro ${c}"><div class="k">${esc(k)}</div><div class="v ${c ? '' : cls(parseFloat(v))}">${c ? v : esc(v)}</div><div class="d">${esc(d)}</div></div>`).join('');
}
function renderTokens(it, ed, base, sm, smb, order, am) {
  const T = S.data.tracked, tb = $('#toktable tbody'); const rows = []; const shown = new Set();
  const pinned = ['<tool_call>', 'NO', 'NE', 'I', 'To', 'The', '<', '</', 'No', 'None'].map((w) => S.ids[w]).filter((x) => x !== undefined);
  const row = (i, cls) => { shown.add(i); const d = ed.logits[i] - base.logits[i]; return `<tr class="${cls}${T.ids[i] === am ? ' arg' : ''}"><td class="tok">${esc(JSON.stringify(T.text[i]).slice(1, -1))}</td><td class="num">${fmt(ed.logits[i], 2)}</td><td class="num">${d >= 0 ? '+' : ''}${fmt(d, 2)}</td><td class="num">${fmtP(sm.p[i])}</td><td class="barcell"><span class="bar" style="width:${(100 * sm.p[i]).toFixed(1)}%"></span></td></tr>`; };
  for (let r = 0; r < 10; r++) rows.push(row(order[r], 'top'));
  for (const id of pinned) { const i = S.tidx.get(id); if (i !== undefined && !shown.has(i)) rows.push(row(i, 'pinned')); }
  rows.push(`<tr class="tail"><td class="tok">all ${(S.data.constants.vocab - T.ids.length).toLocaleString()} other tokens</td><td class="num">—</td><td class="num">—</td><td class="num">${fmtP(sm.tail)}</td><td class="barcell"><span class="bar" style="width:${(100 * sm.tail).toFixed(1)}%;background:var(--muted)"></span></td></tr>`);
  tb.innerHTML = rows.join('');
  const tm = S.data.meta.checks.tail_mass; const errV = S.variant === 'original' ? 0 : (tm[S.variant] ? tm[S.variant].max_abs_prob_err_top10 : Math.max(...Object.values(tm).map((x) => x.max_abs_prob_err_top10)));
  $('#dist-hint').textContent = `Top 10 of the edited distribution plus pinned tokens; ${T.ids.length} tokens are tracked exactly, the other tokens keep their original logits (${S.variant === 'original' ? 'exact for the original' : `max top-10 probability error at the recorded ${VLABEL[S.variant]} settings over all 224 prompts: ${errV.toExponential(1)}; sliders moved away from them are not covered by that bound`}). Original top-3 here: ${it.decision.base_top.slice(0, 3).map(([id, l]) => `${JSON.stringify(T.text[S.tidx.get(id)] ?? id)} ${l.toFixed(1)}`).join(', ')}.`;
}
const fmtP = (p) => p >= 0.01 ? p.toFixed(3) : p >= 1e-4 ? p.toExponential(1) : p > 0 ? p.toExponential(0) : '0';

function renderSchematic(it, n, ed, base, am, amText) {
  const C = S.data.constants, v = S.variant, off = v === 'original';
  const col = (x, pos = 'var(--green)', neg = 'var(--red)') => x > 0 ? pos : x < 0 ? neg : 'var(--muted)';
  const [dc, dl] = decisionOf(am, amText);
  const lamTxt = v === 'rand13' ? 'random gate direction g₁₃' : `λ=${fmt(S.sl.lam, 2)} λ₂=${fmt(S.sl.lam2, 2)}`;
  const uAB = v === 'v1' ? 'u_AB (v1 readout)' : v === 'rand13' ? 'random u' : 'u_AB (NO, NE up; call, I, To, The down)';
  const uR = v === 'rand13' ? 'random u′' : 'u_route = u_NE − u_NO';
  const w = 640, h = 330;
  $('#schematic').innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Two-neuron circuit schematic with live values">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>
  <style>text{font:12px var(--sans);fill:var(--ink)} .k{fill:var(--muted);font-size:10.5px} .m{font-family:var(--mono)} .box{fill:var(--bg);stroke:var(--rule);rx:8} line,path.w{stroke:var(--muted);stroke-width:1.3;fill:none;marker-end:url(#arr)}</style>
  <rect class="box" x="10" y="20" width="150" height="290"/>
  <text x="85" y="40" text-anchor="middle" font-weight="600">layer-23 MLP input x</text>
  <text class="k" x="85" y="55" text-anchor="middle">post-attention RMSNorm, 1024-d</text>
  ${[['s·x', n.sx, 'abstain feature'], ['r·x', n.rx, 'route feature'], ['c·x', n.cx, 'tool-context'], ['c₂·x', n.c2x, 'generated-pos.']].map(([k, val, d], i) => `<text class="m" x="22" y="${90 + i * 46}">${k}</text><text class="m" x="150" y="${90 + i * 46}" text-anchor="end" font-weight="700" fill="${col(val, 'var(--ink)', 'var(--ink)')}">${fmt(val, 2)}</text><text class="k" x="22" y="${104 + i * 46}">${d}</text>`).join('')}
  <line x1="160" y1="110" x2="235" y2="110"/><line x1="160" y1="155" x2="235" y2="200"/><line x1="160" y1="200" x2="235" y2="115"/><line x1="160" y1="245" x2="235" y2="120"/>
  <rect class="box" x="238" y="70" width="150" height="90"/>
  <text x="313" y="90" text-anchor="middle" font-weight="600">shared gate</text>
  <text class="k" x="313" y="104" text-anchor="middle">k=${C.k} · ${esc(lamTxt)}</text>
  <text class="m" x="313" y="128" text-anchor="middle" font-size="16" font-weight="700" fill="${off ? 'var(--muted)' : col(n.gate)}">${off ? 'off' : fmt(n.gate, 2)}</text>
  <text class="k" x="313" y="148" text-anchor="middle">silu(gate) = ${off ? '—' : fmt(silu(n.gate), 3)}</text>
  <rect class="box" x="238" y="190" width="150" height="56"/>
  <text x="313" y="210" text-anchor="middle">up 3468 = k·s·x <tspan class="m" font-weight="700">${off ? '—' : fmt(C.k * n.sx, 2)}</tspan></text>
  <text x="313" y="232" text-anchor="middle">up 3421 = k<tspan baseline-shift="sub" font-size="9">r</tspan>·r·x <tspan class="m" font-weight="700">${off ? '—' : fmt(C.kr * n.rx, 2)}</tspan></text>
  <line x1="388" y1="115" x2="440" y2="115"/><line x1="388" y1="218" x2="440" y2="130"/><line x1="388" y1="130" x2="440" y2="215"/><line x1="388" y1="232" x2="440" y2="230"/>
  <rect x="442" y="92" width="110" height="46" rx="23" fill="var(--teal-soft)" stroke="var(--teal)"/>
  <text x="497" y="110" text-anchor="middle" font-weight="600" fill="var(--teal)">neuron 3468</text><text class="m" x="497" y="128" text-anchor="middle" font-weight="700">${fmt(n.act1, 3)}</text>
  <rect x="442" y="200" width="110" height="46" rx="23" fill="var(--amber-soft)" stroke="var(--amber)"/>
  <text x="497" y="218" text-anchor="middle" font-weight="600" fill="var(--amber)">neuron 3421</text><text class="m" x="497" y="236" text-anchor="middle" font-weight="700">${fmt(n.act2, 3)}</text>
  <text class="k" x="497" y="152" text-anchor="middle">× a₁ = ${fmt(S.sl.a1, 2)}</text><text class="k" x="497" y="166" text-anchor="middle">${esc(uAB)}</text>
  <text class="k" x="497" y="260" text-anchor="middle">× a₂ = ${fmt(S.sl.a2, 2)}</text><text class="k" x="497" y="274" text-anchor="middle">${esc(uR)}</text>
  <path class="w" d="M552 115 C 600 115, 600 165, 612 165"/><path class="w" d="M552 223 C 600 223, 600 175, 612 175"/>
  <text x="${w - 8}" y="150" text-anchor="end" font-weight="600">+ h₂₄ → RMSNorm → E<tspan baseline-shift="super" font-size="9">T</tspan></text>
  <text class="k" x="${w - 8}" y="166" text-anchor="end">rms ${fmt(base.rms, 3)} → ${fmt(ed.rms, 3)}</text>
  <text x="${w - 8}" y="195" text-anchor="end" class="m" font-weight="700" font-size="13">${esc(dl)}</text>
  <text class="k" x="${w - 8}" y="212" text-anchor="end">original first token: ${esc(JSON.stringify(S.data.tracked.text[S.tidx.get(it.decision.base_top[0][0])] ?? '?'))}</text>
  <text class="k" x="22" y="300">original (dead) neurons here: ${fmt(n.old1, 3)} / ${fmt(n.old2, 3)}, removed from h₂₄ when the edit is on</text>
  </svg>`;
}

function renderTrace(it) {
  const P = it.positions, n = P.tokens.length, W = Math.max(600, n * 14 + 60), H = 230, left = 44, top = 14, plotH = 150;
  const vals = []; let amax = 1e-6, gmax = 1e-6;
  for (let j = 0; j < n; j++) { const r = neuron(it, j); vals.push(r); amax = Math.max(amax, Math.abs(r.act1), Math.abs(r.act2)); if (Number.isFinite(r.gate)) gmax = Math.max(gmax, Math.abs(r.gate)); }
  const scale = Math.max(amax, 1); const y0 = top + plotH / 2, sy = (plotH / 2 - 4) / scale, bw = (W - left - 10) / n;
  const yv = (v) => y0 - v * sy;
  let bars = '', gate = '', labels = '';
  for (let j = 0; j < n; j++) {
    const x = left + j * bw, r = vals[j];
    bars += `<rect x="${x + 1}" y="${Math.min(y0, yv(r.act1))}" width="${Math.max(1, bw / 2 - 1)}" height="${Math.abs(yv(r.act1) - y0)}" fill="var(--teal)"><title>pos ${j} ${JSON.stringify(P.text[j])}: act3468 ${r.act1.toFixed(3)} act3421 ${r.act2.toFixed(3)} gate ${Number.isFinite(r.gate) ? r.gate.toFixed(2) : '—'}</title></rect>`;
    bars += `<rect x="${x + bw / 2}" y="${Math.min(y0, yv(r.act2))}" width="${Math.max(1, bw / 2 - 1)}" height="${Math.abs(yv(r.act2) - y0)}" fill="var(--amber)"><title>pos ${j} ${JSON.stringify(P.text[j])}: act3421 ${r.act2.toFixed(3)}</title></rect>`;
    if (Number.isFinite(r.gate)) gate += `${j ? 'L' : 'M'}${(x + bw / 2).toFixed(1)} ${(y0 - r.gate / Math.max(gmax, 1) * (plotH / 2 - 4)).toFixed(1)} `;
    const t = P.text[j].replace(/\n/g, '↵'); const hot = Math.abs(r.act1) > 1 || Math.abs(r.act2) > 1;
    if (n <= 64 || j % 2 === 0) labels += `<text x="${x + bw / 2}" y="${top + plotH + 14}" transform="rotate(-60 ${x + bw / 2} ${top + plotH + 14})" font-size="${n > 64 ? 8 : 9.5}" text-anchor="end" fill="${hot ? 'var(--red)' : 'var(--muted)'}" font-family="var(--mono)">${esc(t.length > 9 ? t.slice(0, 8) + '…' : t)}</text>`;
  }
  const hot = vals.filter((r) => Math.abs(r.act1) > 1 || Math.abs(r.act2) > 1).length;
  $('#trace').innerHTML = `<p class="hint">${n} generated positions · max |act| ${fmt(amax, 2)} · ${hot} position${hot === 1 ? '' : 's'} with |act| &gt; 1 (red labels) · position 0 is the decision token. Gate line scaled to ±${fmt(Math.max(gmax, 1), 1)}.</p>
  <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Per-position neuron activations">
  <line x1="${left}" y1="${y0}" x2="${W - 8}" y2="${y0}" stroke="var(--rule)"/>
  <text x="4" y="${yv(scale) + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">+${fmt(scale, 1)}</text><text x="4" y="${yv(-scale) + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">−${fmt(scale, 1)}</text><text x="4" y="${y0 + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">0</text>
  ${bars}<path d="${gate}" fill="none" stroke="var(--ink)" stroke-width="1.2" opacity=".7"/>${labels}</svg>`;
}

function renderProvenance() {
  const m = S.data.meta, c = m.checks, ag = c.argmax_agreement || {};
  const li = [];
  li.push(`${m.n_items} prompts · base <code>${esc(m.base_model)}</code> @ <code>${esc(m.base_revision.slice(0, 7))}</code>`);
  if (ag.original) li.push(`original first token reproduced by the exact readout: ${ag.original.match}/${ag.original.n}`);
  for (const v of ['g23v3', 'v2', 'v1', 'rand13']) if (ag[v]) li.push(`${VLABEL[v]} recorded first token predicted at default sliders: ${ag[v].match}/${ag[v].n}`);
  const ti = c.original_generation_token_identical || {}; li.push(`fresh original generations token-identical to the recorded runs: ${Object.entries(ti).map(([k, v]) => `${k} ${v}`).join(', ')}`);
  li.push(`closed-form readout vs direct recomputation: max |Δlogit| ${(+c.formula_vs_direct_max_abs).toExponential(1)}`);
  if (c.fresh_vs_p03_decision_logits_max_abs != null) li.push(`fresh vs historical (fp16) decision logits, practice: max |Δ| ${fmt(c.fresh_vs_p03_decision_logits_max_abs, 2)} (BF16 run-to-run noise)`);
  $('#provenance').innerHTML = `<ul>${li.map((x) => `<li>${x}</li>`).join('')}</ul><p>Generated ${esc(m.generated_utc)}. ${esc(m.provenance)}</p>`;
}

// ------------------------------------------------------------------ tabs
function setupTabs() { $$('.tab').forEach((b) => b.onclick = () => switchTab(b.dataset.tab)); }
function switchTab(name) {
  $$('.tab').forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); });
  $('#tab-explore').hidden = name !== 'explore'; $('#tab-live').hidden = name !== 'live';
  history.replaceState(null, '', name === 'live' ? '#live' : location.pathname + location.search);
  if (name === 'live') checkGPU();
}

// ------------------------------------------------------------------ Mode 2: WebGPU runner (UI side; model work lives in worker.js)
const L = { worker: null, gpu: null, loaded: new Set(), running: false, t: {} };
function log(msg) { const el = $('#live-log'); el.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`; el.scrollTop = el.scrollHeight; }
async function checkGPU() {
  if (L.gpu !== null) return L.gpu;
  const el = $('#gpu-status');
  if (!('gpu' in navigator)) { L.gpu = false; el.className = 'notice bad'; el.innerHTML = 'This browser does not expose <code>navigator.gpu</code>. The live runner needs WebGPU (Chrome/Edge 113+, Safari 26+, Firefox 141+ on supported hardware). Mode 1 above works everywhere and needs no download.'; return false; }
  try {
    const ad = await navigator.gpu.requestAdapter(); if (!ad) throw new Error('requestAdapter() returned null');
    const info = ad.info || {}; const f16 = ad.features.has('shader-f16');
    L.gpu = { f16, vendor: info.vendor, arch: info.architecture };
    el.className = 'notice ok'; el.innerHTML = `WebGPU available${info.vendor ? ` · ${esc(info.vendor)}${info.architecture ? ' / ' + esc(info.architecture) : ''}` : ''} · shader-f16 ${f16 ? 'yes' : 'no (fp16 kernels will be emulated; expect slower generation)'} · max buffer ${(ad.limits.maxBufferSize / 2 ** 30).toFixed(1)} GiB.`;
    $('#btn-load').disabled = false;
  } catch (e) { L.gpu = false; el.className = 'notice bad'; el.innerHTML = `WebGPU adapter unavailable (${esc(e.message || e)}). Use Mode 1, which needs no GPU and no download.`; }
  return L.gpu;
}
function setupLive() {
  $('#live-system').value = S.cur?.system ?? ''; $('#live-tools').value = JSON.stringify(S.cur?.tools ?? [], null, 1); $('#live-user').value = S.cur?.user ?? '';
  $('#btn-use-item').onclick = () => { if (!S.cur) return; $('#live-system').value = S.cur.system; $('#live-tools').value = JSON.stringify(S.cur.tools, null, 1); $('#live-user').value = S.cur.user; log(`prompt set from ${S.cur.id}`); };
  $('#btn-load').onclick = loadModels; $('#btn-run').onclick = runBoth; $('#btn-stop').onclick = () => { L.worker?.postMessage({ type: 'stop' }); log('stop requested'); };
}
function ensureWorker() {
  if (L.worker) return L.worker;
  const w = new Worker('worker.js', { type: 'module' }); L.worker = w;
  w.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'log') log(m.text);
    else if (m.type === 'progress') progress(m);
    else if (m.type === 'loaded') { L.loaded.add(m.variant); log(`${m.variant} ready in ${(m.ms / 1000).toFixed(1)} s${m.verified === true ? ' · data file SHA-256 verified' : m.verified === false ? ' · WARNING: data file hash mismatch' : ''}`); $('#btn-run').disabled = false; $('#btn-load').disabled = false; $('#btn-load').textContent = 'Reload models'; }
    else if (m.type === 'token') { const g = $('#gen-' + m.variant); g.querySelector('.out').textContent += m.text; const st = L.t[m.variant]; st.n = m.n; st.last = performance.now(); if (!st.first) st.first = st.last; stats(m.variant); }
    else if (m.type === 'first_logits') { const g = $('#gen-' + m.variant); g.querySelector('.firstlogits').innerHTML = `<span>first-token logits:</span>${Object.entries(m.named).map(([k, v]) => `<span>${esc(k)} <b>${v.toFixed(2)}</b></span>`).join('')}<span>· top: ${m.top.map(([t, v]) => `${esc(JSON.stringify(t))} ${v.toFixed(1)}`).join(', ')}</span>`; }
    else if (m.type === 'done') { const st = L.t[m.variant]; st.done = true; st.prefill = m.prefillMs; st.total = m.genMs; st.n = m.nTokens; stats(m.variant); log(`${m.variant}: ${m.nTokens} tokens, prefill ${m.prefillMs.toFixed(0)} ms, ${(m.nTokens / Math.max(m.genMs, 1) * 1000).toFixed(1)} tok/s${m.stopped ? ' (stopped)' : ''}`); }
    else if (m.type === 'error') { log('ERROR ' + m.text); $('#btn-load').disabled = false; $('#btn-run').disabled = L.loaded.size === 0; $('#btn-stop').disabled = true; L.running = false; }
    else if (m.type === 'all_done') { L.running = false; $('#btn-run').disabled = false; $('#btn-stop').disabled = true; }
  };
  w.onerror = (e) => log('worker error: ' + (e.message || e));
  return w;
}
function progress(m) {
  const box = $('#load-progress'); const id = 'pb-' + m.file.replace(/[^a-z0-9]/gi, '_'); let el = document.getElementById(id);
  if (!el) { el = document.createElement('div'); el.id = id; el.className = 'pbar'; el.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`; box.appendChild(el); }
  el.querySelector('.name').textContent = `${m.variant ? m.variant + ' · ' : ''}${m.file}${m.status ? ' · ' + m.status : ''}`;
  const pct = m.total ? Math.min(100, 100 * m.loaded / m.total) : (m.status === 'done' ? 100 : 0);
  el.querySelector('.fill').style.width = pct + '%'; el.querySelector('.pct').textContent = m.total ? `${(m.loaded / 2 ** 20).toFixed(0)}/${(m.total / 2 ** 20).toFixed(0)} MB` : '';
}
function loadModels() {
  const w = ensureWorker(); $('#load-progress').innerHTML = ''; $('#btn-load').disabled = true; $('#btn-run').disabled = true; L.loaded.clear();
  const src = $('#model-source').value.trim(); log(`loading from ${src}`);
  w.postMessage({ type: 'load', source: src, keepBoth: $('#keep-both').checked, f16: !!(L.gpu && L.gpu.f16) });
}
function runBoth() {
  let tools; try { tools = JSON.parse($('#live-tools').value || '[]'); if (!Array.isArray(tools)) throw new Error('tools must be a JSON array'); } catch (e) { log('tools JSON invalid: ' + e.message); return; }
  const w = ensureWorker(); L.running = true; $('#btn-run').disabled = true; $('#btn-stop').disabled = false;
  for (const v of ['original', 'edited']) { const g = $('#gen-' + v); g.querySelector('.out').textContent = ''; g.querySelector('.firstlogits').textContent = ''; g.querySelector('.stats').textContent = '…'; L.t[v] = { n: 0, first: 0, last: 0, start: performance.now() }; }
  w.postMessage({ type: 'generate', system: $('#live-system').value, user: $('#live-user').value, tools, maxNewTokens: Math.max(1, Math.min(512, +$('#live-max').value || 128)), keepBoth: $('#keep-both').checked });
}
function stats(v) {
  const st = L.t[v], g = $('#gen-' + v); if (!st) return;
  const tps = st.n > 1 && st.last > st.first ? (st.n - 1) / ((st.last - st.first) / 1000) : 0;
  g.querySelector('.stats').textContent = st.done ? `${st.n} tok · prefill ${st.prefill.toFixed(0)} ms · ${(st.n / Math.max(st.total, 1) * 1000).toFixed(1)} tok/s` : `${st.n} tok · ${tps.toFixed(1)} tok/s`;
}
