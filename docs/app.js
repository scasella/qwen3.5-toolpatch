// Four-Neuron Circuit Explorer — Tab 1 (exact closed-form readout of g23v3 / g23v4) and Tab 2 (WebGPU runner UI; model work in worker.js).
// Tab 1 math. Everything the two edits add to the final residual h24 lives in a fixed 10-vector basis b_k (the original down columns of
// J1/J2/J3, u_AB, u_route, v1's u_AB, u_conv, the random-control output vectors). With coef_k the weight on b_k,
//   logit_t = (L_t*rms + sum_k coef_k B_kt) / sqrt((sumsq + 2 sum_k coef_k hb_k + coef^T G coef)/d + eps)
// where L_t are the logits of the model with the ORIGINAL layer-23 MLP at the decision position for the current stage-1 state
// (the layer-23 input with the clause-end marker b * act_J0 * p6 added, replayed exactly on the CPU for b in {0, 1/2, 1, 2} and for the
// random control), B_kt = (b_k*gamma).E_t, hb_k = h24.b_k, G = Gram(b). Head-6 attention to the clause end is an exact closed form in the
// write magnitude w (six stored scalars per head). All of it was checked against direct recomputation in scripts/export_circuit_traces.py.
// The math block (stateName … writeAtEnd) is unchanged by the UI revamp; `window.__explorer.selfTest()` reproduces the recorded first
// tokens (original 143/143, g23v3 142/143, g23v4 143/143, v1 65/66, v2 76/76, random 33/33, random-control 64/64).

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : (Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const silu = (z) => z / (1 + Math.exp(-z));
const fmtP = (p) => p >= 0.01 ? p.toFixed(3) : p >= 1e-4 ? p.toExponential(1) : p > 0 ? p.toExponential(0) : '0';

const VARIANTS = ['original', 'v1', 'v2', 'g23v3', 'g23v4', 'rand13', 'rand17', 'conv3', 'conv4'];
const VLABEL = { original: 'Original', v1: 'v1', v2: 'v2', g23v3: 'g23v3', g23v4: 'g23v4', rand13: 'Random J1/J2', rand17: 'Random J0/J3', conv3: 'g23v3 conversational', conv4: 'g23v4 conversational' };
const VGROUPS = [['Edits', ['original', 'v1', 'v2', 'g23v3', 'g23v4']], ['Controls', ['rand13', 'rand17']], ['Conversational readout', ['conv3', 'conv4']]];
const OUT_PRIMARY = ['original', 'g23v3', 'g23v4'], OUT_EXTRA = ['v1', 'v2', 'rand13'];
const CTRL_VARIANTS = [['ctrl_half', 'Half dose (a3 = 0.05, b = 0.5)'], ['ctrl_double', 'Double dose (a3 = 0.2, b = 2)'], ['ctrl_reversed', 'Reversed (a3 = −0.1)'], ['ctrl_rand17', 'Random norm-matched (seed 17)']];
const COHORTS = [['valid_call', 'Valid call'], ['no_tool', 'Explanation → NO_TOOL'], ['need_clarification', 'Missing value → NEED_CLARIFICATION'], ['ambiguous', 'Ambiguous "either X or Y"'], ['stress_ambiguous', 'Stress suite: unseen ambiguous phrasing'], ['stress_resolved', 'Stress suite: resolved two-value call'], ['prose', 'Non-tool prose QA (128 tokens)']];
const COHORT_ROW = { valid_call: 'call', no_tool: 'NO_TOOL', need_clarification: 'NEED_CLAR.', ambiguous: 'ambiguous', stress_ambiguous: 'stress · amb.', stress_resolved: 'stress · res.', prose: 'prose' };
const COHORT_CHIP = { valid_call: 'Valid call', no_tool: 'NO_TOOL', need_clarification: 'NEED_CLARIFICATION', ambiguous: 'Ambiguous', stress_ambiguous: 'Stress · ambiguous', stress_resolved: 'Stress · resolved', prose: 'Prose' };
const SETS = [['practice', 'Practice'], ['selection', 'Selection (unseen families)'], ['renamed_A', 'Renamed A'], ['renamed_B', 'Renamed B'], ['final', 'Final-split proxy'], ['stress', 'Ambiguity Stress Suite'], ['prose_qa', 'Prose QA'], ['prose_qa2', 'Prose QA 2']];
const SET_CHIP = { practice: 'Practice', selection: 'Selection', renamed_A: 'Renamed A', renamed_B: 'Renamed B', final: 'Final split', stress: 'Stress suite', prose_qa: 'Prose QA', prose_qa2: 'Prose QA 2' };
const DECISION_NAME = { call: '<tool_call> → tool call', answer: 'NO_TOOL', clarify: 'NEED_CLARIFICATION', prose: 'prose', null: '—' };
const SECTIONS = [['s-prompt', 'Prompt'], ['s-outputs', 'Recorded outputs'], ['s-circuit', 'The circuit, live'], ['s-tokens', 'Prompt tokens & head 6'], ['s-dist', 'First-token distribution'], ['s-trace', 'Along the generation'], ['s-provenance', 'Provenance']];
const X22 = { w1: 0, r1: 1, gJ0_old: 2, uJ0_old: 3, g22_r17: 4, u22_r17: 5 };
const X23 = { s: 0, r: 1, c: 2, c2: 3, t: 4, r3: 5, gJ1_old: 6, uJ1_old: 7, gJ2_old: 8, uJ2_old: 9, gJ3_old: 10, uJ3_old: 11, s13: 12, r13: 13, g13: 14, g23_r17: 15, u23_r17: 16 };
const B = { dJ1_old: 0, dJ2_old: 1, dJ3_old: 2, uAB: 3, uroute: 4, uAB_v1: 5, uconv: 6, uAB_r13: 7, uroute_r13: 8, d23_r17: 9 };
const NB = 10;

const S = { data: null, items: [], filtered: [], cur: null, variant: 'g23v4', stage1: true, stage2: true, sl: { a1: 1, a2: 1, lam: 0.3, lam2: 0.5, b: 1, a3: 0.1, rho: 1 }, ids: {}, tidx: null, allTok: false, lastDec: undefined };

// ------------------------------------------------------------------ boot
init().catch((e) => { const l = $('#loading'); l.classList.add('bad'); l.innerHTML = `<b>Failed to load the data.</b> ${esc(e.message || e)}`; console.error(e); });
async function readWithProgress(r) {
  const el = $('#loading .msg'); if (!r.body || !r.body.getReader) return r.text();
  const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (el) el.textContent = `Loading the recorded runs and replay states… ${(got / 2 ** 20).toFixed(1)} MB received`; }
  const all = new Uint8Array(got); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
  if (el) el.textContent = 'Parsing…';
  return new TextDecoder().decode(all);
}
async function init() {
  const r = await fetch('data/circuit_traces.json'); if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  S.data = JSON.parse(await readWithProgress(r)); S.items = S.data.items;
  S.ids = S.data.tracked.named; S.tidx = new Map(S.data.tracked.ids.map((t, i) => [t, i]));
  buildFilters(); buildVariants(); buildSliders(); renderProvenance(); setupTabs(); setupOnPage(); setupBrowser();
  $('#ck-stage1').addEventListener('change', (e) => { S.stage1 = e.target.checked; recompute(); });
  $('#ck-stage2').addEventListener('change', (e) => { S.stage2 = e.target.checked; recompute(); });
  $('#ck-alltok').addEventListener('change', (e) => { S.allTok = e.target.checked; renderHeat(S.cur); });
  const q = new URLSearchParams(location.search);
  applyFilters();
  selectItem(S.items.find((it) => it.id === q.get('item')) || S.items.find((it) => it.id === 'discovery-611401-0239') || S.items[0]);
  $('#loading').hidden = true; $('#explorer').hidden = false;
  if (location.hash === '#live') switchTab('live');
  setupLive();
  if (q.has('selftest')) { const res = selfTest(); console.log('selftest', res); $('#provenance').insertAdjacentHTML('beforeend', `<p><b>Self-test (recorded first token reproduced by this page's readout):</b> ${esc(Object.entries(res).map(([k, v]) => `${k} ${v}`).join(' · '))}</p>`); $('#s-provenance').open = true; }
}

// ------------------------------------------------------------------ filters / list
function buildFilters() {
  const fc = $('#filter-cohort'), fs = $('#filter-set');
  const cnt = (key, k) => S.items.filter((it) => it[key] === k).length;
  for (const [k, lab] of COHORTS) fc.insertAdjacentHTML('beforeend', `<label title="${esc(lab)}"><input type="checkbox" data-cohort="${k}" checked><span>${esc(COHORT_CHIP[k])}</span><span class="cnt">${cnt('cohort', k)}</span></label>`);
  for (const [k, lab] of SETS) fs.insertAdjacentHTML('beforeend', `<label title="${esc(lab)}"><input type="checkbox" data-set="${k}" checked><span>${esc(SET_CHIP[k])}</span><span class="cnt">${cnt('set', k)}</span></label>`);
  fc.addEventListener('change', applyFilters); fs.addEventListener('change', applyFilters);
  $('#search').addEventListener('input', applyFilters);
  $('#search-clear').onclick = () => { $('#search').value = ''; applyFilters(); $('#search').focus(); };
  $('#filters-all').onclick = () => { const all = $$('#filter-cohort input, #filter-set input'); const every = all.every((i) => i.checked); all.forEach((i) => { i.checked = !every; }); applyFilters(); };
}
const jr = (cls, d) => `<span class="j ${cls}">${esc(shortDec(d))}</span>`;
function applyFilters() {
  const coh = new Set($$('#filter-cohort input:checked').map((i) => i.dataset.cohort));
  const sets = new Set($$('#filter-set input:checked').map((i) => i.dataset.set));
  const q = $('#search').value.trim().toLowerCase(); $('#search-clear').hidden = !q;
  S.filtered = S.items.filter((it) => coh.has(it.cohort) && sets.has(it.set) && (!q || (it.id + ' ' + it.family + ' ' + it.user).toLowerCase().includes(q)));
  const ul = $('#itemlist'); ul.innerHTML = '';
  for (const it of S.filtered) {
    const o = it.outputs, d0 = o.original?.decision, d3 = o.g23v3?.decision, d4 = o.g23v4?.decision;
    const li = document.createElement('li'); li.role = 'option'; li.dataset.id = it.id; const sel = S.cur && S.cur.id === it.id;
    li.className = sel ? 'sel' : ''; li.tabIndex = sel ? 0 : -1; li.setAttribute('aria-selected', sel);
    li.innerHTML = `<span class="id" title="${esc(it.id)}">${esc(it.id)}</span><span class="chip cohort ${it.cohort}" title="${esc(cohortLabel(it.cohort))}">${esc(cohortShort(it))}</span>
      <span class="fam"><span>${esc(it.family)} · ${esc(it.set)}</span><span class="journey" title="first-token decision: Original → g23v3 → g23v4">${jr('base', d0)}<span class="sep">→</span>${jr('g23v3', d3)}<span class="sep">→</span>${jr('g23v4', d4)}${o.g23v4 ? `<b class="${o.g23v4.correct ? 'ok' : 'bad'}">${o.g23v4.correct ? '✓' : '✗'}</b>` : ''}</span></span>`;
    li.onclick = () => selectItem(it, true); ul.appendChild(li);
  }
  if (S.filtered.length === 0) ul.innerHTML = '<li class="empty" style="color:var(--muted);cursor:default;display:block;padding:14px 8px">No prompt matches these filters.</li>';
  $('#count').textContent = `${S.filtered.length} of ${S.items.length}`;
}
const cohortShort = (it) => COHORT_ROW[it.cohort] ?? it.cohort;
const shortDec = (d) => ({ call: 'call', answer: 'NO_TOOL', clarify: 'NE', prose: 'prose' })[d] ?? '—';
const cohortLabel = (c) => (COHORTS.find((x) => x[0] === c) || [c, c])[1];
function setupBrowser() {
  const ul = $('#itemlist');
  ul.addEventListener('keydown', (e) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key) || !S.filtered.length) return;
    e.preventDefault(); const i = Math.max(0, S.filtered.findIndex((it) => S.cur && it.id === S.cur.id));
    const j = e.key === 'Home' ? 0 : e.key === 'End' ? S.filtered.length - 1 : Math.min(S.filtered.length - 1, Math.max(0, i + (e.key === 'ArrowDown' ? 1 : -1)));
    selectItem(S.filtered[j]); $(`#itemlist li[data-id="${CSS.escape(S.filtered[j].id)}"]`)?.focus();
  });
  document.addEventListener('keydown', (e) => { if (e.key === '/' && !/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName) && !$('#tab-explore').hidden) { e.preventDefault(); $('#search').focus(); } });
  const tg = $('#browser-toggle'), br = $('#browser');
  tg.onclick = () => { const open = br.classList.toggle('open'); tg.setAttribute('aria-expanded', open); tg.textContent = open ? 'Close list' : 'Change prompt'; if (open) $('#search').focus(); };
}
function closeBrowserOnMobile() { if (window.matchMedia('(max-width: 960px)').matches) { const br = $('#browser'), tg = $('#browser-toggle'); br.classList.remove('open'); tg.setAttribute('aria-expanded', 'false'); tg.textContent = 'Change prompt'; } }

// ------------------------------------------------------------------ item
function selectItem(it, fromClick = false) {
  if (!it) return; S.cur = it;
  $$('#itemlist li').forEach((li) => { const on = li.dataset.id === it.id; li.classList.toggle('sel', on); li.tabIndex = on ? 0 : -1; li.setAttribute('aria-selected', on); if (on) { const ul = li.parentElement; if (li.offsetTop < ul.scrollTop || li.offsetTop + li.offsetHeight > ul.scrollTop + ul.clientHeight) ul.scrollTop = li.offsetTop - ul.clientHeight / 2 + li.offsetHeight / 2; } });
  const u = new URL(location); u.searchParams.set('item', it.id); history.replaceState(null, '', u);
  $('#item-title').textContent = it.id; $('#mobile-cur').textContent = it.id;
  $('#item-badges').innerHTML = `<span class="chip cohort ${it.cohort}">${esc(cohortLabel(it.cohort))}</span><span class="chip">${esc(it.set)}</span><span class="chip">${esc(it.family)}</span>${it.renamed ? '<span class="chip">renamed identifiers</span>' : ''}<span class="chip">${it.T} prompt tokens</span>`;
  $('#p-system').textContent = it.system; $('#p-user').textContent = it.user; $('#p-expected').textContent = it.expected;
  $('#p-tools').textContent = it.tools.length ? JSON.stringify(it.tools, null, 1) : '(no tools in this prompt)';
  $('#p-tools-summary').textContent = it.tools.length ? it.tools.map((t) => t.function.name).join(', ') + ` (${it.tools.length} function${it.tools.length === 1 ? '' : 's'}, click to expand)` : 'no tools';
  renderOutputs(it); recompute(); renderHeat(it);
  if (fromClick) closeBrowserOnMobile();
}
function outCol(it, v, label, o, note) {
  const col = document.createElement('div'); col.className = 'out-col ' + v + (o ? '' : ' missing');
  if (!o) { col.innerHTML = `<h3><span class="nm">${esc(label)}</span></h3><pre>not run on this item in the campaigns</pre><div class="meta">${esc(note)}</div>`; return col; }
  const ok = o.correct ? 'ok' : 'bad'; const same = (ref) => ref && ref.token_ids && JSON.stringify(ref.token_ids) === JSON.stringify(o.token_ids);
  const verdict = it.cohort === 'prose' ? (o.correct ? 'intact prose' : 'corrupted') : (o.correct ? 'correct' : 'wrong');
  const ident = v === 'original' ? '' : same(it.outputs.original) ? 'token-identical to Original' : (same(it.outputs.g23v3) && v !== 'g23v3') ? 'token-identical to g23v3' : 'differs from Original';
  col.innerHTML = `<h3><span class="nm">${esc(label)}</span><span class="dec ${o.decision}">${esc(DECISION_NAME[o.decision] ?? o.decision)}</span></h3><pre>${esc(o.raw_output)}</pre>
    <div class="meta"><span class="verdict ${ok}">${o.correct ? '✓' : '✗'} ${verdict}${o.correct_strict && it.cohort !== 'prose' ? ' · strict' : ''}</span><span>${o.n_tokens} tok${o.truncated ? ' (cut at 128)' : ''}</span>${ident ? `<span>${ident}</span>` : ''}<span title="campaign pass" class="mono">${esc(o.run)}</span></div>`;
  return col;
}
function renderOutputs(it) {
  const box = $('#outputs'); box.innerHTML = '';
  const notes = { rand13: 'random controls (seed 13) were run on the practice set only', v1: 'v1 was dropped before the final-split pass', v2: 'v2 was not run on this set', g23v4: 'no recorded g23v4 pass', original: 'no recorded pass', g23v3: 'no recorded pass' };
  for (const v of OUT_PRIMARY) box.appendChild(outCol(it, v, VLABEL[v], it.outputs[v], notes[v]));
  const ex = $('#outputs-extra'); ex.innerHTML = ''; let anyX = false;
  for (const v of OUT_EXTRA) { if (!it.outputs[v] && it.set === 'stress') continue; anyX = true; ex.appendChild(outCol(it, v, VLABEL[v], it.outputs[v], notes[v])); }
  $('#outputs-more').hidden = !anyX;
  const cb = $('#outputs-ctrl'); cb.innerHTML = ''; let any = false;
  for (const [v, lab] of CTRL_VARIANTS) { if (it.outputs[v]) { any = true; cb.appendChild(outCol(it, v, lab, it.outputs[v], '')); } }
  cb.parentElement.hidden = !any;
}

// ------------------------------------------------------------------ variants & sliders
function buildVariants() {
  const box = $('#variants'); box.innerHTML = '';
  for (const [title, vs] of VGROUPS) {
    const g = document.createElement('div'); g.className = 'pgroup'; g.innerHTML = `<span class="pl">${esc(title)}</span>`;
    for (const v of vs) { const b = document.createElement('button'); b.type = 'button'; b.textContent = VLABEL[v]; b.dataset.v = v; b.className = v; b.title = S.data.variants[v].label; b.onclick = () => setVariant(v); g.appendChild(b); }
    box.appendChild(g);
  }
  setVariant('g23v4', false);
}
function setVariant(v, re = true) {
  S.variant = v; $$('#variants button').forEach((b) => { b.classList.toggle('active', b.dataset.v === v); b.setAttribute('aria-pressed', b.dataset.v === v); });
  $('#variant-desc').textContent = S.data.variants[v].label;
  const C3 = S.data.constants.g23v3, C4 = S.data.constants.g23v4, K = S.data.constants;
  const presets = {
    original: { stage1: false, stage2: false, sl: { a1: 0, a2: 0, a3: 0 } },
    v1: { stage1: true, stage2: false, sl: { a1: 1, a2: 1, lam: 0, lam2: 0, a3: 0 } },
    v2: { stage1: true, stage2: false, sl: { a1: 1, a2: 1, lam: C3.lam, lam2: 0, a3: 0 } },
    g23v3: { stage1: true, stage2: false, sl: { a1: C3.a1, a2: C3.a2, lam: C3.lam, lam2: C3.lam2, a3: 0 } },
    g23v4: { stage1: true, stage2: true, sl: { a1: C3.a1, a2: C3.a2, lam: C4.lam, lam2: C4.lam2, b: 1, a3: C4.a3, rho: C4.rho } },
    rand13: { stage1: true, stage2: false, sl: { a1: 1, a2: 1, a3: 0 } },
    rand17: { stage1: true, stage2: true, sl: { a1: C3.a1, a2: C3.a2, lam: C3.lam, lam2: C3.lam2, b: 1, a3: 1, rho: 0 } },
    conv3: { stage1: true, stage2: false, sl: { a1: K.conv_a1, a2: 0, lam: C3.lam, lam2: C3.lam2, a3: 0 } },
    conv4: { stage1: true, stage2: true, sl: { a1: K.conv_a1, a2: 0, lam: C4.lam, lam2: C4.lam2, b: 1, a3: K.conv_a3, rho: 0 } },
  }[v];
  S.stage1 = presets.stage1; S.stage2 = presets.stage2; S.sl = { ...S.sl, ...presets.sl };
  $('#ck-stage1').checked = S.stage1; $('#ck-stage2').checked = S.stage2;
  const gateFree = v === 'original' || v === 'rand13';
  for (const k of ['lam', 'lam2']) $('#sl-' + k).disabled = gateFree;
  for (const k of ['a1', 'a2']) $('#sl-' + k).disabled = v === 'original';
  for (const k of ['b', 'a3', 'rho']) $('#sl-' + k).disabled = !['g23v4', 'conv4', 'rand17'].includes(v);
  $('#sl-rho').disabled = $('#sl-rho').disabled || v === 'rand17' || v === 'conv4';
  syncSliders(); if (re) recompute();
}
const DOSE_STEPS = [0, 0.5, 1, 2];
function buildSliders() {
  for (const k of ['a1', 'a2', 'lam', 'lam2', 'a3', 'rho']) $('#sl-' + k).addEventListener('input', (e) => { S.sl[k] = +e.target.value; $('#o-' + k).value = fmt(S.sl[k], k === 'a3' ? 3 : 2); recompute(); });
  $('#sl-b').addEventListener('input', (e) => { S.sl.b = DOSE_STEPS[+e.target.value]; $('#o-b').value = fmt(S.sl.b, 1); recompute(); });
  syncSliders();
}
function syncSliders() {
  for (const k of ['a1', 'a2', 'lam', 'lam2', 'a3', 'rho']) { $('#sl-' + k).value = S.sl[k]; $('#o-' + k).value = fmt(S.sl[k], k === 'a3' ? 3 : 2); }
  $('#sl-b').value = Math.max(0, DOSE_STEPS.indexOf(S.sl.b)); $('#o-b').value = fmt(S.sl.b, 1);
}

// ------------------------------------------------------------------ the math (unchanged by the UI revamp)
function stateName() {
  if (!S.stage2) return 'b0';
  if (S.variant === 'rand17') return 'rand17';
  return 'b' + (S.sl.b === 0.5 ? '0.5' : String(S.sl.b));
}
function j0At(x, variant = S.variant) {
  // x = x22 projections [w1, r1, gJ0_old, uJ0_old, g22_r17, u22_r17]
  const C4 = S.data.constants.g23v4; const old = silu(x[X22.gJ0_old]) * x[X22.uJ0_old];
  if (!S.stage2) return { act: 0, z: NaN, old };
  if (variant === 'rand17') { const z = x[X22.g22_r17]; return { act: silu(z) * x[X22.u22_r17], z, old }; }
  const z = C4.k1 * (x[X22.w1] - C4.theta1 * x[X22.r1]); return { act: silu(z) * z, z, old };
}
function circuit(x, variant = S.variant, sl = S.sl, st1 = S.stage1, st2 = S.stage2) {
  // x = x23 projections; returns activations and the coefficient vector over the basis
  const C3 = S.data.constants.g23v3, C4 = S.data.constants.g23v4, k = C3.k, kr = C3.kr, k3 = C4.k3;
  const old1 = silu(x[X23.gJ1_old]) * x[X23.uJ1_old], old2 = silu(x[X23.gJ2_old]) * x[X23.uJ2_old], old3 = silu(x[X23.gJ3_old]) * x[X23.uJ3_old];
  const coef = new Array(NB).fill(0); const r = { gate: NaN, act1: 0, act2: 0, gate3: NaN, act3: 0, old1, old2, old3, comps: { s: NaN, c: NaN, c2: NaN }, coef, sx: x[X23.s], rx: x[X23.r], cx: x[X23.c], c2x: x[X23.c2], tx: x[X23.t], r3x: x[X23.r3] };
  if (variant === 'original' || (!st1 && !st2)) return r;
  if (st1) {
    coef[B.dJ1_old] = -old1; coef[B.dJ2_old] = -old2;
    if (variant === 'rand13') { const gate = k * x[X23.g13]; r.gate = gate; r.act1 = silu(gate) * k * x[X23.s13]; r.act2 = silu(gate) * kr * x[X23.r13]; coef[B.uAB_r13] = sl.a1 * r.act1; coef[B.uroute_r13] = sl.a2 * r.act2; r.comps = { s: gate, c: 0, c2: 0 }; }
    else {
      const gs = k * x[X23.s], gc = -k * sl.lam * x[X23.c], gc2 = -k * sl.lam2 * x[X23.c2]; const gate = gs + gc + gc2; r.gate = gate; r.comps = { s: gs, c: gc, c2: gc2 };
      r.act1 = silu(gate) * k * x[X23.s]; r.act2 = silu(gate) * kr * x[X23.r];
      if (variant === 'v1') { coef[B.uAB_v1] = sl.a1 * r.act1; coef[B.uroute] = sl.a2 * r.act2; }
      else if (variant === 'conv3' || variant === 'conv4') { coef[B.uconv] = sl.a1 * r.act1; coef[B.uroute] = sl.a2 * r.act2; }
      else { coef[B.uAB] = sl.a1 * r.act1; coef[B.uroute] = sl.a2 * r.act2; }
    }
  }
  if (st2 && ['g23v4', 'conv4', 'rand17'].includes(variant)) {
    coef[B.dJ3_old] = -old3;
    if (variant === 'rand17') { const g3 = x[X23.g23_r17]; r.gate3 = g3; r.act3 = silu(g3) * x[X23.u23_r17]; coef[B.d23_r17] = sl.a3 * r.act3; }
    else {
      const g3 = k3 * (x[X23.t] - sl.lam * x[X23.c] - sl.lam2 * x[X23.c2] - C4.theta3 * x[X23.r3]); const up3 = k3 * (x[X23.t] - C4.theta3 * x[X23.r3]);
      r.gate3 = g3; r.act3 = silu(g3) * up3;
      if (variant === 'conv4') coef[B.uconv] += sl.a3 * r.act3; else { coef[B.uAB] += sl.a3 * r.act3; coef[B.uroute] += sl.a3 * sl.rho * r.act3; }
    }
  }
  return r;
}
function logitsFor(st, coef) {
  const Bk = S.data.tracked.B, G = S.data.gram, C = S.data.constants, K = st.logits.length;
  let r2 = st.sumsq;
  for (let k = 0; k < NB; k++) { if (coef[k] === 0) continue; r2 += 2 * coef[k] * st.hb[k]; for (let l = 0; l < NB; l++) r2 += coef[k] * G[k][l] * coef[l]; }
  const rn = Math.sqrt(r2 / C.d + C.eps), out = new Float64Array(K);
  for (let t = 0; t < K; t++) { let num = st.logits[t] * st.rms; for (let k = 0; k < NB; k++) if (coef[k] !== 0) num += coef[k] * Bk[k][t]; out[t] = num / rn; }
  return { logits: out, rms: rn };
}
function softmaxWithTail(st, logits) {
  let m = -Infinity; for (const v of logits) if (v > m) m = v;
  let sTr = 0; for (let t = 0; t < st.logits.length; t++) sTr += Math.exp(st.logits[t] - st.lse);
  const tailLse = sTr < 1 ? st.lse + Math.log(1 - sTr) : -Infinity;
  const mm = Math.max(m, tailLse); let z = Math.exp(tailLse - mm); const e = new Float64Array(logits.length);
  for (let t = 0; t < logits.length; t++) { e[t] = Math.exp(logits[t] - mm); z += e[t]; }
  for (let t = 0; t < logits.length; t++) e[t] /= z;
  return { p: e, tail: Math.exp(tailLse - mm) / z };
}
function alphaFormula(H, w) {
  const C = S.data.constants; const l = (H.A + w * H.B) / Math.sqrt((H.y0sq + 2 * w * H.y01 + w * w * H.y1sq) / C.head_dim + C.eps) / Math.sqrt(C.head_dim);
  const m = Math.max(l, H.lse_rest); return Math.exp(l - m) / (Math.exp(l - m) + Math.exp(H.lse_rest - m));
}
function writeAtEnd(it) {
  // the marker magnitude at the clause end for the current settings, |p6| included
  const j = j0At(it.prompt.x22[it.end]); const bscale = S.variant === 'rand17' ? 1 : S.sl.b; return { w: bscale * j.act * S.data.constants.p6_norm, act: j.act, z: j.z };
}

// ------------------------------------------------------------------ render live circuit
function recompute() {
  const it = S.cur; if (!it) return;
  const sn = stateName(), st = it.states[sn], st0 = it.states.b0;
  const n = circuit(st.x23); const base = logitsFor(st0, new Array(NB).fill(0)); const ed = logitsFor(st, n.coef);
  const sm = softmaxWithTail(st, ed.logits), smb = softmaxWithTail(st0, base.logits);
  const order = [...ed.logits.keys()].sort((a, b) => ed.logits[b] - ed.logits[a]);
  const am = S.data.tracked.ids[order[0]], amText = S.data.tracked.text[order[0]];
  const wr = writeAtEnd(it); const alpha6 = S.stage2 ? st.alpha[6] : st0.alpha[6];
  renderSchematic(it, n, ed, base, am, amText, st, wr, alpha6);
  renderReadouts(it, n, ed, base, am, amText, sm.p[order[0]], st, wr, alpha6);
  renderTokens(it, ed, base, sm, smb, order, am, st, sn);
  renderEq(); renderTrace(it); renderStrip(it, wr, alpha6, st);
}
function decisionOf(tokId, text) { const N = S.ids; if (tokId === N['<tool_call>']) return ['call', '<tool_call> → tool call']; if (tokId === N['NO']) return ['answer', 'NO → NO_TOOL']; if (tokId === N['NE']) return ['clarify', 'NE → NEED_CLARIFICATION']; return ['prose', `“${text}” → prose`]; }
function renderEq() {
  const v = S.variant, conv = v === 'conv3' || v === 'conv4';
  $('#eq').innerHTML = `J0: z = k<sub>1</sub>(w<sub>1</sub>·x<sub>22</sub> − θ<sub>1</sub> r<sub>1</sub>·x<sub>22</sub>), act<sub>J0</sub> = silu(z)·z, marker = b·act<sub>J0</sub>·p<sub>6</sub> added at the clause end → head 6 → x<sub>23</sub><br>
  J1/J2: gate = k(s·x − λ c·x − λ<sub>2</sub> c<sub>2</sub>·x), act<sub>J1</sub> = silu(gate)·k s·x, act<sub>J2</sub> = silu(gate)·k<sub>r</sub> r·x &nbsp;·&nbsp; J3: g<sub>3</sub> = k<sub>3</sub>(t·x − λ c·x − λ<sub>2</sub> c<sub>2</sub>·x − θ<sub>3</sub> r<sub>3</sub>·x), act<sub>J3</sub> = silu(g<sub>3</sub>)·k<sub>3</sub>(t·x − θ<sub>3</sub> r<sub>3</sub>·x)<br>
  h′ = h<sub>24</sub> − Σ old<sub>J</sub>·W<sub>d</sub>[:,J] + a<sub>1</sub>act<sub>J1</sub>·${conv ? 'u<sub>conv</sub>' : 'u<sub>AB</sub>'} + a<sub>2</sub>act<sub>J2</sub>·u<sub>route</sub> + a<sub>3</sub>act<sub>J3</sub>·${conv ? 'u<sub>conv</sub>' : '(u<sub>AB</sub> + ρ u<sub>route</sub>)'} &nbsp;·&nbsp; logits = ((h′/rms(h′))⊙γ)·E<sup>T</sup>`;
}
function renderReadouts(it, n, ed, base, am, amText, pam, st, wr, alpha6) {
  const [dc, dl] = decisionOf(am, amText);
  const idx = (w) => S.tidx.get(S.ids[w]);
  const g = (arr, w) => idx(w) === undefined ? NaN : arr[idx(w)];
  const cls = (v) => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
  const top2 = [...ed.logits.keys()].sort((a, b) => ed.logits[b] - ed.logits[a]);
  const orig = S.data.tracked.text[S.tidx.get(it.states.b0.base_top[0][0])] ?? '?';
  const tile = (c, k, v, d, signed = true) => `<div class="ro ${c}"><div class="k">${esc(k)}</div><div class="v ${signed ? cls(parseFloat(v)) : ''}">${esc(v)}</div><div class="d">${esc(d)}</div></div>`;
  const s2on = S.stage2 && ['g23v4', 'conv4', 'rand17'].includes(S.variant), s1on = S.stage1 && S.variant !== 'original';
  const hero = `<div class="ro-hero" id="ro-hero"><span class="k">First token at the decision position</span><span class="tok">${esc(JSON.stringify(amText).slice(1, -1))}</span><span class="dl"><span class="dec ${dc}">${esc(dl)}</span></span><span class="d">p = ${fmt(pam, 3)} · margin over the runner-up ${fmt(ed.logits[top2[0]] - ed.logits[top2[1]], 2)} logits · original model's first token: ${esc(JSON.stringify(orig))}</span></div>`;
  const g2 = [
    tile('j0', 'J0 = 2525 · act at the clause end', fmt(wr.act, 2), `layer 22 · z = ${fmt(wr.z, 2)} · original neuron here ${fmt(j0At(it.prompt.x22[it.end]).old, 4)}`),
    tile('head', 'head 6 · attention → clause end', fmt(alpha6, 4), `natural ${fmt(it.heads[6].alpha_nat, 4)} · spill on heads 4/5/7: ${[4, 5, 7].map((h) => fmt(st.alpha[h], 4)).join(' / ')}`, false),
    tile('j3', 'J3 = 2331 · act', fmt(n.act3, 3), `g₃ = ${fmt(n.gate3, 2)} · t·x ${fmt(n.tx, 2)} · original neuron here ${fmt(n.old3, 4)}`),
  ];
  const g1 = [
    tile('', 'J1/J2 gate pre-activation', fmt(n.gate, 3), S.variant === 'rand13' ? 'k·g₁₃·x (random direction)' : !Number.isFinite(n.gate) ? 'edit off' : `k·s·x ${fmt(n.comps.s, 2)} − λ term ${fmt(-n.comps.c, 2)} − λ₂ term ${fmt(-n.comps.c2, 2)}`),
    tile('j1', 'J1 = 3468 · abstain act', fmt(n.act1, 3), `original neuron here ${fmt(n.old1, 3)}`),
    tile('j2', 'J2 = 3421 · route act', fmt(n.act2, 3), `original neuron here ${fmt(n.old2, 3)}`),
    tile('', 's·x / r·x / c·x / c₂·x', `${fmt(n.sx, 2)} / ${fmt(n.rx, 2)} / ${fmt(n.cx, 2)} / ${fmt(n.c2x, 2)}`, 'abstain, route, tool-context and generated-position features at x₂₃', false),
  ];
  const g0 = [
    tile('', '<tool_call> − NO', `${fmt(g(ed.logits, '<tool_call>') - g(ed.logits, 'NO'), 2)}`, `original: ${fmt(g(base.logits, '<tool_call>') - g(base.logits, 'NO'), 2)}`),
    tile('', 'NE − NO', `${fmt(g(ed.logits, 'NE') - g(ed.logits, 'NO'), 2)}`, `original: ${fmt(g(base.logits, 'NE') - g(base.logits, 'NO'), 2)}`),
    tile('', 'rms(h′) / rms(h₂₄)', `${fmt(ed.rms, 3)} / ${fmt(base.rms, 3)}`, 'final RMSNorm denominator, edited / original', false),
  ];
  $('#readouts').innerHTML = hero + `<div class="ro-groups"><div class="ro-group" data-title="Act II · layer 22 → head 6 → layer 23${s2on ? '' : ' · off'}">${g2.join('')}</div><div class="ro-group" data-title="Act I · layer 23${s1on ? '' : ' · off'}">${g1.join('')}</div><div class="ro-group" data-title="Readout at the decision position">${g0.join('')}</div></div>`;
  if (S.lastDec !== undefined && S.lastDec !== dl) { $('#ro-hero').classList.add('flash'); $('#a11y-decision').textContent = `First token now ${dl}`; }
  S.lastDec = dl;
}
function renderTokens(it, ed, base, sm, smb, order, am, st, sn) {
  const T = S.data.tracked, tb = $('#toktable tbody'); const rows = []; const shown = new Set();
  const pinned = ['<tool_call>', 'NO', 'NE', 'I', 'To', 'The', '<', '</', 'No', 'None'].map((w) => S.ids[w]).filter((x) => x !== undefined);
  const row = (i, cls) => { shown.add(i); const d = ed.logits[i] - base.logits[i]; return `<tr class="${cls}${T.ids[i] === am ? ' arg' : ''}"><td class="tok">${esc(JSON.stringify(T.text[i]).slice(1, -1))}</td><td class="num">${fmt(ed.logits[i], 2)}</td><td class="num">${d >= 0 ? '+' : ''}${fmt(d, 2)}</td><td class="num">${fmtP(sm.p[i])}</td><td class="barcell"><span class="bar" style="width:${(100 * sm.p[i]).toFixed(1)}%"></span></td></tr>`; };
  for (let r = 0; r < 10; r++) rows.push(row(order[r], 'top'));
  for (const id of pinned) { const i = S.tidx.get(id); if (i !== undefined && !shown.has(i)) rows.push(row(i, 'pinned')); }
  rows.push(`<tr class="tail"><td class="tok">all ${(S.data.constants.vocab - T.ids.length).toLocaleString()} other tokens</td><td class="num">—</td><td class="num">—</td><td class="num">${fmtP(sm.tail)}</td><td class="barcell"><span class="bar" style="width:${(100 * sm.tail).toFixed(1)}%;background:var(--muted)"></span></td></tr>`);
  tb.innerHTML = rows.join('');
  const tm = S.data.meta.checks.tail; const key = { original: 'original@b0', g23v3: 'g23v3@b0', v1: 'v1@b0', v2: 'v2@b0', rand13: 'rand13@b0', g23v4: 'g23v4@' + sn, rand17: 'rand17@rand17', conv3: 'conv3@b0', conv4: 'conv4@b1' }[S.variant];
  const errV = S.variant === 'original' ? 0 : (tm[key] ? tm[key].max_abs_prob_err_top10 : Math.max(...Object.values(tm).map((x) => x.max_abs_prob_err_top10)));
  $('#dist-hint').textContent = `Top 10 of the edited distribution plus pinned tokens; ${T.ids.length} tokens are tracked exactly, the other tokens keep the logits of the current stage-1 state (${S.variant === 'original' ? 'exact for the original' : `max top-10 probability error at the recorded ${VLABEL[S.variant]} settings over all ${S.data.n_items} prompts: ${errV.toExponential(1)}; sliders moved away from them are not covered by that bound`}). Stage-1 state: ${sn}. Original top-3 here: ${it.states.b0.base_top.slice(0, 3).map(([id, l]) => `${JSON.stringify(T.text[S.tidx.get(id)] ?? id)} ${l.toFixed(1)}`).join(', ')}.`;
}

function renderSchematic(it, n, ed, base, am, amText, st, wr, alpha6) {
  const C3 = S.data.constants.g23v3, v = S.variant, off = v === 'original' || (!S.stage1 && !S.stage2);
  const col = (x, pos = 'var(--good)', neg = 'var(--bad)') => x > 0 ? pos : x < 0 ? neg : 'var(--muted)';
  const [, dl] = decisionOf(am, amText);
  const lamTxt = v === 'rand13' ? 'random gate direction g₁₃' : `λ = ${fmt(S.sl.lam, 2)} · λ₂ = ${fmt(S.sl.lam2, 2)}`;
  const uAB = v === 'v1' ? 'u_AB (v1 readout)' : v === 'rand13' ? 'random u' : (v === 'conv3' || v === 'conv4') ? 'u_conv: removes <tool_call> only' : 'u_AB: NO, NE up · call, I, To, The down';
  const uR = v === 'rand13' ? 'random u′' : 'u_route = u_NE − u_NO';
  const s1 = S.stage1 && v !== 'original', s2 = S.stage2 && ['g23v4', 'conv4', 'rand17'].includes(v);
  const w = 980, h = 480;
  const endTok = it.prompt.text[it.end].replace(/\n/g, '↵');
  const orig = JSON.stringify(S.data.tracked.text[S.tidx.get(it.states.b0.base_top[0][0])] ?? '?');
  const pill = (x, y, wd, color, name, val, sub) => `<rect x="${x}" y="${y}" width="${wd}" height="52" rx="26" style="fill:color-mix(in srgb,${color} 14%,var(--panel));stroke:${color};stroke-width:1.5"/>
    <text x="${x + wd / 2}" y="${y + 21}" text-anchor="middle" font-weight="600" fill="${color}">${name}</text><text class="m big" x="${x + wd / 2}" y="${y + 41}" text-anchor="middle" font-weight="700">${val}</text>${sub ? `<text class="k" x="${x + wd / 2}" y="${y + 68}" text-anchor="middle">${sub}</text>` : ''}`;
  $('#schematic').innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Four-neuron circuit schematic with live values">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>
  <style>text{font:12.5px var(--sans);fill:var(--ink)} .k{fill:var(--muted);font-size:10.8px} .m{font-family:var(--mono)} .big{font-size:14px} .box{fill:var(--panel-2);stroke:var(--rule-2);rx:10} .lane{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;fill:var(--muted)} line,path.w{stroke:var(--muted);stroke-width:1.3;fill:none;marker-end:url(#arr)} .dim{opacity:.38}</style>
  <text class="lane" x="12" y="14" fill="var(--c-g23v4)">Act II · layer 22 → head 6 → layer 23</text>
  <g class="${s2 ? '' : 'dim'}">
    <rect class="box" x="12" y="22" width="222" height="104"/>
    <text x="123" y="42" text-anchor="middle" font-weight="600">layer-22 MLP · <tspan fill="var(--n-j0)">J0 = 2525</tspan></text>
    <text class="k" x="123" y="57" text-anchor="middle">reads x₂₂ at the clause end “${esc(endTok)}”</text>
    <text class="m" x="24" y="80">z = k₁(w₁·x − θ₁ r₁·x)</text><text class="m" x="222" y="80" text-anchor="end" font-weight="700">${s2 ? fmt(wr.z, 2) : 'off'}</text>
    <text class="m" x="24" y="99">act = silu(z)·z</text><text class="m" x="222" y="99" text-anchor="end" font-weight="700" fill="${s2 && wr.act > 1 ? 'var(--n-j0-ink)' : 'var(--ink)'}">${s2 ? fmt(wr.act, 1) : '—'}</text>
    <text class="k" x="24" y="117">marker b·act·p₆, |·| = ${s2 ? fmt(wr.w, 1) : '0'} (residual rms ≈ 9.6)</text>
    <line x1="234" y1="74" x2="286" y2="74"/>
    <rect class="box" x="290" y="22" width="236" height="104"/>
    <text x="408" y="42" text-anchor="middle" font-weight="600">layer-23 attention · head 6</text>
    <text class="k" x="408" y="57" text-anchor="middle">decision query → marked key · weights unchanged</text>
    <text class="m" x="408" y="86" text-anchor="middle" font-size="22" font-weight="700" fill="${alpha6 > 0.5 ? 'var(--c-g23v4-ink)' : 'var(--muted)'}">α₆ = ${fmt(alpha6, 3)}</text>
    <text class="k" x="408" y="104" text-anchor="middle">natural ${fmt(it.heads[6].alpha_nat, 4)} · heads 4/5/7: ${[4, 5, 7].map((hh) => fmt(st.alpha[hh], 3)).join(' / ')}</text>
    <text class="k" x="408" y="119" text-anchor="middle">copies the value → t·x₂₃ = ${fmt(n.tx, 2)}</text>
    <line x1="526" y1="74" x2="578" y2="74"/>
    ${pill(582, 48, 150, 'var(--n-j3)', 'J3 = 2331', s2 ? fmt(n.act3, 2) : '—', `g₃ = ${s2 ? fmt(n.gate3, 2) : '—'} · × a₃ = ${fmt(S.sl.a3, 3)}${v === 'rand17' ? ' (random)' : ''}`)}
    <path class="w" d="M732 74 C 790 74, 790 262, 836 262"/>
  </g>
  <text class="lane" x="12" y="152" fill="var(--c-g23v3)">Act I · layer 23, decision position</text>
  <rect class="box" x="12" y="160" width="222" height="250"/>
  <text x="123" y="180" text-anchor="middle" font-weight="600">layer-23 MLP input x₂₃</text>
  <text class="k" x="123" y="194" text-anchor="middle">post-attention RMSNorm, 1024-d</text>
  ${[['s·x', n.sx, 'abstain feature'], ['r·x', n.rx, 'route feature (NE vs NO)'], ['c·x', n.cx, 'tool-context gate'], ['c₂·x', n.c2x, 'generated-position gate'], ['t·x', n.tx, 'delivered marker (Act II)']].map(([k, val, d], i) => `<text class="m" x="24" y="${220 + i * 38}">${k}</text><text class="m" x="222" y="${220 + i * 38}" text-anchor="end" font-weight="700">${fmt(val, 2)}</text><text class="k" x="24" y="${233 + i * 38}">${d}</text>`).join('')}
  <line x1="234" y1="226" x2="286" y2="226"/><line x1="234" y1="264" x2="286" y2="336"/><line x1="234" y1="302" x2="286" y2="232"/><line x1="234" y1="340" x2="286" y2="238"/>
  <g class="${s1 ? '' : 'dim'}">
    <rect class="box" x="290" y="190" width="236" height="96"/>
    <text x="408" y="210" text-anchor="middle" font-weight="600">shared gate J1 / J2</text>
    <text class="k" x="408" y="224" text-anchor="middle">k = ${C3.k} · ${esc(lamTxt)}</text>
    <text class="m" x="408" y="252" text-anchor="middle" font-size="20" font-weight="700" fill="${!Number.isFinite(n.gate) ? 'var(--muted)' : col(n.gate)}">${!Number.isFinite(n.gate) ? 'off' : fmt(n.gate, 2)}</text>
    <text class="k" x="408" y="274" text-anchor="middle">silu(gate) = ${!Number.isFinite(n.gate) ? '—' : fmt(silu(n.gate), 3)}</text>
    <rect class="box" x="290" y="312" width="236" height="60"/>
    <text x="408" y="334" text-anchor="middle">up J1 = k·s·x <tspan class="m" font-weight="700">${off ? '—' : fmt(C3.k * n.sx, 2)}</tspan></text>
    <text x="408" y="358" text-anchor="middle">up J2 = k<tspan baseline-shift="sub" font-size="9">r</tspan>·r·x <tspan class="m" font-weight="700">${off ? '—' : fmt(C3.kr * n.rx, 2)}</tspan></text>
    <line x1="526" y1="236" x2="578" y2="236"/><line x1="526" y1="340" x2="578" y2="252"/><line x1="526" y1="252" x2="578" y2="332"/><line x1="526" y1="356" x2="578" y2="350"/>
    ${pill(582, 206, 150, 'var(--n-j1)', 'J1 = 3468 abstain', fmt(n.act1, 3), `× a₁ = ${fmt(S.sl.a1, 2)}`)}
    <text class="k" x="657" y="288" text-anchor="middle">${esc(uAB)}</text>
    ${pill(582, 318, 150, 'var(--n-j2)', 'J2 = 3421 route', fmt(n.act2, 3), `× a₂ = ${fmt(S.sl.a2, 2)}`)}
    <text class="k" x="657" y="400" text-anchor="middle">${esc(uR)}</text>
    <path class="w" d="M732 232 C 790 232, 790 256, 836 256"/><path class="w" d="M732 344 C 790 344, 790 268, 836 268"/>
  </g>
  <rect class="box" x="840" y="196" width="130" height="150" rx="10"/>
  <text x="905" y="216" text-anchor="middle" font-weight="600">+ h₂₄ → RMSNorm</text>
  <text x="905" y="231" text-anchor="middle" font-weight="600">→ E<tspan baseline-shift="super" font-size="9">T</tspan> logits</text>
  <text class="k" x="905" y="249" text-anchor="middle">rms ${fmt(base.rms, 3)} → ${fmt(ed.rms, 3)}</text>
  <text class="m" x="905" y="285" text-anchor="middle" font-weight="700" font-size="22" fill="${dl.startsWith('NE') ? 'var(--c-g23v4-ink)' : dl.startsWith('NO') ? 'var(--ink)' : dl.startsWith('<') ? 'var(--c-base-ink)' : 'var(--n-j0-ink)'}">${esc(JSON.stringify(amText).slice(1, -1))}</text>
  <text class="k" x="905" y="306" text-anchor="middle">${esc(dl.split(' → ')[1] ?? dl)}</text>
  <text class="k" x="905" y="328" text-anchor="middle">original: ${esc(orig)}</text>
  <text class="k" x="12" y="446">original (dead) neurons at this position: J0 ${fmt(j0At(it.prompt.x22[it.end]).old, 4)} · J1 ${fmt(n.old1, 3)} · J2 ${fmt(n.old2, 3)} · J3 ${fmt(n.old3, 4)} — their columns are removed from the residual when the edit is on.</text>
  <text class="k" x="12" y="464">Edited tensors: layers.22.mlp.{gate,up,down}_proj row/column J0 · layers.23.mlp.{gate,up,down}_proj rows/columns J1, J2, J3. No attention weight changed.</text>
  </svg>`;
}

// ------------------------------------------------------------------ prompt-token strip, heat map and arc
function renderStrip(it, wr, alpha6, st) {
  const P = it.prompt, T = P.text.length; const acts = P.x22.map((x) => j0At(x).act); const amax = Math.max(1, ...acts);
  const W = 900, H = 150, left = 36, top = 44, plotH = 70, bw = (W - left - 10) / T; const yv = (v) => top + plotH - Math.log10(1 + v) / Math.log10(1 + amax) * plotH;
  let bars = ''; let hot = 0;
  for (let j = 0; j < T; j++) { const a = acts[j]; if (a > 1) hot++; const x = left + j * bw; bars += `<rect x="${x.toFixed(1)}" y="${yv(a).toFixed(1)}" width="${Math.max(0.6, bw - 0.2).toFixed(2)}" height="${(top + plotH - yv(a)).toFixed(1)}" fill="${j === it.end ? 'var(--ink)' : 'var(--n-j0)'}"><title>pos ${j} ${JSON.stringify(P.text[j])}: J0 act ${a.toFixed(3)}</title></rect>`; }
  const xe = left + (it.end + 0.5) * bw, xd = left + (T - 0.5) * bw; const arcOn = alpha6 > 0.01; const ymid = top - 30 * Math.min(1, Math.sqrt(alpha6));
  $('#posstrip').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="J0 activation per prompt token and head-6 attention arc">
    <text x="4" y="${top + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">${fmt(amax, 0)}</text><text x="4" y="${top + plotH}" font-size="10" fill="var(--muted)" font-family="var(--mono)">0</text>
    <line x1="${left}" y1="${top + plotH}" x2="${W - 8}" y2="${top + plotH}" stroke="var(--rule-2)"/>${bars}
    <path d="M${xd.toFixed(1)} ${top + plotH + 2} C ${xd.toFixed(1)} ${ymid}, ${xe.toFixed(1)} ${ymid}, ${xe.toFixed(1)} ${top + plotH + 2}" fill="none" stroke="${arcOn ? 'var(--c-g23v4)' : 'var(--muted)'}" stroke-width="${(1 + 3 * alpha6).toFixed(1)}" stroke-dasharray="${arcOn ? '' : '3 3'}"/>
    <text x="${((xd + xe) / 2).toFixed(1)}" y="${Math.max(12, ymid - 6).toFixed(1)}" text-anchor="middle" font-size="11" fill="${arcOn ? 'var(--c-g23v4-ink)' : 'var(--muted)'}" font-family="var(--mono)">head 6: decision → clause end, α = ${fmt(alpha6, 4)}</text>
    <text x="${xe.toFixed(1)}" y="${top + plotH + 14}" text-anchor="middle" font-size="10" fill="var(--ink)" font-weight="600" font-family="var(--mono)">${esc(P.text[it.end].replace(/\n/g, '↵'))} (pos ${it.end})</text>
    <text x="${xd.toFixed(1)}" y="${top + plotH + 14}" text-anchor="end" font-size="10" fill="var(--c-g23v4-ink)" font-family="var(--mono)">decision (pos ${T - 1})</text>
    <text x="${left}" y="${top + plotH + 28}" font-size="10" fill="var(--muted)">${T} prompt tokens · J0 bars on a log scale · ${hot} position${hot === 1 ? '' : 's'} with act &gt; 1</text></svg>`;
  $('#heat-hint').textContent = `J0 = 2525 activation at every prompt token (${S.stage2 ? 'current stage-1 settings' : 'Act II off: the original dead neuron, ≤ 0.2 everywhere'}); the arc is head 6's attention from the decision query to the clause-end token (exact replay at the snapped dose; the closed form in the write magnitude gives ${fmt(alphaFormula(it.heads[6], wr.w), 4)} for |marker| = ${fmt(wr.w, 1)}).`;
}
function renderHeat(it) {
  if (!it) return; const P = it.prompt, T = P.text.length; const acts = P.x22.map((x) => j0At(x).act); const amax = Math.max(1, ...acts);
  const start = S.allTok ? 0 : Math.max(0, (it.roles && it.roles.v1 ? Math.min(it.roles.v1, it.end) : it.end) - 60);
  let html = start > 0 ? `<span class="tk more">… ${start} earlier tokens (system prompt and tool schema) hidden …</span>` : '';
  for (let j = start; j < T; j++) {
    const a = acts[j]; const heat = Math.log10(1 + a) / Math.log10(1 + amax); const isEnd = j === it.end, isDec = j === T - 1;
    const roleName = it.roles ? Object.entries(it.roles).filter(([k, p]) => p === j && !['dec', 'imend', 'sysamb'].includes(k)).map(([k]) => k).join(',') : '';
    html += `<span class="tk${isEnd ? ' end' : ''}${isDec ? ' dec' : ''}" style="--heat:${heat.toFixed(3)}" title="pos ${j}${roleName ? ' · ' + roleName : ''} · J0 act ${a.toFixed(3)}">${esc(P.text[j].replace(/\n/g, '↵').replace(/ /g, '␣'))}</span>`;
  }
  $('#tokheat').innerHTML = html;
}

function renderTrace(it) {
  const Gn = it.generated.text.length, W = Math.max(600, Gn * 14 + 60), H = 240, left = 44, top = 14, plotH = 150;
  const vals = []; let amax = 1e-6;
  for (let j = 0; j < Gn; j++) { const r = circuit(it.generated.x23[j]); const j0 = j0At(it.generated.x22[j]); vals.push({ ...r, j0: j0.act }); amax = Math.max(amax, Math.abs(r.act1), Math.abs(r.act2), Math.abs(r.act3), Math.abs(j0.act)); }
  const scale = Math.max(amax, 1); const y0 = top + plotH / 2, sy = (plotH / 2 - 4) / scale, bw = (W - left - 10) / Gn; const yv = (v) => y0 - v * sy;
  let bars = '', labels = '';
  for (let j = 0; j < Gn; j++) {
    const x = left + j * bw, r = vals[j]; const q = bw / 4;
    const bar = (k, v, color, name) => `<rect x="${(x + k * q + 0.5).toFixed(1)}" y="${Math.min(y0, yv(v)).toFixed(1)}" width="${Math.max(1, q - 0.8).toFixed(1)}" height="${Math.abs(yv(v) - y0).toFixed(1)}" fill="${color}"><title>${name} at pos ${j} ${JSON.stringify(it.generated.text[j])}: ${v.toFixed(3)} · J1 ${r.act1.toFixed(3)} J2 ${r.act2.toFixed(3)} J3 ${r.act3.toFixed(3)} J0 ${r.j0.toFixed(3)} gate ${Number.isFinite(r.gate) ? r.gate.toFixed(2) : '—'}</title></rect>`;
    bars += bar(0, r.act1, 'var(--n-j1)', 'J1') + bar(1, r.act2, 'var(--n-j2)', 'J2') + bar(2, r.act3, 'var(--n-j3)', 'J3') + bar(3, r.j0, 'var(--n-j0)', 'J0');
    const t = it.generated.text[j].replace(/\n/g, '↵'); const hot = Math.max(Math.abs(r.act1), Math.abs(r.act2), Math.abs(r.act3), Math.abs(r.j0)) > 1;
    if (Gn <= 64 || j % 2 === 0) labels += `<text x="${x + bw / 2}" y="${top + plotH + 14}" transform="rotate(-60 ${x + bw / 2} ${top + plotH + 14})" font-size="${Gn > 64 ? 8 : 9.5}" text-anchor="end" fill="${hot ? 'var(--ink)' : 'var(--muted)'}" font-weight="${hot ? '700' : '400'}" font-family="var(--mono)">${esc(t.length > 9 ? t.slice(0, 8) + '…' : t)}</text>`;
  }
  const hot = vals.filter((r) => Math.max(Math.abs(r.act1), Math.abs(r.act2), Math.abs(r.act3), Math.abs(r.j0)) > 1).length;
  $('#trace-hint').innerHTML = `Layer-22 and layer-23 MLP inputs at every generated position of the <em>original</em> model, re-read through the four edited neurons with the current sliders; the circuit must stay silent inside tool calls and 128-token prose. Here: ${Gn} generated positions · max |act| ${fmt(amax, 2)} · ${hot} position${hot === 1 ? '' : 's'} with |act| &gt; 1 (bold labels) · position 0 is the decision token.`;
  $('#trace').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Per-position neuron activations">
  <line x1="${left}" y1="${y0}" x2="${W - 8}" y2="${y0}" stroke="var(--rule-2)"/>
  <text x="4" y="${yv(scale) + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">+${fmt(scale, 1)}</text><text x="4" y="${yv(-scale) + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">−${fmt(scale, 1)}</text><text x="4" y="${y0 + 4}" font-size="10" fill="var(--muted)" font-family="var(--mono)">0</text>
  ${bars}${labels}</svg>`;
}

function renderProvenance() {
  const m = S.data.meta, c = m.checks, ag = c.argmax_agreement || {};
  const li = [];
  li.push(`${m.n_items} prompts · base <code>${esc(m.base_model)}</code> @ <code>${esc(m.base_revision.slice(0, 7))}</code>`);
  if (ag.original_fresh) li.push(`fresh original first token reproduced by the exact readout: ${ag.original_fresh.match}/${ag.original_fresh.n}`);
  for (const v of ['original', 'g23v3', 'g23v4', 'v2', 'v1', 'rand13', 'ctrl_rand17']) if (ag[v]) li.push(`${VLABEL[v] ?? v} recorded first token predicted at the recorded settings: ${ag[v].match}/${ag[v].n}`);
  const ti = c.original_generation_token_identical || {}; li.push(`fresh original generations token-identical to the recorded runs: ${Object.entries(ti).map(([k, v]) => `${k} ${v}`).join(', ')}`);
  li.push(`closed-form readout vs direct recomputation: max |Δlogit| ${(+c.formula_vs_direct_max_abs).toExponential(1)}`);
  li.push(`head-6 attention closed form vs exact replay at the snapped doses: max |Δα| ${(+c.attention_formula_vs_replay_max_abs).toExponential(1)}`);
  $('#provenance').innerHTML = `<ul>${li.map((x) => `<li>${x}</li>`).join('')}</ul><p>Generated ${esc(m.generated_utc)}. ${esc(m.provenance)}</p>`;
}

// ------------------------------------------------------------------ tabs, on-page navigation
function setupTabs() {
  const tabs = $$('.tab');
  tabs.forEach((b, i) => {
    b.onclick = () => switchTab(b.dataset.tab);
    b.onkeydown = (e) => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); const n = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); switchTab(n.dataset.tab); } };
  });
}
function switchTab(name) {
  $$('.tab').forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1; });
  $('#tab-explore').hidden = name !== 'explore'; $('#tab-live').hidden = name !== 'live';
  history.replaceState(null, '', name === 'live' ? '#live' : location.pathname + location.search);
  if (name === 'live') checkGPU();
}
function setupOnPage() {
  const ol = $('#onpage'); ol.innerHTML = SECTIONS.map(([id, lab], i) => `<li><a href="#${id}" data-sec="${id}"><span class="n">${String(i + 1).padStart(2, '0')}</span><span>${esc(lab)}</span></a></li>`).join('');
  const links = $$('#onpage a');
  if (!('IntersectionObserver' in window)) return;
  const vis = new Map();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) vis.set(e.target.id, e.isIntersecting ? e.boundingClientRect.top : Infinity);
    let best = null, bt = Infinity; for (const [id, t] of vis) { if (t !== Infinity && Math.abs(t - 90) < bt) { bt = Math.abs(t - 90); best = id; } }
    if (best) links.forEach((a) => a.classList.toggle('current', a.dataset.sec === best));
  }, { rootMargin: '-80px 0px -55% 0px', threshold: [0, 0.2, 0.5] });
  SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
}

// ------------------------------------------------------------------ Tab 2: WebGPU runner (UI side; model work lives in worker.js)
const L = { worker: null, gpu: null, loaded: new Set(), running: false, t: {} };
function log(msg) { const el = $('#live-log'); el.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`; el.scrollTop = el.scrollHeight; }
async function checkGPU() {
  if (L.gpu !== null) return L.gpu;
  const el = $('#gpu-status');
  if (!('gpu' in navigator)) { L.gpu = false; el.className = 'notice bad'; el.innerHTML = 'This browser does not expose <code>navigator.gpu</code>. The live runner needs WebGPU (Chrome/Edge 113+, Safari 26+, Firefox 141+ on supported hardware). The explorer mode works everywhere and needs no download.'; return false; }
  try {
    const ad = await navigator.gpu.requestAdapter(); if (!ad) throw new Error('requestAdapter() returned null');
    const info = ad.info || {}; const f16 = ad.features.has('shader-f16');
    L.gpu = { f16, vendor: info.vendor, arch: info.architecture };
    el.className = 'notice ok'; el.innerHTML = `WebGPU available${info.vendor ? ` · ${esc(info.vendor)}${info.architecture ? ' / ' + esc(info.architecture) : ''}` : ''} · shader-f16 ${f16 ? 'yes' : 'no (fp16 kernels will be emulated; expect slower generation)'} · max buffer ${(ad.limits.maxBufferSize / 2 ** 30).toFixed(1)} GiB.`;
    $('#btn-load').disabled = false;
  } catch (e) { L.gpu = false; el.className = 'notice bad'; el.innerHTML = `WebGPU adapter unavailable (${esc(e.message || e)}). Use the explorer mode, which needs no GPU and no download.`; }
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
  const pct = m.total ? Math.min(100, 100 * m.loaded / m.total) : (/done|ready|cache/.test(m.status || '') ? 100 : 0);
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

// ------------------------------------------------------------------ self-test: recorded first tokens vs the exact readout at each variant's recorded settings
function selfTest() {
  const C3 = S.data.constants.g23v3, C4 = S.data.constants.g23v4;
  const cfg = {
    original: { st: 'b0', v: 'original', sl: {}, s1: false, s2: false },
    v1: { st: 'b0', v: 'v1', sl: { a1: 1, a2: 1, lam: 0, lam2: 0, a3: 0 }, s1: true, s2: false },
    v2: { st: 'b0', v: 'v2', sl: { a1: 1, a2: 1, lam: C3.lam, lam2: 0, a3: 0 }, s1: true, s2: false },
    g23v3: { st: 'b0', v: 'g23v3', sl: { a1: C3.a1, a2: C3.a2, lam: C3.lam, lam2: C3.lam2, a3: 0 }, s1: true, s2: false },
    g23v4: { st: 'b1', v: 'g23v4', sl: { a1: C3.a1, a2: C3.a2, lam: C4.lam, lam2: C4.lam2, b: 1, a3: C4.a3, rho: C4.rho }, s1: true, s2: true },
    rand13: { st: 'b0', v: 'rand13', sl: { a1: 1, a2: 1, a3: 0 }, s1: true, s2: false },
    ctrl_rand17: { st: 'rand17', v: 'rand17', sl: { a1: C3.a1, a2: C3.a2, lam: C3.lam, lam2: C3.lam2, b: 1, a3: 1, rho: 0 }, s1: true, s2: true },
  };
  const res = {};
  for (const [name, c] of Object.entries(cfg)) {
    let n = 0, match = 0;
    for (const it of S.items) {
      const o = it.outputs[name], st = it.states[c.st]; if (!o || !o.token_ids || !st) continue;
      const r = circuit(st.x23, c.v, { ...S.sl, ...c.sl }, c.s1, c.s2), ed = logitsFor(st, r.coef);
      let best = 0; for (let t = 1; t < ed.logits.length; t++) if (ed.logits[t] > ed.logits[best]) best = t;
      n++; if (S.data.tracked.ids[best] === o.token_ids[0]) match++;
    }
    res[name] = `${match}/${n}`;
  }
  return res;
}
window.__explorer = { S, circuit, logitsFor, softmaxWithTail, alphaFormula, j0At, writeAtEnd, selfTest };
