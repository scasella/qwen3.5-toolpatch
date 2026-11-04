# g23v4: routing the "two candidate values" signal to the decision — a two-stage weight edit on top of g23v3 (Qwen3.5-0.8B)

Workspace `/Users/scasella/Downloads/qwen-7` (qwen-6 read-only). Model `Qwen/Qwen3.5-0.8B` rev `2fc0636…`, BF16, MLX-LM 0.32.2 / MLX 0.32.4, Apple M4 Pro.
Protocol identical to qwen-6: official chat template with native `<tools>`, `enable_thinking=False`, greedy, max 128 new tokens, context ≤ 2048, system prompt `standard`
(no format examples), lenient/strict scoring as defined there. Every model forward is a budgeted pass in the hash-chained `ledger.jsonl`: **20 of 50 passes used**
(1,184 executions; one pass, `d01_triples_D1`, was charged without executing because of a tokenizer-wrapper bug and re-run as `d01b`). No gradients, optimizers, LoRA or
output-fitting; no runtime hooks in the deliverable; `embed_tokens`, `lm_head`, final norm and vision untouched. Public write-ups credit the agent as Mythos 5.1.

**Headline.** g23v4 = g23v3 + 6,139 new BF16 entries (12,281 total, 1.6 × 10⁻⁵ of the 752,393,024 text parameters; 6 tensors in layers 22 and 23). On the 30 existing
ambiguous `either "X" or "Y"` items it goes 0/30 → **30/30**, with every one of the other 162 tool-context items and all 64 non-tool items token-identical or unchanged
in correctness (0 right→wrong anywhere). On the new 64-item Ambiguity Stress Suite (held-out families, renamed tools, six never-captured phrasings) it goes
base 0/32, g23v3 9/32 → **g23v4 27/32** ambiguous → NEED_CLARIFICATION, and all 32 contrastive resolved two-value calls are token-identical to g23v3 and base
(0 false clarifications). The circuit is explicit: a layer-22 neuron detects "clause end AND unresolved", writes a marker whose layer-23 key is aligned with the decision
query of attention head 6, head 6 carries it to the decision position (α ≈ 0.99), and a third layer-23 neuron reads it under g23v3's gates. An exact CPU replay of layer 23
predicted the real edited model's decision token on 64/64 calibration items (median |logit error| 0.06, max 0.5) and the direction and sign of every dose/random control before it was run.

## 1. Diagnosis: where the ambiguity signal lives (Hypotheses A/B/C)

Data: matched triples per family — (i) complete call, (ii) resolved two-value call (both values mentioned, choice unambiguous: "use X rather than Y", "previously Y, but today
use X", …), (iii) unresolved ambiguous — six derivation phrasings each (set A/R), renamed twins, and held-out phrasings (set B/S) on selection/final families
(`build_triples.py`; D1 64 discovery items + 10 existing practice ambiguous, D2 their 64 renamed twins, D3 64 selection/final items with set B + 10 existing). Three
teacher-forced passes on the original model captured the residual at all 25 layer boundaries at role-tagged positions (first value, connector, second value, indecision tail,
clause-end '.', `<|im_end|>`, decision, the word "ambiguous" in the system prompt), per-head gated attention outputs there, attention rows from the decision query over every
key at the six full-attention layers, and all-position MLP inputs of layers 18/22/23 and residual inputs of layers 19/23. All probes are leave-one-family-out (LOFO) over
the 36 families; d′ = (μ₊−μ₋)/pooled sd. (`analysis/diag.py`, `analysis/diag2.py`; `analysis/diag*_….json`.)

**Hypothesis A (within-pair subspace masking) — refuted as a sufficient mechanism.** At the decision position, ambiguous vs {call, resolved}:

| probe at the decision position (192 items, LOFO) | L12 | L14 | L18 | L22 | L23 | x23 (layer-23 MLP input) |
|---|---|---|---|---|---|---|
| raw diff-of-means d′ (acc / gap) | 1.46 | 2.55 (0.92 / −0.24) | 2.49 | 2.75 | 2.73 (0.92 / −1.02) | 2.66 (0.90 / −3.84) |
| family-subspace masked (top-4 / 8 / 16 family-mean PCs projected out) | 1.5 / 1.8 / 2.0 | 3.4 / 3.4 / 3.4 | 2.9 / 3.0 / 2.9 | 3.0 / 3.0 / 3.0 | 3.0 / 3.0 / 3.0 | 3.0 / 3.0 / 3.0 |
| oracle family-triple centering (not implementable) | 3.8 | 4.1 | 3.5 | 3.7 | 3.7 | 3.6 |
| shrinkage LDA 0.5 / 0.2 d′ (acc) | 2.5 / 2.3 | 3.9 / 3.7 (0.97) | 3.5 / 3.2 | 3.7 / 3.7 | 3.7 / 3.8 (0.97) | 3.5 / 3.5 (0.96) |

A consistent direction exists (centered accuracy 1.00, D1-fit direction transfers to D2+D3 with d′ 2.7), but no linear readout of the decision position exceeds d′ ≈ 3.9 or
closes the overlap (gaps negative everywhere), name-invariantly (D2) and on held-out phrasings/families (D3 raw 2.2, LDA 3.5). A J3-only readout cannot reach 0 regressions.

**Hypothesis B (positional bottleneck) — supported.** The same probes at the clause tokens (ambiguous vs resolved at shared roles; vs {call, resolved} at the clause end):

| role, LOFO d′ raw / shrinkage-LDA 0.5 (acc, gap) | L10 | L12 | L14 | L15 | L18 | L22 | L23 |
|---|---|---|---|---|---|---|---|
| clause-end '.' (84 vs 108) | 3.4 / 11.5 | 4.1 / 13.4 | 4.5 / 14.2 (1.00, +0.41) | 5.0 / 13.5 | 4.9 / 13.3 | 4.7 / 9.5 | 4.6 / 9.5 (1.00, +0.83) |
| connector (84 vs 54) | 4.1 / 19.0 | 4.4 / 18.0 | 4.3 / 17.5 | 4.7 / 19.3 | 4.4 / 17.4 | 3.8 / 13.5 | 3.6 / 13.7 |
| second value (84 vs 54) | 3.2 / 10.8 | 3.7 / 9.8 | 3.5 / 10.4 | 3.2 / 9.2 | 2.7 / 8.8 | 2.4 / 7.8 | 2.2 / 8.1 |
| indecision tail (66 vs 17) | 3.6 / 18.3 | 3.7 / 21.8 | 3.8 / 18.7 | 3.7 / 20.3 | 3.6 / 15.4 | 3.4 / 12.7 | 3.3 / 11.9 |
| `<|im_end|>` (raw) | 0.9 | 1.0 | 1.4 | 1.4 | 1.8 | 1.6 | 1.5 |
| decision (raw) | 1.7 | 1.5 | 2.6 | 2.7 | 2.7 | 2.9 | 2.9 |

The representation of "unresolved alternatives" is strong and linearly clean at the clause tokens (LDA d′ 13–22, accuracy 1.00, positive gaps from layer 10 on) but is not
routed: attention mass from the decision query onto the clause-end token is 0.000–0.003 at layer 23 (heads 2 and 6 attend 97 % / 87 % to themselves), ≤ 0.008 at layer 19,
and 0.015–0.062 at layer 15 (head 1: 0.062 ambiguous / 0.048 resolved / 0.037 call; heads 0–2 put 0.18/0.18/0.10 of their mass on the whole clause span for ambiguous items
vs 0.07/0.04/0.005 for calls — presumably how the weak d′ ≈ 2.5 signal reaches the decision from layer 13–15 on). It is also absent at `<|im_end|>` (d′ 1.5).
**Hypothesis C** (single-head output at the decision): no head exceeds d′ 2.3 (L23 h2 2.34, h6 2.31; L15 ≤ 2.3; L19 ≤ 1.9).

Transport through the existing layer-15 heads was rejected quantitatively: OV gain 0.6, α 0.06, residual RMS 0.09 at the clause end → ≈ 0.2 units delivered per unit write;
a usable feature at layer 23 would need a write of ≈ 3 (the whole residual norm there) reprocessed by eight layers. Near-dead neurons: none below max |a| 0.1 at layers
15/19/21/22/23 over 116k prompt positions; quietest at L22: 2525 (0.205); at L23 besides J1/J2: 2331 (0.237).

## 2. The circuit (g23v4), closed form

Stage 1 — layer-22 MLP neuron **J0 = 2525** (rows/column replaced), reading the layer-22 MLP input x₂₂:
```
gate[J0] = k1·(w1 − θ1·r1)      up[J0] = k1·(w1 − θ1·r1)      down[:,J0] = b·p6          k1 = 4, θ1 = 11.7, b = 1
w1 = unit(0.5·g + u)
u  = shrinkage-LDA(0.5) 'ambiguous clause end vs call/resolved clause end' (LOFO d′ 10.0: 5.8..10.6 vs −1.9..2.0) — alone it also fires on the system prompt
     ('reply exactly NO_TOOL.', 'missing or ambiguous') and inside resolved clauses before the resolution (up to 9.6), so
g  = shrinkage-LDA 'user fact-clause end vs other positions' (LOFO at ends 8.9..19.3) makes the gate the conjunction 'clause end AND unresolved':
     w1·x: ambiguous ends 13.7..16.6 vs every harmful position ≤ 9.8 (all prompt+generated positions of call/resolved/NO_TOOL items, prose, short answers; 141k positions)
r1 = constant-projection direction C⁻¹μ/(μᵀC⁻¹μ) over {ambiguous ends, 3,000 hardest negatives}: r1·x ∈ 0.99..1.03 / 0.94..1.04 there, so (w1 − θ1 r1)·x ≈ w1·x − θ1
     without a bias term; pre-activation z: ambiguous ends 1.90..5.06, harmful max −2.10 → activation silu(k1 z)·k1 z = 58..410 vs ≤ 0.016.
p6 = unit( pinv(Wk_kv1 ⊙ γ_in23) · k_target ),  k_target = γ_k ⊙ mean decision query of layer-23 head 6 on the unrotated dims 64:256, orthogonalized against the
     mean decision queries of heads 4/5/7 (same kv group); written with norm 58..410 it dominates the clause-end residual (norm ≈ 9.6), so that position's layer-23 key
     is exactly aligned with head 6's decision query (its attention-logit cap 30 vs natural max 17.7). The only consumer of a prompt position at layer 23's input is
     layer-23 attention, so nothing else sees the marker.
```
Transport — unchanged layer-23 attention: head 6 at the decision position attends to the marked clause end (replay α 0.992..0.999, real 0.995..0.999; natural 0.0001)
and copies its value; zero spill onto heads 4/5/7; Δx₂₃ at the decision has norm 11.9 and cosine 0.99 to its mean over all 84 ambiguous items.

Stage 2 — layer-23 MLP neuron **J3 = 2331**, reading the layer-23 MLP input x₂₃ (J1 = 3468, J2 = 3421 of g23v3 kept verbatim, bit-identical):
```
gate[J3] = k3·(t − λc − λ₂c₂ − θ3·r3)    up[J3] = k3·(t − θ3·r3)    down[:,J3] = a3·(u_AB + ρ·u_route)     k3 = 2, θ3 = 2, λ = 0.3, λ₂ = 0.5, a3 = 0.1, ρ = 1
t  = unit mean Δx₂₃ delivered by head 6; t·x₂₃: delivered 5.5..7.0, natural ≤ −4.1 at every tool-context decision, ≤ −2.3 at non-tool first tokens, ≤ 3.6 at prose
     positions (where c₂ ≥ 6.8 closes the gate as in g23v3); r3 = constant-projection direction at x₂₃ (0.96..1.03); z: delivered 4.2..5.7, natural decisions ≤ −6.4,
     generated positions ≤ −2.6 → activation 59..109 on ambiguous items, ≤ 0.0003 elsewhere; c, c₂, u_AB, u_route are g23v3's vectors.
```
Changed entries: layer 22 gate/up 1,024 each, down 1,023; layer 23 gate 3,069, up 3,070, down 3,071 (incl. g23v3's 6,142) = **12,281 = 1.63 × 10⁻⁵** of the text
parameters (limit 0.05 %); 6 tensors (limit 8), 2 layers (limit 3). `edits/g23v4.npz` (BF16 bits), `.spec.json` (every scalar and receipt), `.vectors.npz`.

## 3. Simulation and calibration

`analysis/replay23.py` replays layer 23 (attention + MLP + final norm + tied unembedding) on the CPU with the stock MLX modules from the captured layer-23 input sequences
(validated on the unedited model: argmax 72/72, max |logit err| 0.19 from fp16 capture rounding). `analysis/sim4.py` adds the stage-1 writes at every prompt position (from the
captured x₂₂, with J0's original contribution removed) and the J1/J2/J3 rows, and checks stage-1/J3 silence on every captured generated position (qwen-6 p04c/p05 calls and
literals, p17/p35/p16, and the new g01 capture of the 32 prose answers and 32 short non-tool answers at layers 22/23). Prediction for the intended dose: 84/84 captured
ambiguous items → NE (NE−`<tool_call>` ≥ 31, NE−NO ≥ 12.5, NE−runner-up ≥ 11.6), 108/108 call/resolved items keep the g23v3 argmax (max |Δlogit| 0.125 on their top-20
tokens), max J0/J3 write norm at any generated position 0.016. Calibration pass c01 (real g23v4, teacher-forced, 64 D1 items): argmax agreement **64/64**, |logit error| on
NE/NO/`<tool_call>` median 0.06 / max 0.50, α₆ 0.995..0.999, J0 at ambiguous ends 149..360 (replay 219.15 vs real 219.0 on the first item) and ≤ 0.0002 at call/resolved
ends, J3 59..100 vs ≤ 0.0002. Dose scan (replay): a3 = 0.1 with ρ = 0.5 leaves one item at NO (NE−NO −2.0); ρ = 1 chosen.

## 4. Results — Base vs g23v3 vs g23v4 on every cohort (paired, greedy, identical prompts; `analysis/final_tables.md`)

Base and g23v3 rows for the existing cohorts are qwen-6's recorded runs; this workspace's original model reproduced qwen-6's practice baseline token-for-token (g09: 64/64),
and the exported checkpoint in a fresh stock process reproduced both in-process streams (§6). "identical" = token-identical streams.

| cohort | model | lenient | strict | false calls | per class (correct/n, decisions) | mean tokens | vs Base: w->r / r->w / identical | vs g23v3: w->r / r->w / identical |
|---|---|---|---|---|---|---|---|---|
| Practice (64) | Base | 24/64 | 7/64 | 38 | valid_call:24/24 {'call': 24}, no_tool:0/20 {'call': 18, 'prose': 2}, need_clarification:0/10 {'call': 10}, ambiguous:0/10 {'call': 10} | 55.8 | - | 0 / 28 / 36 |
| Practice (64) | g23v3 | 52/64 | 35/64 | 12 | valid_call:24/24 {'call': 24}, no_tool:19/20 {'answer': 19, 'call': 1}, need_clarification:9/10 {'clarify': 9, 'call': 1}, ambiguous:0/10 {'call': 10} | 33.5 | 28 / 0 / 36 | - |
| Practice (64) | g23v4 | 62/64 | 45/64 | 2 | valid_call:24/24 {'call': 24}, no_tool:19/20 {'answer': 19, 'call': 1}, need_clarification:9/10 {'clarify': 9, 'call': 1}, ambiguous:10/10 {'clarify': 10} | 24.7 | 38 / 0 / 26 | 10 / 0 / 54 |
| Selection (32) | Base | 12/32 | 4/32 | 17 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 7, 'prose': 3}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 56.8 | - | 0 / 15 / 17 |
| Selection (32) | g23v3 | 27/32 | 19/32 | 5 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 34.0 | 15 / 0 / 17 | - |
| Selection (32) | g23v4 | 32/32 | 24/32 | 0 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 24.1 | 20 / 0 / 12 | 5 / 0 / 27 |
| Renamed A (32) | Base | 8/32 | 6/32 | 20 | valid_call:8/12 {'call': 12}, no_tool:0/10 {'call': 10}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 45.8 | - | 0 / 15 / 17 |
| Renamed A (32) | g23v3 | 23/32 | 21/32 | 5 | valid_call:8/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 24.5 | 15 / 0 / 17 | - |
| Renamed A (32) | g23v4 | 28/32 | 26/32 | 0 | valid_call:8/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 18.8 | 20 / 0 / 12 | 5 / 0 / 27 |
| Renamed B (32) | Base | 12/32 | 10/32 | 20 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 10}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 45.9 | - | 0 / 14 / 17 |
| Renamed B (32) | g23v3 | 26/32 | 25/32 | 6 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:4/5 {'clarify': 4, 'call': 1}, ambiguous:0/5 {'call': 5} | 26.8 | 14 / 0 / 17 | - |
| Renamed B (32) | g23v4 | 31/32 | 30/32 | 1 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:4/5 {'clarify': 4, 'call': 1}, ambiguous:5/5 {'clarify': 5} | 20.8 | 19 / 0 / 12 | 5 / 0 / 27 |
| Final-split proxy (32) | Base | 12/32 | 3/32 | 19 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 9, 'prose': 1}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 54.7 | - | 0 / 15 / 17 |
| Final-split proxy (32) | g23v3 | 27/32 | 18/32 | 5 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 32.1 | 15 / 0 / 17 | - |
| Final-split proxy (32) | g23v4 | 32/32 | 23/32 | 0 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 23.6 | 20 / 0 / 12 | 5 / 0 / 27 |
| Short non-tool regression2 (32) | Base | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | - | 0 / 0 / 32 |
| Short non-tool regression2 (32) | g23v3 | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | 0 / 0 / 32 | - |
| Short non-tool regression2 (32) | g23v4 | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | 0 / 0 / 32 | 0 / 0 / 32 |
| Prose QA 128-token (16 derivation set) | Base | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | - | 0 / 0 / 11 |
| Prose QA 128-token (16 derivation set) | g23v3 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | 0 / 0 / 11 | - |
| Prose QA 128-token (16 derivation set) | g23v4 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | 0 / 0 / 7 | 0 / 0 / 11 |
| Prose QA 128-token (16 fresh) | Base | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 127.9 | - | 0 / 0 / 9 |
| Prose QA 128-token (16 fresh) | g23v3 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 128.0 | 0 / 0 / 9 | - |
| Prose QA 128-token (16 fresh) | g23v4 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 128.0 | 0 / 0 / 8 | 0 / 0 / 12 |
| Stress: ambiguous (32) | Base | 0/32 | 0/32 | 32 | ambiguous:0/32 {'call': 32} | 54.2 | - | 0 / 9 / 16 |
| Stress: ambiguous (32) | g23v3 | 9/32 | 9/32 | 16 | ambiguous:9/32 {'clarify': 9, 'call': 16, 'answer': 6, 'prose': 1} | 32.5 | 9 / 0 / 16 | - |
| Stress: ambiguous (32) | g23v4 | 27/32 | 27/32 | 5 | ambiguous:27/32 {'clarify': 27, 'call': 5} | 13.2 | 27 / 0 / 5 | 18 / 0 / 14 |
| Stress: resolved two-value calls (32) | Base | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | - | 0 / 0 / 32 |
| Stress: resolved two-value calls (32) | g23v3 | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | 0 / 0 / 32 | - |
| Stress: resolved two-value calls (32) | g23v4 | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | 0 / 0 / 32 | 0 / 0 / 32 |

Tool-context totals over the 192 existing items: lenient 68 → 155 (g23v3) → **185 (g23v4)**, strict 30 → 118 → 148; NEED_CLARIFICATION 0/60 → 28/60 → **58/60**
(ambiguous 0/30 → 0/30 → **30/30**; the two remaining misses are the weak explicit-missing practice item and its renamed twin, unchanged from g23v3); NO_TOOL 59/60,
required calls 68/72 with all 72 streams token-identical to g23v3; false calls 114 → 33 → 3. Every non-ambiguous tool-context item is token-identical to g23v3
(practice 54/54, selection/renamed/final 27/27 each); the only changed streams are the 30 ambiguous items. Non-tool: regression2 32/32 token-identical; the 32 prose
answers all intact, mean length unchanged (125.1 → 125.1), 23/32 token-identical to g23v3 (g23v3 was 20/32 identical to base; the drift source is the removal of J0's
original ≤ 0.2 activation times a 0.15-norm column, which can flip near-tied late tokens).

**Ambiguity Stress Suite (64 items, `data/stress.jsonl`): base 0/32 ambiguous, g23v3 9/32, g23v4 27/32; resolved 30/32 for all three with 32/32 identical streams.**

| ambiguous items by phrasing (n; base / g23v3 / g23v4) | B1 "X or Y, whichever — I have not picked one" | B2 "could be X, could be Y; it is not settled" | B3 "not sure yet: X or Y" | B4 "torn between X and Y" | B5 "X or Y, I keep going back and forth" | B6 "maybe X, maybe Y, I cannot say which" | A1 (qwen-6 phrasing) | A2 "X or possibly Y — I haven't decided" | A3 "unsure whether to use X or Y" |
|---|---|---|---|---|---|---|---|---|---|
| | 5; 0/3/**5** | 4; 0/1/**3** | 4; 0/2/**4** | 5; 0/0/**1** | 4; 0/1/**4** | 4; 0/1/**4** | 2; 0/0/**2** | 2; 0/0/**2** | 2; 0/1/**2** |

Held-out phrasings (B1–B6, never captured or used for any direction): 26 items, base 0, g23v3 8, g23v4 21; seen phrasings on new families/names: 6 items, 0 / 1 / 6.
Renamed tools: 11 items, 0 / 3 / 8; selection (never-captured-with-these-phrasings) families: 14 items, 0 / 2 / 13; discovery families: 18 items, 0 / 7 / 14. Resolved
two-value calls by phrasing (S1–S6, R1/R2/R4): correct 5/5, 4/4, 5/5, 4/4, 4/4, 3/4, 2/2, 2/3, 1/1 for base, g23v3 and g23v4 alike — the two misses are function-choice
errors shared by all three models; g23v4 never emits NEED_CLARIFICATION on a resolved item. g23v3's 9 stress repairs are its abstain feature responding to some set-B
phrasings (6 of its other outputs are NO_TOOL, 1 prose), not ambiguity detection.

Mechanism readout on the suite (pass c02, g23v4 teacher-forced on the 32 ambiguous items): the 5 misses have J0 activation 0.01..0.41 at the clause end (gate shut), head-6
attention 0.000, J3 0.00 — stage-1 misses, as predicted; the 27 hits have J0 10.5..452, α₆ 0.992..0.999, J3 21..107. The miss pattern is a phrasing without a disjunction
or an indecision tail ("torn between X and Y", 1/5; one renamed "could be X, could be Y"): the model's clause-end representation of those is weaker along u and the gate's
margin (1.9 above threshold on captured items) does not cover it. A direction fit on D1+D2 only transferred to the held-out D3 set with ambiguous min 10.2 vs harmful max
9.0 (margin 1.2) — this is the honest out-of-distribution picture; the shipped directions use all 192 captured items and the suite is the held-out test.

## 5. Controls on the same support (64-item control set: 32 stress ambiguous, 16 stress resolved, 8 practice calls, 4 NO_TOOL, 4 explicit-missing; `analysis/controls.md`)

| edit | ambiguous -> NEED_CLARIFICATION (32) | resolved calls correct (16) | practice calls correct (8) | NO_TOOL (4) | explicit-missing (4) | total lenient (64) | vs g23v3: w->r / r->w / token-identical | vs Base: r->w |
|---|---|---|---|---|---|---|---|---|
| Zero (g23v3) | 9/32 {'clarify': 9, 'call': 16, 'answer': 6, 'prose': 1} | 16/16 {'call': 16} | 8/8 {'call': 8} | 4/4 {'answer': 4} | 4/4 {'clarify': 4} | 41/64 | 0 / 0 / 64 | 0 |
| Half (a3=0.05, b=0.5) | 24/32 {'clarify': 24, 'call': 8} | 16/16 {'call': 16} | 8/8 {'call': 8} | 4/4 {'answer': 4} | 4/4 {'clarify': 4} | 56/64 | 15 / 0 / 49 | 0 |
| Intended (g23v4) | 27/32 {'clarify': 27, 'call': 5} | 16/16 {'call': 16} | 8/8 {'call': 8} | 4/4 {'answer': 4} | 4/4 {'clarify': 4} | 59/64 | 18 / 0 / 46 | 0 |
| Double (a3=0.2, b=2) | 27/32 {'clarify': 27, 'call': 5} | 16/16 {'call': 16} | 8/8 {'call': 8} | 4/4 {'answer': 4} | 4/4 {'clarify': 4} | 59/64 | 18 / 0 / 46 | 0 |
| Reversed (a3=-0.1) | 0/32 {'answer': 12, 'call': 20} | 16/16 {'call': 16} | 8/8 {'call': 8} | 4/4 {'answer': 4} | 4/4 {'clarify': 4} | 32/64 | 0 / 9 / 52 | 0 |
| Random norm-matched (seed 17) | 0/32 {'prose': 9, 'call': 20, 'answer': 3} | 0/16 {'call': 16} | 0/8 {'call': 8} | 4/4 {'answer': 4} | 0/4 {'prose': 4} | 4/64 | 0 / 37 / 6 | 24 |

Predictions (replay) were written into `notes.md` before any control pass was read and held: half dose keeps stage 1/transport but weakens the readout (24/32, the misses
stay calls), double is token-identical to the intended dose on all 64 items (a larger dose cannot open a shut stage-1 gate and breaks nothing), reversed output sign gives
0/32 (12 become NO_TOOL, the nine items g23v3 had repaired are lost), and norm-matched random stage-1/stage-2 vectors on the same support (g23v3 kept) are catastrophic:
4/64, every call broken (0/16 resolved, 0/8 practice calls), 37 right→wrong vs g23v3 — the random gate fires at arbitrary prompt positions (replay max activation 521) and
writes random markers that all later attention reads. A perturbation of this size is destructive unless it is the structured one. Two deviations from the replay's
pre-registered expectations: the half-dose misses came out as calls rather than the NO_TOOL the replay suggested (NE−NO −11.8 on captured items), and the random control's
replay kept the g23v3 argmax on all 108 non-ambiguous *decision* positions while generation broke every call — the replay covers the decision position only, so random
stage-1 firing at generated positions (replay max activation 521) is outside what it can predict. The intended edit's stage-1/J3 activations at generated positions are
≤ 0.016, which is why its generated streams match the replay-based expectation (all non-ambiguous streams identical).

## 6. Export and fresh stock reload

`stock_reload.py --edit g23v4` copied the base HF checkpoint, replaced only the six BF16 tensors (`artifacts/exports/g23v4/`, manifest with SHA-256 of base and edited
safetensors; every other tensor byte-identical) and ran a fresh `python -I` MLX-LM subprocess importing no campaign module on 16 practice + 16 stress items for both
checkpoints: original 32/32 and edited 32/32 token-identical to the in-process runs (pass s01).

## 7. Accounting, limitations

The g01 capture's prediction ("recorded original continuations reproduced") held: 99.46 % of the 4,246 teacher-forced gold tokens rank 1 (4,223; median log-probability −0.30, minimum −2.97).
Ledger: 20/50 passes (d01 charged, no executions; d01b–d03 diagnosis 192; g01 generated-position capture 64; c01 calibration 64; g02–g09 generation 512; c02 64;
s01 stock 64; k01–k04 controls 256), 1,184 executions, 688,144 input / 34,385 generated tokens, hash chain verified; peak RSS 2.42 GiB, backend peak 1.87 GiB (never summed);
shared accelerator locks polled, never broken. CPU analyses (probes, replay simulator) are uncharged, like qwen-6's closed-form simulator.

Limitations: (1) phrasings without a disjunction/indecision cue ("torn between X and Y") are mostly missed — they fail gracefully (the model calls as base/g23v3 do);
(2) a direction fit on discovery families only transfers to unseen families with a thinner margin (1.2 vs 3.9); (3) the marker also fires at explicit-missing clause
ends on some items (harmless: it strengthens their gold NEED_CLARIFICATION) and at other tokens inside ambiguous clauses (harmless: same payload); (4) like g23v3, prose
is not bit-identical (23/32) though intact; (5) validated under the `standard` framing only, single turn, no tool results in context, no vision.

## 8. Reproduction (from `/Users/scasella/Downloads/qwen-7`, `P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python`, `HF_HUB_OFFLINE=1`)

```
$P build_triples.py                                                        # D1/D2/D3 triples + 64-item stress suite (deterministic)
$P fp2.py teacher --set triples_D1 --model original --continuation none --roles --layers 23 --hfull 19,23 --mlpin 18,22,23 --colstats 15,19,21,22,23 --attnrows --label d01b_triples_D1 ...   # D2, D3 likewise
$P fp2.py teacher --set prose_qa+prose_qa2+regression2 --prompt regression --model original --continuation fromqwen6:...  --mlpin 22,23 --label g01_genpos ...
$P -m analysis.diag d01b_triples_D1; $P -m analysis.diag2 d01b_triples_D1 d02_triples_D2 d03_triples_D3     # Hypotheses A/B/C
$P -m analysis.stage1; $P -m analysis.stage1d; $P -m analysis.stage1e; $P -m analysis.transport; $P -m analysis.bias   # stage-1 / transport / bias designs
$P g23v4_build.py --refit; $P -m analysis.sim4 '{"a3":0.1,"rho":1.0}'; $P build_g23v4.py                    # directions, exact simulation, edits + controls
$P fp2.py teacher --set triples_D1 --model g23v4 --continuation none --roles --layers 22,23 --attnrows --label c01_calib_D1 ...; $P -m analysis.calib
sh runs/chain1.sh; sh runs/chain2.sh; sh runs/chain3.sh                                                     # generation, stress calibration + stock reload, controls
$P -m analysis.tables4; $P -m analysis.controls; $P -m analysis.stress_calib; $P -m analysis.stock_cmp
```
Artifacts: `edits/g23v4.{npz,spec.json,vectors.npz}` (+ `_half`, `_double`, `_rev`, `_rand17`), `artifacts/exports/g23v4/`, `data/stress.jsonl` + `.meta.json`,
`data/triples_D{1,2,3}.jsonl`, `runs/*.json|npz`, `analysis/*.json|md`, `notes.md`, `ledger.jsonl`.
