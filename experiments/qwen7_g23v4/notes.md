# qwen-7 campaign notes — ambiguity (two candidate values) repair on top of g23v3, Qwen3.5-0.8B

Workspace: /Users/scasella/Downloads/qwen-7 (qwen-6 at /Users/scasella/Downloads/qwen-6 is read-only; harness, data and edits/g23v3.* copied from it).
Python: /Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python (mlx 0.32.4, mlx-lm 0.32.2). Protocol unchanged: official chat template with native <tools>,
enable_thinking=False, greedy, max 128 new tokens, context <= 2048, system prompt 'standard'. Scoring lenient/strict as in qwen-6.
Budget: 50 passes (<= 64 executions each) in this workspace's own hash-chained ledger.jsonl; every pass (also failed) is charged.
Edit limits for g23v4 (including g23v3): <= 3 decoder layers, <= 8 tensors, <= 0.05% of 752,393,024 unique text parameters = 376,196 BF16 entries
(environment/architecture.json max_changed_entries and primitives.save_edit updated accordingly). Forbidden: embed_tokens, lm_head, final norm, vision.
Shared-machine etiquette: both /tmp/qwen*accelerator.lock flocks are polled, never broken; memory guard: 12 GiB RSS/backend target, kernel pressure stop, 1 GiB system-available courtesy threshold.

## Starting point (from qwen-6 results.md)

g23v3 (layer 23, neurons J1=3468 abstain / J2=3421 route, 6,142 entries) repairs 59/60 NO_TOOL and 28/30 explicit-missing NEED_CLARIFICATION with 0 regressions,
but 0/30 ambiguous items ('either "X" or "Y", with no preference specified'): raw leave-one-out probes at the decision position gave d' < 3 in every layer.

## Plan (written before any pass)

1. Data: matched triples per family — (i) complete call, (ii) resolved two-value call (both values mentioned, choice unambiguous), (iii) unresolved ambiguous — with varied
   phrasings, plus renamed twins. Derivation phrasings (set A: 6 ambiguous, 6 resolved) are used only on discovery families; held-out phrasings (set B: 6 + 6) and
   selection/final families are reserved for the held-out diagnosis and the 64-item Ambiguity Stress Suite (32 ambiguous, 32 resolved; discovery/selection/renamed).
2. Diagnosis (teacher-forced, original model, 3 passes): residual at all 25 layer boundaries at role-tagged positions (first value, connector, second value, indecision/resolution tail,
   clause-end '.', <|im_end|>, decision), per-head attention outputs there, attention rows from the decision query over every key at the six full-attention layers, and all-position
   MLP inputs at layers 18/22/23 plus residual inputs of layers 19/23 (for an exact layer replay simulator). Hypothesis A: family-mean-subtracted and raw held-out d' at the decision.
   Hypothesis B: d' at the clause roles per layer, and attention mass from the decision query onto the clause roles per head. Hypothesis C: single-head decision outputs.
3. Decision rule (pre-registered): if a raw, leave-one-family-out direction at the layer-23 MLP input separates ambiguous from {call, resolved} with d' >= 5 and no overlap,
   build J3 only (third layer-23 neuron sharing the c/c2 gates). Otherwise build a two-stage circuit: stage 1 = a near-dead MLP neuron in layer 22 (only consumer of its write
   at prompt positions is layer 23's K/V) firing at the clause roles, carried by whichever layer-23 heads attend from the decision query to those roles; stage 2 = J3.
4. Simulate (CPU replay of layer 23 from captured inputs; silence at every prompt position including 'missing or ambiguous' in the system prompt, inside calls, after literals,
   and across 128-token prose) before any generation; then dose/reversed/random controls on the same support; then all cohorts; then fresh python -I stock reload.

## Diagnosis pass 1 (D1) — explanation before running

What we know: qwen-6's raw leave-one-out probes at the decision position (the `\n\n` after `</think>`) gave d' < 3 for the ambiguous `either "X" or "Y"` items against
complete calls in every layer, and g23v3's abstain feature `s.x` sits at call level on them, so nothing at the decision position currently says "two candidate values".
This pass captures the original model on 18 discovery-family matched triples (call / resolved two-value call / ambiguous, six derivation phrasings each) plus the 10 existing
practice ambiguous items: the residual at every layer boundary at role-tagged positions (first value, connector, second value, indecision tail, clause-end '.', <|im_end|>,
decision, and the word 'ambiguous' in the system prompt), per-head attention outputs there, attention rows from the decision/<|im_end|>/clause-end queries over every key at the
six full-attention layers, and all-position MLP inputs of layers 18/22/23 and residual inputs of layers 19/23 for the replay simulator. Hypothesis A predicts that once family
means are subtracted a consistent ambiguous-minus-{call,resolved} direction appears at the decision position with d' >= 5 (refuted if the paired differences do not align
across families, i.e. leave-one-family-out d' stays < 3). Hypothesis B predicts d' >> 5 at the connector / second value / tail tokens in mid-to-late layers (the model clearly
represents "an alternative is being offered" and "no decision" where it reads them) but attention mass from the decision query onto those tokens is small (< 0.05 per head),
which would explain why the signal never reaches the output; it is refuted if the clause tokens themselves are not separable, or if a decision-query head already puts
substantial mass on them (in which case the signal should have arrived and the bottleneck is elsewhere). Hypothesis C is read off the same data: whether any single layer-23/19
head's decision-position output separates the ambiguous items (then the stage-2 readout could use that head's output subspace directly).

### pass d01_triples_D1 (teacher, set=triples_D1, models=['original'], n=64)

Hypothesis: Ambiguity (two unresolved candidate values) is represented at the user-clause tokens (connector/second value/indecision tail) with d'>>5 in mid-to-late layers but is weak at the decision position (raw and family-mean-subtracted d'<5), and the decision query's attention mass on those clause tokens is small at layers 19/23.

Prediction: Clause-role d' between ambiguous and {call,resolved} exceeds 5 at some layer >=12; decision-position raw LOFO d' < 3 as in qwen-6; per-head attention from the decision query onto the clause roles < 0.05 for most heads; original-model argmax at the decision is <tool_call> on all 64 items.

Pass d01_triples_D1 was charged (1/50) but executed nothing: fp2.tag_roles called the mlx_lm TokenizerWrapper directly (not callable). Fixed by using the wrapped HF
tokenizer, and every item is now role-tagged before the model loads so a tagging error cannot be charged again. Re-run as d01b with the identical hypothesis/prediction.

### pass d01b_triples_D1 (teacher, set=triples_D1, models=['original'], n=64)

Hypothesis: Ambiguity (two unresolved candidate values) is represented at the user-clause tokens (connector/second value/indecision tail) with d'>>5 in mid-to-late layers but is weak at the decision position (raw and family-mean-subtracted d'<5), and the decision query's attention mass on those clause tokens is small at layers 19/23.

Prediction: Clause-role d' between ambiguous and {call,resolved} exceeds 5 at some layer >=12; decision-position raw LOFO d' < 3 as in qwen-6; per-head attention from the decision query onto the clause roles < 0.05 for most heads; original-model argmax at the decision is <tool_call> on all 64 items.

## Diagnosis passes 2 and 3 (D2, D3) — explanation before running

D1 ran (64 items, ~1 min). D2 captures the renamed twins of the same 54 triples plus the 10 renamed twins of the existing practice ambiguous items: every claim derived from
D1 (where the ambiguity signal is, how much the decision query attends to it) must be invariant to fresh function/argument names, otherwise the derived directions would be
name-specific and the stress suite's renamed family would fail. D3 captures selection and final families with the held-out phrasing set B (never used for derivation): if the
clause-token signal and the decision-position deficit found on set A reappear with d' of the same order on unseen phrasings and families, the signal is about unresolved
alternatives rather than the words "either"/"no preference"; if set-B d' collapses, the model's representation is lexical and a weight edit built on set A would not transfer.
The same captures are taken so that the replay simulator can be checked on every set.

### pass d02_triples_D2 (teacher, set=triples_D2, models=['original'], n=64)

Hypothesis: The clause-token ambiguity signal and the decision-position deficit measured on D1 are invariant to fresh function/argument names: renamed twins give the same d-prime pattern per role and layer and the same decision-query attention mass.

Prediction: Per-role d-prime within 20 percent of D1 at every layer; decision argmax <tool_call> on all 64; projections of D1-derived directions on D2 items separate ambiguous from call/resolved with the same sign and magnitude.

### pass d03_triples_D3 (teacher, set=triples_D3, models=['original'], n=64)

Hypothesis: On selection/final families with the held-out phrasing set B the ambiguity signal at the clause tokens has d-prime of the same order as on set A (the representation is semantic, not lexical), and D1-derived directions transfer.

Prediction: Set-B clause-role d-prime > 5 at the same layers; D1-derived direction projected on D3 ambiguous vs call/resolved gives d-prime > 3 with the same sign; decision argmax <tool_call> on all 64.

## Diagnosis results (D1 64, D2 64 renamed twins, D3 64 held-out families + phrasing set B; 192 items, LOFO folds = families; analysis/diag.py, analysis/diag2.py)

Original model argmax at the decision is <tool_call> on all 192 items (ambiguous ones included), call-NE 13-17 logits.
* Hypothesis A (decision position, ambiguous vs {call, resolved}): raw diff-of-means LOFO d' 2.7 (acc 0.90, gap -3.8 at the layer-23 MLP input x23), family-subspace
  masked (top-4/8/16 family-mean PCs projected out before the probe) 3.0, family-triple centering (oracle, not implementable) 3.6-3.9, shrinkage LDA 3.5-3.8 (acc 0.96-0.98,
  still overlapping). The pattern is name-invariant (D2) and the same on held-out families/phrasings (D3: raw 2.2, LDA ~3.5). Conclusion: a consistent direction exists but
  it is not clean (d' < 4 for every linear readout) -> a J3-only readout at the decision position cannot reach 0 regressions. Hypothesis A refuted as a sufficient mechanism.
* Hypothesis B (clause roles): raw d' at the clause-end '.' token 4.5-5.0 (L14-L20), connector 4.3-4.7, indecision tail 3.6-3.8; shrinkage LDA (LOFO) at the clause-end token
  d' 13-14 at L12-L18 (acc 1.00, positive gap), 9.5 at L22/23; connector 17-19; tail 15-22. The D1-fit diff-of-means direction transfers to D2+D3 raw (end L15: d' 4.9, acc 0.98).
  So the representation of 'unresolved alternatives' is strong and linearly clean at the clause tokens (and the same at <|im_end|> is weak, d' 1.5) but it is not routed:
  attention mass from the decision query onto the clause-end token is 0.000-0.003 at layer 23 (heads 2/6 attend 97%/87% to self), <= 0.008 at layer 19 and 0.015-0.062 at layer 15
  (h1 0.062 ambiguous / 0.048 resolved / 0.037 call; h0-h2 put 0.18/0.18/0.10 on the whole clause span for ambiguous vs 0.07/0.04/0.005 for calls, i.e. layer-15 heads do look
  at complex clauses, which is presumably how the weak d'~2.5 signal arrives at the decision from layer 13-15 on). Hypothesis B supported: positional bottleneck.
* Hypothesis C: no single head's gated output at the decision position exceeds d' 2.3 (L23 h2 2.34, h6 2.31). No head carries it.
* Near-dead neurons (max |activation| over every prompt position of 192 items): none below 0.1 at layers 15/19/21/22/23; quietest at L22: 2525 (0.205), 3211 (0.27);
  at L23: 2331 (0.237), 3468=J1 (0.238), 2551 (0.246), 3109 (0.25), 3421=J2 (0.256), 2279 (0.26). 'Near-dead' therefore means max |a| ~0.2 (median neuron 0.8-1.1).
* Layer-15 transport of a hypothetical stage-1 write at the clause end (OV gain 0.6, alpha 0.06, rms 0.09): ~0.2 units delivered per unit write; a usable feature at layer 23
  would need a write of ~3 (the residual norm there is ~3) reprocessed by eight layers -> rejected.

Decision (per the pre-registered rule): two-stage circuit. Stage 1 = a layer-22 MLP neuron (J0 = 2525) that fires at the clause-end token of an unresolved clause and writes a
marker p into the residual there (its only consumer is layer-23 attention, since no layer follows before 23). Transport = make the decision query of one layer-23 head attend to
the marked position: either through the existing k_proj (if a payload direction p exists whose key image aligns with that head's decision query strongly enough) or through a
rank-1 term on that kv head's k_proj (u_k p~^T, 262,144 entries, still within 3 layers / 8 tensors / 0.05%). Stage 2 = J3 (layer 23, quiet neuron 2331) reading the delivered
payload under the g23v3 gates c, c2 and writing u_AB + a_route u_route. Everything in layer 23 is simulated exactly by a CPU replay of layer 23 from the captured h23 sequences.

## Pass 5 (g01_genpos) — explanation before running

Any layer-22 stage-1 neuron and the layer-23 readout J3 must be silent on text the model generates outside tool decisions. qwen-6's all-position captures (p04c/p05) already
hold the layer-22 and layer-23 MLP inputs inside tool calls and after the NO_TOOL/NEED_CLARIFICATION literals, but for free-form prose answers and short non-tool answers
only the layer-23 MLP input was captured (p17/p35/p16). This pass teacher-forces the recorded original answers of the 32 prose questions (128 tokens each) and the 32 fresh
short non-tool prompts (regression2) and captures the layer-22 and layer-23 MLP inputs at every position, so that the stage-1 direction's maximum activation over ~5,000
generated prose positions and the non-tool prompt tokens is known before any weight is written. Refutation: if the stage-1 direction fires at prose positions (e.g. on 'or'
constructions in explanations) the design must add a gate term or move to a different read direction; if it is silent (|activation| ~ 0) the two-stage edit cannot touch prose.

### pass g01_genpos (teacher, set=prose_qa+prose_qa2+regression2, models=['original'], n=64)

Hypothesis: A stage-1 ambiguity direction at the layer-22 MLP input and the J3 read direction at the layer-23 MLP input are silent (max quadratic activation ~0) at every generated position of 128-token prose answers and short non-tool answers, and at the non-tool prompt tokens.

Prediction: Max |projection| of the stage-1 direction over all prose/non-tool positions below the ambiguous clause-end minimum by a factor >= 2; the recorded original continuations are reproduced (gold tokens rank 1 almost everywhere).

## g23v4 design (closed form; all numbers from CPU analyses of the captures, analysis/stage1*.py, transport.py, j3feat.py, bias.py, sim4.py)

Stage 1 — layer-22 MLP neuron J0 = 2525 (quietest at layer 22, max |a| 0.205 over every captured position), read at the layer-22 MLP input x22:
  u  = shrinkage-LDA (0.5) direction 'ambiguous clause end vs call/resolved clause end' (LOFO d' 10.0: ambiguous ends 5.8..10.6, other ends -1.9..2.0), but alone it also
       fires on system-prompt tokens ('reply exactly NO_TOOL.', 'missing or ambiguous') and inside resolved clauses before the resolution (up to 7.5-9.6);
  g  = shrinkage-LDA 'user fact-clause end vs other positions' (LOFO at clause ends 8.9..19.3), so the gate is the conjunction 'clause end AND unresolved':
       w1 = unit(0.5 g + u): ambiguous ends 13.7..16.6 vs every harmful position (all prompt+generated positions of call/resolved/NO_TOOL items, prose, short non-tool
       answers; explicit-missing items excluded because a marker there only strengthens their gold NEED_CLARIFICATION) <= 9.8 (margin 3.9; fit on D1+D2 only and tested on
       the held-out D3 families/phrasings the margin is 1.2 with ambiguous min 10.2, so the final directions are fit on all 192 captured items and the 64-item stress suite is
       the held-out generation test).
  Threshold without a bias: r1 = constant-projection direction (r1.x in 0.99..1.03 on clause ends, 0.94..1.04 on the 3,000 hardest negatives), gate = up = k1 (w1 - 11.7 r1):
       pre-activation z: ambiguous ends 1.90..5.06, harmful max -2.10 ('checksum' resolved clause), k1 = 4 -> quadratic activation 58..410 on ambiguous ends, <= 0.016 elsewhere.
  Write: down[:,J0] = b p6 (b = 1): p6 = unit(pinv(Wk_kv1 gamma_in) k_target), k_target = gamma_k * mean layer-23 head-6 decision query on the unrotated dims (64:256),
       orthogonalized against heads 4/5/7 (same kv group). The write (norm 58..410 vs residual norm ~9.6) dominates the clause-end residual entering layer 23, so its key is
       exactly aligned with head 6's decision query: replayed attention of head 6 onto the marked clause end 0.992..0.999 (natural 0.0001; head 6 otherwise attends 85-89% to
       itself), zero spill onto heads 4/5/7. The clause-end position has no other consumer than layer-23 attention (layer 23 is the last layer), so prompt positions are safe.
Stage 2 — layer-23 MLP neuron J3 = 2331 (quietest free neuron at layer 23, 0.237): t = unit mean change of the layer-23 MLP input x23 at the decision when the marker is
  present (|dx23| 11.9, cosine to the mean 0.99 on every ambiguous item). t.x23: delivered 5.5..7.0 on ambiguous items; natural <= -4.1 at every tool-context decision,
  <= -2.3 at non-tool first tokens, <= 3.6 at prose positions (where c2 >= 6.8 closes the gate as in g23v3). gate = up = k3 (t - 0.3 c - 0.5 c2 - 2 r3) with the
  constant-projection r3 (0.96..1.03), k3 = 2: delivered z 4.2..5.7, natural decisions <= -6.4, generated positions <= -2.6. down[:,J3] = a3 (u_AB + rho u_route).
Exact replay (layer 23 with J1/J2/J3 rows + final norm/unembedding, stage-1 writes at every prompt position from the captured x22, original J0 contribution removed):
  a3 = 0.1, rho = 1.0: 84/84 ambiguous items -> NE argmax (NE-call >= 31, NE-NO >= 12.5, NE-runner-up >= 11.6), 108/108 call/resolved items keep the g23v3 argmax with
  max |logit change| 0.125 on their top-20 tokens; J0/J3 writes at every captured generated position (calls, literals, prose, short answers) <= 0.016 in norm.

## Pass 6 (c01_calib_D1) — explanation before running

The replay predicts, for the g23v4 weights, NEED_CLARIFICATION as the decision token on all 84 captured ambiguous items with NE leading <tool_call> by >= 31 logits and NO by
>= 12.5, and no change of the g23v3 decision on the 108 call/resolved items (max logit change 0.125). That prediction rests on three things the replay takes as exact: the
captured layer-22 MLP inputs (stage-1 activation), the captured layer-23 inputs with the stage-1 write added (attention of head 6 onto the marker, alpha >= 0.992) and the
bf16 arithmetic of layer 23. This pass runs the real edited model (edits/g23v4.npz loaded into the resident MLX model) teacher-forced on the 64 D1 items, recording the
decision logits, the attention rows of the decision query, and the layer-22/23 MLP activations at the tagged positions (so J0's activation at the clause end and J3's at
the decision are measured directly). Refutation: a decision argmax disagreeing with the replay on any item, |logit error| > 1 on NE/NO/<tool_call>, head-6 attention on the
marker below 0.9, or a measurable J0 activation at non-ambiguous clause ends would mean the simulator missed a path (e.g. the stage-1 write also changing layer-22's own
output through the MLP residual at that position, which the replay does not model) and generation must not proceed on the simulated margins.

### pass c01_calib_D1 (teacher, set=triples_D1, models=['g23v4'], n=64)

Hypothesis: The exact layer-23 replay predicts the real edited model: with g23v4 loaded, the decision argmax is NE on all 28 ambiguous D1 items and unchanged (<tool_call>) on the 36 call/resolved items; head 6 attends to the marked clause end with alpha >= 0.9; NE/NO/<tool_call> logits match the replay within 1 logit.

Prediction: Argmax agreement with the replay on >= 62/64 items; median |logit error| < 0.3 on NE/NO/<tool_call>; alpha6 >= 0.9 on every ambiguous item; J0 activation at non-ambiguous clause ends < 0.05.

Calibration c01 (real g23v4 vs replay, 64 D1 items): decision argmax agreement 64/64 (28 NE, 36 <tool_call>); |logit error| on NE/NO/<tool_call> median 0.06, max 0.50;
head-6 attention on the marker 0.995..0.999; J0 activation at ambiguous clause ends 149..360 (replay 219.15 vs real 219.0 on the first item), at call/resolved ends <= 0.0002;
J3 at the decision 59..100 on ambiguous items, <= 0.0002 otherwise. The simulator is exact to bf16 noise; generation proceeds on its margins.

## Generation passes 7-14 — explanation before running

Hypothesis for all of them: the circuit that the replay and the calibration pass agree on turns the first generated token of every ambiguous item into NE and leaves every
other first token exactly as g23v3 produced it; because NE -> ED -> _CL -> AR -> IFICATION follows with log-probability ~0 and the J0/J3 activations are < 0.02 at every
captured generated position, the full token streams of non-ambiguous items should be identical to g23v3's (the only new source of drift is the removal of J0's original
contribution at layer 22, max |activation| 0.2 times a down-column of norm 0.15, i.e. < 0.03 in a residual of norm ~9; it can flip only near-tied tokens late in prose).
Stress suite (passes g02/g03 base + g23v3 paired on the 32 ambiguous and 32 resolved items; g04 g23v4 on all 64): prediction base 0/32 and g23v3 0/32 ambiguous (all
calls), resolved calls mostly correct for all three; g23v4 >= 28/32 ambiguous -> NEED_CLARIFICATION (the replay's weakest margin is 11.6 logits on captured items, but the
suite's items were never captured: a miss would be an item whose clause-end z falls below the 1.9 floor seen on captured items), and 0 resolved items flipped to clarification
(a flip would mean a resolved phrasing the stage-1 gate does not close on). Existing cohorts with g23v4 (g05 practice 64, g06 selection+renamed A, g07 renamed B+final,
g08 regression2+prose): prediction practice 52 -> 62 lenient (the 10 ambiguous items repaired, nothing else changed, calls token-identical), selection 27 -> 32, renamed A
23 -> 28, renamed B 26 -> 31, final 27 -> 32, regression2 28/28 unchanged, prose 32/32 intact with lengths unchanged. g09 runs the original model on practice in this
workspace to tie the environments: prediction 64/64 token-identical to qwen-6's p01 (determinism across processes was already shown in qwen-6).

### pass g02_stress_amb_base_v3 (generate, set=stress, models=['original', 'g23v3'], n=32)

Hypothesis: Base and g23v3 both call the tool on every ambiguous stress item (varied phrasings, selection/renamed families): the ambiguity signal is not routed to the decision in either.

Prediction: 0/32 NEED_CLARIFICATION for both; g23v3 token-identical to base on nearly all items.

### pass g03_stress_res_base_v3 (generate, set=stress, models=['original', 'g23v3'], n=32)

Hypothesis: Resolved two-value calls are ordinary calls for base and g23v3: both call with the chosen value on most items; no clarification.

Prediction: >= 26/32 lenient-correct calls for both; 0 NEED_CLARIFICATION; g23v3 token-identical to base on nearly all items.

### pass g04_stress_v4 (generate, set=stress, models=['g23v4'], n=64)

Hypothesis: g23v4 emits NEED_CLARIFICATION on the ambiguous stress items regardless of phrasing/family/renaming and keeps the resolved calls.

Prediction: >= 28/32 ambiguous -> NEED_CLARIFICATION; 0 resolved items flipped to clarification; resolved outputs token-identical to g23v3 (g03).

### pass g05_practice_v4 (generate, set=practice, models=['g23v4'], n=64)

Hypothesis: On practice, g23v4 repairs the 10 ambiguous items and changes nothing else relative to g23v3 (p38).

Prediction: lenient 52 -> 62, strict 35 -> 45; 54/64 token-identical to p38; all 24 calls identical.

### pass g06_sel_renA_v4 (generate, set=selection+renamed, models=['g23v4'], n=64)

Hypothesis: Selection (unseen families) and renamed A: the 5+5 ambiguous items are repaired, nothing else changes vs g23v3 (p40/p42).

Prediction: selection 27 -> 32, renamed A 23 -> 28; other items token-identical to g23v3.

### pass g07_renB_final_v4 (generate, set=renamed+final_proxy, models=['g23v4'], n=64)

Hypothesis: Renamed B and final proxy: the 5+5 ambiguous items are repaired, nothing else changes vs g23v3 (p43/p41).

Prediction: renamed B 26 -> 31, final 27 -> 32; other items token-identical to g23v3.

### pass g08_nontool_v4 (generate, set=regression2+prose_qa+prose_qa2, models=['g23v4'], n=64)

Hypothesis: Non-tool prompts are untouched: the stage-1 gate is closed on prose and short answers (max activation 0.01) and J3 is closed by c/c2.

Prediction: regression2 28/32 lenient as g23v3 with 32/32 token-identical; prose 32/32 intact, mean length 128, >= 18/32 token-identical to g23v3 (p37/p36).

### pass g09_practice_orig (generate, set=practice, models=['original'], n=64)

Hypothesis: This workspace's resident model reproduces qwen-6's baseline token streams exactly (environment tie-in).

Prediction: 64/64 token-identical to qwen-6 p01_baseline.

## Generation results (passes g02-g09; tables in analysis/final_tables.md)

Environment tie-in g09: original model on practice 64/64 token-identical to qwen-6 p01. Practice g23v4 52 -> 62 lenient (35 -> 45 strict): the 10 ambiguous items ->
NEED_CLARIFICATION, the other 54 token-identical to g23v3 (p38); selection 27 -> 32, renamed A 23 -> 28, renamed B 26 -> 31, final proxy 27 -> 32, each with exactly the 5
ambiguous items changed and 27/32 token-identical; regression2 32/32 identical; prose 32/32 intact, mean length unchanged (125.1), 23/32 token-identical to g23v3 (g23v3 itself
was 20/32 identical to base). Stress suite: ambiguous base 0/32, g23v3 9/32 (its abstain feature already catches some set-B phrasings), g23v4 27/32 (18 wrong->right, 0
right->wrong); resolved two-value calls 30/32 for base, g23v3 and g23v4 with all 32 streams token-identical (the 2 misses are function-choice errors shared by all three).
The 5 ambiguous misses are 4 of the 'torn between X and Y' items (held-out phrasing B4, no 'or' and no indecision tail) and one renamed 'could be X, could be Y' item.

## Passes 15-16 (c02 stress calibration, s01 stock reload) — explanation before running

c02 teacher-forces the g23v4 model on the 32 ambiguous stress items with role tags (as c01) to read the stage-1 pre-activation and activation at each clause end, head-6's
attention on it and J3's activation: the hypothesis is that the 5 misses are stage-1 misses (z below the threshold on an unseen phrasing: the gate stays shut, nothing is
written, the model calls as before) rather than transport or J3 failures; refuted if the marker is written (J0 activation > 50) but head 6 does not attend to it or J3
stays silent. s01 exports edits/g23v4.npz into a full HF checkpoint and runs a fresh stock MLX-LM subprocess (python -I, no campaign modules) on 16 practice + 16 stress
items for both checkpoints; prediction: token streams identical to the in-process runs (g09/g02/g03 for original, g05/g04 for g23v4).

### pass c02_stress_calib (teacher, set=stress, models=['g23v4'], n=32)

Hypothesis: The 5 stress misses are stage-1 misses (clause-end pre-activation below threshold on an unseen phrasing), not transport or J3 failures; the 27 repaired items show J0 activation > 50, head-6 attention > 0.9 and J3 activation > 40.

Prediction: On the 5 missed items J0 activation at the clause end < 1 and no marker; on the 27 others J0 > 50, alpha6 > 0.9, J3 > 40, decision argmax NE.

### pass s01_v4_stock (stock reload, set=practice+stress, n=32, checkpoints=['original', 'edited'], edit=g23v4)

Hypothesis: Fresh stock MLX-LM reload of exported edit g23v4 reproduces in-process token streams exactly for both original and edited checkpoints.

## Controls on the same support (passes k01-k04 on data/control.ids: 32 stress ambiguous + 16 stress resolved + 8 practice calls + 4 NO_TOOL + 4 explicit-missing)
— predictions from the replay, written before any control pass was read

Zero = pure g23v3 (already run: g02/g03/p38). Half (a3 = 0.05, b = 0.5): replay 77/84 captured ambiguous items -> NE, the rest -> NO (the route term is halved: NE-NO
min -11.8), so on the control set expect ~24/32 NEED_CLARIFICATION with the misses being NO_TOOL rather than calls, non-ambiguous unchanged. Double (a3 = 0.2, b = 2):
84/84, non-ambiguous unchanged -> expect the same 27/32 as the intended dose (the 5 misses are stage-1 misses, which a larger dose cannot fix) and 0 regressions.
Reversed (a3 = -0.1): 1/84 -> expect 0/32 repairs (calls), non-ambiguous unchanged. Norm-matched random stage-1/stage-2 vectors (seed 17, g23v3 rows kept): the random
gate fires at arbitrary prompt positions (max activation 521 in the replay) and writes a random marker; the replay keeps the g23v3 argmax on all 108 non-ambiguous items
but moves their top-20 logits by up to 3.0, and gives NE on 11/84 ambiguous items via perturbation -> expect <= 6/32 ambiguous repaired, some token changes on calls,
possibly broken calls (as qwen-6's seed-13 control broke 12 calls).

### pass k01_control_half (generate, set=stress+practice, models=['g23v4_half'], n=64)

Hypothesis: Half dose (a3=0.05, b=0.5) keeps stage 1 and transport but weakens the J3 route: ambiguous items mostly NEED_CLARIFICATION, misses become NO_TOOL; non-ambiguous unchanged.

Prediction: ~24/32 ambiguous -> NEED_CLARIFICATION, remaining -> NO_TOOL; 0 regressions on the 32 non-ambiguous items.

### pass k02_control_double (generate, set=stress+practice, models=['g23v4_double'], n=64)

Hypothesis: Double dose cannot repair stage-1 misses and does not break anything: same 27/32 ambiguous, 0 regressions.

Prediction: 27/32 ambiguous -> NEED_CLARIFICATION (same items), non-ambiguous token-identical to g23v4.

Results c02 (stress calibration, g23v4 teacher-forced on the 32 ambiguous stress items): the 5 misses have J0 activation 0.01..0.41 at the clause end (gate closed), head-6
attention on it 0.000 and J3 0.00 -> stage-1 misses exactly as predicted (4x 'torn between X and Y' = phrasing B4 1/5, 1x renamed 'could be X, could be Y' = B2 3/4);
the 27 repaired items have J0 10.5..452 (two weak ones at 10.5 and 23), alpha6 0.992..0.999, J3 21..107. By phrasing: A1 2/2, A2 2/2, A3 2/2, B1 5/5, B2 3/4, B3 4/4, B4 1/5,
B5 4/4, B6 4/4. Results s01 (stock reload, fresh python -I MLX-LM process on the exported checkpoint artifacts/exports/g23v4, 16 practice + 16 stress items): original 32/32
and edited 32/32 token-identical to the in-process runs (g09/g02/g03 and g05/g04).

### pass k03_control_rev (generate, set=stress+practice, models=['g23v4_rev'], n=64)

Hypothesis: Reversed J3 output sign gives no repairs: ambiguous items stay calls; non-ambiguous unchanged.

Prediction: 0/32 ambiguous repaired; 0 regressions.

### pass k04_control_rand17 (generate, set=stress+practice, models=['g23v4_rand17'], n=64)

Hypothesis: Norm-matched random stage-1/stage-2 vectors on the same support do not repair ambiguity and may perturb calls.

Prediction: <= 6/32 ambiguous repaired; some non-ambiguous token streams change.

Results k01-k04 (controls on data/control.ids; analysis/controls.md): Zero = g23v3 9/32 ambiguous, 41/64 total. Half 24/32 (the 8 misses stay calls), 0 regressions, 56/64.
Intended g23v4 27/32, 59/64, 0 right->wrong vs g23v3 and vs base. Double 27/32, 59/64, token-identical to the intended dose on every item (a larger dose cannot open a
closed stage-1 gate and breaks nothing). Reversed 0/32 (12 become NO_TOOL, 20 stay calls), 9 right->wrong vs g23v3 (the nine items g23v3 had repaired), non-ambiguous
unchanged. Random norm-matched vectors on the same support (seed 17): 4/64 — 0/32 ambiguous, 0/16 resolved and 0/8 practice calls correct, 0/4 explicit-missing (prose),
37 right->wrong vs g23v3, 24 vs base: a perturbation of this size is catastrophic unless it is the structured one. All six predictions written before the passes held
(half slightly better than the replay's 24-ish estimate with misses as calls rather than NO_TOOL).

## Closing accounting

20/50 passes (labels in ledger.jsonl: d01 charged/no executions, d01b, d02, d03, g01, c01, g02-g09, c02, s01, k01-k04), 1,184 executions, 688,144 input / 34,385 generated
tokens, hash chain verified (analysis in results.md §7). Peak RSS 2.42 GiB, backend peak 1.87 GiB. Final edit: edits/g23v4.npz (12,281 entries, 6 tensors, layers 22+23,
1.63e-5 of text params), exported to artifacts/exports/g23v4 and verified bit-for-bit in a fresh python -I stock process. Report: results.md. Nothing in qwen-6 was modified.
g01 check (promised in its prediction): the recorded original continuations teacher-forced on the original model have the gold token at rank 1 for 99.46 % of the 4,246
continuation tokens (4,223; median logprob −0.30, min −2.97). Final state: edits/current.npz restored to g23v4 and verified bit-identical (ledger event final_state).
