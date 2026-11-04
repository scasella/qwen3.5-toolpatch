# Semantic tool-calling repair of Qwen3.5-0.8B by a two-neuron final-layer weight edit

Workspace: `/Users/scasella/Downloads/qwen-6`. Model: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`, unquantized BF16, native MLX-LM 0.32.2 / MLX 0.32.4 (venv `/Users/scasella/Downloads/qwen-3/toolpatch/.venv`). Protocol throughout: official chat template with native `<tools>` declarations, `enable_thinking=False`, greedy decoding, max 128 new tokens, context <= 2048, one fixed system prompt (`standard`, no format examples). Scoring: *lenient* = semantic correctness with an omitted optional default treated as equivalent to the explicit default; *strict* = additionally no extra arguments. Both are always reported; the decision mechanism is identical under both.

## 1. What goes wrong (diagnosis)

Baseline on 64 practice problems (24 required calls, 20 NO_TOOL explanation requests, 20 NEED_CLARIFICATION requests; 18 discovery families, half with opaque function/argument names, distractor notes, three request phrasings):

| class | baseline lenient | baseline decisions |
|---|---|---|
| required call | 24/24 (strict 7/24: 17 add the optional `detail=brief`) | 24 calls |
| NO_TOOL | 0/20 | 18 false `<tool_call>`, 2 prose |
| NEED_CLARIFICATION | 0/20 | 20 false `<tool_call>` (10 write the literal phrase `not provided` as the argument value, 10 pick one of two offered values) |

The failure is unconditional `<tool_call>` onset: at the decision position (the `\n\n` after `</think>`) `<tool_call>` leads `NO` by 2-15 logits on non-call items and `NO` always leads `NE` by 0.5-5 logits. Once the first literal token is forced, the rest of each literal follows with log-prob ~ 0 (`NO`->`_TOOL`, `NE`->`ED`->`_CL`->`AR`->`IFICATION`), so the whole problem is the first-token decision.

## 2. Localization (teacher-forced captures at the decision position, 60 token-length-matched teacher triples x {original, renamed names} and 64 practice items; all-position captures of the final-layer MLP input)

* **No culprit neurons.** Direct logit attribution of every MLP neuron (24 layers x 3584) to the `<tool_call>`-minus-`NO` margin finds no non-call-selective pusher larger than ~0.2 logits. The call push is a distributed, class-independent prior written mostly by layers 14-23 (layer 20's block adds ~5-6 logits, layer 23 ~2), and explanation requests receive almost the same push as real calls. The failure is a **missing inhibition**, not a spurious excitation.
* **The missing evidence exists in the residual stream.** (A) An "explanation request" direction (teacher diff of class means) separates NO_TOOL items from calls with d' ~ 10-14 at layers 12-16 with zero overlapping practice cases, and it is still linearly present in the residual entering layer 23. (B) "Required value explicitly missing" is separable for `not provided`-type phrasings but entangled with (A). (C) The clarify-vs-explain distinction needed to choose `NE` over `NO` is extremely clean at layers 8-13 (d' ~ 10) and a layer-14 direction still separates the two when read at the layer-23 MLP input (practice 0.79-2.23 vs <= 0.28), yet the model's own output never uses it (`NE`-`NO` ~ -2 for both classes).
* **The ambiguous variant has no feature.** Items offering `either "X" or "Y", with no preference specified` are indistinguishable from complete calls at the output (margins 12.6-14.8 vs 13.9) and in every probed layer (leave-one-out probes d' < 3); the model simply picks one value. Nothing in the network represents "two candidate values = ambiguity", so no readout edit can fix these.
* **Why the write must be in the last layer.** The residual RMS grows from 0.07 (layer 14) to 0.38 (layer 24). A mid-layer write large enough to move the output by ~15 logits would exceed the local residual norm and be reprocessed by nine layers; a layer-23 MLP write is scaled exactly once by the final RMSNorm and, because the KV cache is built from the layer input, affects only the logits at its own position.
* **Name invariance.** Renamed teacher twins (fresh function and argument names, identical token lengths) give near-identical margins and features; all directions are teacher-derived and evaluated per case on practice before use.

## 3. The edit (`g23v3`, final; `g23v2` and `g23_t05_a1.0` = v1 are the earlier candidates whose failures led to it)

Two layer-23 MLP neurons that are near-dead everywhere (max |activation| <= 0.25 at every captured position) are re-purposed: J1 = 3468 (abstain readout) and J2 = 3421 (route readout). Three tensors change, 6,142 BF16 entries = 8.2e-6 of the 752,393,024 text parameters (limit 0.25%). With x = the layer-23 MLP input (post-attention RMSNorm output), `k = 2`, `kr = 1`, `a1 = a2 = 1.0`:

```
gate_proj[J1,:] = k * (s - lam*c - lam2*c2)     up_proj[J1,:] = k * s      down_proj[:,J1] = a1 * u_AB
gate_proj[J2,:] = k * (s - lam*c - lam2*c2)     up_proj[J2,:] = kr * r     down_proj[:,J2] = a2 * (u_NE - u_NO)
activation_J1 = silu(k*(s - lam*c - lam2*c2).x) * (k*s.x)    ~ k^2 (s.x)^2 at a tool-context decision position with s.x > 0, ~0 elsewhere
activation_J2 = silu(k*(s - lam*c - lam2*c2).x) * (kr*r.x)   signed NE-vs-NO route, same gate
lam = 0.3, lam2 = 0.5, k = 2, kr = 1, a1 = a2 = 1.0, t = 0.5, prose_target = 0.5
```
* `s` (abstain feature): unit teacher difference of class means (abstain minus call) at residual layer 18, orthogonalized against the end-of-literal and first-continuation signatures (mean x at the last/first generated position of abstain items minus the call decision mean), then against `ref = mu_call + 0.5 (mu_abstain_clear - mu_call)` so that its zero-crossing sits midway between the call cluster and the clear abstain cluster (practice explanations + explicit-missing, teacher missing).
* `r` (route feature): unit teacher difference of means (clarify minus answer) at residual layer 14, orthogonalized against the midpoint of the explanation and explicit-missing clusters.
* `c` (tool-context gate term, `lam = 0.3`, v2 and v3): unit( ortho( mean non-tool decision x - mean tool-context decision x , mean tool-context decision x ) ). `c.x` is -2.5..3.7 on tool-context decision positions but 17.9..24.7 on non-tool decision positions, so the SiLU gate closes at the first token of non-tool prompts.
* `c2` (generated-position gate term, `lam2 = 0.5`, v3 only): unit( ortho( mean x over every captured generated position - mean tool-context decision x , mean tool-context decision x ) ), where the generated positions are literal continuations, positions inside tool calls, explanatory prose in tool contexts (p17) and the model's own prose answers to 16 non-tool questions (p35). `c2.x` is -2.0..2.9 (sd 1.0) at tool-context decision positions but >= 9.3 (1st percentile) at literal continuations, >= 12.4 inside calls, >= 10.1 at tool-context prose and >= 6.8 at non-tool prose positions (minimum 3.3), so the gate opens only at the decision position: simulated |activation| 0.00 at literal and in-call positions and on non-tool first tokens, and above 1.0 at only 6 of 1,953 non-tool prose positions (max 2.3). Larger `lam2` closes the last leak but flips the borderline inverted-interval call (its `c2.x` is negative), so 0.5 was chosen.
* `u_AB`, `u_NE - u_NO`: minimum-norm least-squares directions in the pre-final-norm residual space (rows E_t * gamma_final) with unit logit effect +1 `NO`, +1 `NE`, -1 `<tool_call>` and -0.5 on the prose starters `I`, `To`, `The` (v2/v3; v1 had 0 there) resp. +1 `NE`, -1 `NO`, and zero effect on 39 competitor tokens (`I To The No None Based This Since Select < <think> Please NOT no Not Need Sure Here Okay You It Yes 1 \n <|im_end|> _TOOL ED <|endoftext|> Hello Certainly Let For In If We A As There Of`).
* Everything is computed in closed form from captured activations and the weights (difference of means, orthogonalization, least squares); no gradients, no optimizer, no fitting of weight updates to task outputs. An exact CPU simulator of the edit (final norm + tied unembedding) predicted the decision token of the real edited model on 64/64 calibration items (median logit error 0.06, max 0.38).

## 4. Failed or refuted hypotheses and negative controls

1. *Culprit neurons / column ablation*: no neuron or small column set drives the false call (DLA <= 0.2 logits), so the suppression-style edits of earlier campaigns had nothing to remove.
2. *Mid-layer rank-1 column edits*: the required write (~4 in residual units) exceeds the whole residual norm at layers 12-16 (~2.2); such edits are attenuated/reprocessed and were not pursued.
3. *Linear, position-gated readout* (gate = decision-position selector, up = signed feature): the SiLU gate cannot carry a bias, continuation positions still leak ~20%, and the linear feature flips a borderline call whose request is genuinely odd (interval with lower bound 46 > upper bound 43). The quadratic readout keeps that call.
4. *Unorthogonalized quadratic readout*: fires at the position after the literal (quadratic activation ~150 at `IFICATION`), which at double dose produces `NENONONENE...`; orthogonalizing `s` against the literal-continuation signatures cut this 15x.
5. *Mean-signature orthogonalization against prose*: the abstain feature genuinely persists throughout explanatory prose (2-6, quadratic 190), so removing a mean direction only drops it to 133; not adopted.
6. *Context term in the up-projection*: flips the sign of the write on non-tool prompts and produces false `<tool_call>` answers there; the gate-side term has no such sign problem.
7. *v1 outside the tool protocol*: the first candidate (quadratic readout, no gate terms) turned one non-tool instruction prompt into `NENE` (exactly the item the simulation flagged) and destroyed all 16 non-tool prose answers (`NONENENE...`), because the abstain feature stays positive while the model explains anything. *v2* (tool-context gate term only) fixed the first tokens (32/32 identical on both non-tool sets) but still corrupted most prose answers (1/16 token-identical). The generated-position term `c2` (v3) fixes this (16/16 and 16/16 intact, lengths unchanged).
8. *Controls on the same two neurons*: zero edit reproduces the baseline token-for-token (64/64); reversed sign gives 0 repairs (the two prose explanations become false calls); norm-matched random gate/up/down directions (three seeds, matched to the v1, v2 and v3 norms) give 0 repairs: two are neutral (24/64, 57-60/64 token-identical) and the third breaks 12 calls (24 -> 12) by writing a random direction inside tool calls, i.e. a perturbation of this size is at best harmless and never helpful unless it is the structured one; half and double doses show the expected monotone dose response, with the double dose hitting the post-literal leakage ceiling.

## 5. Results (all paired on identical items; greedy; lenient / strict as defined above)

Row format: original -> edited. "both" rows ran original and edited weights in the same pass; practice rows compare each edited pass with the baseline pass p01 (the zero edit proved cross-pass determinism token-for-token). Transition counts are lenient; for every structured-edit row the strict transition counts were identical (the strict/lenient difference is the pre-existing optional-argument habit, which the edit neither fixes nor worsens); the random-control rows differ under strict scoring because their perturbations also disturb optional arguments.

| set (edit) | n | lenient correct | strict correct | calls | NO_TOOL | NEED_CLARIFICATION | false calls | wrong->right / right->wrong | token-identical | mean new tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| practice, g23v3 (final) | 64 | 24 -> 52 | 7 -> 35 | 24 -> 24 (n=24) | 0 -> 19 (n=20) | 0 -> 9 (n=20) | 38 -> 12 | 28 / 0 | 36 | 55.8 -> 33.5 |
| practice, g23v3 random | 64 | 24 -> 12 | 7 -> 1 | 24 -> 12 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 35 | 0 / 12 | 24 | 55.8 -> 57.8 |
| practice, g23v2 (intended) | 64 | 24 -> 53 | 7 -> 36 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 9 (n=20) | 38 -> 11 | 29 / 0 | 35 | 55.8 -> 32.5 |
| practice, g23v2 half | 64 | 24 -> 52 | 7 -> 35 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 8 (n=20) | 38 -> 12 | 28 / 0 | 36 | 55.8 -> 33.0 |
| practice, g23v2 double | 64 | 24 -> 53 | 7 -> 36 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 9 (n=20) | 38 -> 10 | 29 / 0 | 34 | 55.8 -> 31.9 |
| practice, g23v2 reversed | 64 | 24 -> 24 | 7 -> 7 | 24 -> 24 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 40 | 0 / 0 | 61 | 55.8 -> 54.7 |
| practice, g23v2 random | 64 | 24 -> 24 | 7 -> 8 | 24 -> 24 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 38 | 0 / 0 | 57 | 55.8 -> 55.3 |
| practice, g23 v1 intended | 64 | 24 -> 52 | 7 -> 35 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 8 (n=20) | 38 -> 10 | 28 / 0 | 34 | 55.8 -> 33.8 |
| practice, v1 zero | 64 | 24 -> 24 | 7 -> 7 | 24 -> 24 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 38 | 0 / 0 | 64 | 55.8 -> 55.8 |
| practice, v1 half | 64 | 24 -> 52 | 7 -> 35 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 8 (n=20) | 38 -> 11 | 28 / 0 | 35 | 55.8 -> 32.5 |
| practice, v1 double | 64 | 24 -> 53 | 7 -> 36 | 24 -> 24 (n=24) | 0 -> 20 (n=20) | 0 -> 9 (n=20) | 38 -> 9 | 29 / 0 | 33 | 55.8 -> 31.2 |
| practice, v1 reversed | 64 | 24 -> 24 | 7 -> 7 | 24 -> 24 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 40 | 0 / 0 | 56 | 55.8 -> 56.4 |
| practice, v1 random | 64 | 24 -> 24 | 7 -> 6 | 24 -> 24 (n=24) | 0 -> 0 (n=20) | 0 -> 0 (n=20) | 38 -> 39 | 0 / 0 | 60 | 55.8 -> 54.5 |
| selection (both), g23v3 | 32 | 12 -> 27 | 4 -> 19 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 17 -> 5 | 15 / 0 | 17 | 56.8 -> 34.0 |
| renamed A (both), g23v3 | 32 | 8 -> 23 | 6 -> 21 | 8 -> 8 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 20 -> 5 | 15 / 0 | 17 | 45.8 -> 24.5 |
| renamed B (both), g23v3 | 32 | 12 -> 26 | 10 -> 25 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 4 (n=10) | 20 -> 6 | 14 / 0 | 17 | 45.9 -> 26.8 |
| final proxy (both), g23v3 | 32 | 12 -> 27 | 3 -> 18 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 19 -> 5 | 15 / 0 | 17 | 54.7 -> 32.1 |
| regression2 fresh non-tool (both), g23v3 | 32 | 28 -> 28 | 28 -> 28 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 0 | 32 | 8.7 -> 8.7 |
| prose QA fresh (both), g23v3 | 16 | 16 -> 16 | 16 -> 16 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 0 | 9 | 127.9 -> 128.0 |
| prose QA derivation set (both), g23v3 | 16 | 16 -> 16 | 16 -> 16 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 0 | 11 | 122.1 -> 122.1 |
| regression2 fresh non-tool (both), g23v2 | 32 | 28 -> 28 | 28 -> 28 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 0 | 32 | 8.7 -> 8.7 |
| prose QA (both), g23v2 | 16 | 16 -> 12 | 16 -> 12 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 4 | 1 | 122.1 -> 95.1 |
| prose QA (both), v1 | 16 | 16 -> 0 | 16 -> 0 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 16 | 0 | 122.1 -> 120.5 |
| practice concise framing (both), g23v2 | 32 | 12 -> 27 | 3 -> 18 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 20 -> 5 | 15 / 0 | 17 | 56.2 -> 30.8 |
| selection (both), g23v2 | 32 | 12 -> 27 | 4 -> 19 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 17 -> 5 | 15 / 0 | 17 | 56.8 -> 34.0 |
| renamed A (both), g23v2 | 32 | 8 -> 23 | 6 -> 21 | 8 -> 8 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 20 -> 5 | 15 / 0 | 17 | 45.8 -> 24.5 |
| renamed B (both), g23v2 | 32 | 12 -> 26 | 10 -> 25 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 4 (n=10) | 20 -> 6 | 14 / 0 | 17 | 45.9 -> 26.8 |
| final proxy (both), g23v2 | 32 | 12 -> 27 | 3 -> 18 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 19 -> 5 | 15 / 0 | 17 | 54.7 -> 32.1 |
| regression non-tool (both), g23v2 | 32 | 27 -> 27 | 27 -> 27 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 0 / 0 | 32 | 7.1 -> 7.1 |
| selection (both), v1 | 32 | 12 -> 26 | 4 -> 18 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 4 (n=10) | 17 -> 5 | 14 / 0 | 17 | 56.8 -> 37.8 |
| renamed A (both), v1 | 32 | 8 -> 24 | 6 -> 22 | 8 -> 9 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 20 -> 5 | 16 / 0 | 16 | 45.8 -> 24.5 |
| renamed B (both), v1 | 32 | 12 -> 26 | 10 -> 25 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 4 (n=10) | 20 -> 6 | 14 / 0 | 17 | 45.9 -> 26.8 |
| regression non-tool (both), v1 | 32 | 29 -> 29 | 29 -> 29 | - -> - (n=-) | - -> - (n=-) | - -> - (n=-) | 0 -> 0 | 1 / 1 | 30 | 8.7 -> 8.6 |
| stock reload (fresh process), g23v3 | 32 | 12 -> 26 | 3 -> 17 | 12 -> 12 (n=12) | 0 -> 9 (n=10) | 0 -> 5 (n=10) | 20 -> 6 | 14 / 0 | 18 | 54.9 -> 32.8 |
| stock reload (fresh process), g23v2 | 32 | 12 -> 27 | 3 -> 18 | 12 -> 12 (n=12) | 0 -> 10 (n=10) | 0 -> 5 (n=10) | 20 -> 5 | 15 / 0 | 17 | 54.9 -> 30.8 |

p44_v3_stock stock vs in-process: {'edited': {'different': 0, 'different_ids': [], 'identical': 32}}

p34_v2_stock stock vs in-process: {'edited': {'different': 0, 'different_ids': [], 'identical': 32}}


**Final candidate `g23v3` summary.** Tool-context sets (practice 64, selection 32, renamed 64, final-split proxy 32 = 192 items; 72 required calls, 60 NO_TOOL, 60 NEED_CLARIFICATION): lenient 68 -> 155, strict 30 -> 118, **87 wrong->right and 0 right->wrong**, required calls 68/72 -> 68/72 with 71/72 token-identical (the four calls wrong at baseline are renamed-set function-choice errors that stay wrong; the one call whose tokens differ is a correct renamed call that drops its explicit optional default at a near-tied parameter-boundary position and stays correct under both scorings), NO_TOOL 0/60 -> 59/60, NEED_CLARIFICATION 0/60 -> 28/60 (the 32 misses are the 30 ambiguous `either X or Y` items plus one weak explicit-missing item and its renamed twin), false calls 114 -> 33. Non-tool: fresh-seed short-answer suite 32/32 token-identical; 32 prose questions (16 used to derive `c2`, 16 fresh) all intact with mean length unchanged (127.9 -> 128.0 tokens) and no literal fragments, 20/32 token-identical, the rest diverge late into benign paraphrases (e.g. "Bacteria are single-celled euka..." -> "Bacteria are prokaryotic cells."). Fresh-process stock MLX-LM reload of the exported checkpoint reproduces the in-process original (32/32 vs p01) and edited (32/32 vs p38) token streams exactly, with no campaign modules imported by the worker.

Calibration: the exact CPU simulator predicted the real edited decision token on 64/64 items for v1 (median |logit error| 0.06, max 0.38); v2/v3 differ only in the gate direction and the output target and were taken straight to generation, where every simulated prediction (which items flip, which call is borderline, which non-tool prompt breaks at a given dose) was borne out.

## 6. Candidate decision, limitations and accounting

**Decision.** The selection rule below was written into notes.md before any v2 result was read; v2 then failed its non-tool clause, v3 was built specifically to fix that failure and was evaluated under the same rule, with 16 fresh prose questions (`prose_qa2`) that were never used in any derivation providing the independent non-tool evidence. Rule: choose the revised edit iff practice has 0 right->wrong with calls token-identical, all non-tool checks have 0 right->wrong with no injected literal fragments and mean length within 10%, and selection/renamed/final each have 0 right->wrong. `g23v3` satisfies every clause; `g23v2` fails the prose clause (4 of 16 prose answers corrupted, 1/16 identical); v1 fails the non-tool clause (one `NENE` answer, 0/16 prose intact). Final edit: **`g23v3`** (`edits/g23v3.npz`, exported checkpoint `artifacts/exports/g23v3/`).

Limitations and residual failures:
* Ambiguous requests (`either "X" or "Y", with no preference specified`) are untouched (0/23 across all sets): the network has no linear feature for "two candidate values" at any probed layer; a readout edit cannot create one.
* Weak-cue explanation requests ("Describe this operation now: <op>. <values>", the teacher set's minimal variant) mostly stay calls (2/20 flipped in simulation); the hidden test's NO_TOOL items are expected to be clearer than this, but the edit's reach scales with how strongly the model itself represents the request as explanatory.
* One practice explanation item (0238, meters-to-feet, a prose explanation at baseline) is not repaired by v3 (it was by v2) because its `c2.x` jitter lowers its gate value; it fails gracefully (same prose as baseline).
* Prose answers in non-tool prompts are not bit-identical (20/32): a handful of generated positions still carry activations up to ~2 units (~5 logits on `NO`/`NE`), enough to flip near-tied word choices after which the text diverges benignly. Short-answer non-tool prompts are identical.
* Inside tool calls the readout is not perfectly silent at near-tied positions: v1 flipped two near-tied tokens in renamed calls (one function name, favourably; one optional argument), v2 none, v3 one (a correct renamed call drops its explicit optional default; still correct). Across 72 paired calls per candidate no call changed correctness.
* The explicit-missing feature is phrasing-dependent: `not provided`-style phrasings are repaired (practice 9/10, teacher 19-20/20 in simulation), other phrasings only when the model itself represents them as missing.
* The edit is specific to the evaluated protocol's first-token decision; it was validated under two system-prompt framings (`standard` and `concise`, the latter for v2 only: 12 -> 27 of 32, 15/0) but not under framings with format examples.

Accounting (ledger.jsonl, hash-chained): **47 of 50 passes** used, 2,619 charged executions (2,585 completed), 1,431,007 input and 93,537 generated tokens. Three passes were charged without producing results: p04 and p04b (system-wide available memory fell below the reused guard's 3 GiB threshold while a sibling campaign shared the machine; threshold relaxed to 1 GiB with kernel-pressure and 15 GiB stops intact) and p29 (the stock worker, living in `runtime/`, shadowed the stdlib `inspect` module imported by mlx_lm; fixed with `python -I`). Two attempted passes failed before the budget request was opened (lock-acquisition errors, not charged) and were re-run. Peak process RSS 3.46 GiB and backend (Metal) peak 1.85 GiB, measured separately and never summed; one resident model at a time under the shared accelerator lock, which was shared with a concurrently running sibling campaign (`qwen-5`) by polling. No gradients, optimizers, LoRA, distillation or output-fitting regressions were used; prompts, template, decoding and post-processing were identical for original and edited weights in every pass. No held-out test instances were generated or read; the `final_proxy` set is a local proxy built from the 12 final-split family definitions that were never used in localization.

## 7. Reproduction

All commands run from `/Users/scasella/Downloads/qwen-6` with `P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python` and `HF_HUB_OFFLINE=1`. Every forward pass is one budgeted request in `ledger.jsonl` (hash-chained, append-only); labels below are the ledger labels.

```
$P build_data.py                                   # practice/selection/final_proxy/renamed/teacher/regression sets (deterministic)
$P build_extra.py                                  # regression2 (seed 60132) and prose_qa; prose_qa2 is written inline (see notes.md)
$P fp.py generate --set practice --model original --label p01_baseline --hypothesis ... --prediction ...
$P fp.py teacher  --set teacher  --model original --label p02_teacher_dec ...        # decision/prefix captures, gold continuations
$P fp.py teacher  --set practice --model original --label p03_practice_dec ...
$P fp.py teacher  --set practice --model original --label p04c_practice_allpos --mlpin 21,22,23 --colstats 21,22,23 ...
$P fp.py teacher  --set teacher  --model original --label p05_teacher_allpos  --mlpin 21,22,23 --colstats 21,22,23 ...
$P fp.py teacher  --set regression --prompt regression --model original --label p16_regression_capture --layers 23 --mlpin 23 ...
$P fp.py teacher  --set practice --ids data/practice_noncall.ids --model original --label p17_prose_capture --layers 23 --mlpin 23 --continuation "text:I cannot perform this operation. Here is an explanation of what it means instead." ...
$P fp.py teacher  --set prose_qa --prompt regression --model original --label p35_prose_capture_nontool --layers 23 --mlpin 23 --continuation fromrun:p31_v2_prose_both:original ...
$P -m analysis.neurons p02_teacher_dec ; $P -m analysis.separability ; $P -m analysis.features2 ; $P -m analysis.dla ; $P -m analysis.crosslayer ; $P -m analysis.leak2   # CPU localization
$P build_g23.py --label g23v3 --a1 1.0 --a2 1.0 --t 0.5 --lam 0.3 --lam2 0.5 --prose_target 0.5      # final edit -> edits/g23v3.npz (+ .spec.json, .vectors.npz)
$P build_g23.py --label g23v3_rand13 ... --random 13                                                   # norm-matched random control on the same support
$P fp.py generate --set practice --model edited --edit g23v3 --label p38_v3_gen_practice ...
$P fp.py generate --set selection --model both --edit g23v3 --label p40_v3_sel_both ...               # paired original vs edited
$P stock_reload.py --edit g23v3 --set practice --ids data/practice_stock.ids --label p44_v3_stock --compare p38_v3_gen_practice   # export + fresh stock MLX-LM worker
$P -m analysis.pair p01_baseline p38_v3_gen_practice ; $P -m analysis.tables                           # paired transitions / tables
```

Artifacts: final edit `edits/g23v3.npz` (BF16 bit patterns of the three replaced tensors; `edits/g23v3.spec.json` has every scalar, `edits/g23v3.vectors.npz` the vectors s, r, c, c2, u_AB, u_route, gate_dir), full HF-format checkpoint with only those tensors replaced in `artifacts/exports/g23v3/` (manifest with SHA-256 of base and edited safetensors), earlier candidates `edits/g23_t05_a1.0.npz` (v1) and `edits/g23v2.npz` (v2) with their controls, all per-case outputs in `runs/*.json`, the ledger in `ledger.jsonl`, the running notebook in `notes.md`.
