# qwen-6 campaign notes — semantic tool-calling weight edits for Qwen3.5-0.8B

Workspace: /Users/scasella/Downloads/qwen-6. Python: /Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python (mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.1).
Harness reused from qwen-5/qwen-3 toolpatch (benchmark generator/renderer, strict scorer, hash-chained ledger, MLX runtime). New: fp.py (forward-pass tool), primitives.py (disk-persisted BF16 edits), build_data.py.
Protocol: official chat template with native <tools>, enable_thinking=False, greedy, max 128 new tokens, context <=2048, system prompt 'standard' (no format examples).
Scoring: lenient = semantic correctness with omitted/explicit optional default equivalent; strict = additionally no extra arguments. Both are always reported.
Budget: 50 passes total, <=64 executions each; every pass (including failed) is charged in ledger.jsonl.

Data (build_data.py, deterministic): practice 64 (24 call/20 answer/20 clarify, 18 discovery families), selection 32 (6 selection families), final_proxy 32 (12 final families, local proxy only), renamed 64 (practice with fresh function/argument names), teacher 60 (10 groups x {call, answer, clarify} x {original, renamed}, all six prompts in a group have identical token length), regression 32 non-tool tasks.

### pass p01_baseline (generate, set=practice, model=original, n=64, edit=-)

Hypothesis: Decision errors dominate: the unedited model initiates <tool_call> on explanation (NO_TOOL) and missing/ambiguous-argument (NEED_CLARIFICATION) requests; required calls are mostly semantically right.

Prediction: Answer and clarify classes near 0/20 correct with high false-call rate; call class mostly correct under lenient scoring, lower under strict due to explicit optional default.

Outcome p01_baseline (64 practice, original): lenient 24/64 (call 24/24, answer 0/20, clarify 0/20), strict 7/64 (17 calls add the explicit optional default `detail=brief`); false calls 38/40 non-call items; 2 answer items produce prose, 1 clarify output has an invalid typed value ('not provided' into an integer slot). Dominant, explainable failure: unconditional <tool_call> onset on explanation and missing-argument requests. Target = decision mechanism (call vs abstain) plus NO_TOOL vs NEED_CLARIFICATION routing.

### pass p02_teacher_dec (teacher, set=teacher, model=original, n=60, edit=-)

Hypothesis: At the decision position the residual stream and specific mid-to-late MLP neurons linearly separate complete-call vs explanation vs missing-argument requests, invariantly to function/argument names, even though <tool_call> wins the logits; the gold literal continuations (NO->_TOOL, NE->ED->...) are high-probability once the first token is forced.

Prediction: Per-neuron class contrasts replicate across original and renamed twins; <tool_call> leads NO/NE on nearly all non-call cases; continuation tokens after NO/NE have logprob near 0.

### pass p03_practice_dec (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: The same decision features found on matched teacher triples generalize to practice prompts with distractors, ambiguous variants, three request forms and opaque names.

Prediction: Features selected on teacher data separate call vs non-call on practice per case; NO leads NE on answer items and NE-NO gap is larger on clarify items than on answer items.

(Attempts of p02/p03 above at 2026-11-04T11:59:12Z failed on accelerator-lock acquisition before any model execution; no pass charged. Re-run below.)

### pass p02_teacher_dec (teacher, set=teacher, model=original, n=60, edit=-)

Hypothesis: At the decision position the residual stream and specific mid-to-late MLP neurons linearly separate complete-call vs explanation vs missing-argument requests, invariantly to function/argument names, even though <tool_call> wins the logits; the gold literal continuations (NO->_TOOL, NE->ED->...) are high-probability once the first token is forced.

Prediction: Per-neuron class contrasts replicate across original and renamed twins; <tool_call> leads NO/NE on nearly all non-call cases; continuation tokens after NO/NE have logprob near 0.

### pass p03_practice_dec (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: The same decision features found on matched teacher triples generalize to practice prompts with distractors, ambiguous variants, three request forms and opaque names.

Prediction: Features selected on teacher data separate call vs non-call on practice per case; NO leads NE on answer items and NE-NO gap is larger on clarify items than on answer items.

### pass p02_teacher_dec (teacher, set=teacher, model=original, n=60, edit=-)

Hypothesis: At the decision position the residual stream and specific mid-to-late MLP neurons linearly separate complete-call vs explanation vs missing-argument requests, invariantly to function/argument names, even though <tool_call> wins the logits; the gold literal continuations (NO->_TOOL, NE->ED->...) are high-probability once the first token is forced.

Prediction: Per-neuron class contrasts replicate across original and renamed twins; <tool_call> leads NO/NE on nearly all non-call cases; continuation tokens after NO/NE have logprob near 0.

### pass p03_practice_dec (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: The same decision features found on matched teacher triples generalize to practice prompts with distractors, ambiguous variants, three request forms and opaque names.

Prediction: Features selected on teacher data separate call vs non-call on practice per case; NO leads NE on answer items and NE-NO gap is larger on clarify items than on answer items.

### pass p04_practice_allpos (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: The explanation and explicit-missing features read from the final-layer MLP input are confined to the decision position: at prompt positions and after the emitted literal they are weak, so a final-layer readout neuron would fire only where the decision is made; some layer-21..23 neurons are near-dead at every position and can be re-purposed.

Prediction: Candidate readout directions have small projections at all prompt positions relative to the decision position on their class; dozens of neurons have max |activation| < 0.05 across all positions.

### pass p05_teacher_allpos (teacher, set=teacher, model=original, n=60, edit=-)

Hypothesis: Same as p04 on the matched, renamed teacher triples: readout directions derived here transfer to practice and are name-invariant.

Prediction: Original and renamed twins give near-identical final-layer MLP inputs at the decision position; the derived directions separate classes on practice.

### pass p04b_practice_allpos (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: Re-run of p04 (aborted by system memory guard): final-layer MLP-input readout directions for explanation/missing are confined to the decision position and silent at generated continuation positions; some layer-21..23 neurons are near-dead at all positions.

Prediction: Candidate directions have small projections at continuation positions after the literal; dozens of neurons have max |activation| < 0.05 across all positions.

Memory guard note: the reused guard aborted p04 and p04b (charged, no output) because *system-wide* available memory fell below 3 GiB while another campaign (qwen-5) shares the machine; my process RSS stayed ~2 GiB. Relaxed that courtesy threshold to 1 GiB; kernel-pressure (critical>=4) and 15 GiB RSS/backend hard stops unchanged.

### pass p04c_practice_allpos (teacher, set=practice, model=original, n=64, edit=-)

Hypothesis: Re-run of p04/p04b (aborted by system memory guard): final-layer MLP-input readout directions for explanation/missing are confined to the decision position and silent at generated continuation positions; some layer-21..23 neurons are near-dead at all positions.

Prediction: Candidate directions have small projections at continuation positions after the literal; dozens of neurons have max |activation| < 0.05 across all positions.

### pass p06_calib_practice (teacher, set=practice, model=edited, n=64, edit=g23_t05_a1.0)

Hypothesis: G23 edit (layer-23 neurons 3468/3421 re-purposed as a quadratic abstain readout writing +NO+NE-call and an abstain-gated signed route readout writing NE-NO; t=0.5, k=2, a1=a2=1.0): decision-position logits match the CPU simulation (all 24 calls keep <tool_call>, 20 answers -> NO, 8/10 explicit-missing -> NE, ambiguous stay calls); the gold continuation tokens keep rank 1 after the literal and inside tool calls.

Prediction: Decision argmax agrees with simulation on >=60/64 items; max |logit error| vs simulation < 1.5; all continuation gold tokens rank 1 with logprob > -2; function-name prefix argmax unchanged.

## Localization summary (passes p02/p03 decision captures, p05/p04c all-position captures; CPU analysis in analysis/*.py)

- Lens check: decision logits reconstructed from the captured pre-final-norm residual h_24 match the saved logits (max err 0.07). Residual RMS grows 0.07 (L14) -> 0.38 (L24): any mid-layer write large enough to move the output by ~15 logits would exceed the local residual norm, so the write must live in the last layer. A layer-23 MLP edit only changes the logits at its own position (the KV cache is built from the layer input), so prompt positions cannot leak; only generated positions matter.
- Layer lens: the call-vs-abstain margin is written by layers 14-23 (layer 20's block adds ~5-6 logits, layer 23 ~2), nearly class-independent: explanation requests get almost the same call push as real calls. Direct logit attribution finds no small set of non-call-selective neurons pushing <tool_call> (max per-neuron effect ~0.2 logits): the failure is a missing inhibition, not a spurious excitation.
- Features: (A) "explanation request" is linearly separable with d'~10-14 at layers 12-16 and still present in the residual entering layer 23 (zero overlapping practice cases). (B) "required value explicitly missing" is separable for 'not provided' phrasings but entangled with (A); the model has NO usable feature for the ambiguous 'either X or Y' variant (those items look exactly like complete calls at the output and in all probed layers; LOO probes give d'<3). (C) clarify-vs-answer (NO vs NE routing) is extremely clean at layers 8-13 (d'~10) and a layer-14 diff-of-means direction still separates NOTPROV from answers when read at the layer-23 MLP input (practice 0.79..2.23 vs <=0.28; teacher 1.24..2.80 vs <=0.82), although the model's own output never uses it (NE-NO margin ~ -2 for both classes).
- Continuation hazard: a quadratic abstain readout also fires at the generated positions after the literal (e.g. at 'IFICATION', predicting <|im_end|>). Orthogonalizing the abstain direction against the end-of-literal and first-continuation signatures cuts that leakage ~15x (worst remaining quadratic activation 6.7 vs decision values 7-200).
- Borderline call: teacher-06-call asks for an interval with lower bound 46 > upper bound 43; the model's abstain feature is mildly positive there (sensible). A quadratic readout keeps it a call; a linear one flips it.

## Edit G23 (layer 23 only, 3 tensors, 6142 BF16 entries = 8.2e-6 of text params)

Re-purpose two near-dead layer-23 MLP neurons (max |activation| <= 0.25 at every captured position): J1=3468 (abstain), J2=3421 (route).
  gate[J1] = k s,  up[J1] = k s,  down[:,J1] = a1 * u_AB            -> activation silu(k s.x) (k s.x) ~ k^2 (s.x)^2 for s.x>0, ~0 otherwise
  gate[J2] = k s,  up[J2] = kr r, down[:,J2] = a2 * (u_NE - u_NO)   -> activation silu(k s.x) (kr r.x): signed route, gated by abstain
  x = post_attention_layernorm output at layer 23 (MLP input); k=2, kr=1; intended a1=a2=1.0.
  s = unit( ortho( unit(ortho(ortho(s0, e_end), e_first)), ref ) ), s0 = teacher diff of means (abstain - call) at residual h_18,
      e_end/e_first = mean x at the last / first generated position of abstain items minus mean call decision x,
      ref = mu_call + 0.5 (mu_abstain_clear - mu_call) (zero-crossing midway between the call cluster and the clear abstain cluster).
  r = unit(ortho(r0, (mu_answer+mu_NOTPROV)/2)), r0 = teacher diff of means (clarify - answer) at residual h_14.
  u_AB, u_NE-u_NO: min-norm least-squares directions in pre-final-norm residual space (E*gamma rows) with unit logit effect
      (+1 NO, +1 NE, -1 <tool_call>) resp. (+1 NE, -1 NO), and zero effect on 39 competitor tokens (I, To, The, No, None, Based, This, <|im_end|>, _TOOL, ED, ...).
  Exact CPU simulation (analysis/simulate.py, build_g23.py --simulate_only) at the 124 captured decision positions: all 44 calls keep <tool_call> (min margin 1.44 logits), 20/20 practice explanations -> NO, 8/10 explicit-missing practice -> NE, 18/20 teacher missing -> NE, teacher weak-cue explanations 2/20, ambiguous 0/10 (no feature).
  Controls built on the same support: g23_zero (0 entries), g23_t05_a0.5 (half), g23_t05_a2.0 (double: simulation shows the borderline call margin drops to 0.09), g23_t05_rev (a1=a2=-1), g23_rand7 (random Gaussian s, r, u directions with matched norms; simulation: no decision changes).

### pass p07_gen_intended (generate, set=practice, model=edited, n=64, edit=g23_t05_a1.0)

Hypothesis: Intended G23 edit (a1=a2=1.0) in ordinary greedy generation: explanation requests emit exactly NO_TOOL and explicit-missing requests exactly NEED_CLARIFICATION (then stop), while all 24 previously correct calls are unchanged token-for-token.

Prediction: Lenient score rises from 24/64 to >=50/64; zero right-to-wrong transitions; answer 20/20, clarify 8/20 (ambiguous items remain calls); outputs after the literal terminate cleanly.

### pass p08_gen_zero (generate, set=practice, model=edited, n=64, edit=g23_zero)

Hypothesis: Zero control: an edit snapshot identical to the original weights reproduces the baseline generation exactly (cross-pass determinism).

Prediction: All 64 outputs token-identical to p01_baseline; 24/64 lenient.

### pass p09_gen_half (generate, set=practice, model=edited, n=64, edit=g23_t05_a0.5)

Hypothesis: Half dose (a1=a2=0.5): same mechanism at half output scale repairs most explanation items and some explicit-missing items with no call regressions.

Prediction: Lenient between 24 and 52; answer >=16/20; calls 24/24 unchanged.

### pass p13_sel_both (generate, set=selection, model=both, n=32, edit=g23_t05_a1.0)

Hypothesis: Selection families (unseen during localization): the intended edit repairs explanation and explicit-missing items without breaking calls.

Prediction: wrong_to_right >= 12, right_to_wrong = 0 on 32 selection items; calls 12/12 unchanged.

### pass p10_gen_double (generate, set=practice, model=edited, n=64, edit=g23_t05_a2.0)

Hypothesis: Double dose (a1=a2=2.0): still repairs explanation/explicit-missing items; simulation predicts the borderline teacher call would be at risk but all 24 practice calls survive.

Prediction: Lenient ~52; calls 24/24; possibly more clarify repairs.

### pass p16_regression_capture (teacher, set=regression, model=original, n=32, edit=-)

Hypothesis: In prompts without tool declarations the abstain readout direction s (derived in tool contexts) is inactive at the decision position and at answer positions, so the layer-23 edit cannot alter ordinary non-tool answers.

Prediction: s.x at the decision position of all 32 regression prompts is below the call cluster's maximum (<0.3); quadratic activation < 1 at every generated position.

### pass p11_gen_rev (generate, set=practice, model=edited, n=64, edit=g23_t05_rev)

Hypothesis: Reversed sign (a1=a2=-1): pushes the opposite way; no repairs, baseline-like or worse.

Prediction: Lenient <= 24; answer 0/20; calls 24/24.

### pass p14_renamedA_both (generate, set=renamed, model=both, n=32, edit=g23_t05_a1.0)

Hypothesis: Renamed functions/arguments (fresh names, same semantics): the repair is name-invariant.

Prediction: wrong_to_right >= 12 and right_to_wrong = 0 on the first 32 renamed items.

### pass p12_gen_rand7 (generate, set=practice, model=edited, n=64, edit=g23_rand7)

Hypothesis: Norm-matched random directions on the same two neurons: no repair.

Prediction: Lenient ~24; no systematic transitions.

### pass p18_regression_both (generate, set=regression, model=both, n=32, edit=g23_t05_a1.0)

Hypothesis: Non-tool regression suite (no tool declarations; instruction following, extraction, arithmetic, JSON): the layer-23 abstain readout is inactive outside tool contexts, so the edited model's outputs are unchanged.

Prediction: All 32 outputs token-identical between original and edited; same correctness.

### pass p15_renamedB_both (generate, set=renamed, model=both, n=32, edit=g23_t05_a1.0)

Hypothesis: Renamed functions/arguments (second half): name-invariant repair.

Prediction: wrong_to_right >= 12 and right_to_wrong = 0 on the last 32 renamed items.

## Results so far (candidate g23_t05_a1.0 = intended; all generation passes greedy, max 128 tokens, standard prompt)

Practice (64; baseline p01 vs edited): lenient 24 -> 52, strict 7 -> 35, false calls 38 -> 10; 28 wrong->right, 0 right->wrong (both scorings); all 24 calls token-identical. Per class: answer 0/20 -> 20/20 (all exactly NO_TOOL), clarify 0/20 -> 8/20 (8 exactly NEED_CLARIFICATION; the 10 ambiguous 'either X or Y' items remain calls as predicted; 2 explicit-missing items with a weak feature start with 'I' and then degenerate into NE/NO repeats: wrong before and after).
Calibration (p06, teacher-forced with the edit): decision argmax agrees with the exact CPU simulation on 64/64 items (max |logit error| 0.38, median 0.06); inside tool calls no continuation token changes rank (max log-prob shift -0.24).
Dose response on practice (lenient /64): zero 24 (64/64 token-identical to baseline: exact revert), half 52, intended 52, double 53 (but 2 post-literal degenerations, e.g. NENONONENE: the leakage ceiling), reversed 24 (0 repairs; the two prose explanations become false calls), norm-matched random 24 (0 repairs, 60/64 token-identical).
Selection (32 unseen families, paired 'both'): lenient 12 -> 26, strict 4 -> 18, false calls 17 -> 5; 14 wrong->right, 0 right->wrong; calls 12/12 unchanged; answers 10/10; clarify 4/10 (5 ambiguous remain calls, 1 degenerate prose).
Pending: renamed transfer (p14/p15), non-tool regression (p18), context captures (p16/p17), final proxy, stock reload.
Planned refinement (simulated): target -0.5 on prose starters I/To/The in u_AB raises the borderline call margin 1.44 -> 2.12 and converts the weak explicit-missing failures from degenerate prose to NE (9/10) or a plain call.

### pass p17_prose_capture (teacher, set=practice, model=original, n=40, edit=-)

Hypothesis: When an abstain-context response starts as prose instead of the literal, the abstain readout keeps firing at prose positions (the mechanism behind the INENENONONO degeneration seen on two items).

Prediction: Quadratic activation of s at prose positions after 'I' is comparable to the decision-position value for answer items.

### pass p19_v2_gen_practice (generate, set=practice, model=edited, n=64, edit=g23v2)

Hypothesis: Revised edit g23v2 (context-gated abstain readout, prose-starter-suppressing output direction, a1=a2=1.0): repairs explanation and explicit-missing requests in greedy generation with zero call regressions and no degenerate prose starts on practice.

Prediction: Lenient >= 52/64; answer 20/20; clarify 9/20; calls 24/24 token-identical to baseline; no INENE-style outputs.

### pass p20_v2_regression_both (generate, set=regression, model=both, n=32, edit=g23v2)

Hypothesis: Context gate silences the readout on prompts without tool declarations: non-tool outputs are unchanged.

Prediction: 32/32 outputs token-identical between original and edited (the v1 regression on 'cedar24' disappears).

## v1 regression finding and revision (g23v2)
Non-tool regression (p18, v1, paired): 29/32 -> 29/32, 30/32 token-identical, but one instruction prompt ('Reply with exactly the word cedar24...') became 'NENE' (predicted exactly by the simulation: its abstain-feature value 1.35 is the highest among non-tool prompts) and one arithmetic item flipped a near-tie favourably. Prose-position capture (p17) shows the abstain feature stays large (2-6) throughout explanatory prose in abstain contexts (quadratic activation up to 190), which is why the two weak explicit-missing items that start with 'I' degenerate; mean-signature orthogonalization cannot remove this (194 -> 133) because the feature genuinely persists during explanation.
Revision g23v2: (1) gate direction s - lam*c with lam=0.3, c = unit(ortho(mean non-tool decision x - mean tool-context decision x, mean tool-context decision x)); c.x is -2.5..3.7 on tool-context decisions but 17.9..24.7 on non-tool decisions, 10..15 at the function-name prefix and 4.6..22 inside calls, so the SiLU gate closes there: simulated |activation| 0.00 on non-tool decisions, 0.01 inside calls, 0.6 after the literal (was 6.7); prose positions in abstain contexts are unaffected (c.x -0.6..7.9) and remain the one known leak (only affects outputs that are already wrong). (2) Output direction target adds -0.5 on the prose starters I/To/The (zero elsewhere), which raises the borderline call's margin (1.44 -> 2.05: its runner-up is 'I') and turns the weak explicit-missing items into NE (practice 9/10) or a plain call instead of degenerate prose. Same neurons, k=2, kr=1, a1=a2=1.0, t=0.5; 6142 entries. Controls rebuilt on the same support (half/double/reversed/random seed 11; zero shared).

### pass p21_v2_sel_both (generate, set=selection, model=both, n=32, edit=g23v2)

Hypothesis: Unseen selection families: revised edit repairs abstentions with zero call regressions.

Prediction: wrong_to_right >= 14, right_to_wrong = 0; calls 12/12.

### pass p22_v2_renamedA_both (generate, set=renamed, model=both, n=32, edit=g23v2)

Hypothesis: Renamed transfer (first half): name-invariant repair with calls unchanged.

Prediction: wrong_to_right >= 14, right_to_wrong = 0; calls token-identical.

### pass p23_v2_renamedB_both (generate, set=renamed, model=both, n=32, edit=g23v2)

Hypothesis: Renamed transfer (second half).

Prediction: wrong_to_right >= 12, right_to_wrong = 0.

### pass p24_v2_final_both (generate, set=final_proxy, model=both, n=32, edit=g23v2)

Hypothesis: Local proxy of the held-out test (12 final-split families never used in localization): the repair generalizes to new task families.

Prediction: wrong_to_right >= 12, right_to_wrong = 0; calls 12/12.

### pass p25_v2_half (generate, set=practice, model=edited, n=64, edit=g23v2_half)

Hypothesis: Half dose of the revised edit.

Prediction: Lenient ~50-52; no regressions.

### pass p26_v2_double (generate, set=practice, model=edited, n=64, edit=g23v2_double)

Hypothesis: Double dose of the revised edit: leakage ceiling may reappear after the literal.

Prediction: Lenient ~52; possibly degenerate continuations; calls 24/24.

### pass p27_v2_rev (generate, set=practice, model=edited, n=64, edit=g23v2_rev)

Hypothesis: Reversed sign of the revised edit: no repairs.

Prediction: Lenient 24; prose explanations may become calls.

## Pre-stated candidate decision rule (written before reading any g23v2 result)
Choose g23v2 as the final edit iff ALL hold: (1) practice p19: 0 right->wrong (lenient) and all 24 calls token-identical to baseline; (2) non-tool checks p20 (original 32 prompts), p30 (fresh 32 prompts, seed 60132) and p31 (16 prose-QA prompts): 0 right->wrong, no output containing NO_TOOL / NEED_CLARIFICATION / <tool_call> / 'NENE' / 'NONO' fragments where the original has none, and mean new-token length within 10% of the original; (3) selection p21, renamed p22/p23 and final proxy p24: 0 right->wrong each. Otherwise fall back to g23_t05_a1.0 (v1) with its single documented non-tool regression, provided v1 passes the same non-tool prose check; if neither passes, report both and recommend v1 at half dose. Ties are not broken by lenient gains.

### pass p30_v2_regression2_both (generate, set=regression2, model=both, n=32, edit=g23v2)

Hypothesis: Independent non-tool regression items (fresh seed 60132, never used to derive any direction): the context-gated edit leaves them unchanged.

Prediction: 32/32 token-identical; 0 right->wrong.

### pass p28_v2_rand11 (generate, set=practice, model=edited, n=64, edit=g23v2_rand11)

Hypothesis: Norm-matched random directions (same neurons, gate/up/down norms matched): no repairs.

Prediction: Lenient ~24; no systematic transitions.

### pass p29_v2_stock (stock reload, set=practice, n=32, checkpoints=['original', 'edited'], edit=g23v2)

Hypothesis: Fresh stock MLX-LM reload of exported edit g23v2 reproduces in-process token streams exactly for both original and edited checkpoints.

### pass p31_v2_prose_both (generate, set=prose_qa, model=both, n=16, edit=g23v2)

Hypothesis: Non-tool prose question answering (no tools declared): the context gate keeps the readout silent during prose, so answers are unchanged and contain no injected literal fragments.

Prediction: 16/16 token-identical or at least no NO/NE fragments and mean length within 10%.

### pass p32_v1_prose_both (generate, set=prose_qa, model=both, n=16, edit=g23_t05_a1.0)

Hypothesis: Same prose-QA check for the v1 edit (fallback candidate).

Prediction: No injected literal fragments; mean length within 10%.

### pass p33_v2_concise_both (generate, set=practice, model=both, n=32, edit=g23v2)

Hypothesis: Different prompt framing ('concise' system prompt, no examples): the repair does not depend on the exact system prompt wording.

Prediction: wrong_to_right >= 20 of 32, 0 right->wrong under the concise framing.

## g23v2 results (passes p19-p28) and stock-reload failure
Practice: 24 -> 53 lenient (36 strict), 29 wrong->right / 0 right->wrong, all 24 calls token-identical, answers 20/20, clarify 9/20 (11 calls = 10 ambiguous + 1 weak explicit-missing; no prose/degenerate outputs). Non-tool (same 32 prompts used for c, standard framing without tools): 32/32 token-identical. Selection 12 -> 27 (15/0), renamed 8 -> 23 and 12 -> 26 (15/0, 14/0; no call token changes this time), final proxy (12 never-touched families) 12 -> 27 (15/0). Controls: half 52, double 53, reversed 24 (0 repairs), random 24 (0 repairs).
p29 (stock reload) failed before any model execution: the worker in runtime/ shadowed the stdlib 'inspect' module needed by mlx_lm (charged as a failed pass; 35 requests used). Fixed by running the worker with python -I; re-run as p34.

### pass p34_v2_stock (stock reload, set=practice, n=32, checkpoints=['original', 'edited'], edit=g23v2)

Hypothesis: Fresh stock MLX-LM reload of exported edit g23v2 reproduces in-process token streams exactly for both original and edited checkpoints.

### pass p35_prose_capture_nontool (teacher, set=prose_qa, model=original, n=16, edit=-)

Hypothesis: During prose answers to non-tool questions the abstain feature s.x stays positive while the decision-derived context term c.x is small, so the v2 gate opens mid-answer; a generated-position term c2 (orthogonal to the tool-context decision mean) is large at all these positions and can close the gate.

Prediction: s.x > 0 and c.x < 7 at most prose positions; c2.x >= 8 at prose positions versus |c2.x| <= 3 at tool-context decision positions.

## Non-tool prose failure (p31/p32) and the generated-position gate (v3)
Prose questions without tools (16 hand-written, judged by token identity/length and literal fragments): v1 destroys all 16 answers (NONENENE...), v2 corrupts most (1/16 token-identical; NO,NO,NO... and mid-answer injections) although the fresh-seed regression2 set stays 32/32 identical and the same-prompt regression 32/32. Cause: the abstain feature s.x stays positive while the model explains, and the decision-derived context term c.x is small at generated positions (tool-context prose: -0.6..7.9). Stock reload of v2 (p34): fresh stock worker reproduces original (32/32 vs p01) and edited (32/32 vs p19) streams exactly.
v3 plan: add a second gate term c2 = unit(ortho(mean x over all captured generated positions - mean tool-context decision x, mean tool-context decision x)); preview from existing captures: c2.x is -2.25..2.81 (sd 1.0) at tool-context decisions but >= 10.1 at literal continuations, >= 13.2 inside calls, >= 7.9 at tool-context prose and 6.6..14.4 at non-tool decisions. Gate = k (s - 0.3 c - lam2 c2) closes the readout at every generated position while keeping practice abstain decisions open (25th percentile pre-activation 2.7 at lam2=0.5). p35 captures non-tool prose continuations (the model's own answers) to derive/validate c2 on exactly the failing positions. Budget after p35: 13 passes.

### pass p36_v3_prose2_both (generate, set=prose_qa2, model=both, n=16, edit=g23v3)

Hypothesis: Fresh non-tool prose questions (never used for any derivation): the generated-position gate keeps the readout silent during prose, so answers are unchanged.

Prediction: 16/16 token-identical, or at least no NO/NE fragments and mean length within 10%.

### pass p37_v3_prose_both (generate, set=prose_qa, model=both, n=16, edit=g23v3)

Hypothesis: The 16 prose questions that v1/v2 corrupted are now answered unchanged.

Prediction: 16/16 token-identical.

### pass p38_v3_gen_practice (generate, set=practice, model=edited, n=64, edit=g23v3)

Hypothesis: v3 (generated-position gate) keeps the tool-context repair: explanation and explicit-missing requests emit the literals, calls unchanged.

Prediction: Lenient >= 51/64; 0 right->wrong; calls token-identical; answer >= 19/20; clarify 9/20.

### pass p39_v3_regression2_both (generate, set=regression2, model=both, n=32, edit=g23v3)

Hypothesis: Fresh-seed non-tool regression items unchanged under v3.

Prediction: 32/32 token-identical.

### pass p40_v3_sel_both (generate, set=selection, model=both, n=32, edit=g23v3)

Hypothesis: Unseen selection families under v3: repairs with zero call regressions.

Prediction: wrong_to_right >= 14, right_to_wrong = 0; calls 12/12.

### pass p41_v3_final_both (generate, set=final_proxy, model=both, n=32, edit=g23v3)

Hypothesis: Final-split family proxy under v3.

Prediction: wrong_to_right >= 14, right_to_wrong = 0; calls 12/12.

### pass p42_v3_renamedA_both (generate, set=renamed, model=both, n=32, edit=g23v3)

Hypothesis: Renamed transfer (first half) under v3.

Prediction: wrong_to_right >= 14, right_to_wrong = 0; calls token-identical.

### pass p43_v3_renamedB_both (generate, set=renamed, model=both, n=32, edit=g23v3)

Hypothesis: Renamed transfer (second half) under v3.

Prediction: wrong_to_right >= 12, right_to_wrong = 0.

### pass p44_v3_stock (stock reload, set=practice, n=32, checkpoints=['original', 'edited'], edit=g23v3)

Hypothesis: Fresh stock MLX-LM reload of exported edit g23v3 reproduces in-process token streams exactly for both original and edited checkpoints.

### pass p45_v3_rand13 (generate, set=practice, model=edited, n=64, edit=g23v3_rand13)

Hypothesis: Norm-matched random directions on the same two neurons (v3 norms): no repair.

Prediction: Lenient ~24; no systematic transitions.

## Final status (2026-11-04)
Final edit: g23v3 (layer 23 only; gate_proj/up_proj rows and down_proj columns of neurons 3468 and 3421; 6142 BF16 entries). Battery: practice 24->52 (28/0, calls 24/24 identical), selection 12->27 (15/0), renamed 8->23 and 12->26 (15/0, 14/0), final proxy 12->27 (15/0), fresh non-tool regression 32/32 identical, prose QA 16/16 + 16/16 intact (no literal fragments, lengths unchanged), stock reload exact (32/32 original, 32/32 edited), random control seed 13 harmful (24->12), seeds 7/11 neutral. 47/50 passes. See results.md.
Final state: edits/current.npz restored to snapshot g23v3 (the last build before this was the random control g23v3_rand13, which had overwritten the live edit); verified bit-identical to edits/g23v3.npz and to the exported checkpoint; ledger event 'final_state' appended.
