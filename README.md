# qwen3.5-toolpatch — programming four dead neurons in Qwen3.5-0.8B

Two closed-form weight edits of `Qwen/Qwen3.5-0.8B` (revision `2fc0636…`, BF16), built without training, gradients, LoRA or fitting to outputs, that repair semantic tool calling:

| edit | neurons | tensors / layers | BF16 entries | 192 tool items (lenient) | NO_TOOL (60) | explicit missing (30) | ambiguous "either X or Y" (30) | valid calls (72) | stress suite, unseen ambiguous (32) | stress suite, resolved two-value calls (32) |
|---|---|---|---|---|---|---|---|---|---|---|
| Base `Qwen3.5-0.8B` | — | — | — | 68 (35.4%) | 0 | 0 | 0 | 68 | 0 | 30 |
| **g23v3** | J1 = 3468, J2 = 3421 (layer 23) | 3 / 1 | 6,142 (0.00082%) | 155 (80.7%) | 59 | 28 | 0 | 68 (token-identical) | 9 | 30 (identical) |
| **g23v4** | + J0 = 2525 (layer 22), J3 = 2331 (layer 23) | 6 / 2 | 12,281 (0.00163%) | **185 (96.4%)** | 59 | 28 | **30** | 68 (token-identical) | **27** | 30 (identical) |

Base → g23v4: 117 items wrong→right, 0 right→wrong; selection and final-split families 32/32 each. Non-tool cohorts: 32 short regression answers token-identical, 32 free-form 128-token essays intact under both edits. The same CPU simulator that designed each edit predicted the real edited model's decision token on 64/64 calibration items (median logit error 0.06) and the direction of every dose and random control before it ran. Research agent: **Mythos 5.1**. Hardware: one Apple M4 Pro (24 GB), MLX-LM 0.32.2.

* **Blog post** (the full story, figures, controls, limitations): [scasella.github.io/qwen3.5-toolpatch/blog.html](https://scasella.github.io/qwen3.5-toolpatch/blog.html)
* **Interactive explorer** (exact closed-form circuit in the browser, no download) + **live WebGPU runner**: [scasella.github.io/qwen3.5-toolpatch](https://scasella.github.io/qwen3.5-toolpatch/)
* **Weights**: [scasella91/Qwen3.5-0.8B-toolpatch-g23v4](https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v4) (full checkpoint + 50 KB patch + `apply_patch.py`), [g23v3](https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v3), ONNX fp16 for Transformers.js: [g23v4-ONNX](https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v4-ONNX)

## 1. The circuit in one paragraph each

**Act I — g23v3 (missing inhibition).** Direct logit attribution over all 86,016 MLP neurons finds no neuron that pushes `<tool_call>` on the wrong prompts: the call prior is distributed and the model has simply not learned to stop. The evidence it needs ("this is an explanation request", "clarify rather than explain") is linearly present in the residual stream and still readable at the last layer. Two near-dead layer-23 MLP neurons are re-purposed into a gated readout: `gate = k(s − λc − λ₂c₂)` opens only at the decision token (`c₂`) of a tool-context prompt (`c`) whose abstain feature `s` is high; J1 writes `u_AB` (least squares: −1 on `<tool_call>`, +1 on `NO` and `NE`, −0.5 on the prose starters `I`/`To`/`The`, 0 on 39 other competitor tokens), J2 writes `u_NE − u_NO` (route). The gates came from two failed candidates: v1 (no gates) collapsed 128-token prose into `NONENENE…`, v2 (tool-context gate only) still fired at later generated positions.

**Act II — g23v4 (positional bottleneck).** Ambiguous "either X or Y" requests are indistinguishable from complete calls at the decision token (leave-one-family-out d′ < 3.9 in every layer) but cleanly separable at the user's clause tokens (d′ 9.5–21.8 at the connector, the indecision tail and the clause-end "."), and the decision query's attention on those tokens is 0.000–0.003 at layer 23: the feature is stranded in the prompt. A layer-22 neuron J0 reads the conjunction "clause end AND unresolved" (`w₁ = unit(0.5g + u)`), realises a −11.7 threshold without a bias term through a constant-projection direction (`r₁·x ≈ 1` on the critical positions), and writes a beacon `p₆` built from the pseudo-inverse of layer 23's key projection so that its key — in the unrotated RoPE dimensions 64:256, orthogonal to the sibling GQA heads 4/5/7 — matches head 6's decision query. Head 6's attention to the clause end goes from 0.0001 to 0.99 with no weight changed; a third layer-23 neuron J3 reads the delivered direction `t` under the Act I gates and writes the same abstain-plus-route payload. Resolved two-value calls ("use X rather than Y") never trigger it: J0 ≤ 0.016 on 141k non-target positions.

Design equations, every scalar and the per-tensor receipts: [`patches/g23v4.spec.json`](patches/g23v4.spec.json), [`patches/g23v3.spec.json`](patches/g23v3.spec.json); the two campaign reports: [`experiments/qwen6_g23v3/results.md`](experiments/qwen6_g23v3/results.md), [`experiments/qwen7_g23v4/results.md`](experiments/qwen7_g23v4/results.md).

## 2. Quickstart

Requirements: Python ≥ 3.10 with `numpy` (that is all `apply_patch.py` needs); the demo and the derived analyses need `mlx-lm` on Apple Silicon (`pip install -r requirements.txt`). The base checkpoint comes from the Hugging Face hub (`huggingface-cli download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17`).

**Patch the stock checkpoint (under a second, numpy only):**

```bash
python apply_patch.py --base ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 --out /tmp/qwen35-g23v4
```

```bash
python apply_patch.py --base <snapshot> --out /tmp/qwen35-g23v3 --version g23v3
```

```bash
python apply_patch.py --base <snapshot> --out /tmp/qwen35-g23v4-conv --mode conversational
```

`--version g23v4` (default) places the four neurons' rows and columns (12,281 entries, six tensors); `--version g23v3` the two layer-23 neurons (6,142 entries, three tensors); `--version v1` the first qwen-6 candidate, kept for the prose-collapse demonstration. Every edited tensor's SHA-256 is checked against the published values; `--verify` additionally compares all 488 tensors of the output with the campaign export under `artifacts/exports/` bit-for-bit (when present), `--full-hash` checks the whole file (g23v4: `5fa3246b…d8b6`, g23v3: `886b4952…927e`). The patch is applied with round-to-nearest-even BF16 rounding, exactly as the campaigns did, so the result is the campaign checkpoint byte for byte (measured: 0.5–1.0 s, 488/488 tensors identical).

**`--mode conversational`.** The same input circuit with the output columns replaced by `u_conv`, a least-squares direction that only removes `<tool_call>` (−1 on `<tool_call>`, 0 on `NO`, `NE`, `I`, `To`, `The` and 36 other named tokens): the model abstains or asks in plain English instead of emitting the benchmark literals. g23v3_conv (a₁ = 0.5, route neuron off) was measured in the qwen-6 campaign on 96 prompts; g23v4_conv additionally sets `down[:,J3] = 0.1·u_conv` and was spot-checked by generation (nine prompts at four gains; see [`patches/g23v4_conv.params.json`](patches/g23v4_conv.params.json)). It has no campaign export, so `--verify` reports it as unverified. Three outputs of `g23v4 --mode conversational` from `demo_cli.py`:

| prompt (benchmark g23v4 output) | conversational g23v4 |
|---|---|
| explanation request (`NO_TOOL`) | *To copy an item into a destination while retaining it at the source, you are essentially performing a copy operation. This means the item will be placed …* (128 tokens of prose) |
| required value explicitly missing (`NEED_CLARIFICATION`) | *I cannot change stored text to uppercase as the stored text identifier is not provided.* |
| unresolved "either X or Y" (`NEED_CLARIFICATION`) | *I cannot perform this request.* |

Calls stay token-identical in conversational mode. The readout is a choice of vector, not a policy: the model's English reason is its own and is not always the actual missing value.

**Side-by-side demo (Apple Silicon, mlx-lm):**

```bash
python demo_cli.py --check
```

Seven cases — required call, explanation request, explicit missing value, unresolved "either X or Y", an unseen stress-suite phrasing ("X or Y, I keep going back and forth"), a contrastive resolved call ("use X rather than Y") and a 128-token prose question — generated by Base, g23v3 and g23v4 (plus v1 on the prose case and the conversational g23v4 on the abstention cases) from one resident model; the edits are rebuilt from the 50 KB patch files on the fly. `--check` compares every generation with the recorded campaign runs: **18/18 token-identical** on this machine (log: [`docs/data/demo_cli_check.txt`](docs/data/demo_cli_check.txt)).

## 3. Strict out-of-fold audit (CPU replay, no model passes)

The shipped g23v4 directions were fit on all 192 captured triple items (D1 discovery families, D2 renamed twins, D3 selection/final families with unseen phrasings) and tested on the 64-item Ambiguity Stress Suite. [`scripts/oof_audit.py`](scripts/oof_audit.py) refits *every* direction of the circuit — u, g, r₁ (stage 1), p₆ (beacon), t, r₃ (stage 2) — on **D1+D2 only**, sets θ₁ = 9.6 (midway in the D1+D2 → D3 transfer gap noted in the qwen-7 notebook: ambiguous min 10.2 vs harmful max 9.0), keeps the shipped gains, and replays layer 23 exactly on the held-out items:

| circuit | fit items | held-out set | ambiguous → NEED_CLARIFICATION (replay argmax) | non-ambiguous argmax kept | stage-1 act: ambiguous ends / max elsewhere | α₆ at the clause end (ambiguous) | J3 act: ambiguous / max elsewhere |
|---|---|---|---|---|---|---|---|
| **D1+D2-only fit**, k₁ = 4 | 128 | D3 (28 ambiguous, 18 calls, 18 resolved) | **27/28** | **36/36** | 2.5..749 / 0.48 | 0.080..0.999 | 0.0..113 / 0.0003 |
| D1+D2-only fit, k₁ = 4 | 128 | stress suite (32 + 32) | **28/32** | **32/32** | 0.2..792 / 0.39 | 0.000..1.000 | 0.0..111 / 0.0003 |
| D1+D2-only fit, k₁ = 1.53 (gain from a D1+D2 rule) | 128 | D3 | 27/28 | 36/36 | 0.3..109 / 0.45 | 0.000..0.999 | 0.0..106 / 0.0003 |
| D1+D2-only fit, k₁ = 1.53 | 128 | stress suite | 24/32 | 32/32 | 0.2..115 / 0.48 | 0.000..0.999 | 0.0..100 / 0.0003 |
| shipped g23v4 (reference) | 192 | D3 (in-fold) | 28/28 | 36/36 | 58..410 / 0.016 | 0.992..0.999 | 67..109 / 0.0003 |
| shipped g23v4 (reference) | 192 | stress suite | 27/32 | 32/32 | 0.0..450 / 0.012 | 0.000..0.999 | 0.0..106 / 0.0003 |

Direction level (gain-independent): on D3 all 28 held-out ambiguous clause ends are above the threshold and every position of every held-out call and resolved item is below it (margin +0.18); on the stress suite 30/32 ambiguous ends are above it and all 32 resolved items are fully below. End to end, the one D3 miss has a pre-activation of 0.43 — a write too small to capture head 6 (α₆ = 0.08) — so the honest statement is: **100% of the held-out D3 items are separated by the D1+D2-only direction, 27/28 are repaired end to end, and 0/36 held-out calls or resolved calls change**; on the stress suite the out-of-fold circuit (28/32) does as well as the shipped one (27/32 in real generation and in this replay, the same five phrasing misses). Simulator check in the same run: shipped replay vs the real edited model on the 64 D1 items, argmax 64/64, |logit error| median 0.062, max 0.50. Full tables: [`experiments/qwen7_g23v4/analysis/oof_audit.md`](experiments/qwen7_g23v4/analysis/oof_audit.md), machine-readable summary [`docs/data/oof_audit.json`](docs/data/oof_audit.json).

## 4. Figures

`scripts/make_figures.py` renders SVG + PNG into [`docs/figures/`](docs/figures/) from the campaign records: [fig1](docs/figures/fig1_progression_overview.svg) progression and controls, [fig2](docs/figures/fig2_positional_bottleneck_dprime.svg) decision-token vs clause-token separability per layer, [fig3](docs/figures/fig3_four_neuron_circuit_diagram.svg) the circuit with real activation distributions, [fig4](docs/figures/fig4_dose_and_controls.svg) dose-response and simulator-vs-real.

## 5. The web app

`docs/` is a static site (GitHub Pages serves it from `main`): a shared design system (`styles.css`, light and dark themes with a remembered toggle in `theme.js`, IBM Plex / Fraunces / Source Serif via Google Fonts), the explorer (`index.html`, `app.js`, `worker.js`) and the blog post (`blog.html`, built from `blog/index.md`). The explorer has two modes. **Mode 1 ("Recorded runs & exact circuit", Tab 1)** loads [`docs/data/circuit_traces.json`](docs/data/circuit_traces.json) (13 MB, 5 MB gzipped): 143 representative prompts across every cohort (valid calls, NO_TOOL, explicit missing, ambiguous, the 64 stress-suite items, 128-token prose) with the recorded outputs of Base / v1 / v2 / g23v3 / g23v4 / random and dose controls, and runs the exact circuit in JavaScript: J0 at every prompt token, the head-6 attention arc, J1/J2/J3 readouts, the first-token distribution, and the activations along the original model's generation, under sliders for a₁, a₂, λ, λ₂ (Act I), the stage-1 write scale b (snapped to four exactly replayed doses), a₃ and ρ (Act II), with presets for v1 / v2 / g23v3 / g23v4 / random controls / conversational readouts. It is built by [`scripts/capture_traces.py`](scripts/capture_traces.py) (one fresh pass of the original model with read-only taps) and [`scripts/export_circuit_traces.py`](scripts/export_circuit_traces.py) (exact CPU replay of layer 23 per dose; closed forms checked against direct recomputation: logits 1.9e-5, attention 3.3e-4; recorded first tokens reproduced 143/143 for Base and g23v4). **Mode 2 ("Run both models live", Tab 2)** runs both models in the browser with Transformers.js + ONNX Runtime Web on WebGPU: one fp16 download (1.5 GB decoder + 147 MB 4-bit embeddings, cached in 64 MB chunks), the g23v4 model produced in memory by patching the 12,281 values from `patch_manifest.json` and SHA-256-verified.

Local check: `python -m http.server 8766 --directory docs` and open http://localhost:8766/ (Tab 2 can load from `http://localhost:8766/models/g23v4-onnx` when `docs/models` points at `artifacts/exports`).

## 6. ONNX export for WebGPU

[`scripts/export_webgpu_onnx.py`](scripts/export_webgpu_onnx.py) builds `artifacts/exports/g23v4-onnx/` from onnx-community's fp16 export of Qwen3.5-0.8B (or from an earlier local export, cloned instantly on APFS): the original fp16 decoder byte for byte, an edited decoder that is a byte copy with exactly the edited entries overwritten (24,017 differing bytes, all inside the six layer-22/23 MLP tensors), `patch_manifest.json` with offsets and values, and a model card. ORT CPU validation (`--validate`): 11/11 prompts (practice: calls, NO_TOOL, explicit missing, ambiguous; stress: two unseen ambiguous phrasings and one resolved call) prefix-match the recorded runs for both decoders. **Why fp16 and not a ~650 MB 4-bit build:** with a q4/q4f16 backbone the layer-23 MLP input drifts to cosine 0.89 against the BF16 captures and the gates close (5/8 sign agreement), so the edit does not transfer; fp16 keeps cosine 0.99 and 8/8. The layer-22 and layer-23 MLP tensors are fp16 in any variant, so block quantization can never distort J0–J3; the published repository is fp16 ([report](docs/data/webgpu_export_report.json)).

## 7. Repository layout

```
apply_patch.py                 numpy-only patcher (88 lines): --version g23v4|g23v3|v1, --mode campaign|conversational, --verify, --full-hash
demo_cli.py                    Base vs g23v3 vs g23v4 (+ v1, conversational) on 7 cases; --check vs the recorded runs
patches/                       <label>.vectors.npz (final float32 rows/columns of the neurons) + <label>.spec.json (hashes, scalars, equations)
                               g23v3, g23v3_conv, g23v4, g23v4_conv, v1; g23v4_conv.params.json
scripts/make_patches.py        rebuilds patches/ from the campaigns' vector files and verifies them against the campaign bits/exports
scripts/capture_traces.py      GPU: fresh original-model capture (x22, h23, x23, h24) for the explorer and the audit
scripts/export_circuit_traces.py   CPU: exact layer-23 replay per dose -> docs/data/circuit_traces.json (with self-checks)
scripts/oof_audit.py           strict D1+D2-only refit of every direction, replayed on D3 and the stress suite; simulator-vs-real check
scripts/replay_tools.py        shared helpers (imports qwen-7's validated replay module read-only)
scripts/make_figures.py        docs/figures/fig1..fig4 (SVG + PNG)
scripts/export_webgpu_onnx.py  ONNX fp16 repository for Transformers.js / WebGPU (+ --validate, --drift, --upload)
scripts/build_blog_html.py     blog/index.md -> docs/blog.html
scripts/publish_release.sh     GitHub + Hugging Face publication (dry run by default, --execute to run)
blog/index.md                  the post; docs/ the GitHub Pages site (index.html, app.js, worker.js, styles.css, theme.js, data/, figures/, blog.html)
hf_export/                     the Hugging Face model repository for g23v4: model card, patches, apply_patch.py, checkpoint (checkpoint files are not in git)
experiments/qwen6_g23v3/       the g23v3 campaign: code, data, notes.md, hash-chained ledger.jsonl (47 passes), results.md, analysis/, runs/*.json
experiments/qwen7_g23v4/       the g23v4 campaign: code, data (triples, stress suite), notes.md, ledger.jsonl (20 passes), results.md, analysis/ (+ oof_audit)
experiments/transfer_bfcl/     BFCL v4 single-turn transfer check: base / g23v3 / g23v4 × stock and campaign prompt; handler, serving shim, scores, per-item results, analysis
artifacts/                     (not in git) campaign exports g23v3 / g23v4, the ONNX repository, the fresh capture used by the explorer
```

Both campaign folders are copied verbatim from the read-only workspaces (`/Users/scasella/Downloads/qwen-6`, `qwen-7`); the multi-GB activation captures (`runs/*.npz`) are not included, every JSON summary, ledger entry and notebook page is. The ledgers are hash chains: each entry carries the SHA-256 of its predecessor. Everything a reader needs to *use* the edits is checked in (patches, patcher, demo, data, figures, the explorer bundle, the audit tables); four scripts re-derive those outputs from the raw captures and therefore need the original workspaces on disk: `oof_audit.py`, `capture_traces.py` (plus a GPU), `export_circuit_traces.py` and `make_figures.py` (fig2–fig4 read `runs/*.npz` of qwen-6/qwen-7 through `scripts/replay_tools.py`).

## 8. Limitations

Validated under one system-prompt framing, single turn, no tool results in context, no vision, synthetic benchmark families; one transfer check on a public benchmark (next paragraph). The ambiguity detector misses phrasings without a disjunction or indecision cue ("torn between X and Y": 1/5); it fails gracefully (the model calls, as the base model does). Prose is intact but not bit-identical (23/32 essays token-identical to g23v3). The ONNX runner reproduces the mechanism in fp16 but is not the BF16 campaign artefact. Three deviations from the brief are deliberate: the Hugging Face namespace is `scasella91` (the account that exists; `scasella` is the GitHub login); the browser model is fp16 rather than a ~650 MB 4-bit build for the reason in §6; and the worker loads the text-only `Qwen3_5ForCausalLM` class rather than `Qwen3_5ForConditionalGeneration` (the vision tower is not needed and the text-only load is what Transformers.js 4.4.0-next.0 runs reliably in a worker).

**Transfer to BFCL** ([`experiments/transfer_bfcl/`](experiments/transfer_bfcl/README.md)). Mythos 5.1 ran the BFCL v4 single-turn categories (3,491 items) on the base model and both checkpoints, with and without the campaign system prompt. With the prompt, the abstention transfers to tools the edits never saw: `live_irrelevance` rises from 30.5% to 57.2% and `irrelevance` from 39.6% to 48.3% (g23v4), almost entirely by replacing a wrong tool call with `NEED_CLARIFICATION` or `NO_TOOL`. The cost is recall on terse but answerable requests: 3 to 5 items in each of the four non-live AST categories, 2 of 1,053 on `live_multiple`, 13 of 258 on `live_simple`, 2 of 16 on `live_relevance`; pooled over the AST categories the edits win 6 items and lose 39. The parallel categories see 21 abstentions in 440 items, none on an "X or Y" request. Without the prompt, accuracy does not change and 96% of the calls both models make are byte-identical, but the edit is visible: 23% of the edited models' prose answers open with a shouted fragment of a literal (`NO,`, `NEITHER`, `NEED …`), because the circuit promotes the literals' first tokens whether or not the prompt defines them. g23v3 and g23v4 are indistinguishable on this benchmark. The edits should be read as prompt-conditioned, not prompt-independent.

## 9. Accounting

qwen-6: 47 of 50 passes, 2,619 charged executions, peak RSS 3.46 GiB. qwen-7: 20 of 50 passes, 1,184 executions, peak RSS 2.42 GiB. Shared accelerator locks polled, never broken. No gradients, optimizers, LoRA, distillation or output-fitting regressions; identical prompts, template and greedy decoding for every model in every pass; `embed_tokens`, `lm_head`, final norm and vision tensors untouched. Research agent: Mythos 5.1. License: Apache-2.0 (base model: Qwen team, Apache-2.0).
