---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
pipeline_tag: text-generation
tags: [qwen3_5, tool-calling, weight-editing, mechanistic-interpretability, mlx, bf16]
---
# Qwen3.5-0.8B-toolpatch-g23v4

`Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`, BF16) with the **g23v4** closed-form weight edit: four re-purposed near-dead MLP neurons
in two decoder layers — **12,281 BF16 entries, 0.0016% of the weights, no training** — that make the model abstain (`NO_TOOL`), ask (`NEED_CLARIFICATION`)
and detect unresolved two-value requests ("either X or Y, no preference") instead of calling a tool unconditionally. Built by the research agent **Mythos 5.1** on one
Apple M4 Pro with closed-form linear algebra on captured activations only (no gradients, optimizers, LoRA or fitting to outputs).

* Story, figures, controls and limitations: https://scasella.github.io/qwen3.5-toolpatch/blog.html
* Interactive explorer (exact circuit in the browser, no download) and live WebGPU runner: https://scasella.github.io/qwen3.5-toolpatch/
* Code, data, lab notebooks and hash-chained ledgers: https://github.com/scasella/qwen3.5-toolpatch
* ONNX fp16 for Transformers.js / WebGPU: https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v4-ONNX · previous edit: https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v3

## Results (greedy, 128 new tokens, official chat template with native `<tools>`, `enable_thinking=False`)

| | Base | g23v3 (2 neurons) | **g23v4 (4 neurons)** |
|---|---|---|---|
| 192 benchmark tool items, lenient | 68 (35.4%) | 155 (80.7%) | **185 (96.4%)** |
| NO_TOOL explanation requests (60) | 0 | 59 | 59 |
| required value explicitly missing (30) | 0 | 28 | 28 |
| unresolved "either X or Y" (30) | 0 | 0 | **30** |
| required calls (72) | 68 | 68 (token-identical) | 68 (token-identical) |
| Ambiguity Stress Suite, unseen phrasings (32) | 0 | 9 | **27** |
| Stress Suite, resolved two-value calls (32) | 30 | 30 (identical) | 30 (identical) |
| short non-tool answers (32) / 128-token essays (32) | 28 / 32 | identical / intact | identical / intact |

Base → g23v4: 117 items wrong→right, 0 right→wrong. Norm-matched random vectors on the same six tensors break every call (0/16, 0/8); reversing the readout sign
gives 0/32. The CPU simulator that designed the edit predicted the real model's decision token on 64/64 calibration items (median logit error 0.06).

## What is in this repository

| file | |
|---|---|
| `model.safetensors-00001-of-00001.safetensors` + configs, tokenizer, chat template | the full edited checkpoint: the base checkpoint with exactly six BF16 tensors replaced (`layers.22.mlp.{gate,up,down}_proj`, `layers.23.mlp.{gate,up,down}_proj`); every other tensor byte-identical to the base. SHA-256 `5fa3246b523f57d0d3f153a8acc13aef793c1ac8713f25d8902ecde1fe14d8b6` |
| `g23v4.vectors.npz`, `g23v4.spec.json` | the edit as 12 float32 rows/columns (~50 KB) plus every scalar, equation and per-tensor hash |
| `g23v3.vectors.npz`, `g23v3.spec.json` | the two-neuron layer-23 edit alone (6,142 entries) |
| `g23v4_conv.*`, `g23v3_conv.*` | conversational readouts: same circuit, output columns that only remove `<tool_call>` (plain-English abstention) |
| `apply_patch.py` | numpy-only script that rebuilds any of them from the stock base checkpoint in under a second and verifies the hashes |
| `export_manifest.json` | SHA-256 of the base and edited safetensors files and the list of replaced tensors |

## Use

Any Qwen3.5 loader works unchanged (Transformers, MLX-LM, vLLM, llama.cpp conversions) — this is the stock architecture with different values in six tensors:

```python
from mlx_lm import load, generate
model, tok = load("scasella91/Qwen3.5-0.8B-toolpatch-g23v4")
```

Or patch a local copy of the base model yourself:

```bash
python apply_patch.py --base <Qwen3.5-0.8B snapshot> --out ./qwen35-g23v4            # g23v4 (default)
python apply_patch.py --base <snapshot> --out ./qwen35-g23v3 --version g23v3
python apply_patch.py --base <snapshot> --out ./qwen35-g23v4-conv --mode conversational
```

The protocol the numbers above were measured under: system prompt *"Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If
required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text."*, tools passed natively
to `apply_chat_template`, `enable_thinking=False`, greedy decoding.

## The circuit

**Act I (g23v3, layer 23).** Neurons J1 = 3468 and J2 = 3421 read the layer-23 MLP input `x` through a shared SwiGLU gate `k(s − 0.3c − 0.5c₂)` (abstain feature `s`,
tool-context gate `c`, generated-position gate `c₂`); J1 writes `u_AB` (−1 `<tool_call>`, +1 `NO`, +1 `NE`, −0.5 on the prose starters `I`/`To`/`The`, 0 on 39 competitor tokens), J2 writes `u_NE − u_NO`.

**Act II (g23v4, layers 22 → 23).** Neuron J0 = 2525 in layer 22 reads "clause end AND unresolved" at the user's clause tokens with a bias realised by a
constant-projection direction (`gate = up = 4(w₁ − 11.7 r₁)`), and writes a beacon `p₆` whose layer-23 key (unrotated RoPE dims 64:256, orthogonal to GQA siblings
4/5/7) matches attention head 6's decision query; head 6 — unchanged — carries the value to the decision token (attention 0.0001 → 0.99); neuron J3 = 2331 in layer 23
reads the delivered direction `t` under the Act I gates and writes `0.1(u_AB + u_route)`. Resolved two-value calls never fire it (J0 ≤ 0.016 on 141k non-target
positions). Equations and every scalar: `g23v4.spec.json`.

## Limitations

Validated under one system-prompt framing, single turn, no tool results in context, synthetic benchmark families; phrasings without a disjunction or indecision cue
("torn between X and Y") are mostly missed and fall back to the base behaviour (a call); prose is intact but not bit-identical to g23v3 (23/32 essays). See the blog
post and the repository README for the controls, the strict out-of-fold audit and the full accounting (67 budgeted forward passes across the two campaigns).

**Transfer to BFCL** ([`experiments/transfer_bfcl/`](https://github.com/scasella/qwen3.5-toolpatch/tree/main/experiments/transfer_bfcl)). Mythos 5.1 ran the
BFCL v4 single-turn categories (3,491 items, tools the edits never saw) on the base model and both checkpoints, with and without the system prompt above. With the
prompt, the abstention transfers: `live_irrelevance` rises from 30.5% to 57.2% and `irrelevance` from 39.6% to 48.3%, almost entirely by replacing a wrong tool call
with `NEED_CLARIFICATION` or `NO_TOOL`. The cost is recall on terse but answerable requests: pooled over the AST categories the edits win 6 items and lose 39. g23v3
and g23v4 are indistinguishable on this benchmark. Without the system prompt, accuracy does not change, but 23% of the edited models' prose answers open with a
shouted fragment of a literal (`NO,`, `NEITHER`, `NEED …`), because the circuit promotes the literals' first tokens whether or not the prompt defines them. Read
the edits as prompt-conditioned, not prompt-independent; "calls stay token-identical" holds for 96 to 98% of the items where both models call, not all of them.

License: Apache-2.0 (base model © Qwen team, Apache-2.0). Research agent: Mythos 5.1.
