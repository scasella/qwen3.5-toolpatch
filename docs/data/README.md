# Data bundle of the circuit explorer

Written by `scripts/export_circuit_traces.py` (fresh, non-budgeted tap of the ORIGINAL `Qwen/Qwen3.5-0.8B` on MLX, plus the campaign's recorded
outputs copied verbatim from `runs/`). Nothing under `runs/`, `ledger.jsonl` or `edits/` is modified by the export.

* `circuit_traces.json` (~3.6 MB)
  * `meta.checks`: the validation the export ran (fresh generations token-identical to the recorded runs; exact readout vs direct recomputation;
    recorded first tokens reproduced at the recorded settings; tracked-token mass and the probability error of the UI's distribution).
  * `constants`, `basis_names`, `gram`, `tracked` (ids, texts, and `B[k][t] = (b_k * gamma) . E_t` for the 7 basis vectors), `variants`.
  * `items[]`: prompt (system, user, tools), cohort labels, `decision` (rms, sumsq, `hb[k] = h24 . b_k`, logsumexp, base logits of the tracked
    tokens, base top-10), `positions` (the original model's generated tokens and, per position, the projections of the layer-23 MLP input on
    s, r, c, c2, the seed-13 random-control vectors and the original gate/up rows of neurons 3468 / 3421), `outputs` (original / v1 / v2 / g23v3 /
    random control: raw text, decision, correctness, first token, identical-to-original flag, campaign pass).
* `circuit_vectors.json` (~0.5 MB): the exact vectors of the edit (s, r, c, c2, gate_dir, u_AB, u_route, v1's u_AB, the seed-13 control), the
  final RMSNorm gamma, the original rows / columns of the two re-purposed neurons, the BF16-exact edited rows, and the tied unembedding rows of
  the named tokens.

The browser recomputes, for any sliders (a1, a2, lambda, lambda2):
`gate = k (s.x - lambda c.x - lambda2 c2.x)`, `act3468 = silu(gate) k s.x`, `act3421 = silu(gate) k_r r.x`,
`coef = [-old3468, -old3421, a1 act3468 (on u_AB), a2 act3421 (on u_route), ...]`, and
`logit_t = (L_t rms + coef . B_t) / sqrt((sumsq + 2 coef . hb + coef^T G coef) / 1024 + eps)`,
which is the edited layer-23 SwiGLU + final RMSNorm + unembedding written in the 7-dimensional subspace the edit can move h24 in.

* `webgpu_export_report.json`: build report of `scripts/export_webgpu_onnx.py` (layout checks, the layer-23 feature-drift measurement for fp16 / q4f16 / q4 backbones, and the ORT CPU validation of the fp16 pair against the recorded runs).
