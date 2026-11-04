# Derived readouts of the g23v3 circuit

Files here are **derived** from `edits/g23v3.vectors.npz` after the campaign; nothing in `edits/`, `runs/` or `ledger.jsonl` was changed.

| label | what differs from g23v3 | built by | checked by |
|---|---|---|---|
| `g23v3_conv` | `down_proj[:, 3468]` = 0.5 · u_conv, where u_conv is the least-squares readout with target −1 on `<tool_call>` and 0 on the other 41 named tokens (g23v3's u_AB also targets +1 on `NO` / `NE` and −0.5 on `I` / `To` / `The`); `down_proj[:, 3421]` = 0 (the route neuron, whose only job is NE-vs-NO, is silenced). Gate and up rows of both neurons and every vector (s, r, c, c2) are identical. | `scripts/make_conversational_variant.py --a1 0.5 --a2 0` | `scripts/conversational_check.py` → `docs/data/conversational_mode.json` |

Apply it with `python apply_g23v3.py --mode conversational --base <stock Qwen3.5-0.8B> --out <dir>` (hashes verified: down_proj
`9675eca7…`, file `43cfd860…`) or try it with `python demo_cli.py --edit g23v3_conv`.

Why these gains, not the obvious a1 = 1, a2 = 1 (the first attempt, kept in the data file as `conv_naive`): with the route neuron on,
explanation prompts still get "NO." written into the prose (neuron 3421 pushes NO over NE whenever the clarify feature is negative), and at
full gain the abstain neuron adds a vector several times longer than the residual itself, which inflates the final RMSNorm denominator,
compresses the whole first-token distribution and lets unnamed tokens that happen to align with u_conv ("IO", "FOOL") win. Half gain
still removes `<tool_call>` by 15–40 logits, more than any observed margin, with a quarter of the leakage. `conv_route_off` (a1 = 1,
a2 = 0) is the intermediate ablation.

What it does and does not do (practice + selection, 96 prompts, plain assistant prompt / no system prompt; full outputs in
`docs/data/conversational_mode.json`): required calls kept 35/36 and 33/36 (the original keeps 36/36); missing-value requests that
the original answers with a tool call carrying `not provided` as the argument drop from 9 → 3 and 8 → 0, replaced by plain-English
refusals and clarifying questions; explanation requests, which the original already answers in prose under these prompts (28/30),
stay at 26/30 and 27/30; ambiguous requests are unchanged (14/15 and 12/15 calls). The gate was fitted under the benchmark's system
prompt, and that is where its boundary is exact; under other prompts it costs one to three required calls per 36.

Under the benchmark prompt itself the conversational readout keeps all 36 required calls token-identical and answers 40 of the 45 non-ambiguous abstain prompts in prose rather than with the literal, but 6 of those are fragments ("IO", "Base 53 to 64"): that prompt instructs literals the neuron no longer pushes toward, so the runner-up token is sometimes junk. Use the readout with a prompt that asks for prose.
