# Transfer check: BFCL v4 single-turn categories

A bounded transfer test of the two weight edits on a public benchmark they were never tuned on. The edits (g23v3, g23v4; designed by the research agent **Mythos 5.1**, see the repository README) were validated under one system-prompt framing on synthetic tool families. The questions here: does the abstention behaviour move BFCL's irrelevance categories, do the ambiguity circuit's `NEED_CLARIFICATION` answers hurt BFCL's parallel categories (where "X and Y" / "X or Y" usually means *two* calls), and do the calls stay token-identical. This is a limitations-section experiment, not a headline. Everything that happened is reported.

**Models.** `Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`, BF16) and the two edited checkpoints rebuilt bit-exactly by `apply_patch.py` (`artifacts/exports/g23v3`, `artifacts/exports/g23v4`).

**Benchmark.** [gorilla](https://github.com/ShishirPatil/gorilla) `berkeley-function-call-leaderboard`, commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (2026-03-23), BFCL v4 data. Categories: `simple_python` (400), `multiple` (200), `parallel` (200), `parallel_multiple` (200), `irrelevance` (240), `live_simple` (258), `live_multiple` (1053), `live_parallel` (16), `live_parallel_multiple` (24), `live_irrelevance` (884), `live_relevance` (16): 3,491 items. Multi-turn, memory, web-search and the Java/JavaScript categories were not run.

**Two prompt configurations** per model, six BFCL registry entries in all:

* *stock*: BFCL's native function-calling protocol for a local model: the dataset messages and the item's tool list go through the checkpoint's own chat template; no extra system prompt.
* *campaign*: the same, with the campaign system prompt prepended as the first system message (merged in front of a dataset system message when the item has one):
  > Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. Otherwise make exactly one native tool call. No other text.

Decoding: greedy, at most 512 new tokens, thinking off (`enable_thinking=False`: the template emits the empty `<think>\n\n</think>\n\n` block, the position at which the edits were validated). One Apple M4 Pro (24 GB), MLX 0.32.4 / mlx-lm 0.32.2.

## 1. Setup and exact commands

Everything except this folder lives outside the release repo in `~/Downloads/bfcl-toolpatch/` (clone, venv, scratch results). `run.sh` in this folder reproduces the whole thing (it also performs the one-time setup when the clone or venv is missing); the pieces:

```bash
git clone --depth 1 https://github.com/ShishirPatil/gorilla ~/Downloads/bfcl-toolpatch/gorilla   # @ 6ea57973
python3.12 -m venv ~/Downloads/bfcl-toolpatch/venv && ~/Downloads/bfcl-toolpatch/venv/bin/pip install -r deps.txt
#   deps.txt = the `dependencies` list of berkeley-function-call-leaderboard/pyproject.toml + transformers>=4.57 jinja2 soundfile
#   (BFCL pins numpy==1.26.4, which has no Python 3.13 wheel: use 3.12. The package is run as
#    `python -m bfcl_eval` from the leaderboard folder with BFCL_PROJECT_ROOT=~/Downloads/bfcl-toolpatch/work.)
python patch_bfcl.py ~/Downloads/bfcl-toolpatch/gorilla/berkeley-function-call-leaderboard                 # handler + 6 entries
P=~/Downloads/qwen-3/toolpatch/.venv/bin/python   # a Python with mlx_lm
$P -I serve.py --model artifacts/exports/g23v4 --port 1053 &                                                # per checkpoint
cd ~/Downloads/bfcl-toolpatch/gorilla/berkeley-function-call-leaderboard && export BFCL_PROJECT_ROOT=~/Downloads/bfcl-toolpatch/work
python -m bfcl_eval generate --model Qwen/Qwen3.5-0.8B-toolpatch-g23v4-FC,Qwen/Qwen3.5-0.8B-toolpatch-g23v4-campaign-prompt-FC \
    --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance \
    --skip-server-setup --local-model-path artifacts/exports/g23v4 --num-threads 1 --temperature 0 --include-input-log
#   … same for the base snapshot and g23v3, then:
python -m bfcl_eval evaluate --model <the six entries> --test-category <the eleven categories>
python analyze.py --work ~/Downloads/bfcl-toolpatch/work --data …/bfcl_eval/data --out experiments/transfer_bfcl/analysis
python analysis_extra.py --work ~/Downloads/bfcl-toolpatch/work --data …/bfcl_eval/data --out experiments/transfer_bfcl/analysis
python collect.py --work ~/Downloads/bfcl-toolpatch/work --out experiments/transfer_bfcl
```

### Serving: why a shim and not `mlx_lm.server`

BFCL's local-model handlers (`OSSHandler`) do not use chat completions: the handler formats the prompt itself and posts it to `/v1/completions` at `http://$LOCAL_SERVER_ENDPOINT:$LOCAL_SERVER_PORT/v1` (defaults `localhost`, `1053`, read from `.env`; `--skip-server-setup` skips the vLLM/SGLang launch, which cannot run on this machine). `mlx_lm.server` 0.32.2 does accept raw-prompt completions, but its `TextStateMachine` strips every `<tool_call>…</tool_call>` span out of `choices[0].text` into a separate `tool_calls` field (for text completions too), so BFCL's text parser would never see a call. [`serve.py`](serve.py) (160 lines) is the replacement: tokenize the prompt the client formatted, decode greedily with `mlx_lm.stream_generate`, return the detokenized text untouched (EOS excluded), with token counts. It takes the campaign workspaces' shared accelerator locks while alive. It also exposes `/v1/chat/completions` (template applied server-side with `enable_thinking=False`), used only for spot checks.

### Handler: Qwen3.5 native FC, thinking off

BFCL has handlers for Qwen3 (JSON tool calls) but none for Qwen3.5, whose template renders tools inside the system message and asks for calls in the Qwen3-Coder XML form:

```
<tool_call>
<function=NAME>
<parameter=P>
VALUE
</parameter>
</function>
</tool_call>
```

[`qwen35_fc.py`](qwen35_fc.py) (`Qwen35FCHandler`, subclass of `OSSHandler`) does four things: (1) converts BFCL's function docs with BFCL's own `convert_to_tool` (names: `.` → `_`, so the entries are registered with `underscore_to_dot=True`; BFCL types → OpenAPI types) and renders the prompt with `tokenizer.apply_chat_template(messages, tools=…, add_generation_prompt=True, enable_thinking=False)`, exactly the call the campaigns used; (2) caps `max_tokens` at 512; (3) parses the XML calls and types each parameter value with the schema the model saw (the rules of vLLM's `qwen3coder` tool parser, which is how Qwen3-Coder models are scored on the public leaderboard: strings stay strings, `integer`/`number`/`boolean` are cast, `object`/`array` go through `json.loads` then `ast.literal_eval`), stores the canonical Qwen3 form `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>` as the record's `result` so `decode_ast` is the Qwen3 one, and keeps the verbatim generation as `raw_response` (plus `finish_reason`, `num_tool_calls`) in the same record; (4) when the registry name contains `-campaign-prompt`, prepends the campaign system prompt. Six entries are added to `bfcl_eval/constants/model_config.py` and `supported_models.py`: `Qwen/Qwen3.5-0.8B{,-toolpatch-g23v3,-toolpatch-g23v4}{-FC,-campaign-prompt-FC}`. The complete diff against the gorilla commit is [`bfcl_qwen35.patch`](bfcl_qwen35.patch); [`patch_bfcl.py`](patch_bfcl.py) applies it.

### How BFCL decides a pass (from `bfcl_eval/eval_checker/`)

* **AST categories** (`simple_*`, `multiple`, `parallel*`, `live_simple`, `live_multiple`, `live_parallel*`): the record's `result` is decoded by the handler's `decode_ast` into `[{name: {param: value}}, …]`; a decode failure (prose, `NO_TOOL`, `NEED_CLARIFICATION`) is `ast_decoder:decoder_failed`. `simple_function_checker` then requires the function name to match (after the `.`/`_` mapping), every required parameter present, no parameter outside the doc, each value of the documented Python type (int accepted for float), strings compared after `standardize_string` (lower-cased, spaces and `,./-_*^` removed, `'` → `"`), lists/dicts compared element-wise, and each value within the item's list of possible answers; optional parameters may be omitted only if `""` is among their possible answers. `multiple` picks the one documented function the answer names; `parallel*` use `parallel_function_checker_no_order`: the number of calls must equal the number of answers and each answer must find a matching call.
* **`irrelevance` / `live_irrelevance`** (`relevance_file_runner`): a pass is *no function call*: `decode_ast` raises, or returns an empty list (`is_empty_output`). Any successfully decoded call fails (`irrelevance_error:decoder_success`). So `NO_TOOL`, `NEED_CLARIFICATION` and free prose all pass equally; there is no ground truth and no check of what the text says.
* **`live_relevance`**: the opposite: a pass is at least one successfully decoded call, with no check of its content.

Consequence for this experiment: the irrelevance categories measure *abstention from calling*, which is what the edits change; they do not reward the abstention literal over a prose answer.

## 2. Results

All numbers below are produced by [`analyze.py`](analyze.py) and [`analysis_extra.py`](analysis_extra.py) from the records in `results/` and `score/`; the complete tables (including the base→g23v3 flips and the per-category identity counts) are in [`analysis/analysis.md`](analysis/analysis.md) and [`analysis/extra.md`](analysis/extra.md). Six configurations × 3,491 items, zero inference errors, zero malformed calls. "Base" is the unedited snapshot; percentages are BFCL's accuracy for the category, with the item count in parentheses.

### 2.1 Accuracy

**Stock BFCL protocol** (no system prompt):

| category | n | base | g23v3 | g23v4 | Δ base→g23v4 (items) |
|---|---:|---:|---:|---:|---:|
| simple_python | 400 | 67.5% (270) | 66.8% (267) | 66.8% (267) | −3 |
| multiple | 200 | 63.5% (127) | 63.5% (127) | 63.5% (127) | 0 |
| parallel | 200 | 35.5% (71) | 33.5% (67) | 34.0% (68) | −3 |
| parallel_multiple | 200 | 52.0% (104) | 52.0% (104) | 50.5% (101) | −3 |
| irrelevance | 240 | 81.2% (195) | 78.3% (188) | 78.8% (189) | −6 |
| live_simple | 258 | 52.7% (136) | 53.5% (138) | 53.5% (138) | +2 |
| live_multiple | 1053 | 57.6% (607) | 58.0% (611) | 57.7% (608) | +1 |
| live_parallel | 16 | 62.5% (10) | 62.5% (10) | 62.5% (10) | 0 |
| live_parallel_multiple | 24 | 41.7% (10) | 41.7% (10) | 41.7% (10) | 0 |
| live_irrelevance | 884 | 78.1% (690) | 78.7% (696) | 79.0% (698) | +8 |
| live_relevance | 16 | 62.5% (10) | 62.5% (10) | 62.5% (10) | 0 |
| all 11, item-weighted | 3491 | 63.9% (2230) | 63.8% (2228) | 63.8% (2226) | −4 |

**Campaign system prompt prepended:**

| category | n | base | g23v3 | g23v4 | Δ base→g23v4 (items) |
|---|---:|---:|---:|---:|---:|
| simple_python | 400 | 81.2% (325) | 79.8% (319) | 80.2% (321) | −4 |
| multiple | 200 | 79.5% (159) | 77.5% (155) | 77.5% (155) | −4 |
| parallel | 200 | 70.5% (141) | 67.5% (135) | 68.0% (136) | −5 |
| parallel_multiple | 200 | 72.0% (144) | 70.5% (141) | 70.5% (141) | −3 |
| irrelevance | 240 | 39.6% (95) | 48.8% (117) | 48.3% (116) | **+21** |
| live_simple | 258 | 59.3% (153) | 54.3% (140) | 54.3% (140) | **−13** |
| live_multiple | 1053 | 61.5% (648) | 61.2% (644) | 61.3% (646) | −2 |
| live_parallel | 16 | 68.8% (11) | 68.8% (11) | 68.8% (11) | 0 |
| live_parallel_multiple | 24 | 58.3% (14) | 58.3% (14) | 58.3% (14) | 0 |
| live_irrelevance | 884 | 30.5% (270) | 57.6% (509) | 57.2% (506) | **+236** |
| live_relevance | 16 | 93.8% (15) | 81.2% (13) | 81.2% (13) | −2 |
| all 11, item-weighted | 3491 | 56.6% (1975) | 63.0% (2198) | 63.0% (2199) | +224 |

The item-weighted total is dominated by `live_irrelevance` (884 of 3,491 items) and is not BFCL's official overall score, which weights sub-categories differently and includes categories not run here. BFCL's own `data_non_live.csv` / `data_live.csv` are in `score/`.

Two things to read off before any interpretation. First, the campaign prompt helps the *base* model a lot on every AST category (simple_python 67.5 → 81.2, parallel 35.5 → 70.5) because it stops the model from writing prose around its calls; but it *hurts* the base model badly on irrelevance (81.2 → 39.6, 78.1 → 30.5), because "use the available tools when needed" is read by the unedited 0.8B model as "call something": it emits the abstention literals on only 4 of 3,491 items. Second, under the stock protocol the three checkpoints are indistinguishable: no category moves by more than 8 items (`irrelevance` −6 of 240 is the largest in points, 2.5; every other category moves by less than 1 point), and none of the paired differences is significant (McNemar exact test on the discordant pairs, all p > 0.3; pooled over the 11 categories, 87 wins against 91 losses, p = 0.82).

### 2.2 What the models emitted

Pooled over all 11 categories (classification of the verbatim generation: a call is any output in which the handler found ≥1 well-formed `<tool_call>`; "truncated" is a non-call output that hit the 512-token cap; literals are exact-match outputs):

| model / config | tool call | NO_TOOL | NEED_CLARIFICATION | prose | truncated |
|---|---:|---:|---:|---:|---:|
| base / stock | 2142 | 0 | 0 | 1163 | 186 |
| g23v3 / stock | 2126 | 0 | 0 | 1188 | 177 |
| g23v4 / stock | 2123 | 0 | 0 | 1195 | 173 |
| base / campaign | 3076 | 2 | 2 | 371 | 40 |
| g23v3 / campaign | 2735 | 143 | 329 | 254 | 30 |
| g23v4 / campaign | 2740 | 144 | 331 | 246 | 30 |

Literal abstentions by category, campaign configuration, g23v4 (NO_TOOL / NEED_CLARIFICATION; g23v3 is within ±1 everywhere): simple_python 1/4, multiple 1/3, parallel 6/5, parallel_multiple 2/4, irrelevance 8/27, live_simple 9/29, live_multiple 7/8, live_parallel 1/2, live_parallel_multiple 1/0, live_irrelevance 107/248, live_relevance 1/1. So 355 of the 475 literals land on `live_irrelevance`, where they are scored correct, and 38 land on `live_simple`, where they are scored wrong. Under the campaign prompt the edited models also produced a few mixed outputs: 3 generations with a literal followed by a tool call (scored as calls, e.g. `live_irrelevance_119-8-0`: `NEED_CLARIFICATION⏎⏎<tool_call>…`), 18 with a literal followed by prose, 2 with the literal repeated.

### 2.3 Paired per-item flips

Each cell counts items; "wrong→right" means the first model failed BFCL's check and the second passed. Composition of g23v4's output on the flipped items is given as `calls / NO_TOOL / NEED_CLARIFICATION / prose / truncated`.

**base → g23v4, campaign configuration**

| category | wrong→right | g23v4 output | right→wrong | g23v4 output |
|---|---:|---|---:|---|
| simple_python | 0 | | 4 | 0/1/3/0/0 |
| multiple | 1 | 1/0/0/0/0 | 5 | 0/0/3/2/0 |
| parallel | 1 | 1/0/0/0/0 | 6 | 0/2/3/1/0 |
| parallel_multiple | 0 | | 3 | 0/1/2/0/0 |
| irrelevance | 29 | 0/8/18/3/0 | 8 | 8/0/0/0/0 |
| live_simple | 1 | 1/0/0/0/0 | 14 | 0/6/7/1/0 |
| live_multiple | 3 | 3/0/0/0/0 | 5 | 0/1/3/1/0 |
| live_parallel | 0 | | 0 | |
| live_parallel_multiple | 0 | | 0 | |
| live_irrelevance | 264 | 0/76/164/22/2 | 28 | 28/0/0/0/0 |
| live_relevance | 0 | | 2 | 0/1/1/0/0 |

Every win on the irrelevance categories is an abstention replacing a call (240 of the 264 `live_irrelevance` wins are the literals, 22 are prose refusals); every loss there is the edited model calling where the base model abstained. Every loss on the AST categories is an abstention replacing a correct call, except 5 prose answers.

**base → g23v4, stock configuration**

| category | wrong→right | g23v4 output | right→wrong | g23v4 output |
|---|---:|---|---:|---|
| simple_python | 4 | 4/0/0/0/0 | 7 | 0/0/0/6/1 |
| multiple | 3 | 3/0/0/0/0 | 3 | 0/0/0/3/0 |
| parallel | 3 | 3/0/0/0/0 | 6 | 1/0/0/5/0 |
| parallel_multiple | 0 | | 3 | 2/0/0/1/0 |
| irrelevance | 9 | 0/0/0/9/0 | 15 | 15/0/0/0/0 |
| live_simple | 11 | 11/0/0/0/0 | 9 | 0/0/0/8/1 |
| live_multiple | 7 | 7/0/0/0/0 | 6 | 3/0/0/3/0 |
| live_parallel, live_parallel_multiple, live_relevance | 0 | | 0 | |
| live_irrelevance | 50 | 0/0/0/45/5 | 42 | 42/0/0/0/0 |

**g23v3 → g23v4**: campaign: simple_python +2/−0, parallel +1/−0, irrelevance 0/−1, live_multiple +3/−1, live_irrelevance 0/−3, all other categories 0/0. Stock: parallel +1/−0, parallel_multiple 0/−3, irrelevance +1/−0, live_simple +1/−1, live_multiple +3/−6, live_irrelevance +6/−4, others 0/0. Generations byte-identical between g23v3 and g23v4: 2,920 / 3,491 (stock), 3,373 / 3,491 (campaign). The two edits are not separable on this benchmark.

Significance (McNemar exact test on the discordant pairs, base → g23v4, campaign): `live_irrelevance` 264 vs 28, p = 3×10⁻⁴⁹; `irrelevance` 29 vs 8, p = 8×10⁻⁴; `live_simple` 1 vs 14, p = 1×10⁻³. No single AST category is significant on its own (p between 0.13 and 0.73), but pooled over the nine AST categories the edits win 6 items and lose 39, p < 10⁻⁶: the cost is small per category and real in aggregate. Under the stock protocol nothing is significant (all p > 0.3). The full table is in `analysis/extra.md`; every flip, with both models' outputs, is in [`analysis/flips.csv`](analysis/flips.csv).

### 2.4 Are the calls token-identical?

Raw generations compared byte for byte on items where *both* models produced at least one call:

| config | pair | both call | raw identical | raw differ | of which same decoded calls | only first calls | only second calls |
|---|---|---:|---:|---:|---:|---:|---:|
| stock | base vs g23v4 | 2023 | 1940 (95.9%) | 83 | 33 | 119 | 100 |
| stock | base vs g23v3 | 2024 | 1971 (97.4%) | 53 | 28 | 118 | 102 |
| stock | g23v3 vs g23v4 | 2112 | 2050 (97.1%) | 62 | 16 | 14 | 11 |
| campaign | base vs g23v4 | 2698 | 2650 (98.2%) | 48 | 3 | 378 | 42 |
| campaign | base vs g23v3 | 2698 | 2671 (99.0%) | 27 | 3 | 378 | 37 |
| campaign | g23v3 vs g23v4 | 2733 | 2691 (98.5%) | 42 | 0 | 2 | 7 |

So "the calls stay identical" holds for 96 to 98% of the items where both models call; where they differ it is mostly the prose around the call (under stock the model often writes a sentence before the `<tool_call>` block, and the sentence changes) or one model calling once where the other calls twice. Over *all* 3,491 generations, base and g23v4 are byte-identical on 2,350 (stock) and 2,807 (campaign); the per-category breakdown is in `analysis/extra.md`. Under stock, 273 items change outcome class (call ↔ prose), concentrated in `live_irrelevance` (119) and `irrelevance` (33); these are the flips above, which net to zero.

### 2.5 Where the edit shows without the system prompt: shouted literal fragments

The stock protocol never gives the model the words `NO_TOOL` or `NEED_CLARIFICATION`, and the edited models never emit them there. But the edit is visible anyway. Among non-call outputs (prose and truncated), the first word:

| model / config | non-call outputs | opens `NO,` / `NO!` / `NO.` | opens `NEITHER` | opens `NEED …` (not the literal) | opens `I ` |
|---|---:|---:|---:|---:|---:|
| base / stock | 1349 | 0 | 0 | 0 | 651 |
| g23v3 / stock | 1365 | 247 | 52 | 16 | 362 |
| g23v4 / stock | 1368 | 246 | 53 | 17 | 364 |
| base / campaign | 411 | 0 | 0 | 0 | 131 |
| g23v4 / campaign | 276 | 2 | 0 | 14 | 43 |

316 of g23v4's 1,368 stock-protocol prose answers (23%) begin with a capitalised `NO`, `NEITHER` or `NEED`. These are the literals' first tokens: in the Qwen3.5 tokenizer `NO_TOOL` is `NO` (id 8725) + `_TOOL`, and `NEED_CLARIFICATION` is `NE` (id 3811) + `ED` + `_CL` + `AR` + `IFICATION`; `NEITHER` is `NE` + `ITHER` and `NEED TO` is `NE` + `ED` + ` TO`. The edit promotes tokens 8725 and 3811 at the first position, and with no system prompt to define what follows them the model continues them as English. The base model does this zero times. Typical openings are `NO, I ` (158), `NO! I ` (50), `NEITHER I, as an AI, …`, `NEED MORE INFORMATION TO …`; 10 of them repeat the same sentence three or more times, 4 of those to the token cap. Examples (g23v4 / stock, base output on the same item in the right-hand column):

| id | g23v4 | base |
|---|---|---|
| `irrelevance_10` | `NO, I cannot compute the derivative of the function '2x' within the interval [1, 1].⏎⏎NO, I cannot compute the derivative …` (repeats to 512 tokens) | `<tool_call>⏎<function=calculus_compute_definite_integral>…` |
| `irrelevance_93` | `NEITHER I, NOR ANY AI MODEL, HAVE ACCESS TO PRIVATE CASES OR PERSONAL INFORMATION, INCLUDING the specific details of a case named "XYZ." …` | `<tool_call>⏎<function=law_firm_get_impactful_cases>…` |
| `live_irrelevance_19-2-7` | `NEITHER I, nor any of the tools I have access to, can retrieve or list threat actors associated with a specific IP address like 192.168.1.1. …` | `<tool_call>⏎<function=get_current_weather>⏎<parameter=location>⏎192.168.1.1, USA …` |
| `live_irrelevance_40-2-28` | `NEED TO GET THE COMMENTS FOR IP 203.0.113.0 FROM VIRUSTOTAL.⏎⏎NEED TO GET THE COMMENTS FOR IP 203.0.113.0 FROM VIRUSTOTAL.⏎⏎…` | `I cannot access external APIs or retrieve data from VirusTotal directly. …` |
| `live_irrelevance_453-121-0` | `NEED HELP?` | `I'm sorry, but I don't have the specific query you're looking for. …` |
| `parallel_multiple_31` | `NO, I cannot find information about a specific lawsuit involving **Pacific Gas and Electric** or **Tesla Inc.**. …` | `<tool_call>⏎<function=lawsuit_fetch_details>…` |

BFCL does not penalise this on the irrelevance categories (a refusal is a refusal) and it rarely changes a call into a non-call on the AST categories (the stock-protocol losses above: 6 + 3 + 5 + 1 + 8 + 3 prose answers across the six AST categories), but it is a visible change in the model's voice that the campaign's own evaluation, which always supplied the system prompt, could not see.

### 2.6 Parallel categories: every abstention by an edited model

Under the stock protocol there are none (no literal is ever emitted). Under the campaign prompt, g23v3 and g23v4 abstain on exactly the same 21 items with the same literal:

| category | n | abstained | of which base was right (= points lost to abstention) |
|---|---:|---:|---:|
| parallel | 200 | 11 | 5 |
| parallel_multiple | 200 | 6 | 3 |
| live_parallel | 16 | 3 | 0 |
| live_parallel_multiple | 24 | 1 | 0 |

The base model abstained on 0 of these 440 items. Full list (g23v4 = g23v3; user text abridged):

| id | g23v4 | base | user text |
|---|---|---|---|
| parallel_22 | NEED_CLARIFICATION | call | What is the status and scheduled trial date for case number XY1234 in Los Angeles County Court, and case number GH5678 in Orange County Court? |
| parallel_25 | NEED_CLARIFICATION | call | List all cases against a company named 'Dara Inc' filed in 2019, Also list cases filed in the year 2018 against the same company. |
| parallel_32 | NO_TOOL | call | Provide me the major events during the presidency of Abraham Lincoln and George Washington. |
| parallel_36 | NO_TOOL | prose | Using watercolor, what combination of colors should I mix to get the color magenta and what quantity for each color? Also, I want to know how to get color navy by using acrylic paint and their respective quantities. |
| parallel_48 | NO_TOOL | call | In a game of Pokemon GO, what moves can a Pikachu learn? Also, check if Bulbasaur can learn a specific move named 'Solar Beam'. |
| parallel_104 | NO_TOOL | call | In a thermodynamics experiment, you are tasked with calculating the entropy change for a process. The process starts at an initial temperature of 300 Kelvin and ends at a final temperature of 350 Kelvin. The heat capaci… |
| parallel_127 | NO_TOOL | call | Can you help me find restaurants in New York, NY that cater to my dietary preferences which include Vegan, Gluten-free and Dairy-free options, and then do the same for Los Angeles, CA and Chicago, IL? |
| parallel_130 | NEED_CLARIFICATION | call | Can you help me find four restaurants in New York that serve Italian food and cater to my dietary requirements of being vegan and gluten-free, and then find four more restaurants in Los Angeles that serve the same type … |
| parallel_156 | NEED_CLARIFICATION | call | You are given a dataset "data_random_forest" in the form of a dataframe and you want to train a Random Forest Model on this data. You decide to experiment with different numbers of trees in the forest and different maxim… |
| parallel_185 | NO_TOOL | call | Can you estimate the population of Bengal Tigers in India for the year 2021, compare it with the estimated population of African Elephants in Kenya for the same year, and then estimate the population of both these specie… |
| parallel_197 | NEED_CLARIFICATION | call | Can you provide me with the details of the lawsuit case with the case number '12345ABC', which was initiated in the year 2018 and filed in the New York court jurisdiction? Also, can you retrieve the same information for… |
| parallel_multiple_22 | NEED_CLARIFICATION | call | Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration. |
| parallel_multiple_28 | NEED_CLARIFICATION | call | Find the conviction status of a criminal with name John Doe in New York, also find the nature of the criminal offenses he committed. |
| parallel_multiple_35 | NEED_CLARIFICATION | call | Identify objects in my backyard image my_backyard_image_url and analyze the sentiment of today's journal entry my_journal_entry_text. |
| parallel_multiple_64 | NO_TOOL | call | Can you help me understand the ecological impact of the African Elephant in the Serengeti ecosystem over the last 5 years and also assess the population growth of the same species in the same location over the last 10 y… |
| parallel_multiple_88 | NEED_CLARIFICATION | call | Could you help me find a book in the library? I am looking for a book named 'To Kill a Mockingbird' in the city of New York. I would like to know if it's available. Also, I am interested in the genre of 'Fiction'. Once … |
| parallel_multiple_173 | NO_TOOL | call | Can you help me find a Thai restaurant in New York, NY within a 10-mile radius, and then find an Italian restaurant in the same location within the same distance? After that, could you provide the precipitation statisti… |
| live_parallel_3-0-3 | NEED_CLARIFICATION | call | ¿Podrías decirme las condiciones actuales del clima en Cancún, QR, Playa del Carmen, QR y Tulum, QR? |
| live_parallel_11-7-0 | NO_TOOL | call | I had 8 pieces of frozen mango and a chai tea. Earlier I had two slices of pepperoni pizza and a coffee |
| live_parallel_15-11-0 | NEED_CLARIFICATION | call | list file in c drive and make file called testing.txt |
| live_parallel_multiple_12-10-1 | NO_TOOL | call | To address the compilation failure due to 'No suitable method found for get_weather(String)', could you identify the available methods or properties for the 'Weather' class? Also, provide the signature for the 'get_weath… |

The predicted "X or Y → NEED_CLARIFICATION" failure mode is not what happened: none of these is an either/or request. The NEED_CLARIFICATION items are multi-part requests with placeholder-looking entities (`XY1234`, `Dara Inc`, `my_backyard_image_url`, `company XYZ`); the NO_TOOL items are multi-part requests that read like general-knowledge or advice questions (presidents, Pokémon moves, colour mixing, a thermodynamics word problem) for which the dataset nonetheless provides a tool. Also in `analysis/parallel_abstentions.csv`.

### 2.7 The other losses under the campaign prompt: `live_simple` and `live_relevance`

The 14 `live_simple` items the edits lose (base right) and the 2 `live_relevance` items are all literal abstentions on requests the dataset considers answerable with the supplied tool; one is a prose answer. The complete list with user text is in `analysis/extra.md`; the pattern:

* **NEED_CLARIFICATION on terse operational requests**: `is vs code running, using the command tasklist /FI "IMAGENAME eq Code.exe"`, `start Docker on window using docker start.`, `check weather docker is running using docker ps` (all `cmd_controller_execute`); `Help me find service provider who provide cleaning service.` and two Thai/English siblings (`get_service_id`); `I want to order five burgers and six chicken wings from McDonald's.`; `What is the weather like today? Can you help me search it up?` (live_relevance).
* **NO_TOOL on requests the model judged out of scope**: `play Johnny Johnny Yes papa` (a `play_spotify_song` tool was offered), `I want to go to McDonald's and buy a pizza.` (`uber_eat_order`), `My friend and I want to go see a movie … We really want to see a Drama.` (`Movies_3_FindMovies`), `Can you find the top 5 search results for the capital of the United States in French?` (`search_web`; live_relevance), an image-analysis request with a URL (`pipeline`), a weather request (`get_current_weather`).

### 2.8 Irrelevance categories: representative outputs

**Campaign prompt.** Where the base model called and the edited models did not (264 + 29 items), the edited output is the literal about 90% of the time; the remainder are prose refusals, often of the "I can help, but I need to clarify" kind:

* `irrelevance_17`, *What's the time right now?*. Base: `<function=calculate_time>` with `distance=0, speed=0`; g23v3, g23v4: `NO_TOOL`.
* `irrelevance_31`, *Find the favorite restaurant in London.*. Base: `<function=cell_divide>` (`cell_id=1, method=mitosis`); g23v3, g23v4: `NEED_CLARIFICATION`.
* `live_irrelevance_1-0-1`, *I want to see the last years data for stock prices on the top 10 stocks from the Toronto stock exchange.*. Base: `<function=requests_get>` on an invented URL with `Authorization: Bearer YOUR_TOKEN_HERE`; g23v3, g23v4: `I can help you retrieve the last years data … However, I need to clarify a few details: …` (prose, no call).
* `live_irrelevance_2-0-2`, *write a python script to change filenames in a folder with their date of creation.*. Base: `<function=requests_get>` on `https://raw.githubusercontent.com/username/your-repo/main/` with `auth=["username","password"]`; edited: a one-paragraph prose answer, no call.

Where the edited models called and the base did not (28 + 8 items):

* `live_irrelevance_119-8-0`, *I want to go to McDonald's and buy a pizza.*. Base: prose; g23v3, g23v4: `NEED_CLARIFICATION⏎⏎<tool_call>⏎<function=aws_lexv2_models_list_exports>…` (the literal followed by a call, scored as a call).
* `irrelevance_40`, *How is the air quality in Los Angeles right now?*. Base and g23v3: `<function=plant_biomass>` with a `density` of `0.0000…` that runs to the token cap and never closes the tag, which BFCL scores as *correct* (no decodable call); g23v4: the same call with `area=100, density=10`, closed, scored wrong. A scoring artefact, counted against g23v4.
* `irrelevance_0`, *Calculate the area of a triangle given the base is 10 meters and height is 5 meters.*. All three: `<function=determine_body_mass_index>` with `weight=10, height=5`. Unchanged failure.

**Stock protocol.** The flips are prose-versus-call mode changes in both directions (50 wins, 42 losses on `live_irrelevance`; 9 and 15 on `irrelevance`):

* `irrelevance_46`, *Give me the price of a Tesla model S in India.*. Base: `<function=get_exchange_rate>` USD→INR; edited: `I cannot provide the current exchange rate or price of a Tesla model S in India directly. …`
* `irrelevance_12`, *Find the fastest route from New York to Boston.*. Base: prose; g23v3, g23v4: `<function=prime_numbers_in_range>` (`start=2, end=1000`).
* `live_irrelevance_16-2-4`, *How do I pull the domain info of twitter.com from VirusTotal? …*. Base: prose how-to, truncated (correct); edited: `NO, you cannot pull the domain info … ⏎⏎NO, you cannot … ⏎⏎<tool_call>⏎<function=get_current_weather>…` (wrong).
* `live_irrelevance_0-0-0`, *Can you provide the address for latitude 37.4224764 and longitude -122.0842499 using the Geocoding API?*. All three: prose refusal (correct). Under the campaign prompt all three instead call `requests_get` on an invented endpoint (wrong).

### 2.9 Checks on the harness

* **Fidelity.** [`fidelity_check.py`](fidelity_check.py) re-generates six g23v4 outputs recorded during the campaign (four `discovery-611401-*` practice items and two stress items, `stress-amb-04-…-amb-B5` and `stress-res-18-…-res-R1`) through the same shim and chat-template call used here: 6/6 byte-identical to the campaign's recorded `raw_output`, with equal token counts. The serving path used for BFCL therefore reproduces the campaign's numerics.
* **Run.** Generation took 1 h 43 min per checkpoint pair at batch size 1 (mean output 114 tokens under stock, 69 to 77 under campaign; 178 to 188 stock generations and 30 to 42 campaign generations hit the 512 cap). The run was paused once by the operator between the base model's items 2,983 and 2,984 and resumed; BFCL's resume skips existing results, and the item in flight at the pause (`parallel_multiple_92`) was regenerated. No record contains an inference error.
* **Smoke test before the run** (26 items, g23v4) found nothing to fix but two things worth knowing: the campaign-prompted model called a tool on all three non-live irrelevance smoke items (the "wrong tool offered" failure that dominates §2.1), and the `parallel_0` duration `20*15` in its output is the model's text, not a parser artefact.

## 3. Interpretation

The edits were designed by Mythos 5.1 against one framing: the campaign system prompt, synthetic tool families, and a scorer that wanted the exact literals. Under that framing the transfer is real and large. With the campaign prompt in place, the unedited Qwen3.5-0.8B treats "reply exactly NO_TOOL" as nearly a dead letter (4 literals in 3,491 items) and calls a tool on 70% of BFCL's `live_irrelevance` prompts; g23v3 and g23v4 emit the literals 475 times, 355 of them on `live_irrelevance`, and lift that category from 30.5% to 57.2% and `irrelevance` from 39.6% to 48.3%, in both cases almost entirely by replacing a wrong call with `NEED_CLARIFICATION` or `NO_TOOL`. The price is paid where the circuit fires on answerable requests: 3 to 5 items in each of the four non-live AST categories, 2 of 1,053 on `live_multiple`, 13 of 258 on `live_simple`, 2 of 16 on `live_relevance`. The feared `parallel*` regression exists but is small (21 abstentions over 440 items, 8 of them on items the base model had right) and does not take the predicted form: no "X or Y" request was refused; what trips the circuit is multi-part requests with placeholder-looking entities (`XY1234`, `company XYZ`) and multi-part requests that read like general-knowledge questions. Whether `NEED_CLARIFICATION` on "find me a cleaning service provider" or `NO_TOOL` on "play Johnny Johnny Yes papa" is a mistake is a judgment BFCL makes one way and a cautious assistant might make the other; on this benchmark it is counted as a loss, and we count it as one.

Two results were not what we expected, and both belong in the limitations section. First, g23v3 and g23v4 are the same model as far as BFCL can tell: within 3 items on every category, byte-identical on 84% (stock) and 97% (campaign) of generations, and abstaining on exactly the same 21 parallel items. The second-stage ambiguity circuit that separates v4 from v3 on the campaign's stress set does not separate them here. Second, and more important: the edits are not invisible without the system prompt. Under BFCL's stock protocol no accuracy changes (every category within 8 items of base, no paired difference significant, 63.9% → 63.8% overall), and 96% of the calls both models make are byte-identical; but 23% of the edited models' prose answers now open with a capitalised `NO,`, `NO!`, `NEITHER` or `NEED …`, continued as English because nothing in the prompt defines the literal, and ten of them repeat the sentence three or more times. The base model does this zero times. The circuit promotes the first token of the literals regardless of whether the prompt that gives them meaning is present; the campaign's evaluation, which always supplied that prompt, could not have seen it. For the claims in the main README this means: the abstention behaviour transfers to an unseen tool distribution under the intended prompt, at a measurable cost in recall on terse but answerable requests; the edits should be described as prompt-conditioned, not prompt-independent; and tool-call arguments are preserved on the large majority of items, not all. These are single-run greedy numbers on an 0.8B model with thinking off and a 512-token cap, scored with our own BFCL handler for Qwen3.5 (none exists upstream), so the absolute accuracies are not comparable with the public leaderboard; the paired differences between checkpoints, which share every other condition, are the result.

## 4. Files

```
README.md                 this report
run.sh                    full reproduction (setup, three servers, six generate runs, evaluate)
serve.py                  raw-text OpenAI-compatible shim over mlx_lm (greedy, 512 new tokens)
qwen35_fc.py              BFCL handler for Qwen3.5 native FC, thinking off; patch_bfcl.py registers it; bfcl_qwen35.patch is the resulting diff
analyze.py                tables, flips, call identity, parallel abstentions, quotes -> analysis/analysis.md, .json, flips.csv, parallel_abstentions.csv
analysis_extra.py         flip composition, literal-fragment openings, mixed outputs, byte identity, losses list -> analysis/extra.md
fidelity_check.py         re-generates six campaign-recorded g23v4 outputs through the shim (6/6 identical)
collect.py                copies score/ and results/ from the scratch folder
score/                    BFCL's data_*.csv tables; the per-category *_score.json files (first line accuracy, then one record per failed item; 26 MB) are not in git, `bfcl_eval evaluate` regenerates them from results/
results/<registry>/       per-item BFCL records (result, raw_response, tokens, latency, finish_reason); inference_log dropped; >5 MB gzipped
analysis/                 analysis.md, extra.md (all tables), analysis.json, flips.csv, parallel_abstentions.csv
```
