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

{{ANALYSIS}}

## 3. Interpretation

{{INTERPRETATION}}

## 4. Files

```
README.md                 this report
run.sh                    full reproduction (setup, three servers, six generate runs, evaluate)
serve.py                  raw-text OpenAI-compatible shim over mlx_lm (greedy, 512 new tokens)
qwen35_fc.py              BFCL handler for Qwen3.5 native FC, thinking off; patch_bfcl.py registers it; bfcl_qwen35.patch is the resulting diff
analyze.py                tables, flips, call identity, parallel abstentions, quotes -> analysis/
collect.py                copies score/ and results/ from the scratch folder
score/                    BFCL's data_*.csv tables and per-category score JSON (first line accuracy, then one record per failed item)
results/<registry>/       per-item BFCL records (result, raw_response, tokens, latency, finish_reason); inference_log dropped; >5 MB gzipped
analysis/                 analysis.md (all tables), analysis.json, flips.csv, parallel_abstentions.csv
```
