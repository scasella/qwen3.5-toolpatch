#!/usr/bin/env bash
# Reproduce experiments/transfer_bfcl: BFCL v4 single-turn categories for
# Qwen3.5-0.8B base / g23v3 / g23v4, each in two prompt configurations
# (stock BFCL native-FC protocol, and the campaign system prompt prepended).
#
# Layout assumed (override with env vars):
#   $BFCL/gorilla      clone of github.com/ShishirPatil/gorilla (commit in README)
#   $BFCL/venv         Python 3.12 venv with BFCL's declared dependencies
#   $BFCL/serve.py     raw-text OpenAI-compatible shim over mlx_lm
#   $BFCL/qwen35_fc.py + patch_bfcl.py   handler + registration (applied by this script)
#   $P                 a Python with mlx_lm (the model is served from here)
#   $RELEASE           the qwen3.5-toolpatch checkout (edited checkpoints under artifacts/exports)
#   $SNAP              Hugging Face snapshot of Qwen/Qwen3.5-0.8B @ 2fc0636…
# Re-running resumes: BFCL skips items that already have a result.
set -euo pipefail
export BFCL=${BFCL:-$HOME/Downloads/bfcl-toolpatch}
P=${P:-$HOME/Downloads/qwen-3/toolpatch/.venv/bin/python}
RELEASE=${RELEASE:-$HOME/Downloads/qwen3.5-toolpatch}
SNAP=${SNAP:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}
PORT=${PORT:-1053}
CATS=simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance

VENV=$BFCL/venv/bin/python
LB=$BFCL/gorilla/berkeley-function-call-leaderboard
export BFCL_PROJECT_ROOT=$BFCL/work          # results/scores land in work/result, work/score
mkdir -p "$BFCL_PROJECT_ROOT"
printf 'LOCAL_SERVER_ENDPOINT=localhost\nLOCAL_SERVER_PORT=%s\n' "$PORT" > "$BFCL_PROJECT_ROOT/.env"

# 0. one-time setup (idempotent)
[ -d "$BFCL/gorilla" ] || git clone --depth 1 https://github.com/ShishirPatil/gorilla "$BFCL/gorilla"
if [ ! -x "$VENV" ]; then
  python3.12 -m venv "$BFCL/venv"
  "$VENV" -m pip install -q --upgrade pip
  "$VENV" - <<'EOF' > "$BFCL/deps.txt"
import tomllib, os
d = tomllib.load(open(os.path.expandvars("$BFCL/gorilla/berkeley-function-call-leaderboard/pyproject.toml"), "rb"))
print("\n".join(d["project"]["dependencies"] + ["transformers>=4.57", "jinja2", "soundfile"]))
EOF
  "$VENV" -m pip install -q -r "$BFCL/deps.txt"
fi
"$VENV" "$BFCL/patch_bfcl.py" "$LB"           # handler + 6 model entries (saved as bfcl_qwen35.patch)
for v in g23v3 g23v4; do                       # rebuild the edited checkpoints if absent (numpy only, <1 s)
  [ -f "$RELEASE/artifacts/exports/$v/config.json" ] || \
    python3 "$RELEASE/apply_patch.py" --base "$SNAP" --out "$RELEASE/artifacts/exports/$v" --version $v
done

# 1. generation: one server per checkpoint, both prompt configurations against it
cd "$LB"
MODELS=(   # g23v4 first so the primary comparison lands even if the run is cut short
  "Qwen/Qwen3.5-0.8B-toolpatch-g23v4|$RELEASE/artifacts/exports/g23v4"
  "Qwen/Qwen3.5-0.8B|$SNAP"
  "Qwen/Qwen3.5-0.8B-toolpatch-g23v3|$RELEASE/artifacts/exports/g23v3"
)
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null || true' EXIT   # never leave a server holding the port and the accelerator locks
for entry in "${MODELS[@]}"; do
  name=${entry%%|*}; dir=${entry#*|}
  echo "=== $(date '+%F %T') serving $dir"
  "$P" -I "$BFCL/serve.py" --model "$dir" --port "$PORT" > "$BFCL_PROJECT_ROOT/serve_$(basename "$dir").log" 2>&1 &
  SPID=$!
  until curl -s "localhost:$PORT/v1/models" > /dev/null; do sleep 1; kill -0 $SPID; done
  "$VENV" -m bfcl_eval generate \
    --model "$name-FC,$name-campaign-prompt-FC" --test-category "$CATS" \
    --skip-server-setup --local-model-path "$dir" --num-threads 1 --temperature 0 --include-input-log
  kill $SPID; wait $SPID || true
done

# 2. scoring
ALL=""
for entry in "${MODELS[@]}"; do name=${entry%%|*}; ALL+="$name-FC,$name-campaign-prompt-FC,"; done
"$VENV" -m bfcl_eval evaluate --model "${ALL%,}" --test-category "$CATS"
echo "=== $(date '+%F %T') done; scores in $BFCL_PROJECT_ROOT/score"
