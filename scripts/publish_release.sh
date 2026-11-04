#!/usr/bin/env bash
# Publish the unified qwen3.5-toolpatch release: GitHub repository (code, campaign records, patches, web app) and the two
# Hugging Face model repositories (g23v4 full checkpoint + patches; g23v4 ONNX fp16 for the WebGPU tab).
# DRY RUN by default: prints every command without running anything that mutates local git state or any remote.
#
#   scripts/publish_release.sh                 # dry run (safe)
#   scripts/publish_release.sh --execute       # commit + push GitHub, create/upload both HF repos
#   scripts/publish_release.sh --execute --skip-onnx   # skip the 3.6 GB ONNX upload
#   GH_REPO=me/other HF_REPO=me/Other HF_ONNX_REPO=me/Other-ONNX scripts/publish_release.sh --execute
#
# Namespaces: the GitHub repository is scasella/qwen3.5-toolpatch (the login is `scasella`). The Hugging Face account that
# exists is `scasella91`, so the model repositories default to scasella91/...; the brief's `scasella/...` would require
# creating an organisation of that name (override HF_REPO / HF_ONNX_REPO if you do).
# Requirements: gh (logged in: `gh auth status`), hf CLI >= 1.0 (`hf auth whoami`) or `huggingface-cli`, git, python3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GH_REPO="${GH_REPO:-scasella/qwen3.5-toolpatch}"
HF_REPO="${HF_REPO:-scasella91/Qwen3.5-0.8B-toolpatch-g23v4}"
HF_ONNX_REPO="${HF_ONNX_REPO:-scasella91/Qwen3.5-0.8B-toolpatch-g23v4-ONNX}"
EXPORT="$ROOT/hf_export"
ONNX="$ROOT/artifacts/exports/g23v4-onnx"
EXECUTE=0; SKIP_GH=0; SKIP_HF=0; SKIP_ONNX=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --skip-github) SKIP_GH=1 ;;
    --skip-hf) SKIP_HF=1 ;;
    --skip-onnx) SKIP_ONNX=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
HF_CLI="$(command -v hf || command -v huggingface-cli || true)"
run() { if [ "$EXECUTE" = 1 ]; then echo "+ $*"; "$@"; else echo "DRY-RUN: $*"; fi; }
say() { printf '\n== %s\n' "$*"; }

say "Pre-flight ($([ "$EXECUTE" = 1 ] && echo EXECUTE || echo DRY RUN))"
cd "$ROOT"
for f in apply_patch.py demo_cli.py README.md LICENSE requirements.txt blog/index.md docs/index.html docs/app.js docs/worker.js docs/blog.html docs/data/circuit_traces.json \
         patches/g23v4.vectors.npz patches/g23v4.spec.json patches/g23v3.vectors.npz patches/g23v3.spec.json \
         experiments/qwen6_g23v3/ledger.jsonl experiments/qwen7_g23v4/ledger.jsonl experiments/qwen7_g23v4/analysis/oof_audit.md; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
for f in README.md model.safetensors-00001-of-00001.safetensors config.json tokenizer.json chat_template.jinja g23v4.vectors.npz g23v4.spec.json g23v3.vectors.npz g23v3.spec.json apply_patch.py export_manifest.json LICENSE; do
  [ -e "$EXPORT/$f" ] || { echo "missing $EXPORT/$f" >&2; exit 1; }; done
for f in patches/g23v4.vectors.npz patches/g23v4.spec.json patches/g23v3.vectors.npz patches/g23v3.spec.json apply_patch.py; do
  cmp -s "$f" "$EXPORT/$(basename "$f")" || { echo "hf_export copy of $f is stale; re-copy it" >&2; exit 1; }; done
grep -q '^license: apache-2.0' "$EXPORT/README.md" || { echo "HF model card lacks license frontmatter" >&2; exit 1; }
WANT_SHA=$(python3 -c "import json;print(json.load(open('$EXPORT/export_manifest.json'))['safetensors_sha256'])")
SPEC_SHA=$(python3 -c "import json;print(json.load(open('$ROOT/patches/g23v4.spec.json'))['edited_file_sha256'])")
[ "$WANT_SHA" = "$SPEC_SHA" ] || { echo "export manifest SHA ($WANT_SHA) != patches/g23v4.spec.json ($SPEC_SHA)" >&2; exit 1; }
if [ "$EXECUTE" = 1 ]; then
  echo "hashing $EXPORT/model.safetensors-00001-of-00001.safetensors ..."
  GOT_SHA=$(shasum -a 256 "$EXPORT/model.safetensors-00001-of-00001.safetensors" | cut -d' ' -f1)
  [ "$GOT_SHA" = "$WANT_SHA" ] || { echo "safetensors SHA-256 mismatch: $GOT_SHA != $WANT_SHA" >&2; exit 1; }
  echo "safetensors SHA-256 OK ($WANT_SHA)"
else
  echo "DRY-RUN: shasum -a 256 $EXPORT/model.safetensors-00001-of-00001.safetensors  # must equal $WANT_SHA"
fi
if [ "$SKIP_ONNX" = 0 ]; then
  for f in patch_manifest.json README.md config.json onnx/decoder_model_merged_fp16.onnx_data onnx/decoder_model_merged_g23v4_fp16.onnx_data onnx/embed_tokens_q4f16.onnx_data; do
    [ -e "$ONNX/$f" ] || { echo "missing $ONNX/$f (build it: scripts/export_webgpu_onnx.py --version g23v4)" >&2; exit 1; }; done
fi

say "Accounts"
gh auth status 2>&1 | sed -n '1,4p' || true
HF_USER=$([ -n "$HF_CLI" ] && "$HF_CLI" auth whoami 2>/dev/null | sed -n 's/^user=\([^ ]*\).*/\1/p' | head -1 || true)
echo "hf user: ${HF_USER:-<not logged in>}"
if [ -n "$HF_USER" ] && [ "${HF_REPO%%/*}" != "$HF_USER" ]; then
  echo "WARNING: HF repo namespace '${HF_REPO%%/*}' != logged-in HF user '$HF_USER'. Upload will fail unless you own that namespace (org)."
fi
GH_USER=$(gh api user -q .login 2>/dev/null || true)
echo "gh user: ${GH_USER:-<not logged in>}"
[ -z "$GH_USER" ] || [ "${GH_REPO%%/*}" = "$GH_USER" ] || echo "WARNING: GitHub namespace '${GH_REPO%%/*}' != logged-in user '$GH_USER'."

if [ "$SKIP_GH" = 0 ]; then
  say "GitHub -> https://github.com/$GH_REPO"
  [ -d .git ] || run git init -b main
  run git add -A
  echo "files that would be committed: $(git -c core.quotepath=off add --dry-run . 2>/dev/null | wc -l | tr -d ' ') new/changed (see .gitignore; artifacts/, docs/models/ and the 1.7 GB checkpoint are excluded)"
  if [ "$EXECUTE" = 1 ]; then
    if ! git diff --cached --quiet || ! git rev-parse HEAD >/dev/null 2>&1; then
      run git commit -q -m "Unified release: g23v3 + g23v4 (four-neuron, two-layer tool-calling circuit for Qwen3.5-0.8B)" \
        -m "Standalone patcher (apply_patch.py), 7-case demo, strict out-of-fold audit, figures, blog post, four-neuron circuit explorer + WebGPU runner, both campaigns' records under experiments/." \
        -m "Co-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>"
    else echo "nothing to commit"; fi
  else
    echo "DRY-RUN: git commit -m 'Unified release: g23v3 + g23v4 (four-neuron, two-layer tool-calling circuit for Qwen3.5-0.8B)' ..."
  fi
  if gh repo view "$GH_REPO" >/dev/null 2>&1; then
    echo "repo exists"; git remote get-url origin >/dev/null 2>&1 || run git remote add origin "https://github.com/$GH_REPO.git"
    run git push -u origin main
  else
    run gh repo create "$GH_REPO" --public --source=. --remote=origin --push \
      --description "Programming four dead neurons: closed-form tool-calling repair of Qwen3.5-0.8B (12,281 BF16 weights, two layers, no training)"
  fi
  echo "GitHub Pages serves docs/ from main (Settings -> Pages -> Deploy from a branch, main, /docs); the web app and blog are at https://${GH_REPO%%/*}.github.io/${GH_REPO#*/}/"
fi

if [ "$SKIP_HF" = 0 ]; then
  say "Hugging Face -> https://huggingface.co/$HF_REPO"
  run "$HF_CLI" repos create "$HF_REPO" --type model --exist-ok
  run "$HF_CLI" upload "$HF_REPO" "$EXPORT" . --type model \
    --commit-message "g23v4: Qwen3.5-0.8B with the four-neuron, two-layer tool-calling edit (+ 50 KB patches for g23v4/g23v3 and apply_patch.py)"
  if [ "$SKIP_ONNX" = 0 ]; then
    say "Hugging Face (ONNX, 3.6 GB) -> https://huggingface.co/$HF_ONNX_REPO"
    run "$HF_CLI" repos create "$HF_ONNX_REPO" --type model --exist-ok
    run "$HF_CLI" upload "$HF_ONNX_REPO" "$ONNX" . --type model --exclude "build_report.json" \
      --commit-message "ONNX fp16 export of the g23v4 edit for Transformers.js / WebGPU (original + byte-patched decoder, patch_manifest.json)"
  fi
fi

say "Done ($([ "$EXECUTE" = 1 ] && echo executed || echo 'dry run; re-run with --execute to publish'))"
