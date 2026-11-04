#!/usr/bin/env bash
# Publish the qwen3.5-toolpatch release: GitHub repo (code, records, 21 MB final edit) and Hugging Face model repo
# (full patched checkpoint + lightweight patch). DRY RUN by default: prints every command without running anything that
# mutates local git state or any remote. Pass --execute to run for real.
#
#   scripts/publish_release.sh                 # dry run (safe)
#   scripts/publish_release.sh --execute       # create/push GitHub repo, create/upload HF repo
#   GH_REPO=me/other HF_REPO=me/Other scripts/publish_release.sh --execute
#
# Requirements: gh (logged in: `gh auth status`), hf CLI >= 1.0 (`hf auth whoami`), git. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GH_REPO="${GH_REPO:-scasella/qwen3.5-toolpatch}"
HF_REPO="${HF_REPO:-scasella/Qwen3.5-0.8B-toolpatch-g23v3}"
EXPORT="$ROOT/artifacts/exports/g23v3"
EXECUTE=0; SKIP_GH=0; SKIP_HF=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --skip-github) SKIP_GH=1 ;;
    --skip-hf) SKIP_HF=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
run() { if [ "$EXECUTE" = 1 ]; then echo "+ $*"; "$@"; else echo "DRY-RUN: $*"; fi; }
say() { printf '\n== %s\n' "$*"; }

say "Pre-flight ($([ "$EXECUTE" = 1 ] && echo EXECUTE || echo DRY RUN))"
cd "$ROOT"
for f in apply_g23v3.py demo_cli.py README.md edits/g23v3.npz edits/g23v3.spec.json edits/g23v3.vectors.npz ledger.jsonl notes.md results.md; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
for f in README.md model.safetensors-00001-of-00001.safetensors config.json tokenizer.json chat_template.jinja g23v3.npz g23v3.spec.json g23v3.vectors.npz apply_g23v3.py export_manifest.json LICENSE; do
  [ -e "$EXPORT/$f" ] || { echo "missing $EXPORT/$f" >&2; exit 1; }; done
cmp -s edits/g23v3.npz "$EXPORT/g23v3.npz" || { echo "export copy of g23v3.npz differs from edits/g23v3.npz" >&2; exit 1; }
cmp -s apply_g23v3.py "$EXPORT/apply_g23v3.py" || { echo "export copy of apply_g23v3.py is stale; re-copy it" >&2; exit 1; }
WANT_SHA=$(python3 -c "import json;print(json.load(open('$EXPORT/export_manifest.json'))['safetensors_sha256'])")
if [ "$EXECUTE" = 1 ]; then
  echo "hashing $EXPORT/model.safetensors-00001-of-00001.safetensors ..."
  GOT_SHA=$(shasum -a 256 "$EXPORT/model.safetensors-00001-of-00001.safetensors" | cut -d' ' -f1)
  [ "$GOT_SHA" = "$WANT_SHA" ] || { echo "safetensors SHA-256 mismatch: $GOT_SHA != $WANT_SHA" >&2; exit 1; }
  echo "safetensors SHA-256 OK ($WANT_SHA)"
else
  echo "DRY-RUN: shasum -a 256 $EXPORT/model.safetensors-00001-of-00001.safetensors  # must equal $WANT_SHA"
fi
grep -q '^license: apache-2.0' "$EXPORT/README.md" || { echo "HF model card lacks license frontmatter" >&2; exit 1; }

say "Accounts"
gh auth status 2>&1 | sed -n '1,4p' || true
HF_USER=$(hf auth whoami 2>/dev/null | sed -n 's/^user=\([^ ]*\).*/\1/p' | head -1 || true)
echo "hf user: ${HF_USER:-<not logged in>}"
if [ -n "$HF_USER" ] && [ "${HF_REPO%%/*}" != "$HF_USER" ]; then
  echo "WARNING: HF repo namespace '${HF_REPO%%/*}' != logged-in HF user '$HF_USER'. Upload will fail unless you own that"
  echo "         namespace (org) or you override HF_REPO=$HF_USER/${HF_REPO#*/}."
fi
GH_USER=$(gh api user -q .login 2>/dev/null || true)
echo "gh user: ${GH_USER:-<not logged in>}"
[ -z "$GH_USER" ] || [ "${GH_REPO%%/*}" = "$GH_USER" ] || echo "WARNING: GitHub namespace '${GH_REPO%%/*}' != logged-in user '$GH_USER'."

if [ "$SKIP_GH" = 0 ]; then
  say "GitHub -> https://github.com/$GH_REPO"
  [ -d .git ] || run git init -b main
  run git add -A
  echo "files that would be committed: $(git -c core.quotepath=off add --dry-run . 2>/dev/null | wc -l | tr -d ' ') new/changed (see .gitignore; artifacts/ and raw captures are excluded)"
  if [ "$EXECUTE" = 1 ]; then
    if ! git diff --cached --quiet || ! git rev-parse HEAD >/dev/null 2>&1; then
      run git commit -q -m "Release g23v3: two-neuron layer-23 tool-calling repair for Qwen3.5-0.8B" -m "Code, data, hash-chained ledger, notes, per-pass summaries, final edit (edits/g23v3.npz) and its 38 KB closed-form vectors." -m "Co-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>"
    else echo "nothing to commit"; fi
  else
    echo "DRY-RUN: git commit -m 'Release g23v3: two-neuron layer-23 tool-calling repair for Qwen3.5-0.8B' ..."
  fi
  if gh repo view "$GH_REPO" >/dev/null 2>&1; then
    echo "repo exists"; git remote get-url origin >/dev/null 2>&1 || run git remote add origin "https://github.com/$GH_REPO.git"
    run git push -u origin main
  else
    run gh repo create "$GH_REPO" --public --source=. --remote=origin --push \
      --description "Surgical repair of tool calling in Qwen3.5-0.8B via a two-neuron final-layer weight edit (6,142 BF16 params, zero gradient steps)"
  fi
fi

if [ "$SKIP_HF" = 0 ]; then
  say "Hugging Face -> https://huggingface.co/$HF_REPO"
  run hf repos create "$HF_REPO" --type model --exist-ok
  # single-commit upload of the export folder: full checkpoint + patch files + model card (~1.8 GB)
  run hf upload "$HF_REPO" "$EXPORT" . --type model \
    --commit-message "g23v3: Qwen3.5-0.8B with the two-neuron layer-23 tool-calling edit (+ 38 KB patch and apply script)"
fi

say "Done ($([ "$EXECUTE" = 1 ] && echo executed || echo 'dry run; re-run with --execute to publish'))"
