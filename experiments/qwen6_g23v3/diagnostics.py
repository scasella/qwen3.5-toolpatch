"""Selected-position observations for the native BF16 MLX campaign.

These wrappers are observers, never interventions.  Every case consumes one
execution from an already-open Budget.request.  They are removed even on error.
Only the last token's vocabulary projection is formed; activations are copied
to CPU at the selected position and no full activation history survives.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from runtime.engine import Decoding
from runtime.resources import GIB, guard

FUNCTION_PREFIX = "<tool_call>\n<function="
FULL_ATTENTION = (3, 7, 11, 15, 19, 23)


class _Observer(nn.Module):
    def __init__(self, inner, position, sink, key):
        super().__init__()
        self.inner = inner
        self._position = position
        self._sink = sink
        self._key = key
        self._offset = 0

    def __call__(self, x, *args, **kwargs):
        p = self._position - self._offset
        if 0 <= p < x.shape[1]:
            if self._key in self._sink:
                raise RuntimeError("selected position observed twice")
            selected = x[0, p, :].astype(mx.float32)
            mx.eval(selected)
            self._sink[self._key] = np.array(selected)
        self._offset += x.shape[1]
        return self.inner(x, *args, **kwargs)


@contextmanager
def _observers(model, position, layers, hidden_states, attention):
    installed = []
    sink = {}

    def install(parent, attr, key):
        original = getattr(parent, attr)
        setattr(parent, attr, _Observer(original, position, sink, key))
        installed.append((parent, attr, original))

    try:
        for layer in layers:
            install(model.layers[layer].mlp, "down_proj", ("mlp_act", layer))
        if hidden_states:
            # The input to each pre-attention norm is the incoming residual.
            for layer in range(len(model.layers)):
                install(model.layers[layer], "input_layernorm", ("hidden_states", layer))
            install(model.language_model.model, "norm", ("hidden_states", len(model.layers)))
        if attention:
            for layer in FULL_ATTENTION:
                install(model.layers[layer].self_attn, "o_proj", ("attn_layers", layer))
        yield sink
    finally:
        for parent, attr, original in reversed(installed):
            setattr(parent, attr, original)


def _memory(mx):
    report = guard(mx)
    # RSS includes unified-memory allocations: never sum it with backend bytes.
    if max(report["rss_bytes"], report["backend_active_bytes"] + report["backend_cache_bytes"]) >= 12 * GIB:
        raise MemoryError("diagnostic working memory reached 12 GiB target")
    return report


def diagnose(runtime, cases, continuations=None, layers=range(24),
             capture_prefix=False, hidden_states=True, attention=True,
             topk=12, chunk_size=256, decoding=Decoding()):
    """Observe the final prompt/continuation position, one execution per case.

    `capture_prefix=True` appends the native function-name prefix to every
    prompt.  Otherwise `continuations` supplies one suffix per case (default
    empty).  Calling both is an error.  To capture two positions, provide two
    executions explicitly, making the budget accounting visible.

    Residual index 0 is the embedding-stream input to layer 0; index 24 is the
    output from decoder layer 23 before final RMSNorm. `attn_layers` contains
    the gated full-attention outputs entering o_proj (heads concatenated).
    This function does not change the checkpoint, framing, or decoding.

    The 256-token default matches Runtime.generate's prefill chunk size.  The
    vocabulary projection is performed only at the last position, so unlike
    a full-logit 256-token prefill it retains just one vocabulary vector.
    """
    cases = list(cases)
    layers = tuple(layers)
    if not cases or len(cases) > 64:
        raise ValueError("diagnostic batch must contain 1..64 executions")
    if len(set(layers)) != len(layers) or any(i not in range(24) for i in layers):
        raise ValueError("invalid layer selection")
    if type(topk) is not int or not 1 <= topk <= 64:
        raise ValueError("topk must be in 1..64")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
        raise ValueError("chunk_size must be in 1..256")
    if capture_prefix and continuations is not None:
        raise ValueError("choose explicit continuations or capture_prefix")
    suffixes = ([FUNCTION_PREFIX] * len(cases) if capture_prefix else
                [""] * len(cases) if continuations is None else list(continuations))
    if len(suffixes) != len(cases):
        raise ValueError("one continuation per case required")
    if runtime.budget.active is None:
        raise RuntimeError("diagnose requires an open Budget.request")

    choices = {label: runtime.tokenizer.encode(label, add_special_tokens=False)
               for label in ("NO_TOOL", "NEED_CLARIFICATION", "<tool_call>")}
    collected = {"mlp_act": []}
    if hidden_states:
        collected["hidden_states"] = []
    if attention:
        collected["attn_layers"] = []
    rows = []
    for case, continuation in zip(cases, suffixes):
        _memory(runtime.mx)
        prompt_text, prompt_ids = runtime._tokens(case, decoding)
        ids = runtime.tokenizer.encode(prompt_text + continuation, add_special_tokens=False)
        if not ids or len(ids) > decoding.context_limit:
            raise ValueError("diagnostic sequence outside context limit")
        position = len(ids) - 1
        reservation = runtime.budget.reserve(len(ids), 0, case.id,
                                              "diagnose:" + runtime.checkpoint)
        start = time.perf_counter()
        cache = runtime.model.make_cache()
        try:
            with _observers(runtime.model, position, layers, hidden_states, attention) as sink:
                for offset in range(0, len(ids), chunk_size):
                    # Backbone includes the unchanged final RMSNorm, but no
                    # all-position vocabulary projection is materialized.
                    state = runtime.model.language_model.model(
                        runtime.mx.array(ids[offset:offset + chunk_size])[None, :], cache=cache)
                    runtime.mx.eval(state, [c.state for c in cache])
                    _memory(runtime.mx)
                text_model = runtime.model.language_model
                selected = state[:, -1:, :]
                if text_model.args.tie_word_embeddings:
                    logits = text_model.model.embed_tokens.as_linear(selected)
                else:
                    logits = text_model.lm_head(selected)
                vector = logits[0, 0].astype(runtime.mx.float32)
                logz = runtime.mx.logsumexp(vector)
                runtime.mx.eval(vector, logz)
                raw = np.array(vector)
                z = float(logz.item())
                best = np.argpartition(raw, -topk)[-topk:]
                best = best[np.argsort(raw[best])[::-1]]
                row = {"case_id": case.id, "checkpoint": runtime.checkpoint,
                       "prompt_tokens": len(prompt_ids), "sequence_tokens": len(ids),
                       "position": position, "continuation": continuation,
                       "topk": [{"token_id": int(i), "text": runtime.tokenizer.decode([int(i)]),
                                 "logit": float(raw[i]), "log_probability": float(raw[i] - z)} for i in best],
                       "decision": {label: {"token_id": tokens[0], "sequence_ids": tokens,
                                             "logit": float(raw[tokens[0]]),
                                             "log_probability": float(raw[tokens[0]] - z)}
                                    for label, tokens in choices.items()}}
                d = row["decision"]
                row["logit_gaps"] = {
                    "call_minus_no_tool": d["<tool_call>"]["logit"] - d["NO_TOOL"]["logit"],
                    "call_minus_need_clarification": d["<tool_call>"]["logit"] - d["NEED_CLARIFICATION"]["logit"],
                    "need_clarification_minus_no_tool": d["NEED_CLARIFICATION"]["logit"] - d["NO_TOOL"]["logit"],
                }
                collected["mlp_act"].append(np.stack([sink[("mlp_act", i)] for i in layers])
                                              if layers else np.empty((0, 3584), np.float32))
                if hidden_states:
                    collected["hidden_states"].append(np.stack([sink[("hidden_states", i)] for i in range(25)]))
                if attention:
                    collected["attn_layers"].append(np.stack([sink[("attn_layers", i)] for i in FULL_ATTENTION]))
            row["seconds"] = time.perf_counter() - start
            row["memory"] = _memory(runtime.mx)
            runtime.budget.finish(reservation, len(ids), 0, seconds=row["seconds"], memory=row["memory"])
            rows.append(row)
        finally:
            del cache
            runtime.mx.clear_cache()
        del state, selected, logits, vector, logz, raw, sink
    result = {"rows": rows,
            "memory": {key: max(row["memory"][key] for row in rows)
                       for key in ("rss_bytes", "backend_peak_bytes", "backend_active_bytes", "backend_cache_bytes")},
            "metadata": {"layers": list(layers), "residual_layers": list(range(25)) if hidden_states else [],
                         "attention_layers": list(FULL_ATTENTION) if attention else [],
                         "position_policy": "last prompt or teacher-forced continuation token",
                         "observer_only": True, "chunk_size": chunk_size,
                         "vocabulary_projection": "selected final position only"}}
    return result, {k: np.stack(v) for k, v in collected.items()}


def save_diagnostics(stem, result):
    """Write compact scalar JSON and selected-position CPU arrays separately."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    scalars, arrays = result
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(scalars, indent=2) + "\n")
    return str(stem.with_suffix(".json")), str(stem.with_suffix(".npz"))
