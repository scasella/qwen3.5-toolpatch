#!/usr/bin/env python
"""Minimal OpenAI-compatible server over mlx_lm that returns the model's RAW text.

Why not `mlx_lm.server`?  Its text state machine strips every `<tool_call>…</tool_call>`
span out of `choices[0].text` into a separate `tool_calls` field, even for
`/v1/completions`, so BFCL's text-based Qwen parser would never see the call.
This shim does what the campaign harness did: tokenize the prompt the client
formatted (BFCL applies the chat template itself), decode greedily, return the
detokenized text untouched (EOS `<|im_end|>` excluded), with usage counts.

Endpoints
  GET  /v1/models            readiness probe used by BFCL
  GET  /health
  POST /v1/completions       {"prompt": str, "max_tokens": int}  -> raw text
  POST /v1/chat/completions  {"messages", "tools", "chat_template_kwargs"} -> raw text
                             (convenience; enable_thinking=False unless overridden)

Decoding is always greedy (temperature in the request is logged, not used).
`max_tokens` is capped at --max-tokens (default 512).  Requests are served one
at a time behind a lock (the model is small, the GPU is one).

Run with the only Python that has a working mlx_lm on this machine:
  P=/Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python
  $P -I serve.py --model <checkpoint dir> --port 1053
"""
import argparse
import fcntl
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

LOCKS = ["/tmp/qwen-local-accelerator.lock", "/tmp/qwen35-mlx-accelerator.lock"]


def acquire_locks(paths, poll=5.0):
    """Shared accelerator-lock etiquette of the campaign workspaces: poll, never kill."""
    handles = []
    for p in paths:
        fh = open(p, "a+")
        waited = 0.0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited == 0:
                    logging.info("waiting for accelerator lock %s", p)
                time.sleep(poll)
                waited += poll
        handles.append(fh)
    return handles


class Engine:
    def __init__(self, model_path, max_tokens):
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        mx.eval(self.model.parameters())
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.sampler = make_sampler(temp=0.0)
        self.lock = threading.Lock()
        self.n = 0
        self.t_gen = 0.0
        logging.info("loaded %s in %.1fs", model_path, time.time() - t0)

    def complete(self, prompt_ids, max_tokens):
        max_tokens = min(int(max_tokens or self.max_tokens), self.max_tokens)
        text, n_out, finish = [], 0, "length"
        with self.lock:
            t0 = time.time()
            for r in stream_generate(
                self.model, self.tokenizer, prompt_ids, max_tokens=max_tokens, sampler=self.sampler
            ):
                text.append(r.text)
                n_out = r.generation_tokens
                if r.finish_reason is not None:
                    finish = r.finish_reason
            dt = time.time() - t0
            self.n += 1
            self.t_gen += dt
        out = "".join(text)
        logging.info(
            "req %d: prompt %d tok, out %d tok, %.2fs (%s) | avg %.2fs",
            self.n, len(prompt_ids), n_out, dt, finish, self.t_gen / self.n,
        )
        return out, len(prompt_ids), n_out, finish


ENGINE: Engine = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet access log; Engine logs per request
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": ENGINE.model_path, "object": "model", "created": 0, "owned_by": "mlx"}]})
        elif self.path == "/health":
            self._json(200, {"status": "ok", "requests": ENGINE.n})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        tok = ENGINE.tokenizer
        try:
            if self.path == "/v1/completions":
                prompt = body["prompt"]
                if isinstance(prompt, list):
                    prompt = prompt[0]
                ids = tok.encode(prompt, add_special_tokens=False)
                obj_type, key = "text_completion", "text"
            elif self.path == "/v1/chat/completions":
                kw = {"enable_thinking": False}
                kw.update(body.get("chat_template_kwargs") or {})
                ids = tok.apply_chat_template(
                    body["messages"], tools=body.get("tools") or None, tokenize=True, add_generation_prompt=True, **kw
                )
                obj_type, key = "chat.completion", "message"
            else:
                return self._json(404, {"error": "not found"})
        except Exception as e:  # template errors etc.
            return self._json(400, {"error": f"{type(e).__name__}: {e}"})
        if body.get("temperature") not in (None, 0, 0.0) and ENGINE.n == 0:
            logging.info("note: temperature=%s requested; decoding is greedy regardless", body.get("temperature"))
        text, n_in, n_out, finish = ENGINE.complete(ids, body.get("max_tokens") or body.get("max_completion_tokens"))
        choice = {"index": 0, "finish_reason": finish, "logprobs": None}
        choice[key] = text if key == "text" else {"role": "assistant", "content": text}
        self._json(200, {
            "id": f"cmpl-{uuid.uuid4()}", "object": obj_type, "created": int(time.time()), "model": ENGINE.model_path,
            "choices": [choice],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=1053)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-lock", action="store_true", help="do not take the shared accelerator locks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    locks = [] if args.no_lock else acquire_locks(LOCKS)
    global ENGINE
    ENGINE = Engine(args.model, args.max_tokens)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    logging.info("serving %s on http://%s:%d/v1 (pid %d)", args.model, args.host, args.port, os.getpid())
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for fh in locks:
            fcntl.flock(fh, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
