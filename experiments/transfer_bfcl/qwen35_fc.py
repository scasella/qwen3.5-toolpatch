"""BFCL handler for Qwen3.5 in native function-calling mode (thinking off).

Differences from the Qwen3 FC handler (`qwen_fc.py`):

* The prompt is built by the checkpoint's own chat template
  (`tokenizer.apply_chat_template(messages, tools=..., enable_thinking=False)`),
  exactly as the qwen-6/qwen-7 edit campaigns did, instead of a hand-written
  Qwen3 template.  Qwen3.5's template renders tools inside the system message
  and asks for calls in the XML form
  `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`
  (the Qwen3-Coder format), not Qwen3's JSON form.
* The raw text comes back from a raw-text completion server (see `serve.py`).
  The XML calls are parsed here and the parameter values are coerced with the
  tool schema the model saw (same rules as vLLM's `qwen3coder` tool parser,
  which is how Qwen3-Coder models are scored on the public leaderboard).  The
  canonical JSON form `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`
  is stored as the model response so `decode_ast` is the Qwen3 one; the raw text
  is kept verbatim in the result record (`raw_response`).
* Thinking is off: the template emits an empty `<think>\n\n</think>\n\n` block
  before the decision token, the position at which the weight edits were
  validated.
* `max_tokens` is capped at 512.
* Registry names containing `-campaign-prompt` prepend the campaign system
  prompt ("Use the available tools when needed. ... reply exactly NO_TOOL ...
  NEED_CLARIFICATION ...") as the first system message (merged in front of a
  dataset system message when the item has one).
"""
import ast
import json
import re
import threading
import time
from types import SimpleNamespace
from typing import Any

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
from overrides import override

CAMPAIGN_SYSTEM_PROMPT = (
    "Use the available tools when needed. For a request requiring no tool, reply exactly NO_TOOL. "
    "If required information is missing or ambiguous, reply exactly NEED_CLARIFICATION. "
    "Otherwise make exactly one native tool call. No other text."
)
MAX_NEW_TOKENS = 512

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _loose(v: str):
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return v


def coerce_value(raw: str, prop: dict | None):
    """Mirror vLLM's qwen3coder tool parser: type the XML string by the schema."""
    v = raw
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    if v.strip().lower() == "null":
        return None
    if prop is None:
        return _loose(v)
    t = str(prop.get("type", "string")).strip().lower()
    if t in ("string", "str", "text", "varchar", "char", "enum"):
        return v
    if t.startswith("int") or t in ("integer", "long"):
        try:
            return int(v)
        except ValueError:
            try:
                f = float(v)
                return int(f) if f.is_integer() else f
            except ValueError:
                return v
    if t in ("number", "float", "double"):
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v
    if t in ("boolean", "bool"):
        s = v.strip().lower()
        return s == "true" if s in ("true", "false") else v
    if t in ("object", "array", "dict", "list", "tuple"):
        return _loose(v)
    return _loose(v)


def parse_xml_tool_calls(text: str, schemas: dict[str, dict]) -> list[dict]:
    """Return [{"name": str, "arguments": dict}, ...] from Qwen3.5 XML tool calls."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = (schemas.get(name) or {}).get("properties", {})
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            pname = pm.group(1).strip()
            args[pname] = coerce_value(pm.group(2), props.get(pname) if pname in props else None)
        calls.append({"name": name, "arguments": args})
    return calls


def to_json_tool_call_text(calls: list[dict]) -> str:
    return "\n".join(
        "<tool_call>\n" + json.dumps(c, ensure_ascii=False) + "\n</tool_call>" for c in calls
    )


class Qwen35FCHandler(OSSHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.use_campaign_prompt = "campaign-prompt" in registry_name
        # Qwen3.5-0.8B's config nests max_position_embeddings under text_config; the
        # base class only looks at the top level, so give it a fallback.
        self.max_context_length = 32768
        self._tls = threading.local()

    # ---- prompt -----------------------------------------------------------
    def _tools(self, function: list[dict]) -> list[dict]:
        # convert_to_tool: '.' -> '_' in names (underscore_to_dot=True in the config),
        # BFCL types -> OpenAPI types ("dict" -> "object", "float" -> "number", ...)
        return [{"type": "function", "function": f} for f in convert_to_tool(function, GORILLA_TO_OPENAPI, self.model_style)]

    @override
    def _format_prompt(self, messages, function):
        tools = self._tools(function)
        messages = [dict(m) for m in messages]
        if self.use_campaign_prompt:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = CAMPAIGN_SYSTEM_PROMPT + "\n\n" + str(messages[0]["content"])
            else:
                messages.insert(0, {"role": "system", "content": CAMPAIGN_SYSTEM_PROMPT})
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Native FC: the template carries the tools; no BFCL prompting system prompt.
        return {"message": [], "function": test_entry["function"]}

    # ---- query ------------------------------------------------------------
    @override
    def _query_prompting(self, inference_data: dict):
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]
        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in self._tools(function)}
        start = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=MAX_NEW_TOKENS,
            timeout=72000,
        )
        latency = time.time() - start
        return SimpleNamespace(response=api_response, schemas=schemas), latency

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        resp = api_response.response
        raw = resp.choices[0].text
        calls = parse_xml_tool_calls(raw, api_response.schemas)
        self._tls.extra = {
            "raw_response": raw,
            "finish_reason": resp.choices[0].finish_reason,
            "num_tool_calls": len(calls),
        }
        return {
            "model_responses": to_json_tool_call_text(calls) if calls else raw,
            "raw_response": raw,
            "finish_reason": resp.choices[0].finish_reason,
            "num_tool_calls": len(calls),
            "input_token": resp.usage.prompt_tokens,
            "output_token": resp.usage.completion_tokens,
        }

    @override
    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        # `inference_single_turn_prompting` is @final in OSSHandler and only keeps token
        # counts; attach the raw text (verbatim generation) to the result record here.
        self._tls.extra = None
        result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
        extra = getattr(self._tls, "extra", None)
        if extra:
            metadata.update(extra)
        return result, metadata

    # ---- decode (same JSON form as qwen_fc.py) -----------------------------
    @staticmethod
    def _extract_tool_calls(input_string):
        result = []
        for match in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", input_string, re.DOTALL):
            try:
                result.append(json.loads(match))
            except Exception:
                pass
        return result

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or len(tool_calls) == 0 or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{call["name"]: {k: v for k, v in call["arguments"].items()}} for call in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call([{item["name"]: item["arguments"]} for item in tool_calls])
