"""Minimal client for an OpenAI-compatible /chat/completions endpoint.

Non-streaming, stdlib only. Returns (content, meta) where meta carries the
raw tool_calls (arguments left as the model's string: parsing them, and
failing to, is the runner's business), usage and finish_reason."""

import json
import time
import urllib.error
import urllib.request


def chat(base, model, messages, tools=None, api_key=None, max_tokens=4000,
         temperature=0.0, timeout=300, extra_body=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
    body.update(extra_body or {})
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code} from {base}: {raw.strip()[:400]}") \
            from None
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {})
        tool_calls.append({"id": tc.get("id"), "type": "function",
                           "function": {"name": fn.get("name"),
                                        "arguments": args}})
    meta = {"tool_calls": tool_calls,
            "usage": data.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
            "served_model": data.get("model"),
            "elapsed_s": round(time.time() - t0, 3)}
    return msg.get("content") or "", meta
