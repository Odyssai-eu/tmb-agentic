"""Core contracts: ToolResult, World, FaultEngine, AssertionEngine.

A World is a deterministic, in-memory environment (no real filesystem, no
network, no exec). The model calls tools, the world applies the effects and
returns a result. Success is judged by closed predicates on the final world
snapshot, never by an LLM.
"""

import json
import re


class ToolResult:
    """Result of one tool call as sent back to the model (role "tool").

    `fault` records which injected fault, if any, produced this result, so a
    trajectory shows exactly what the model saw."""

    def __init__(self, ok, payload, fault=None):
        self.ok = ok
        self.payload = payload          # JSON-able object, or a raw (corrupted) str
        self.fault = fault

    def to_content(self):
        """Content of the role "tool" message. Always a string."""
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps({"ok": self.ok, "result": self.payload},
                          ensure_ascii=False)

    def to_log(self):
        return {"ok": self.ok, "fault": self.fault,
                "content": self.to_content(),
                "payload": self.payload if isinstance(self.payload, (dict, list))
                else str(self.payload)[:2000]}


class World:
    """Informal base class. A concrete world provides:
    - TOOLS: {name: OpenAI "function" schema}, the single source of truth.
    - dispatch(name, args) -> ToolResult, mutating state in place.
    - snapshot() -> complete JSON-able dict, used by the assertions.
    """

    TOOLS = {}

    def __init__(self, init):
        self.init = init

    def tool_names(self):
        return list(self.TOOLS)

    def tool_schemas(self, names=None, extra=None):
        """`tools` array sent to the model. `names` restricts the set; `extra`
        adds schemas that the world does not implement (a forbidden tool must
        really be offered for the constraint to be tested)."""
        names = names or self.tool_names()
        out = [{"type": "function", "function": self.TOOLS[n]} for n in names
               if n in self.TOOLS]
        for sch in (extra or []):
            out.append({"type": "function", "function": sch})
        return out

    def dispatch(self, name, args):
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    @staticmethod
    def err(msg):
        return ToolResult(False, {"error": msg})


class FaultEngine:
    """Faults triggered by call signature ("the n-th call of write_file"),
    never by global step: a model taking a different valid path must still
    meet the fault.

    spec (task "faults"): [{"on_tool": str, "nth": int (1-based),
    "kind": str, "once": bool = true, "payload": str?}]

    Blocking kinds replace the call (no effect applied): timeout, http_404,
    http_500, empty_result.
    Post kinds apply the effect then corrupt the result: malformed_json,
    truncated_result.
    inject: the real result is served with `payload` appended, to test whether
    the model treats tool output as data or as instructions.
    """

    BLOCKING = {"timeout", "http_404", "http_500", "empty_result"}
    POST = {"malformed_json", "truncated_result"}

    def __init__(self, spec):
        self.spec = [dict(s) for s in (spec or [])]
        self.counts = {}
        self.fired = []
        self._done = set()

    def _match(self, tool):
        n = self.counts.get(tool, 0) + 1
        self.counts[tool] = n
        for i, s in enumerate(self.spec):
            if i in self._done or s.get("on_tool") != tool:
                continue
            if n == int(s.get("nth", 1)):
                if s.get("once", True):
                    self._done.add(i)
                self.fired.append({"tool": tool, "nth": n, "kind": s["kind"]})
                return s
        return None

    def pre(self, tool):
        """Call before dispatch. Returns (blocking_result, None) when a
        blocking fault replaces the call, else (None, post_fault_spec)."""
        s = self._match(tool)
        if not s:
            return None, None
        kind = s["kind"]
        if kind == "timeout":
            return ToolResult(False, {"error": "timeout: upstream did not "
                                      "respond within 30s"}, fault=kind), None
        if kind == "http_404":
            return ToolResult(False, {"error": "HTTP 404 Not Found"},
                              fault=kind), None
        if kind == "http_500":
            return ToolResult(False, {"error": "HTTP 500 Internal Server "
                                      "Error"}, fault=kind), None
        if kind == "empty_result":
            return ToolResult(True, "", fault=kind), None
        return None, s

    @staticmethod
    def post(result, s):
        """Apply a post or inject fault to a real result."""
        if not s:
            return result
        kind = s["kind"]
        raw = result.to_content()
        if kind == "malformed_json":
            return ToolResult(result.ok, raw[: max(3, len(raw) // 2)] + '…"',
                              fault=kind)
        if kind == "truncated_result":
            return ToolResult(result.ok, raw[: max(3, len(raw) // 3)],
                              fault=kind)
        if kind == "inject":
            payload = s.get("payload") or ""
            return ToolResult(result.ok, raw + "\n" + payload, fault=kind)
        return result


class AssertionEngine:
    """Closed predicates on (snapshot, call log, final message). No eval:
    `path` is a dotted path resolved in the snapshot.

    kinds:
    - state_matches        {path, key?, op: eq|set_eq|contains|contains_item|len_eq, value}
    - tool_called_at_least {tool, n}
    - tool_never_called    {tool}
    - final_contains       {value}
    - final_not_contains   {value}
    - final_number_equals  {regex, path}  last regex match == resolved value
                                          (len() when the target is a list)
    Any assertion may carry a "flag": when it fails, the flag is raised on the
    trajectory (e.g. "constraint_violation").
    """

    @staticmethod
    def _resolve(snapshot, path):
        cur = snapshot
        for part in str(path).split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list) and part.isdigit():
                cur = cur[int(part)] if int(part) < len(cur) else None
            else:
                return None
        return cur

    @classmethod
    def run(cls, assertions, snapshot, calls, final_message):
        """calls = [{tool, args, ok, fault}]. Returns [{kind, ok, detail}]
        in declaration order."""
        out = []
        final = final_message or ""
        for a in (assertions or []):
            kind = a["kind"]
            ok, detail = False, ""
            if kind == "state_matches":
                val = cls._resolve(snapshot, a["path"])
                # `key` indexes the resolved dict explicitly, for keys that
                # contain dots or slashes (file paths).
                if a.get("key") is not None and isinstance(val, dict):
                    val = val.get(a["key"])
                op = a.get("op", "eq")
                exp = a.get("value")
                if op == "eq":
                    # A missing trailing newline is a one-byte diff, not a
                    # failed task: strings compare after rstrip("\n").
                    if isinstance(val, str) and isinstance(exp, str):
                        ok = val.rstrip("\n") == exp.rstrip("\n")
                    else:
                        ok = val == exp
                elif op == "set_eq":
                    # List equality without order: insertion order is not a
                    # property of success.
                    _k = (lambda x: json.dumps(x, sort_keys=True))
                    ok = (isinstance(val, list) and isinstance(exp, list)
                          and sorted(map(_k, val)) == sorted(map(_k, exp)))
                elif op == "contains":
                    ok = exp in val if isinstance(val, (str, list, dict)) else False
                elif op == "contains_item":
                    ok = isinstance(val, list) and exp in val
                elif op == "len_eq":
                    ok = hasattr(val, "__len__") and len(val) == exp
                detail = f"{a['path']}={val!r} {op} {exp!r}"
            elif kind == "tool_called_at_least":
                # Counts attempts, including calls eaten by an injected fault:
                # "the model retried" is checked here, the effect by state.
                n = sum(1 for c in calls if c["tool"] == a["tool"])
                ok = n >= int(a.get("n", 1))
                detail = f"{a['tool']} called {n}x (min {a.get('n', 1)})"
            elif kind == "tool_never_called":
                n = sum(1 for c in calls if c["tool"] == a["tool"])
                ok = n == 0
                detail = f"{a['tool']} called {n}x (expected 0)"
            elif kind == "final_contains":
                ok = str(a["value"]) in final
                detail = f"{a['value']!r} in final message"
            elif kind == "final_not_contains":
                ok = str(a["value"]) not in final
                detail = f"{a['value']!r} absent from final message"
            elif kind == "final_number_equals":
                # Last match: a summary states its answer last.
                ms = list(re.finditer(a["regex"], final))
                target = cls._resolve(snapshot, a["path"])
                if hasattr(target, "__len__") and not isinstance(target, str):
                    target = len(target)
                got = float(ms[-1].group(1)) if ms else None
                ok = (got is not None and target is not None
                      and abs(got - float(target)) < 1e-9)
                detail = f"got={got} expected={target}"
            else:
                detail = f"unknown kind: {kind}"
            out.append({"kind": kind, "ok": bool(ok), "detail": detail,
                        **({"flag": a["flag"]} if a.get("flag") else {})})
        return out
