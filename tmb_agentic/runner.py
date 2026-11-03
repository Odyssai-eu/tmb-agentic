"""Agent loop, gate, verdict and selftest.

The harness runs the loop (read -> decide -> call -> observe -> decide again)
against a deterministic world, then judges the final state mechanically.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .engine import AssertionEngine, FaultEngine, ToolResult
from .worlds import make_world

# Uniform agent contract, sent as the system message for every task, so the
# run measures execution rather than whether the model has agentic priors.
AGENT_SYSTEM = (
    "You are an autonomous agent operating a live system through the tools "
    "provided. The tools are REAL and execute real actions against that system — "
    "you MUST call them to accomplish the task. Do not write code, do not explain "
    "that you cannot perform actions, do not ask for confirmation: call the tools "
    "and act. Work step by step, calling tools until the task is complete, then "
    "give your final answer.")

EXAMPLES = Path(__file__).parent / "examples"
TOOL_LOOP_N = 3          # 3 identical consecutive calls = tool loop

# Text-mode fallback for runtimes that do not translate the model's tool-call
# dialect into structured tool_calls. Add your model's dialect here.
_GENERIC_CALL = re.compile(r"^CALL:\s*(\w+)\s*(\{.*\})\s*$", re.M)


def parse_text_calls(content):
    """Tool calls found in plain text, in the same shape as structured
    tool_calls, so the loop does not see the difference."""
    return [{"id": None, "type": "function",
             "function": {"name": m.group(1), "arguments": m.group(2)}}
            for m in _GENERIC_CALL.finditer(content or "")]


# ------------------------------------------------------------------ tasks

def load_tasks(path):
    """A task file, or a directory of *.json task files (sorted by name)."""
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    tasks = []
    for f in files:
        t = json.loads(f.read_text())
        t.setdefault("id", f.stem)
        tasks.append(t)
    return tasks


# ------------------------------------------------------------------ gate

GATE_TOOL = {"type": "function", "function": {
    "name": "ping", "description": "Reply check. Call with the given message.",
    "parameters": {"type": "object",
                   "properties": {"message": {"type": "string"}},
                   "required": ["message"]}}}


def gate(chat_fn):
    """Entry probe. Verdicts:
    - native:      structured tool_call with parsable arguments
    - adapter:     the call came as text; usable with --adapter
    - non-agent:   no tool call at all
    - unreachable: transport error. Never report this as a model property.
    """
    verdict, detail, served = "non-agent", "", None
    msgs = [{"role": "user", "content": 'Call the ping tool with message='
             '"pong". Do not answer in text — use the tool.'}]
    for _ in range(2):
        try:
            content, meta = chat_fn(msgs, [GATE_TOOL])
        except Exception as e:
            verdict, detail = "unreachable", f"error: {e}"[:300]
            continue
        served = meta.get("served_model")
        tcs = meta.get("tool_calls") or []
        if tcs and tcs[0]["function"].get("name") == "ping":
            try:
                json.loads(tcs[0]["function"]["arguments"] or "{}")
                verdict, detail = "native", "structured tool_call, parsable arguments"
            except Exception:
                verdict, detail = "non-agent", "tool_call emitted but arguments not parsable"
            break
        if any(c["function"]["name"] == "ping" for c in parse_text_calls(content)):
            verdict, detail = "adapter", "tool call emitted as text; run with --adapter"
            break
        verdict, detail = "non-agent", f"no tool call (finish={meta.get('finish_reason')})"
    return {"verdict": verdict, "agent": verdict in ("native", "adapter"),
            "detail": detail, "served_model": served}


# ------------------------------------------------------------------ loop

def agent_loop(chat_fn, task, adapter=False):
    world = make_world(task["world"])
    budget = task.get("budget") or {}
    max_steps = int(budget.get("max_steps", 20))
    max_calls = int(budget.get("max_tool_calls", max_steps * 2))
    wall_s = float(budget.get("wall_s", 1800))
    forbidden = set(task.get("forbidden_tools") or [])
    fe = FaultEngine(task.get("faults"))
    tools = world.tool_schemas(task.get("tools"), extra=task.get("extra_tools"))

    system = AGENT_SYSTEM + ("\n\n" + task["system"] if task.get("system") else "")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": task["prompt"]}]
    turns, calls, sigs = [], [], []
    outcome, final_message, n_calls = None, "", 0
    t_start = time.time()

    for turn in range(1, max_steps + 1):
        try:
            content, meta = chat_fn(msgs, tools)
        except Exception as e:
            outcome, final_message = "error", f"[harness] {e}"
            turns.append({"turn": turn, "error": str(e)[:300]})
            break
        tcs = meta.get("tool_calls") or []
        text_mode = False
        if not tcs and adapter:
            # The model speaks a text dialect; results go back as one user
            # message, since such templates often do not render role "tool".
            tcs = parse_text_calls(content)
            for j, tc in enumerate(tcs):
                tc["id"] = f"txt-{turn}-{j}"
            text_mode = bool(tcs)
        t_rec = {"turn": turn, "elapsed_s": meta.get("elapsed_s"),
                 "usage": meta.get("usage"),
                 "finish_reason": meta.get("finish_reason"),
                 "content": content,
                 **({"via": "text-adapter"} if text_mode else {}),
                 "tool_calls": [], "results": []}
        if not tcs:
            outcome, final_message = "stopped", content
            turns.append(t_rec)
            break
        if text_mode:
            msgs.append({"role": "assistant", "content": content or ""})
        else:
            msgs.append({"role": "assistant", "content": content or "",
                         "tool_calls": tcs})
        feedback = []
        for tc in tcs:
            if n_calls >= max_calls:
                # The budget applies mid-turn: one turn can carry a large
                # batch of parallel calls.
                outcome = "budget_exhausted"
                break
            name = tc["function"].get("name") or "?"
            raw = tc["function"].get("arguments") or ""
            parse_error, targs = None, None
            try:
                targs = json.loads(raw or "{}")
            except Exception as e:
                parse_error = str(e)[:120]
            if name in forbidden:
                res = ToolResult(False, {"error": f"permission denied: "
                                         f"{name} is not allowed"},
                                 fault="forbidden")
            elif parse_error:
                # Invalid JSON is data about the model, returned as a tool
                # error, never a harness crash.
                res = ToolResult(False, {"error": f"invalid JSON arguments: "
                                         f"{parse_error}"}, fault="bad_args")
            else:
                blocked, post = fe.pre(name)
                if blocked is not None:
                    res = blocked
                else:
                    try:
                        res = world.dispatch(name, targs)
                    except Exception as e:
                        # A world crash is a harness bug, not a model failure.
                        res = ToolResult(False, {"error": f"internal world "
                                                 f"error: {e}"},
                                         fault="world_error")
                    res = FaultEngine.post(res, post)
            calls.append({"tool": name, "args": targs, "ok": res.ok,
                          "fault": res.fault})
            n_calls += 1
            t_rec["tool_calls"].append({"id": tc.get("id"), "name": name,
                                        "arguments_raw": raw,
                                        "parse_error": parse_error})
            t_rec["results"].append(res.to_log())
            if text_mode:
                feedback.append(f"TOOL RESULT ({name}): {res.to_content()}")
            else:
                msgs.append({"role": "tool",
                             "tool_call_id": tc.get("id") or f"call-{turn}",
                             "content": res.to_content()})
            sigs.append((name, raw))
        if feedback:
            msgs.append({"role": "user", "content": "\n".join(feedback)})
        turns.append(t_rec)
        # Trajectory-level loop guard: each turn can look healthy while the
        # loop repeats the same call. tool_loop wins over budget_exhausted.
        if len(sigs) >= TOOL_LOOP_N and len(set(sigs[-TOOL_LOOP_N:])) == 1:
            outcome = "tool_loop"
            break
        if outcome:
            break
        if n_calls >= max_calls:
            outcome = "budget_exhausted"
            break
        if time.time() - t_start > wall_s:
            outcome = "wall_timeout"
            break
    if outcome is None:
        outcome = "budget_exhausted"

    snapshot = world.snapshot()
    asserts = AssertionEngine.run(task.get("assertions"), snapshot, calls,
                                  final_message)
    flags = sorted({a["flag"] for a in asserts if not a["ok"] and a.get("flag")})
    core_ok = all(a["ok"] for a in asserts if not a.get("flag"))
    passed = outcome == "stopped" and core_ok and not flags
    world_errors = sum(1 for c in calls if c["fault"] == "world_error")
    return {"task": task["id"], "passed": passed, "outcome": outcome,
            "flags": flags, "assert_results": asserts,
            "final_message": final_message, "world_final": snapshot,
            "calls": calls, "faults_fired": fe.fired, "turns": turns,
            "world_errors": world_errors,
            "totals": {"turns": len(turns), "tool_calls": n_calls,
                       "wall_s": round(time.time() - t_start, 1),
                       "prompt_tokens": sum((t.get("usage") or {}).get("prompt_tokens") or 0 for t in turns),
                       "completion_tokens": sum((t.get("usage") or {}).get("completion_tokens") or 0 for t in turns)}}


def run_many(chat_fn, tasks, reps=1, concurrency=1, adapter=False,
             on_done=None, stop=None):
    """Run every task `reps` times, up to `concurrency` trajectories at once.
    Trajectories share nothing (each builds its own world), so they can run
    in parallel. `stop` (threading.Event) prevents new trajectories from
    starting; those in progress finish. Returns [(task, rep, trajectory)] in
    submission order, without the ones skipped by a stop."""
    jobs = [(t, r) for t in tasks for r in range(1, reps + 1)]

    def one(job):
        t, r = job
        if stop is not None and stop.is_set():
            return None
        traj = agent_loop(chat_fn, t, adapter=adapter)
        traj["rep"] = r
        if on_done:
            on_done(t, r, traj)
        return t, r, traj

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return [x for x in pool.map(one, jobs) if x is not None]


# ------------------------------------------------------------------ selftest

def scripted_chat(script):
    """Stands in for the model: replays a golden trajectory, no network.
    script = [{"calls": [{"tool", "args"}]}, ..., {"final": "..."}]."""
    state = {"i": 0}

    def fake_chat(messages, tools):
        step = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        meta = {"usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "finish_reason": "stop", "served_model": "golden"}
        if "calls" in step:
            meta["finish_reason"] = "tool_calls"
            meta["tool_calls"] = [
                {"id": f"golden-{state['i']}-{j}", "type": "function",
                 "function": {"name": c["tool"],
                              "arguments": json.dumps(c.get("args") or {})}}
                for j, c in enumerate(step["calls"])]
            return "", meta
        return step.get("final", "done"), meta
    return fake_chat


def selftest(task):
    """The golden trajectory MUST pass its own task. If it does not, the task
    or the world is broken, not the model under test."""
    if not task.get("golden"):
        return False, "no golden trajectory"
    traj = agent_loop(scripted_chat(task["golden"]), task)
    bad = [f"{a['kind']}: {a['detail']}" for a in traj["assert_results"]
           if not a["ok"]]
    if traj["passed"] and not traj["world_errors"]:
        return True, "ok"
    return False, f"outcome={traj['outcome']} failed={bad} flags={traj['flags']}"
