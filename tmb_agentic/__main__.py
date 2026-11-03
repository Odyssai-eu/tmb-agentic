"""Command line: python -m tmb_agentic {ui,gate,run,selftest} ..."""

import argparse
import json
import os
import sys
from functools import partial

from . import __version__, store
from .client import chat
from .runner import EXAMPLES, gate, load_tasks, run_many, selftest


def _chat_fn(args):
    extra = json.loads(args.extra_body) if args.extra_body else None
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    call = partial(chat, args.base, args.model, api_key=api_key,
                   max_tokens=args.max_tokens, temperature=args.temperature,
                   timeout=args.timeout, extra_body=extra)
    return lambda messages, tools: call(messages, tools=tools)


def cmd_ui(args):
    from .server import serve
    serve(store.home(args.home), args.tasks, port=args.port,
          open_browser=not args.no_browser)
    return 0


def cmd_gate(args):
    g = gate(_chat_fn(args))
    print(json.dumps(g, indent=2))
    return 0 if g["agent"] else 1


def cmd_run(args):
    tasks = load_tasks(args.tasks)
    d = store.new_run(store.home(args.home), args.model, args.base,
                      [t["id"] for t in tasks], args.reps, args.concurrency,
                      args.adapter, out=args.out)

    def save(task, rep, traj):
        store.record(d, task["id"], rep, traj)
        print(f"{task['id']} r{rep}: {'PASS' if traj['passed'] else 'FAIL'}"
              f" outcome={traj['outcome']}"
              + (f" flags={traj['flags']}" if traj["flags"] else ""), flush=True)

    status = "error"
    try:
        run_many(_chat_fn(args), tasks, reps=args.reps,
                 concurrency=args.concurrency, adapter=args.adapter,
                 on_done=save)
        status = "done"
    except KeyboardInterrupt:
        status = "stopped"
    finally:
        summary = store.finish(d, status)
    for tid, r in summary["results"].items():
        print(f"{tid}: {r['passed']}/{r['runs']} passed {r['outcomes']}"
              + (f" flags={r['flags']}" if r["flags"] else ""))
    print(f"{summary['passed']}/{summary['total']} runs passed -> {d}")
    return 0


def cmd_selftest(args):
    failed = 0
    tasks = load_tasks(args.tasks)
    for task in tasks:
        ok, detail = selftest(task)
        print(f"{task['id']}: {'OK' if ok else 'BROKEN'} ({detail})")
        failed += 0 if ok else 1
    print(f"{len(tasks) - failed}/{len(tasks)} tasks valid")
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="tmb-agentic")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def home(sp):
        sp.add_argument("--home", default=None,
                        help="data folder (default $TMB_AGENTIC_HOME or ~/.tmb-agentic)")

    def tasks(sp):
        sp.add_argument("tasks", nargs="?", default=str(EXAMPLES),
                        help="task file or folder (default: the bundled examples)")

    def endpoint(sp):
        sp.add_argument("--base", required=True,
                        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
        sp.add_argument("--model", required=True)
        sp.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="environment variable holding the API key")
        sp.add_argument("--temperature", type=float, default=0.0)
        sp.add_argument("--max-tokens", type=int, default=4000,
                        help="max tokens per turn")
        sp.add_argument("--timeout", type=int, default=300,
                        help="seconds per request")
        sp.add_argument("--extra-body", default=None,
                        help="JSON merged into each request body")

    sp = sub.add_parser("ui", help="open the web interface")
    sp.add_argument("--tasks", default=str(EXAMPLES),
                    help="task folder shown at start (default: the bundled examples)")
    sp.add_argument("--port", type=int, default=8790)
    sp.add_argument("--no-browser", action="store_true")
    home(sp)
    sp.set_defaults(func=cmd_ui)

    sp = sub.add_parser("gate", help="check that the endpoint emits usable tool calls")
    endpoint(sp)
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("run", help="run a model on tasks")
    tasks(sp)
    sp.add_argument("--out", default=None,
                    help="write the run here instead of the data folder")
    sp.add_argument("--adapter", action="store_true",
                    help="parse tool calls from text (gate verdict 'adapter')")
    sp.add_argument("--reps", type=int, default=1, help="runs per task")
    sp.add_argument("--concurrency", type=int, default=1,
                    help="trajectories run in parallel")
    home(sp)
    endpoint(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("selftest", help="replay each task's golden trajectory")
    tasks(sp)
    sp.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
