"""Run store: one directory per run under <home>/runs/<run_id>/.

  meta.json       written at start, updated at the end (status, counts)
  progress.ndjson one line per finished trajectory (survives a crash)
  <task>.r<k>.json full trajectories
  summary.json    written at the end

The CLI and the UI write here the same way, so every run shows up in the
history. Home is $TMB_AGENTIC_HOME or ~/.tmb-agentic.
"""

import csv
import io
import json
import os
import re
import threading
import time
import zipfile
from pathlib import Path

from . import __version__

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_lock = threading.Lock()


def home(path=None):
    h = Path(path or os.environ.get("TMB_AGENTIC_HOME") or
             Path.home() / ".tmb-agentic").expanduser()
    (h / "runs").mkdir(parents=True, exist_ok=True)
    return h


def safe_name(s):
    """File-system safe token: task ids and model names come from users."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip(".-")
    return out[:80] or "x"


def traj_file(task_id, rep):
    return f"{safe_name(task_id)}.r{int(rep)}.json"


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def new_run(h, model, base, task_ids, reps, concurrency, adapter, out=None):
    """Create the run directory and its meta.json. `out` forces a directory
    (CLI --out); otherwise the run lives under <home>/runs/."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{safe_name(model)}"
    d = Path(out) if out else h / "runs" / run_id
    n = 2
    while not out and d.exists():
        d = h / "runs" / f"{run_id}-{n}"
        n += 1
    d.mkdir(parents=True, exist_ok=True)
    meta = {"id": d.name, "version": __version__, "model": model, "base": base,
            "tasks": list(task_ids), "reps": reps, "concurrency": concurrency,
            "adapter": adapter, "expected": len(task_ids) * reps,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": None,
            "status": "running", "pid": os.getpid()}
    _write_json(d / "meta.json", meta)
    return d


def record(d, task_id, rep, traj):
    """Save one finished trajectory and append its progress line."""
    _write_json(d / traj_file(task_id, rep), traj)
    line = {"task": task_id, "rep": rep, "passed": traj["passed"],
            "outcome": traj["outcome"], "flags": traj["flags"],
            "tool_calls": traj["totals"]["tool_calls"],
            "completion_tokens": traj["totals"]["completion_tokens"],
            "wall_s": traj["totals"]["wall_s"], "t": time.time()}
    with _lock, open(d / "progress.ndjson", "a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_progress(d):
    p = d / "progress.ndjson"
    if not p.exists():
        return []
    out = []
    for raw in p.read_text().splitlines():
        try:
            out.append(json.loads(raw))
        except ValueError:
            pass                      # a line cut by a crash
    return out


def summarize(progress):
    """Per-task counts: passed / runs / outcomes / flags."""
    res = {}
    for p in progress:
        r = res.setdefault(p["task"], {"passed": 0, "runs": 0,
                                       "outcomes": {}, "flags": {}})
        r["runs"] += 1
        r["passed"] += bool(p["passed"])
        r["outcomes"][p["outcome"]] = r["outcomes"].get(p["outcome"], 0) + 1
        for f in p["flags"]:
            r["flags"][f] = r["flags"].get(f, 0) + 1
    return res


def finish(d, status="done"):
    meta = json.loads((d / "meta.json").read_text())
    results = summarize(read_progress(d))
    summary = {"version": __version__, "model": meta["model"],
               "base": meta["base"], "reps": meta["reps"], "results": results,
               "passed": sum(r["passed"] for r in results.values()),
               "total": sum(r["runs"] for r in results.values())}
    _write_json(d / "summary.json", summary)
    meta.update(status=status, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_json(d / "meta.json", meta)
    return summary


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def run_info(d):
    """meta + live counts. A run still marked running whose process is gone
    is reported as incomplete."""
    meta = json.loads((d / "meta.json").read_text())
    prog = read_progress(d)
    meta["done"] = len(prog)
    meta["passed"] = sum(1 for p in prog if p["passed"])
    if meta["status"] == "running" and not _alive(meta.get("pid")):
        meta["status"] = "incomplete"
    if meta["status"] == "running" and prog:
        t0 = time.mktime(time.strptime(meta["created"], "%Y-%m-%dT%H:%M:%S"))
        per = (time.time() - t0) / len(prog)
        meta["eta_s"] = round(per * max(0, meta["expected"] - len(prog)))
    return meta


def list_runs(h):
    out = []
    for d in sorted((h / "runs").iterdir(), reverse=True):
        if (d / "meta.json").exists():
            try:
                out.append(run_info(d))
            except (ValueError, KeyError):
                pass
    return out


def run_dir(h, run_id):
    """Resolve a run id coming from a client. Never a free path."""
    if not run_id or not _RUN_ID.match(run_id):
        return None
    d = h / "runs" / run_id
    return d if (d / "meta.json").exists() else None


# ------------------------------------------------------------------ export

CSV_FIELDS = ["run_id", "created", "model", "base", "status", "task", "runs",
              "passed", "pass_rate", "outcomes", "flags"]


def export_rows(runs):
    rows = []
    for meta, results in runs:
        for task, r in sorted(results.items()):
            rows.append({
                "run_id": meta["id"], "created": meta["created"],
                "model": meta["model"], "base": meta["base"],
                "status": meta["status"], "task": task, "runs": r["runs"],
                "passed": r["passed"],
                "pass_rate": round(r["passed"] / r["runs"], 4) if r["runs"] else "",
                "outcomes": ";".join(f"{k}={v}" for k, v in sorted(r["outcomes"].items())),
                "flags": ";".join(f"{k}={v}" for k, v in sorted(r["flags"].items()))})
    return rows


def failures(d, limit=200):
    """Failed runs with their failed assertions, for the report."""
    out = []
    for p in read_progress(d):
        if p["passed"]:
            continue
        f = d / traj_file(p["task"], p["rep"])
        failed = []
        if f.exists():
            t = json.loads(f.read_text())
            failed = [a["detail"] for a in t.get("assert_results", [])
                      if not a["ok"]]
        out.append({"task": p["task"], "rep": p["rep"],
                    "outcome": p["outcome"], "flags": p["flags"],
                    "failed": failed})
        if len(out) >= limit:
            break
    return out


def export_zip(h, run_ids=None, lang="en"):
    """One mechanism for one run and for everything: a ZIP with the raw run
    directories, report.csv (one row per run x task) and report.html."""
    from .report import render_report
    ids = run_ids or [m["id"] for m in list_runs(h)]
    runs, dirs = [], []
    for rid in ids:
        d = run_dir(h, rid)
        if d is None:
            continue
        meta = run_info(d)
        runs.append((meta, summarize(read_progress(d))))
        dirs.append(d)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.is_file() and not f.name.endswith(".tmp"):
                    z.write(f, f"runs/{d.name}/{f.name}")
        s = io.StringIO()
        w = csv.DictWriter(s, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(export_rows(runs))
        z.writestr("report.csv", s.getvalue())
        z.writestr("report.html", render_report(
            [(m, r, failures(d)) for (m, r), d in zip(runs, dirs)], lang))
    return buf.getvalue(), len(runs)
