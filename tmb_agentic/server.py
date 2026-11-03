"""Local web UI: `python -m tmb_agentic ui`.

Standard-library HTTP server bound to 127.0.0.1. It serves one page and a
JSON API. Requests whose Host is not localhost are refused (DNS rebinding),
and every POST must carry the X-TMB-Agentic header, which a cross-site form
cannot send. One run at a time. API keys stay in this process's memory.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__, store
from .client import chat
from .runner import EXAMPLES, gate, load_tasks, run_many, selftest

UI_DIR = Path(__file__).parent / "ui"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class App:
    """Process-wide state shared by the request threads."""

    def __init__(self, home, tasks_dir):
        self.home = home
        self.tasks_dir = Path(tasks_dir)
        self.lock = threading.Lock()
        self.active = None            # {"id", "dir", "stop": Event}

    def chat_fn(self, ep):
        key = ep.get("api_key") or os.environ.get("OPENAI_API_KEY")
        extra = ep.get("extra_body") or None
        if extra is not None and not isinstance(extra, dict):
            raise ValueError("extra_body must be a JSON object")

        def fn(messages, tools):
            return chat(ep["base"], ep["model"], messages, tools=tools,
                        api_key=key,
                        max_tokens=int(ep.get("max_tokens") or 4000),
                        temperature=float(ep.get("temperature") or 0.0),
                        extra_body=extra)
        return fn

    def tasks(self):
        out = []
        for t in load_tasks(self.tasks_dir) if self.tasks_dir.is_dir() else []:
            ok, detail = selftest(t)
            out.append({"id": t["id"], "valid": ok, "detail": detail,
                        "world": t["world"]["type"], "task": t})
        return out

    def start_run(self, ep, task_ids, reps, concurrency, adapter):
        with self.lock:
            if self.active:
                return None, "busy"
            chosen = [t for t in load_tasks(self.tasks_dir)
                      if t["id"] in set(task_ids)]
            chosen = [t for t in chosen if selftest(t)[0]]
            if not chosen:
                return None, "no_tasks"
            chat_fn = self.chat_fn(ep)
            d = store.new_run(self.home, ep["model"], ep["base"],
                              [t["id"] for t in chosen], reps, concurrency,
                              adapter)
            stop = threading.Event()
            self.active = {"id": d.name, "dir": d, "stop": stop}

        def work():
            status = "done"
            try:
                run_many(chat_fn, chosen, reps=reps, concurrency=concurrency,
                         adapter=adapter, stop=stop,
                         on_done=lambda t, r, traj: store.record(d, t["id"], r, traj))
                if stop.is_set():
                    status = "stopped"
            except Exception:
                status = "error"
            finally:
                store.finish(d, status)
                with self.lock:
                    self.active = None
        threading.Thread(target=work, daemon=True).start()
        return d.name, None


def _list_models(base, api_key):
    req = urllib.request.Request(
        base.rstrip("/") + "/models",
        headers={"authorization": f"Bearer {api_key}"} if api_key else {})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))


def make_handler(app):

    class Handler(BaseHTTPRequestHandler):
        server_version = f"tmb-agentic/{__version__}"

        def log_message(self, fmt, *args):
            pass

        # -- plumbing ------------------------------------------------------
        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        # -- GET -----------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden host"})
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = [unquote(p) for p in u.path.split("/") if p]
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, (UI_DIR / "index.html").read_bytes(),
                                      "text/html; charset=utf-8")
                if u.path == "/i18n.json":
                    return self._send(200, (UI_DIR / "i18n.json").read_bytes())
                if u.path == "/api/state":
                    a = app.active
                    return self._send(200, {
                        "version": __version__, "home": str(app.home),
                        "tasks_dir": str(app.tasks_dir),
                        "active": a["id"] if a else None,
                        "stopping": bool(a and a["stop"].is_set())})
                if u.path == "/api/tasks":
                    return self._send(200, {"dir": str(app.tasks_dir),
                                            "tasks": app.tasks()})
                if u.path == "/api/runs":
                    return self._send(200, store.list_runs(app.home))
                if parts[:2] == ["api", "runs"] and len(parts) >= 3:
                    d = store.run_dir(app.home, parts[2])
                    if d is None:
                        return self._send(404, {"error": "no such run"})
                    if len(parts) == 3:
                        prog = store.read_progress(d)
                        return self._send(200, {
                            "meta": store.run_info(d), "progress": prog,
                            "results": store.summarize(prog)})
                    if len(parts) == 6 and parts[3] == "traj":
                        f = d / store.traj_file(parts[4], parts[5])
                        if not f.exists():
                            return self._send(404, {"error": "no such trajectory"})
                        return self._send(200, f.read_bytes())
                if u.path == "/api/export":
                    ids = [x for x in (q.get("runs") or [""])[0].split(",") if x]
                    lang = (q.get("lang") or ["en"])[0]
                    data, n = store.export_zip(app.home, ids or None, lang)
                    name = (f"tmb-agentic-{ids[0]}.zip" if len(ids) == 1
                            else f"tmb-agentic-export-{time.strftime('%Y%m%d-%H%M%S')}.zip")
                    return self._send(200, data, "application/zip", {
                        "Content-Disposition": f'attachment; filename="{name}"'})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)[:300]})

        # -- POST ----------------------------------------------------------
        def do_POST(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden host"})
            if self.headers.get("X-TMB-Agentic") != "1":
                return self._send(403, {"error": "missing X-TMB-Agentic header"})
            try:
                body = self._json_body()
                if self.path == "/api/models":
                    try:
                        key = body.get("api_key") or os.environ.get("OPENAI_API_KEY")
                        return self._send(200, {"models": _list_models(body["base"], key)})
                    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                        return self._send(200, {"models": [], "error": str(e)[:200]})
                if self.path == "/api/gate":
                    return self._send(200, gate(app.chat_fn(body)))
                if self.path == "/api/tasks":
                    app.tasks_dir = Path(body.get("dir") or app.tasks_dir).expanduser()
                    return self._send(200, {"dir": str(app.tasks_dir),
                                            "tasks": app.tasks()})
                if self.path == "/api/run":
                    rid, err = app.start_run(
                        body["endpoint"], body.get("tasks") or [],
                        max(1, int(body.get("reps") or 1)),
                        max(1, min(64, int(body.get("concurrency") or 1))),
                        bool(body.get("adapter")))
                    if err == "busy":
                        return self._send(409, {"error": "busy"})
                    if err:
                        return self._send(400, {"error": err})
                    return self._send(200, {"id": rid})
                if self.path == "/api/stop":
                    a = app.active
                    if a:
                        a["stop"].set()
                    return self._send(200, {"stopping": bool(a)})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)[:300]})

    return Handler


def serve(home, tasks_dir, port=8790, open_browser=True):
    app = App(home, tasks_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"TMB Agentic UI: {url}  (runs in {home}, Ctrl-C to stop)", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
