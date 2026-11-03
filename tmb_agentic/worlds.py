"""Built-in worlds: fs (fake filesystem), db (fake tables), api (mock HTTP).

All three are pure in-memory and deterministic: listings are sorted, rows
keep insertion order."""

import json
import re

from .engine import World, ToolResult


def _as_dict(v):
    """Models sometimes double-encode objects ('body': '"{...}"'). That is
    data about the model, never a crash of the harness."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def _obj_param(name):
    return {"type": "object", "properties": {name: {"type": "string"}},
            "required": [name]}


class FsWorld(World):
    """init: {"files": {path: content}}. Paths have no leading slash;
    directories are implicit prefixes."""

    TOOLS = {
        "list_dir": {
            "name": "list_dir",
            "description": "List files and directories under a path.",
            "parameters": _obj_param("path")},
        "read_file": {
            "name": "read_file",
            "description": "Return the full content of a file.",
            "parameters": _obj_param("path")},
        "write_file": {
            "name": "write_file",
            "description": "Create or overwrite a file with new content.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}},
                           "required": ["path", "content"]}},
        "delete_file": {
            "name": "delete_file",
            "description": "Delete a file.",
            "parameters": _obj_param("path")},
        "search": {
            "name": "search",
            "description": "Search a regex pattern in files under a path; "
                           "returns matching lines as path:line:text.",
            "parameters": {"type": "object",
                           "properties": {"pattern": {"type": "string"},
                                          "path": {"type": "string"}},
                           "required": ["pattern", "path"]}},
    }

    def __init__(self, init):
        super().__init__(init)
        self.files = dict(init.get("files") or {})

    @staticmethod
    def _norm(p):
        return str(p or "").strip().lstrip("./").rstrip("/")

    def dispatch(self, name, args):
        if name == "list_dir":
            base = self._norm(args.get("path"))
            if base in self.files:
                # An empty listing would read as "empty file" to the model.
                return self.err(f"not a directory: {base} is a file")
            prefix = base + "/" if base else ""
            entries = set()
            for f in self.files:
                if not f.startswith(prefix):
                    continue
                rest = f[len(prefix):]
                entries.add(rest.split("/")[0] + ("/" if "/" in rest else ""))
            if not entries and base and not any(
                    f == base or f.startswith(prefix) for f in self.files):
                return self.err(f"no such directory: {base}")
            return ToolResult(True, {"entries": sorted(entries)})
        if name == "read_file":
            p = self._norm(args.get("path"))
            if p not in self.files:
                return self.err(f"no such file: {p}")
            return ToolResult(True, {"path": p, "content": self.files[p]})
        if name == "write_file":
            p = self._norm(args.get("path"))
            if not p:
                return self.err("empty path")
            self.files[p] = str(args.get("content", ""))
            return ToolResult(True, {"written": p, "bytes": len(self.files[p])})
        if name == "delete_file":
            p = self._norm(args.get("path"))
            if p not in self.files:
                return self.err(f"no such file: {p}")
            del self.files[p]
            return ToolResult(True, {"deleted": p})
        if name == "search":
            base = self._norm(args.get("path"))
            try:
                rx = re.compile(args.get("pattern") or "")
            except re.error as e:
                return self.err(f"bad pattern: {e}")
            hits = []
            for f in sorted(self.files):
                if base and not (f == base or f.startswith(base + "/")):
                    continue
                for i, line in enumerate(self.files[f].splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f}:{i}:{line.strip()[:200]}")
            return ToolResult(True, {"matches": hits[:50],
                                     "total": len(hits)})
        return self.err(f"unknown tool: {name}")

    def snapshot(self):
        return {"files": dict(self.files)}


class DbWorld(World):
    """init: {"tables": {name: [row, ...]}}. `where` matches by equality on
    columns. Enough for a read -> decide -> write cycle; not SQL."""

    TOOLS = {
        "db_tables": {
            "name": "db_tables",
            "description": "List the tables and their row counts.",
            "parameters": {"type": "object", "properties": {}, "required": []}},
        "db_select": {
            "name": "db_select",
            "description": "Return rows of a table, optionally filtered by "
                           "equality on columns.",
            "parameters": {"type": "object",
                           "properties": {"table": {"type": "string"},
                                          "where": {"type": "object"}},
                           "required": ["table"]}},
        "db_insert": {
            "name": "db_insert",
            "description": "Insert one row into a table.",
            "parameters": {"type": "object",
                           "properties": {"table": {"type": "string"},
                                          "row": {"type": "object"}},
                           "required": ["table", "row"]}},
        "db_update": {
            "name": "db_update",
            "description": "Update rows matching `where` with values `set`. "
                           "Returns the number of rows changed.",
            "parameters": {"type": "object",
                           "properties": {"table": {"type": "string"},
                                          "where": {"type": "object"},
                                          "set": {"type": "object"}},
                           "required": ["table", "where", "set"]}},
        "db_delete": {
            "name": "db_delete",
            "description": "Delete rows matching `where`. Returns the number "
                           "of rows deleted.",
            "parameters": {"type": "object",
                           "properties": {"table": {"type": "string"},
                                          "where": {"type": "object"}},
                           "required": ["table", "where"]}},
    }

    def __init__(self, init):
        super().__init__(init)
        self.tables = {t: [dict(r) for r in rows]
                       for t, rows in (init.get("tables") or {}).items()}

    @staticmethod
    def _match(row, where):
        return all(row.get(k) == v for k, v in (where or {}).items())

    def dispatch(self, name, args):
        if name == "db_tables":
            return ToolResult(True, {t: len(r) for t, r in
                                     sorted(self.tables.items())})
        t = args.get("table")
        if t not in self.tables:
            return self.err(f"no such table: {t}")
        rows = self.tables[t]
        where = _as_dict(args.get("where")) if args.get("where") is not None \
            else {}
        if name == "db_select":
            if where is None:
                return self.err("where must be a JSON object")
            hits = [r for r in rows if self._match(r, where)]
            return ToolResult(True, {"rows": hits[:100], "total": len(hits)})
        if name == "db_insert":
            row = _as_dict(args.get("row"))
            if not row:
                return self.err("row must be a non-empty object")
            rows.append(dict(row))
            return ToolResult(True, {"inserted": 1, "table_size": len(rows)})
        if name == "db_update":
            new = _as_dict(args.get("set"))
            if where is None or new is None:
                return self.err("where and set must be JSON objects")
            n = 0
            for r in rows:
                if self._match(r, where):
                    r.update(new)
                    n += 1
            return ToolResult(True, {"updated": n})
        if name == "db_delete":
            if where is None:
                return self.err("where must be a JSON object")
            keep = [r for r in rows if not self._match(r, where)]
            n = len(rows) - len(keep)
            self.tables[t] = keep
            return ToolResult(True, {"deleted": n})
        return self.err(f"unknown tool: {name}")

    def snapshot(self):
        return {"tables": {t: [dict(r) for r in rows]
                           for t, rows in self.tables.items()}}


class ApiWorld(World):
    """init: {"routes": {...}, "state": {store: value}}. Routes are
    declarative, no per-task code:
      {"GET /items":          {"kind": "read_store", "store": "items"},
       "POST /upload":        {"kind": "append", "store": "uploaded", "field": "name"},
       "GET /status":         {"kind": "static", "payload": {"status": "ok"}},
       "DELETE /items/{id}":  {"kind": "remove", "store": "items", "field": "id"}}
    Stores live in state and are what assertions check."""

    TOOLS = {
        "http_get": {
            "name": "http_get",
            "description": "HTTP GET on the API. Returns the JSON body.",
            "parameters": _obj_param("path")},
        "http_post": {
            "name": "http_post",
            "description": "HTTP POST with a JSON body on the API.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "body": {"type": "object"}},
                           "required": ["path", "body"]}},
        "http_delete": {
            "name": "http_delete",
            "description": "HTTP DELETE on the API.",
            "parameters": _obj_param("path")},
    }

    def __init__(self, init):
        super().__init__(init)
        self.routes = dict(init.get("routes") or {})
        self.state = {k: (list(v) if isinstance(v, list) else v)
                      for k, v in (init.get("state") or {}).items()}

    def _route(self, method, path):
        """Exact match first, then {param} templates segment by segment."""
        key = f"{method} {path}"
        if key in self.routes:
            return self.routes[key], {}
        parts = path.strip("/").split("/")
        for rk, spec in self.routes.items():
            m, rp = rk.split(" ", 1)
            if m != method:
                continue
            rparts = rp.strip("/").split("/")
            if len(rparts) != len(parts):
                continue
            params = {}
            for a, b in zip(rparts, parts):
                if a.startswith("{") and a.endswith("}"):
                    params[a[1:-1]] = b
                elif a != b:
                    break
            else:
                return spec, params
        return None, None

    def dispatch(self, name, args):
        method = {"http_get": "GET", "http_post": "POST",
                  "http_delete": "DELETE"}.get(name)
        if not method:
            return self.err(f"unknown tool: {name}")
        path = str(args.get("path") or "")
        spec, params = self._route(method, path)
        if spec is None:
            return self.err(f"HTTP 404: no route {method} {path}")
        kind = spec.get("kind", "static")
        if kind == "static":
            return ToolResult(True, spec.get("payload"))
        store = spec.get("store")
        if kind == "read_store":
            return ToolResult(True, {store: self.state.get(store)})
        if kind == "append":
            body = _as_dict(args.get("body"))
            if body is None:
                return self.err("body must be a JSON object")
            field = spec.get("field")
            val = body.get(field) if field else body
            if val in (None, ""):
                return self.err(f"missing field: {field}")
            self.state.setdefault(store, []).append(val)
            return ToolResult(True, {"accepted": val,
                                     "count": len(self.state[store])})
        if kind == "remove":
            field = spec.get("field", "id")
            val = params.get(field) if params else None
            cur = self.state.get(store) or []
            if val not in cur:
                return self.err(f"HTTP 404: {val!r} not in {store}")
            cur.remove(val)
            return ToolResult(True, {"removed": val, "count": len(cur)})
        return self.err(f"bad route kind: {kind}")

    def snapshot(self):
        return {"state": {k: (list(v) if isinstance(v, list) else v)
                          for k, v in self.state.items()}}


WORLDS = {"fs": FsWorld, "db": DbWorld, "api": ApiWorld}


def make_world(spec):
    """spec = task["world"]: {"type": "fs"|"db"|"api", "init": {...}}."""
    cls = WORLDS.get(spec.get("type"))
    if cls is None:
        raise ValueError(f"unknown world type: {spec.get('type')!r}")
    return cls(spec.get("init") or {})
