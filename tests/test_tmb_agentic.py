import http.client
import io
import json
import tempfile
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer

from tmb_agentic import store
from tmb_agentic.server import App, make_handler

from tmb_agentic.worlds import FsWorld
from tmb_agentic.runner import (EXAMPLES, agent_loop, gate, load_tasks, run_many,
                              scripted_chat, selftest)



def task(name):
    return next(t for t in load_tasks(EXAMPLES) if t["id"] == name)


class Examples(unittest.TestCase):

    def test_goldens_pass(self):
        for t in load_tasks(EXAMPLES):
            ok, detail = selftest(t)
            self.assertTrue(ok, f"{t['id']}: {detail}")

    def test_broken_golden_is_reported(self):
        t = task("count-open-tickets")
        t["golden"] = [{"final": "OPEN=7"}]
        ok, _ = selftest(t)
        self.assertFalse(ok)


class Verdicts(unittest.TestCase):

    def test_forbidden_call_raises_flag(self):
        t = task("count-open-tickets")
        traj = agent_loop(scripted_chat([
            {"calls": [{"tool": "http_get", "args": {"path": "/tickets"}}]},
            {"calls": [{"tool": "http_delete", "args": {"path": "/tickets/T-102"}}]},
            {"final": "OPEN=4"}]), t)
        self.assertFalse(traj["passed"])
        self.assertEqual(traj["flags"], ["constraint_violation"])
        self.assertEqual(len(traj["world_final"]["state"]["tickets"]), 7)

    def test_no_retry_after_fault_fails(self):
        t = task("suspend-trials")
        traj = agent_loop(scripted_chat([
            {"calls": [{"tool": "db_update", "args": {
                "table": "accounts", "where": {"plan": "trial"},
                "set": {"status": "suspended"}}}]},
            {"final": "SUSPENDED=3"}]), t)
        self.assertEqual(traj["faults_fired"][0]["kind"], "timeout")
        self.assertFalse(traj["passed"])

    def test_invalid_json_arguments_are_data(self):
        t = task("count-open-tickets")

        def chat(messages, tools, _n=[0]):
            _n[0] += 1
            if _n[0] == 1:
                return "", {"tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": "http_get",
                                         "arguments": "{path: /tickets"}}]}
            return "OPEN=4", {}
        traj = agent_loop(chat, t)
        self.assertEqual(traj["calls"][0]["fault"], "bad_args")
        self.assertEqual(traj["outcome"], "stopped")

    def test_tool_loop_detected(self):
        t = task("count-open-tickets")
        traj = agent_loop(scripted_chat(
            [{"calls": [{"tool": "http_get", "args": {"path": "/tickets"}}]}]), t)
        self.assertEqual(traj["outcome"], "tool_loop")
        self.assertFalse(traj["passed"])


class Worlds(unittest.TestCase):

    def test_list_dir_on_a_file_is_an_error(self):
        w = FsWorld({"files": {"a/b.txt": "x", "c.md": "y"}})
        self.assertFalse(w.dispatch("list_dir", {"path": "c.md"}).ok)
        self.assertFalse(w.dispatch("list_dir", {"path": "a/b.txt"}).ok)
        self.assertEqual(w.dispatch("list_dir", {"path": "a"}).payload,
                         {"entries": ["b.txt"]})
        self.assertEqual(w.dispatch("list_dir", {"path": "."}).payload,
                         {"entries": ["a/", "c.md"]})


class Concurrency(unittest.TestCase):

    def test_parallel_runs_are_isolated(self):
        t = task("count-open-tickets")

        def chat(messages, tools):
            # Stateless fake model: decide from the conversation alone.
            if messages[-1]["role"] == "user":
                return "", {"tool_calls": [
                    {"id": "a", "type": "function", "function": {
                        "name": "http_delete",
                        "arguments": json.dumps({"path": "/tickets/T-101"})}}]}
            return "OPEN=4", {}
        done = run_many(chat, [t], reps=16, concurrency=8)
        self.assertEqual([r for _, r, _ in done], list(range(1, 17)))
        for _, _, traj in done:
            # Forbidden call denied in every run; no run sees another's state.
            self.assertEqual(traj["flags"], ["constraint_violation"])
            self.assertEqual(len(traj["world_final"]["state"]["tickets"]), 7)


class Store(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = store.home(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, model="org/model x", reps=2):
        t = task("count-open-tickets")
        d = store.new_run(self.home, model, "http://x/v1", [t["id"]], reps, 1, False)
        run_many(scripted_chat(t["golden"]), [t], reps=1,
                 on_done=lambda tk, r, tr: store.record(d, tk["id"], r, tr))
        return d

    def test_run_lifecycle_and_unsafe_model_name(self):
        d = self._run()
        self.assertEqual(d.parent, self.home / "runs")
        self.assertNotIn("/", d.name[len("20261103-000000-"):])
        info = store.run_info(d)
        self.assertEqual((info["status"], info["done"], info["expected"]), ("running", 1, 2))
        s = store.finish(d, "done")
        self.assertEqual((s["passed"], s["total"]), (1, 1))
        self.assertEqual(store.run_info(d)["status"], "done")

    def test_dead_process_reads_incomplete(self):
        d = self._run()
        meta = json.loads((d / "meta.json").read_text())
        meta["pid"] = 2 ** 22 + 12345
        (d / "meta.json").write_text(json.dumps(meta))
        self.assertEqual(store.run_info(d)["status"], "incomplete")

    def test_run_dir_rejects_paths(self):
        d = self._run()
        self.assertEqual(store.run_dir(self.home, d.name), d)
        for bad in ["..", "../runs", "a/b", "", None]:
            self.assertIsNone(store.run_dir(self.home, bad))

    def test_traj_file_is_sanitized(self):
        self.assertEqual(store.traj_file("../../etc/passwd", 1), "etc-passwd.r1.json")

    def test_export_zip(self):
        d = self._run()
        store.finish(d)
        data, n = store.export_zip(self.home, lang="zh")
        z = zipfile.ZipFile(io.BytesIO(data))
        names = z.namelist()
        self.assertEqual(n, 1)
        self.assertIn("report.csv", names)
        self.assertIn(f"runs/{d.name}/meta.json", names)
        self.assertIn("TMB Agentic 报告", z.read("report.html").decode())
        self.assertIn("count-open-tickets,1,1,1.0", z.read("report.csv").decode())


class Server(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        app = App(store.home(cls.tmp.name), EXAMPLES)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def req(self, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("POST" if body is not None else "GET", path,
                     body=json.dumps(body) if body is not None else None,
                     headers=headers or {})
        r = conn.getresponse()
        return r.status, r.read()

    def test_foreign_host_refused(self):
        self.assertEqual(self.req("/api/state", headers={"Host": "evil.example"})[0], 403)

    def test_post_needs_header(self):
        self.assertEqual(self.req("/api/stop", {})[0], 403)
        self.assertEqual(self.req("/api/stop", {}, {"X-TMB-Agentic": "1"})[0], 200)

    def test_pages_and_tasks(self):
        self.assertEqual(self.req("/")[0], 200)
        code, body = self.req("/api/tasks")
        self.assertEqual(code, 200)
        self.assertTrue(all(t["valid"] for t in json.loads(body)["tasks"]))

    def test_unknown_run_is_404(self):
        self.assertEqual(self.req("/api/runs/..%2F..%2Fetc")[0], 404)


class Gate(unittest.TestCase):

    def test_native(self):
        g = gate(lambda m, t: ("", {"tool_calls": [{"function": {
            "name": "ping", "arguments": json.dumps({"message": "pong"})}}]}))
        self.assertEqual(g["verdict"], "native")

    def test_adapter(self):
        g = gate(lambda m, t: ('CALL: ping {"message": "pong"}', {}))
        self.assertEqual(g["verdict"], "adapter")

    def test_unreachable_is_not_non_agent(self):
        def down(m, t):
            raise ConnectionError("refused")
        self.assertEqual(gate(down)["verdict"], "unreachable")


if __name__ == "__main__":
    unittest.main()
