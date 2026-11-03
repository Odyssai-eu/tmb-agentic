"""Standalone HTML report for one or more runs (part of every export)."""

import html
import json
import time
from pathlib import Path

_I18N = json.loads((Path(__file__).parent / "ui" / "i18n.json").read_text())


def strings(lang):
    base = dict(_I18N["en"])
    base.update(_I18N.get(lang) or {})
    return base


def _e(s):
    return html.escape(str(s))


def render_report(runs, lang="en"):
    """runs = [(meta, results, failures)]."""
    t = strings(lang)

    def tr(key, **kw):
        s = t.get(key, key)
        for k, v in kw.items():
            s = s.replace("{" + k + "}", str(v))
        return s

    def counts(d, prefix=""):
        return ", ".join(f"{_e(t.get(prefix + k, k))} {v}" for k, v in sorted(d.items())) or "-"

    parts = []
    for meta, results, fails in runs:
        total = sum(r["runs"] for r in results.values())
        passed = sum(r["passed"] for r in results.values())
        rows = "".join(
            f"<tr><td>{_e(task)}</td><td class=n>{r['passed']}/{r['runs']}</td>"
            f"<td>{counts(r['outcomes'], 'outcome.')}</td><td>{counts(r['flags'])}</td></tr>"
            for task, r in sorted(results.items()))
        frows = "".join(
            f"<li><b>{_e(f['task'])}</b> r{f['rep']}: {_e(t.get('outcome.' + f['outcome'], f['outcome']))}"
            + (f" [{_e(', '.join(f['flags']))}]" if f["flags"] else "")
            + "".join(f"<div class=d>{_e(x)}</div>" for x in f["failed"]) + "</li>"
            for f in fails)
        parts.append(f"""
<section>
  <h2>{_e(meta['model'])}</h2>
  <p class=m>{_e(meta['created'])} &middot; {_e(t.get('status.' + meta['status'], meta['status']))}
    &middot; <span class=n>{passed}/{total}</span> {_e(t['hist.passed'].lower())}
    <br>{_e(meta['base'])} &middot; reps {meta['reps']} &middot; {_e(meta['id'])}</p>
  <table><thead><tr><th>{_e(t['detail.task'])}</th><th>{_e(t['hist.passed'])}</th>
    <th>{_e(t['detail.outcomes'])}</th><th>{_e(t['detail.flags'])}</th></tr></thead>
    <tbody>{rows}</tbody></table>
  <h3>{_e(t['report.failures'])}</h3>
  {f'<ul>{frows}</ul>' if frows else f"<p class=m>{_e(t['report.no_failures'])}</p>"}
</section>""")

    return f"""<!doctype html>
<html lang="{'zh-CN' if lang == 'zh' else 'en'}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(t['report.title'])}</title>
<style>
:root {{ --bg:#f6f7f8; --fg:#16181d; --mute:#5b6270; --line:#d9dce1; --acc:#0b6b5c; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#121417; --fg:#e7e9ec; --mute:#9aa1ad; --line:#2a2e35; --acc:#4cc3a8; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:32px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:32px 0 4px; }}
h3 {{ font-size:14px; margin:20px 0 6px; color:var(--mute); }}
.m {{ color:var(--mute); margin:0 0 12px; }} .n {{ font-family:ui-monospace,Menlo,monospace; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-weight:600; color:var(--mute); }}
ul {{ padding-left:18px; }} li {{ margin-bottom:6px; }}
.d {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--mute); word-break:break-all; }}
.note {{ margin-top:40px; padding-top:12px; border-top:1px solid var(--line); color:var(--mute); }}
</style></head><body><main>
<h1>{_e(t['report.title'])}</h1>
<p class=m>{_e(tr('report.generated', date=time.strftime('%Y-%m-%d %H:%M')))}
  &middot; {len(runs)} {_e(t['report.runs'].lower())}</p>
{''.join(parts)}
<p class=note>{_e(t['report.note'])}</p>
</main></body></html>"""
