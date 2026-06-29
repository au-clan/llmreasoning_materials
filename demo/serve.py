#!/usr/bin/env python3
"""Serve all of the tutorial's browser demos over local HTTP from one launcher,
on a single port, so you can pick any of them on stage from a URL.

All four routes share one server / one port:
  * /            — index (pick a demo)
  * /game24      — Game24 combination tree     (demo/instability/game24_tree.py)
  * /protocols   — scoring-protocols app       (demo/instability/protocols_app.py)
  * /strategies  — live reasoning-strategies trace explorer (+ its /events SSE
                   stream)                      (demo/reasoning_demo/liveserver.py)

The first two are self-contained static HTML (built on demand, no network). The
strategies demo runs models live, so it needs OPENROUTER_API_KEY; without a key
(or with --no-strategies) it's skipped and its index card shows the reason — the
two offline demos still work.

Usage:
  python demo/serve.py                        # serve everything; open the index
  python demo/serve.py --open game24          # land on the Game24 tree
  python demo/serve.py --open strategies      # land on the live strategies demo
  python demo/serve.py --model gpt-5-mini --puzzle "6 6 7 12"   # Game24 knobs
  python demo/serve.py --no-strategies        # skip the live strategies demo
  python demo/serve.py --no-open --port 8011  # headless / custom port

Routes: /  ·  /game24  ·  /protocols  ·  /strategies (+ /events)
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))      # the demo/ package dir
ROOT = os.path.dirname(HERE)                           # repo root (parent of demo/)
sys.path.insert(0, ROOT)  # so the `demo` package resolves when run as a script

try:  # mirror client.py: pick up OPENROUTER_API_KEY / MODEL from the repo-root .env
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:  # python-dotenv not installed -> rely on real env vars
    pass

from demo.instability import game24_tree, protocols_app
from demo.reasoning_demo import liveserver

# Small dark/light theme switcher, shared verbatim with the demo pages. Sets
# data-theme on <html> before paint (no flash) and remembers the choice.
_THEME_JS = """<script>
(function(){var K="demoTheme",r=document.documentElement;
 function lbl(t){var b=document.getElementById("themeBtn");if(b)b.textContent=(t==="light"?"\\u{1F319} Dark":"\\u2600 Light");}
 var s="dark";try{s=localStorage.getItem(K)||"dark";}catch(e){}
 r.setAttribute("data-theme",s);
 function set(t){r.setAttribute("data-theme",t);try{localStorage.setItem(K,t);}catch(e){}lbl(t);}
 document.addEventListener("DOMContentLoaded",function(){lbl(r.getAttribute("data-theme"));});
 document.addEventListener("click",function(e){if(e.target&&e.target.id==="themeBtn")
   set(r.getAttribute("data-theme")==="light"?"dark":"light");});
})();
</script>"""


def _index(strategies_enabled: bool, strategies_note: str | None) -> str:
    if strategies_enabled:
        strat_card = (
            '<a class="card" href="/strategies">'
            '<span class="seclabel">§2.1 · Inference-time reasoning strategies</span>'
            '<b>Reasoning strategies (live) <span class="arr">→</span></b>'
            '<span>Seven test-time strategies streamed step by step over SSE — '
            'input–output, self-consistency, ReAct, agentic tool-use, self-refine, '
            'tree-of-thoughts, fleet-of-agents. Runs models live.</span></a>')
    else:
        strat_card = (
            '<div class="card off">'
            '<span class="seclabel">§2.1 · Inference-time reasoning strategies</span>'
            '<b>Reasoning strategies (live)</b>'
            f'<span>{strategies_note or "disabled"}</span></div>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Current Advances in LLM Reasoning — ACL 2026 demos</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --line:#2a3343;
    --text:#e6edf3; --muted:#8b949e; --hero-top:#10151e; --hover:#3b4758;
  }}
  :root[data-theme="light"] {{
    --bg:#ffffff; --panel:#ffffff; --panel2:#f4f6f8; --line:#e2e7ec;
    --text:#1d2126; --muted:#5c6773; --hero-top:#eef1f6; --hover:#c4c4c4;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .theme-toggle {{ position:fixed; top:14px; right:16px; z-index:99; cursor:pointer;
    font:inherit; font-size:13px; padding:6px 12px; border-radius:8px;
    border:1px solid var(--line); background:var(--panel2); color:var(--text); }}
  .theme-toggle:hover {{ border-color:var(--hover); }}
  .hero {{ border-bottom:1px solid var(--line); background:
      radial-gradient(1100px 220px at 18% -60%, rgba(88,166,255,.12), transparent 60%),
      radial-gradient(900px 220px at 90% -80%, rgba(247,120,186,.10), transparent 60%),
      linear-gradient(180deg,var(--hero-top), var(--bg)); }}
  .wrap {{ max-width:680px; margin:0 auto; padding:24px; }}
  .hero .wrap {{ padding:34px 24px 28px; }}
  h1 {{ font-size:28px; font-weight:800; letter-spacing:-.5px; margin:0 0 10px;
    display:flex; align-items:center; gap:12px; }}
  .grad {{ background:linear-gradient(90deg,#58a6ff 0%,#bc8cff 48%,#f778ba 100%);
    -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .dot-live {{ width:12px; height:12px; border-radius:50%; background:#f85149; flex:none;
    box-shadow:0 0 0 4px rgba(248,81,73,.16); animation:pulse 1.4s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
  .lede {{ color:var(--text); opacity:.82; max-width:560px; font-size:15px;
    line-height:1.65; margin:0; }}
  .sub {{ color:var(--muted); margin:0 0 6px; font-size:13px; text-transform:uppercase;
    letter-spacing:.6px; }}
  .card {{ display:block; text-decoration:none; color:inherit; background:var(--panel);
    border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin:14px 0;
    transition:.12s; }}
  a.card:hover {{ border-color:#3b4758; background:var(--panel2); transform:translateY(-1px); }}
  .card b {{ font-size:16px; }} .card .arr {{ color:#58a6ff; }}
  .card span {{ display:block; color:var(--muted); font-size:13.5px; margin-top:5px;
    line-height:1.55; }}
  .card.off {{ opacity:.5; }} .card.off b {{ color:var(--muted); }}
  .seclabel {{ display:block; font-size:11px; text-transform:uppercase; letter-spacing:.6px;
    color:#58a6ff; opacity:.85; margin-bottom:7px; }}
  .lede b {{ font-weight:600; opacity:1; }}
  a.tut {{ color:#58a6ff; text-decoration:none; white-space:nowrap; }}
  a.tut:hover {{ text-decoration:underline; }}
</style>
{_THEME_JS}
</head>
<body>
  <button id="themeBtn" class="theme-toggle"></button>
  <div class="hero"><div class="wrap">
    <h1><span class="dot-live"></span><span class="grad">Current Advances in LLM Reasoning</span></h1>
    <p class="lede">Live demos for our <b>ACL 2026</b> tutorial — a hands-on tour of how well
      LLMs reason, how to make them reason better, and where the field is heading next.
      <a class="tut" href="https://llmreasoning.github.io" target="_blank" rel="noopener">llmreasoning.github.io ↗</a></p>
  </div></div>
  <div class="wrap">
    <p class="sub">Pick a demo</p>
    <a class="card" href="/game24"><span class="seclabel">§1 · How well can models reason?</span>
      <b>Game24 combination tree <span class="arr">→</span></b>
      <span>All of one model's repeated attempts at a single puzzle, overlaid on one
      tree. The valid runs collapse into a shared trunk; the failures peel off where
      they fabricate a number.</span></a>
    <a class="card" href="/protocols"><span class="seclabel">§1 · How well can models reason?</span>
      <b>Scoring protocols <span class="arr">→</span></b>
      <span>The same ~30 reruns scored four ways (single-run / mean@N / maj@N /
      pass@N). Drag N and watch perceived capability move.</span></a>
    {strat_card}
  </div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet; one startup line is enough
        pass

    def _send(self, body: str, status: int = 200):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/events":  # SSE: manages its own response/headers, no html wrapper
            if not self.server.cfg["strategies_enabled"]:
                self.send_error(404)
                return
            llm, problems, defaults = self.server.strategies()
            liveserver.stream_events(self, parse_qs(u.query), llm, problems, defaults)
            return
        try:
            if path == "/":
                self._send(self.server.index_html())
            elif path == "/game24":
                self._send(self.server.page("game24"))
            elif path == "/protocols":
                self._send(self.server.page("protocols"))
            elif path == "/strategies":
                if not self.server.cfg["strategies_enabled"]:
                    self._send("<h1>strategies unavailable</h1><p>"
                               + (self.server.cfg["strategies_note"] or "") + "</p>", 404)
                else:
                    llm, problems, defaults = self.server.strategies()
                    self._send(liveserver.render_page(llm, problems, defaults))
            else:
                self._send(f"<h1>404</h1><p>no route {path}</p>", 404)
        except Exception as e:  # surface build errors in the browser, not just the console
            self._send(f"<h1>build failed</h1><pre>{type(e).__name__}: {e}</pre>", 500)
            raise


class Server(ThreadingHTTPServer):
    """Builds each page on first request and caches it (parsing logs/parquet is slow)."""

    def __init__(self, addr, cfg: dict):
        super().__init__(addr, Handler)
        self.cfg = cfg
        self._cache: dict[str, str] = {}
        self._strat = None  # (llm, problems, defaults) for the live demo, built lazily

    def page(self, which: str) -> str:
        if which not in self._cache:
            if which == "game24":
                self._cache[which] = game24_tree.render_html_string(
                    self.cfg["model"], self.cfg["puzzle"], self.cfg["target"])
            else:
                self._cache[which] = protocols_app.render_html_string()
        return self._cache[which]

    def index_html(self) -> str:
        return _index(self.cfg["strategies_enabled"], self.cfg["strategies_note"])

    def strategies(self):
        """Lazily build the live-strategies backend (LLM + problems) on first hit."""
        if self._strat is None:
            from demo.reasoning_demo.client import LLM
            from demo.reasoning_demo.data import load_problems
            self._strat = (LLM(model=self.cfg["strategies_model"]),
                           load_problems(), liveserver.DEFAULTS)
        return self._strat


def serve(host: str, port: int, cfg: dict) -> None:
    httpd = Server((host, port), cfg)
    url = f"http://{host}:{port}/"
    routes = "/  ·  /game24  ·  /protocols" + ("  ·  /strategies" if cfg["strategies_enabled"] else "")
    print(f"Tutorial demos on {url}  "
          f"(game24: {cfg['model']} · {' '.join(map(str, cfg['puzzle']))})")
    if cfg["strategies_enabled"]:
        print(f"  · reasoning strategies (live) on {url}strategies")
    elif cfg["strategies_note"]:
        print(f"  · strategies: {cfg['strategies_note']}")
    print(f"Routes: {routes}   — press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.shutdown()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    ap.add_argument("--open", default="index",
                    choices=["index", "game24", "protocols", "strategies"],
                    help="which page to open in the browser (default: index)")
    ap.add_argument("--model", default="gpt-5-mini", help="Game24: which model's attempts")
    ap.add_argument("--puzzle", default="6 6 7 12", help="Game24: the four numbers")
    ap.add_argument("--target", type=int, default=24, help="Game24: the target value")
    ap.add_argument("--no-strategies", action="store_true",
                    help="don't serve the live reasoning-strategies demo")
    ap.add_argument("--strategies-model", default=None,
                    help="OpenRouter model id for the strategies demo (default: MODEL env)")
    args = ap.parse_args()

    # The strategies demo runs models live, so it needs a key; skip it gracefully
    # otherwise and mark its index card disabled with the reason.
    if args.no_strategies:
        strategies_enabled, strategies_note = False, "disabled with --no-strategies"
    elif not os.getenv("OPENROUTER_API_KEY"):
        strategies_enabled, strategies_note = False, "needs OPENROUTER_API_KEY in your environment or .env"
    else:
        strategies_enabled, strategies_note = True, None

    cfg = {
        "model": args.model,
        "puzzle": tuple(int(x) for x in args.puzzle.split()),
        "target": args.target,
        "strategies_enabled": strategies_enabled,
        "strategies_note": strategies_note,
        "strategies_model": args.strategies_model,
    }

    open_to = args.open
    if open_to == "strategies" and not strategies_enabled:
        print("[strategies] not available; opening the index instead.")
        open_to = "index"
    base = f"http://{args.host}:{args.port}"
    routes = {"index": base + "/", "game24": base + "/game24",
              "protocols": base + "/protocols", "strategies": base + "/strategies"}
    if not args.no_open:
        target = routes[open_to]
        threading.Timer(1.5, lambda: webbrowser.open(target)).start()
    serve(args.host, args.port, cfg)


if __name__ == "__main__":
    main()
