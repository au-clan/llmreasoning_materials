"""Live trace server: runs the strategies and STREAMS each step to the browser
over Server-Sent Events (SSE), so the timeline fills in real time.

Stdlib only -- no Flask, no websockets. Each strategy has its own Run button and
its own SSE connection, so they can stream independently / concurrently.
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .client import LLM
from .data import load_problems
from .extract import is_correct
from .strategies import (agentic, fleet_of_agents, input_output, iterative,
                         react, self_consistency, tree_of_thoughts)


def _title(name, k, rounds, max_steps, depth, n_agents) -> str:
    return {"input_output": "Input–output · direct answer",
            "self_consistency": f"Self-consistency · k={k}",
            "react": f"ReAct · ≤{max_steps} thought→action turns",
            "agentic": f"Agent (tool-use) · ≤{max_steps} steps",
            "iterative": f"Self-Refine · ≤{rounds} feedback→refine round(s)",
            "tree_of_thoughts": f"Tree of Thoughts · depth={depth}",
            "fleet_of_agents": f"Fleet of Agents · {n_agents} agents"}[name]


def _run(llm, prob, q, names, k, rounds, max_steps, depth, n_agents):
    """Worker thread: run the requested strategies, pushing events onto `q`.

    Each /events connection runs ONE strategy (its own Run button), so two
    strategies can stream concurrently over separate connections.
    """
    plan = {
        "input_output": (input_output, {}),
        "self_consistency": (self_consistency, {"k": k}),
        "react": (react, {"max_steps": max_steps}),
        "agentic": (agentic, {"max_steps": max_steps}),
        "iterative": (iterative, {"rounds": rounds}),
        "tree_of_thoughts": (tree_of_thoughts, {"depth": depth}),
        "fleet_of_agents": (fleet_of_agents, {"n_agents": n_agents}),
    }
    one = names[0] if len(names) == 1 else None
    try:
        q.put({"type": "meta", "question": prob.question, "gold": prob.answer})
        for name in names:
            fn, kwargs = plan[name]
            q.put({"type": "trace_start", "strategy": name,
                   "title": _title(name, k, rounds, max_steps, depth, n_agents)})
            res = fn(llm, prob.question,
                     emit=lambda ev, nm=name: q.put({**ev, "strategy": nm}),
                     **kwargs)
            q.put({"type": "trace_done", "strategy": name,
                   "answer": res.answer, "gold": prob.answer,
                   "correct": is_correct(res.answer, prob.answer),
                   "latency_s": round(res.latency_s, 3),
                   "tokens": res.usage.total_tokens, "calls": res.usage.calls})
        q.put({"type": "all_done", "strategy": one})
    except Exception as e:  # surface backend/auth errors to the page
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}", "strategy": one})
    finally:
        q.put(None)  # sentinel -> close the SSE stream


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet during the talk
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._html()
        elif u.path == "/events":
            self._events(parse_qs(u.query))
        else:
            self.send_error(404)

    def _html(self):
        body = render_page(self.server.llm, self.server.problems,
                           self.server.defaults).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _events(self, qs):
        stream_events(self, qs, self.server.llm, self.server.problems,
                      self.server.defaults)


DEFAULTS = {"k": 5, "rounds": 2, "max_steps": 6, "depth": 2, "n_agents": 4}


def render_page(llm, problems, defaults=None) -> str:
    """Build the live-trace HTML page. Reused by the standalone server and by the
    unified launcher (demo/serve.py), so both can serve it from one port."""
    probs = [{"id": p.id, "q": p.question} for p in problems]
    return (PAGE
            .replace("__PROBLEMS__", json.dumps(probs))
            .replace("__MODEL__", json.dumps(llm.model))
            .replace("__DEFAULTS__", json.dumps(defaults or DEFAULTS)))


def stream_events(handler, qs, llm, problems, defaults=None) -> None:
    """Run the requested strategy (or all) and stream each step to `handler.wfile`
    as SSE. The page fetches the root-relative `/events`, so whatever server owns
    that route (standalone or the launcher) can host the live demo."""
    defaults = defaults or DEFAULTS

    def g(name, default):
        try:
            return int(qs.get(name, [default])[0])
        except (ValueError, TypeError):
            return default

    pid = qs.get("id", [None])[0]
    prob = next((p for p in problems if p.id == pid), problems[0])
    k, rounds = g("k", defaults["k"]), g("rounds", defaults["rounds"])
    max_steps = g("max_steps", defaults["max_steps"])
    depth, n_agents = g("depth", defaults["depth"]), g("n_agents", defaults["n_agents"])
    valid = ("input_output", "self_consistency", "react", "agentic",
             "iterative", "tree_of_thoughts", "fleet_of_agents")
    strat = qs.get("strategy", [None])[0]
    names = [strat] if strat in valid else list(valid)

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()

    ev_q: queue.Queue = queue.Queue()
    threading.Thread(target=_run,
                     args=(llm, prob, ev_q, names, k, rounds,
                           max_steps, depth, n_agents),
                     daemon=True).start()
    while True:
        ev = ev_q.get()
        if ev is None:
            break
        try:
            handler.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            break  # browser navigated away / closed the stream


def serve(model: str | None = None, host: str = "127.0.0.1", port: int = 8000,
          defaults: dict | None = None) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.llm = LLM(model=model)
    httpd.problems = load_problems()
    httpd.defaults = defaults or DEFAULTS
    url = f"http://{host}:{port}/"
    print(f"Live trace server on {url}  (model: {httpd.llm.model})")
    print("Open it in a browser; press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.shutdown()


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Reasoning Trace</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --line:#2a3343;
    --text:#e6edf3; --muted:#8b949e; --ok:#3fb950; --bad:#f85149;
    --io:#8b949e; --solve:#58a6ff; --feedback:#f2cc60; --refine:#bc8cff;
    --sample:#39c5cf; --vote:#e3b341;
    --think:#ffa657; --tool:#56d364; --final:#ff7b72;
    --act:#d2a8ff; --observe:#7ce0d3;
    --propose:#f778ba; --value:#d29922; --prune:#7d8590;
    --agent:#2dd4bf; --resample:#2dd4bf;
    --hero-top:#10151e; --badge-ink:#0d1117; --code:#cdd9e5; --hover:#3b4758;
  }
  :root[data-theme="light"] {
    --bg:#ffffff; --panel:#ffffff; --panel2:#f3f5f8; --line:#e2e7ec;
    --text:#1d2126; --muted:#5c6773; --ok:#1a8a3a; --bad:#cf2f25;
    --io:#6b7785; --solve:#1f6feb; --feedback:#9a7400; --refine:#8957e5;
    --sample:#0b8aa0; --vote:#9a7400;
    --think:#c25a00; --tool:#1a8f3c; --final:#cf432b;
    --act:#8250df; --observe:#0b8a82;
    --propose:#d6336c; --value:#9a6b00; --prune:#8b949e;
    --agent:#0b8a82; --resample:#0b8a82;
    --hero-top:#eef1f6; --badge-ink:#ffffff; --code:#2b3038; --hover:#c4c4c4;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .theme-toggle, .home-btn { position:fixed; right:16px; z-index:99; cursor:pointer;
    font:inherit; font-size:13px; padding:6px 12px; border-radius:8px; text-decoration:none;
    border:1px solid var(--line); background:var(--panel2); color:var(--text); }
  .home-btn { top:14px; } .theme-toggle { top:52px; }
  .theme-toggle:hover, .home-btn:hover { border-color:var(--hover); }
  .wrap { max-width:1100px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .hero { border-bottom:1px solid var(--line); background:
      radial-gradient(1100px 220px at 18% -60%, rgba(88,166,255,.12), transparent 60%),
      radial-gradient(900px 220px at 90% -80%, rgba(247,120,186,.10), transparent 60%),
      linear-gradient(180deg,var(--hero-top), var(--bg)); }
  .hero h1 { font-size:30px; font-weight:800; letter-spacing:-.5px;
    margin:0 0 10px; display:flex; align-items:center; gap:12px; }
  .grad { background:linear-gradient(90deg,#58a6ff 0%,#bc8cff 48%,#f778ba 100%);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
  .dot-live { width:12px; height:12px; border-radius:50%; background:var(--bad); flex:none;
    box-shadow:0 0 0 4px rgba(248,81,73,.16); animation:pulse 1.4s infinite; }
  .lede { color:var(--text); opacity:.82; max-width:790px; font-size:15px;
    line-height:1.65; margin:0 0 14px; }
  .lede b { font-weight:600; opacity:1; }
  .sub { color:var(--muted); margin-bottom:0; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px;
    background:var(--panel2); border:1px solid var(--line); font-size:12px;
    color:var(--muted); margin-right:6px; }
  a.chip { text-decoration:none; }
  a.chip:hover { border-color:var(--hover); color:var(--text); }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; margin-bottom:16px; }
  select, input[type=number] { background:var(--panel2); color:var(--text);
    border:1px solid var(--line); border-radius:8px; padding:7px 9px; font:inherit; }
  input[type=number] { width:60px; }
  label { color:var(--muted); font-size:12px; }
  button { background:#238636; color:#fff; border:0; border-radius:8px;
    padding:8px 16px; font:inherit; font-weight:700; cursor:pointer; }
  button:hover { background:#2ea043; } button:disabled { opacity:.5; cursor:default; }
  .q { font-size:15px; line-height:1.55; color:var(--text); }
  .q b { color:var(--vote); }
  .pick-label { display:block; font-size:11px; text-transform:uppercase;
    letter-spacing:.6px; color:var(--muted); margin-bottom:8px; }
  #prob { width:100%; max-width:580px; }
  .strat { border-left:5px solid var(--line); }
  .strat-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
  .strat-title { font-size:16px; font-weight:700; }
  .strat-head .spacer { flex:1; }
  .status { color:var(--muted); font-size:13px; min-width:90px; }
  .live-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
    background:var(--ok); margin-right:6px; animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  @keyframes livebar { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.25)} }
  .stats { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 4px; }
  .stat { background:var(--panel2); border:1px solid var(--line);
    border-radius:10px; padding:7px 11px; min-width:84px; }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; }
  .stat .v { font-size:17px; font-weight:700; } .v.ok{color:var(--ok)} .v.bad{color:var(--bad)}
  .section-h { font-size:12px; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:16px 0 8px; }
  .section-h i { text-transform:none; font-style:italic; }
  .tl-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
  .tl-label { width:150px; flex:none; font-size:12px; color:var(--muted);
    text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tl-track { position:relative; flex:1; height:22px; background:var(--panel2);
    border-radius:6px; border:1px solid var(--line); overflow:hidden; }
  .tl-bar { position:absolute; top:2px; height:16px; border-radius:5px;
    min-width:3px; cursor:pointer; transition:width .12s linear, left .12s linear; }
  .tl-bar.inflight { animation:livebar 1.2s ease-in-out infinite; }
  .tl-axis { display:flex; justify-content:space-between; color:var(--muted);
    font-size:11px; margin:6px 0 0 160px; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; font-style:italic; }
  .empty { color:var(--muted); font-style:italic; font-size:13px; }
  .step { border:1px solid var(--line); border-left:4px solid var(--line);
    border-radius:10px; margin:8px 0; background:var(--panel); overflow:hidden; }
  .step-h { display:flex; align-items:center; gap:10px; padding:10px 14px; cursor:pointer; }
  .step-h:hover { background:var(--panel2); }
  .badge { font-size:10px; font-weight:700; text-transform:uppercase;
    padding:2px 8px; border-radius:6px; color:var(--badge-ink); }
  .step-title { font-weight:600; } .step-meta { margin-left:auto; color:var(--muted);
    font-size:12px; display:flex; gap:12px; align-items:center; }
  .ans { padding:2px 8px; border-radius:6px; font-weight:700; font-size:12px;
    background:var(--panel2); border:1px solid var(--line); }
  .step-body { display:none; padding:12px 14px; border-top:1px solid var(--line); }
  .step.open .step-body { display:block; }
  pre.txt { margin:0; white-space:pre-wrap; word-break:break-word;
    font:12.5px/1.55 "SF Mono",Menlo,Consolas,monospace; color:var(--code); }
  .running { color:var(--muted); font-style:italic; }
  .tally { display:flex; flex-direction:column; gap:5px; }
  .tally-row { display:flex; align-items:center; gap:8px; }
  .tally-row .lab { width:70px; font-weight:700; }
  .tally-bar { height:16px; border-radius:4px; background:var(--vote); }
  /* tree-of-thoughts search tree */
  .tree { position:relative; overflow-x:auto; padding:6px 4px 2px; }
  .tree-rows { position:relative; z-index:1; }
  .tree-rowcap { font-size:10.5px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.6px; margin:14px 0 2px 2px; }
  .tree-rowcap:first-child { margin-top:2px; }
  .tree-row { display:flex; justify-content:center; gap:16px; margin:4px 0 6px;
    flex-wrap:wrap; }
  .tnode { position:relative; min-width:54px; padding:8px 13px; border-radius:9px;
    border:1px solid var(--line); background:var(--panel2); text-align:center;
    cursor:pointer; transition:.15s; }
  .tnode:hover { border-color:var(--hover); }
  .tnode .tnode-label { font-weight:700; font-size:15px; display:block; }
  .tnode .tnode-score { font-size:13px; color:var(--value); }  /* value = gold-orange */
  .tnode .tnode-ans { font-size:12px; color:var(--muted); }
  .tnode.pruned { opacity:.38; }
  .tnode.inflight { animation:livebar 1.2s ease-in-out infinite; }
  .tnode.final { border-color:var(--final); }
  .tnode.final.kept { box-shadow:0 0 0 1px var(--final) inset; }
  .tnode.root { background:var(--panel); border-color:var(--vote);
    color:var(--vote); }
  .tnode.sel { outline:2px solid #4b91ff; outline-offset:1px; }
  svg.tree-edges { position:absolute; left:0; top:0; pointer-events:none;
    z-index:0; overflow:visible; }
  .tree-detail { margin-top:6px; }
  .tree-detail pre.txt { background:var(--panel2); border:1px solid var(--line);
    border-radius:8px; padding:10px 12px; }
  /* summary comparison table */
  .sumtab { width:100%; border-collapse:collapse; font-size:13px; }
  .sumtab th { text-align:left; color:var(--muted); font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:.5px; padding:6px 10px;
    border-bottom:1px solid var(--line); }
  .sumtab td { padding:7px 10px; border-bottom:1px solid var(--line); }
  .sumtab td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .sumtab td.best { color:var(--ok); font-weight:700; }
  .sumtab tr:last-child td { border-bottom:0; }
  .sumtab .mlabel { font-weight:600; }
</style>
<script>
(function(){var K="demoTheme",r=document.documentElement;
 function lbl(t){var b=document.getElementById("themeBtn");if(b)b.textContent=(t==="light"?"\u{1F319} Dark":"☀ Light");}
 var s="dark";try{s=localStorage.getItem(K)||"dark";}catch(e){}
 r.setAttribute("data-theme",s);
 function set(t){r.setAttribute("data-theme",t);try{localStorage.setItem(K,t);}catch(e){}lbl(t);
   if(window.onThemeChange){try{window.onThemeChange(t);}catch(e){}}}
 document.addEventListener("DOMContentLoaded",function(){lbl(r.getAttribute("data-theme"));});
 document.addEventListener("click",function(e){if(e.target&&e.target.id==="themeBtn")
   set(r.getAttribute("data-theme")==="light"?"dark":"light");});
})();
</script>
</head>
<body>
<a class="home-btn" href="/">&larr; Main</a>
<button id="themeBtn" class="theme-toggle"></button>
<div class="hero"><div class="wrap" style="padding:30px 24px 26px">
  <h1><span class="dot-live"></span><span class="grad">Test-Time Reasoning</span>&nbsp;</h1>
  <p class="lede">Seven ways to spend more compute at inference, racing on the
    <b>same grade-school math problem</b>. Each method streams its reasoning
    <b>step by step</b> as it runs and watch how sampling, searching, refining, and
    tool-use change the answer.</p>
  <div class="sub" id="sub"></div>
</div></div>
<div class="wrap">
  <div class="card">
    <span class="pick-label">Pick a problem</span>
    <select id="prob"></select>
    <div class="q" id="question" style="margin-top:12px"></div>
  </div>
  <div id="boxes"></div>
  <div class="card" id="summary-card">
    <div class="section-h">Summary — latency &amp; tokens across the methods you've run</div>
    <div id="summary"><div class="empty">Run one or more methods to compare them here.</div></div>
  </div>
</div>
<script>
const PROBLEMS = __PROBLEMS__, MODEL = __MODEL__, DEF = __DEFAULTS__;
const STRATS = [
  {name:"input_output", label:"Input–output (direct)", param:null, knobLabel:null,
   accent:"--io"},
  {name:"self_consistency", label:"Self-consistency", param:"k", knobLabel:"k",
   def:DEF.k, min:1, max:16, accent:"--sample"},
  {name:"react", label:"ReAct", param:"max_steps", knobLabel:"steps",
   def:DEF.max_steps, min:1, max:10, accent:"--act"},
  {name:"agentic", label:"Agent (tool-use)", param:"max_steps", knobLabel:"steps",
   def:DEF.max_steps, min:1, max:10, accent:"--think"},
  {name:"iterative", label:"Self-Refine", param:"rounds", knobLabel:"rounds",
   def:DEF.rounds, min:0, max:6, accent:"--refine"},
  {name:"tree_of_thoughts", label:"Tree of Thoughts", param:"depth", knobLabel:"depth",
   def:DEF.depth, min:1, max:4, accent:"--propose"},
  {name:"fleet_of_agents", label:"Fleet of Agents", param:"n_agents", knobLabel:"agents",
   def:DEF.n_agents, min:2, max:8, accent:"--agent"},
];
const KINDVAR = {io:"--io",solve:"--solve",feedback:"--feedback",refine:"--refine",
  sample:"--sample",
  vote:"--vote",think:"--think",tool:"--tool",final:"--final",
  act:"--act",observe:"--observe",
  propose:"--propose",value:"--value",prune:"--prune",
  agent:"--propose",resample:"--resample"};   // propose=pink, resample=teal
// every strategy renders a node-link search tree (shapes differ: a single node,
// a chain, a fan-in, a branching beam, a resampling fleet)
const TREE_STRATS = ["input_output","self_consistency","react","agentic",
                     "iterative","tree_of_thoughts","fleet_of_agents"];
const TREE_NOTE = {
  input_output:"no search — one call straight from question to answer",
  iterative:"a chain: each round is a feedback step then a refine step",
  self_consistency:"K chains sampled in parallel, converging on a majority vote",
  react:"a chain: thought + action, then an observation, repeated until Finish",
  agentic:"a chain of reason → tool-call steps",
  tree_of_thoughts:"branches split by depth · highlighted = kept, faded = pruned",
  fleet_of_agents:"agents resample each step · highlighted = survived, faded = died",
};
// per-kind tree display: short node label + row caption
const TKIND = {
  root:    {disp:()=>"Q",                                  cap:"question"},
  io:      {disp:()=>"answer",                             cap:"direct answer"},
  solve:   {disp:()=>"solve",                              cap:"initial solution"},
  feedback:{disp:n=>n.label,                               cap:"feedback"},
  refine:  {disp:n=>n.label,                               cap:"refine"},
  sample:  {disp:n=>n.label.replace("chain ","#"),         cap:"sampled chains (parallel)"},
  vote:    {disp:()=>"vote",                               cap:"majority vote"},
  think:   {disp:()=>"reason",                             cap:"reason"},
  tool:    {disp:()=>"calc",                               cap:"calculator"},
  act:     {disp:()=>"act",                                cap:"thought + action"},
  observe: {disp:()=>"obs",                                cap:"observation"},
  propose: {disp:n=>n.label.replace(/^branch /,""),        cap:null},
  agent:   {disp:n=>n.label.replace(/^agent /,""),         cap:null},
  resample:{disp:n=>n.label.replace(/^agent /,""),         cap:null},
  final:   {disp:()=>"★ final",                            cap:"answer"},
};
const cssv = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const col = k => cssv(KINDVAR[k]||"--line");
const esc = s => (s==null?"":(""+s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

document.getElementById("sub").innerHTML =
  '<span class="chip">🧠 model: '+esc(MODEL)+'</span>'+
  '<span class="chip">⚡ streaming over SSE</span>'+
  '<span class="chip">'+PROBLEMS.length+' GSM8K problems</span>'+
  '<span class="chip">§2.1 · inference-time reasoning</span>'+
  '<a class="chip" href="https://llmreasoning.github.io" target="_blank" rel="noopener">'+
  'ACL 2026 · Current Advances in LLM Reasoning ↗</a>';
const sel = document.getElementById("prob");
PROBLEMS.forEach(p => { const o=document.createElement("option");
  o.value=p.id; o.textContent=p.id+" — "+p.q.slice(0,52)+"…"; sel.appendChild(o); });
function showQuestion(){ const p=PROBLEMS.find(x=>x.id===sel.value)||PROBLEMS[0];
  document.getElementById("question").textContent=p.q; }
sel.onchange=showQuestion; showQuestion();

// ---- build the stacked strategy boxes -------------------------------------
const ST={}, es={}, SUMMARY={};
const boxes=document.getElementById("boxes");
STRATS.forEach(c=>{
  const box=document.createElement("div");
  box.className="card strat"; box.setAttribute("data-strat",c.name);
  box.style.borderLeftColor=cssv(c.accent);
  const treeSection = TREE_STRATS.includes(c.name)
    ? '<div class="section-h">Search tree <i>('+(TREE_NOTE[c.name]||"")+
      ' · click a node)</i></div>'+
      '<div class="tree" id="tree-'+c.name+'"></div>'+
      '<div class="tree-detail" id="treedet-'+c.name+'"></div>'
    : '';
  const knobCtl = c.param
    ? '<label>'+c.knobLabel+' <input id="knob-'+c.name+'" type="number" value="'+c.def+
      '" min="'+c.min+'" max="'+c.max+'"></label>'
    : '<span class="chip">no knob — single call</span>';
  box.innerHTML=
    '<div class="strat-head"><span class="strat-title">'+esc(c.label)+'</span>'+
    knobCtl+
    '<button id="run-'+c.name+'">▶ Run</button>'+
    '<span class="spacer"></span><span class="status" id="status-'+c.name+'"></span></div>'+
    '<div id="stats-'+c.name+'"></div>'+ treeSection +
    '<div class="section-h">Execution timeline</div><div id="tl-'+c.name+'"></div>'+
    '<div class="section-h">Steps <i>(click to expand)</i></div>'+
    '<div id="steps-'+c.name+'"><div class="empty">Press Run to stream this strategy.</div></div>';
  boxes.appendChild(box);
  document.getElementById("run-"+c.name).onclick=()=>runStrategy(c);
});

function runStrategy(c){
  const name=c.name;
  if(es[name]) es[name].close();
  ST[name]={bars:{},order:[],stats:null,done:false,t0c:null,scale:4,selNode:null};
  renderStats(name); document.getElementById("tl-"+name).innerHTML="";
  const tEl=document.getElementById("tree-"+name); if(tEl) tEl.innerHTML="";
  const tdEl=document.getElementById("treedet-"+name); if(tdEl) tdEl.innerHTML="";
  document.getElementById("steps-"+name).innerHTML='<div class="empty">starting…</div>';
  const p=new URLSearchParams({strategy:name, id:sel.value});
  if(c.param) p.set(c.param, document.getElementById("knob-"+name).value);
  status(name,"connecting…",true);
  es[name]=new EventSource("/events?"+p.toString());
  es[name].onmessage=e=>onEvent(JSON.parse(e.data));
  es[name].onerror=()=>{ if(es[name].readyState===2) status(name,"disconnected",false); };
  ensureLoop();
}

// ---- auto-expand: every running step + the latest finished step -----------
function applyAutoOpen(name){
  const s=ST[name]; let lf=null, lfT=-1;
  s.order.forEach(id=>{ const b=s.bars[id]; if(b.t1!=null && b.t1>=lfT){lfT=b.t1; lf=id;} });
  s.order.forEach(id=>{ const b=s.bars[id]; if(b.userToggled) return;
    b.open = (b.t1==null) || (""+id===""+lf); });
}

function elapsed(s){ return s.done||s.t0c==null?null:(performance.now()-s.t0c)/1000; }

function renderStats(name){
  const s=ST[name], el=document.getElementById("stats-"+name); if(!el) return;
  const st=(k,v,c="")=>'<div class="stat"><div class="k">'+k+'</div><div class="v '+c+'">'+v+'</div></div>';
  if(!s||(!s.stats && !s.order.length)){ el.innerHTML=""; return; }
  let h='<div class="stats">';
  if(s.stats){ h+=st("predicted",esc(s.stats.answer),s.stats.correct?"ok":"bad")
    +st("gold",esc(s.stats.gold))+st("latency",s.stats.latency_s.toFixed(2)+"s")
    +st("calls",s.stats.calls)+st("tokens",s.stats.tokens)+st("steps",s.order.length); }
  else { h+=st("status",'<span class="live-dot"></span>running')+st("steps so far",s.order.length); }
  el.innerHTML=h+'</div>';
}

function renderTL(name){
  const s=ST[name], el=document.getElementById("tl-"+name); if(!el||!s) return;
  const ela=elapsed(s);
  let needed=s.stats?s.stats.latency_s:0;
  s.order.forEach(id=>{ const b=s.bars[id]; needed=Math.max(needed, b.t1==null?(ela||0):b.t1); });
  needed=Math.max(needed,0.5);
  // Stepped, monotonic scale: time flows left->right; the in-flight bar's right
  // edge advances as time passes (the axis only jumps in coarse steps).
  while(needed>s.scale) s.scale*=1.5;
  const maxT=s.scale;
  let rows="";
  s.order.forEach(id=>{ const b=s.bars[id];
    const end=b.t1==null?(ela||b.t0):b.t1;
    const left=100*b.t0/maxT, w=Math.max(100*(end-b.t0)/maxT,0.6);
    rows+='<div class="tl-row"><div class="tl-label">'+esc(b.label)+'</div>'+
      '<div class="tl-track"><div class="tl-bar'+(b.t1==null?" inflight":"")+'" data-t0="'+b.t0+'" '+
      'style="left:'+left+'%;width:'+w+'%;background:'+col(b.kind)+'"></div></div></div>'; });
  let note="";
  if(s.done && s.order.length){ let work=0; s.order.forEach(id=>{const b=s.bars[id]; if(b.t1!=null) work+=b.t1-b.t0;});
    const sp=work/Math.max(s.stats.latency_s,1e-6);
    note=sp>1.5?'Σ step work = '+work.toFixed(1)+'s but wall-clock = '+s.stats.latency_s.toFixed(1)
      +'s → ≈ ×'+sp.toFixed(1)+' parallel speedup.'
      :'Steps run sequentially: each bar starts where the previous ends.'; }
  el.innerHTML=(rows||'<div class="empty">no steps yet</div>')+
    '<div class="tl-axis"><span>0s</span><span>'+(maxT/2).toFixed(1)+'s</span><span>'
    +maxT.toFixed(1)+'s</span></div>'+(note?'<div class="note">'+note+'</div>':'');
}

// Per-frame: stretch in-flight bars to the current elapsed time WITHOUT rebuilding
// the DOM, so the CSS width transition shows a smooth left->right fill. Falls back
// to a full redraw only when the (coarse, stepped) time scale actually jumps.
function growInflight(name){
  const s=ST[name]; const ela=elapsed(s); if(ela==null) return;
  let needed=s.stats?s.stats.latency_s:0;
  s.order.forEach(id=>{ const b=s.bars[id]; needed=Math.max(needed, b.t1==null?ela:b.t1); });
  const old=s.scale; while(needed>s.scale) s.scale*=1.5;
  if(s.scale!==old){ renderTL(name); return; }     // scale jumped -> reposition all bars
  const el=document.getElementById("tl-"+name); if(!el) return;
  el.querySelectorAll(".tl-bar.inflight").forEach(bar=>{
    const t0=parseFloat(bar.getAttribute("data-t0"));
    bar.style.left=(100*t0/s.scale)+"%";
    bar.style.width=Math.max(100*(ela-t0)/s.scale,0.6)+"%";
  });
}

function tally(meta){ const t=meta.tally||{}, mx=Math.max(1,...Object.values(t));
  return '<div class="tally">'+Object.entries(t).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
    '<div class="tally-row"><span class="lab">'+esc(k)+'</span><div class="tally-bar" style="width:'
    +(180*v/mx)+'px"></div><span>'+v+'</span></div>').join("")+'</div>'; }

function renderSteps(name){
  const s=ST[name], el=document.getElementById("steps-"+name); if(!el||!s) return;
  if(!s.order.length){ el.innerHTML='<div class="empty">no steps yet</div>'; return; }
  el.innerHTML=s.order.map(id=>{ const b=s.bars[id], c=col(b.kind);
    const ans=(b.meta&&b.meta.answer!=null)?'<span class="ans" style="color:'+c+'">'+esc(b.meta.answer)+'</span>':'';
    let body; if(b.t1==null) body='<span class="running"><span class="live-dot"></span>running…</span>';
      else if(b.kind==="vote") body=tally(b.meta); else body='<pre class="txt">'+esc(b.text)+'</pre>';
    const time=b.t0.toFixed(2)+(b.t1==null?"–…":"–"+b.t1.toFixed(2))+"s";
    return '<div class="step'+(b.open?" open":"")+'" data-sid="'+id+'" style="border-left-color:'+c+'">'+
      '<div class="step-h"><span class="badge" style="background:'+c+'">'+esc(b.kind)+'</span>'+
      '<span class="step-title">'+esc(b.label)+'</span><span class="step-meta">'+
      (b.tokens?'<span>'+b.tokens+' tok</span>':'')+'<span>'+time+'</span>'+ans+'</span></div>'+
      '<div class="step-body">'+body+'</div></div>'; }).join("");
}

// ---- tree-of-thoughts: reconstruct the search tree from the step stream and
// ---- draw it as a node-link diagram (root -> branches per depth -> final). ---
function buildTree(s){
  const nodes={root:{id:"root",depth:-1,parent:null,parents:null,kind:"root",
    label:"Q",score:null,answer:null,kept:true,inflight:false,scoring:false,text:""}};
  const keptByDepth={};                              // ToT prune markers
  const resParents=new Set(), resDepths=new Set();   // FoA resampling
  s.order.forEach(id=>{ const b=s.bars[id], m=b.meta||{};
    if(b.kind==="value" && m.node){ const n=nodes[m.node];   // annotates a node
      if(n){ if(m.score!=null) n.score=m.score; n.scoring=(b.t1==null); } return; }
    if(b.kind==="prune"){ keptByDepth[m.depth]=new Set(m.kept||[]); return; }
    if(b.kind==="resample"){                 // FoA: a whole row of survivor nodes
      (m.nodes||[]).forEach(rn=>{
        nodes[rn.node]={id:rn.node,depth:m.depth,parent:rn.parent,parents:null,
          kind:"resample",label:"agent "+rn.name,score:(rn.score!=null?rn.score:null),
          answer:null,kept:true,inflight:false,scoring:false,text:""};
        resParents.add(rn.parent); });
      resDepths.add(m.depth); return; }
    if(m.node){                              // any other kind = one tree node
      const prunable=(b.kind==="propose"||b.kind==="agent");
      nodes[m.node]={id:m.node,depth:m.depth,parent:(m.parent!=null?m.parent:null),
        parents:(m.parents||null),kind:b.kind,label:b.label,
        score:(m.score!=null?m.score:null),answer:(m.answer!=null?m.answer:null),
        kept:(prunable?null:true),inflight:b.t1==null,scoring:false,text:b.text||""};
    }});
  Object.values(nodes).forEach(n=>{
    if(n.kind==="propose"){ const ks=keptByDepth[n.depth]; n.kept = ks?ks.has(n.id):null; }
    else if(n.kind==="agent"){       // faded once its resample row exists and it wasn't sampled
      n.kept = resDepths.has(n.depth+1) ? resParents.has(n.id) : null; }
  });
  return nodes;
}

function renderTree(name){
  const el=document.getElementById("tree-"+name); const s=ST[name];
  if(!el||!s) return;
  const nodes=buildTree(s);
  if(Object.keys(nodes).length<=1){
    el.innerHTML='<div class="empty">the tree fills in as branches are proposed…</div>'; return; }
  const rows={}; Object.values(nodes).forEach(n=>{ (rows[n.depth]=rows[n.depth]||[]).push(n); });
  const depths=Object.keys(rows).map(Number).sort((a,b)=>a-b);
  const disp=n=>(TKIND[n.kind]?TKIND[n.kind].disp(n):n.label);
  const finalNode=Object.values(nodes).find(n=>n.kind==="final");
  const finalDepth=finalNode?finalNode.depth:Infinity;
  const cap=d=>{
    if(d===-1) return "question";
    if(d===finalDepth) return "answer";
    if(name==="tree_of_thoughts") return "depth "+(d+1);
    if(name==="fleet_of_agents") return d%2===0 ? ("step "+(d/2+1)+" · propose + score")
                                                : ("step "+((d-1)/2+1)+" · resample");
    const k=(rows[d][0]||{}).kind;               // chain / sampling strategies
    return (TKIND[k]&&TKIND[k].cap) || ("step "+(d+1));
  };
  let html='<svg class="tree-edges"></svg><div class="tree-rows">';
  depths.forEach(d=>{ const ns=rows[d].slice().sort((a,b)=> a.id<b.id?-1:1);
    html+='<div class="tree-rowcap">'+esc(cap(d))+'</div>';
    html+='<div class="tree-row">'+ns.map(n=>{
      const active=n.kept!==false;
      const kc=n.kind==="root"?cssv("--vote"):col(n.kind);
      const style=active?('border-color:'+kc+';box-shadow:0 0 0 1px '+kc+' inset'):'';
      const cls=['tnode',n.kind, n.kept===false?'pruned':'',
        (n.inflight||n.scoring)?'inflight':'',
        (s.selNode===n.id?'sel':'')].join(' ').replace(/\s+/g,' ').trim();
      // sub-line: gold value score for scored strategies, else the running answer
      const sub = n.score!=null ? '<span class="tnode-score">★'+n.score+'</span>'
        : n.scoring ? '<span class="tnode-score">scoring…</span>'
        : (n.answer!=null && n.kind!=="root") ? '<span class="tnode-ans">→ '+esc(n.answer)+'</span>'
        : '';
      return '<div class="'+cls+'" style="'+style+'" data-node="'+esc(n.id)+'" data-strat="'+name+'">'+
        '<span class="tnode-label">'+esc(disp(n))+'</span>'+sub+'</div>';
    }).join('')+'</div>'; });
  html+='</div>';
  el.innerHTML=html;
  drawEdges(el,nodes);
}

function drawEdges(container,nodes){
  const svg=container.querySelector('svg.tree-edges'); if(!svg) return;
  const cr=container.getBoundingClientRect();
  const W=container.scrollWidth, H=container.scrollHeight;
  svg.setAttribute('width',W); svg.setAttribute('height',H);
  let paths='';
  Object.values(nodes).forEach(n=>{
    const plist=(n.parents&&n.parents.length)?n.parents:(n.parent?[n.parent]:[]);
    plist.forEach(pid=>{
      const ce=container.querySelector('.tnode[data-node="'+n.id+'"]');
      const pe=container.querySelector('.tnode[data-node="'+pid+'"]');
      if(!ce||!pe) return;
      const c=ce.getBoundingClientRect(), p=pe.getBoundingClientRect();
      const sx=container.scrollLeft, sy=container.scrollTop;
      const x1=p.left+p.width/2-cr.left+sx, y1=p.bottom-cr.top+sy;
      const x2=c.left+c.width/2-cr.left+sx, y2=c.top-cr.top+sy;
      const my=(y1+y2)/2;
      // edge takes the colour of the child node's kind (faded if it died)
      const ecol = n.kept===false ? cssv("--prune")
        : (n.kind==="root" ? cssv("--line") : col(n.kind));
      const dash = n.kept===false ? ' stroke-dasharray="3 3"' : '';
      paths+='<path d="M'+x1+' '+y1+' C '+x1+' '+my+', '+x2+' '+my+', '+x2+' '+y2+
        '" fill="none" stroke="'+ecol+'" stroke-width="1.6"'+dash+
        ' opacity="'+(n.kept===false?0.5:0.9)+'"/>';
    });
  });
  svg.innerHTML=paths;
}

// ---- end-of-run summary: compare latency & tokens across methods run -------
function renderSummary(){
  const el=document.getElementById("summary"); if(!el) return;
  const names=STRATS.map(c=>c.name).filter(n=>SUMMARY[n]);
  if(!names.length){ el.innerHTML='<div class="empty">Run one or more methods to compare them here.</div>'; return; }
  const minLat=Math.min(...names.map(n=>SUMMARY[n].latency_s));
  const minTok=Math.min(...names.map(n=>SUMMARY[n].tokens));
  let h='<table class="sumtab"><thead><tr><th>method</th><th>setting</th>'+
    '<th>latency</th><th>tokens</th><th>calls</th><th>answer</th><th>✓?</th></tr>'+
    '</thead><tbody>';
  names.forEach(n=>{ const r=SUMMARY[n];
    h+='<tr><td class="mlabel">'+esc(r.label)+'</td><td>'+esc(r.setting)+'</td>'+
      '<td class="num'+(r.latency_s===minLat?' best':'')+'">'+r.latency_s.toFixed(2)+'s</td>'+
      '<td class="num'+(r.tokens===minTok?' best':'')+'">'+r.tokens+'</td>'+
      '<td class="num">'+r.calls+'</td>'+
      '<td class="num">'+esc(r.answer)+'</td>'+
      '<td>'+(r.correct?'<span style="color:var(--ok)">✓</span>'
                       :'<span style="color:var(--bad)">✗</span>')+'</td></tr>';
  });
  el.innerHTML=h+'</tbody></table><div class="note">Fastest latency and fewest '+
    'tokens are highlighted; each row is the most recent run of that method.</div>';
}

// Delegated clicks survive the per-event re-renders; a manual toggle pins the
// step (userToggled) so auto-expand stops managing it.
document.getElementById("boxes").addEventListener("click", e=>{
  // clicking a tree node shows that branch's thought + score below the tree
  const tn=e.target.closest(".tnode");
  if(tn){ const name=tn.getAttribute("data-strat"), nid=tn.getAttribute("data-node");
    const s=ST[name]; if(!s) return;
    s.selNode = (s.selNode===nid? null : nid);
    let thought="", score="";
    s.order.forEach(id=>{ const b=s.bars[id], m=b.meta||{};
      if(m.node===nid && (b.kind==="propose"||b.kind==="agent"||b.kind==="final")) thought=b.text||thought;
      if(m.node===nid && b.kind==="value") score=b.text||score; });
    const det=document.getElementById("treedet-"+name);
    if(det) det.innerHTML = s.selNode
      ? '<pre class="txt"><b>'+esc(nid)+'</b>\n\n'+esc(thought||"(no reasoning yet)")+
        (score?'\n\n'+esc(score):"")+'</pre>'
      : "";
    renderTree(name); return; }
  const h=e.target.closest(".step-h"); if(!h) return;
  const card=h.parentNode, box=h.closest(".strat");
  const name=box.getAttribute("data-strat"), sid=card.getAttribute("data-sid");
  const b=ST[name]&&ST[name].bars[sid]; if(!b) return;
  b.open=!b.open; b.userToggled=true; card.classList.toggle("open",b.open);
});
// node positions shift when the window reflows -> redraw the tree edges
window.addEventListener("resize", ()=>{
  TREE_STRATS.forEach(n=>{ if(ST[n]) renderTree(n); }); });

function status(name,t,live){ const el=document.getElementById("status-"+name);
  if(el) el.innerHTML=(live?'<span class="live-dot"></span>':'')+esc(t); }

function onEvent(ev){
  const name=ev.strategy;
  if(ev.type==="meta"){ return; }
  if(ev.type==="error"){ status(name,"error",false);
    document.getElementById("steps-"+name).innerHTML=
      '<div class="empty" style="color:var(--bad)">Backend error: '+esc(ev.message)+'</div>';
    if(es[name]) es[name].close(); return; }
  if(ev.type==="all_done"){ status(name,"done",false); if(es[name]) es[name].close(); return; }
  if(ev.type==="trace_start"){
    if(!ST[name]) ST[name]={bars:{},order:[],stats:null,done:false,t0c:null,scale:4};
    ST[name].t0c=performance.now(); status(name,"streaming",true); ensureLoop(); return; }
  const s=ST[name]; if(!s) return;
  if(ev.type==="call_start"){
    if(!s.bars[ev.sid]){ s.bars[ev.sid]={sid:ev.sid,kind:ev.kind,label:ev.label,
      t0:ev.t0,t1:null,text:"",tokens:0,
      meta:{node:ev.node,parent:ev.parent,depth:ev.depth}}; s.order.push(ev.sid); }
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name);
    renderTree(name); return; }
  if(ev.type==="step"){ let b=s.bars[ev.sid];
    if(!b){ b={sid:ev.sid,userToggled:false}; s.bars[ev.sid]=b; s.order.push(ev.sid); }
    Object.assign(b,{kind:ev.kind,label:ev.label,t0:ev.t0,t1:ev.t1,
      text:ev.text,tokens:ev.tokens,meta:ev.meta});
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name);
    renderTree(name); return; }
  if(ev.type==="trace_done"){ s.stats={answer:ev.answer,correct:ev.correct,gold:ev.gold,
    latency_s:ev.latency_s,tokens:ev.tokens,calls:ev.calls}; s.done=true;
    const c=STRATS.find(x=>x.name===name), knob=document.getElementById("knob-"+name);
    SUMMARY[name]={label:c?c.label:name,
      setting:(c&&!c.param)?"direct":((c?c.knobLabel:"")+"="+(knob?knob.value:"?")),
      latency_s:ev.latency_s, tokens:ev.tokens, calls:ev.calls,
      answer:ev.answer, correct:ev.correct};
    renderSummary();
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name);
    renderTree(name); return; }
}

let raf=null;
function ensureLoop(){ if(!raf) raf=requestAnimationFrame(loop); }
function loop(){ let running=false;
  STRATS.forEach(c=>{ const s=ST[c.name]; if(s&&!s.done&&s.t0c!=null){ running=true; growInflight(c.name); } });
  raf = running ? requestAnimationFrame(loop) : null;
}
// re-render already-drawn SVG (timeline bars, search trees) so the colours,
// which are read from CSS vars at draw time, follow a dark<->light toggle.
window.onThemeChange = function(){
  STRATS.forEach(c=>{ const s=ST[c.name]; if(!s) return;
    renderStats(c.name); renderTL(c.name); renderSteps(c.name);
    if(TREE_STRATS.includes(c.name)) renderTree(c.name); });
  renderSummary();
};
</script></body></html>
"""
