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
from .strategies import agentic, iterative, self_consistency


def _title(name, k, rounds, max_steps) -> str:
    return {"iterative": f"Iterative · {rounds} refine round(s)",
            "self_consistency": f"Self-consistency · k={k}",
            "agentic": f"Agentic · ≤{max_steps} steps"}[name]


def _run(llm, prob, q, names, k, rounds, max_steps):
    """Worker thread: run the requested strategies, pushing events onto `q`.

    Each /events connection runs ONE strategy (its own Run button), so two
    strategies can stream concurrently over separate connections.
    """
    plan = {
        "iterative": (iterative, {"rounds": rounds}),
        "self_consistency": (self_consistency, {"k": k}),
        "agentic": (agentic, {"max_steps": max_steps}),
    }
    one = names[0] if len(names) == 1 else None
    try:
        q.put({"type": "meta", "question": prob.question, "gold": prob.answer})
        for name in names:
            fn, kwargs = plan[name]
            q.put({"type": "trace_start", "strategy": name,
                   "title": _title(name, k, rounds, max_steps)})
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
        probs = [{"id": p.id, "q": p.question} for p in self.server.problems]
        page = (PAGE
                .replace("__PROBLEMS__", json.dumps(probs))
                .replace("__MODEL__", json.dumps(self.server.llm.model))
                .replace("__DEFAULTS__", json.dumps(self.server.defaults)))
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _events(self, qs):
        def g(name, default):
            try:
                return int(qs.get(name, [default])[0])
            except (ValueError, TypeError):
                return default

        pid = qs.get("id", [None])[0]
        prob = next((p for p in self.server.problems if p.id == pid),
                    self.server.problems[0])
        k, rounds, max_steps = g("k", 5), g("rounds", 2), g("max_steps", 6)
        valid = ("iterative", "self_consistency", "agentic")
        strat = qs.get("strategy", [None])[0]
        names = [strat] if strat in valid else list(valid)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        ev_q: queue.Queue = queue.Queue()
        threading.Thread(target=_run,
                         args=(self.server.llm, prob, ev_q, names, k, rounds, max_steps),
                         daemon=True).start()
        while True:
            ev = ev_q.get()
            if ev is None:
                break
            try:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break  # browser navigated away / closed the stream


def serve(model: str | None = None, host: str = "127.0.0.1", port: int = 8000,
          defaults: dict | None = None) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.llm = LLM(model=model)
    httpd.problems = load_problems()
    httpd.defaults = defaults or {"k": 5, "rounds": 2, "max_steps": 6}
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
    --solve:#58a6ff; --refine:#bc8cff; --sample:#39c5cf; --vote:#e3b341;
    --think:#ffa657; --tool:#56d364; --final:#ff7b72;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:16px; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px;
    background:var(--panel2); border:1px solid var(--line); font-size:12px;
    color:var(--muted); margin-right:6px; }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; margin-bottom:16px; }
  select, input[type=number] { background:var(--panel2); color:var(--text);
    border:1px solid var(--line); border-radius:8px; padding:7px 9px; font:inherit; }
  input[type=number] { width:60px; }
  label { color:var(--muted); font-size:12px; }
  button { background:#238636; color:#fff; border:0; border-radius:8px;
    padding:8px 16px; font:inherit; font-weight:700; cursor:pointer; }
  button:hover { background:#2ea043; } button:disabled { opacity:.5; cursor:default; }
  .q b { color:var(--vote); }
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
    padding:2px 8px; border-radius:6px; color:#0d1117; }
  .step-title { font-weight:600; } .step-meta { margin-left:auto; color:var(--muted);
    font-size:12px; display:flex; gap:12px; align-items:center; }
  .ans { padding:2px 8px; border-radius:6px; font-weight:700; font-size:12px;
    background:var(--panel2); border:1px solid var(--line); }
  .step-body { display:none; padding:12px 14px; border-top:1px solid var(--line); }
  .step.open .step-body { display:block; }
  pre.txt { margin:0; white-space:pre-wrap; word-break:break-word;
    font:12.5px/1.55 "SF Mono",Menlo,Consolas,monospace; color:#cdd9e5; }
  .running { color:var(--muted); font-style:italic; }
  .tally { display:flex; flex-direction:column; gap:5px; }
  .tally-row { display:flex; align-items:center; gap:8px; }
  .tally-row .lab { width:70px; font-weight:700; }
  .tally-bar { height:16px; border-radius:4px; background:var(--vote); }
</style></head>
<body><div class="wrap">
  <h1>🔴 Live Reasoning Trace</h1>
  <div class="sub" id="sub"></div>
  <div class="card"><label>problem
    <select id="prob"></select></label>
    <div class="q" id="question" style="margin-top:10px"></div>
  </div>
  <div id="boxes"></div>
</div>
<script>
const PROBLEMS = __PROBLEMS__, MODEL = __MODEL__, DEF = __DEFAULTS__;
const STRATS = [
  {name:"iterative", label:"Iterative (self-refine)", param:"rounds", knobLabel:"rounds",
   def:DEF.rounds, min:0, max:6, accent:"--solve"},
  {name:"self_consistency", label:"Self-consistency", param:"k", knobLabel:"k",
   def:DEF.k, min:1, max:16, accent:"--sample"},
  {name:"agentic", label:"Agentic (tool use)", param:"max_steps", knobLabel:"steps",
   def:DEF.max_steps, min:1, max:10, accent:"--think"},
];
const KINDVAR = {solve:"--solve",refine:"--refine",sample:"--sample",
  vote:"--vote",think:"--think",tool:"--tool",final:"--final"};
const cssv = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const col = k => cssv(KINDVAR[k]||"--line");
const esc = s => (s==null?"":(""+s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

document.getElementById("sub").innerHTML =
  '<span class="chip">model: '+esc(MODEL)+'</span><span class="chip">streaming via SSE</span>';
const sel = document.getElementById("prob");
PROBLEMS.forEach(p => { const o=document.createElement("option");
  o.value=p.id; o.textContent=p.id+" — "+p.q.slice(0,52)+"…"; sel.appendChild(o); });
function showQuestion(){ const p=PROBLEMS.find(x=>x.id===sel.value)||PROBLEMS[0];
  document.getElementById("question").textContent=p.q; }
sel.onchange=showQuestion; showQuestion();

// ---- build the three stacked strategy boxes -------------------------------
const ST={}, es={};
const boxes=document.getElementById("boxes");
STRATS.forEach(c=>{
  const box=document.createElement("div");
  box.className="card strat"; box.setAttribute("data-strat",c.name);
  box.style.borderLeftColor=cssv(c.accent);
  box.innerHTML=
    '<div class="strat-head"><span class="strat-title">'+esc(c.label)+'</span>'+
    '<label>'+c.knobLabel+' <input id="knob-'+c.name+'" type="number" value="'+c.def+
      '" min="'+c.min+'" max="'+c.max+'"></label>'+
    '<button id="run-'+c.name+'">▶ Run</button>'+
    '<span class="spacer"></span><span class="status" id="status-'+c.name+'"></span></div>'+
    '<div id="stats-'+c.name+'"></div>'+
    '<div class="section-h">Execution timeline</div><div id="tl-'+c.name+'"></div>'+
    '<div class="section-h">Steps <i>(click to expand)</i></div>'+
    '<div id="steps-'+c.name+'"><div class="empty">Press Run to stream this strategy.</div></div>';
  boxes.appendChild(box);
  document.getElementById("run-"+c.name).onclick=()=>runStrategy(c);
});

function runStrategy(c){
  const name=c.name;
  if(es[name]) es[name].close();
  ST[name]={bars:{},order:[],stats:null,done:false,t0c:null,scale:4};
  renderStats(name); document.getElementById("tl-"+name).innerHTML="";
  document.getElementById("steps-"+name).innerHTML='<div class="empty">starting…</div>';
  const p=new URLSearchParams({strategy:name, id:sel.value});
  p.set(c.param, document.getElementById("knob-"+name).value);
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

// Delegated clicks survive the per-event re-renders; a manual toggle pins the
// step (userToggled) so auto-expand stops managing it.
document.getElementById("boxes").addEventListener("click", e=>{
  const h=e.target.closest(".step-h"); if(!h) return;
  const card=h.parentNode, box=h.closest(".strat");
  const name=box.getAttribute("data-strat"), sid=card.getAttribute("data-sid");
  const b=ST[name]&&ST[name].bars[sid]; if(!b) return;
  b.open=!b.open; b.userToggled=true; card.classList.toggle("open",b.open);
});

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
      t0:ev.t0,t1:null,text:"",tokens:0,meta:{}}; s.order.push(ev.sid); }
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name); return; }
  if(ev.type==="step"){ let b=s.bars[ev.sid];
    if(!b){ b={sid:ev.sid,userToggled:false}; s.bars[ev.sid]=b; s.order.push(ev.sid); }
    Object.assign(b,{kind:ev.kind,label:ev.label,t0:ev.t0,t1:ev.t1,
      text:ev.text,tokens:ev.tokens,meta:ev.meta});
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name); return; }
  if(ev.type==="trace_done"){ s.stats={answer:ev.answer,correct:ev.correct,gold:ev.gold,
    latency_s:ev.latency_s,tokens:ev.tokens,calls:ev.calls}; s.done=true;
    applyAutoOpen(name); renderStats(name); renderTL(name); renderSteps(name); return; }
}

let raf=null;
function ensureLoop(){ if(!raf) raf=requestAnimationFrame(loop); }
function loop(){ let running=false;
  STRATS.forEach(c=>{ const s=ST[c.name]; if(s&&!s.done&&s.t0c!=null){ running=true; growInflight(c.name); } });
  raf = running ? requestAnimationFrame(loop) : null;
}
</script></body></html>
"""
