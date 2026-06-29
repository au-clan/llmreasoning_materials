"""Render strategy traces to a standalone, interactive HTML trace explorer.

No server, no CDN, no build step -- one self-contained .html you can open from
the filesystem (or drop into slides). The centrepiece is a timeline that makes
the sequential-vs-parallel execution of each strategy visible at a glance.
"""
from __future__ import annotations

import json
import os

from .extract import is_correct


def _title(result) -> str:
    d = result.detail
    if result.strategy == "input_output":
        return "Input–output · direct answer (1 call)"
    if result.strategy == "iterative":
        n = d.get("rounds_run", d.get("rounds", 0))
        return f"Self-Refine · {n} feedback→refine round(s)"
    if result.strategy == "self_consistency":
        return f"Self-consistency · k={d.get('k')}"
    if result.strategy == "react":
        return f"ReAct · ≤{d.get('max_steps')} thought→action turns"
    if result.strategy == "agentic":
        return f"Agent (tool-use) · {d.get('tool_calls', 0)} tool call(s)"
    if result.strategy == "tree_of_thoughts":
        return (f"Tree of Thoughts · depth={d.get('depth')}, "
                f"breadth={d.get('breadth')}")
    if result.strategy == "fleet_of_agents":
        return (f"Fleet of Agents · {d.get('n_agents')} agents, "
                f"{d.get('steps')} steps")
    return result.strategy


def trace_dict(result, problem) -> dict:
    return {
        "strategy": result.strategy,
        "title": _title(result),
        "answer": result.answer,
        "gold": problem.answer,
        "correct": is_correct(result.answer, problem.answer),
        "latency_s": round(result.latency_s, 3),
        "tokens": result.usage.total_tokens,
        "calls": result.usage.calls,
        "steps": result.steps,
    }


def render(results, problem, model: str, out_path: str) -> str:
    payload = {
        "model": model,
        "question": problem.question,
        "gold": problem.answer,
        "traces": [trace_dict(r, problem) for r in results],
    }
    data = json.dumps(payload).replace("</", "<\\/")  # safe inside <script>
    html = _TEMPLATE.replace("__DATA__", data)
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reasoning Trace Explorer</title>
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
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin:0 0 4px; letter-spacing:.2px; }
  .sub { color:var(--muted); margin-bottom:18px; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px;
    background:var(--panel2); border:1px solid var(--line); font-size:12px;
    color:var(--muted); margin-right:6px; }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; margin-bottom:16px; }
  .q { font-size:15px; }
  .q b { color:var(--vote); }
  .tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
  .tab { cursor:pointer; padding:9px 14px; border-radius:10px; font-weight:600;
    background:var(--panel); border:1px solid var(--line); color:var(--muted);
    transition:.15s; }
  .tab:hover { color:var(--text); border-color:#3b4758; }
  .tab.active { color:var(--text); border-color:#4b91ff;
    box-shadow:0 0 0 1px #4b91ff inset; }
  .tab .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:7px; vertical-align:middle; }
  .stats { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  .stat { background:var(--panel2); border:1px solid var(--line);
    border-radius:10px; padding:8px 12px; min-width:96px; }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.5px; }
  .stat .v { font-size:18px; font-weight:700; margin-top:2px; }
  .v.ok { color:var(--ok); } .v.bad { color:var(--bad); }
  .section-h { font-size:12px; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:18px 0 8px; }
  /* timeline */
  .tl-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
  .tl-label { width:160px; flex:none; font-size:12px; color:var(--muted);
    text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tl-track { position:relative; flex:1; height:22px; background:var(--panel2);
    border-radius:6px; border:1px solid var(--line); }
  .tl-bar { position:absolute; top:2px; height:16px; border-radius:5px;
    min-width:3px; opacity:.9; cursor:pointer; transition:.1s; }
  .tl-bar:hover { opacity:1; outline:1px solid #fff3; }
  .tl-axis { display:flex; justify-content:space-between; color:var(--muted);
    font-size:11px; margin:6px 0 0 170px; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; font-style:italic; }
  /* steps */
  .step { border:1px solid var(--line); border-left:4px solid var(--line);
    border-radius:10px; margin:8px 0; background:var(--panel); overflow:hidden; }
  .step-h { display:flex; align-items:center; gap:10px; padding:10px 14px;
    cursor:pointer; }
  .step-h:hover { background:var(--panel2); }
  .badge { font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:.5px; padding:2px 8px; border-radius:6px; color:#0d1117; }
  .step-title { font-weight:600; }
  .step-meta { margin-left:auto; color:var(--muted); font-size:12px;
    display:flex; gap:12px; align-items:center; }
  .ans { padding:2px 8px; border-radius:6px; font-weight:700; font-size:12px;
    background:var(--panel2); border:1px solid var(--line); }
  .step-body { padding:0 14px; max-height:0; overflow:hidden; transition:.0s;
    border-top:0 solid var(--line); }
  .step.open .step-body { max-height:none; padding:12px 14px;
    border-top:1px solid var(--line); }
  pre.txt { margin:0; white-space:pre-wrap; word-break:break-word;
    font:12.5px/1.55 "SF Mono",Menlo,Consolas,monospace; color:#cdd9e5; }
  .tally { display:flex; flex-direction:column; gap:5px; margin-top:6px; }
  .tally-row { display:flex; align-items:center; gap:8px; }
  .tally-row .lab { width:70px; font-weight:700; }
  .tally-bar { height:16px; border-radius:4px; background:var(--vote); }
  .caret { color:var(--muted); transition:.15s; }
  .step.open .caret { transform:rotate(90deg); }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔍 Reasoning Trace Explorer</h1>
  <div class="sub" id="sub"></div>
  <div class="card q" id="question"></div>
  <div class="tabs" id="tabs"></div>
  <div id="view"></div>
  <div class="sub" style="margin-top:24px;font-size:12px">
    ACL'26 tutorial — input-output · self-consistency · ReAct · agent · self-refine · tree-of-thoughts · fleet-of-agents, under latency budgets
  </div>
</div>
<script>
const DATA = __DATA__;
const KIND = {
  io:"--io", solve:"--solve", feedback:"--feedback", refine:"--refine",
  sample:"--sample", vote:"--vote",
  think:"--think", tool:"--tool", final:"--final",
  act:"--act", observe:"--observe",
  propose:"--propose", value:"--value", prune:"--prune",
  agent:"--propose", resample:"--resample"
};
const css = k => getComputedStyle(document.documentElement).getPropertyValue(KIND[k]||"--line").trim();
const esc = s => (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

document.getElementById("sub").innerHTML =
  '<span class="chip">model: '+esc(DATA.model)+'</span>' +
  '<span class="chip">'+DATA.traces.length+' strategies</span>';
document.getElementById("question").innerHTML =
  esc(DATA.question) + '<br><br>gold answer: <b>'+esc(DATA.gold)+'</b>';

let active = 0;
const tabs = document.getElementById("tabs");
DATA.traces.forEach((t,i) => {
  const b = document.createElement("div");
  b.className = "tab" + (i===0?" active":"");
  const c = t.correct ? "var(--ok)" : "var(--bad)";
  b.innerHTML = '<span class="dot" style="background:'+c+'"></span>' +
    esc(t.strategy.replace("_"," ")) + (t.correct?" ✓":" ✗");
  b.onclick = () => { active=i; render();
    [...tabs.children].forEach((x,j)=>x.classList.toggle("active",j===i)); };
  tabs.appendChild(b);
});

function timeline(t) {
  const maxT = Math.max(t.latency_s, ...t.steps.map(s=>s.t1), 0.001);
  let rows = "";
  t.steps.forEach(s => {
    const left = 100*s.t0/maxT, w = Math.max(100*(s.t1-s.t0)/maxT, 0.6);
    const dur = (s.t1-s.t0).toFixed(2);
    rows += '<div class="tl-row"><div class="tl-label">'+esc(s.label)+'</div>' +
      '<div class="tl-track"><div class="tl-bar" title="'+esc(s.label)+
      ' · '+s.t0.toFixed(2)+'–'+s.t1.toFixed(2)+'s ('+dur+'s)" ' +
      'style="left:'+left+'%;width:'+w+'%;background:'+css(s.kind)+'" ' +
      'onclick="jump('+t.steps.indexOf(s)+')"></div></div></div>';
  });
  const work = t.steps.reduce((a,s)=>a+(s.t1-s.t0),0);
  const speed = work/Math.max(t.latency_s,1e-6);
  const note = speed>1.5
    ? 'Sum of step work = '+work.toFixed(1)+'s but wall-clock = '+t.latency_s.toFixed(1)
      +'s → ≈ ×'+speed.toFixed(1)+' parallel speedup (overlapping bars).'
    : 'Steps run sequentially: each bar starts where the previous ends.';
  return rows +
    '<div class="tl-axis"><span>0s</span><span>'+(maxT/2).toFixed(1)+
    's</span><span>'+maxT.toFixed(1)+'s</span></div>' +
    '<div class="note">'+note+'</div>';
}

function tally(meta) {
  const t = meta.tally || {}; const max = Math.max(1,...Object.values(t));
  return '<div class="tally">' + Object.entries(t).sort((a,b)=>b[1]-a[1]).map(
    ([k,v]) => '<div class="tally-row"><span class="lab">'+esc(k)+'</span>' +
    '<div class="tally-bar" style="width:'+(180*v/max)+'px"></div>' +
    '<span>'+v+'</span></div>').join("") + '</div>';
}

function steps(t) {
  return t.steps.map((s,i) => {
    const col = css(s.kind);
    const ans = (s.meta && s.meta.answer!=null)
      ? '<span class="ans" style="color:'+col+'">'+esc(s.meta.answer)+'</span>' : '';
    const body = (s.kind==="vote") ? tally(s.meta)
      : '<pre class="txt">'+esc(s.text)+'</pre>';
    return '<div class="step" id="step'+i+'" style="border-left-color:'+col+'">' +
      '<div class="step-h">' +
      '<span class="badge" style="background:'+col+'">'+esc(s.kind)+'</span>' +
      '<span class="step-title">'+esc(s.label)+'</span>' +
      '<span class="step-meta">' +
        (s.tokens?'<span>'+s.tokens+' tok</span>':'') +
        '<span>'+s.t0.toFixed(2)+'–'+s.t1.toFixed(2)+'s</span>' + ans +
        '<span class="caret">▶</span></span></div>' +
      '<div class="step-body">'+body+'</div></div>';
  }).join("");
}

function render() {
  const t = DATA.traces[active];
  const ansCls = t.correct ? "ok" : "bad";
  const stat = (k,v,cls="") => '<div class="stat"><div class="k">'+k+
    '</div><div class="v '+cls+'">'+v+'</div></div>';
  document.getElementById("view").innerHTML =
    '<div class="card"><div style="font-size:16px;font-weight:700;margin-bottom:12px">'+
      esc(t.title)+'</div><div class="stats">' +
      stat("predicted", esc(t.answer), ansCls) +
      stat("gold", esc(t.gold)) +
      stat("latency", t.latency_s.toFixed(2)+"s") +
      stat("model calls", t.calls) +
      stat("tokens", t.tokens) +
      stat("steps", t.steps.length) +
    '</div>' +
    '<div class="section-h">Execution timeline</div>'+timeline(t) +
    '<div class="section-h">Steps &nbsp;<span style="text-transform:none;font-style:italic">(click to expand)</span></div>' +
    steps(t) + '</div>';
}
function jump(i){ const el=document.getElementById("step"+i);
  el.classList.add("open"); el.scrollIntoView({behavior:"smooth",block:"center"}); }
// Delegated click so expanding a step keeps working after tab re-renders.
document.getElementById("view").addEventListener("click", e=>{
  const h=e.target.closest(".step-h"); if(h) h.parentNode.classList.toggle("open");
});
render();
</script>
</body>
</html>
"""
