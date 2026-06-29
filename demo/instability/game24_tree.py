"""Build a self-contained web page that draws all of a model's attempts at one
Game24 puzzle as a *combination tree*.

Each Game24 solution is a sequence of binary steps: pick two of the available
numbers, combine them with +-*/, and replace them with the result, until one
number is left (which should be 24). So every final expression decomposes into a
path through a tree whose **nodes are the numbers still on the board** and whose
**edges are the combine-steps**.

Overlaying all 20 of gpt-5-mini's attempts at [6 6 7 12] onto one tree shows the
instability structurally:
  * the 9 valid runs collapse into a tiny shared subtree (one solid green trunk),
  * the 11 failures each fabricate a number they weren't given (e.g. 6/6, 7/7) and
    so peel off as dashed-red branches at the exact step the board runs out.

This module reuses the log parsing/grading in ``instability.py`` and emits a single
static HTML file (D3 from a CDN, tree data embedded as JSON) — no server needed.
"""
from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(  # repo root (../../.. from demo/instability/<file>)
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo.instability.attempts import load_attempts

HERE = os.path.dirname(os.path.abspath(__file__))

_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_PRETTY = {"+": "+", "-": "−", "*": "×", "/": "÷"}


def _num(x):
    """Show 42.0 as 42, keep real fractions."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, float):
        return round(x, 4)
    return x


def steps_from_expr(expr: str):
    """Decompose an expression into ordered (a, b, op, result) combine-steps
    using a deterministic left-first post-order."""
    node = ast.parse(expr, mode="eval").body
    steps: list[tuple] = []

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp):
            a = ev(n.left)
            b = ev(n.right)
            r = eval(compile(ast.Expression(n), "<e>", "eval"), {"__builtins__": {}})
            steps.append((a, b, _OPS[type(n.op)], r))
            return r
        if isinstance(n, ast.UnaryOp):
            return eval(compile(ast.Expression(n), "<e>", "eval"), {"__builtins__": {}})
        raise ValueError(f"unsupported node {ast.dump(n)}")

    ev(node)
    return steps


def _state_list(avail: dict) -> list:
    out = []
    for k in sorted(avail):
        out += [_num(k)] * max(0, avail[k])
    return out


def build_data(model: str, puzzle, target: int = 24) -> dict:
    puzzle = list(puzzle)
    attempts = load_attempts(model, puzzle, target)

    counter = [0]

    def new_node(state, edge):
        counter[0] += 1
        return {
            "id": f"n{counter[0]}", "state": state, "edge": edge,
            "terminal": False, "value": None, "leafValid": None,
            "cheated": False, "runs": [], "leafRuns": [], "children": [],
        }

    root = new_node(_state_list({k: puzzle.count(k) for k in set(puzzle)}), None)
    by_path = {(): root}
    runs_out = []

    for a in attempts:
        if not a.expr:
            runs_out.append({"run": a.run, "expr": None, "valid": False,
                             "reason": a.reason, "nodePath": [root["id"]]})
            continue
        steps = steps_from_expr(a.expr)
        avail = {}
        for k in puzzle:
            avail[k] = avail.get(k, 0) + 1
        path, cur, cheated = (), root, False
        node_path = [root["id"]]
        for (x, y, op, r) in steps:
            need = {}
            need[x] = need.get(x, 0) + 1
            need[y] = need.get(y, 0) + 1
            step_legal = all(avail.get(k, 0) >= v for k, v in need.items())
            fabricated = sorted({_num(k) for k, v in need.items()
                                 if avail.get(k, 0) < v})
            was_cheated = cheated
            for k, v in need.items():
                avail[k] = avail.get(k, 0) - v
            avail[r] = avail.get(r, 0) + 1
            if not step_legal:
                cheated = True
            first_cheat = (not step_legal) and (not was_cheated)
            sig = f"{_num(x)}|{op}|{_num(y)}|{_num(r)}"
            new_path = path + (sig,)
            child = by_path.get(new_path)
            if child is None:
                child = new_node(_state_list(avail), {
                    "a": _num(x), "b": _num(y), "op": op,
                    "label": f"{_num(x)} {_PRETTY[op]} {_num(y)}",
                    "result": _num(r), "legal": step_legal,
                    "cheated": cheated, "firstCheat": first_cheat,
                    "fabricated": fabricated, "count": 0, "runs": [],
                })
                by_path[new_path] = child
                cur["children"].append(child)
            child["edge"]["count"] += 1
            child["edge"]["runs"].append(a.run)
            child["cheated"] = child["cheated"] or cheated
            cur, path = child, new_path
            node_path.append(child["id"])
        cur["terminal"] = True
        cur["value"] = _num(steps[-1][3]) if steps else None
        cur["leafValid"] = bool(a.valid)
        cur["leafRuns"].append(a.run)
        runs_out.append({"run": a.run, "expr": a.expr, "valid": bool(a.valid),
                         "reason": a.reason, "nodePath": node_path})

    return {
        "model": model, "puzzle": puzzle, "target": target,
        "nRuns": len(attempts), "nValid": sum(a.valid for a in attempts),
        "tree": root, "runs": runs_out,
    }


def render_html_string(model: str = "gpt-5-mini", puzzle=(6, 6, 7, 12),
                       target: int = 24) -> str:
    """Build the self-contained page as a string (no file written)."""
    data = build_data(model, puzzle, target)
    return _TEMPLATE.replace("__DATA__", json.dumps(data))


def render_html(model: str = "gpt-5-mini", puzzle=(6, 6, 7, 12), target: int = 24,
                out_path: str | None = None) -> str:
    html = render_html_string(model, puzzle, target)
    out_path = out_path or os.path.join(HERE, "game24_tree.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Game24 combination tree</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  :root { --green:#3fb950; --red:#f85149; --ink:#e6edf3; --muted:#8b949e;
          --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --line:#2a3343;
          --gold:#e3b341; --hero-top:#10151e; --edge:#56607a;
          --win-fill:#14271b; --lose-fill:#2a1719; --root-stroke:#6e7681;
          --step:#9aa4b2; --fab:#ff9b8f; --mono:#c9d1d9; --hover:#3b4758; }
  :root[data-theme="light"] {
          --green:#1a9850; --red:#c0392b; --ink:#1d1d1f; --muted:#777;
          --bg:#fbfbfb; --panel:#ffffff; --panel2:#f2f2f2; --line:#e3e3e3;
          --gold:#c8860a; --hero-top:#eef1f6; --edge:#bbb;
          --win-fill:#eaf6ec; --lose-fill:#fbecea; --root-stroke:#888;
          --step:#555; --fab:#8a0b0b; --mono:#444; --hover:#c4c4c4; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--ink); background:var(--bg); }
  .theme-toggle, .home-btn { position:fixed; right:16px; z-index:99; cursor:pointer;
    font:inherit; font-size:13px; padding:6px 12px; border-radius:8px; text-decoration:none;
    border:1px solid var(--line); background:var(--panel2); color:var(--ink); }
  .home-btn { top:14px; } .theme-toggle { top:52px; }
  .theme-toggle:hover, .home-btn:hover { border-color:var(--hover); }
  header { padding:26px 24px 14px; border-bottom:1px solid var(--line); background:
      radial-gradient(1100px 220px at 18% -60%, rgba(88,166,255,.12), transparent 60%),
      radial-gradient(900px 220px at 90% -80%, rgba(247,120,186,.10), transparent 60%),
      linear-gradient(180deg,var(--hero-top), var(--bg)); }
  h1 { font-size:24px; font-weight:800; letter-spacing:-.4px; margin:0 0 6px;
       background:linear-gradient(90deg,#58a6ff 0%,#bc8cff 48%,#f778ba 100%);
       -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub { color:var(--muted); font-size:13.5px; line-height:1.55; max-width:820px; }
  .sub b.g { color:var(--green); }
  .tutline { font-size:12px; color:var(--muted); margin-top:7px; }
  .tutline a { color:#58a6ff; text-decoration:none; }
  .tutline a:hover { text-decoration:underline; }
  .bar { padding:14px 24px 12px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  button { font:inherit; padding:7px 14px; border:1px solid var(--line); border-radius:8px;
           background:var(--panel2); color:var(--ink); cursor:pointer; transition:.1s; }
  button:hover { border-color:var(--hover); }
  .legend { font-size:12.5px; color:var(--muted); display:flex; gap:16px; align-items:center;
            flex-wrap:wrap; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .legend b.fab { color:var(--fab); }
  .swatch { width:22px; height:0; border-top:3px solid; }
  .caption { font-size:13.5px; min-height:20px; padding:0 24px 6px; color:var(--muted); }
  .caption b { color:var(--ink); }
  .caption .mono { font-family:ui-monospace,Menlo,Consolas,monospace; color:var(--mono); }
  #wrap { overflow:auto; padding:0 16px 30px; }
  .link { fill:none; stroke:var(--edge); }
  .link.cheat { stroke:var(--red); stroke-dasharray:5 4; }
  .link.lit { stroke:var(--gold); }
  .nodebox { stroke:var(--line); fill:var(--panel); rx:7; ry:7; }
  .nodebox.root { stroke:var(--root-stroke); fill:var(--panel2); }
  .nodebox.cheat { stroke:var(--red); stroke-dasharray:4 3; }
  .nodebox.win { stroke:var(--green); stroke-width:1.4px; fill:var(--win-fill); }
  .nodebox.lose { stroke:var(--red); stroke-width:1.4px; fill:var(--lose-fill); }
  .nlabel { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px;
            text-anchor:middle; dominant-baseline:middle; fill:var(--ink); }
  .nlabel .step { fill:var(--step); }
  .nlabel .stepfab { fill:var(--fab); font-weight:700; }
  .errnote { fill:var(--fab); font-size:11px; font-style:italic; text-anchor:middle;
             font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  .tip { position:fixed; pointer-events:none; background:var(--panel2); color:var(--ink);
         border:1px solid var(--line); padding:7px 10px;
         border-radius:8px; font-size:12.5px; max-width:300px; opacity:0; transition:opacity .1s;
         z-index:10; line-height:1.45; }
  .tip .ok { color:var(--green); } .tip .bad { color:var(--red); }
</style>
<script>
(function(){var K="demoTheme",r=document.documentElement;
 function lbl(t){var b=document.getElementById("themeBtn");if(b)b.textContent=(t==="light"?"\u{1F319} Dark":"☀ Light");}
 var s="dark";try{s=localStorage.getItem(K)||"dark";}catch(e){}
 r.setAttribute("data-theme",s);
 function set(t){r.setAttribute("data-theme",t);try{localStorage.setItem(K,t);}catch(e){}lbl(t);}
 document.addEventListener("DOMContentLoaded",function(){lbl(r.getAttribute("data-theme"));});
 document.addEventListener("click",function(e){if(e.target&&e.target.id==="themeBtn")
   set(r.getAttribute("data-theme")==="light"?"dark":"light");});
})();
</script>
</head>
<body>
<a class="home-btn" href="/">&larr; Main</a>
<button id="themeBtn" class="theme-toggle"></button>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="subtitle"></div>
  <div class="tutline">§1 · How well can models reason? — part of the
    <a href="https://llmreasoning.github.io" target="_blank" rel="noopener">ACL 2026 tutorial “Current Advances in LLM Reasoning”</a> ↗</div>
</header>
<div class="bar">
  <button id="stepcol">▸ Reveal step by step</button>
  <button id="play">▶ Replay the 20 runs</button>
  <button id="reset">Reset</button>
  <div class="legend">
    <span><span class="swatch" style="border-color:#bbb"></span>combine step (thicker = more runs)</span>
    <span><span class="swatch" style="border-color:var(--red);border-top-style:dashed"></span>fabricates a number (<b class="fab">bold label</b> = first)</span>
    <span><span class="swatch" style="border-color:var(--green)"></span>reaches 24 legally</span>
  </div>
</div>
<div class="caption" id="caption">Hover any edge or node. Press <b>Replay</b> to paint the runs one at a time.</div>
<div id="wrap"><svg id="svg"></svg></div>
<div class="tip" id="tip"></div>

<script>
const DATA = __DATA__;
const tip = document.getElementById('tip');
function showTip(html, ev){ tip.innerHTML=html; tip.style.opacity=1;
  tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+14)+'px'; }
function hideTip(){ tip.style.opacity=0; }

document.getElementById('title').textContent =
  `${DATA.model} · make ${DATA.target} from [${DATA.puzzle.join(' ')}] · ${DATA.nRuns} identical calls`;
document.getElementById('subtitle').innerHTML =
  `Each node = numbers still on the board · each edge = one combine step. ` +
  `<b class="g">${DATA.nValid}</b> / ${DATA.nRuns} runs reach 24 legally — ` +
  `the rest fabricate a number (dashed red).`;

const root = d3.hierarchy(DATA.tree);
const dx = 46, dy = 250;
const tree = d3.tree().nodeSize([dx, dy])
  .separation((a,b)=> (a.parent===b.parent?1.15:1.7));
tree(root);

let x0=Infinity, x1=-Infinity, y1=-Infinity;
root.each(d=>{ if(d.x>x1)x1=d.x; if(d.x<x0)x0=d.x; if(d.y>y1)y1=d.y; });
const margin={top:24,right:230,bottom:40,left:90};   // bottom roomy for the error note
const width = y1 + margin.left + margin.right;
const height = (x1-x0) + margin.top + margin.bottom;

const svg = d3.select('#svg').attr('width',width).attr('height',height)
  .attr('viewBox',[0,0,width,height]);
const g = svg.append('g').attr('transform',`translate(${margin.left},${margin.top-x0})`);

// links (keyed by child id so we can light them up during replay)
const wScale = c => Math.min(c*2.2, 20);          // thicker = more runs (proportional)
const link = g.append('g').selectAll('path').data(root.links()).join('path')
  .attr('class', d=> 'link'+(d.target.data.edge.cheated?' cheat':''))
  .attr('data-id', d=> d.target.data.id)
  .style('stroke-width', d=> wScale(d.target.data.edge.count)+'px')
  .attr('d', d3.linkHorizontal().x(d=>d.y).y(d=>d.x));

// edge hover (the step itself now lives in the child node label)
link.style('cursor','default').on('mouseover',(ev,d)=>{ const e=d.target.data.edge;
    let note='';
    if(e.firstCheat){ const f=e.fabricated.join(', ');
      note=`<br><span class="bad">⚠ fabricates ${f} — no ${f} left on the board</span>`; }
    else if(e.cheated){ note=`<br><span class="bad">⚠ downstream of a fabricated number</span>`; }
    showTip(`<b>${e.label} = ${e.result}</b><br>taken by ${e.count} run(s)`+note, ev); })
  .on('mousemove',ev=>showTip(tip.innerHTML,ev)).on('mouseout',hideTip);

// nodes
const node = g.append('g').selectAll('g').data(root.descendants()).join('g')
  .attr('transform', d=>`translate(${d.y},${d.x})`);
node.each(function(d){
  const data=d.data, isRoot=d.depth===0, gsel=d3.select(this);
  const tail = data.terminal ? (data.value + (data.leafValid?' ✓':' ✗')) : data.state.join(' ');
  // label: "<combine step> → <numbers on the board>"  (root has no step)
  const t = gsel.append('text').attr('class','nlabel');
  if(isRoot){
    t.text(data.state.join(' '));
  } else {
    t.append('tspan').attr('class', data.edge.firstCheat?'stepfab':'step').text(data.edge.label);
    t.append('tspan').text('  →  ' + tail);
  }
  const bb = t.node().getBBox();
  const w = Math.max(34, bb.width+18), h=24;
  let cls='nodebox';
  if(isRoot) cls+=' root';
  else if(data.terminal && data.leafValid) cls+=' win';
  else if(data.terminal && data.leafValid===false) cls+=' lose';
  else if(data.cheated) cls+=' cheat';
  gsel.insert('rect','text').attr('class',cls)
    .attr('x',-w/2).attr('y',-h/2).attr('width',w).attr('height',h);
  // annotate the FIRST step where this run fabricates a number
  if(!isRoot && data.edge.firstCheat){
    const fab=data.edge.fabricated;
    const msg = fab.length ? ('invents an extra '+fab.join(' & '))
                           : 'uses a number that is gone';
    gsel.append('text').attr('class','errnote').attr('y', h/2+14).text(msg);
  }
  gsel.datum(d);
});
node.style('cursor','default').on('mouseover',(ev,d)=>{
  const data=d.data;
  if(data.terminal){
    const runs=DATA.runs.filter(r=>r.nodePath[r.nodePath.length-1]===data.id);
    const ex=[...new Set(runs.map(r=>r.expr))].map(e=>`<span class="mono">${e}</span>`).join('<br>');
    showTip(`<b>${data.leafValid?'<span class="ok">valid — 24</span>':'<span class="bad">'+data.value+'</span>'}</b>`+
      `<br>${runs.length} run(s)<br>${ex}`+(data.leafValid?'':`<br><i>${runs[0]?.reason||''}</i>`), ev);
  } else {
    showTip(`<b>on the board:</b> ${data.state.join(' ')}`+(d.depth?'':'  (start)'), ev);
  }
}).on('mousemove',ev=>showTip(tip.innerHTML,ev)).on('mouseout',hideTip);

// ---- controls ----
const cap=document.getElementById('caption');
const DEFAULT_CAP='Hover any edge or node. <b>Reveal step by step</b> grows the tree one column at a time; <b>Replay</b> paints the runs one at a time.';
let timer=null;
function clearLit(){ link.classed('lit',false); }

// column-by-column reveal: each column = one further combine step from the left
const maxDepth = d3.max(root.descendants(), d=>d.depth);
let shown = 0;                                   // launch collapsed: no steps shown
function setShown(k, dur){
  if(dur===undefined) dur=300;
  shown=k;
  const op = (sel,fn)=>{ (dur ? sel.transition().duration(dur) : sel).style('opacity',fn); };
  op(node, d=> d.depth<=k?1:0);
  node.style('pointer-events', d=> d.depth<=k?null:'none');
  op(link, d=> d.target.depth<=k?1:0);
  link.style('pointer-events', d=> d.target.depth<=k?null:'none');
  // crop the canvas to just the revealed nodes, anchored top-left (no blank space)
  const shownNodes = root.descendants().filter(d=> d.depth<=k);
  let mnx=Infinity, mxx=-Infinity;
  shownNodes.forEach(d=>{ if(d.x<mnx)mnx=d.x; if(d.x>mxx)mxx=d.x; });
  const w = margin.left + k*dy + 215;
  const h = (mxx-mnx) + margin.top + margin.bottom;
  svg.attr('width',w).attr('height',h).attr('viewBox',[0,0,w,h]);
  g.attr('transform',`translate(${margin.left},${margin.top-mnx})`);
}
function stepCaption(k){
  cap.innerHTML = (k===0)
    ? 'The four starting numbers — no combine steps yet. Press <b>Next step</b> to grow the tree.'
    : `After <b>${k}</b> combine step${k>1?'s':''} (column ${k+1} of ${maxDepth+1}).`;
}
const stepBtn=document.getElementById('stepcol');
stepBtn.onclick=()=>{
  clearLit();
  const k = (shown>=maxDepth) ? 0 : shown+1;   // at the end, next press collapses to the start
  setShown(k);
  stepBtn.textContent = (k>=maxDepth) ? '↻ Start over' : '▸ Next step';
  stepCaption(k);
};

function reset(){ if(timer){clearTimeout(timer);timer=null;} clearLit(); setShown(maxDepth);
  stepBtn.textContent='↻ Start over'; cap.innerHTML=DEFAULT_CAP; }
document.getElementById('reset').onclick=reset;

// initial state: collapsed, nothing but the starting numbers
setShown(0, 0);
stepBtn.textContent='▸ Next step';
stepCaption(0);
document.getElementById('play').onclick=()=>{
  reset(); let i=0;
  const step=()=>{
    if(i>=DATA.runs.length){ cap.innerHTML += '  —  done.'; return; }
    const r=DATA.runs[i];
    link.classed('lit',false);
    link.filter(function(){ return r.nodePath.includes(this.getAttribute('data-id')); })
        .classed('lit',true);
    cap.innerHTML = `run ${r.run+1}/${DATA.nRuns}: `+
      (r.expr?`<span class="mono">${r.expr}</span> `:'<i>empty answer</i> ')+
      (r.valid?'<b style="color:var(--green)">✓ 24</b>'
              :`<b style="color:var(--red)">✗ ${r.reason}</b>`);
    i++; timer=setTimeout(step, 900);
  };
  step();
};
</script>
</body>
</html>
"""


def _count_nodes(n):
    return 1 + sum(_count_nodes(c) for c in n["children"])


if __name__ == "__main__":
    d = build_data("gpt-5-mini", (6, 6, 7, 12))
    p = render_html()
    print(f"wrote {p}")
    print(f"{d['nValid']}/{d['nRuns']} valid · {_count_nodes(d['tree'])} nodes")
