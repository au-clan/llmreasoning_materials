"""Build `protocols_app.html` — an interactive version of the "same runs, four
scoring protocols" panel from `consistency.py` / the notebook.

What it adds over the static SVG:
  * a live **N** slider (how many of the ~30 reruns you can afford), driving
    mean@N / maj@N / pass@N;
  * a benchmark switcher;
  * honest *distributions*, not point estimates. "single run" is no longer the
    average-over-seeds (that's a different thing) — it's literally one randomly
    chosen rerun, shown as the empirical min–max range of the actual reruns. The
    grand mean is kept as a separate reference tick.

Definitions, per (model, benchmark) cell of S reruns x 50 problems (scores
thresholded at 0.5 for "solved"; humaneval is fractional so this is "≥ half its
unit tests"):
  single run  one random rerun's accuracy   → empirical range over the S reruns
  mean        average of all S reruns        → fixed reference
  mean@N      average of N sampled reruns     → centred on mean, band shrinks with N
  maj@N       sample N reruns; a problem counts if a strict majority solved it
  pass@N      sample N reruns; a problem counts if ≥1 solved it (the ceiling)

mean@N / maj@N / pass@N bands are 10–90% over seeded Monte-Carlo resamples
(without replacement), so they're stable as you drag. Self-contained: no CDN, no
network — embeds all the matrices as bitstrings and computes in the browser.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(  # repo root (../../.. from demo/instability/<file>)
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo.instability import consistency as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "protocols_app.html")


def build_payload() -> dict:
    data: dict = {}
    for b in C.benchmarks():
        cell = {}
        for m in C.models(b):
            M = C.matrix(m, b)
            if M.size == 0:
                continue
            solved = (M >= 0.5).astype(int)               # S x P
            cell[C.short(m)] = {
                "s": [round(float(x), 4) for x in M.mean(1)],      # per-rerun accuracy
                "m": ["".join(map(str, row)) for row in solved],   # bitstring per rerun
            }
        data[b] = cell
    return {"benchmarks": C.benchmarks(), "data": data}


# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Scoring protocols — same runs, different verdicts</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{ --ink:#e6edf3; --mut:#8b949e; --line:#2a3343;
         --bg:#0d1117; --panel:#161b22; --panel2:#1c2230;
         --mean:#58a6ff; --maj:#e3b341; --pass:#3fb950; --single:#30363d;
         --hero-top:#10151e; --row0:#0d1117; --row1:#11161f; --hover:#3b4758; }
  :root[data-theme="light"]{ --ink:#1d1d1f; --mut:#777; --line:#ececec;
         --bg:#ffffff; --panel:#fafafa; --panel2:#ffffff;
         --mean:#2563c9; --maj:#d98000; --pass:#1a9850; --single:#c8d2dc;
         --hero-top:#eef1f6; --row0:#ffffff; --row1:#fcfcfc; --hover:#c4c4c4; }
  *{box-sizing:border-box}
  body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       color:var(--ink);background:var(--bg);margin:0}
  .theme-toggle,.home-btn{position:fixed;right:16px;z-index:99;cursor:pointer;
    font:inherit;font-size:13px;padding:6px 12px;border-radius:8px;text-decoration:none;
    border:1px solid var(--line);background:var(--panel2);color:var(--ink)}
  .home-btn{top:14px} .theme-toggle{top:52px}
  .theme-toggle:hover,.home-btn:hover{border-color:var(--hover)}
  .wrap{max-width:1040px;padding:26px 30px}
  .hero{border-bottom:1px solid var(--line);background:
      radial-gradient(1100px 220px at 18% -60%, rgba(88,166,255,.12), transparent 60%),
      radial-gradient(900px 220px at 90% -80%, rgba(247,120,186,.10), transparent 60%),
      linear-gradient(180deg,var(--hero-top), var(--bg))}
  .hero .wrap{padding:30px 30px 22px}
  h1{font-size:24px;font-weight:800;letter-spacing:-.4px;margin:0 0 6px;
     background:linear-gradient(90deg,#58a6ff 0%,#bc8cff 48%,#f778ba 100%);
     -webkit-background-clip:text;background-clip:text;color:transparent}
  .sub{color:var(--mut);font-size:13.5px;margin:0;max-width:820px;line-height:1.55}
  .sub b{color:var(--ink)}
  .tutline{font-size:12px;color:var(--mut);margin-top:9px}
  .tutline a{color:#58a6ff;text-decoration:none}
  .tutline a:hover{text-decoration:underline}
  .controls{display:flex;align-items:center;gap:26px;flex-wrap:wrap;
            background:var(--panel);border:1px solid var(--line);border-radius:10px;
            padding:12px 18px;margin-bottom:8px}
  .ctl{display:flex;align-items:center;gap:10px;font-size:13px}
  .ctl label{color:var(--mut)}
  select{font-size:13px;padding:5px 9px;border:1px solid var(--line);border-radius:7px;
         background:var(--panel2);color:var(--ink)}
  input[type=range]{width:300px;accent-color:var(--mean)}
  #nval{font-variant-numeric:tabular-nums;font-weight:700;font-size:15px;min-width:2.4em}
  .nnote{color:var(--mut);font-size:12px}
  .legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);
          margin:10px 2px 4px}
  .legend span{display:inline-flex;align-items:center;gap:7px}
  .sw{width:24px;height:0;border-top-width:0;display:inline-block;position:relative}
  svg{display:block}
  .axislab{fill:var(--mut);font-size:10px}
  .grid{stroke:var(--line)}
  .mname{font-size:12.5px;fill:var(--ink)}
  .caption{font-size:12px;color:var(--mut);margin-top:12px;line-height:1.6;
           border-top:1px solid var(--line);padding-top:12px;max-width:880px}
  .caption b{color:#c9d1d9}
  .focus{background:rgba(88,166,255,.12)}
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
  <div class="hero"><div class="wrap">
  <h1>Same runs, four ways to score them</h1>
  <div class="sub">Each model was rerun ~30 times on the same 50 problems at
  identical settings. A benchmark reports <i>one</i> number; here are four, and how
  they evolve as you accumulate <b>N</b> reruns one at a time, in order. Drag N and watch
  <b>mean@N</b> walk from your first run toward the overall mean as the noise averages
  out, the run-to-run spread widen, and <b>pass@N</b> climb to the ceiling.</div>
  <div class="tutline">§1 · How well can models reason? — part of the
    <a href="https://llmreasoning.github.io" target="_blank" rel="noopener">ACL 2026 tutorial “Current Advances in LLM Reasoning”</a> ↗</div>
  </div></div>

  <div class="wrap">
  <div class="controls">
    <div class="ctl"><label>benchmark</label>
      <select id="bench"></select></div>
    <div class="ctl"><label>N reruns sampled</label>
      <input type="range" id="n" min="1" max="30" value="5">
      <span id="nval">5</span></div>
    <div class="ctl nnote" id="nnote"></div>
  </div>

  <div class="legend">
    <span><svg width="26" height="12"><rect x="0" y="3" width="24" height="6" rx="3" fill="var(--single)"/></svg> run-to-run spread (min–max of the N runs)</span>
    <span><svg width="26" height="12"><line x1="12" y1="1" x2="12" y2="11" stroke="#8b949e" stroke-dasharray="2 2"/></svg> overall mean (mean@N's limit)</span>
    <span><svg width="26" height="12"><circle cx="12" cy="6" r="4" fill="var(--mean)"/></svg> mean@N (running average of first N runs)</span>
    <span><svg width="26" height="12"><line x1="2" y1="6" x2="22" y2="6" stroke="var(--maj)" stroke-width="2"/><path d="M12,2 L16,6 L12,10 L8,6 Z" fill="var(--maj)"/></svg> maj@N</span>
    <span><svg width="26" height="12"><line x1="2" y1="6" x2="22" y2="6" stroke="var(--pass)" stroke-width="2"/><circle cx="12" cy="6" r="4.5" fill="none" stroke="var(--pass)" stroke-width="2"/></svg> pass@N (≥1 rerun)</span>
  </div>

  <div id="chart"></div>

  <div class="caption" id="cap"></div>
  </div>

<script>
const PAYLOAD = __DATA__;
const ROWH = 34, X0 = 168, XW = 690, TOP = 30;
const cssvar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// parse a model's bitstring matrix into Uint8 rows once
function prep(cell){
  const rows = cell.m.map(s => Uint8Array.from(s, c => c==='1'?1:0));
  return {seedAcc: cell.s, rows, S: rows.length, P: rows[0].length};
}

// Everything is cumulative over the FIRST n runs, taken in order (run 1, then 2, ...):
//   mean@N = running average of those n run-scores  (moves, converging to the overall mean)
//   maj@N  = per problem, did a majority of the n runs solve it? (ties count as half)
//   pass@N = per problem, did at least one of the n runs solve it?
//   obs    = min..max of those n run-scores (a point at n=1, widening as runs accrue)
// Deterministic — no resampling. Depends on run order, which is the actual log/seed order.
function stats(model, N){
  const S = model.S, P = model.P, n = Math.min(N, S);
  let sm=0, lo=2, hi=-1;
  const acc = new Int16Array(P);
  for(let i=0;i<n;i++){
    const a = model.seedAcc[i]; sm+=a; if(a<lo)lo=a; if(a>hi)hi=a;
    const row = model.rows[i]; for(let p=0;p<P;p++) acc[p]+=row[p];
  }
  let mj=0, ps=0;
  for(let p=0;p<P;p++){ const c=acc[p]; if(c*2>n) mj++; else if(c*2===n) mj+=0.5; if(c>=1) ps++; }
  const grand = model.seedAcc.reduce((a,b)=>a+b,0)/S;
  return {n, meanN: sm/n, grand, obsLo: lo, obsHi: hi, maj: mj/P, pass: ps/P};
}

let CUR = {bench:null, models:[], N:5};

function render(){
  const COL = {mean:cssvar('--mean'), maj:cssvar('--maj'), pass:cssvar('--pass'), single:cssvar('--single')};
  const ROW0 = cssvar('--row0'), ROW1 = cssvar('--row1'), MUT = cssvar('--mut');
  const cell = PAYLOAD.data[CUR.bench];
  const names = Object.keys(cell);
  const prepped = names.map(nm => prep(cell[nm]));
  const st = prepped.map(m => stats(m, CUR.N));
  // stable order by overall mean (doesn't reshuffle as you drag N)
  const ord = names.map((nm,i)=>i).sort((a,b)=> st[b].grand - st[a].grand);

  const H = TOP + ROWH*names.length + 24;
  const W = X0 + XW + 50;
  const X = v => X0 + v*XW;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  // gridlines
  for(let t=0;t<=10;t+=2){ const x=X(t/10);
    s += `<line class="grid" x1="${x}" y1="${TOP-6}" x2="${x}" y2="${TOP+ROWH*names.length}"/>`;
    s += `<text class="axislab" x="${x}" y="${TOP-12}" text-anchor="middle">${t*10}%</text>`; }

  ord.forEach((i,k)=>{
    const nm = names[i], d = st[i], y = TOP + k*ROWH + ROWH/2;
    s += `<rect x="0" y="${y-ROWH/2+1}" width="${W}" height="${ROWH-2}" fill="${k%2?ROW1:ROW0}"/>`;
    s += `<text class="mname" x="${X0-12}" y="${y+4}" text-anchor="end">${nm}</text>`;
    // run-to-run spread (behind) — min..max of the first n runs: a point at n=1, widening
    s += `<rect x="${X(d.obsLo)}" y="${y-7}" width="${Math.max(X(d.obsHi)-X(d.obsLo),1.5)}" height="14" rx="7" fill="${COL.single}"><title>runs 1–${d.n} scored ${(d.obsLo*100).toFixed(0)}–${(d.obsHi*100).toFixed(0)}%</title></rect>`;
    // dashed tick at the overall mean — the value mean@N is converging toward
    s += `<line x1="${X(d.grand)}" y1="${y-9}" x2="${X(d.grand)}" y2="${y+9}" stroke="${MUT}" stroke-dasharray="2 2"><title>overall mean (all ${d.n>0?'':''}reruns): ${(d.grand*100).toFixed(1)}%</title></line>`;
    // maj@N diamond
    const mx=X(d.maj);
    s += `<path d="M${mx},${y-5} L${mx+5},${y} L${mx},${y+5} L${mx-5},${y} Z" fill="${COL.maj}"><title>maj@${d.n} (majority vote of the first ${d.n} runs): ${(d.maj*100).toFixed(0)}%</title></path>`;
    // pass@N ring + label
    s += `<circle cx="${X(d.pass)}" cy="${y}" r="5" fill="none" stroke="${COL.pass}" stroke-width="2"><title>pass@${d.n} (any of the first ${d.n} runs): ${(d.pass*100).toFixed(0)}%</title></circle>`;
    s += `<text x="${X(d.pass)+9}" y="${y+4}" font-size="11" fill="${COL.pass}">${(d.pass*100).toFixed(0)}%</text>`;
    // mean@N dot — the running average, drawn on top
    s += `<circle cx="${X(d.meanN)}" cy="${y}" r="4.5" fill="${COL.mean}"><title>mean@${d.n} (running average of the first ${d.n} run${d.n>1?'s':''}): ${(d.meanN*100).toFixed(1)}%</title></circle>`;
  });
  s += `</svg>`;
  document.getElementById('chart').innerHTML = s;

  // caption: call out the model with the widest run-to-run spread so far
  let best=ord[0], gap=-1;
  ord.forEach(i=>{ const g = st[i].obsHi - st[i].obsLo; if(g>gap){gap=g; best=i;} });
  const d=st[best], one=d.n===1;
  document.getElementById('cap').innerHTML =
    `<b>Reading it.</b> A run = one full pass over the 50 problems. <b>mean@N</b> (blue dot) is the `+
    `running average of your first N runs — for <b>${names[best]}</b> on ${CUR.bench} it ` +
    (one
      ? `starts at run 1's <b>${(d.meanN*100).toFixed(0)}%</b>; add runs and it walks toward the dashed <b>${(d.grand*100).toFixed(0)}%</b> overall mean as the noise averages out. `
      : `has averaged ${d.n} runs to <b>${(d.meanN*100).toFixed(0)}%</b>, settling toward the dashed <b>${(d.grand*100).toFixed(0)}%</b> mean. `) +
    `The grey bar is the min–max of those runs (<b>${(d.obsLo*100).toFixed(0)}–${(d.obsHi*100).toFixed(0)}%</b>), and <b>pass@${d.n}</b> (green) reaches <b>${(d.pass*100).toFixed(0)}%</b> — solved if any one of the ${d.n} run${one?'':'s'} got it. `+
    `Same runs, same model — the number you report is a choice.`;
}

// ---- wire up controls ----
const benchSel=document.getElementById('bench'), nSlider=document.getElementById('n'),
      nVal=document.getElementById('nval'), nNote=document.getElementById('nnote');
PAYLOAD.benchmarks.forEach(b=>{ const o=document.createElement('option'); o.value=b; o.textContent=b; benchSel.appendChild(o); });
function maxSeeds(b){ return Math.max(...Object.values(PAYLOAD.data[b]).map(c=>c.s.length)); }
function setBench(b){ CUR.bench=b; const mx=maxSeeds(b); nSlider.max=mx;
  if(+nSlider.value>mx) nSlider.value=mx; setN(+nSlider.value); }
function setN(n){ CUR.N=n; nVal.textContent=n; nNote.textContent=`of up to ${nSlider.max} reruns`; render(); }
benchSel.onchange=e=>setBench(e.target.value);
nSlider.oninput=e=>setN(+e.target.value);
setBench(PAYLOAD.benchmarks.includes('game24')?'game24':PAYLOAD.benchmarks[0]);
benchSel.value=CUR.bench;
window.onThemeChange = render;
</script>
</body></html>
"""


def render_html_string() -> str:
    """Build the self-contained page as a string (no file written)."""
    payload = build_payload()
    return PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))


def main():
    html = render_html_string()
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
