"""HTML for the viewer. Three pages sharing one nav bar and one stylesheet.

Kept separate from watch.py so the server stays readable: watch.py routes and
streams, this file renders.
"""

from __future__ import annotations

CSS = r"""
:root{--bg:#0f1115;--fg:#d8dee9;--dim:#707a8c;--acc:#8fbcbb;--warn:#ebcb8b;--err:#bf616a;--think:#8a7fb5;--line:#262b36}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0}
nav{position:sticky;top:0;z-index:9;background:#161922;border-bottom:1px solid var(--line);
    padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
nav a{color:var(--dim);text-decoration:none;padding:3px 9px;border-radius:4px}
nav a:hover{color:var(--fg)} nav a.on{color:#0f1115;background:var(--acc)}
nav .sp{flex:1} nav b{color:var(--acc)} nav span{color:var(--dim)}
label{color:var(--dim);cursor:pointer;user-select:none}
main{padding:12px 16px}
pre{background:#12151c;border-left:2px solid var(--line);margin:4px 0 8px;padding:8px 10px;
    overflow-x:auto;white-space:pre-wrap;word-break:break-word;color:#c3cad6}
"""


# Shared by every page: HTML escaping, LaTeX-to-text, markdown, Python highlighting.
COMMON_JS = r"""
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// The agent writes LaTeX in prose. Rather than ship a math engine, fold the
// commands it actually uses down to Unicode so the text reads normally.
const TEX={'\\times':'×','\\approx':'≈','\\ge':'≥','\\geq':'≥','\\le':'≤','\\leq':'≤',
 '\\mu':'µ','\\dots':'…','\\cdot':'·','\\pm':'±','\\ll':'≪','\\gg':'≫',
 '\\alpha':'α','\\beta':'β','\\sigma':'σ','\\Delta':'Δ','\\to':'→','\\neq':'≠'};
function detex(s){
  s=s.replace(/\$\$([\s\S]*?)\$\$/g,(m,x)=>x).replace(/\$([^$\n]+)\$/g,(m,x)=>x);
  for(let i=0;i<4;i++){
    s=s.replace(/\\(?:text|mathrm|mathit|mathbf|mathrel|operatorname)\s*\{([^{}]*)\}/g,'$1');
    s=s.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g,'($1)/($2)');
    s=s.replace(/\\lceil\s*([^\\]*?)\s*\\rceil/g,'⌈$1⌉');
    s=s.replace(/\\lfloor\s*([^\\]*?)\s*\\rfloor/g,'⌊$1⌋');
  }
  s=s.replace(/\\(left|right|!|;|:)/g,'').replace(/\\,/g,' ').replace(/\\quad|\\qquad/g,'  ');
  for(const k in TEX) s=s.split(k).join(TEX[k]);
  return s.replace(/\\(min|max|log|exp|sum|sqrt)\b/g,'$1').replace(/[{}]/g,'');
}

// Minimal Python highlighter: protect strings and comments, then colour keywords.
const PYKW=/\b(def|class|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|import|from|as|with|try|except|finally|raise|lambda|yield|pass|break|continue|global|nonlocal|assert|async|await|del)\b/g;
function hl(src){
  let out=esc(src), holes=[];
  // Private-use chars, not digits: a numeric placeholder gets rewritten by the
  // number pass below, which destroys the stashed string or comment.
  const stash=h=>{holes.push(h);return String.fromCharCode(0xE000+holes.length-1);};
  out=out.replace(/("{3}[\s\S]*?"{3}|'{3}[\s\S]*?'{3}|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/g,
                  m=>stash('<span class="s">'+m+'</span>'));
  out=out.replace(/(#[^\n]*)/g, m=>stash('<span class="c">'+m+'</span>'));
  out=out.replace(/(@[\w.]+)/g,'<span class="d">$1</span>');
  out=out.replace(PYKW,'<span class="k">$1</span>');
  out=out.replace(/\b(\d+\.?\d*(?:e-?\d+)?)\b/g,'<span class="n">$1</span>');
  // Placeholders nest: a quoted span stashed first can end up inside a comment
  // stashed second, so one pass leaves the inner ones unresolved.
  for(let i=0;i<6&&/[\uE000-\uF8FF]/.test(out);i++)
    out=out.replace(/[\uE000-\uF8FF]/g,m=>holes[m.charCodeAt(0)-0xE000]);
  return out;
}

// Markdown -> HTML. Input is agent-written, so escape before anything else.
function md(src){
  let s=esc(detex(src)), fences=[];
  s=s.replace(/```(\w*)\n([\s\S]*?)```/g,(m,lang,code)=>{
    const body=/^(py|python)$/i.test(lang)?hl(code.replace(/\n$/,'')):code.replace(/\n$/,'');
    fences.push('<pre class="code"><code>'+body+'</code></pre>');
    return String.fromCharCode(0xE000+fences.length-1);});
  const lines=s.split('\n'), out=[]; let list=null, tbl=false;
  const cl=()=>{if(list){out.push('</'+list+'>');list=null;}};
  const ct=()=>{if(tbl){out.push('</tbody></table>');tbl=false;}};
  for(let i=0;i<lines.length;i++){
    const l=lines[i];
    if(/^\s*\|.*\|\s*$/.test(l)&&/^\s*\|[\s:|-]+\|\s*$/.test(lines[i+1]||'')){
      cl(); out.push('<table><thead><tr>'+l.split('|').slice(1,-1).map(c=>'<th>'+c.trim()+'</th>').join('')+'</tr></thead><tbody>');
      tbl=true; i++; continue;}
    if(tbl){ if(/^\s*\|.*\|\s*$/.test(l)){
        out.push('<tr>'+l.split('|').slice(1,-1).map(c=>'<td>'+c.trim()+'</td>').join('')+'</tr>'); continue;} ct();}
    let m;
    if(m=l.match(/^(#{1,6})\s+(.*)$/)){cl();out.push('<h'+m[1].length+'>'+m[2]+'</h'+m[1].length+'>');continue;}
    if(/^\s*(-{3,}|\*{3,})\s*$/.test(l)){cl();out.push('<hr>');continue;}
    if(m=l.match(/^\s*[-*+]\s+(.*)$/)){if(list!=='ul'){cl();out.push('<ul>');list='ul';}out.push('<li>'+m[1]+'</li>');continue;}
    if(m=l.match(/^\s*\d+[.)]\s+(.*)$/)){if(list!=='ol'){cl();out.push('<ol>');list='ol';}out.push('<li>'+m[1]+'</li>');continue;}
    if(!l.trim()){cl();continue;}
    cl(); out.push('<p>'+l+'</p>');
  }
  cl(); ct();
  return out.join('\n')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>')
    .replace(/[\uE000-\uF8FF]/g,m=>fences[m.charCodeAt(0)-0xE000]||m);
}
"""

MD_CSS = r"""
.md h1,.md h2,.md h3{color:var(--acc);margin:12px 0 6px;font-size:15px}
.md h1{font-size:17px;border-bottom:1px solid var(--line);padding-bottom:4px}
.md h3{font-size:13px;color:var(--warn)}
.md p{margin:5px 0}
.md code{background:#1b1f28;padding:1px 4px;border-radius:3px;color:#c3cad6}
.md pre.code{margin:8px 0} .md pre.code code{background:none;padding:0}
.md table{border-collapse:collapse;margin:8px 0;font-size:12px}
.md th,.md td{border:1px solid var(--line);padding:3px 9px;text-align:left}
.md th{color:var(--dim);font-weight:normal;background:#161922}
.md ul,.md ol{margin:5px 0;padding-left:22px} .md li{margin:1px 0}
.md hr{border:none;border-top:1px solid var(--line);margin:10px 0}
.md strong{color:#fff} .md a{color:var(--acc)}
.k{color:#b48ead} .s{color:#a3be8c} .c{color:#616e88;font-style:italic}
.n{color:#d08770} .d{color:#ebcb8b}
"""

NAV = """<nav>
  <a href="/" id="n-log">log</a>
  <a href="/chart" id="n-chart">chart</a>
  <a href="/files" id="n-files">experiments</a>
  <span class="sp"></span>
  {extra}
</nav>"""


def _shell(title: str, nav_extra: str, body: str, script: str) -> str:
    return (
        "<!doctype html>\n<meta charset=\"utf-8\">"
        f"<title>kopt · {title}</title>\n<style>{CSS}{MD_CSS}{{extra_css}}</style>\n"
        + NAV.format(extra=nav_extra)
        + f"\n{body}\n<script>\n{COMMON_JS}\n"
        "document.querySelectorAll('nav a').forEach(a=>{"
        "if(a.getAttribute('href')===location.pathname)a.classList.add('on')});\n"
        f"{script}\n</script>\n"
    )


# ----------------------------------------------------------------- log page
LOG_CSS = r"""
.row{padding:1px 0;white-space:pre-wrap;word-break:break-word}
.txt{margin:6px 0} .meta{color:var(--dim)}
.iter{color:var(--warn);border-top:1px solid #2a2f3c;margin-top:14px;padding-top:8px}
details{margin:2px 0}
details>summary{cursor:pointer;list-style:none;color:var(--acc)}
details>summary::-webkit-details-marker{display:none}
details>summary:before{content:'▸ ';color:var(--dim)}
details[open]>summary:before{content:'▾ '}
details.think>summary{color:var(--think);font-style:italic}
details.bad>summary{color:var(--err)}
details pre{max-height:60vh}
.hide-read details.read{display:none}
#log{max-width:1100px}
"""

LOG_JS = r"""
const $=i=>document.getElementById(i), log=$('log');
let tools=0, logged=0, seen=new Set();
const near=()=>innerHeight+scrollY>=document.body.scrollHeight-60;
function put(el){const b=near();log.appendChild(el);if(b)scrollTo(0,document.body.scrollHeight);}
function row(cls,text){const d=document.createElement('div');d.className='row '+cls;d.textContent=text;put(d);}
function prose(text){const d=document.createElement('div');d.className='row txt md';d.innerHTML=md(text);put(d);}
function block(cls,summary,body,asMd){
  const d=document.createElement('details');d.className=cls;
  const s=document.createElement('summary');s.textContent=summary;d.appendChild(s);
  if(asMd){const w=document.createElement('div');w.className='md';w.innerHTML=md(body);d.appendChild(w);}
  else{const pre=document.createElement('pre');pre.textContent=body;d.appendChild(pre);}
  if($('expand').checked)d.open=true; put(d); return d;
}
const oneline=s=>String(s==null?'':s).replace(/\s+/g,' ').trim();
const nlines=s=>String(s==null?'':s).split('\n').length;
function gist(name,a){
  a=a||{}; const path=a.path||a.file_path||a.file||'';
  if(name==='bash')  return oneline(a.command);
  if(name==='write') return path+'  ('+nlines(a.content)+' lines)';
  if(name==='edit'){const m=String(a.input||'').match(/^\[([^\]#]+)/);return (m?m[1]:'')+'  (patch)';}
  if(name==='read')  return path;
  if(name==='task')  return (Array.isArray(a.tasks)?a.tasks.length:1)+' subagent(s)';
  if(name==='glob'||name==='grep') return oneline((a.pattern||'')+' '+path);
  return oneline(JSON.stringify(a)).slice(0,120);
}
function argBody(name,a){
  a=a||{};
  if(name==='bash')  return String(a.command||'');
  if(name==='edit')  return String(a.input||'');
  if(name==='write') return String(a.content||'');
  return JSON.stringify(a,null,2);
}
function assistant(m){
  for(const b of (m.content||[])){
    if(b.type==='thinking'){const s=String(b.thinking||'');if(s.trim())block('think','thinking · '+s.length+' chars',s,true);}
    else if(b.type==='text'){const s=String(b.text||'');if(s.trim())prose(s);}
    else if(b.type==='toolCall'){
      tools++;$('tools').textContent=tools;
      const d=block('tool'+(b.name==='read'?' read':''),'→ '+b.name+'  '+gist(b.name,b.arguments),argBody(b.name,b.arguments));
      if(b.name==='write'&&/\.py$/.test((b.arguments||{}).path||'')){const q=d.querySelector('pre');if(q)q.innerHTML=hl(q.textContent);}
      if(b.intent)d.querySelector('summary').title=b.intent;
    }
  }
}
function toolResult(m){
  const txt=(m.content||[]).map(b=>b.text||'').join('\n'); if(!txt.trim())return;
  const bad=/^(Error|Path .* not found)/i.test(txt.trim());
  block('res'+(bad?' bad':''),(bad?'✗ ':'')+'output · '+nlines(txt)+' lines',txt);
}
const es=new EventSource('/api/events');
es.onopen=()=>$('status').textContent='live';
es.onerror=()=>$('status').textContent='disconnected — retrying';
es.onmessage=e=>{
  const r=JSON.parse(e.data);
  if(r.kind==='run_start'){$('model').textContent=r.model;
    row('meta','run start — '+r.project+'  max_iterations='+r.max_iterations+(r.budget?'  budget=$'+r.budget:''));return;}
  if(r.kind==='run_end'){$('status').textContent='finished';row('iter','run end — '+r.reason);return;}
  if(r.kind==='reconnect'){row('meta','session ended — reconnecting');return;}
  if(r.kind==='iteration'){
    $('iter').textContent=r.idx;$('spent').textContent='$'+Number(r.spent).toFixed(4);
    if(r.experiment){logged++;$('logged').textContent=logged;}
    row('iter','■ iteration '+r.idx+' — '+(r.experiment||'NOTHING LOGGED')+'  '+Math.round(r.seconds)+'s  '
      +r.tool_calls+' tools  '+r.tokens+' tok  $'+Number(r.cost).toFixed(4));
    return;}
  if(r.kind!=='event'||r.type!=='message_end')return;
  const m=(r.data||{}).message||{}; const id=m.responseId||m.timestamp;
  if(id&&seen.has(id))return; if(id)seen.add(id);
  if(m.role==='assistant')assistant(m); else if(m.role==='toolResult')toolResult(m);
};
$('hideRead').onchange=e=>log.classList.toggle('hide-read',e.target.checked);
$('expand').onchange=e=>document.querySelectorAll('#log details').forEach(d=>d.open=e.target.checked);
log.classList.add('hide-read');
"""

LOG_NAV = """<b id="model">—</b>
  <span>iter</span> <b id="iter">0</b>
  <span>logged</span> <b id="logged">0</b>
  <span>tools</span> <b id="tools">0</b>
  <span>spent</span> <b id="spent">$0.0000</b>
  <span id="status">connecting…</span>
  <label><input type="checkbox" id="hideRead" checked> hide reads</label>
  <label><input type="checkbox" id="expand"> expand all</label>"""


# --------------------------------------------------------------- chart page
CHART_CSS = r"""
#chart svg{width:100%;max-width:1200px;height:60vh;min-height:320px;background:#12151c;border:1px solid var(--line)}
#chart text{fill:var(--dim);font:11px ui-monospace,monospace}
.best{fill:none;stroke:var(--acc);stroke-width:1.5}
.pt{fill:var(--warn)} .ptbad{fill:var(--err)} .ptq{fill:none;stroke:var(--dim);stroke-width:1}
.ax{stroke:#2a2f3c;stroke-width:1}
#tbl{margin-top:16px;border-collapse:collapse;font-size:12px}
#tbl td,#tbl th{padding:3px 12px 3px 0;text-align:left;border-bottom:1px solid var(--line)}
#tbl th{color:var(--dim);font-weight:normal}
"""

CHART_JS = r"""
const $=i=>document.getElementById(i);
async function draw(){
  let h=[],ex=[]; try{h=await (await fetch('/api/history')).json();
    ex=await (await fetch('/api/experiments')).json();}catch(e){return;}
  const byKernel={}; ex.forEach(e=>{if(e.kernel)byKernel[e.kernel]=e;});
  // `quick` samples only the smallest+largest workload, so its mean is not comparable
  // with a full/stride sweep. Probes never move the curve.
  const showQ=$('showQuick').checked;
  const real=h.filter(d=>d.mean_latency_ms!=null&&d.mode!=='quick');
  const probes=h.filter(d=>d.mean_latency_ms!=null&&d.mode==='quick');
  $('nprobe').textContent=showQ?'':probes.length+' probes hidden';
  const pts=showQ?real.concat(probes).sort((a,b)=>a.t-b.t):real;
  if(!pts.length){$('chart').innerHTML='<p style="color:var(--dim)">no measurements yet</p>';return;}
  const W=1200,H=420,L=64,R=16,T=18,B=30;
  const ys=pts.map(d=>d.mean_latency_ms),lo=Math.min(...ys),hi=Math.max(...ys);
  const ly=v=>Math.log10(Math.max(v,1e-6)), y0=ly(lo*0.85), y1=ly(hi*1.15);
  const X=i=>L+(pts.length<2?0:i*(W-L-R)/(pts.length-1));
  const Y=v=>T+(y1-ly(v))/((y1-y0)||1)*(H-T-B);
  let best=Infinity,seg=[];
  pts.forEach((d,i)=>{if(d.mode==='quick')return;best=Math.min(best,d.mean_latency_ms);
    seg.push((seg.length?'L':'M')+X(i).toFixed(1)+' '+Y(best).toFixed(1));});
  const dots=pts.map((d,i)=>{
    const bad=d.num_passed<d.num_workloads,q=d.mode==='quick';
    return '<circle class="'+(bad?'ptbad':q?'ptq':'pt')+'" cx="'+X(i).toFixed(1)+'" cy="'+Y(d.mean_latency_ms).toFixed(1)
      +'" r="'+(q?2:4)+'"><title>'+d.mean_latency_ms.toFixed(4)+' ms · '+d.num_passed+'/'+d.num_workloads
      +' · '+d.mode+' · '+(d.kernel||'').slice(0,8)
      +((byKernel[d.kernel]||{}).desc?'\n'+byKernel[d.kernel].exp+': '+byKernel[d.kernel].desc:'')
      +'</title></circle>';}).join('');
  const labels=pts.map((d,i)=>{const e=byKernel[d.kernel]; if(!e||d.mode==='quick')return '';
    return '<text x="'+X(i).toFixed(1)+'" y="'+(Y(d.mean_latency_ms)-10).toFixed(1)
      +'" text-anchor="middle" style="fill:var(--warn)">'+esc(e.exp)+'</text>';}).join('');
  const ticks=[hi,Math.sqrt(hi*lo),lo].map(v=>
    '<text x="6" y="'+(Y(v)+4).toFixed(1)+'">'+v.toFixed(4)+' ms</text>'
    +'<line class="ax" x1="'+L+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-R)+'" y2="'+Y(v).toFixed(1)+'"/>').join('');
  $('chart').innerHTML='<svg viewBox="0 0 '+W+' '+H+'">'+ticks+'<path class="best" d="'+seg.join(' ')+'"/>'+dots+labels+'</svg>';
  const bestOf=Math.min(...real.map(d=>d.mean_latency_ms));
  $('tbl').innerHTML='<tr><th>#</th><th>exp</th><th>mean</th><th>min</th><th>max</th><th>pass</th><th>mode</th><th>change</th></tr>'
    +pts.map((d,i)=>'<tr'+(d.mean_latency_ms===bestOf?' style="color:var(--acc)"':'')+'><td>'+(i+1)+'</td><td>'
      +esc((byKernel[d.kernel]||{}).exp||'')+'</td><td>'
      +d.mean_latency_ms.toFixed(4)+'</td><td>'+(d.min_latency_ms||0).toFixed(4)+'</td><td>'
      +(d.max_latency_ms||0).toFixed(4)+'</td><td>'+d.num_passed+'/'+d.num_workloads+'</td><td>'
      +d.mode+'</td><td>'+esc((byKernel[d.kernel]||{}).desc||'')+'</td></tr>').join('');
}
draw(); setInterval(draw,15000); $('showQuick').onchange=draw;
"""


# --------------------------------------------------------------- files page
FILES_CSS = r"""
#wrap{display:flex;gap:14px;align-items:flex-start}
#tree{flex:0 0 320px;max-height:82vh;overflow:auto;border:1px solid var(--line);padding:8px;background:#12151c}
#tree div{padding:1px 4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#tree div:hover{background:#1b1f28} #tree div.on{background:var(--acc);color:#0f1115}
#tree .dir{color:var(--dim);cursor:default;margin-top:6px}
#view{flex:1;min-width:0} #view pre{max-height:82vh;overflow:auto}
#path{color:var(--acc);margin-bottom:6px}
#md{max-height:82vh;overflow:auto;padding:4px 14px 20px;background:#12151c;border:1px solid var(--line)}
#md h1,#md h2,#md h3{color:var(--acc);margin:14px 0 6px;font-size:15px}
#md h1{font-size:18px;border-bottom:1px solid var(--line);padding-bottom:4px}
#md h3{font-size:13px;color:var(--warn)}
#md code{background:#1b1f28;padding:1px 4px;border-radius:3px;color:#c3cad6}
#md pre{margin:8px 0}#md pre code{background:none;padding:0}
#md table{border-collapse:collapse;margin:8px 0;font-size:12px}
#md th,#md td{border:1px solid var(--line);padding:3px 9px;text-align:left}
#md th{color:var(--dim);font-weight:normal;background:#161922}
#md ul,#md ol{margin:6px 0;padding-left:22px} #md li{margin:2px 0}
#md hr{border:none;border-top:1px solid var(--line);margin:12px 0}
#md strong{color:#fff} #md a{color:var(--acc)}
"""

FILES_JS = r"""
const $=i=>document.getElementById(i);
let cur=null, raw=false, last={path:'',text:''};

async function tree(){
  const files=await (await fetch('/api/tree')).json();
  const byDir={};
  files.forEach(f=>{const i=f.lastIndexOf('/');const d=i<0?'.':f.slice(0,i);(byDir[d]=byDir[d]||[]).push(f);});
  const el=$('tree'); el.innerHTML='';
  Object.keys(byDir).sort().forEach(d=>{
    const h=document.createElement('div');h.className='dir';h.textContent=d==='.'?'experiments/':d+'/';el.appendChild(h);
    byDir[d].sort().forEach(f=>{
      const n=document.createElement('div');n.textContent='  '+f.slice(f.lastIndexOf('/')+1);
      n.onclick=()=>open_(f,n); el.appendChild(n);});
  });
}
async function open_(f,node){
  if(cur)cur.classList.remove('on'); if(node){node.classList.add('on');cur=node;}
  $('path').textContent=f;
  const r=await fetch('/api/file?path='+encodeURIComponent(f));
  last={path:f,text:r.ok?await r.text():'('+r.status+') cannot read'};
  show(); location.hash=f;
}
function show(){
  const isMd=/\.md$/i.test(last.path)&&!raw, isPy=/\.(py|cu|cuh)$/i.test(last.path);
  $('md').style.display=isMd?'':'none'; $('body').style.display=isMd?'none':'';
  if(isMd)$('md').innerHTML=md(last.text);
  else if(isPy)$('body').innerHTML=hl(last.text);
  else $('body').textContent=last.text;
  $('rawBtn').textContent=raw?'rendered':'raw';
  $('rawBtn').style.display=/\.md$/i.test(last.path)?'':'none';
}
$('rawBtn').onclick=()=>{raw=!raw;show();};
tree().then(()=>{const h=decodeURIComponent(location.hash.slice(1)); if(h)open_(h,null);});
"""


def log_page() -> str:
    return _shell("log", LOG_NAV, '<main><div id="log"></div></main>', LOG_JS).replace(
        "{extra_css}", LOG_CSS
    )


def chart_page() -> str:
    nav = ('<label><input type="checkbox" id="showQuick"> show quick probes</label>'
           '<span id="nprobe"></span>')
    body = '<main><div id="chart"></div><table id="tbl"></table></main>'
    return _shell("chart", nav, body, CHART_JS).replace("{extra_css}", CHART_CSS)


def files_page() -> str:
    body = ('<main><div id="wrap"><div id="tree"></div>'
            '<div id="view"><div id="path"></div>'
            '<div id="md" class="md"></div><pre id="body"></pre></div></div></main>')
    nav = '<button id="rawBtn" style="display:none">raw</button>'
    return _shell("experiments", nav, body, FILES_JS).replace("{extra_css}", FILES_CSS)
