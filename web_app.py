"""Local website for the pipeline.

    python web_app.py        ->  opens http://localhost:8000

The page runs the three steps one after another (so slow CPU models don't hit a browser timeout,
and each result appears as soon as it is ready):
    POST /api/retrieve      Step 1  RAG retrieval
    POST /api/requirements  Step 2  LLM 1 -> requirements (+ grounding check)
    POST /api/sdlc          Step 3  LLM 2 -> SDLC model only
    GET  /api/config        which Ollama models are installed + sensible defaults
"""
from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_clients import (DEFAULT_GEMINI_MODEL, LLMError, build_client, ollama_installed_models,
                         pick_default_ollama_model)
from pipeline import (MIN_GROUNDING, build_index, generate_requirements, ground_requirements,
                      identify_sdlc, load_documents, retrieve_evidence)

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Blockchain Finance: Requirements &amp; SDLC Pipeline</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#f8fafc; --muted:#94a3b8; --accent:#38bdf8;
          --ok:#22c55e; --warn:#f59e0b; --bad:#f87171; }
  * { box-sizing:border-box; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
  body { margin:0; padding:28px 16px 60px; background:var(--bg); color:var(--text); }
  .container { max-width:960px; margin:0 auto; }
  h1 { margin:0 0 6px; font-size:25px; } h2 { margin:0 0 12px; font-size:18px; }
  p.sub { margin:0 0 18px; color:var(--muted); font-size:14px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:18px; margin-bottom:16px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
  @media (max-width:700px){ .grid{ grid-template-columns:1fr; } }
  label { display:block; font-size:12.5px; color:var(--muted); margin-bottom:5px; font-weight:600; }
  input, select, textarea { width:100%; padding:9px 11px; border-radius:6px; border:1px solid var(--border); background:#0b1120; color:var(--text); font-size:14px; }
  textarea { resize:vertical; min-height:64px; }
  details { margin-bottom:14px; } summary { cursor:pointer; color:var(--muted); font-size:13px; margin-bottom:8px; }
  button { background:var(--accent); color:#0f172a; border:0; padding:12px 18px; border-radius:6px; font-weight:700; font-size:15px; cursor:pointer; width:100%; }
  button:disabled { opacity:.5; cursor:wait; }
  .status { font-size:13px; line-height:1.6; margin-bottom:16px; padding:12px 14px; border-radius:8px; border:1px solid var(--border); background:#0b1120; }
  .status .ok { color:var(--ok); } .status .warn { color:var(--warn); } .status .bad { color:var(--bad); }
  .stepper { display:flex; gap:8px; margin:0 0 16px; flex-wrap:wrap; }
  .pill { flex:1; min-width:150px; padding:8px 12px; border-radius:20px; border:1px solid var(--border); font-size:12.5px; color:var(--muted); text-align:center; }
  .pill.active { border-color:var(--accent); color:var(--accent); } .pill.done { border-color:var(--ok); color:var(--ok); } .pill.error { border-color:var(--bad); color:var(--bad); }
  .item { padding:11px 12px; border:1px solid var(--border); border-radius:8px; margin-bottom:9px; background:#0b1120; font-size:14px; }
  .item.review { border-color:var(--warn); }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11.5px; font-weight:700; background:#0284c7; color:#fff; margin-right:6px; }
  .badge.alt { background:#475569; } .badge.src { background:#0d9488; }
  .meta { margin-top:6px; font-size:12.5px; color:var(--muted); } .meta.warn { color:var(--warn); }
  .sdlc { font-size:22px; font-weight:800; color:var(--ok); margin:4px 0 10px; }
  .error { display:none; color:var(--bad); background:#450a0a; border:1px solid #ef4444; padding:12px; border-radius:8px; margin-bottom:16px; font-size:14px; white-space:pre-wrap; }
  #results { display:none; }
</style>
</head>
<body>
<div class="container">
  <h1>Blockchain Finance: Requirements &amp; SDLC Pipeline</h1>
  <p class="sub">Input &rarr; <b>RAG retrieval</b> &rarr; <b>LLM 1</b> (requirements + reasoning) &rarr; <b>LLM 2</b> (SDLC model only) &rarr; final output</p>

  <div class="status" id="status">Checking Ollama&hellip;</div>

  <div class="card">
    <div class="grid">
      <div><label>Domain</label><input id="domain" value="DeFi Lending" /></div>
      <div><label>Document folder (RAG corpus: .txt / .md files)</label><input id="docs" value="sample_docs" /></div>
    </div>
    <div style="margin-bottom:14px"><label>Functionality to generate requirements for</label>
      <textarea id="functionality">Multi-signature approval for large withdrawals from a lending pool</textarea></div>
    <details><summary>Optional: paste extra document text to add to the corpus</summary>
      <textarea id="extra" placeholder="Paste an existing requirement document or policy here..."></textarea></details>
    <div class="grid">
      <div><label>LLM 1 provider (requirements)</label>
        <select id="p1" onchange="providerChanged(1)"><option value="ollama">Ollama (local)</option><option value="gemini">Gemini (cloud)</option></select></div>
      <div><label>LLM 1 model</label><input id="m1" list="modelList" /></div>
    </div>
    <div class="grid">
      <div><label>LLM 2 provider (SDLC only)</label>
        <select id="p2" onchange="providerChanged(2)"><option value="ollama">Ollama (local)</option><option value="gemini">Gemini (cloud)</option></select></div>
      <div><label>LLM 2 model</label><input id="m2" list="modelList" /></div>
    </div>
    <datalist id="modelList"></datalist>
    <div class="grid">
      <div id="keyBox" style="display:none"><label>Gemini API key (kept in memory, sent only to this local server)</label><input type="password" id="apiKey" placeholder="AIza..." /></div>
      <div><label>Evidence chunks to retrieve (top-k)</label><input type="number" id="topk" value="4" min="1" max="8" /></div>
    </div>
    <button id="runBtn" onclick="runPipeline()">Run pipeline</button>
  </div>

  <div class="error" id="errorBox"></div>
  <div class="stepper">
    <div class="pill" id="s1">1 &middot; RAG retrieval</div>
    <div class="pill" id="s2">2 &middot; LLM 1: requirements</div>
    <div class="pill" id="s3">3 &middot; LLM 2: SDLC</div>
  </div>

  <div id="results">
    <div class="card" id="c1" style="display:none"><h2>Step 1 &mdash; Evidence retrieved from your documents</h2><div id="evidence"></div></div>
    <div class="card" id="c2" style="display:none"><h2>Step 2 &mdash; Requirements <span id="m1used" class="meta"></span></h2>
      <div id="reqs"></div><p class="meta"><b>Reasoning:</b> <span id="reasoning"></span></p></div>
    <div class="card" id="c3" style="display:none"><h2>Step 3 &mdash; Recommended SDLC <span id="m2used" class="meta"></span></h2>
      <div class="sdlc" id="sdlcName"></div><label>Key factors</label><ul id="factors"></ul>
      <p style="font-size:14px;line-height:1.55"><b>Software-engineering justification:</b> <span id="justification"></span></p></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let CFG = null;

async function api(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json().catch(() => ({error:'The server sent an unreadable response.'}));
  if (!r.ok || d.error) throw new Error(d.error || 'Request failed');
  return d;
}

function setDefaultModel(step) {
  const prov = $('p' + step).value;
  $('m' + step).value = prov === 'gemini' ? CFG.gemini_model : CFG.defaults['llm' + step].model;
}
function providerChanged(step) {
  setDefaultModel(step);
  $('keyBox').style.display = ($('p1').value === 'gemini' || $('p2').value === 'gemini') && !CFG.gemini_key_in_env ? 'block' : 'none';
}

async function loadConfig() {
  const r = await fetch('/api/config'); CFG = await r.json();
  setDefaultModel(1); setDefaultModel(2);
  $('modelList').innerHTML = CFG.installed.map(n => `<option value="${esc(n)}">`).join('');
  const lines = [];
  if (!CFG.ollama_up) {
    lines.push('<span class="bad">Ollama is not running.</span> Start the Ollama app (or run <code>ollama serve</code>), then reload this page &mdash; or switch a step to Gemini.');
  } else {
    lines.push('<span class="ok">Ollama is running.</span> Installed models: ' + (CFG.installed.length ? esc(CFG.installed.join(', ')) : 'none'));
    [1, 2].forEach(n => {
      const d = CFG.defaults['llm' + n], have = CFG.installed.some(x => x.split(':')[0] === d.model.split(':')[0]);
      if (d.finetuned) lines.push(`LLM ${n}: <span class="ok">using your fine-tuned model <b>${esc(d.model)}</b></span>`);
      else if (have) lines.push(`LLM ${n}: fine-tuned model not installed &mdash; using base model <b>${esc(d.model)}</b> (works fine; fine-tuning is optional, see README)`);
      else lines.push(`LLM ${n}: <span class="warn">base model <b>${esc(d.model)}</b> is not installed &mdash; run <code>ollama pull ${esc(d.model)}</code></span>`);
    });
  }
  $('status').innerHTML = lines.join('<br>');
}

function setStep(n, state) { $('s' + n).className = 'pill ' + state; }

async function runPipeline() {
  const btn = $('runBtn'), err = $('errorBox');
  btn.disabled = true; err.style.display = 'none'; $('results').style.display = 'none';
  ['c1','c2','c3'].forEach(id => $(id).style.display = 'none'); [1,2,3].forEach(n => setStep(n, ''));
  let current = 1;
  const common = {docs: $('docs').value.trim(), extra_text: $('extra').value, domain: $('domain').value.trim(),
                  functionality: $('functionality').value.trim(), top_k: parseInt($('topk').value || '4', 10)};
  const key = $('apiKey').value.trim();
  try {
    $('results').style.display = 'block';
    setStep(1, 'active'); btn.textContent = 'Step 1/3: retrieving evidence...';
    const r1 = await api('/api/retrieve', common);
    $('evidence').innerHTML = r1.evidence.length ? r1.evidence.map(e =>
      `<div class="item"><span class="badge">${esc(e.id)}</span><span class="badge src">${esc(e.source)}</span><span class="meta">score ${esc(e.score)}</span>
       <div style="margin-top:6px;font-size:13px;color:#cbd5e1">${esc(e.text)}</div></div>`).join('')
      : '<div class="meta warn">No relevant evidence was found in your documents. The model will rely on general domain knowledge and every requirement will be flagged for review.</div>';
    $('c1').style.display = 'block'; setStep(1, 'done');

    current = 2; setStep(2, 'active'); btn.textContent = 'Step 2/3: LLM 1 is writing requirements (can take a minute on CPU)...';
    const r2 = await api('/api/requirements', {...common, evidence: r1.evidence, provider: $('p1').value, model: $('m1').value.trim(), api_key: key});
    $('reqs').innerHTML = r2.requirements.map(r => `<div class="item ${r.needs_review ? 'review' : ''}">
      <span class="badge">${esc(r.id)}</span><span class="badge alt">${esc(r.category)}</span>
      <div style="margin-top:6px">${esc(r.statement)}</div>
      <div class="meta ${r.needs_review ? 'warn' : ''}">${r.needs_review ? '&#9888; Needs human review &mdash; weakly supported by the retrieved evidence' : 'Supported by the evidence'}
        (closest: ${esc(r.support_source || 'none')}, similarity ${esc(r.support_score)})</div></div>`).join('');
    $('reasoning').textContent = r2.reasoning; $('m1used').textContent = '(' + r2.model + ')';
    $('c2').style.display = 'block'; setStep(2, 'done');

    current = 3; setStep(3, 'active'); btn.textContent = 'Step 3/3: LLM 2 is choosing the SDLC model...';
    const r3 = await api('/api/sdlc', {domain: common.domain, evidence: r1.evidence, requirements: r2.requirements,
                                       provider: $('p2').value, model: $('m2').value.trim(), api_key: key});
    $('sdlcName').textContent = r3.sdlc.recommended_sdlc; $('m2used').textContent = '(' + r3.model + ')';
    $('factors').innerHTML = r3.sdlc.key_factors.map(f => `<li>${esc(f)}</li>`).join('');
    $('justification').textContent = r3.sdlc.justification;
    $('c3').style.display = 'block'; setStep(3, 'done');
  } catch (e) {
    setStep(current, 'error'); err.textContent = 'Error: ' + e.message; err.style.display = 'block';
  } finally { btn.disabled = false; btn.textContent = 'Run pipeline'; }
}
loadConfig();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------- API helpers
def _text(body: dict, key: str, required: bool = False) -> str:
    value = str(body.get(key) or "").strip()
    if required and not value:
        raise ValueError(f"'{key}' is required.")
    return value


def _retriever(body: dict):
    folder = _text(body, "docs")
    if folder:
        path = Path(folder).expanduser()
        folder = str(path if path.is_absolute() else BASE_DIR / path)
    return build_index(load_documents(folder, body.get("extra_text") or ""))


def _list_of_dicts(body: dict, key: str, fields: tuple) -> list:
    items = body.get(key)
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        raise ValueError(f"'{key}' must be a list.")
    for i in items:
        for f in fields:
            i.setdefault(f, "")
    return items


def api_config() -> dict:
    installed = ollama_installed_models()
    return {
        "ollama_up": installed is not None,
        "installed": installed or [],
        "defaults": {"llm1": pick_default_ollama_model("llm1", installed), "llm2": pick_default_ollama_model("llm2", installed)},
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "gemini_key_in_env": bool(os.environ.get("GEMINI_API_KEY")),
        "min_grounding": MIN_GROUNDING,
    }


def api_retrieve(body: dict) -> dict:
    top_k = max(1, min(int(body.get("top_k") or 4), 8))
    return {"evidence": retrieve_evidence(_retriever(body), _text(body, "domain", True), _text(body, "functionality", True), top_k)}


def api_requirements(body: dict) -> dict:
    domain, functionality = _text(body, "domain", True), _text(body, "functionality", True)
    evidence = _list_of_dicts(body, "evidence", ("id", "source", "text"))
    llm1 = build_client(_text(body, "provider") or "ollama", _text(body, "model") or None, _text(body, "api_key") or None, role="llm1")
    draft = generate_requirements(domain, functionality, evidence, llm1)
    reqs = ground_requirements(_retriever(body), draft["requirements"], evidence)
    return {"requirements": reqs, "reasoning": draft["reasoning"], "model": llm1.label,
            "needs_review_count": sum(r["needs_review"] for r in reqs)}


def api_sdlc(body: dict) -> dict:
    domain = _text(body, "domain", True)
    evidence = _list_of_dicts(body, "evidence", ("id", "source", "text"))
    reqs = _list_of_dicts(body, "requirements", ("id", "statement", "category"))
    llm2 = build_client(_text(body, "provider") or "ollama", _text(body, "model") or None, _text(body, "api_key") or None, role="llm2")
    return {"sdlc": identify_sdlc(domain, reqs, evidence, llm2), "model": llm2.label}


ROUTES = {"/api/retrieve": api_retrieve, "/api/requirements": api_requirements, "/api/sdlc": api_sdlc}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, content_type: str = "application/json") -> None:
        data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE, "text/html")
        elif self.path == "/api/config":
            self._send(200, api_config())
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        handler = ROUTES.get(self.path)
        if not handler:
            return self._send(404, {"error": "Not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
            self._send(200, handler(body))
        except (ValueError, LLMError) as e:       # user / model problems: show the message in the page
            self._send(400, {"error": str(e)})
        except Exception as e:                    # genuine bugs: log full trace in the terminal
            traceback.print_exc()
            self._send(500, {"error": f"Unexpected error: {e}"})

    def log_message(self, fmt, *args):            # keep the terminal readable (and never echo request bodies)
        pass


def main() -> None:
    url = f"http://localhost:{PORT}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        raise SystemExit(f"Cannot start on port {PORT} ({e}). Close whatever is using it, or set PORT=8001.")
    print(f"\n[ok] Website running at {url}   (Ctrl+C to stop)")
    threading.Timer(0.8, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
