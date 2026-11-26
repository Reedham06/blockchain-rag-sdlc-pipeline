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
import ssl
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from access_control import AuthorizationError, authorize_approval
from agent_team import run_review_agents, validate_agent_reviews
from audit_store import create_run_id, get_run_events, record_event
from llm_clients import (DEFAULT_GEMINI_MODEL, LLMError, build_client, ollama_installed_models,
                         pick_default_ollama_model)
from pipeline import (MIN_GROUNDING, build_index, generate_requirements, ground_requirements,
                      identify_sdlc, load_documents, retrieve_evidence, stage1_messages, stage2_messages)
from project_engine import (clarification_questions, generate_artifacts, generate_workflow,
                            ranked_sdlc, run_specialist_agents,
                            sanitize_prompt_text)

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Blockchain &amp; Digital Assets: Agentic Requirements Engineering</title>
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
  <h1>Blockchain &amp; Digital Assets: Agentic Requirements Engineering</h1>
  <p class="sub">Four role-specific LLM agents &bull; six rule-based specialist agents &bull; coordinator-led evidence sharing &bull; human approval gates</p>

  <div class="status" id="status">Checking Ollama&hellip;</div>

  <div class="card">
    <div class="grid">
      <div><label>Blockchain project domain</label><input id="domain" value="Blockchain / Digital Assets - DeFi Lending" /></div>
      <div><label>Knowledge-base folder (.txt / .md / .docx)</label><input id="docs" value="sample_docs" /></div>
    </div>
    <div class="grid">
      <div><label>Stakeholder role/source</label><select id="stakeholderRole">
        <option>Product Owner</option><option>Customer / End User</option><option>Business Analyst</option>
        <option>Smart-Contract Engineer</option><option>Software Architect</option><option>Security Officer</option>
        <option>Privacy Officer</option><option>Compliance Officer</option><option>Risk Manager</option>
        <option>Operations / Validator Operator</option><option>Auditor</option><option>Regulator</option></select></div>
      <div><label>Functionality / blockchain use case</label><textarea id="functionality">Multi-signature approval for large withdrawals from a smart-contract lending pool</textarea></div>
    </div>
    <details open><summary>Stakeholder conversation and source material</summary>
      <p class="meta">Select the role who is speaking, paste that person's notes, then click <b>Add interview to this run</b>.
        Repeat for other roles. Added interviews are held in this browser tab until you run the pipeline; they are not saved as reusable files.</p>
      <label>Interview transcript / conversation / email or meeting notes</label>
      <textarea id="interview" placeholder="Paste anonymised notes from the selected stakeholder role."></textarea>
      <button type="button" onclick="addStakeholderInterview()">Add interview to this run</button>
      <div id="interviewStatus" class="meta" aria-live="polite">No interviews added yet.</div>
      <div id="savedInterviews"></div>
      <button id="clarificationButton" type="button" onclick="askClarifications()">Suggest missing topics (rule-based)</button>
      <div id="clarificationStatus" class="meta" aria-live="polite"></div>
      <div id="clarificationBox"></div>
      <label style="margin-top:12px">Answers to clarification questions</label>
      <textarea id="clarificationAnswers" placeholder="Answer any follow-up questions here; these answers are sent with the run."></textarea>
      <label style="margin-top:12px">Questionnaire answers / incident or audit findings / legacy and API notes</label>
      <textarea id="extra" placeholder="Additional approved source material (anonymise sensitive data first)."></textarea>
    </details>
    <div class="meta">The SDLC agent infers project characteristics from your use case, stakeholder interviews, answers, requirements, and retrieved evidence. You do not need to rate them.</div>
    <div class="grid">
      <div><label>LLM 1 provider (requirements + security/compliance reviews)</label>
        <select id="p1" onchange="providerChanged(1)"><option value="ollama">Ollama (local)</option><option value="gemini">Gemini (cloud)</option></select></div>
      <div><label>LLM 1 model</label><input id="m1" list="modelList" /></div>
    </div>
    <div class="grid">
      <div><label>LLM 2 provider (SDLC explanation)</label>
        <select id="p2" onchange="providerChanged(2)"><option value="ollama">Ollama (local)</option><option value="gemini">Gemini (cloud)</option></select></div>
      <div><label>LLM 2 model</label><input id="m2" list="modelList" /></div>
    </div>
    <datalist id="modelList"></datalist>
    <div class="grid">
      <div id="keyBox" style="display:none"><label>Gemini API key (kept in memory, sent only to this local server)</label><input type="password" id="apiKey" placeholder="AIza..." /></div>
      <div><label>Evidence chunks to retrieve (top-k)</label><input type="number" id="topk" value="4" min="1" max="8" /></div>
    </div>
    <label><input type="checkbox" id="cloudConsent" style="width:auto" /> I authorise sending masked project prompts and retrieved evidence to Gemini when selected.</label>
    <button id="runBtn" onclick="runPipeline()">Run pipeline</button>
  </div>

  <div class="error" id="errorBox"></div>
  <div class="stepper">
    <div class="pill" id="s1">1 &middot; Stakeholder intake + RAG</div>
    <div class="pill" id="s2">2 &middot; Requirement agents</div>
    <div class="pill" id="s3">3 &middot; SDLC decision + workflow</div>
  </div>

  <div id="results">
    <div class="card" id="c1" style="display:none"><h2>Step 1 &mdash; Evidence retrieved from approved knowledge sources</h2><div id="evidence"></div></div>
    <div class="card" id="c2" style="display:none"><h2>Step 2 &mdash; Requirements <span id="m1used" class="meta"></span></h2>
      <div id="reqs"></div><p class="meta"><b>Reasoning:</b> <span id="reasoning"></span></p>
      <h3>Coordinated agent team</h3><div id="agents"></div><div id="agentReviews"></div>
      <h3>Conflicts / duplicates</h3><div id="conflicts"></div></div>
    <div class="card" id="c3" style="display:none"><h2>Step 3 &mdash; Recommended SDLC <span id="m2used" class="meta"></span></h2>
      <label>SDLC profile inferred by the agent</label><div id="projectProfile"></div>
      <div class="sdlc" id="sdlcName"></div><label>Key factors</label><ul id="factors"></ul>
      <p style="font-size:14px;line-height:1.55"><b>Software-engineering justification:</b> <span id="justification"></span></p>
      <p class="meta" id="sdlcMismatch"></p></div>
    <div class="card" id="c4" style="display:none"><h2>Governance artefacts</h2>
      <label>Ranked SDLC recommendations</label><div id="ranked"></div>
      <label>Generated workflow</label><div id="workflow"></div>
      <label>Compliance control mappings (advisory)</label><div id="governance"></div>
      <label>Blockchain risk register</label><div id="riskRegister"></div>
      <label>Traceability matrix</label><div id="traceability"></div>
      <label>Generated SRS / use-case / data artifacts</label><div id="artifactSummary"></div>
      <label>Requirement quality / clarification items</label><div id="openIssues"></div>
      <label>Human approval (configured reviewer token and TOTP are required)</label>
      <div class="grid">
        <div><label>Approval type</label><select id="approvalType"><option value="requirement">Requirement baseline</option><option value="security">High-risk security</option><option value="compliance">Regulatory interpretation</option><option value="sdlc">SDLC selection</option><option value="production">Production readiness</option></select></div>
        <div><label>Target ID (requirement ID or SDLC)</label><input id="approvalTarget" placeholder="REQ-001 or SDLC" /></div>
        <div><label>Decision</label><select id="approvalAction"><option>accept</option><option>reject</option><option>modify</option><option>regenerate</option></select></div>
        <div><label>Reviewer name</label><input id="reviewer" placeholder="Authorised reviewer" /></div>
        <div><label>Role token</label><input id="approvalToken" type="password" /></div>
        <div><label>Current 6-digit MFA code</label><input id="mfaCode" inputmode="numeric" /></div>
        <div><label>Proposed modified statement (for modify)</label><input id="modifiedStatement" /></div>
        <div><label>Review comment</label><input id="approvalComment" /></div>
      </div>
      <button type="button" onclick="submitApproval()">Record approval / review decision</button><div id="approvalStatus" class="meta"></div>
      <button type="button" onclick="showAudit()">View this run's audit trail</button><div id="auditTrail"></div>
      <button type="button" onclick="downloadArtifacts()">Download JSON artifact bundle</button></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let CFG = null;
let ARTIFACT_BUNDLE = null;
let RUN_ID = null;
let RESULT_CONTEXT = null;
let STAKEHOLDER_INTERVIEWS = [];

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
  lines.push(CFG.approvals_enabled
    ? '<span class="ok">Human approvals enabled:</span> role tokens and TOTP MFA are configured.'
    : '<span class="warn">Approvals are fail-closed until role tokens and TOTP MFA are configured.</span>');
  lines.push(`<span class="warn">${esc(CFG.warning)}</span>`);
  $('status').innerHTML = lines.join('<br>');
}

function setStep(n, state) { $('s' + n).className = 'pill ' + state; }
async function askClarifications() {
  const button = $('clarificationButton');
  const text = [
    $('functionality').value, $('interview').value, $('extra').value, $('clarificationAnswers').value,
    ...STAKEHOLDER_INTERVIEWS.map(item => `${item.role}: ${item.transcript}`)
  ].filter(Boolean).join('\n');
  $('clarificationStatus').textContent = 'Checking the supplied context for missing blockchain requirements...';
  $('clarificationBox').replaceChildren();
  button.disabled = true;
  try {
    const data = await api('/api/interview', {text, run_id:ensureRunId(), stakeholder_role:$('stakeholderRole').value});
    $('clarificationBox').innerHTML = data.questions.map((q, index) =>
      `<div class="item"><b>Question ${index + 1}</b>: ${esc(q)}</div>`).join('');
    $('clarificationStatus').textContent = data.questions.length
      ? 'These are rule-based suggestions for missing topics. Review them, enter answers below, and they will be sent with the run.'
      : 'No follow-up suggestions returned.';
  } catch (e) { $('clarificationStatus').textContent = 'Could not generate questions: ' + e.message; }
  finally { button.disabled = false; }
}
function renderStakeholderInterviews() {
  $('savedInterviews').innerHTML = STAKEHOLDER_INTERVIEWS.map((item, index) =>
    `<div class="item"><b>${esc(item.role)}</b> &mdash; interview ${index + 1}
      <div class="meta">${esc(item.transcript.slice(0, 180))}${item.transcript.length > 180 ? '...' : ''}</div>
      <button type="button" onclick="removeStakeholderInterview(${index})">Remove</button></div>`).join('');
  $('interviewStatus').textContent = STAKEHOLDER_INTERVIEWS.length
    ? `${STAKEHOLDER_INTERVIEWS.length} interview(s) added for this run. They will be combined with any text still in the box.`
    : 'No interviews added yet.';
}
function addStakeholderInterview() {
  const transcript = $('interview').value.trim();
  if (!transcript) {
    $('interviewStatus').textContent = 'Paste interview notes before adding them.';
    return;
  }
  STAKEHOLDER_INTERVIEWS.push({role: $('stakeholderRole').value, transcript});
  $('interview').value = '';
  renderStakeholderInterviews();
}
function removeStakeholderInterview(index) {
  STAKEHOLDER_INTERVIEWS.splice(index, 1);
  renderStakeholderInterviews();
}
function ensureRunId() {
  if (!RUN_ID) RUN_ID = crypto.randomUUID().replace(/-/g, '');
  return RUN_ID;
}
function downloadArtifacts() {
  if (!ARTIFACT_BUNDLE) return;
  const blob = new Blob([JSON.stringify(ARTIFACT_BUNDLE, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'blockchain_requirements_artifacts.json'; a.click();
  URL.revokeObjectURL(url);
}
function renderRequirements(requirements) {
  $('reqs').innerHTML = requirements.map(r => `<div class="item ${r.needs_review ? 'review' : ''}">
    <span class="badge">${esc(r.id)}</span><span class="badge alt">${esc(r.priority || 'Medium')}</span>
    <div><b>${esc(r.categories.join(', '))}</b></div>
    <div style="margin-top:6px">${esc(r.statement)}</div>
    <div class="meta">Source: ${esc(r.source_stakeholder || 'missing')} &middot; Confidence: ${esc(r.confidence_score)}
      &middot; Risk: ${esc(r.risk_level || 'Unassessed')} &middot; Evidence: ${esc((r.evidence_ids || []).join(', ') || 'none')}</div>
    <div class="meta">Business justification: ${esc(r.business_justification || 'Not supplied')}</div>
    <div class="meta">Acceptance: ${esc((r.acceptance_criteria || []).join('; ') || 'missing')}</div>
    <div class="meta">Dependencies: ${esc((r.dependencies || []).join('; ') || 'none stated')}
      &middot; Assumptions: ${esc((r.assumptions || []).join('; ') || 'none stated')}</div>
    <div class="meta ${r.needs_review ? 'warn' : ''}">${r.needs_review ? 'Human review required' : 'Evidence similarity checked'}
      &middot; Quality issues: ${esc((r.quality_issues || []).join('; ') || 'none detected')}</div>
  </div>`).join('');
}
function renderAgentBoard(agents, reviews) {
  const uniqueAgents = [...new Map(agents.map(a => [a.name, a])).values()];
  $('agents').innerHTML = uniqueAgents.map(a => `<div class="item">
    <b>${esc(a.name)}</b> &mdash; ${esc(a.status)}
    <span class="badge alt">${esc(a.implementation || 'Rule-based')}</span>
    ${a.model ? `<span class="badge">${esc(a.model)}</span>` : ''}
    <div class="meta">${esc(a.responsibility || '')}</div></div>`).join('');
  $('agentReviews').innerHTML = reviews.map(review => `<div class="item">
    <b>${esc(review.name)}</b><div class="meta">${esc(review.summary || 'No summary returned.')}</div>
    ${review.rejected_unlinked_findings ? `<div class="meta warn">${esc(review.rejected_unlinked_findings)} finding(s) with unknown requirement IDs were excluded.</div>` : ''}
    ${review.findings.length ? review.findings.map(f => `<div class="item review">
      <b>${esc(f.requirement_id)} &middot; ${esc(f.severity)}</b>: ${esc(f.issue)}
      <div class="meta">Suggested follow-up: ${esc(f.recommendation)} &middot; Evidence: ${esc(f.evidence_ids.join(', ') || 'none linked')}</div>
      </div>`).join('') : '<div class="meta">No findings returned by this reviewer.</div>'}</div>`).join('');
}

async function runPipeline() {
  const btn = $('runBtn'), err = $('errorBox');
  if (($('p1').value === 'gemini' || $('p2').value === 'gemini') && !$('cloudConsent').checked) {
    err.textContent = 'Confirm cloud-provider data sharing before using Gemini, or choose Ollama for local inference.';
    err.style.display = 'block';
    return;
  }
  btn.disabled = true; err.style.display = 'none'; $('results').style.display = 'none';
  ['c1','c2','c3','c4'].forEach(id => $(id).style.display = 'none'); [1,2,3].forEach(n => setStep(n, ''));
  let current = 1;
  if (RESULT_CONTEXT) { RUN_ID = null; RESULT_CONTEXT = null; }
  ensureRunId();
  const common = {run_id:RUN_ID, docs:$('docs').value.trim(), extra_text:$('extra').value,
                  interview_transcript:$('interview').value, stakeholder_interviews:STAKEHOLDER_INTERVIEWS,
                  clarification_answers:$('clarificationAnswers').value,
                  stakeholder_role:$('stakeholderRole').value,
                  domain:$('domain').value.trim(),
                  functionality:$('functionality').value.trim(), top_k:parseInt($('topk').value || '4', 10)};
  const key = $('apiKey').value.trim();
  try {
    $('results').style.display = 'block';
    setStep(1, 'active'); btn.textContent = 'Step 1/3: asking context questions and retrieving evidence...';
    const r1 = await api('/api/retrieve', common);
    $('evidence').innerHTML = r1.evidence.length ? r1.evidence.map(e =>
      `<div class="item"><span class="badge">${esc(e.id)}</span><span class="badge src">${esc(e.source)}</span><span class="meta">score ${esc(e.score)}</span>
       <div class="meta">Jurisdiction: ${esc(e.metadata.jurisdiction)} &middot; Version: ${esc(e.metadata.version)}
       &middot; Effective date: ${esc(e.metadata.effective_date)} &middot; Applicability: ${esc(e.metadata.applicability)}</div>
       <div style="margin-top:6px;font-size:13px;color:#cbd5e1">${esc(e.text)}</div></div>`).join('')
      : '<div class="meta warn">No evidence matched. Any generated requirement will need human review; no regulatory facts should be inferred.</div>';
    $('c1').style.display = 'block'; setStep(1, 'done');

    current = 2; setStep(2, 'active'); btn.textContent = 'Step 2/3: requirement, security/privacy, and compliance agents are reviewing...';
    const r2 = await api('/api/requirements', {...common, evidence:r1.evidence, provider:$('p1').value,
      model:$('m1').value.trim(), api_key:key});
    renderRequirements(r2.requirements);
    $('reasoning').textContent = r2.reasoning; $('m1used').textContent = '(' + r2.model + ')';
    renderAgentBoard(r2.agents, r2.agent_reviews);
    $('conflicts').innerHTML = r2.conflicts.length ? r2.conflicts.map(c =>
      `<div class="item review">${esc(c.type)} (${esc(c.requirements.join(', '))}): ${esc(c.detail)}</div>`).join('')
      : '<div class="meta">No likely duplicates or direct conflicts detected by the prototype rules.</div>';
    $('c2').style.display = 'block'; setStep(2, 'done');

    current = 3; setStep(3, 'active'); btn.textContent = 'Step 3/3: ranking SDLC models and generating approval-gated workflow...';
    const r3 = await api('/api/sdlc', {...common, domain: common.domain, evidence: r1.evidence, requirements: r2.requirements,
      agent_reviews:r2.agent_reviews,
      functionality:common.functionality,
      stakeholder_role:common.stakeholder_role, run_id:RUN_ID,
      provider:$('p2').value, model:$('m2').value.trim(), api_key:key});
    $('sdlcName').textContent = r3.sdlc.recommended_sdlc; $('m2used').textContent = '(' + r3.model + ')';
    renderAgentBoard([...r2.agents, ...r3.agents], r3.agent_reviews);
    $('projectProfile').innerHTML = Object.entries(r3.project_factors).map(([name, level]) =>
      `<span class="badge alt">${esc(name.replace(/_/g, ' '))}: ${esc(level)}</span>`).join(' ');
    $('factors').innerHTML = r3.sdlc.key_factors.map(f => `<li>${esc(f)}</li>`).join('');
    $('justification').textContent = r3.sdlc.justification;
    $('sdlcMismatch').textContent = r3.sdlc.llm_agrees_with_ranking
      ? 'Language-model explanation agrees with the deterministic ranking.'
      : `Model suggested ${r3.sdlc.llm_suggestion}; the explainable matrix ranks ${r3.sdlc.recommended_sdlc} first. Human review is required.`;
    $('ranked').innerHTML = (r3.sdlc.ranked_recommendations || []).map(r =>
      `<div class="item"><b>${esc(r.model)}</b> &mdash; ${esc(r.score_percent)} relative fit points
      ${r.recommended ? '<span class="badge">Top ranked</span>' : ''}<div class="meta">Matrix score: ${esc(r.raw_score)}; high factors: ${esc(r.selection_factors.join(', '))}</div></div>`).join('');
    $('workflow').innerHTML = (r3.sdlc.workflow.phases || []).map(p =>
      `<div class="item"><b>${esc(p.number)}. ${esc(p.name)}</b>${p.human_approval_required ? ' <span class="badge">Human approval gate</span>' : ''}
       <div class="meta">Activities: ${esc(p.activities.join('; '))}</div>
       <div class="meta">Roles: ${esc(p.responsible_roles.join(', '))} &middot; Deliverables: ${esc(p.deliverables.join(', '))}</div>
       <div class="meta">Gate: ${esc(p.validation_gate)} &middot; Entry/exit criteria and traceability included</div></div>`).join('');
    $('governance').innerHTML = (r3.compliance.length ? r3.compliance.map(c =>
      `<div class="item"><b>${esc(c.control_domain)}</b> / ${esc(c.requirement_id)}: ${esc(c.control)}
      <div class="meta">Jurisdiction: ${esc(c.jurisdiction)}; ${esc(c.approval_checkpoint)}</div>
      <div class="meta">${esc(c.determination)}</div></div>`).join('')
      : '<div class="meta">No rule-based blockchain control mapping matched.</div>') +
      (r3.compliance_gaps || []).map(g => `<div class="item review"><b>Open control gap: ${esc(g.requirement_id)}</b>
        <div class="meta">${esc(g.gap)} &middot; Owner: ${esc(g.owner)}</div></div>`).join('') +
      '<div class="meta">All mappings are advisory; legal and compliance officers must confirm jurisdiction, applicability and current rules.</div>';
    $('riskRegister').innerHTML = r3.risk_register.map(r =>
      `<div class="item"><b>${esc(r.requirement_id)}</b> &mdash; ${esc(r.risk_level)} risk (${esc(r.risk_score)}/5)
      <div class="meta">Risk domains: ${esc(r.risk_domains.join(', '))} &middot; Owner: ${esc(r.owner)}
      &middot; Treatment: ${esc(r.treatment)}</div></div>`).join('');
    $('traceability').innerHTML = r3.artifacts.traceability_matrix.map(t =>
      `<div class="item"><b>${esc(t.requirement_id)}</b> &rarr; ${esc(t.source_stakeholder || 'unattributed')}
      &rarr; evidence ${esc(t.evidence_ids.join(', ') || 'none')} &rarr; ${esc(t.user_story_ids.join(', '))}, ${esc(t.use_case_ids.join(', '))}
      <div class="meta">Approval: ${esc(t.approval_status || 'Pending human approval')}</div></div>`).join('');
    $('artifactSummary').innerHTML =
      `<div class="item">SRS generated (${r3.artifacts.srs_markdown.length} characters); ${r3.artifacts.user_stories.length} user stories;
      ${r3.artifacts.use_cases.length} use cases; ${r3.artifacts.process_workflows.length} process workflows;
      ${r3.artifacts.data_requirements.length} data requirements; ${r3.artifacts.interface_requirements.length} interface requirements;
      ${r3.artifacts.assumption_dependency_register.length} assumption/dependency records.</div>`;
    $('openIssues').innerHTML = r3.artifacts.open_issues.length ? r3.artifacts.open_issues.map(i =>
      `<div class="item review">${esc(i.requirement_id)}: ${esc(i.issues.join('; ') || 'Evidence/quality review required')}</div>`).join('')
      : '<div class="meta">No outstanding rule-generated quality issue.</div>';
    ARTIFACT_BUNDLE = {run_id:RUN_ID, input:common, evidence:r1.evidence, requirements:r3.requirements,
      conflicts:r3.conflicts, project_factors:r3.project_factors, sdlc:r3.sdlc, compliance:r3.compliance,
      compliance_gaps:r3.compliance_gaps, risk_register:r3.risk_register, agents:r3.agents, artifacts:r3.artifacts};
    RESULT_CONTEXT = {common, evidence:r1.evidence, requirements:r3.requirements, result:r3};
    $('approvalTarget').value = r3.sdlc.recommended_sdlc;
    $('c4').style.display = 'block';
    $('c3').style.display = 'block'; setStep(3, 'done');
  } catch (e) {
    setStep(current, 'error'); err.textContent = 'Error: ' + e.message; err.style.display = 'block';
  } finally { btn.disabled = false; btn.textContent = 'Run pipeline'; }
}
async function submitApproval() {
  const body = {
    ...(RESULT_CONTEXT ? RESULT_CONTEXT.common : {}),
    run_id:RUN_ID, approval_type:$('approvalType').value, target_id:$('approvalTarget').value.trim(),
    action:$('approvalAction').value, reviewer:$('reviewer').value.trim(),
    role_token:$('approvalToken').value, mfa_code:$('mfaCode').value.trim(),
    modified_statement:$('modifiedStatement').value.trim(), comment:$('approvalComment').value.trim(),
    domain:RESULT_CONTEXT?.common.domain, functionality:RESULT_CONTEXT?.common.functionality,
    requirements:RESULT_CONTEXT?.requirements, evidence:RESULT_CONTEXT?.evidence,
    stakeholder_role:RESULT_CONTEXT?.common.stakeholder_role,
    provider:$('approvalType').value === 'sdlc' ? $('p2').value : $('p1').value,
    model:$('approvalType').value === 'sdlc' ? $('m2').value.trim() : $('m1').value.trim(),
    api_key:$('apiKey').value.trim(),
  };
  try {
    const data = await api('/api/approve', body);
    $('approvalStatus').textContent = `${data.approval.status}; reviewer role ${data.approval.role}; audit event ${data.approval.audit_event_id}`;
    ARTIFACT_BUNDLE.approvals = ARTIFACT_BUNDLE.approvals || [];
    ARTIFACT_BUNDLE.approvals.push(data.approval);
    if (data.approval.approval_type === 'requirement') {
      const current = ARTIFACT_BUNDLE.requirements.find(r => r.id === data.approval.target_id);
      if (current && data.approval.action === 'accept') current.approval_status = data.approval.status;
      if (current && data.approval.action === 'reject') current.approval_status = 'Rejected by ' + data.approval.role;
    }
    if (data.updated_requirements) {
      ARTIFACT_BUNDLE.requirements = data.updated_requirements;
      ARTIFACT_BUNDLE.artifacts = data.updated_artifacts;
      ARTIFACT_BUNDLE.compliance = data.updated_compliance;
      ARTIFACT_BUNDLE.compliance_gaps = data.updated_compliance_gaps;
      ARTIFACT_BUNDLE.risk_register = data.updated_risk_register;
      ARTIFACT_BUNDLE.conflicts = data.updated_conflicts;
      RESULT_CONTEXT.requirements = data.updated_requirements;
      renderRequirements(data.updated_requirements);
      if (data.sdlc_reassessment_required) {
        $('approvalStatus').textContent += ' SDLC recommendation should be re-evaluated after this change.';
      }
    }
    if (data.regenerated) {
      ARTIFACT_BUNDLE = {...ARTIFACT_BUNDLE, ...data.regenerated};
      RESULT_CONTEXT.requirements = data.regenerated.requirements;
      renderRequirements(data.regenerated.requirements);
      if (data.regenerated.artifacts) ARTIFACT_BUNDLE.artifacts = data.regenerated.artifacts;
    }
    $('approvalToken').value = ''; $('mfaCode').value = '';
  } catch (e) { $('approvalStatus').textContent = 'Approval was not recorded: ' + e.message; }
}
async function showAudit() {
  if (!RUN_ID) return;
  try {
    const response = await fetch('/api/audit?run_id=' + encodeURIComponent(RUN_ID));
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Audit query failed');
    $('auditTrail').innerHTML = data.events.map(e =>
      `<div class="item"><b>${esc(e.event_type)}</b> &middot; ${esc(e.actor)} &middot; ${esc(e.timestamp)}</div>`).join('')
      || '<div class="meta">No audit records found.</div>';
  } catch (e) { $('auditTrail').textContent = 'Could not load audit trail: ' + e.message; }
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


def _input_sources(body: dict) -> list:
    sources = []
    for key, label in (("questionnaire", "questionnaire.txt"),
                       ("meeting_notes", "meeting_notes.txt")):
        value = _text(body, key)
        if value:
            sources.append((label, value))
    interviews = body.get("stakeholder_interviews", [])
    if not isinstance(interviews, list) or len(interviews) > 30:
        raise ValueError("'stakeholder_interviews' must be a list of at most 30 interview records.")
    for index, item in enumerate(interviews):
        if not isinstance(item, dict):
            raise ValueError("Each stakeholder interview must include a role and transcript.")
        role = str(item.get("role") or "Stakeholder").strip()[:100]
        transcript = str(item.get("transcript") or "").strip()
        if len(transcript) > 500_000:
            raise ValueError(f"Stakeholder interview {index + 1} exceeds the 500 KB limit.")
        if transcript:
            sources.append((f"stakeholder_interview_{index + 1}.txt",
                            f"Stakeholder role/source: {role}\n{transcript}"))
    transcript = _text(body, "interview_transcript")
    if transcript:
        role = _text(body, "stakeholder_role") or "Stakeholder"
        sources.append(("current_stakeholder_interview.txt",
                        f"Stakeholder role/source: {role}\n{transcript}"))
    clarification_answers = _text(body, "clarification_answers")
    if clarification_answers:
        sources.append(("clarification_answers.txt", clarification_answers))
    supplied = body.get("source_documents", [])
    if not isinstance(supplied, list):
        raise ValueError("'source_documents' must be a list.")
    for index, item in enumerate(supplied):
        if not isinstance(item, dict):
            raise ValueError("Each source document must be an object with a name and text.")
        name, text = str(item.get("name") or f"stakeholder_source_{index + 1}.txt").strip(), str(item.get("text") or "")
        if len(text) > 500_000:
            raise ValueError(f"Source document '{name}' exceeds the 500 KB limit.")
        if text.strip():
            sources.append((name[:120], text))
    return sources


def _stakeholder_context(body: dict) -> str:
    entries = []
    interviews = body.get("stakeholder_interviews", [])
    if not isinstance(interviews, list) or len(interviews) > 30:
        raise ValueError("'stakeholder_interviews' must be a list of at most 30 interview records.")
    for index, item in enumerate(interviews):
        if not isinstance(item, dict):
            raise ValueError("Each stakeholder interview must include a role and transcript.")
        role = str(item.get("role") or "Stakeholder").strip()[:100]
        transcript = str(item.get("transcript") or "").strip()
        if len(transcript) > 500_000:
            raise ValueError(f"Stakeholder interview {index + 1} exceeds the 500 KB limit.")
        if transcript:
            entries.append(f"Stakeholder role/source: {role}\n{transcript}")
    transcript = _text(body, "interview_transcript")
    if transcript:
        entries.append(f"Stakeholder role/source: {_text(body, 'stakeholder_role') or 'Stakeholder'}\n{transcript}")
    for key in ("questionnaire", "meeting_notes", "extra_text", "clarification_answers"):
        value = _text(body, key)
        if value:
            entries.append(value)
    return "\n\n".join(entries)


def _retriever(body: dict):
    folder = _text(body, "docs")
    if folder:
        path = Path(folder).expanduser()
        folder = str(path if path.is_absolute() else BASE_DIR / path)
    documents = load_documents(folder, body.get("extra_text") or "", _input_sources(body))
    return build_index(documents, documents_root=folder)


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
        "approvals_enabled": bool(os.environ.get("BLOCKCHAIN_RAG_ROLE_TOKENS") and os.environ.get("BLOCKCHAIN_RAG_MFA_SECRET")),
        "source_extensions": [".txt", ".md", ".docx"],
        "audit_store": "Local SQLite; sensitive identifiers masked before storage",
        "warning": "Prototype: run on loopback only. Configure TLS, managed identity, MFA, and encrypted storage before handling sensitive or production data.",
        "agent_architecture": {
            "llm_agents": ["Requirement extraction", "Blockchain security and privacy reviewer",
                           "Blockchain compliance and governance reviewer", "SDLC selection"],
            "rule_based_agents": ["Classification", "Clarification and quality validation",
                                  "Conflict and duplicate detection", "Blockchain security and privacy rule checks",
                                  "Compliance control mapping", "Risk analysis"],
            "coordination": "Coordinator passes grounded requirements and evidence to independent specialist reviews, then includes their findings in SDLC analysis.",
            "model_note": "Four LLM agent roles use two configured model slots: requirements/review roles use LLM 1, and the SDLC agent infers project factors and makes an SDLC suggestion with LLM 2 before the deterministic matrix chooses the final recommendation.",
        },
    }


def api_retrieve(body: dict) -> dict:
    top_k = max(1, min(int(body.get("top_k") or 4), 8))
    domain, functionality = _text(body, "domain", True), _text(body, "functionality", True)
    context = " ".join([
        _text(body, key) for key in ("interview_transcript", "questionnaire", "meeting_notes",
                                     "extra_text", "clarification_answers")
    ])
    context += " " + " ".join(text for _, text in _input_sources(body))
    evidence = retrieve_evidence(_retriever(body), domain, f"{functionality}\n{context[:4000]}", top_k)
    run_id = _text(body, "run_id")
    if run_id:
        record_event(run_id, "retrieval_completed", _text(body, "stakeholder_role") or "Coordinator",
                     {"domain": domain, "functionality": functionality, "evidence": evidence})
    return {"evidence": evidence, "run_id": run_id}


def api_interview(body: dict) -> dict:
    role = _text(body, "stakeholder_role") or "Stakeholder"
    text = _text(body, "text")
    questions = clarification_questions(text, role)
    run_id = _text(body, "run_id")
    if run_id:
        record_event(run_id, "clarification_questions_generated", role, {"context": text, "questions": questions})
    return {"questions": questions}


def api_requirements(body: dict) -> dict:
    domain, functionality = _text(body, "domain", True), _text(body, "functionality", True)
    evidence = _list_of_dicts(body, "evidence", ("id", "source", "text", "metadata"))
    stakeholder_role = _text(body, "stakeholder_role") or "Stakeholder"
    stakeholder_context = _stakeholder_context(body)
    llm1 = build_client(_text(body, "provider") or "ollama", _text(body, "model") or None, _text(body, "api_key") or None, role="llm1")
    draft = generate_requirements(domain, functionality, evidence, llm1, stakeholder_role, stakeholder_context)
    reqs = ground_requirements(_retriever(body), draft["requirements"], evidence)
    specialist = run_specialist_agents(reqs, evidence, stakeholder_role)
    review_board = run_review_agents(llm1, specialist["requirements"], evidence)
    conflict_ids = {rid for conflict in specialist["conflicts"] for rid in conflict["requirements"]}
    reqs = specialist["requirements"]
    reviewed_ids = {
        finding["requirement_id"]
        for review in review_board["reviews"] for finding in review["findings"]
    }
    for req in reqs:
        req["needs_review"] = (req["needs_review"] or bool(req["quality_issues"])
                               or req["id"] in conflict_ids or req["id"] in reviewed_ids)
    extraction_agent = {
        "name": "Requirement extraction", "status": "completed", "implementation": "LLM",
        "model": llm1.label, "responsibility": "Draft evidence-grounded structured requirements",
    }
    agents = [extraction_agent] + specialist["agents"] + review_board["agents"]
    run_id = _text(body, "run_id")
    if run_id:
        record_event(run_id, "requirement_extraction_prompt", stakeholder_role, draft["prompt"])
        record_event(run_id, "requirements_and_validation", stakeholder_role, {
            "requirements": reqs, "conflicts": specialist["conflicts"], "agents": agents,
            "agent_reviews": review_board["reviews"],
        })
    return {
        "requirements": reqs, "reasoning": draft["reasoning"], "model": llm1.label,
        "needs_review_count": sum(r["needs_review"] for r in reqs),
        "conflicts": specialist["conflicts"], "agents": agents, "agent_reviews": review_board["reviews"],
        "clarification_questions": specialist["clarification_questions"],
    }


def api_sdlc(body: dict) -> dict:
    domain = _text(body, "domain", True)
    evidence = _list_of_dicts(body, "evidence", ("id", "source", "text", "metadata"))
    reqs = _list_of_dicts(body, "requirements", ("id", "statement", "category"))
    stakeholder_role = _text(body, "stakeholder_role") or "Stakeholder"
    stakeholder_context = _stakeholder_context(body)
    specialist = run_specialist_agents(reqs, evidence, stakeholder_role)
    agent_reviews = validate_agent_reviews(body.get("agent_reviews", []), reqs, evidence)
    llm2 = build_client(_text(body, "provider") or "ollama", _text(body, "model") or None, _text(body, "api_key") or None, role="llm2")
    sdlc = identify_sdlc(domain, specialist["requirements"], evidence, llm2, agent_reviews=agent_reviews,
                         stakeholder_context=stakeholder_context)
    factors = sdlc["project_factors"]
    ranked = ranked_sdlc(factors)
    artifacts = generate_artifacts(domain, _text(body, "functionality"), specialist["requirements"],
                                   specialist["compliance"], specialist["risk_register"], specialist["conflicts"])
    result = {
        "sdlc": {**sdlc, "ranked_recommendations": ranked,
                 "workflow": generate_workflow(sdlc["recommended_sdlc"])},
        "requirements": specialist["requirements"],
        "compliance": specialist["compliance"], "risk_register": specialist["risk_register"],
        "compliance_gaps": specialist["compliance_gaps"],
        "conflicts": specialist["conflicts"], "agents": specialist["agents"] + [
            {"name": "SDLC selection", "status": "completed", "implementation": "Weighted matrix + LLM explanation",
             "model": llm2.label, "responsibility": "Rank approaches deterministically and explain trade-offs"},
            {"name": "Documentation and traceability", "status": "completed", "implementation": "Rule-based",
             "responsibility": "Generate linked project artifacts"},
            {"name": "Human approval", "status": "required before baseline/release", "implementation": "Human reviewer"},
        ],
        "agent_reviews": agent_reviews, "artifacts": artifacts, "project_factors": factors, "model": llm2.label,
    }
    run_id = _text(body, "run_id")
    if run_id:
        system, prompt = stage2_messages(domain, specialist["requirements"], evidence,
                                         agent_reviews=agent_reviews, stakeholder_context=stakeholder_context)
        record_event(run_id, "sdlc_decision_prompt", stakeholder_role, {"system": system, "user": prompt})
        record_event(run_id, "sdlc_and_artifacts_generated", "Coordinator", result)
    return result


def api_approve(body: dict) -> dict:
    target_id = _text(body, "target_id", True)
    approval_type = _text(body, "approval_type", True).lower()
    action = _text(body, "action", True).lower()
    if action not in ("accept", "reject", "modify", "regenerate"):
        raise ValueError("action must be accept, reject, modify or regenerate")
    role = authorize_approval(_text(body, "role_token"), _text(body, "mfa_code"), approval_type)
    modified_statement = sanitize_prompt_text(_text(body, "modified_statement"))
    if action == "modify" and not modified_statement:
        raise ValueError("A proposed modified statement is required for the modify action.")
    if action == "modify" and approval_type != "requirement":
        raise ValueError("Only requirement decisions can be modified directly; request regeneration for other outputs.")
    modified_requirements = None
    if action == "modify":
        modified_requirements = _list_of_dicts(body, "requirements", ("id", "statement", "category"))
        match = next((item for item in modified_requirements if item.get("id") == target_id), None)
        if not match:
            raise ValueError(f"Requirement '{target_id}' was not found in this run.")
        match["statement"] = modified_statement
        match["approval_status"] = "Pending re-approval after modification"
    run_id = _text(body, "run_id", True)
    approval_record = {
        "target_id": target_id,
        "approval_type": approval_type,
        "action": action,
        "reviewer": _text(body, "reviewer", True),
        "role": role,
        "comment": _text(body, "comment"),
        "modified_statement": modified_statement if action == "modify" else "",
    }
    event = record_event(run_id, "human_approval_decision", role, approval_record)
    required_roles = sorted({
        "project_manager", "software_architect", "security_officer", "compliance_officer"
    }) if approval_type == "sdlc" else [role]
    prior = get_run_events(run_id)
    role_decisions = {}
    for item in prior:
        payload = item["payload"]
        if (item["event_type"] == "human_approval_decision"
                and payload.get("approval_type") == approval_type and payload.get("target_id") == target_id):
            role_decisions[payload.get("role")] = payload.get("action")
    approved_roles = sorted(role for role, decision in role_decisions.items() if decision == "accept")
    status = "approved" if set(required_roles).issubset(approved_roles) else "pending required reviewers"
    if action == "reject":
        status = "rejected"
    elif action == "modify":
        status = "modification proposed; regenerate and re-review"
    elif action == "regenerate":
        status = "regeneration requested"
    response = {
        "approval": {
            "target_id": target_id, "approval_type": approval_type, "action": action,
            "reviewer": approval_record["reviewer"], "role": role, "status": status,
            "approved_roles": approved_roles, "required_roles": required_roles,
            "audit_event_id": event["event_id"],
        }
    }

    if action == "modify":
        evidence = _list_of_dicts(body, "evidence", ("id", "source", "text", "metadata"))
        requirements_with_support = ground_requirements(_retriever(body), modified_requirements, evidence)
        specialist = run_specialist_agents(requirements_with_support, evidence, _text(body, "stakeholder_role") or role)
        conflict_ids = {rid for conflict in specialist["conflicts"] for rid in conflict["requirements"]}
        for req in specialist["requirements"]:
            req["needs_review"] = (req.get("needs_review", False) or bool(req.get("quality_issues"))
                                   or req["id"] in conflict_ids)
        updated_artifacts = generate_artifacts(
            _text(body, "domain", True), _text(body, "functionality"),
            specialist["requirements"], specialist["compliance"], specialist["risk_register"],
            specialist["conflicts"],
        )
        record_event(run_id, "human_requirement_modified", role, {
            "requirement_id": target_id, "new_statement": modified_statement,
            "requirements": specialist["requirements"], "artifacts": updated_artifacts,
        })
        response["updated_requirements"] = specialist["requirements"]
        response["updated_artifacts"] = updated_artifacts
        response["updated_compliance"] = specialist["compliance"]
        response["updated_compliance_gaps"] = specialist["compliance_gaps"]
        response["updated_risk_register"] = specialist["risk_register"]
        response["updated_conflicts"] = specialist["conflicts"]
        response["sdlc_reassessment_required"] = True

    if approval_type == "requirement" and action in ("accept", "reject"):
        requirements = _list_of_dicts(body, "requirements", ("id", "statement", "category"))
        match = next((item for item in requirements if item.get("id") == target_id), None)
        if not match:
            raise ValueError(f"Requirement '{target_id}' was not found in this run.")
        match["approval_status"] = ("Approved by " if action == "accept" else "Rejected by ") + role
        evidence = _list_of_dicts(body, "evidence", ("id", "source", "text", "metadata"))
        specialist = run_specialist_agents(requirements, evidence, _text(body, "stakeholder_role") or role)
        artifacts = generate_artifacts(
            _text(body, "domain", True), _text(body, "functionality"),
            specialist["requirements"], specialist["compliance"], specialist["risk_register"], specialist["conflicts"],
        )
        response["updated_requirements"] = specialist["requirements"]
        response["updated_artifacts"] = artifacts
        response["updated_compliance"] = specialist["compliance"]
        response["updated_compliance_gaps"] = specialist["compliance_gaps"]
        response["updated_risk_register"] = specialist["risk_register"]
        response["updated_conflicts"] = specialist["conflicts"]
        record_event(run_id, "approved_artifacts_updated", role, {
            "requirement_id": target_id, "approval_status": match["approval_status"], "artifacts": artifacts,
        })

    if action == "regenerate":
        if approval_type in ("requirement", "security", "compliance"):
            evidence = _list_of_dicts(body, "evidence", ("id", "source", "text", "metadata"))
            domain, functionality = _text(body, "domain", True), _text(body, "functionality", True)
            llm1 = build_client(_text(body, "provider") or "ollama", _text(body, "model") or None,
                                _text(body, "api_key") or None, role="llm1")
            draft = generate_requirements(domain, functionality, evidence, llm1,
                                          _text(body, "stakeholder_role") or "Stakeholder",
                                          "\n".join(_text(body, key) for key in ("interview_transcript", "questionnaire", "meeting_notes")))
            requirements = ground_requirements(_retriever(body), draft["requirements"], evidence)
            specialist = run_specialist_agents(requirements, evidence, _text(body, "stakeholder_role") or "Stakeholder")
            conflict_ids = {rid for conflict in specialist["conflicts"] for rid in conflict["requirements"]}
            for requirement in specialist["requirements"]:
                requirement["needs_review"] = (requirement.get("needs_review", False)
                                               or bool(requirement.get("quality_issues"))
                                               or requirement["id"] in conflict_ids)
            regenerated = {
                "requirements": specialist["requirements"],
                "requirements_reasoning": draft["reasoning"],
                "conflicts": specialist["conflicts"],
                "compliance": specialist["compliance"],
                "risk_register": specialist["risk_register"],
                "artifacts": generate_artifacts(domain, functionality, specialist["requirements"],
                                                specialist["compliance"], specialist["risk_register"],
                                                specialist["conflicts"]),
            }
            response["regenerated"] = regenerated
            record_event(run_id, "requirements_regenerated", role, regenerated)
        elif approval_type == "sdlc":
            response["regenerated"] = api_sdlc(body)
        else:
            raise ValueError("Regeneration is supported for requirements and SDLC decisions.")
    return response


ROUTES = {"/api/retrieve": api_retrieve, "/api/interview": api_interview,
          "/api/requirements": api_requirements, "/api/sdlc": api_sdlc, "/api/approve": api_approve}


def api_audit(run_id: str) -> dict:
    return {"run_id": run_id, "events": get_run_events(run_id)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, content_type: str = "application/json") -> None:
        data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE, "text/html")
        elif parsed.path == "/api/config":
            self._send(200, api_config())
        elif parsed.path == "/api/audit":
            run_id = parse_qs(parsed.query).get("run_id", [""])[0]
            try:
                self._send(200, api_audit(run_id))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        handler = ROUTES.get(self.path)
        if not handler:
            return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 2_000_000:
                return self._send(413, {"error": "Request exceeds the 2 MB prototype limit."})
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            self._send(200, handler(body))
        except AuthorizationError as e:
            self._send(403, {"error": str(e)})
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
    cert, key = os.environ.get("BLOCKCHAIN_RAG_TLS_CERT"), os.environ.get("BLOCKCHAIN_RAG_TLS_KEY")
    if cert or key:
        if not cert or not key:
            server.server_close()
            raise SystemExit("Set both BLOCKCHAIN_RAG_TLS_CERT and BLOCKCHAIN_RAG_TLS_KEY for HTTPS.")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert, keyfile=key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        url = f"https://localhost:{PORT}"
    print(f"\n[ok] Website running at {url}   (Ctrl+C to stop)")
    threading.Timer(0.8, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
