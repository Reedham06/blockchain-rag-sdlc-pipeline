# Blockchain & Digital Asset Requirements Engineering

This project adapts the supplied Agentic AI requirements-engineering problem statement to the **blockchain and
digital-asset domain**. It is an advisory prototype for use cases such as DeFi lending, wallet custody, stablecoins,
cross-border payments, tokenised assets, KYC/AML and CBDCs. It does not treat the source problem's financial-sector
wording as the project's domain; blockchain concepts (smart contracts, wallets, keys, transaction integrity,
consensus/finality, oracles, bridges and on-chain/off-chain data) shape the intake, controls and outputs.

## What the prototype does

1. **Stakeholder intake:** add multiple role-tagged interviews to the current browser run, plus clarification
   answers, questionnaire answers, meeting notes and existing knowledge-base documents. The interview helper
   suggests rule-based follow-up questions about goals, roles, chain/contract scope, keys, data placement,
   integrations, measurable quality targets, acceptance and jurisdiction. Added interview entries are held in the
   browser tab until submitted; they are not saved as reusable files.
2. **Approved-source RAG:** retrieve relevant `.txt`, `.md` and `.docx` material. Sources can carry jurisdiction,
   effective date, version and applicability in a `<filename>.meta.json` sidecar or the corpus-level
   `source_metadata.json`. Unknown values are not guessed. The bundled policy examples are synthetic, not law.
3. **Coordinated multi-agent workflow:** a requirement-extraction LLM agent drafts requirements; independent,
   role-prompted blockchain security/privacy and compliance/governance LLM agents review the same grounded
   requirement/evidence board; and an SDLC LLM agent explains a separately calculated decision-matrix result.
   The coordinator passes the specialist findings into the SDLC step. These four LLM agent roles use two
   configured model slots: extraction and the two reviews call LLM 1 as separate task-scoped interactions, while
   SDLC explanation calls LLM 2. They are distinct agent tasks, not four separately installed foundation models.
4. **Rule-based specialist agents:** six explicit rule-driven agents classify requirements; test quality and
   clarification gaps; detect likely conflicts/duplicates; check blockchain security/privacy keywords; map
   advisory control themes; and assess risk. Their implementation is identified as rule-based in the website rather
   than being presented as independent LLM reasoning.
5. **Structured outputs:** requirement IDs/statements/categories, stakeholder source, business justification,
   priority, dependencies, assumptions, acceptance criteria, applicable-regulation fields, risk, confidence,
   evidence and pending human-approval status.
6. **Explainable SDLC selection:** LLM 2 infers low/medium/high project ratings from the use case, stakeholder
   interviews and answers, requirements, specialist reviews and retrieved evidence. A deterministic weighted
   decision matrix ranks Waterfall, V-Model, Spiral, Agile, Incremental, and RAD from those inferred ratings;
   LLM 2's own suggestion is advisory and does not overrule the matrix. The inferred profile is displayed for
   review. Scores are relative fit points, not probabilities.
7. **Project artifacts:** SRS, user stories, use cases, process/data/interface requirements, a risk register,
   compliance-control matrix, assumptions/dependencies, open issues and a requirements traceability matrix. The
   selected SDLC gets phases with roles, activities, deliverables, validation gates, security/compliance activities,
   entry/exit criteria and human approvals.
8. **Human decisions and audit:** accept, reject, propose a modification or request regeneration. Approvals are
   role-authorised and require TOTP MFA when configured; SDLC approval requires separate project-manager,
   architect, security and compliance approvals. Masked prompts, retrieved evidence, generated decisions,
   modifications and approvals are written to a run-scoped SQLite audit log.

## Run locally

Requirements: Python 3.9+ and [Ollama](https://ollama.com). The normal pipeline runs locally; Gemini is an
optional cloud provider.

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
python web_app.py
```

Open `http://localhost:8000` if the browser does not open automatically. Choose a blockchain use case; for each
interview, select the speaker's stakeholder role, paste notes, and click **Add interview to this run**. Use
**Suggest missing topics (rule-based)** to see clarification questions and enter any answers. The SDLC agent derives
the project profile from your supplied material; there are no characteristic-rating fields to fill in. Evidence,
quality issues, conflicts, risk/control mappings, SDLC ranking, workflow and downloadable JSON artifacts appear in
the results. Ollama must be running; on Windows, start the Ollama app before running the website.

The bundled `sample_docs/` corpus is an example, not authoritative guidance. Replace it with approved blockchain
policies, contract standards, architecture documents, incident reports, control catalogs and jurisdiction-specific
sources before relying on its mappings.

The stakeholder role selector tags the next interview entry; it does not create a separate saved record by itself.
Interview text in the browser is not durably stored as a transcript file. Retrieved masked evidence and pipeline
events are recorded in the local audit database; masking is not guaranteed anonymisation.

The website's **Coordinated agent team** panel labels each entry as `LLM`, `Rule-based`, `Weighted matrix + LLM
explanation`, or `Human reviewer`. The security/privacy and compliance/governance agents return separate finding
lists; the SDLC prompt receives those lists, and the website displays their findings and linked evidence IDs.
The SDLC selector considers six approaches: Waterfall, V-Model, Spiral, Agile, Incremental and RAD.

### CLI

```powershell
python main.py --domain "Blockchain Custody" --functionality "Multi-signature approval for high-value withdrawals" `
  --stakeholder-role "Product Owner" --json-out result.json
```

Run without an Ollama server by using the Python test suite, which supplies deterministic fake LLM responses:

```powershell
python -m unittest -v
```

## Knowledge-base sources and metadata

Place approved `.txt`, `.md` or `.docx` files in the selected folder. For a source named `wallet_policy.md`, an
optional `wallet_policy.md.meta.json` can contain:

```json
{
  "jurisdiction": "organisation-confirmed jurisdiction",
  "effective_date": "2026-01-01",
  "version": "2.1",
  "applicability": "approved custody systems"
}
```

Alternatively, `source_metadata.json` can map document filenames to those fields. Missing metadata is surfaced as
unknown/unconfirmed. Source extensions are allowlisted; symbolic links are skipped. The browser also accepts
interview transcripts and notes. Anonymise them before use. Sensitive identifiers are masked before model prompts
and audit logging, but masking is not a guarantee of anonymisation.

## Human approvals and security setup

The server binds to loopback (`127.0.0.1`) only. By default, generation works but approval actions fail closed.
To enable approval testing, configure one token for each needed reviewer role and a shared Base32 TOTP secret in
the same PowerShell session that starts the server:

```powershell
$env:BLOCKCHAIN_RAG_ROLE_TOKENS='{"business_analyst":"replace-with-a-long-random-token","product_owner":"replace-with-a-different-long-random-token","project_manager":"replace-token","software_architect":"replace-token","security_officer":"replace-token","compliance_officer":"replace-token","release_manager":"replace-token"}'
$env:BLOCKCHAIN_RAG_MFA_SECRET='YOUR_BASE32_TOTP_SECRET'
python web_app.py
```

Use unique random tokens and enrol the Base32 secret in an authenticator app. Supported approval roles and scopes
are enforced in `access_control.py`. Do not commit credentials. To enable HTTPS for the local server, set both
`BLOCKCHAIN_RAG_TLS_CERT` and `BLOCKCHAIN_RAG_TLS_KEY` to a certificate/key pair. Audit records are stored in
`data/audit.sqlite3`; `BLOCKCHAIN_RAG_AUDIT_DB` can point elsewhere and
`BLOCKCHAIN_RAG_AUDIT_RETENTION_DAYS` controls retention (1-3650 days).

**Prototype security boundary:** SQLite audit data is not application-encrypted; use OS/full-volume encryption and
keep the service on loopback. This prototype does not provide enterprise identity federation, production-grade
key management, network deployment hardening, or legal compliance decisions. Do not send sensitive production
data or expose the server to a network without a security review and additional controls. When Gemini is selected,
the UI requires explicit consent to send masked prompts/evidence to the cloud provider.

## Evaluation

Use representative, anonymised blockchain case studies (DeFi lending/liquidation, custody/key recovery,
stablecoin reserves, cross-chain payments, tokenised assets, KYC/AML, and CBDC). Compare against expert-labelled
requirements and experienced architecture decisions. Measure precision/recall/F1, completeness, ambiguity and
conflict detection, control coverage, citation correctness, unsupported-claim rate, SDLC suitability, correction
rate, processing time, stakeholder satisfaction and traceability coverage. See [EVALUATION.md](./EVALUATION.md) for
a measurement worksheet and interpretation notes.

## Fine-tuning (optional)

The included `seed_data.py`, generated JSONL files and `finetune_colab.ipynb` contain 28 synthetic blockchain
scenarios per model. Stage 1 trains LLM 1 on stakeholder/use-case inputs to draft structured requirements. Stage 2
trains LLM 2 on requirements, evidence and project-context notes to infer all twelve low/medium/high project
characteristics and recommend an SDLC approach. The dataset builder creates illustrative factor labels from the
hand-authored examples; review and improve those labels with expert-annotated project cases before relying on a
fine-tuned model. Fine-tuning is optional; the base models use the same runtime output schemas. If project code
changes, regenerate the embedded Colab notebook with:

```powershell
python make_notebook.py
```

Fine-tuned Ollama models can be registered using `python install_models.py`; model export files are intentionally
not bundled because of their size.

## Project map

| File | Purpose |
|---|---|
| `web_app.py` | Local stakeholder UI, API, workflow, and human review |
| `pipeline.py` | Retrieval, prompts, extraction, grounding and orchestration |
| `agent_team.py` | Task-scoped security/privacy and compliance/governance LLM review agents |
| `project_engine.py` | Specialist agents, quality/conflict rules, SDLC matrix and artifacts |
| `retriever.py` | Paragraph chunking and TF-IDF retrieval |
| `audit_store.py` | Run-scoped persistent masked audit events |
| `access_control.py` | Role-token authorization and TOTP MFA for approvals |
| `main.py` | CLI runner |
| `sample_docs/` | Synthetic blockchain/digital-asset policies and source metadata |
| `EVALUATION.md` | Evaluation measurements and guidance |
| `test_pipeline.py` | Offline regression, security-control and API tests |

## Known limitations

- Rule-based checks and TF-IDF are useful prototype baselines, not proof of correctness or legal compliance.
- The bundled policies are synthetic. Replace and version them with approved sources, and have legal/compliance
  owners validate jurisdiction, effective dates, applicability and mappings.
- Conflict, ambiguity, testability and risk detectors can miss issues or raise false positives; human review is
  mandatory, especially for smart-contract, key-custody, transaction and production decisions.
- TOTP is a local prototype approval control, not a substitute for an organisation's identity provider or
  privileged-access management.
- Scores are relative decision-matrix values rather than calibrated probabilities. Validate weights with project
  architects and stakeholders.
- Evaluate on anonymised labelled cases before any operational use. The project makes no accuracy or time-saving
  claims without that evaluation.
