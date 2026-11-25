"""Pipeline (retrieve first, then generate - the order used by retrieval-grounded generation):

  input (domain + functionality)
    -> Step 1  RAG retrieval: find the most relevant chunks of the existing requirement documents
    -> Step 2  LLM 1: generate requirements + reasoning, grounded in that evidence
               (+ a deterministic check of how well each requirement is supported by the evidence)
    -> Step 3  LLM 2: identify the SDLC model (its ONLY job) from the requirements + evidence
    -> final output

The prompt builders below are also used by generate_dataset.py, so the fine-tuning data
uses exactly the same prompts the pipeline uses at run time.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from llm_clients import LLMClient, LLMError
from retriever import Chunk, TfidfRetriever, chunk_text

TEXT_EXTENSIONS = (".txt", ".md")
MIN_EVIDENCE_SCORE = 0.02      # chunks scoring below this are treated as irrelevant
MIN_GROUNDING = 0.10           # requirement/evidence similarity below this => flag for human review

SDLC_LABELS = [
    "Waterfall", "V-Model", "Spiral", "Agile", "DevSecOps",
    "Agile-V-Model Hybrid", "DevSecOps-V-Model Hybrid",
]

SDLC_HEURISTICS = """Reference SDLC heuristics:
- Stable requirements and extensive approvals needed -> Waterfall
- Strict verification and validation needed (e.g. formally verified logic) -> V-Model
- High uncertainty or technical risk -> Spiral
- Frequently changing requirements -> Agile
- Continuous secure deployment with rigorous automated security testing -> DevSecOps
- High regulation with evolving requirements and high-stakes deployment -> Agile-V-Model or DevSecOps hybrid"""

STAGE1_SCHEMA = """Return ONLY strict JSON (no markdown fences):
{
 "requirements": [{"id": "REQ-001", "statement": "one clear, testable requirement", "category": "short category"}],
 "reasoning": "one paragraph explaining how the evidence and the functionality led to these requirements"
}"""

STAGE2_SCHEMA = """Return ONLY strict JSON (no markdown fences). Identifying the SDLC model is your ONLY task:
{
 "recommended_sdlc": "model name",
 "key_factors": ["3-4 short project characteristics, drawn from the requirements and evidence, that drove this choice"],
 "justification": "a software-engineering justification paragraph, briefly noting why the main alternative(s) fit less well"
}"""


# ----------------------------------------------------------------------------- documents / index
def load_documents(folder: str, extra_text: str = "") -> List[Tuple[str, str]]:
    """Read every .txt/.md file in `folder`; optionally add pasted text as one more document."""
    docs: List[Tuple[str, str]] = []
    if folder:
        if not os.path.isdir(folder):
            raise ValueError(f"Document folder not found: {folder}")
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(TEXT_EXTENSIONS):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append((name, f.read()))
    if extra_text and extra_text.strip():
        docs.append(("pasted_text", extra_text))
    if not docs:
        raise ValueError("No requirement documents found (.txt/.md files, or pasted text).")
    return docs


def build_index(documents: List[Tuple[str, str]]) -> TfidfRetriever:
    chunks: List[Chunk] = []
    for source, text in documents:
        chunks.extend(chunk_text(text, source))
    retriever = TfidfRetriever()
    retriever.index(chunks)
    return retriever


# ----------------------------------------------------------------------------- prompts
def format_evidence(evidence: List[dict]) -> str:
    if not evidence:
        return "(no relevant evidence was found in the existing documents)"
    return "\n\n".join(f"[{e['id']}] (source: {e['source']}) {e['text']}" for e in evidence)


def stage1_messages(domain: str, functionality: str, evidence: List[dict]) -> Tuple[str, str]:
    system = f"You are a requirements engineer for the '{domain}' domain in blockchain-based finance."
    user = (
        f'Functionality: "{functionality}"\n\n'
        f'Evidence retrieved from the existing requirement documents:\n"""\n{format_evidence(evidence)}\n"""\n\n'
        "Write 4-6 specific, testable requirements for this functionality. Base them on the evidence wherever it "
        "applies; add standard domain requirements only where the evidence is silent. Each statement must be one "
        "sentence using \"shall\". Then explain your reasoning in one paragraph.\n\n" + STAGE1_SCHEMA
    )
    return system, user


def requirements_text(requirements: List[dict]) -> str:
    return "\n".join(f"{r.get('id', '')}: {r.get('statement', '')} [{r.get('category', '')}]" for r in requirements)


def stage2_messages(domain: str, requirements: List[dict], evidence: List[dict]) -> Tuple[str, str]:
    system = (
        f"You identify the single most appropriate SDLC model for a blockchain-finance project in the '{domain}' "
        "domain. This is your only task -- do not propose or revise requirements."
    )
    user = (
        f'Requirements:\n"""\n{requirements_text(requirements)}\n"""\n\n'
        f'Supporting evidence from the existing documents:\n"""\n{format_evidence(evidence)}\n"""\n\n'
        f"{SDLC_HEURISTICS}\n\n"
        "Using the requirements and the evidence together, identify the single most appropriate SDLC model "
        f"(choose from: {', '.join(SDLC_LABELS)}).\n\n" + STAGE2_SCHEMA
    )
    return system, user


# ----------------------------------------------------------------------------- normalisation (small models are messy)
def _get(d: dict, *names: str, default=""):
    lowered = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n in lowered and lowered[n] not in (None, ""):
            return lowered[n]
    return default


def normalize_requirements(raw) -> List[dict]:
    items = raw if isinstance(raw, list) else []
    out: List[dict] = []
    for item in items:
        if isinstance(item, str):
            item = {"statement": item}
        if not isinstance(item, dict):
            continue
        statement = str(_get(item, "statement", "requirement", "text", "description")).strip()
        if not statement:
            continue
        out.append({
            "id": str(_get(item, "id", default="")).strip() or f"REQ-{len(out) + 1:03d}",
            "statement": statement,
            "category": str(_get(item, "category", "type", default="General")).strip() or "General",
        })
    return out


def normalize_sdlc(raw: dict) -> dict:
    factors = _get(raw, "key_factors", "factors", default=[])
    if isinstance(factors, str):
        factors = [factors]
    return {
        "recommended_sdlc": str(_get(raw, "recommended_sdlc", "sdlc", "model", default="")).strip(),
        "key_factors": [str(f) for f in factors if str(f).strip()],
        "justification": str(_get(raw, "justification", "reasoning", default="")).strip(),
    }


# ----------------------------------------------------------------------------- the three steps
def retrieve_evidence(retriever: TfidfRetriever, domain: str, functionality: str, top_k: int = 4) -> List[dict]:
    """Step 1 - RAG: query the existing documents with the user's input."""
    hits = retriever.retrieve_scored(f"{domain}: {functionality}", top_k=top_k, min_score=MIN_EVIDENCE_SCORE)
    return [{"id": f"E{i + 1}", "source": c.source, "text": c.text, "score": round(s, 3)} for i, (c, s) in enumerate(hits)]


def generate_requirements(domain: str, functionality: str, evidence: List[dict], llm1: LLMClient) -> dict:
    """Step 2 - LLM 1 drafts requirements, grounded in the retrieved evidence."""
    system, user = stage1_messages(domain, functionality, evidence)
    raw = llm1.generate_json(system, user)
    requirements = normalize_requirements(_get(raw, "requirements", default=[]))
    if not requirements:
        raise LLMError(f"{llm1.label} returned no usable requirements. Try again or use a larger model.")
    return {"requirements": requirements, "reasoning": str(_get(raw, "reasoning", "explanation", default="")).strip()}


def ground_requirements(retriever: TfidfRetriever, requirements: List[dict], evidence: List[dict],
                        min_grounding: float = MIN_GROUNDING) -> List[dict]:
    """Deterministic check (no LLM): how closely does each requirement match the retrieved evidence?
    Requirements with weak support are flagged for human review."""
    chunks = [Chunk(text=e["text"], source=e["source"]) for e in evidence]
    out = []
    for r in requirements:
        chunk, score = retriever.best_support(r["statement"], chunks)
        out.append({**r,
                    "support_source": chunk.source if chunk else None,
                    "support_score": round(score, 3),
                    "needs_review": score < min_grounding})
    return out


def identify_sdlc(domain: str, requirements: List[dict], evidence: List[dict], llm2: LLMClient) -> dict:
    """Step 3 - LLM 2: its only job is naming the SDLC model, with a justification."""
    system, user = stage2_messages(domain, requirements, evidence)
    sdlc = normalize_sdlc(llm2.generate_json(system, user))
    if not sdlc["recommended_sdlc"]:
        raise LLMError(f"{llm2.label} did not name an SDLC model. Try again or use a larger model.")
    return sdlc


def run_pipeline(retriever: TfidfRetriever, domain: str, functionality: str, llm1: LLMClient, llm2: LLMClient,
                 top_k: int = 4, min_grounding: float = MIN_GROUNDING,
                 progress: Optional[Callable[[str], None]] = None) -> Dict:
    say = progress or (lambda _msg: None)
    say("Step 1/3: retrieving evidence (RAG)")
    evidence = retrieve_evidence(retriever, domain, functionality, top_k)
    say("Step 2/3: LLM 1 generating requirements")
    draft = generate_requirements(domain, functionality, evidence, llm1)
    requirements = ground_requirements(retriever, draft["requirements"], evidence, min_grounding)
    say("Step 3/3: LLM 2 identifying the SDLC model")
    sdlc = identify_sdlc(domain, requirements, evidence, llm2)
    return {
        "input": {"domain": domain, "functionality": functionality},
        "evidence": evidence,
        "requirements": requirements,
        "requirements_reasoning": draft["reasoning"],
        "needs_review_count": sum(1 for r in requirements if r["needs_review"]),
        "sdlc": sdlc,
        "models": {"llm1": llm1.label, "llm2": llm2.label},
    }
