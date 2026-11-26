"""Specialist deterministic agents and artifact generation for blockchain projects."""
from __future__ import annotations

import re
import math
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional


def mask_sensitive_text(text: str) -> str:
    """Mask common identifiers before content is sent to a language model or audit store."""
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "[IBAN]", text, flags=re.I)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[NATIONAL_ID]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[PAYMENT_CARD]", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[IP_ADDRESS]", text)
    text = re.sub(r"\b(?:\+?\d[\d\s\-()]{7,}\d)\b", "[PHONE]", text)
    text = re.sub(r"\b0x[a-fA-F0-9]{40}\b", "[WALLET_ADDRESS]", text)
    return text


def sanitize_prompt_text(text: str) -> str:
    """Mask identifiers and neutralize common instruction-override strings."""
    text = mask_sensitive_text(str(text))
    for pattern in (
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
        r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt",
        r"you\s+are\s+now\s+",
        r"</?(?:system|developer)>",
    ):
        text = re.sub(pattern, "[FILTERED_INSTRUCTION]", text, flags=re.I)
    return text


def clarification_questions(text: str, stakeholder_role: str = "Stakeholder") -> List[str]:
    """Ask targeted follow-ups for information commonly missing from blockchain requests."""
    text = text.strip()
    lowered = text.lower()
    questions = []
    probes = [
        (("goal", "objective", "reduce", "enable", "allow"), "What measurable business or user outcome should this blockchain capability achieve?"),
        (("user", "customer", "admin", "operator", "signer"), "Which stakeholder roles can initiate, approve, cancel, and audit this action?"),
        (("threshold", "limit", "amount", "large", "value"), "What exact transaction/value threshold and currency or token denomination applies?"),
        (("contract", "smart contract", "on-chain", "chain"), "Which chain, contract, and upgrade or pause authority are in scope?"),
        (("wallet", "key", "signer", "signature"), "How are wallet keys protected, rotated, recovered, and revoked?"),
        (("data", "personal", "identity", "kyc"), "What data is on-chain versus off-chain, and what are the privacy, retention, and deletion rules?"),
        (("api", "integrat", "oracle", "legacy"), "Which external or legacy systems must integrate, and what are their interface and failure requirements?"),
        (("fast", "performance", "available", "uptime", "latency"), "What measurable latency, throughput, availability, and recovery targets are required?"),
        (("test", "accept", "complete", "shall"), "What observable acceptance criteria will prove that the requirement is met?"),
        (("regulat", "compliance", "jurisdiction", "policy"), "Which jurisdictions, organisation policies, and authorised compliance owner apply?"),
    ]
    for keywords, question in probes:
        if not any(k in lowered for k in keywords):
            questions.append(question)
        if len(questions) == 4:
            break
    if not text:
        questions.insert(0, f"What blockchain project or use case should we gather requirements for, {stakeholder_role}?")
    return questions[:4]


CATEGORY_RULES = {
    "Business": ("business objective", "revenue", "cost", "business outcome", "market"),
    "Stakeholder": ("user", "customer", "operator", "administrator", "stakeholder"),
    "Functional": ("shall", "execute", "display", "submit", "transfer", "approve", "record"),
    "Security": ("security", "auth", "permission", "signer", "signature", "key", "encrypt", "threat", "attack", "contract"),
    "Privacy": ("privacy", "personal", "identity", "consent", "retention", "gdpr", "off-chain"),
    "Regulatory / Compliance": ("kyc", "aml", "sanction", "regulat", "compliance", "jurisdiction", "policy"),
    "Performance": ("latency", "throughput", "performance", "second", "minute", "requests per"),
    "Availability / Reliability": ("availability", "uptime", "reliability", "failover", "recovery", "disaster"),
    "Usability": ("usability", "interface", "dashboard", "accessible", "notification", "user experience"),
    "Data Management": ("data", "backup", "archive", "retention", "database", "ledger"),
    "Integration": ("api", "interface", "oracle", "integration", "legacy", "interoperab"),
    "Audit / Reporting": ("audit", "log", "trace", "report", "timestamp", "event"),
    "Operational / Maintenance": ("monitor", "alert", "maintain", "upgrade", "incident", "support"),
}


def classify_requirement(statement: str, existing: str = "") -> List[str]:
    text = f"{existing} {statement}".lower()
    labels = [label for label, terms in CATEGORY_RULES.items() if any(term in text for term in terms)]
    return labels or ["Stakeholder"]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "the", "a", "an", "shall", "must", "should", "system", "shall", "will", "to", "of", "for", "and", "or"
    }


def quality_checks(requirement: dict, evidence: List[dict]) -> List[str]:
    text = str(requirement.get("statement", "")).strip()
    lowered = text.lower()
    issues = []
    if not requirement.get("source_stakeholder"):
        issues.append("missing source stakeholder")
    if not evidence:
        issues.append("no supporting evidence supplied")
    if len(text.split()) < 6:
        issues.append("possibly incomplete")
    if not re.search(r"\bshall\b", lowered):
        issues.append("not written as a testable shall statement")
    if re.search(r"\b(should|may|appropriate|reasonable|as soon as possible|user-friendly|fast)\b", lowered):
        issues.append("ambiguous or weak obligation")
    if re.search(r"\b(fast|large|small|secure|easy|soon|adequate|robust)\b", lowered):
        issues.append("undefined or subjective terminology")
    if re.search(r"\b(zero risk|100% availability|never fail|all possible chains|instant finality)\b", lowered):
        issues.append("potentially infeasible or absolute guarantee")
    if re.search(r"\b(age|gender|race|ethnicity|nationality|disability)\b", lowered) and re.search(
        r"\b(score|eligib|approve|deny|rate|rank)\w*\b", lowered
    ):
        issues.append("potential fairness or protected-attribute impact requires review")
    if re.search(r"\b(payment|withdrawal|transfer|transaction|smart contract|wallet)\b", lowered) and not re.search(
        r"\b(auth|sign|key|secure|permission|validate|limit|pause|review|check)\w*\b", lowered
    ):
        issues.append("check for missing blockchain security or transaction-integrity controls")
    if any(term in lowered for term in ("performance", "latency", "throughput", "availability")) and not re.search(
        r"\b\d+(?:\.\d+)?\s*(?:%|seconds?|minutes?|hours?|ms|requests?/s|tps)\b", lowered
    ):
        issues.append("missing measurable non-functional threshold")
    if not requirement.get("acceptance_criteria"):
        issues.append("missing acceptance criteria")
    return issues


def detect_conflicts(requirements: List[dict]) -> List[dict]:
    findings = []
    for i, left in enumerate(requirements):
        left_text = str(left.get("statement", "")).lower()
        left_tokens = _tokens(left_text)
        for right in requirements[i + 1:]:
            right_text = str(right.get("statement", "")).lower()
            overlap = len(left_tokens & _tokens(right_text)) / max(1, len(left_tokens | _tokens(right_text)))
            polarity_conflict = bool(re.search(r"\bshall not\b|\bmust not\b|\bprohibit", left_text)) != bool(
                re.search(r"\bshall not\b|\bmust not\b|\bprohibit", right_text)
            )
            shared_subject = overlap >= 0.35
            numeric_left = set(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|seconds?|minutes?|hours?|ms|tps)?\b", left_text))
            numeric_right = set(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|seconds?|minutes?|hours?|ms|tps)?\b", right_text))
            contradictory_values = shared_subject and bool(numeric_left and numeric_right and numeric_left.isdisjoint(numeric_right))
            similarity = SequenceMatcher(None, left_text, right_text).ratio()
            if polarity_conflict and shared_subject:
                findings.append({"type": "potential conflict", "requirements": [left["id"], right["id"]],
                                 "detail": "Requirements express opposite obligations; stakeholder clarification is required."})
            elif contradictory_values:
                findings.append({"type": "potential conflict", "requirements": [left["id"], right["id"]],
                                 "detail": "Requirements use different measurable values for a similar subject; confirm the intended value."})
            elif similarity >= 0.82:
                findings.append({"type": "possible duplicate", "requirements": [left["id"], right["id"]],
                                 "detail": "Statements are highly similar; confirm whether they should be merged."})
    return findings


CONTROL_RULES = [
    ("Smart-contract security", ("contract", "on-chain", "smart contract", "solidity", "upgrade", "pause"),
     "Independent contract review, invariant/property testing, deployment approval, and controlled upgrade/pause access."),
    ("Wallet and key management", ("wallet", "key", "signer", "signature", "multi-sign"),
     "Least-privilege signing, secure key custody, rotation/revocation, recovery controls, and separation of duties."),
    ("Transaction integrity", ("transaction", "transfer", "withdrawal", "payment", "settlement"),
     "Idempotency, replay protection, finality handling, reconciliation, and auditable transaction state."),
    ("KYC / AML", ("kyc", "aml", "identity", "sanction", "customer due diligence"),
     "Customer due diligence and sanctions screening; confirm applicable jurisdiction and legal interpretation."),
    ("Privacy and data placement", ("personal data", "identity", "privacy", "consent", "retention", "wallet-to-identity"),
     "Data minimisation, off-chain personal data storage, purpose limitation, access controls, and retention review."),
    ("Oracle and bridge risk", ("oracle", "bridge", "cross-chain", "validator", "consensus"),
     "Source validation, stale-data handling, quorum/finality checks, failure containment, and monitoring."),
    ("Auditability", ("audit", "log", "trace", "timestamp", "event", "report"),
     "Tamper-evident event history, requirement-to-evidence traceability, and authorised audit review."),
    ("Retention and reporting", ("retention", "reporting", "report", "archive", "record"),
     "Confirm data-retention periods, required reports, evidence preservation and authorised deletion."),
    ("Interoperability and legacy", ("api", "integration", "legacy", "interoperab", "interface"),
     "Validate interface contracts, backward compatibility, reconciliation and dependency-failure handling."),
]


def compliance_mapping(requirements: Iterable[dict]) -> List[dict]:
    mappings = []
    for req in requirements:
        statement = str(req.get("statement", "")).lower()
        for domain, keywords, control in CONTROL_RULES:
            if any(keyword in statement for keyword in keywords):
                mappings.append({
                    "requirement_id": req.get("id"),
                    "control_domain": domain,
                    "control": control,
                    "jurisdiction": req.get("jurisdiction") or "Unspecified - stakeholder/compliance confirmation required",
                    "policy_source": req.get("evidence_sources", []),
                    "source_records": req.get("evidence", []),
                    "approval_checkpoint": "Compliance/security owner approval before baseline or release",
                    "determination": "Advisory control mapping only; not a legal or regulatory determination",
                })
    return mappings


def compliance_gaps(requirements: Iterable[dict], mappings: List[dict]) -> List[dict]:
    mapped_ids = {mapping["requirement_id"] for mapping in mappings}
    gaps = []
    for requirement in requirements:
        if requirement.get("id") not in mapped_ids:
            gaps.append({
                "requirement_id": requirement.get("id"),
                "gap": "No rule-based blockchain control mapping matched; confirm applicable organisation controls and jurisdiction.",
                "owner": "Compliance officer",
                "status": "Open - human review",
            })
        if not requirement.get("jurisdiction"):
            gaps.append({
                "requirement_id": requirement.get("id"),
                "gap": "Jurisdiction and regulatory applicability are not specified.",
                "owner": "Compliance/legal officer",
                "status": "Open - human review",
            })
    return gaps


def risk_assessment(requirement: dict) -> Dict:
    text = str(requirement.get("statement", "")).lower()
    domains = {
        "financial loss": ("payment", "withdrawal", "transfer", "liquidation", "settlement"),
        "smart-contract exploit": ("contract", "on-chain", "oracle", "bridge"),
        "key compromise": ("wallet", "key", "signer", "signature"),
        "privacy / compliance": ("identity", "personal data", "privacy", "kyc", "aml", "sanction"),
        "availability / operational failure": ("availability", "uptime", "recovery", "monitor"),
    }
    risks = [name for name, terms in domains.items() if any(term in text for term in terms)]
    score = min(5, 1 + len(risks) + (1 if requirement.get("needs_review") else 0))
    return {
        "risk_level": "Low" if score <= 2 else "Medium" if score <= 3 else "High",
        "risk_score": score,
        "risk_domains": risks or ["requirement uncertainty"],
        "owner": "To be assigned by project team",
        "treatment": "Human review, control mapping, and test evidence before approval",
        "status": "Open",
    }


FACTOR_KEYS = (
    "requirement_stability", "regulatory_criticality", "security_risk", "project_complexity",
    "system_size", "legacy_dependency", "change_frequency", "continuous_delivery",
    "stakeholder_availability", "verification_need", "failure_impact", "budget_schedule",
)


def normalize_project_factors(factors: Optional[dict]) -> Dict:
    defaults = {
        "requirement_stability": "medium", "regulatory_criticality": "high",
        "security_risk": "high", "project_complexity": "high", "system_size": "medium",
        "legacy_dependency": "medium", "change_frequency": "high", "continuous_delivery": "medium",
        "stakeholder_availability": "medium", "verification_need": "high",
        "failure_impact": "high", "budget_schedule": "medium",
    }
    if factors is None:
        factors = {}
    if not isinstance(factors, dict):
        raise ValueError("Project factors must be supplied as an object.")
    allowed = {"low", "medium", "high"}
    for key in FACTOR_KEYS:
        value = str(factors.get(key, defaults[key])).strip().lower()
        defaults[key] = value if value in allowed else defaults[key]
    return defaults


def ranked_sdlc(project_factors: Optional[dict]) -> List[dict]:
    """Explainable weighted decision matrix; scores are relative suitability points, not probabilities."""
    f = normalize_project_factors(project_factors)
    scores = {
        "Waterfall": 0.0, "V-Model": 0.0, "Spiral": 0.0, "Agile": 0.0,
        "Incremental": 0.0, "RAD": 0.0,
    }
    weights = {
        "requirement_stability": 1.0, "regulatory_criticality": 1.2, "security_risk": 1.2,
        "project_complexity": 0.9, "system_size": 0.5, "legacy_dependency": 0.7,
        "change_frequency": 1.0, "continuous_delivery": 0.8, "stakeholder_availability": 0.5,
        "verification_need": 1.2, "failure_impact": 1.2, "budget_schedule": 0.4,
    }
    # Each approach gets a transparent positive fit per factor level.
    fit = {
        "Waterfall": {"requirement_stability": (0, 1, 2), "regulatory_criticality": (0, 1, 2),
                      "change_frequency": (2, 1, 0), "stakeholder_availability": (2, 1, 0),
                      "system_size": (0, 1, 2), "budget_schedule": (0, 1, 2)},
        "V-Model": {"regulatory_criticality": (0, 1, 2), "security_risk": (0, 1, 2),
                    "verification_need": (0, 1, 2), "failure_impact": (0, 1, 2),
                    "system_size": (0, 1, 2), "legacy_dependency": (0, 1, 2)},
        "Spiral": {"project_complexity": (0, 1, 2), "security_risk": (0, 1, 2),
                   "failure_impact": (0, 1, 2), "budget_schedule": (2, 1, 0),
                   "legacy_dependency": (0, 1, 2)},
        "Agile": {"change_frequency": (0, 1, 2), "stakeholder_availability": (0, 1, 2),
                  "continuous_delivery": (0, 1, 2), "requirement_stability": (2, 1, 0),
                  "budget_schedule": (0, 1, 2)},
        "Incremental": {"project_complexity": (0, 1, 2), "system_size": (0, 1, 2),
                        "legacy_dependency": (0, 1, 2), "change_frequency": (0, 1, 2),
                        "continuous_delivery": (0, 1, 2), "stakeholder_availability": (0, 1, 2),
                        "requirement_stability": (2, 1, 0)},
        "RAD": {"project_complexity": (2, 1, 0), "system_size": (2, 1, 0),
                "change_frequency": (0, 1, 2), "stakeholder_availability": (0, 1, 2),
                "budget_schedule": (0, 1, 2), "continuous_delivery": (0, 1, 2)},
    }
    levels = {"low": 0, "medium": 1, "high": 2}
    for method, matrix in fit.items():
        scores[method] = sum(weights[key] * matrix[key][levels[f[key]]] for key in matrix)
    criteria = {name: list(matrix) for name, matrix in fit.items()}
    max_score = max(scores.values()) or 1
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [{
        "model": name,
        "score_percent": round(100 * value / max_score),
        "raw_score": round(value, 2),
        "recommended": rank == 0,
        "selection_factors": [key for key in criteria[name] if f[key] == "high"],
        "criteria_considered": {key: f[key] for key in criteria[name]},
    } for rank, (name, value) in enumerate(ranked)]


WORKFLOW_PHASES = {
    "Waterfall": [
        ("Discovery and requirements baseline", "Stakeholders, business analyst", "Approved SRS, assumptions and traceability baseline", "Scope and compliance review"),
        ("Architecture and design", "Architect, security and privacy leads", "Architecture, data-flow and threat models", "Security architecture approval"),
        ("Implementation", "Developers, smart-contract engineers", "Reviewed code and deployment artifacts", "Peer review and static analysis"),
        ("Verification and validation", "QA, security testers, business owners", "Unit, integration, property and acceptance evidence", "Independent security and acceptance sign-off"),
        ("Release and operations", "Release manager, operations, compliance", "Runbooks, monitoring and release record", "Human production-readiness approval"),
    ],
    "V-Model": [
        ("Requirements and acceptance design", "Business analyst, stakeholders, compliance", "SRS, acceptance criteria, traceability matrix", "Requirement and legal interpretation approval"),
        ("System and security architecture", "Architect, security/privacy leads", "Architecture, threat model, trust boundaries", "Architecture and threat-model approval"),
        ("Smart-contract and interface design", "Contract engineers, integration lead", "Contract specification, API/interface contracts", "Independent design review"),
        ("Implementation", "Developers, contract engineers", "Versioned source, code review, build artifacts", "Static analysis and peer review"),
        ("Unit and property verification", "Developers, QA, security testers", "Unit tests, invariants, fuzz/property results", "Critical vulnerability resolution"),
        ("Integration and system validation", "QA, integration and operations teams", "Chain, oracle, wallet, bridge and failure test results", "Transaction-integrity and recovery gates"),
        ("Acceptance and controlled deployment", "Product owner, compliance, security, release manager", "Signed acceptance, deployment plan and rollback", "Authorised human release approval"),
    ],
    "Spiral": [
        ("Set objectives and constraints", "Product owner, architect, risk owner", "Iteration goals and risk register", "Sponsor approval of objectives"),
        ("Analyse risks and prototype", "Security, contract, privacy and integration teams", "Threat model, prototype and risk evidence", "Risk acceptance or mitigation approval"),
        ("Engineer and test increment", "Developers, QA, operations", "Working increment, tests and trace links", "Security and quality gates"),
        ("Stakeholder evaluation", "Users, business, compliance, auditors", "Feedback and updated requirements", "Stakeholder sign-off"),
        ("Plan next iteration or release", "Coordinator, project manager, risk owner", "Revised plan, estimates and controls", "Go/no-go decision"),
    ],
    "Agile": [
        ("Prioritised blockchain backlog", "Product owner, business analyst, stakeholders", "Prioritised stories and acceptance criteria", "Product-owner approval"),
        ("Sprint planning and design", "Team, architect, security champion", "Sprint goal, design notes, threat-model updates", "Security story review"),
        ("Build and continuous verification", "Developers, contract engineers, QA", "Increment, CI results, code and trace links", "Peer review, static analysis and contract tests"),
        ("Review and acceptance", "Product owner, users, compliance as needed", "Demo, feedback, accepted/rejected stories", "Human acceptance of high-risk stories"),
        ("Secure release and monitoring", "Release manager, operations, security", "Release notes, monitoring and rollback plan", "Compliance and production approval"),
    ],
    "Incremental": [
        ("Define the product baseline and increments", "Product owner, business analyst, architect", "Prioritised scope, increment plan and traceability baseline", "Scope and dependency approval"),
        ("Design the first usable increment", "Architect, developers, security and privacy leads", "Increment architecture, interfaces and threat review", "Design and security review"),
        ("Build and verify an increment", "Developers, smart-contract engineers, QA", "Working increment, tests and linked requirement evidence", "Peer review and increment acceptance tests"),
        ("Integrate with previous increments", "Integration lead, QA, operations", "Integrated release candidate and regression results", "End-to-end, security and compatibility checks"),
        ("Release and plan the next increment", "Product owner, release manager, compliance, operations", "Approved release, monitoring and next increment backlog", "Human release approval and lessons learned"),
    ],
    "RAD": [
        ("Agree scope, constraints and prototype goals", "Product owner, users, business analyst", "Time-boxed scope, prototype objectives and acceptance measures", "Stakeholder approval of time box"),
        ("Co-design and prototype workflows", "Users, designers, business analyst, developers", "Reviewed interface and process prototypes", "Stakeholder feedback and scope confirmation"),
        ("Build time-boxed components", "Developers, integration team, QA", "Reusable components and working application increment", "Component and interface verification"),
        ("Integrate and validate the application", "Developers, QA, security/privacy reviewers", "Integrated build, test evidence and resolved prototype feedback", "Security, data and acceptance testing"),
        ("Deploy and transition to operations", "Release manager, operations, product owner", "User guidance, support plan, monitoring and release record", "Operational readiness and human release approval"),
    ],
}


def generate_workflow(sdlc: str) -> dict:
    key = sdlc if sdlc in WORKFLOW_PHASES else "Waterfall"
    phases = WORKFLOW_PHASES[key]
    return {
        "model": sdlc,
        "phases": [{
            "number": i + 1, "name": name,
            "activities": [f"Complete the {name.lower()} activities for this blockchain project."],
            "responsible_roles": [role.strip() for role in roles.split(",")],
            "deliverables": [deliverable], "validation_gate": gate,
            "security_controls": ["Threat model maintained", "Static analysis and smart-contract tests as applicable"],
            "compliance_checkpoints": ["Evidence and jurisdiction reviewed by authorised compliance officer"],
            "entry_criteria": "Approved outputs from the previous phase and current versioned evidence.",
            "exit_criteria": "Deliverables reviewed, tests recorded, and unresolved issues assigned.",
            "human_approval_required": any(term in gate.lower() for term in ("approval", "sign-off", "signoff", "go/no-go")),
            "traceability_required": True,
        } for i, (name, roles, deliverable, gate) in enumerate(phases)],
        "cross_cutting_controls": [
            "Smart-contract threat modelling and independent review",
            "Wallet/key permissions, separation of duties, and recovery controls",
            "Transaction integrity, replay/idempotency and finality testing",
            "Static analysis, dependency checks and property/fuzz testing as applicable",
            "On-chain/off-chain privacy and data-retention review",
            "Compliance approval, auditable traceability and human deployment gate",
        ],
    }


def run_specialist_agents(requirements: List[dict], evidence: List[dict], stakeholder_role: str = "Stakeholder") -> dict:
    """Shared-context specialist-agent pass; each agent emits a separately traceable result."""
    conflicts = detect_conflicts(requirements)
    evidence_by_id = {}
    for item in evidence:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        evidence_id = sanitize_prompt_text(str(item["id"]))
        evidence_by_id[evidence_id] = {
            "id": evidence_id,
            "source": sanitize_prompt_text(str(item.get("source") or "")),
            "metadata": {
                sanitize_prompt_text(str(key)): sanitize_prompt_text(str(value))
                for key, value in item.get("metadata", {}).items()
            } if isinstance(item.get("metadata"), dict) else {},
        }
    enriched = []
    for index, req in enumerate(requirements):
        current = dict(req)
        current["id"] = sanitize_prompt_text(str(current.get("id") or f"REQ-{index + 1:03d}"))
        current["statement"] = sanitize_prompt_text(str(current.get("statement") or "")).strip()
        current["category"] = sanitize_prompt_text(str(current.get("category") or "General"))
        current["source_stakeholder"] = sanitize_prompt_text(str(current.get("source_stakeholder") or stakeholder_role))
        current["business_justification"] = sanitize_prompt_text(str(current.get("business_justification") or ""))
        current["priority"] = sanitize_prompt_text(str(current.get("priority") or "Medium"))
        current["jurisdiction"] = sanitize_prompt_text(str(current.get("jurisdiction") or ""))
        for field in ("dependencies", "assumptions", "acceptance_criteria", "applicable_regulations"):
            value = current.get(field, [])
            if isinstance(value, str):
                value = [value]
            current[field] = [sanitize_prompt_text(str(item)) for item in value if str(item).strip()] if isinstance(value, list) else []
        cited_evidence = current.get("evidence", [])
        cited_ids = {
            sanitize_prompt_text(str(item.get("id")))
            for item in cited_evidence if isinstance(item, dict) and item.get("id")
        } if isinstance(cited_evidence, list) else set()
        prior_ids = current.get("evidence_ids", [])
        if isinstance(prior_ids, list):
            cited_ids.update(sanitize_prompt_text(str(item)) for item in prior_ids if item)
        cited_evidence = [evidence_by_id[item] for item in sorted(cited_ids) if item in evidence_by_id]
        current["evidence"] = cited_evidence
        current["evidence_ids"] = [item["id"] for item in cited_evidence]
        current["evidence_sources"] = sorted({e.get("source", "") for e in cited_evidence if e.get("source")})
        current["categories"] = classify_requirement(current["statement"], current.get("category", ""))
        current["quality_issues"] = quality_checks(current, cited_evidence)
        current["risk"] = risk_assessment(current)
        current["risk_level"] = current["risk"]["risk_level"]
        try:
            confidence = float(current.get("confidence_score", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        current["confidence_score"] = round(min(1.0, max(0.0, confidence)), 3) if math.isfinite(confidence) else 0.0
        current["approval_status"] = current.get("approval_status") or "Pending human approval"
        enriched.append(current)
    conflict_ids = {rid for issue in conflicts for rid in issue["requirements"]}
    for req in enriched:
        if req["id"] in conflict_ids:
            req["quality_issues"].append("potential conflict or duplicate requires clarification")
    controls = compliance_mapping(enriched)
    return {
        "requirements": enriched,
        "conflicts": conflicts,
        "compliance": controls,
        "compliance_gaps": compliance_gaps(enriched, controls),
        "risk_register": [{"requirement_id": r["id"], **r["risk"]} for r in enriched],
        "clarification_questions": [
            {"requirement_id": r["id"], "questions": r["quality_issues"]}
            for r in enriched if r["quality_issues"]
        ],
        "agents": [
            {"name": "Coordinator", "status": "completed", "access_scope": "Run context and agent status",
             "implementation": "Orchestration", "tools": [], "responsibility": "Sequence specialist steps and maintain run traceability"},
            {"name": "Classification", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Draft requirements",
             "tools": [], "responsibility": "Apply multi-label blockchain requirement categories"},
            {"name": "Clarification and quality validation", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Requirements and linked evidence",
             "tools": [], "responsibility": "Flag missing, vague, untestable or weakly supported details"},
            {"name": "Conflict and duplicate detection", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Requirement statements",
             "tools": [], "responsibility": "Identify likely duplicate or contradictory statements"},
            {"name": "Blockchain security and privacy rule checks", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Requirement statements and evidence references",
             "tools": [], "responsibility": "Flag smart-contract, key, transaction and data-boundary concerns"},
            {"name": "Compliance control mapping", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Requirements, source metadata and control rules",
             "tools": [], "responsibility": "Create advisory mappings and explicit open gaps"},
            {"name": "Risk analysis", "status": "completed", "implementation": "Rule-based",
             "access_scope": "Requirements and validation findings",
             "tools": [], "responsibility": "Create a rule-based risk register for human review"},
        ],
    }


def generate_artifacts(domain: str, functionality: str, requirements: List[dict],
                       compliance: List[dict], risk_register: List[dict],
                       conflicts: Optional[List[dict]] = None) -> dict:
    srs = "\n".join([
        f"# Software Requirements Specification: {domain}",
        "\n## Scope\n" + functionality,
        "\n## Blockchain-specific considerations\nOn-chain/off-chain data boundary, chain and finality assumptions, smart-contract upgrade/pause authority, wallet/key custody, oracle/bridge dependencies, and transaction integrity require project-owner confirmation.",
        "\n## Requirements",
        *[f"- {r['id']}: {r['statement']} (Categories: {', '.join(r.get('categories', []))}; Priority: {r.get('priority', 'Medium')}; Status: {r.get('approval_status', 'Pending human approval')})" for r in requirements],
    ])
    stories, use_cases, traceability = [], [], []
    for i, req in enumerate(requirements):
        rid = req["id"]
        stories.append({
            "id": f"US-{i + 1:03d}", "requirement_id": rid,
            "story": f"As a {req.get('source_stakeholder') or 'stakeholder'}, I want {req['statement'].removeprefix('The system shall ').rstrip('.').lower()}, so that the blockchain use case meets its approved objective.",
            "acceptance_criteria": req.get("acceptance_criteria", []),
        })
        use_cases.append({
            "id": f"UC-{i + 1:03d}", "requirement_id": rid,
            "name": req["statement"][:90],
            "primary_actor": req.get("source_stakeholder") or "Stakeholder",
            "preconditions": ["Authorised role and approved configuration", "Required chain/network and dependent services are available"],
            "main_flow": [req["statement"]],
            "alternate_and_failure_flows": ["Reject unauthorised or invalid transaction", "Record and alert on dependency, finality, or validation failure"],
            "postconditions": ["Outcome is verifiable and linked to audit evidence"],
        })
        traceability.append({
            "requirement_id": rid, "source_stakeholder": req.get("source_stakeholder"),
            "source_documents": req.get("evidence_sources", []), "evidence_ids": req.get("evidence_ids", []),
            "source_records": req.get("evidence", []),
            "user_story_ids": [f"US-{i + 1:03d}"], "use_case_ids": [f"UC-{i + 1:03d}"],
            "acceptance_criteria": req.get("acceptance_criteria", []), "approval_status": req.get("approval_status"),
        })
    return {
        "srs_markdown": srs,
        "user_stories": stories,
        "use_cases": use_cases,
        "process_workflows": [{"requirement_id": r["id"], "steps": ["Authenticate actor", "Validate request and policy", "Execute or reject operation", "Record outcome and evidence"]} for r in requirements],
        "data_requirements": [{"requirement_id": r["id"], "data_boundary": "Determine whether each field belongs on-chain or off-chain; do not store personal data on-chain without authorised privacy review."} for r in requirements],
        "interface_requirements": [{"requirement_id": r["id"], "interfaces": ["Wallet/client", "Smart contract or ledger", "Approved external APIs/oracles where applicable"]} for r in requirements],
        "traceability_matrix": traceability,
        "compliance_control_matrix": compliance,
        "compliance_gaps": compliance_gaps(requirements, compliance),
        "risk_register": risk_register,
        "assumption_dependency_register": [
            {"requirement_id": r["id"], "assumptions": r.get("assumptions", []), "dependencies": r.get("dependencies", [])}
            for r in requirements
        ],
        "conflicts_and_duplicates": conflicts or [],
        "open_issues": [
            {"requirement_id": r["id"], "issues": r.get("quality_issues", [])}
            for r in requirements if r.get("quality_issues") or r.get("needs_review")
        ],
    }
