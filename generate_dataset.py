"""Build the fine-tuning datasets from seed_data.py.

    python generate_dataset.py            -> train_stage1.jsonl  (LLM 1: requirements)
                                             train_stage2.jsonl  (LLM 2: infer project factors + SDLC)

The prompts come from pipeline.py and the evidence comes from running the real retriever over
sample_docs/, so the training prompts are exactly what the models will see at run time. Factor labels
are illustrative heuristics over the hand-authored seed scenarios and should be expert-reviewed.
If you change sample_docs/ or the prompts, re-run this script and re-train.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from pipeline import SDLC_LABELS, build_index, load_documents, retrieve_evidence, stage1_messages, stage2_messages
from seed_data import SEEDS


def infer_seed_project_factors(seed: dict) -> dict:
    """Create illustrative low/medium/high training labels from each hand-authored seed scenario."""
    narrative = " ".join([
        seed["domain"], seed["functionality"], seed["reasoning"],
        " ".join(seed["factors"]), seed["justification"],
        " ".join(statement for _, statement in seed["requirements"]),
    ]).lower()
    factors = {
        "requirement_stability": "medium",
        "regulatory_criticality": "medium",
        "security_risk": "medium",
        "project_complexity": "medium",
        "system_size": "medium",
        "legacy_dependency": "medium",
        "change_frequency": "medium",
        "continuous_delivery": "medium",
        "stakeholder_availability": "medium",
        "verification_need": "medium",
        "failure_impact": "medium",
        "budget_schedule": "medium",
    }

    def contains(*terms):
        return any(term in narrative for term in terms)

    if contains("frequently change", "evolving", "evolve", "tune frequently", "change often", "market changes"):
        factors["requirement_stability"], factors["change_frequency"] = "low", "high"
    elif contains("stable", "rarely change", "well understood", "well-defined"):
        factors["requirement_stability"], factors["change_frequency"] = "high", "low"
    if seed["domain"] in ("Blockchain KYC / AML", "CBDC", "Stablecoin Issuance", "Cross-Border Payments") or contains(
        "regulatory", "compliance", "regulated", "jurisdiction", "licensing"
    ):
        factors["regulatory_criticality"] = "high"
    if contains("smart contract", "key custody", "multi-signature", "oracle", "bridge", "financial loss", "asset loss"):
        factors["security_risk"], factors["verification_need"], factors["failure_impact"] = "high", "high", "high"
    if contains("bridge", "cross-border", "multiple chain", "oracle", "interoperability", "settlement", "multi-party"):
        factors["project_complexity"] = "high"
    if contains("large platform", "multiple systems", "ecosystem", "network-wide", "interbank"):
        factors["system_size"] = "high"
    if contains("legacy", "rtgs", "existing banking", "legacy system"):
        factors["legacy_dependency"] = "high"
    elif contains("standalone", "read-only", "no legacy"):
        factors["legacy_dependency"] = "low"
    if contains("continuous delivery", "frequent release", "monitoring", "short agile", "deploy frequently"):
        factors["continuous_delivery"] = "high"
    if contains("stakeholder availability", "stakeholders available", "frequent feedback", "user feedback"):
        factors["stakeholder_availability"] = "high"
    if contains("budget constrained", "tight schedule", "limited budget", "schedule pressure"):
        factors["budget_schedule"] = "high"
    return factors


def _record(system: str, user: str, assistant: dict) -> str:
    return json.dumps({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
    ]}, ensure_ascii=False)


def build_examples(docs_dir: str, top_k: int = 4):
    retriever = build_index(load_documents(docs_dir))
    stage1, stage2, problems = [], [], []
    for n, seed in enumerate(SEEDS, 1):
        tag = f"seed {n} ({seed['domain']}: {seed['functionality']})"
        if seed["sdlc"] not in SDLC_LABELS:
            problems.append(f"{tag}: unknown SDLC label {seed['sdlc']!r}")
        evidence = retrieve_evidence(retriever, seed["domain"], seed["functionality"], top_k)
        if not evidence:
            problems.append(f"{tag}: retrieval found no evidence")
        reqs = [{"id": f"REQ-{i:03d}", "statement": st, "category": cat}
                for i, (cat, st) in enumerate(seed["requirements"], 1)]
        for r in reqs:
            if " shall " not in f" {r['statement']} ":
                problems.append(f"{tag}: {r['id']} does not use 'shall'")

        sys1, usr1 = stage1_messages(seed["domain"], seed["functionality"], evidence)
        stage1.append(_record(sys1, usr1, {"requirements": reqs, "reasoning": seed["reasoning"]}))

        stakeholder_context = seed["reasoning"] + "\nProject context notes: " + "; ".join(seed["factors"])
        sys2, usr2 = stage2_messages(seed["domain"], reqs, evidence,
                                     stakeholder_context=stakeholder_context)
        stage2.append(_record(sys2, usr2, {
            "project_factors": infer_seed_project_factors(seed),
            "recommended_sdlc": seed["sdlc"], "key_factors": seed["factors"],
            "justification": seed["justification"],
        }))
    return stage1, stage2, problems


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Generate fine-tuning datasets")
    p.add_argument("--docs", default=os.path.join(here, "sample_docs"))
    p.add_argument("--out-dir", default=here)
    args = p.parse_args()

    stage1, stage2, problems = build_examples(args.docs)
    if problems:
        raise SystemExit("Dataset problems:\n  " + "\n  ".join(problems))
    for name, rows in (("train_stage1.jsonl", stage1), ("train_stage2.jsonl", stage2)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[ok] wrote {len(rows)} examples to {name}")
    print("SDLC label distribution:", dict(Counter(s["sdlc"] for s in SEEDS)))


if __name__ == "__main__":
    main()
