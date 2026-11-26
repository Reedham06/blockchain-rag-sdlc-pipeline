"""Command-line version of the pipeline.

    python main.py --domain "DeFi Lending" --functionality "Multi-signature approval for large withdrawals"

By default both steps run on local Ollama models (two different ones):
  LLM 1 -> blockchain-req-llm (fine-tuned) if installed, otherwise llama3.2:3b
  LLM 2 -> blockchain-sdlc-llm (fine-tuned) if installed, otherwise qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from audit_store import create_run_id, record_event
from llm_clients import LLMError, build_client
from pipeline import (MIN_GROUNDING, build_index, load_documents, run_pipeline,
                      stage1_messages, stage2_messages)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic blockchain requirements engineering with RAG and ranked SDLC")
    p.add_argument("--docs", default=os.path.join(HERE, "sample_docs"), help="Approved .txt/.md/.docx blockchain knowledge sources")
    p.add_argument("--domain", required=True, help='e.g. "DeFi Lending"')
    p.add_argument("--functionality", required=True, help="Feature to generate requirements for")
    p.add_argument("--stakeholder-role", default="Stakeholder", help="Source stakeholder role")
    p.add_argument("--llm1-provider", default="ollama", choices=["ollama", "gemini"])
    p.add_argument("--llm1-model", default=None, help="default: blockchain-req-llm if installed, else llama3.2:3b")
    p.add_argument("--llm2-provider", default="ollama", choices=["ollama", "gemini"])
    p.add_argument("--llm2-model", default=None, help="default: blockchain-sdlc-llm if installed, else qwen2.5:3b")
    p.add_argument("--top-k", type=int, default=4, help="Chunks retrieved by RAG")
    p.add_argument("--min-grounding", type=float, default=MIN_GROUNDING, help="Below this support score a requirement is flagged for review")
    p.add_argument("--json-out", default=None, help="Also save the full result to this JSON file")
    args = p.parse_args()

    try:
        retriever = build_index(load_documents(args.docs), documents_root=args.docs)
        llm1 = build_client(args.llm1_provider, args.llm1_model, role="llm1")
        llm2 = build_client(args.llm2_provider, args.llm2_model, role="llm2")
        result = run_pipeline(retriever, args.domain, args.functionality, llm1, llm2,
                              top_k=args.top_k, min_grounding=args.min_grounding,
                              progress=lambda m: print(f"\n>> {m}"),
                              stakeholder_role=args.stakeholder_role)
    except (LLMError, ValueError) as e:
        sys.exit(f"\nError: {e}")

    run_id = create_run_id()
    result["run_id"] = run_id
    result["audit"].append({"event": "run_id", "run_id": run_id})
    prompt1 = stage1_messages(args.domain, args.functionality, result["evidence"], args.stakeholder_role)
    prompt2 = stage2_messages(args.domain, result["requirements"], result["evidence"],
                              agent_reviews=result["agent_reviews"])
    record_event(run_id, "pipeline_prompts", args.stakeholder_role, {
        "requirements_prompt": {"system": prompt1[0], "user": prompt1[1]},
        "sdlc_prompt": {"system": prompt2[0], "user": prompt2[1]},
    })
    record_event(run_id, "pipeline_completed", "Coordinator", result)

    print("\n=== Step 1: RAG evidence ===")
    for e in result["evidence"] or [{"id": "-", "source": "(none)", "text": "", "score": 0}]:
        print(f"[{e['id']}] {e['source']} (score {e['score']}): {e['text'][:110]}...")
    print(f"\n=== Step 2: Requirements from {result['models']['llm1']} ===")
    for r in result["requirements"]:
        flag = "  <-- NEEDS HUMAN REVIEW" if r["needs_review"] else ""
        print(f"{r['id']} [{', '.join(r['categories'])}] {r['statement']}\n"
              f"      source: {r['source_stakeholder']}; evidence: {r['support_source']} "
              f"(similarity {r['support_score']}); risk: {r['risk_level']}{flag}")
        if r["quality_issues"]:
            print("      quality issues:", "; ".join(r["quality_issues"]))
    print("\nReasoning:", result["requirements_reasoning"])
    s = result["sdlc"]
    print(f"\n=== Step 3: SDLC from {result['models']['llm2']} ===\nRecommended: {s['recommended_sdlc']}")
    print("Inferred project profile:", ", ".join(f"{key}={value}" for key, value in result["project_factors"].items()))
    for f in s["key_factors"]:
        print(f"  - {f}")
    print("Justification:", s["justification"])
    print("\nRanked SDLC approaches (relative fit points):")
    for option in s["ranked_recommendations"]:
        print(f"  {option['model']}: {option['score_percent']} points")
    print(f"\nAudit run ID: {run_id}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full result to {args.json_out}")


if __name__ == "__main__":
    main()
