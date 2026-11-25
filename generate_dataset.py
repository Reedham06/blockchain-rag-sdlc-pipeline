"""Build the fine-tuning datasets from seed_data.py.

    python generate_dataset.py            -> train_stage1.jsonl  (LLM 1: requirements)
                                             train_stage2.jsonl  (LLM 2: SDLC only)

The prompts come from pipeline.py and the evidence comes from running the real retriever over
sample_docs/, so the training prompts are exactly what the models will see at run time.
If you change sample_docs/ or the prompts, re-run this script and re-train.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from pipeline import SDLC_LABELS, build_index, load_documents, retrieve_evidence, stage1_messages, stage2_messages
from seed_data import SEEDS


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

        sys2, usr2 = stage2_messages(seed["domain"], reqs, evidence)
        stage2.append(_record(sys2, usr2, {"recommended_sdlc": seed["sdlc"], "key_factors": seed["factors"],
                                           "justification": seed["justification"]}))
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
