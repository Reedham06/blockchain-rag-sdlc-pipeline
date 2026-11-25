# Blockchain Finance: RAG → LLM 1 → LLM 2 (SDLC) Pipeline

Give it a blockchain-finance domain and a functionality. It retrieves evidence from your existing requirement
documents, has one LLM write grounded requirements (with reasoning), then has a second, different LLM choose the
SDLC model (with a software-engineering justification).

```
 input: domain + functionality
    │
    ▼
 1. RAG retrieval ─ finds the most relevant chunks in your existing requirement documents (TF-IDF)
    │  evidence
    ▼
 2. LLM 1 (Llama 3.2 3B) ─ requirements + reasoning, grounded in the evidence
    │     + grounding check: every requirement is scored against the evidence;
    │       weakly supported ones are flagged "needs human review"
    ▼
 3. LLM 2 (Qwen 2.5 3B) ─ ONLY identifies the SDLC model (+ key factors + justification)
    ▼
 final output
```

Both LLMs run locally on [Ollama](https://ollama.com) (free, offline). They are two different models, and each can be
swapped for your own fine-tuned version (optional - see below). Gemini is available as an optional cloud alternative.

---

## 1. Run it (no fine-tuning needed)

1. **Python 3.9+**, then in this folder: `pip install -r requirements.txt`
2. **Install Ollama** from https://ollama.com and make sure it is running (open the app, or run `ollama serve`).
3. **Download the two base models** (about 2 GB each, once):
   ```
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```
4. **Start the website:** `python web_app.py` - your browser opens at http://localhost:8000. Click **Run pipeline**.
   (The top box tells you whether Ollama is reachable and which models it will use.)

Command-line version:
```
python main.py --domain "DeFi Lending" --functionality "Multi-signature approval for large withdrawals"
python main.py --domain "CBDC" --functionality "Offline retail payments" --json-out result.json
```

**Speed:** 3B models on a laptop CPU take roughly one to a few minutes per LLM step; a GPU is much faster. The page
shows progress step by step so you can see what is running.

**Your own documents:** put `.txt` / `.md` files in a folder and type its path into "Document folder" (or `--docs`),
and/or paste text into the optional box. The included `sample_docs/` are *illustrative* policy notes written for this
project - they are not real regulations.

**Using Gemini instead (optional):** `pip install google-genai`, set `GEMINI_API_KEY` (or paste it into the page), and
pick "Gemini" as a step's provider. Default model: `gemini-3.5-flash-lite`.

---

## 2. Do you need to fine-tune? - Honest answer

**No, the project works without it.** With base models the whole flow runs; the website says so in its status box.

What fine-tuning *can* do: teach the small models the output style - valid JSON, one-sentence "shall" requirements,
actually using the evidence, choosing among the seven SDLC models and justifying the choice. What it *cannot* do
reliably: add domain knowledge. Knowledge comes from your documents through RAG, which is the right tool for that.

**Do it if** your assignment says the LLMs must be fine-tuned, or if the base models give malformed/weak output.
Otherwise it is optional.

### Where the training data comes from
`seed_data.py` holds **28 hand-written scenarios** across 7 domains (DeFi lending, cross-border payments, KYC/AML,
stablecoin, custody, tokenized trade finance, CBDC). I (an AI assistant) wrote them for this project - they are
illustrative, **not** taken from real project documents, so please read them and edit anything that is wrong.
Each scenario produces one training example per model:
* **LLM 1:** domain + functionality + retrieved evidence → requirements + reasoning
* **LLM 2:** those requirements + evidence → SDLC model + key factors + justification

SDLC answers are deliberately varied (Waterfall 4, V-Model 5, Spiral 4, Agile 3, DevSecOps 3, two hybrids 9) so the
model doesn't learn to always say "hybrid". The evidence in each training prompt comes from running the *real*
retriever over `sample_docs/`, and the prompts come from `pipeline.py`, so training and run time match exactly.

**28 examples is small.** Expect better format and style, not deep expertise, and some risk of memorising the examples.
More data helps: add scenarios to `seed_data.py` (aim for 100+, ideally from real documents), then regenerate.
If you change `sample_docs/` or any prompt in `pipeline.py`, re-run `generate_dataset.py` and re-train.

### How to fine-tune (free Google Colab; my rough estimate is 30-60 minutes of work for both models)
**You only need one file: `finetune_colab.ipynb`.** It already contains all the code and creates the project files
inside Colab itself, so there is **no zip to make or upload**.
1. Go to https://colab.research.google.com (sign in with a Google account) -> **File -> Upload notebook** -> choose `finetune_colab.ipynb`.
2. **Runtime -> Change runtime type -> T4 GPU -> Save.**
3. Run the cells **one by one, in order** (play button on each, wait until it finishes): check the GPU, create the files,
   check the files, install Unsloth, build the data, fine-tune model 1, fine-tune model 2, copy the results to Google Drive.
4. Get the models out of Colab **before the session ends** (Colab deletes its files afterwards; don't restart or
   disconnect the runtime until you have them). Option A: copy them to Google Drive, download that folder and rename it
   `exports`. Option B (if Drive fails): the notebook downloads two zips; unzip both into a folder named `exports`.
   Either way you should have `exports/blockchain-req-llm/` and `exports/blockchain-sdlc-llm/` inside this project
   folder, each with a `Modelfile` and a `model.Q4_K_M.gguf`.
5. With Ollama running: `python install_models.py`
6. Restart `python web_app.py`. The status box now says *"using your fine-tuned model blockchain-req-llm"* (and
   `blockchain-sdlc-llm`). No settings to change - fine-tuned models are picked automatically when installed.

If you edit `seed_data.py` (or anything else) on your computer, run `python make_notebook.py` to rebuild
`finetune_colab.ipynb` with your changes embedded, then upload the new notebook. (Notebook step 10 can also hand you
the complete project as one zip if you ever lose the folder.)

Own NVIDIA GPU instead of Colab (Linux/WSL): `pip install unsloth`, then `python generate_dataset.py`,
`python finetune_llm.py --stage 1`, `python finetune_llm.py --stage 2`, `python install_models.py`.

**Check that it helped:** write 5 functionalities that are *not* in `seed_data.py`, run each with the base model and
with the fine-tuned model (type the model name into the page), and compare: valid output, number of requirements
flagged for review, and whether the SDLC choice and justification make sense.

**Troubleshooting**
* Drive says `mount failed`: allow pop-ups, choose the same Google account Colab uses, click Allow on every screen, and re-run;
  or use Option B (direct download) in step 9. Never restart the runtime before the models are saved.
* Step 3 of the notebook says `Missing: ...`: step 2 did not finish - re-run it and read its output for the error.
* Unsloth install conflicts or API changes: follow the current install notes at https://docs.unsloth.ai.
  The script already tolerates the different `SFTTrainer` signatures used by recent TRL versions.
* Out of disk space on Colab: run stage 2 after deleting stage 1's `*_gguf_work` / `*_checkpoints` folders (the notebook does this).
* "No quantised .gguf found": look inside `exports/<name>_gguf_work*` and copy the Q4_K_M file next to the Modelfile
  as `model.Q4_K_M.gguf`.
* Model returns bad JSON: lower `--epochs` (over-training) or add more varied examples.

---

## 3. Tests

`python -m unittest -v` runs 23 tests with fake models (no GPU, Ollama or API key needed). They check: paragraph
chunking, that retrieval finds the right document, **that retrieval happens before LLM 1 and LLM 2 is SDLC-only and
runs after it**, the grounding flags, messy-output handling, the Ollama HTTP requests (including JSON retry and the
missing-model message), the CLI end to end (fine-tuned LLM 1 auto-selected, base LLM 2), the dataset files, that the
training text equals the Ollama prompt format, the website's three-step API, and that the Colab notebook really recreates every project file exactly.

---

## 4. What was fixed compared with the previous version

| Problem found | Fix |
|---|---|
| Flow was generate → retrieve, but your document's section 6 is retrieve → generate | Now: retrieve → LLM 1 (grounded) → LLM 2 (SDLC only). LLM 2 also receives the evidence. |
| Both steps defaulted to the same Gemini model | Defaults are two different Ollama models; Gemini is optional |
| Fine-tuned models could not be used by the website | `Modelfile` + `install_models.py`; fine-tuned models auto-detected, status box shows what is used |
| Only 6 training examples, one model trained on both tasks, prompts duplicated (could drift) | 28 scenarios, one model per stage, prompts shared with the runtime, varied SDLC labels |
| Fine-tuning script mixed library versions, trained on the whole prompt | Version-tolerant trainer, response-only loss, matching template, GGUF auto-discovery, export folder per model |
| README out of date, no run steps | Rewritten (this file) |
| Page inserted model output as raw HTML (injection risk), single 120 s request | All output escaped; three step-by-step requests; clearer errors; threaded server |
| Chunks cut sentences in half | Paragraph-aware chunking, bigram TF-IDF |
| No way to see which requirements the documents don't support | Grounding score + "needs human review" flag |

## 5. Limitations (please read)

* **Fine-tuning was not run here** - there is no GPU in my environment. The data generation, training-text format,
  Modelfile, export logic and everything else are tested, but the Unsloth training/export steps follow its documented
  API and may need small tweaks for the exact versions Colab installs. If something fails, send me the error.
* TF-IDF retrieval is keyword-based and can miss paraphrases (an embedding retriever can be dropped in later).
* The grounding score is lexical similarity, not proof of correctness. Unflagged requirements still need human review.
* 3B models are modest; larger Ollama models (or Gemini) give better text, just slower or cloud-based.
* The sample documents and training scenarios are illustrative and AI-written, not real regulation or real project data.

## 6. Files

| File | Purpose |
|---|---|
| `web_app.py` | The website (`python web_app.py`) |
| `main.py` | Command-line version |
| `pipeline.py` | The three steps + shared prompts |
| `retriever.py` | Chunking + TF-IDF RAG retrieval + grounding score |
| `llm_clients.py` | Ollama and Gemini clients, JSON repair, fine-tuned-model detection |
| `seed_data.py` | 28 hand-written training scenarios |
| `generate_dataset.py` | Builds `train_stage1.jsonl` / `train_stage2.jsonl` |
| `finetune_llm.py` | Fine-tunes one model (Unsloth QLoRA) and exports GGUF + Modelfile |
| `finetune_colab.ipynb` | Self-contained Colab notebook for the fine-tuning (contains all the code - nothing else to upload) |
| `make_notebook.py` | Rebuilds the notebook with your current files embedded |
| `install_models.py` | Registers the exported models in Ollama |
| `test_pipeline.py` | 23 automated tests |
| `sample_docs/` | 7 illustrative requirement/policy documents (the RAG corpus) |
