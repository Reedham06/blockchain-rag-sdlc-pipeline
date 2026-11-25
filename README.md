# Blockchain Finance: Requirements and SDLC Pipeline

A retrieval-grounded, two-model pipeline for blockchain-based financial software projects. Given a domain and a
functionality, the system retrieves evidence from a set of existing requirement documents, generates grounded
requirements with an explanation, and then recommends the most suitable Software Development Life Cycle (SDLC)
model with a justification.

Both language models run locally on [Ollama](https://ollama.com), so the system works offline and needs no API key.
Each model can optionally be replaced by a fine-tuned version produced with the included training workflow.

## Overview

```
 input: domain + functionality
    |
    v
 1. RAG retrieval   Finds the most relevant chunks of the existing requirement documents (TF-IDF).
    |  evidence
    v
 2. LLM 1           Llama 3.2 3B (or the fine-tuned blockchain-req-llm).
    |               Writes requirements and reasoning, grounded in the evidence.
    |               A grounding check then scores every requirement against the evidence and
    |               flags weakly supported ones for human review.
    v
 3. LLM 2           Qwen 2.5 3B (or the fine-tuned blockchain-sdlc-llm).
    |               Identifies the SDLC model only: recommendation, key factors, justification.
    v
 final output
```

**Supported SDLC outcomes:** Waterfall, V-Model, Spiral, Agile, DevSecOps, Agile-V-Model Hybrid and
DevSecOps-V-Model Hybrid.

**Sample domains:** DeFi lending, cross-border payments, KYC/AML, stablecoin issuance, crypto custody, tokenized
trade finance and CBDC. Other domains work as long as relevant documents are supplied.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com) (running locally)
- About 2 GB of disk space per model, and 8 GB of RAM or more recommended
- Optional: a Google account for fine-tuning on free Colab GPUs; a Gemini API key to use Gemini instead of Ollama

## Quick start

### 1. Install the Python dependencies

```
pip install -r requirements.txt
```

### 2. Install Ollama and the base models

Install Ollama from https://ollama.com and make sure it is running (open the app, or run `ollama serve`). Opening
http://localhost:11434 in a browser should show "Ollama is running". Then download the two base models once:

```
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

### 3. Start the web application

```
python web_app.py
```

The browser opens at http://localhost:8000 (set the `PORT` environment variable to use another port). Enter a
domain and a functionality, then select **Run pipeline**. The status box at the top shows whether Ollama is reachable
and which models will be used. The three steps run one after another and each result appears as soon as it is ready.
On a laptop CPU, each LLM step typically takes from one to a few minutes; a GPU is considerably faster.

Example input:

| Field | Value |
|---|---|
| Domain | `DeFi Lending` |
| Functionality | `Governance timelock before changes to lending pool parameters take effect` |

### Command-line usage

```
python main.py --domain "DeFi Lending" --functionality "Multi-signature approval for large withdrawals"
python main.py --domain "CBDC" --functionality "Offline retail payments" --json-out result.json
```

| Option | Default | Description |
|---|---|---|
| `--docs` | `sample_docs` | Folder of `.txt` / `.md` requirement documents |
| `--llm1-provider`, `--llm1-model` | `ollama`, auto | Model for requirement generation |
| `--llm2-provider`, `--llm2-model` | `ollama`, auto | Model for SDLC identification |
| `--top-k` | `4` | Number of evidence chunks retrieved |
| `--min-grounding` | `0.10` | Support score below which a requirement is flagged for review |
| `--json-out` | none | Save the full result to a JSON file |

When no model is specified, the fine-tuned model is used if it is installed in Ollama, and the base model otherwise.

### Using custom documents

Place `.txt` or `.md` files in a folder and enter its path in the "Document folder" field (or pass `--docs`).
Additional text can be pasted into the optional box on the web page. The files in `sample_docs/` are illustrative
policy notes created for this project; they are not real regulations.

### Using Gemini (optional)

```
pip install google-genai
```

Set the `GEMINI_API_KEY` environment variable (or paste the key into the web page) and choose "Gemini" as the
provider for either step. The default model is `gemini-3.5-flash-lite`.

## Fine-tuning the models (optional)

The pipeline runs fully on the base models. Fine-tuning is useful when a project requires fine-tuned models, or
when the base models produce malformed or weak output.

**What it changes.** Fine-tuning teaches the small models the expected output style: valid JSON, one-sentence
"shall" requirements, use of the supplied evidence, and a justified choice among the seven SDLC outcomes. It does
not reliably add domain knowledge; that comes from the retrieved documents.

### Training data

`seed_data.py` contains 28 synthetic scenarios across seven domains. They are hand-authored examples written for
this project and are not drawn from real organisational or regulatory documents, so they should be reviewed and
edited before being relied upon. Each scenario yields one training example for each model:

- **LLM 1:** domain, functionality and retrieved evidence, producing requirements and reasoning.
- **LLM 2:** the same requirements and evidence, producing the SDLC model, key factors and justification.

The SDLC labels are deliberately varied (Waterfall 4, V-Model 5, Spiral 4, Agile 3, DevSecOps 3, hybrids 9) so that
the model does not default to a single answer. The evidence in every training prompt is produced by running the real
retriever over `sample_docs/`, and the prompts are the ones defined in `pipeline.py`, so training and inference use
identical formats. The generated files are `train_stage1.jsonl` and `train_stage2.jsonl`.

A set of 28 examples is small. It improves format and style but does not give deep expertise, and some memorisation
of the examples is likely. Adding scenarios to `seed_data.py` (100 or more is recommended, ideally based on real
documents) and regenerating the data improves results. Whenever `sample_docs/` or a prompt in `pipeline.py`
changes, run `python generate_dataset.py` again and retrain.

| Model | Base model | Training data | Name in Ollama |
|---|---|---|---|
| LLM 1 (requirements) | Llama-3.2-3B-Instruct | `train_stage1.jsonl` | `blockchain-req-llm` |
| LLM 2 (SDLC only) | Qwen2.5-3B-Instruct | `train_stage2.jsonl` | `blockchain-sdlc-llm` |

Training uses QLoRA through [Unsloth](https://docs.unsloth.ai), with a LoRA rank of 16, three epochs and a learning
rate of 2e-4 by default. Loss is computed on the assistant's answer only (when Unsloth's `train_on_responses_only` is available). The models are exported as 4-bit GGUF
files together with an Ollama `Modelfile`.

### Procedure (Google Colab)

`finetune_colab.ipynb` is self-contained: it contains all the project code and creates the required files inside
Colab, so nothing needs to be zipped or uploaded.

1. Open https://colab.research.google.com, choose **File > Upload notebook**, and select `finetune_colab.ipynb`.
2. Choose **Runtime > Change runtime type > T4 GPU** and save.
3. Run the cells in order, waiting for each to finish: check the GPU, create the project files, verify the files,
   install Unsloth, build the training data, fine-tune LLM 1, fine-tune LLM 2.
4. Save the models before the session ends, because Colab deletes its files afterwards. Option A copies them to
   Google Drive; option B downloads one zip per model. Do not restart or disconnect the runtime until the models
   are saved.
5. On the local machine, create a folder named `exports` inside the project folder, so that the layout is:
   ```
   exports/
     blockchain-req-llm/
       Modelfile
       model.Q4_K_M.gguf
     blockchain-sdlc-llm/
       Modelfile
       model.Q4_K_M.gguf
   ```
6. With Ollama running, register the models:
   ```
   python install_models.py
   ollama list
   ```
   Both `blockchain-` models should be listed.
7. Restart `python web_app.py`. The status box reports that the fine-tuned models are in use. No configuration is
   needed, because installed fine-tuned models are selected automatically.

The exported models are about 2 GB each and are excluded from version control by `.gitignore`. They can be
regenerated at any time with the notebook.

After editing `seed_data.py` or any other file, run `python make_notebook.py` to rebuild `finetune_colab.ipynb`
with the current files embedded, then upload the new notebook.

### Using a local GPU instead of Colab

On a Linux or WSL machine with an NVIDIA GPU:

```
pip install unsloth
python generate_dataset.py
python finetune_llm.py --stage 1
python finetune_llm.py --stage 2
python install_models.py
```

### Evaluating the result

Write several functionalities that do not appear in `seed_data.py`, run each with the base models and with the
fine-tuned models (the model name can be typed into the web page), and compare the validity of the output, the
number of requirements flagged for review, and the quality of the SDLC choice and justification.

### Troubleshooting

| Symptom | Resolution |
|---|---|
| Google Drive reports `mount failed` | Allow pop-ups, choose the same Google account Colab uses, accept every screen and run the cell again; or use the direct-download option. |
| Notebook step 3 reports `Missing: ...` | Step 2 did not complete. Run it again and read its output. |
| Unsloth installation conflicts or API errors | Follow the current installation notes at https://docs.unsloth.ai. `finetune_llm.py` already handles the different `SFTTrainer` signatures used by recent TRL versions. |
| Out of disk space on Colab | Delete the `*_gguf_work` and `*_checkpoints` folders of the first stage before training the second (the notebook does this). |
| `No quantised .gguf found` | Look inside `exports/<name>_gguf_work*` and copy the Q4_K_M file next to the Modelfile as `model.Q4_K_M.gguf`. |
| Model returns invalid JSON | Reduce `--epochs` (over-training) or add more varied training examples. |
| Status box says a model is not installed | Run `ollama list`. Use `ollama pull <name>` for base models or `python install_models.py` for fine-tuned ones. |

## Testing

```
python -m unittest -v
```

The suite contains 23 tests and uses fake models, so no GPU, Ollama installation or API key is required. It covers
paragraph-aware chunking, retrieval of the correct document, the pipeline order (retrieval before LLM 1, and LLM 2
restricted to the SDLC decision and run after LLM 1), grounding flags, handling of malformed model output, the
Ollama HTTP requests (including the JSON retry and the missing-model message), the command-line interface, the
generated datasets, consistency between the training text and the Ollama prompt format, the three-step web API, and
the notebook's recreation of every project file.

## Design notes

- **Retrieval before generation.** Evidence is retrieved first and supplied to LLM 1, so requirements are grounded
  in the existing documents. LLM 2 receives the same evidence together with the requirements.
- **Two different models.** Requirement writing and SDLC selection are separate tasks with separate models, and
  LLM 2 is explicitly restricted to the SDLC decision.
- **Deterministic grounding check.** Each requirement is compared with the retrieved evidence using TF-IDF cosine
  similarity. This does not depend on the LLM, and weak support triggers a "needs human review" flag.
- **Shared prompts.** The prompt builders in `pipeline.py` are used by the pipeline and by the dataset generator.
- **Robust output handling.** Replies are requested as JSON, parsed tolerantly, retried once on invalid JSON, and
  normalised before use. Model output is escaped before it is displayed in the web page.

## Limitations

- TF-IDF retrieval is keyword-based and can miss paraphrased content. A neural embedding retriever can replace it
  by providing the same `index`, `retrieve_scored` and `best_support` methods.
- The grounding score measures lexical similarity, not correctness. Requirements that are not flagged still need
  human review.
- Models of three billion parameters give modest text quality. Larger Ollama models or Gemini produce better results
  at the cost of speed or local execution.
- The sample documents and training scenarios are synthetic and illustrative.
- Unsloth and TRL change frequently. If the training notebook fails after a library update, the version-specific
  calls in `finetune_llm.py` may need adjusting.

## Project structure

| File | Purpose |
|---|---|
| `web_app.py` | Web application (`python web_app.py`) |
| `main.py` | Command-line interface |
| `pipeline.py` | The three pipeline steps and shared prompts |
| `retriever.py` | Chunking, TF-IDF retrieval and grounding score |
| `llm_clients.py` | Ollama and Gemini clients, JSON repair, fine-tuned model detection |
| `seed_data.py` | 28 synthetic training scenarios |
| `generate_dataset.py` | Builds `train_stage1.jsonl` and `train_stage2.jsonl` |
| `finetune_llm.py` | Fine-tunes one model with Unsloth QLoRA and exports GGUF and Modelfile |
| `finetune_colab.ipynb` | Self-contained Colab notebook for fine-tuning |
| `make_notebook.py` | Rebuilds the notebook with the current project files embedded |
| `install_models.py` | Registers the exported models in Ollama |
| `test_pipeline.py` | Automated tests |
| `sample_docs/` | Seven illustrative requirement and policy documents (the retrieval corpus) |
| `.gitignore` | Excludes exported models, caches and local files from version control |
