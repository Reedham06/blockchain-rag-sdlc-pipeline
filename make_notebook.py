"""Build finetune_colab.ipynb with the CURRENT project files embedded inside it.

    python make_notebook.py

Why: the notebook then needs no zip upload - Colab creates every file itself from the notebook, so there is
nothing to zip wrongly. If you edit seed_data.py (to add training examples) or any other file, re-run this
script and upload the new notebook to Colab.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DELIM = "__END_OF_FILE__"

# needed for training
TRAIN_FILES = ["retriever.py", "llm_clients.py", "pipeline.py", "seed_data.py", "generate_dataset.py", "finetune_llm.py"]
# the rest of the project (optional: lets Colab hand you the complete project as one zip)
EXTRA_FILES = ["web_app.py", "main.py", "install_models.py", "make_notebook.py", "test_pipeline.py", "requirements.txt", "README.md"]


def _read(rel: str) -> str:
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n").rstrip("\n") + "\n"
    if any(line == DELIM for line in text.split("\n")):     # only a line that is exactly the delimiter would end the heredoc
        raise ValueError(f"{rel} contains a line equal to the reserved delimiter {DELIM}")
    return text


def _doc_files() -> list:
    return [os.path.join("sample_docs", n) for n in sorted(os.listdir(os.path.join(HERE, "sample_docs")))
            if n.lower().endswith((".txt", ".md"))]


def bash_cell(files: list) -> str:
    """One `%%bash` cell that writes every file with a quoted heredoc."""
    lines = ["%%bash", "set -e", "mkdir -p sample_docs"]
    for rel in files:
        lines += [f"cat > {rel} << '{DELIM}'", _read(rel).rstrip("\n"), DELIM]
    lines.append('echo "Created $(ls | wc -l) items:"; ls')
    return "\n".join(lines)


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def build() -> dict:
    cells = [
        _md("""# Fine-tune the two models (free Colab GPU)

**You do not need to upload or zip anything.** This notebook already contains all the project code and creates the files itself.

**Before you start:** menu *Runtime -> Change runtime type -> **T4 GPU** -> Save*. Then run the cells below **one by one, in order** (click the play button on each, wait for it to finish)."""),
        _md("### 1. Check that a GPU is attached (you should see a table that mentions a Tesla T4)"),
        _code("!nvidia-smi"),
        _md("### 2. Create the project files inside Colab (just run it - there is no need to read the code)"),
        _code(bash_cell(TRAIN_FILES + _doc_files())),
        _md("### 3. Check that everything was created"),
        _code('''import os
need = ["finetune_llm.py", "generate_dataset.py", "pipeline.py", "retriever.py", "llm_clients.py", "seed_data.py"]
missing = [f for f in need if not os.path.exists(f)]
docs = os.listdir("sample_docs") if os.path.isdir("sample_docs") else []
assert not missing and docs, f"Missing: {missing or 'sample_docs/*'}. Re-run step 2 and read its output for errors."
print("OK - files present:", sorted(os.listdir(".")))
print("Sample documents:", docs)'''),
        _md("""### 4. (Optional) Use your own edited training data
Only if you edited `seed_data.py` on your computer and did **not** rebuild this notebook with `python make_notebook.py`.
Remove the `#` at the start of the three lines, run the cell, and choose your `seed_data.py`. Otherwise skip this cell."""),
        _code('''# from google.colab import files
# up = files.upload()                     # choose your seed_data.py
# import shutil; shutil.move(list(up)[0], "seed_data.py")'''),
        _md("### 5. Install Unsloth (takes a few minutes; if Colab asks you to restart the runtime, do it, then continue from step 6 - your files stay)"),
        _code("!pip install -q unsloth"),
        _md("### 6. Build the training data (28 examples per model, from `seed_data.py`)"),
        _code("!python generate_dataset.py"),
        _md("### 7. Fine-tune LLM 1 (Llama-3.2-3B -> requirements). Roughly 10-30 minutes including the GGUF export."),
        _code("""!python finetune_llm.py --stage 1
!rm -rf exports/blockchain-req-llm_gguf_work exports/blockchain-req-llm_checkpoints   # free disk space"""),
        _md("### 8. Fine-tune LLM 2 (Qwen2.5-3B -> SDLC only)"),
        _code("""!python finetune_llm.py --stage 2
!rm -rf exports/blockchain-sdlc-llm_gguf_work exports/blockchain-sdlc-llm_checkpoints"""),
        _md("""### 9. Get the two models out of Colab
**Do this before the session ends: Colab deletes its files afterwards. Do not restart or disconnect the runtime until you have saved them.**
First check that training produced the files (you should see two folders, each with a `Modelfile` and a ~2 GB `model.Q4_K_M.gguf`):"""),
        _code("!ls -lh exports/blockchain-req-llm exports/blockchain-sdlc-llm"),
        _md("""**Option A - Google Drive (recommended).** A pop-up asks you to connect Google Drive: allow pop-ups for this site, pick the **same Google account** that Colab is signed in with, and click *Allow* / *Continue* on every screen.
If you get `mount failed`, run the cell again, or use Option B below."""),
        _code("""from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!mkdir -p /content/drive/MyDrive/blockchain_llms
!cp -r exports/blockchain-req-llm exports/blockchain-sdlc-llm /content/drive/MyDrive/blockchain_llms/
!ls -lh /content/drive/MyDrive/blockchain_llms/*"""),
        _md("""**Option B - download directly (use this if Drive keeps failing).** Makes one zip per model (~2 GB each) and downloads them.
On your computer, create a folder named `exports` inside your project folder and unzip **both** zips into it."""),
        _code("""import os, subprocess
from google.colab import files
for name in ["blockchain-req-llm", "blockchain-sdlc-llm"]:
    subprocess.run(["zip", "-0", "-qr", f"../{name}.zip", name], cwd="exports", check=True)
    print(name, round(os.path.getsize(name + ".zip") / 1e9, 2), "GB - downloading...")
    files.download(name + ".zip")"""),
        _md("""### 10. (Optional) Download the complete project code
Only needed if you don't already have the project folder on your computer. It builds one small zip with every project file."""),
        _code(bash_cell(EXTRA_FILES)),
        _code("""!zip -qr blockchain_rag_project_code.zip *.py *.md requirements.txt sample_docs train_stage1.jsonl train_stage2.jsonl
from google.colab import files
files.download("blockchain_rag_project_code.zip")"""),
        _md("""### 11. On your own computer
1. Put the models in a folder named `exports` inside your project folder, so you have `exports/blockchain-req-llm/` and
   `exports/blockchain-sdlc-llm/` (each with a `Modelfile` and `model.Q4_K_M.gguf`). Option A: download the `blockchain_llms`
   folder from Google Drive and rename it `exports`. Option B: unzip both downloaded zips into `exports`.
2. Make sure the Ollama app is running, then in the project folder run: `python install_models.py`
3. Start the website: `python web_app.py` - the status box will say it is using your fine-tuned models."""),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
        "accelerator": "GPU", "colab": {"name": "finetune_colab.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}


def main() -> None:
    out = os.path.join(HERE, "finetune_colab.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1, ensure_ascii=False)
    print(f"[ok] wrote {out} ({os.path.getsize(out) // 1024} KB) with {len(TRAIN_FILES) + len(EXTRA_FILES)} project files + sample_docs embedded")


if __name__ == "__main__":
    main()
