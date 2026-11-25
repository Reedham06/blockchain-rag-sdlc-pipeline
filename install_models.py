"""Register your fine-tuned models in Ollama (run this on the computer where Ollama is installed).

    python install_models.py                 # looks in ./exports
    python install_models.py --exports D:\\downloads\\exports

For every <exports>/<name>/Modelfile it runs `ollama create <name> -f <that Modelfile>`.
After that, the website and CLI pick the fine-tuned models up automatically.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Create Ollama models from fine-tuned exports")
    p.add_argument("--exports", default="exports", help="folder containing blockchain-req-llm/ and blockchain-sdlc-llm/")
    args = p.parse_args()

    if not shutil.which("ollama"):
        sys.exit("`ollama` was not found on PATH. Install it from https://ollama.com and make sure the app is running.")
    if not os.path.isdir(args.exports):
        sys.exit(f"Folder not found: {args.exports}")

    found = 0
    for name in sorted(os.listdir(args.exports)):
        modelfile = os.path.join(args.exports, name, "Modelfile")
        gguf = os.path.join(args.exports, name, "model.Q4_K_M.gguf")
        if not os.path.isfile(modelfile):
            continue
        if not os.path.isfile(gguf):
            print(f"[skip] {name}: model.Q4_K_M.gguf is missing next to the Modelfile")
            continue
        found += 1
        print(f"\n[*] ollama create {name} ...")
        if subprocess.call(["ollama", "create", name, "-f", modelfile]) != 0:
            sys.exit(f"`ollama create {name}` failed. Is the Ollama app running?")
    if not found:
        sys.exit(f"No <name>/Modelfile found inside {args.exports}.")
    print("\n[ok] Done. Installed models:")
    subprocess.call(["ollama", "list"])


if __name__ == "__main__":
    main()
