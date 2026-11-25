"""Fine-tune ONE of the two models and export it for Ollama.   (needs an NVIDIA GPU - e.g. free Colab T4)

    python generate_dataset.py                 # once: builds train_stage1.jsonl / train_stage2.jsonl
    python finetune_llm.py --stage 1           # LLM 1: Llama-3.2-3B -> requirements   => exports/blockchain-req-llm/
    python finetune_llm.py --stage 2           # LLM 2: Qwen2.5-3B   -> SDLC only      => exports/blockchain-sdlc-llm/

Each export folder holds `model.Q4_K_M.gguf` + a `Modelfile`. Copy the `exports/` folder to the machine that
runs Ollama and run `python install_models.py` there (see README).

Design note: the text format used for training (FORMATS below) and the TEMPLATE written into the Modelfile are
defined side by side, so the model is trained on exactly the prompt format Ollama will send it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

MAX_SEQ_LENGTH = 3072

STAGES = {
    1: {"base": "unsloth/Llama-3.2-3B-Instruct", "family": "llama", "data": "train_stage1.jsonl", "name": "blockchain-req-llm"},
    2: {"base": "unsloth/Qwen2.5-3B-Instruct", "family": "qwen", "data": "train_stage2.jsonl", "name": "blockchain-sdlc-llm"},
}

LLAMA_TEMPLATE = """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

QWEN_TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

FORMATS = {
    "llama": {
        "block": lambda role, content: f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
        "generation_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stops": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        "template": LLAMA_TEMPLATE,
    },
    "qwen": {
        "block": lambda role, content: f"<|im_start|>{role}\n{content}<|im_end|>\n",
        "generation_prefix": "<|im_start|>assistant\n",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "stops": ["<|im_start|>", "<|im_end|>"],
        "template": QWEN_TEMPLATE,
    },
}


# ----------------------------------------------------------------------------- pure helpers (unit-tested, no GPU needed)
def render_example(messages: list, family: str) -> str:
    """Full training text for one {system, user, assistant} example (no BOS token: the tokenizer adds it)."""
    block = FORMATS[family]["block"]
    return "".join(block(m["role"], m["content"]) for m in messages)


def render_prompt(system: str, user: str, family: str) -> str:
    fmt = FORMATS[family]
    return fmt["block"]("system", system) + fmt["block"]("user", user) + fmt["generation_prefix"]


def modelfile_text(gguf_name: str, family: str, num_ctx: int = 4096) -> str:
    fmt = FORMATS[family]
    lines = [f"FROM ./{gguf_name}", f'TEMPLATE """{fmt["template"]}"""']
    lines += [f'PARAMETER stop "{s}"' for s in fmt["stops"]]
    lines += ["PARAMETER temperature 0.2", f"PARAMETER num_ctx {num_ctx}"]
    return "\n".join(lines) + "\n"


def find_gguf(work_dir: str) -> str:
    """Locate the quantised GGUF that Unsloth wrote (its folder/file naming differs between versions)."""
    candidates = []
    for d in (work_dir, work_dir + "_gguf"):
        candidates += glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True)
    candidates = [c for c in candidates if not any(t in os.path.basename(c).lower() for t in ("f16", "bf16", "f32"))]
    if not candidates:
        raise FileNotFoundError(f"No quantised .gguf file found under {work_dir}[_gguf]. Look inside those folders manually.")
    q4 = [c for c in candidates if "q4_k_m" in os.path.basename(c).lower()]
    return (q4 or candidates)[0]


# ----------------------------------------------------------------------------- training
def make_trainer(SFTConfig, SFTTrainer, model, tokenizer, dataset, cfg: dict):
    """Build an SFTTrainer across TRL versions (argument names moved between releases)."""
    attempts = [
        ({"dataset_text_field": "text", "max_seq_length": MAX_SEQ_LENGTH}, {"processing_class": tokenizer}),
        ({"dataset_text_field": "text", "max_length": MAX_SEQ_LENGTH}, {"processing_class": tokenizer}),
        ({}, {"tokenizer": tokenizer, "dataset_text_field": "text", "max_seq_length": MAX_SEQ_LENGTH}),
    ]
    last = None
    for cfg_extra, trainer_kw in attempts:
        try:
            return SFTTrainer(model=model, train_dataset=dataset, args=SFTConfig(**cfg, **cfg_extra), **trainer_kw)
        except TypeError as e:
            last = e
    raise RuntimeError(f"Could not build SFTTrainer with the installed TRL version: {last}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune one stage and export a GGUF + Modelfile for Ollama")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--epochs", type=float, default=3, help="2-3 is plenty for a small dataset; more will over-fit")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-model", default=None, help="override the Hugging Face base model")
    p.add_argument("--data", default=None, help="override the training JSONL")
    p.add_argument("--name", default=None, help="override the Ollama model name")
    p.add_argument("--out", default="exports")
    args = p.parse_args()

    cfg = STAGES[args.stage]
    family, base = cfg["family"], args.base_model or cfg["base"]
    data_path, name = args.data or cfg["data"], args.name or cfg["name"]
    fmt = FORMATS[family]
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found - run `python generate_dataset.py` first.")

    # Heavy imports are here (not at the top) so `--help` and the unit tests work without a GPU.
    from unsloth import FastLanguageModel   # must be imported before trl / transformers
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Use a GPU runtime (Colab: Runtime > Change runtime type > T4 GPU).")

    print(f"[1/5] Loading base model {base} in 4-bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    print(f"[2/5] Preparing {data_path} ...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(lambda ex: {"text": render_example(ex["messages"], family)}, remove_columns=dataset.column_names)
    longest = max(len(tokenizer(t)["input_ids"]) for t in dataset["text"])
    print(f"      {len(dataset)} examples, longest = {longest} tokens (limit {MAX_SEQ_LENGTH})")
    if longest > MAX_SEQ_LENGTH:
        print("      WARNING: some examples are longer than the limit and will be truncated - lower top-k or raise MAX_SEQ_LENGTH.")

    print("[3/5] Training (QLoRA) ...")
    use_bf16 = torch.cuda.is_bf16_supported()
    trainer = make_trainer(SFTConfig, SFTTrainer, model, tokenizer, dataset, dict(
        per_device_train_batch_size=1, gradient_accumulation_steps=4, warmup_steps=3,
        num_train_epochs=args.epochs, learning_rate=args.lr, fp16=not use_bf16, bf16=use_bf16,
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=42, output_dir=os.path.join(args.out, name + "_checkpoints"), report_to="none"))
    try:  # learn only from the assistant's answer, not from the long prompt
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, instruction_part=fmt["instruction_part"], response_part=fmt["response_part"])
    except Exception as e:
        print(f"      (train_on_responses_only unavailable, training on the full text instead: {e})")
    trainer.train()

    print("[4/5] Format check (a sanity check on a training prompt - NOT a quality test) ...")
    FastLanguageModel.for_inference(model)
    first = json.loads(open(data_path, encoding="utf-8").readline())["messages"]
    inputs = tokenizer(render_prompt(first[0]["content"], first[1]["content"], family), return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=700, do_sample=True, temperature=0.2, use_cache=True)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        json.loads(reply[reply.find("{"): reply.rfind("}") + 1])
        print("      model replied with valid JSON.")
    except ValueError:
        print("      WARNING: reply was not valid JSON. Raw reply:\n" + reply[:600])

    print("[5/5] Exporting for Ollama ...")
    os.makedirs(args.out, exist_ok=True)
    adapter_dir = os.path.join(args.out, name + "_lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    work = os.path.join(args.out, name + "_gguf_work")
    model.save_pretrained_gguf(work, tokenizer, quantization_method="q4_k_m")
    export_dir = os.path.join(args.out, name)
    os.makedirs(export_dir, exist_ok=True)
    shutil.copy(find_gguf(work), os.path.join(export_dir, "model.Q4_K_M.gguf"))
    with open(os.path.join(export_dir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(modelfile_text("model.Q4_K_M.gguf", family))
    print(f"\n[done] {export_dir}/ is ready. Copy the whole 'exports' folder to your computer, then run: python install_models.py")


if __name__ == "__main__":
    main()
