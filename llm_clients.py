"""LLM clients.

Default setup = two *different* local Ollama models for blockchain requirements and SDLC selection:
  LLM 1 (requirements)  -> fine-tuned `blockchain-req-llm`  if installed, else base `llama3.2:3b`
  LLM 2 (SDLC only)     -> fine-tuned `blockchain-sdlc-llm` if installed, else base `qwen2.5:3b`
Gemini is available as an optional cloud alternative.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

# role -> (fine-tuned model name created by install_models.py, base model fallback)
DEFAULT_OLLAMA_MODELS = {
    "llm1": ("blockchain-req-llm", "llama3.2:3b"),
    "llm2": ("blockchain-sdlc-llm", "qwen2.5:3b"),
}
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class LLMError(RuntimeError):
    """Raised for any model/connection problem, with a human-readable message."""


def extract_json(text: str) -> dict:
    """Parse a JSON object out of model output, tolerating ``` fences and chatter."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


class LLMClient(ABC):
    label = "llm"

    @abstractmethod
    def _complete(self, system: str, user: str) -> str:
        """Return the raw text reply from the model."""

    def generate_json(self, system: str, user: str, retries: int = 1) -> dict:
        """Call the model and return parsed JSON; re-ask once if the reply wasn't valid JSON."""
        prompt, last_err = user, None
        for _ in range(retries + 1):
            raw = self._complete(system, prompt)
            try:
                return extract_json(raw)
            except ValueError as e:  # json.JSONDecodeError is a ValueError too
                last_err = e
                prompt = user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object, nothing else."
        raise LLMError(f"{self.label} did not return valid JSON ({last_err}).")


class OllamaClient(LLMClient):
    """Local model served by Ollama (https://ollama.com). Free, offline, no API key."""

    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 900, num_ctx: int = 4096):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout, self.num_ctx = timeout, num_ctx
        self.label = f"Ollama '{model}'"

    def _complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx, "num_predict": 1500},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"})
        print(f"  -> {self.label} is generating (this can take a minute on CPU)...", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 404:
                raise LLMError(
                    f"Ollama has no model named '{self.model}'. Install it with `ollama pull {self.model}` "
                    f"(base models) or `python install_models.py` (your fine-tuned models). Installed: "
                    f"{', '.join(ollama_installed_models(self.host) or []) or 'none'}."
                ) from e
            raise LLMError(f"Ollama error {e.code}: {body[:300]}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Start it (open the Ollama app, or run `ollama serve`) "
                f"and try again. Details: {e}"
            ) from e


class GeminiClient(LLMClient):
    """Optional cloud model. Needs `pip install google-genai` and a GEMINI_API_KEY."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is missing. Set the environment variable or paste the key in the web page.")
        try:
            from google import genai
        except ImportError as e:
            raise LLMError("The Gemini provider needs `pip install google-genai`.") from e
        self.model = model
        self.client = genai.Client(api_key=key)
        self.label = f"Gemini '{model}'"

    def _complete(self, system: str, user: str) -> str:
        from google.genai import types
        last_err = None
        for attempt in range(1, 4):
            try:
                print(f"  -> {self.label} (attempt {attempt})...", flush=True)
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=0.2, response_mime_type="application/json"),
                )
                return resp.text
            except Exception as e:  # retry only transient problems, fail fast on bad keys / bad model names
                last_err = e
                if not any(t in str(e) for t in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")):
                    break
                time.sleep(3 * attempt)
        raise LLMError(f"{self.label} failed: {last_err}")


def ollama_installed_models(host: Optional[str] = None) -> Optional[List[str]]:
    """Names of models installed in Ollama, or None if the Ollama server is not reachable."""
    host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
    except Exception:
        return None


def pick_default_ollama_model(role: str, installed: Optional[List[str]]) -> dict:
    """Prefer the fine-tuned model for this role when it is installed, else the base model."""
    finetuned, base = DEFAULT_OLLAMA_MODELS[role]
    names = {n.split(":")[0] for n in (installed or [])}
    if finetuned in names:
        return {"model": finetuned, "finetuned": True}
    return {"model": base, "finetuned": False}


def build_client(provider: str, model: Optional[str] = None, api_key: Optional[str] = None, role: str = "llm1") -> LLMClient:
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        return OllamaClient(model or pick_default_ollama_model(role, ollama_installed_models())["model"])
    if provider == "gemini":
        return GeminiClient(model or DEFAULT_GEMINI_MODEL, api_key=api_key)
    raise LLMError(f"Unknown provider {provider!r}. Choose 'ollama' or 'gemini'.")
