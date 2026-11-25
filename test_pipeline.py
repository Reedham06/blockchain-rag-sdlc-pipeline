"""Run with:  python -m unittest -v      (no GPU, no Ollama, no API key needed - models are faked)"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import finetune_llm
import generate_dataset
import llm_clients
import make_notebook
import pipeline
import shutil
import web_app
from llm_clients import LLMError, OllamaClient, extract_json, pick_default_ollama_model
from retriever import chunk_text

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "sample_docs")


class FakeLLM:
    def __init__(self, label, reply, log):
        self.label, self.reply, self.log = label, reply, log

    def generate_json(self, system, user, retries=1):
        self.log.append((self.label, system, user))
        return self.reply


LLM1_REPLY = {"requirements": [
    {"id": "REQ-001", "statement": "The pool contract shall execute withdrawals below 10,000 USDC equivalent immediately once collateral health checks pass.", "category": "Functional"},
    {"id": "REQ-002", "statement": "The system shall send a push notification to users when the app is updated.", "category": "Usability"}],
    "reasoning": "because"}
LLM2_REPLY = {"recommended_sdlc": "DevSecOps-V-Model Hybrid", "key_factors": ["a", "b"], "justification": "j"}


class RetrievalTests(unittest.TestCase):
    def test_chunks_end_on_paragraph_boundaries(self):
        text = "Para one is short.\n\nPara two is short too.\n\n" + " ".join(["word"] * 300)
        chunks = chunk_text(text, "t.txt", max_words=50, overlap=5)
        self.assertTrue(chunks[0].text.endswith("too."))      # short paragraphs merged, not cut mid-sentence
        self.assertTrue(all(len(c.text.split()) <= 50 for c in chunks))

    def test_retrieval_finds_the_right_document(self):
        r = pipeline.build_index(pipeline.load_documents(DOCS))
        top = lambda d, f: pipeline.retrieve_evidence(r, d, f, 4)[0]["source"]
        self.assertEqual(top("DeFi Lending", "Multi-signature approval for large withdrawals"), "defi_lending_policy.txt")
        self.assertEqual(top("Blockchain KYC / AML", "Sanctions screening of wallet addresses"), "kyc_aml_policy.txt")
        self.assertEqual(top("CBDC", "Offline retail payments"), "cbdc_policy.txt")

    def test_empty_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            pipeline.load_documents("", "")
        with self.assertRaises(ValueError):
            pipeline.load_documents("/definitely/not/a/folder")


class FlowTests(unittest.TestCase):
    def test_retrieve_first_then_llm1_then_llm2_sdlc_only(self):
        log = []
        r = pipeline.build_index(pipeline.load_documents(DOCS))
        result = pipeline.run_pipeline(r, "DeFi Lending", "Multi-signature approval for large withdrawals",
                                       FakeLLM("llm1", LLM1_REPLY, log), FakeLLM("llm2", LLM2_REPLY, log))
        self.assertEqual([c[0] for c in log], ["llm1", "llm2"])                 # LLM 2 runs after LLM 1
        _, _, llm1_prompt = log[0]
        self.assertIn("[E1]", llm1_prompt)                                       # LLM 1 already sees retrieved evidence
        self.assertIn(result["evidence"][0]["text"][:40], llm1_prompt)
        _, llm2_system, llm2_prompt = log[1]
        self.assertIn("do not propose or revise requirements", llm2_system)      # LLM 2 is SDLC-only
        self.assertIn("REQ-001", llm2_prompt)                                    # ...and receives the requirements
        self.assertIn("[E1]", llm2_prompt)                                       # ...plus the evidence
        self.assertEqual(result["sdlc"]["recommended_sdlc"], "DevSecOps-V-Model Hybrid")
        self.assertEqual(result["evidence"][0]["source"], "defi_lending_policy.txt")

    def test_grounding_flags_unsupported_requirement(self):
        log = []
        r = pipeline.build_index(pipeline.load_documents(DOCS))
        result = pipeline.run_pipeline(r, "DeFi Lending", "Multi-signature approval for large withdrawals",
                                       FakeLLM("llm1", LLM1_REPLY, log), FakeLLM("llm2", LLM2_REPLY, log))
        supported, unsupported = result["requirements"]
        self.assertFalse(supported["needs_review"])
        self.assertTrue(unsupported["needs_review"])                             # nothing in the evidence about push notifications
        self.assertEqual(result["needs_review_count"], 1)

    def test_no_evidence_flags_everything(self):
        r = pipeline.build_index(pipeline.load_documents("", "Totally unrelated text about gardening and tomatoes."))
        ev = pipeline.retrieve_evidence(r, "DeFi Lending", "withdrawals", 4)
        self.assertEqual(ev, [])
        grounded = pipeline.ground_requirements(r, [{"id": "REQ-001", "statement": "The pool shall do x.", "category": "F"}], ev)
        self.assertTrue(grounded[0]["needs_review"])

    def test_normalisation_copes_with_messy_model_output(self):
        reqs = pipeline.normalize_requirements(["The system shall do A.", {"Requirement": "The system shall do B.", "Type": "Security"}, 5, {}])
        self.assertEqual([r["id"] for r in reqs], ["REQ-001", "REQ-002"])
        self.assertEqual(reqs[1]["category"], "Security")
        self.assertEqual(pipeline.normalize_sdlc({"SDLC": "Agile", "factors": "one"})["key_factors"], ["one"])

    def test_model_with_no_usable_output_raises_clear_error(self):
        r = pipeline.build_index(pipeline.load_documents(DOCS))
        with self.assertRaises(LLMError):
            pipeline.generate_requirements("DeFi Lending", "x", [], FakeLLM("llm1", {"requirements": []}, []))


class ClientTests(unittest.TestCase):
    def test_extract_json_handles_fences_and_chatter(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here you go: {"a": 1} Hope that helps'), {"a": 1})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_unreachable_ollama_gives_actionable_error(self):
        with self.assertRaises(LLMError) as ctx:
            OllamaClient("whatever", host="http://127.0.0.1:9", timeout=3).generate_json("s", "u")
        self.assertIn("Cannot reach Ollama", str(ctx.exception))

    def test_default_model_prefers_finetuned_when_installed(self):
        self.assertEqual(pick_default_ollama_model("llm1", ["blockchain-req-llm:latest", "llama3.2:3b"]),
                         {"model": "blockchain-req-llm", "finetuned": True})
        self.assertEqual(pick_default_ollama_model("llm2", ["llama3.2:3b"]), {"model": "qwen2.5:3b", "finetuned": False})
        self.assertEqual(pick_default_ollama_model("llm1", None)["model"], "llama3.2:3b")
        self.assertNotEqual(llm_clients.DEFAULT_OLLAMA_MODELS["llm1"], llm_clients.DEFAULT_OLLAMA_MODELS["llm2"])


class DatasetAndFinetuneTests(unittest.TestCase):
    def test_dataset_is_valid_and_uses_runtime_prompts(self):
        s1, s2, problems = generate_dataset.build_examples(DOCS)
        self.assertEqual(problems, [])
        self.assertGreaterEqual(len(s1), 28)
        self.assertEqual(len(s1), len(s2))
        for line in s1 + s2:
            msgs = json.loads(line)["messages"]
            self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
            json.loads(msgs[2]["content"])                                       # assistant answer is valid JSON
        user1 = json.loads(s1[0])["messages"][1]["content"]
        self.assertIn("Evidence retrieved from the existing requirement documents", user1)
        self.assertIn(pipeline.STAGE1_SCHEMA, user1)
        labels = {json.loads(json.loads(l)["messages"][2]["content"])["recommended_sdlc"] for l in s2}
        self.assertGreaterEqual(len(labels), 6)                                  # not collapsed to one answer

    def test_training_text_matches_inference_format(self):
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}, {"role": "assistant", "content": "ANS"}]
        for family in ("llama", "qwen"):
            full = finetune_llm.render_example(msgs, family)
            prompt = finetune_llm.render_prompt("SYS", "USR", family)
            self.assertTrue(full.startswith(prompt))                             # training text = inference prompt + answer
            self.assertTrue(full[len(prompt):].startswith("ANS"))
            self.assertIn(finetune_llm.FORMATS[family]["response_part"], full)
            mf = finetune_llm.modelfile_text("model.Q4_K_M.gguf", family)
            self.assertTrue(mf.startswith("FROM ./model.Q4_K_M.gguf"))
            self.assertIn("{{ .System }}", mf)
            self.assertTrue(all(f'PARAMETER stop "{s}"' in mf for s in finetune_llm.FORMATS[family]["stops"]))

    def test_find_gguf_prefers_q4_and_skips_f16(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "m_gguf_work")
            os.makedirs(work + "_gguf")
            for n in ("unsloth.F16.gguf", "unsloth.Q4_K_M.gguf"):
                open(os.path.join(work + "_gguf", n), "w").close()
            self.assertTrue(finetune_llm.find_gguf(work).endswith("unsloth.Q4_K_M.gguf"))
            with self.assertRaises(FileNotFoundError):
                finetune_llm.find_gguf(os.path.join(tmp, "nothing"))


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = []
        cls._orig = web_app.build_client
        web_app.build_client = lambda provider, model=None, api_key=None, role="llm1": FakeLLM(
            "llm1" if role == "llm1" else "llm2", LLM1_REPLY if role == "llm1" else LLM2_REPLY, cls.log)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        web_app.build_client = cls._orig

    def _call(self, path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page_and_config(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10) as r:
            html = r.read().decode()
        self.assertIn("Run pipeline", html)
        status, cfg = self._call("/api/config")
        self.assertEqual(status, 200)
        self.assertIn("defaults", cfg)

    def test_three_step_flow_through_http(self):
        base = {"docs": "sample_docs", "domain": "DeFi Lending", "functionality": "Multi-signature approval for large withdrawals", "top_k": 4}
        s, r1 = self._call("/api/retrieve", base)
        self.assertEqual(s, 200)
        self.assertEqual(r1["evidence"][0]["source"], "defi_lending_policy.txt")
        s, r2 = self._call("/api/requirements", {**base, "evidence": r1["evidence"], "provider": "ollama"})
        self.assertEqual(s, 200)
        self.assertEqual(len(r2["requirements"]), 2)
        s, r3 = self._call("/api/sdlc", {"domain": "DeFi Lending", "evidence": r1["evidence"], "requirements": r2["requirements"], "provider": "ollama"})
        self.assertEqual(s, 200)
        self.assertEqual(r3["sdlc"]["recommended_sdlc"], "DevSecOps-V-Model Hybrid")

    def test_bad_input_returns_clean_errors(self):
        s, d = self._call("/api/retrieve", {"docs": "sample_docs", "domain": "", "functionality": "x"})
        self.assertEqual(s, 400)
        self.assertIn("required", d["error"])
        s, d = self._call("/api/retrieve", {"docs": "no_such_folder", "domain": "a", "functionality": "b"})
        self.assertEqual(s, 400)
        self.assertEqual(self._call("/nope", {})[0], 404)


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Speaks just enough of Ollama's HTTP API to test our real client + CLI code paths."""
    chat_calls = []
    garbage_first = False

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"models": [{"name": "blockchain-req-llm:latest"}, {"name": "qwen2.5:3b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllamaHandler.chat_calls.append(body)
        if body["model"] == "missing-model":
            return self._send(404, {"error": "model not found"})
        if FakeOllamaHandler.garbage_first and len(FakeOllamaHandler.chat_calls) == 1:
            return self._send(200, {"message": {"content": "Sorry, here are some thoughts..."}})
        reply = LLM2_REPLY if "SDLC" in body["messages"][0]["content"] else LLM1_REPLY
        self._send(200, {"message": {"content": json.dumps(reply)}})

    def log_message(self, *a):
        pass


class OllamaProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeOllamaHandler.chat_calls.clear()
        FakeOllamaHandler.garbage_first = False

    def test_request_format_and_success(self):
        out = OllamaClient("qwen2.5:3b", host=self.host).generate_json("sys", "usr")
        self.assertEqual(out, LLM1_REPLY)
        sent = FakeOllamaHandler.chat_calls[0]
        self.assertEqual((sent["model"], sent["format"], sent["stream"]), ("qwen2.5:3b", "json", False))
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])

    def test_invalid_json_is_retried_once(self):
        FakeOllamaHandler.garbage_first = True
        self.assertEqual(OllamaClient("qwen2.5:3b", host=self.host).generate_json("sys", "usr"), LLM1_REPLY)
        self.assertEqual(len(FakeOllamaHandler.chat_calls), 2)
        self.assertIn("not valid JSON", FakeOllamaHandler.chat_calls[1]["messages"][1]["content"])

    def test_missing_model_message_tells_you_what_to_run(self):
        with self.assertRaises(LLMError) as ctx:
            OllamaClient("missing-model", host=self.host).generate_json("s", "u")
        self.assertIn("ollama pull missing-model", str(ctx.exception))

    def test_cli_end_to_end_uses_two_different_models(self):
        env = {**os.environ, "OLLAMA_HOST": self.host}
        run = subprocess.run([sys.executable, os.path.join(HERE, "main.py"), "--domain", "DeFi Lending",
                              "--functionality", "Multi-signature approval for large withdrawals"],
                             capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Recommended: DevSecOps-V-Model Hybrid", run.stdout)
        self.assertIn("NEEDS HUMAN REVIEW", run.stdout)
        models = [c["model"] for c in FakeOllamaHandler.chat_calls]
        self.assertEqual(models, ["blockchain-req-llm", "qwen2.5:3b"])      # fine-tuned LLM 1 auto-detected; LLM 2 falls back to base


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.nb = make_notebook.build()
        self.bash_cells = [c for c in self.nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%bash")]

    def test_notebook_structure(self):
        json.dumps(self.nb)                                            # serialisable
        own = [c for c in self.nb["cells"] if c["cell_type"] == "code" and c not in self.bash_cells]
        code = "\n".join("".join(c["source"]) for c in own)           # notebook's own cells (not the embedded project files)
        for needle in ("nvidia-smi", "pip install -q unsloth", "generate_dataset.py",
                       "finetune_llm.py --stage 1", "finetune_llm.py --stage 2", "drive.mount"):
            self.assertIn(needle, code)
        self.assertNotIn("zipfile.ZipFile(list(uploaded)", code)       # the fragile zip-upload step is gone
        self.assertEqual(len(self.bash_cells), 2)

    @unittest.skipUnless(shutil.which("bash"), "needs bash (Colab has it)")
    def test_bash_cells_recreate_every_project_file_exactly(self):
        """Run the notebook's file-creation cells the way Colab does (script fed to bash on stdin)."""
        with tempfile.TemporaryDirectory() as tmp:
            for cell in self.bash_cells:
                script = "".join(cell["source"]).split("\n", 1)[1]    # drop the %%bash magic line
                run = subprocess.run(["bash"], input=script, text=True, cwd=tmp, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            wanted = make_notebook.TRAIN_FILES + make_notebook.EXTRA_FILES + make_notebook._doc_files()
            for rel in wanted:
                original = open(os.path.join(HERE, rel), encoding="utf-8").read().replace("\r\n", "\n").rstrip("\n") + "\n"
                with open(os.path.join(tmp, rel), encoding="utf-8") as f:
                    self.assertEqual(f.read(), original, f"{rel} was not recreated exactly")
            # the recreated project must actually work: build the datasets inside the fresh folder
            run = subprocess.run([sys.executable, "generate_dataset.py"], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(os.path.exists(os.path.join(tmp, "train_stage1.jsonl")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
