"""
slm.py
------
Thin wrapper around a locally-running Small Language Model.

Supported backends (tried in order):
  1. Ollama  – recommended, runs many models (phi3, mistral, llama3, gemma2 …)
             Install: https://ollama.com   then: `ollama pull phi3`
  2. llama-cpp-python – for direct .gguf model files (no server needed)
  3. HuggingFace Transformers – any model in the Hub, loaded locally

Set the backend explicitly or let it auto-detect.
"""

from __future__ import annotations

import os
from typing import Generator

# ── backend availability ────────────────────────────────────────────────────

def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


_OLLAMA_AVAILABLE   = _try_import("ollama")
_LLAMA_CPP_AVAILABLE = _try_import("llama_cpp")
_HF_AVAILABLE       = _try_import("transformers")


# ── main class ──────────────────────────────────────────────────────────────

class LocalSLM:
    """
    Unified interface to a local Small Language Model.

    Parameters
    ----------
    backend : str
        "ollama" | "llama_cpp" | "transformers" | "auto"
    model : str
        - Ollama  : model name, e.g. "phi3", "mistral", "llama3.2"
        - llama_cpp: path to a .gguf file
        - transformers: HuggingFace model ID, e.g. "microsoft/phi-2"
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative)
    max_tokens : int
        Maximum tokens to generate per response.

    Quick start (Ollama backend – recommended)
    ------------------------------------------
    # 1. Install Ollama: https://ollama.com
    # 2. Pull a model:  ollama pull phi3
    # 3. Use it:
    slm = LocalSLM(backend="ollama", model="phi3")
    print(slm.generate("What is water treatment?"))
    """

    OLLAMA_DEFAULT_MODEL     = "phi3"
    LLAMA_CPP_DEFAULT_MODEL  = "model.gguf"
    HF_DEFAULT_MODEL         = "microsoft/phi-2"

    def __init__(
        self,
        backend: str = "auto",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs,
    ):
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._kwargs     = kwargs

        self.backend, self.model = self._resolve_backend(backend, model)
        self._client = None
        self._load_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response. Returns the full string."""
        if self.backend == "ollama":
            return self._ollama_generate(prompt, system)
        if self.backend == "llama_cpp":
            return self._llama_cpp_generate(prompt, system)
        if self.backend == "transformers":
            return self._hf_generate(prompt, system)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def stream(self, prompt: str, system: str | None = None) -> Generator[str, None, None]:
        """Stream the response token by token."""
        if self.backend == "ollama":
            yield from self._ollama_stream(prompt, system)
        elif self.backend == "llama_cpp":
            yield from self._llama_cpp_stream(prompt, system)
        elif self.backend == "transformers":
            yield self._hf_generate(prompt, system)  # no streaming for HF
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def __repr__(self) -> str:
        return f"LocalSLM(backend={self.backend!r}, model={self.model!r})"

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self, backend: str, model: str | None):
        if backend != "auto":
            return backend, model or self._default_model(backend)

        if _OLLAMA_AVAILABLE:
            return "ollama", model or self.OLLAMA_DEFAULT_MODEL
        if _LLAMA_CPP_AVAILABLE:
            return "llama_cpp", model or self.LLAMA_CPP_DEFAULT_MODEL
        if _HF_AVAILABLE:
            return "transformers", model or self.HF_DEFAULT_MODEL

        raise RuntimeError(
            "No SLM backend found.\n"
            "Install one of:\n"
            "  pip install ollama          (+ run: ollama pull phi3)\n"
            "  pip install llama-cpp-python\n"
            "  pip install transformers torch\n"
        )

    def _default_model(self, backend: str) -> str:
        return {
            "ollama":       self.OLLAMA_DEFAULT_MODEL,
            "llama_cpp":    self.LLAMA_CPP_DEFAULT_MODEL,
            "transformers": self.HF_DEFAULT_MODEL,
        }.get(backend, "")

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _load_backend(self):
        print(f"[LocalSLM] Using backend='{self.backend}', model='{self.model}'")
        if self.backend == "ollama":
            import ollama
            self._client = ollama
        elif self.backend == "llama_cpp":
            from llama_cpp import Llama
            self._client = Llama(
                model_path=self.model,
                n_ctx=self._kwargs.get("n_ctx", 4096),
                n_gpu_layers=self._kwargs.get("n_gpu_layers", -1),
                verbose=False,
            )
        elif self.backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            print(f"[LocalSLM] Loading HuggingFace model '{self.model}' (this may take a while) …")
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self._client = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
            )

    # ------------------------------------------------------------------
    # Ollama implementation
    # ------------------------------------------------------------------

    def _ollama_generate(self, prompt: str, system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return response["message"]["content"]

    def _ollama_stream(self, prompt: str, system: str | None) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        ):
            yield chunk["message"]["content"]

    # ------------------------------------------------------------------
    # llama-cpp-python implementation
    # ------------------------------------------------------------------

    def _llama_cpp_generate(self, prompt: str, system: str | None) -> str:
        full_prompt = f"<|system|>\n{system}\n" if system else ""
        full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"
        result = self._client(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _llama_cpp_stream(self, prompt: str, system: str | None) -> Generator[str, None, None]:
        full_prompt = f"<|system|>\n{system}\n" if system else ""
        full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"
        for chunk in self._client(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            echo=False,
        ):
            yield chunk["choices"][0]["text"]

    # ------------------------------------------------------------------
    # HuggingFace Transformers implementation
    # ------------------------------------------------------------------

    def _hf_generate(self, prompt: str, system: str | None) -> str:
        full_prompt = f"System: {system}\n\nUser: {prompt}\nAssistant:" if system else f"User: {prompt}\nAssistant:"
        out = self._client(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return out[0]["generated_text"]
