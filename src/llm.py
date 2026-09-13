# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""LLM + embedding backbone (SafeRAG §IV experiments: Llama-3.x / Dr. Llama,
Qwen-2.5 series). Phase-1 uses local Ollama ``qwen2.5:1.5b`` for BOTH
embeddings and generation (Rs 0, offline-capable).

SafeRAG paper's practical route: an embedding model E maps text->unit vector
(Eq.  dribble5) and the LLM G generates the grounded answer. Ollama supplies
both behind one HTTP adapter.
"""

from __future__ import annotations

import json
import os
import time

import requests

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("SAFERAG_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.environ.get("SAFERAG_EMBED_MODEL", "qwen2.5:1.5b")


class OllamaAdapter:
    def __init__(self, host: str = OLLAMA_HOST, model: str = MODEL,
                 embed_model: str = EMBED_MODEL):
        self.host = host
        self.model = model
        self.embed_model = embed_model
        self._session = requests.Session()
        self.tag = self.check() if False else None  # lazy

    # -- state ---------------------------------------------------------- #
    def check(self) -> dict:
        r = self._session.get(f"{self.host}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json()["models"]

    def available(self) -> bool:
        try:
            return len(self.check()) > 0
        except Exception:
            return False

    # -- embeddings ----------------------------------------------------- #
    def embed(self, text: str, dim: int = 64, normalize: bool = True) -> list[float]:
        """
        Embed text via the local model. If the model reports embedding
        support we use it directly; otherwise we build a *deterministic*
        local embedding (token-hash bag-of-words projected to `dim`, then
        L2-normalised).
        """
        try:
            r = self._session.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=300,
            )
            r.raise_for_status()
            vec = r.json().get("embedding")
            if vec:
                return _l2normalize(vec, dim) if normalize else list(vec)[:dim]
        except Exception:
            pass
        # deterministic fallback: fast hashing projection
        vec = _hash_embed(text, dim)
        return _l2normalize(vec, dim) if normalize else vec

    # -- text generation -------------------------------------------------- #
    def complete(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 256) -> str:
        r = self._session.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt,
                  "stream": False, "temperature": temperature,
                  "options": {"num_predict": max_tokens}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()


def _hash_embed(text: str, dim: int) -> list[float]:
    import hashlib
    out = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        out[h % dim] += 1.0
    return out


def _l2normalize(vec, dim):
    if _HAS_NP:
        v = np.array(vec, dtype=float)
        if len(v) > dim:
            v = v[:dim]
        if len(v) < dim:
            v = np.pad(v, (0, dim - len(v)))
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v = v / n
        return v.tolist()
    # pure-python fallback
    v = [float(x) for x in vec[:dim]]
    n = sum(x * x for x in v) ** 0.5
    if n > 1e-12:
        v = [x / n for x in v]
    return v
