# -*- coding: utf-8 -*-
"""Live-LLM smoke (P2-F task (i)) — NOT a determinism battery.

This file exercises the REAL Ollama endpoints on this rig for the first time
(``/api/embeddings`` via ``nomic-embed-text`` + ``/api/generate`` via
``qwen2.5:1.5b``) as an optional integration smoke.  (qwen2.5:1.5b is a
generation-only model — Ollama has no ``/api/embeddings`` backing for it, so
embedding must go through an embedding model.)

Policy:
  * if no Ollama server answers ``OllamaAdapter().available()`` the whole file
    prints a single SKIP line and EXITS 0 — a skip must never masquerade as a
    pass (the checklist counts ``tests passed``; a skipped run reports 0).
  * if live, it runs the smoke cases:
      - embed: returns EXACTLY ``dim`` floats and unit norm (the adapter
        pads/truncates live vectors to ``dim`` — the review item the demo
        also pins at corpus build + query time);
      - embed: live path is NOT the deterministic hash-BoW fallback;
      - complete: returns a non-empty, stripped string (sanitised output).

Run:  python tests\test_llm_live.py      (from repo root)
Exit 0 iff SKIP (clean skip) or ALL_OK.  Also pytest-compatible (skipif).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.llm import OllamaAdapter, _hash_embed, _l2normalize  # noqa: E402

DIM = 768                     # nomic-embed-text native dim
EMBED_PROMPT = "heart rhythm monitoring device prevents arrhythmia related stroke"
GEN_PROMPT = "Answer with one short sentence: what prevents stroke?"

ADAPTER = OllamaAdapter(embed_model="nomic-embed-text")
LIVE = ADAPTER.available()


def test_live_embed_exact_dim_unit_norm():
    v = ADAPTER.embed(EMBED_PROMPT, dim=DIM)
    assert len(v) == DIM, f"live embed returned {len(v)} != {DIM}"
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6, "embedding must be unit norm"


def test_live_embed_is_not_hash_fallback():
    v = ADAPTER.embed(EMBED_PROMPT, dim=DIM)
    hb = _l2normalize(_hash_embed(EMBED_PROMPT, DIM), DIM)
    diff = max(abs(a - b) for a, b in zip(v, hb))
    assert diff > 1e-9, "server appeared to fall back to hash-BoW"


def test_live_generate_sanitised_nonempty():
    out = ADAPTER.complete(GEN_PROMPT, temperature=0.2, max_tokens=64)
    assert isinstance(out, str) and out.strip(), "generation must be non-empty"
    assert "\x00" not in out, "control byte in generated output"


def main():
    if not LIVE:
        print(json.dumps({"module": "llm_live", "status": "SKIP",
                          "reason": "no Ollama server at "
                                    + "http://127.0.0.1:11434 (clean skip; "
                                    "counted as 0/0, NOT as a pass)"}, indent=2))
        print("SKIP::llm_live -> 0/0 (skipped run is not a pass)")
        return 0
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report = {
        "module": "llm_live",
        "host": ADAPTER.host,
        "model": ADAPTER.model,
        "embed_model": ADAPTER.embed_model,
        "status": "PASS" if not fails else "FAIL",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())