# -*- coding: utf-8 -*-
"""
LLM-stage tests (Phase-1 T4) —— OllamaAdapter, fully OFFLINE.

The online path (real Ollama /api/embeddings, /api/generate) is NOT exercised
here — no server, no network. Instead we inject a fake HTTP session so every
test is deterministic and offline:

  - live-up behaviour is simulated at the wire level (embedding JSON echoed
    through, generation echo, tags check),
  - the deterministic hash-BoW fallback `_hash_embed` is asserted directly
    (the same fallback `embed()` uses when the server is down),
  - the no-server path is asserted by a session that raises ConnectionError.

Run:  python tests\test_llm.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.llm import (OllamaAdapter, _hash_embed, _l2normalize)  # noqa: E402
from requests import exceptions as rexc                       # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Wire-level fake: get=tags, post=dispatch on endpoint."""

    def __init__(self, down=False, tags=None, embedding=None, response=None):
        self.down = down
        self.tags = tags or []
        self.embedding = embedding
        self.response = response
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("get", url))
        if self.down:
            raise rexc.ConnectionError("connection refused (fake)")
        return _Resp({"models": self.tags})

    def post(self, url, json=None, timeout=None):
        self.calls.append(("post", url))
        if self.down:
            raise rexc.ConnectionError("connection refused (fake)")
        if "embeddings" in url:
            return _Resp({"embedding": self.embedding})
        return _Resp({"response": self.response})


def _adapter(fake):
    a = OllamaAdapter(host="http://127.0.0.1:9999")
    a._session = fake
    return a


# -- deterministic fallback primitives ------------------------------------ #

def test_hash_embed_deterministic():
    v1 = _hash_embed("the patient presented with fever", 64)
    v2 = _hash_embed("the patient presented with fever", 64)
    assert v1 == v2
    assert len(v1) == 64


def test_hash_embed_changes_with_vocabulary():
    a = _hash_embed("alpha beta george", 256)
    b = _hash_embed("delta epsilon zebra", 256)
    assert a != b


def test_l2normalize_unit_norm():
    out = _l2normalize([3.0, 4.0], 2)
    assert abs(sum(x * x for x in out) - 1.0) < 1e-12, out


def test_l2normalize_pads_and_truncates():
    out = _l2normalize([1.0, 2.0], 4)
    assert len(out) == 4
    assert out[2:] == [0.0, 0.0]
    assert _l2normalize([1.0, 2.0, 3.0, 4.0, 5.0], 3) == _l2normalize([1.0, 2.0, 3.0], 3)


def test_l2normalize_zero_vector_stays_zero():
    assert _l2normalize([0.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0]


# -- no-server behaviour (deterministic fallback) --------------------------- #

def test_available_false_when_server_down():
    a = _adapter(_FakeSession(down=True))
    assert a.available() is False


def test_embed_falls_back_when_server_down():
    a = _adapter(_FakeSession(down=True))
    v = a.embed("discharge summary", dim=64)
    assert len(v) == 64
    # same fallback as _hash_embed normalised
    assert v == _l2normalize(_hash_embed("discharge summary", 64), 64)
    # deterministic across calls
    assert a.embed("discharge summary", dim=64) == v


def test_embed_fallback_unit_norm():
    a = _adapter(_FakeSession(down=True))
    v = a.embed("some clinical note about heart failure", dim=32)
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9


# -- server-wire behaviour (fake responses) --------------------------------- #

def test_embed_uses_live_embedding_when_available():
    a = _adapter(_FakeSession(embedding=[3.0, 4.0]))
    v = a.embed("hello", dim=2, normalize=True)
    assert abs(sum(x * x for x in v) - 1.0) < 1e-12
    assert [round(x, 6) for x in v] == [0.6, 0.8]


def test_embed_no_normalize_slices_to_dim():
    a = _adapter(_FakeSession(embedding=[1.0, 2.0, 3.0, 4.0, 5.0]))
    assert a.embed("hello", dim=3, normalize=False) == [1.0, 2.0, 3.0]


def test_embed_empty_embedding_falls_back():
    # server says nothing usable -> deterministic fallback kicks in
    a = _adapter(_FakeSession(embedding=None))
    v = a.embed("vague server response", dim=64)
    assert v == _l2normalize(_hash_embed("vague server response", 64), 64)


def test_embed_no_normalize_pads_short_embedding():
    # live embedding SHORTER than dim must be zero-padded to exactly dim
    # (IPFE vector length is fixed at Setup(); a short vector would break it)
    a = _adapter(_FakeSession(embedding=[1.0, 2.0]))
    v = a.embed("hello", dim=4, normalize=False)
    assert v == [1.0, 2.0, 0.0, 0.0]
    assert len(v) == 4


def test_embed_normalize_pads_short_embedding():
    a = _adapter(_FakeSession(embedding=[3.0, 4.0]))
    v = a.embed("hello", dim=4, normalize=True)
    assert len(v) == 4
    assert abs(sum(x * x for x in v) - 1.0) < 1e-12


def test_complete_returns_response():
    a = _adapter(_FakeSession(response="The answer is 42."))
    assert a.complete("what is the answer?") == "The answer is 42."


def test_check_returns_models():
    a = _adapter(_FakeSession(tags=[{"name": "qwen2.5:1.5b"}]))
    assert a.check() == [{"name": "qwen2.5:1.5b"}]
    assert a.available() is True


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report = {
        "module": "llm",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
        "all_ok": not fails,
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())