# -*- coding: utf-8 -*-
"""
Retrieval-engine tests (Phase-1 T3) —— real encrypted IPFE top-k.

Acceptance criteria (per plan):
  (a) cosine_ipfe recovers the SAME integer the grid proves against the
      vendored oracle (abs == expected quantised inner product), AND the
      recovered MAGNITUDE matches the plaintext cosine within the documented
      quantisation bound n/B (catches monotonic-but-wrong transforms).
  (b) rank order == plaintext order on the same vectors.
Authorisation gate + k-truncation + negative-cosine path also covered.

Run:  python tests\test_retrieval.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.attributes import build_tree, Attributes       # noqa: E402
from src.ipfe import IPFEScheme                          # noqa: E402
from src.retrieval import RetrievalEngine, RetrievedDoc  # noqa: E402


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _quant(v, prec):
    B = 10 ** prec
    return [int(round(x * B)) for x in v]


class _Doc:
    def __init__(self, doc_id, vec, ct, policy, group_bits=0):
        self.doc_id = doc_id
        self.vec = np.asarray(vec, dtype=np.float64)
        self.ct = ct
        self.policy = policy
        self.group_bits = group_bits

    def policy_cached(self):
        return self.policy


# --------------------------------------------------------------------------- #
# (a) magnitude: recovered cosine == plaintext cosine within quantisation bound
# --------------------------------------------------------------------------- #

def test_cosine_magnitude_matches_plaintext():
    # asserts BOTH: exact integer equality with the rounded inner product
    # (which the grid ties to the FeDDH oracle), AND cos-vs-plaintext within n/B.
    for n in (4, 8, 16, 32):
        prec = 2
        scheme = IPFEScheme.setup(n, quant_bits=prec)
        engine = RetrievalEngine(scheme, k=3)
        rng = np.random.default_rng(10 + n)
        x = _unit(rng.normal(size=n))
        v = _unit(rng.normal(size=n))
        Ct, _ = scheme.encrypt(list(x))
        val = engine.cosine_ipfe(Ct, list(v))
        B = 10 ** prec
        expected_int = int(np.dot(_quant(x, prec), _quant(v, prec)))
        assert abs(val * B * B - expected_int) < 1e-6, (n, val * B * B, expected_int)
        plain = float(np.dot(x, v))
        assert abs(val - plain) <= n / B, (n, val, plain, abs(val - plain), n / B)


def test_cosine_negative_recovered():
    # exercises the BSGS inverse (DiscreteLogNotFound) branch the grid sidesteps
    # by flipping v to keep cos>=0. docs built so cos is clearly negative.
    n, prec = 16, 2
    scheme = IPFEScheme.setup(n, quant_bits=prec)
    engine = RetrievalEngine(scheme, k=3)
    rng = np.random.default_rng(4)
    v = _unit(rng.normal(size=n))
    x = _unit(rng.normal(size=n))
    if float(np.dot(x, v)) >= 0:
        x = -x
    Ct, _ = scheme.encrypt(list(x))
    got = engine.cosine_ipfe(Ct, list(v))
    B = 10 ** prec
    assert abs(got - float(np.dot(x, v))) <= n / B, (got, float(np.dot(x, v)))


# --------------------------------------------------------------------------- #
# (b) rank order == plaintext order
# --------------------------------------------------------------------------- #

def _docs_with_separated_sims(n=8, seed=7):
    cvals = [0.95, 0.80, 0.60, 0.30, -0.20, -0.70]      # well-separated cosines
    scheme = IPFEScheme.setup(n, quant_bits=2)
    rng = np.random.default_rng(seed)
    v = _unit(rng.normal(size=n))
    query = np.zeros(n)
    query[0] = 1.0                                        # dot(query, doc_i) == clue[0]
    docs = []
    for i, c in enumerate(cvals):
        d = np.zeros(n)
        d[0] = c
        if n > 1:
            d[1] = np.sqrt(max(0.0, 1.0 - c * c))          # unit norm
        Ct, _ = scheme.encrypt(list(d))
        docs.append(_Doc(f"D{i}", d, Ct, build_tree("role:Doctor"), group_bits=i))
    return scheme, v, query, docs


def test_rank_order_matches_plaintext_order():
    scheme, _v, query, docs = _docs_with_separated_sims()
    engine = RetrievalEngine(scheme, k=6)
    # query (basis e0) gives each doc exactly its separated cosine c; the
    # engine must rank by that same vector.
    got = [r.doc_id for r in engine.rank(docs, list(query), Attributes.from_roles(["doctor"]), k=6)]
    want = sorted((f"D{i}" for i in range(len(docs))),
                  key=lambda di: float(np.dot(query, docs[int(di[1:])].vec)), reverse=True)
    assert got == want, (got, want)


def test_rank_topk_scores_follow_plaintext():
    n, prec = 8, 2
    scheme = IPFEScheme.setup(n, quant_bits=prec)
    rng = np.random.default_rng(11)
    v = _unit(rng.normal(size=n))
    docs = []
    for i, c in enumerate([0.9, 0.7, 0.5, 0.3, 0.1, -0.3]):
        d = np.zeros(n)
        d[0] = c
        d[1] = np.sqrt(max(0.0, 1.0 - c * c))
        Ct, _ = scheme.encrypt(list(d))
        docs.append(_Doc(f"D{i}", d, Ct, build_tree("role:Doctor")))
    engine = RetrievalEngine(scheme, k=6)
    res = engine.rank(docs, list(v), Attributes.from_roles(["doctor"]), k=6)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True), scores
    # magnitude check on every returned score vs its plaintext cosine
    B = 10 ** prec
    for r in res:
        di = int(r.doc_id[1:])
        plain = float(np.dot(v, docs[di].vec))
        assert abs(r.score - plain) <= n / B, (r.doc_id, r.score, plain)


# --------------------------------------------------------------------------- #
# authorisation gate + k truncation + constructor guard
# --------------------------------------------------------------------------- #

def test_rank_enforces_authorization_gate():
    n, prec = 8, 2
    scheme = IPFEScheme.setup(n, quant_bits=prec)
    rng = np.random.default_rng(13)
    v = _unit(rng.normal(size=n))
    caller = Attributes.from_roles(["doctor"])
    doctor_policy = build_tree("role:Doctor")
    nurse_policy = build_tree("role:Nurse")

    d_high = np.zeros(n); d_high[0] = 0.99; d_high[1] = np.sqrt(1 - 0.99 ** 2)
    d_low = np.zeros(n); d_low[0] = 0.4; d_low[1] = np.sqrt(1 - 0.4 ** 2)
    docs = [_Doc("NURSE-DOC", d_high, scheme.encrypt(list(d_high))[0], nurse_policy),
            _Doc("DOCTOR-DOC", d_low, scheme.encrypt(list(d_low))[0], doctor_policy)]
    engine = RetrievalEngine(scheme, k=5)
    res = engine.rank(docs, list(v), caller, k=5)
    ids = [r.doc_id for r in res]
    assert "NURSE-DOC" not in ids, "highest-sim doc must be excluded by policy"
    assert ids == ["DOCTOR-DOC"]


def test_rank_k_truncation():
    n, prec = 8, 2
    scheme = IPFEScheme.setup(n, quant_bits=prec)
    rng = np.random.default_rng(17)
    v = _unit(rng.normal(size=n))
    docs = []
    for i, c in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]):
        d = np.zeros(n)
        d[0] = c
        d[1] = np.sqrt(max(0.0, 1 - c * c))
        docs.append(_Doc(f"D{i}", list(d), scheme.encrypt(list(d))[0], build_tree("role:Doctor")))
    engine = RetrievalEngine(scheme, k=5)
    res = engine.rank(docs, list(v), Attributes.from_roles(["doctor"]), k=2)
    assert len(res) == 2
    assert res[0].score >= res[1].score
    # engine default k also honours constructor
    engine2 = RetrievalEngine(scheme, k=3)
    assert len(engine2.rank(docs, list(v), Attributes.from_roles(["doctor"]))) == 3


def test_engine_requires_master_secret():
    scheme = IPFEScheme.setup(8, quant_bits=2)
    scheme.msk = None         # simulate a public/cipher-side-only instance
    try:
        RetrievalEngine(scheme)
    except ValueError as exc:
        assert "master secret" in str(exc)
        return
    raise AssertionError("cipher-side IPFE (msk=None) must be refused")


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
        "module": "retrieval",
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