# -*- coding: utf-8 -*-
"""
Phase-2 P2-E baseline battery —— Bayesian dialogue verification + plaintext
retrieval + the v2 §9 comparison harness.

Coverage:

  * ``BayesianAttributeInference`` — belief converges on consistent evidence;
    stays at the prior under evasive/none replies; wrong-level evidence LOWERS
    the true level's posterior; ``rounds_until_grant`` is few and finite;
    reset/prior/bad-input guards.
  * ``plaintext_rank`` — order matches the numpy cosine oracle; the SAME
    ``build_tree(tag).satisfies(attrs)`` authorisation gate as the improved
    path; k-truncation.
  * ``hit_rate`` / ``v9_table`` — arithmetic, required rows, honest framing.
  * mechanism-cost isolation on a SHARED corpus: the encrypted
    ``RetrievalEngine.rank`` (IPFE) reproduces the plaintext top-k for the
    same authorised caller; adversarial dialogue (wrong granted attribute)
    drops the baseline hit-rate while the single-round improved path holds.
  * ``median_ms`` latency recorded median-of-N (T8 policy).

Deterministic: seeded vectors, no RNG in the belief path, no wall-clock
assertions beyond "IPFE is not faster than a numpy dot".

Run:  python tests\\test_baseline.py      (from repo root)
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

from src.attributes import Attributes, build_tree                    # noqa: E402
from src.ipfe import IPFEScheme                                      # noqa: E402
from src.retrieval import RetrievalEngine                            # noqa: E402
from src.baseline import (BayesianAttributeInference, compare,       # noqa: E402
                          hit_rate, median_ms, plaintext_rank,
                          unfiltered_topk, v9_table)

N = 32
PREC = 2           # quant_bits: >9 explodes the BSGS table (machine lock-up)


# --------------------------------------------------------------------------- #
# Shared corpus / IPFE fixtures (mechanism-cost isolation)
# --------------------------------------------------------------------------- #

class _Doc:
    """Duck-typed doc: plaintext ``vec`` AND ciphertext ``ct`` + access policy."""

    def __init__(self, doc_id, vec, ct, tag):
        self.doc_id = doc_id
        self.vec = np.asarray(vec, dtype=np.float64)
        self.ct = ct
        self.policy = build_tree(tag)
        self.tag = tag


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def _corpus():
    """D0,D1,D3 = role:Doctor (best matches first); D2 = role:Nurse."""
    rng = np.random.default_rng(20260918)
    query = _unit(rng.normal(size=N))
    weights = [0.90, 0.65, 0.25, 0.55]          # D0 > D1 > D3 > D2 alignment
    tags = ["role:Doctor", "role:Doctor", "role:Nurse", "role:Doctor"]
    ids = ["D0", "D1", "D2", "D3"]
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    docs = []
    for did, w, tag in zip(ids, weights, tags):
        noise = _unit(rng.normal(size=N))
        vec = _unit(w * query + (1.0 - w) * noise)
        ct, _ = scheme.encrypt(list(vec))
        docs.append(_Doc(did, vec, ct, tag))
    return scheme, docs, query


def _cosine_order(docs, query):
    q = _unit(query)
    return [d.doc_id for d in sorted(
        docs, key=lambda d: float(np.dot(_unit(d.vec), q)), reverse=True)]


# --------------------------------------------------------------------------- #
# Bayesian attribute inference
# --------------------------------------------------------------------------- #

def test_bayesian_converges_on_consistent_evidence():
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    for _ in range(4):
        b.update("Doctor")
    lv, p = b.granted(threshold=0.9)
    assert lv == "Doctor", (lv, p)
    assert p > 0.99, p


def test_bayesian_stays_uncertain_without_evidence():
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    prior = dict(b.belief)
    for _ in range(5):
        b.update(None)
    assert b.belief == prior, "evasive replies must not move the posterior"
    lv, p = b.granted(threshold=0.6)
    assert lv is None and abs(p - 0.5) < 1e-12, (lv, p)


def test_bayesian_wrong_evidence_lowers_access_probability():
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    before = b.belief["Doctor"]
    b.update("Nurse")
    after = b.belief["Doctor"]
    assert after < before, (before, after)
    # the wrong reply does not leave the user un-verified — it verifies them as
    # the WRONG level, so the doctor-protected resource is denied:
    granted_level, _p = b.granted(threshold=0.6)
    assert granted_level == "Nurse", (granted_level, b.belief)


def test_bayesian_rounds_until_grant_is_few():
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    rounds = b.rounds_until_grant(["Doctor", "Doctor", "Doctor"],
                                  threshold=0.6, max_rounds=20)
    assert 1 <= rounds <= 3, rounds


def test_bayesian_evasive_never_grants():
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    sentinel = b.rounds_until_grant([None] * 10, threshold=0.6, max_rounds=10)
    assert sentinel == 11, sentinel


def test_bayesian_prior_honoured_and_reset():
    b = BayesianAttributeInference(["Doctor", "Nurse"],
                                   prior={"Doctor": 3.0, "Nurse": 1.0})
    assert abs(b.belief["Doctor"] - 0.75) < 1e-12, b.belief
    b.update("Nurse")
    assert b.belief != b.prior
    b.reset()
    assert b.belief == b.prior, b.belief


def test_bayesian_rejects_bad_input():
    for bad in (["Doctor"], []):
        try:
            BayesianAttributeInference(bad)
            raise AssertionError(f"accepted levels {bad!r}")
        except ValueError:
            pass
    b = BayesianAttributeInference(["Doctor", "Nurse"])
    try:
        b.update("Engineer")
        raise AssertionError("accepted an unknown reply level")
    except ValueError:
        pass
    try:
        BayesianAttributeInference(["Doctor", "Nurse"],
                                   prior={"Engineer": 0.5})
        raise AssertionError("accepted a prior level outside levels")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Plaintext retrieval baseline
# --------------------------------------------------------------------------- #

def test_plaintext_rank_matches_ground_truth_order():
    _scheme, docs, query = _corpus()
    got = [r[0] for r in plaintext_rank(docs, list(query),
                                        Attributes(["role:Doctor"]), k=5)]
    expected = [d for d in _cosine_order(docs, query)
                if d in {"D0", "D1", "D3"}]
    assert got == expected, (got, expected)


def test_plaintext_rank_enforces_authorization_gate():
    _scheme, docs, query = _corpus()
    nurse = [r[0] for r in plaintext_rank(docs, list(query),
                                          Attributes(["role:Nurse"]), k=5)]
    assert nurse == ["D2"], nurse
    doctor = {r[0] for r in plaintext_rank(docs, list(query),
                                           Attributes(["role:Doctor"]), k=5)}
    assert doctor == {"D0", "D1", "D3"}, doctor


def test_plaintext_rank_k_truncation():
    _scheme, docs, query = _corpus()
    top1 = plaintext_rank(docs, list(query), Attributes(["role:Doctor"]), k=1)
    assert len(top1) == 1, top1


# --------------------------------------------------------------------------- #
# Harness: hit_rate / v9_table
# --------------------------------------------------------------------------- #

def test_hit_rate_arithmetic():
    assert hit_rate(["a", "b"], ["a", "b"]) == 1.0
    assert hit_rate(["a"], ["a", "b"]) == 0.5
    assert hit_rate(["x"], ["a", "b"]) == 0.0
    try:
        hit_rate([], [])
        raise AssertionError("empty true set must raise")
    except ValueError:
        pass


def test_v9_table_rows_complete():
    t = v9_table(true_topk=["a", "b"], baseline_topk=["a", "b"],
                 improved_topk=["a"], baseline_rounds=4)
    for key in ("authorized_hit_rate", "verification_latency_rounds",
                "tamper_detection_rate", "qualitative_g_secret_sharing",
                "qualitative_c_voice_privacy", "qualitative_dp_numeric_answer",
                "hardware", "python", "numpy"):
        assert key in t, key
    assert t["authorized_hit_rate"] == {"baseline": 1.0, "improved": 0.5}
    assert t["verification_latency_rounds"] == {"baseline": 4, "improved": 1}
    assert t["tamper_detection_rate"] == {"baseline_row": "0%",
                                          "improved_row": "100%"}


# --------------------------------------------------------------------------- #
# Mechanism-cost isolation: plaintext vs encrypted IPFE on the SAME corpus
# --------------------------------------------------------------------------- #

def test_encrypted_rank_matches_plaintext_for_same_caller():
    scheme, docs, query = _corpus()
    engine = RetrievalEngine(scheme, k=5)
    attrs = Attributes(["role:Doctor"])
    plain = [r[0] for r in plaintext_rank(docs, list(query), attrs, k=5)]
    enc = [r.doc_id for r in engine.rank(docs, list(query), attrs, k=5)]
    assert enc == plain, (enc, plain)


def test_compare_v9_table_clean_parity():
    scheme, docs, query = _corpus()
    engine = RetrievalEngine(scheme, k=5)
    attrs = ["role:Doctor"]
    true_topk = unfiltered_topk(docs, list(query), k=3)
    assert true_topk == ["D0", "D1", "D3"], true_topk
    table = compare(docs, list(query), attrs=attrs, true_topk=true_topk, k=3,
                    improved_ranker=lambda d, q, a, k: engine.rank(d, q, a, k=k))
    assert table["authorized_hit_rate"] == {"baseline": 1.0, "improved": 1.0}, table


def test_unfiltered_topk_ignores_authorization():
    _scheme, docs, query = _corpus()
    unqualified = unfiltered_topk(docs, list(query), k=4)
    assert unqualified == _cosine_order(docs, query), unqualified
    assert "D2" in unqualified, "unqualified target must include the nurse doc"
    assert [r[0] for r in plaintext_rank(docs, list(query),
                                         Attributes(["role:Doctor"]), k=4)] \
        == ["D0", "D1", "D3"]


def test_compare_requires_explicit_true_topk():
    scheme, docs, query = _corpus()
    engine = RetrievalEngine(scheme, k=5)
    ranker = lambda d, q, a, k: engine.rank(d, q, a, k=k)          # noqa: E731
    try:
        compare(docs, list(query), attrs=["role:Doctor"],
                true_topk=None, k=3, improved_ranker=ranker)
        raise AssertionError("compare() accepted true_topk=None")
    except ValueError:
        pass
    try:
        compare(docs, list(query), attrs=["role:Doctor"],
                k=3, improved_ranker=ranker)                        # omitted
        raise AssertionError("compare() accepted a missing true_topk")
    except TypeError:
        pass


def test_wrong_dialogue_attribute_lowers_authorized_hit_rate():
    """Baseline verification fooled by wrong replies → wrong granted attr →
    authorised hit-rate collapses; improved grants in ONE round → holds 1.0."""
    scheme, docs, query = _corpus()
    engine = RetrievalEngine(scheme, k=5)
    true_topk = unfiltered_topk(docs, list(query), k=3)

    # baseline believed the caller is a Nurse (wrong) → only D2 is authorised
    baseline_topk = [r[0] for r in plaintext_rank(
        docs, list(query), Attributes(["role:Nurse"]), k=3)]
    improved_topk = [r.doc_id for r in engine.rank(
        docs, list(query), Attributes(["role:Doctor"]), k=3)]
    assert hit_rate(baseline_topk, true_topk) < 1.0
    assert hit_rate(improved_topk, true_topk) == 1.0


def test_latency_median_of_n_records_mechanism_cost():
    scheme, docs, query = _corpus()
    engine = RetrievalEngine(scheme, k=5)
    attrs = Attributes(["role:Doctor"])

    def plain():
        plaintext_rank(docs, list(query), attrs, k=5)

    def enc():
        engine.rank(docs, list(query), attrs, k=5)

    plain()
    enc()                                   # warm-up (BSGS cache / imports)
    base_ms = median_ms(plain, n=7)
    enc_ms = median_ms(enc, n=7)
    assert base_ms > 0.0 and enc_ms > 0.0, (base_ms, enc_ms)
    assert enc_ms >= base_ms, (base_ms, enc_ms)   # encrypted path is not faster


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
        "module": "baseline",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
        "all_ok": not fails,
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   "
          f"passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())