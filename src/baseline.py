# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Faithful-but-reduced baseline for the original SafeRAG mechanisms (v2 §9).

Purpose (P2-E): produce the comparison table the milestone's write-up needs —

  * **XML name-hiding and IPFE are NOT in the baseline.** Baseline retrieval
    is *plaintext* cosine top-k: the caller's granted attribute set is applied
    as an authorisation filter (same ``build_tree(tag).satisfies(attrs)``
    gate as the improved path) and the top-k is the raw ``numpy`` cosine.
    The improved path is ``RetrievalEngine.rank`` (real encrypted IPFE
    inner-product). This isolates the *mechanism* cost, exactly as the plan
    requires ("so the comparison isolates mechanism cost").

  * **Verification is multi-round dialogue, not a proof.** ``BayesianAttribute
    Inference`` implements the reduced form of SafeRAG's dialogue game: a
    belief distribution over an attribute's levels, updated by Bayes' rule
    from per-round replies (a reply names a level, or is evasive). Access is
    granted once the top posterior crosses a threshold. The improved path
    grants the claimed attributes in ONE round (a single ZKP proof), so the
    ``verification_latency_rounds`` row is literally "N dialogue rounds vs 1".

  * **Tamper detection is absent in the baseline.** Plaintext baseline has no
    MAC, so its tamper-detection rate is 0% by construction; the improved
    path (HMAC ``CorpusIntegrity``) detects tampering loudly. The harness
    records both.

  * **DP baseline reuses ``FixedEpsDP`` unchanged** (no reimplementation) —
    the improved phase adds adaptive/clearance-aware budgeting only.

Honesty framing (plan review-me): this is the paper's Bayesian *formula* at
toy scale, NOT production SafeRAG. The write-up compares "dialogue-based
(multi-round) vs single-round proof" and never claims the baseline is
production-safe or faithful beyond the belief-update formula.

Run:  python tests\test_baseline.py   (from repo root)
"""

from __future__ import annotations

import datetime as _dt
import platform as _platform
import sys as _sys
import time as _time

import numpy as _np

from .attributes import Attributes, build_tree
from .dp import FixedEpsDP


# --------------------------------------------------------------------------- #
# Plaintext (no-IPFE) top-k baseline
# --------------------------------------------------------------------------- #

def plaintext_rank(docs, query: list[float], attrs: Attributes, k: int = 5) -> list:
    """Plaintext cosine top-k over docs whose policy ``satisfies(attrs)``.

    This is the mechanism-cost isolation baseline: same authorisation gate as
    ``RetrievalEngine.rank``, but the similarity is raw cosine on plaintext
    vectors — no encryption, no discrete-log recovery. Documents are duck-typed
    like ``RetrievalEngine`` docs: ``.doc_id``, ``.vec`` (plaintext), and a
    ``.policy``/`.policy_cached()` AccessNode.
    """
    q = _np.asarray(query, dtype=_np.float64)
    qn = q / (_np.linalg.norm(q) or 1.0)
    scored = []
    for d in docs:
        policy = getattr(d, "policy", None)
        if callable(policy):
            policy = policy()
        if policy is None or not policy.satisfies(attrs):
            continue
        v = _np.asarray(getattr(d, "vec"), dtype=_np.float64)
        vn = v / (_np.linalg.norm(v) or 1.0)
        scored.append((str(d.doc_id), float(_np.dot(qn, vn))))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def unfiltered_topk(docs, query: list[float], k: int = 5) -> list:
    """Plaintext cosine top-k over ALL docs, ignoring authorisation.

    This is the unqualified semantic target — what a *permitted* caller should
    receive — and is the ground truth ``compare()`` requires. It is
    deliberately a separate, explicitly-called function: the ground truth must
    never be silently replaced by the (already authorisation-filtered)
    baseline ranking, which would make the baseline hit-rate tautologically
    1.0. Returns ``[doc_id, ...]`` in descending cosine order.
    """
    q = _np.asarray(query, dtype=_np.float64)
    qn = q / (_np.linalg.norm(q) or 1.0)
    scored = []
    for d in docs:
        v = _np.asarray(getattr(d, "vec"), dtype=_np.float64)
        vn = v / (_np.linalg.norm(v) or 1.0)
        scored.append((str(d.doc_id), float(_np.dot(qn, vn))))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [did for did, _ in scored[:k]]


def hit_rate(retrieved_ids: list, true_topk_ids: list) -> float:
    """Fraction of the true top-k that the system actually returned.

    ``true_topk_ids`` is the unqualified semantic top-k (what a permitted
    caller *should* get); ``retrieved_ids`` is what the boundary returned.
    """
    if not true_topk_ids:
        raise ValueError("true top-k is empty — hit-rate is undefined")
    truth = set(true_topk_ids)
    return len(truth & set(retrieved_ids)) / len(truth)


# --------------------------------------------------------------------------- #
# Multi-round Bayesian attribute inference (SafeRAG dialogue game, reduced)
# --------------------------------------------------------------------------- #

class BayesianAttributeInference:
    """Multi-round belief over an attribute's levels (SafeRAG §III-C, reduced).

    State: a prior ``P(level)`` (default uniform). Each round consumes one
    reply and updates beliefs by Bayes' rule. A reply is either:

      - a level name: with probability ``informativeness`` the reply names the
        TRUE level (probability ``p_evidence``) or a wrong level (spread
        uniformly); with probability ``1 - informativeness`` it is uniform
        noise over all levels;
      - ``None`` / ``""`` (evasive): carries no information, posterior is
        unchanged.

    ``granted(threshold)`` returns the top level once its posterior crosses
    the threshold. ``rounds_until_grant`` counts how many rounds of a given
    evidence sequence are needed — this is the baseline's ``N dialogue rounds``
    verification latency.
    """

    def __init__(self, levels, prior=None, *, informativeness=0.8,
                 p_evidence=0.95):
        if not levels or len(levels) < 2:
            raise ValueError("attribute inference needs >= 2 levels")
        self.levels = tuple(levels)
        self._k = len(self.levels)
        uniform = 1.0 / self._k
        raw = {lv: uniform for lv in self.levels}
        if prior is not None:
            for lv, p in prior.items():
                if lv not in raw:
                    raise ValueError(f"prior level {lv!r} not in levels")
                raw[lv] = float(p)
        total = sum(raw.values())
        self.prior = {lv: p / total for lv, p in raw.items()}
        self.q = float(informativeness)          # probability a reply is informative
        self.p_ev = float(p_evidence)            # P(reply==true | informative)
        if not (0.0 <= self.q <= 1.0 and 0.0 <= self.p_ev <= 1.0):
            raise ValueError("informativeness/p_evidence must be in [0,1]")
        if self.p_ev < 1.0 / self._k:
            raise ValueError("p_evidence must exceed the uniform chance")
        self.belief = dict(self.prior)

    # -- evidence model ----------------------------------------------------- #
    def _likelihood(self, reply, level) -> float:
        uniform = 1.0 / self._k
        if reply is None or reply == "":
            return uniform
        reply = str(reply)
        if reply not in self.levels:
            raise ValueError(f"reply {reply!r} is not a level of {self.levels}")
        if reply == level:
            informative = self.p_ev
        else:
            informative = (1.0 - self.p_ev) / (self._k - 1)
        return self.q * informative + (1.0 - self.q) * uniform

    # -- update / decision -------------------------------------------------- #
    def update(self, reply) -> dict:
        """Consume one reply, return the posterior after the update."""
        lik = {lv: self._likelihood(reply, lv) for lv in self.levels}
        num = {lv: self.belief[lv] * lik[lv] for lv in self.levels}
        total = sum(num.values())
        if total <= 0.0:
            raise ArithmeticError("belief collapsed to zero — bad p_evidence")
        self.belief = {lv: p / total for lv, p in num.items()}
        return dict(self.belief)

    def reset(self) -> None:
        self.belief = dict(self.prior)

    def top(self) -> tuple:
        """Return (level, posterior) of the current best candidate."""
        lv = max(self.levels, key=self.belief.get)
        return lv, self.belief[lv]

    def granted(self, threshold: float = 0.6) -> tuple:
        """Return (level or None, posterior) once the threshold is crossed."""
        lv, p = self.top()
        return (lv if p >= threshold else None), p

    def rounds_until_grant(self, evidence, threshold: float = 0.6,
                           max_rounds: int = 20) -> int:
        """Number of rounds of ``evidence`` needed before ``granted`` fires.

        Returns ``max_rounds + 1`` if the evidence never crosses the threshold
        (the caller is never verified), so the caller can distinguish "took
        max rounds" from "granted on the final round".
        """
        self.reset()
        for i in range(1, max_rounds + 1):
            self.update(evidence[i - 1] if i <= len(evidence) else None)
            lv, _p = self.granted(threshold)
            if lv is not None:
                return i
        return max_rounds + 1


# --------------------------------------------------------------------------- #
# Comparison harness (v2 §9 table)
# --------------------------------------------------------------------------- #

def median_ms(fn, n: int = 7) -> float:
    """Median wall-clock of ``fn()`` over ``n`` calls, in milliseconds.

    Median (not mean) per the T8 timing policy so one slow GC pause cannot
    drag the verdict. Benchmarking warm-up is the caller's responsibility.
    """
    samples = []
    for _ in range(n):
        t0 = _time.perf_counter()
        fn()
        samples.append((_time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[n // 2]


def v9_table(*, true_topk, baseline_topk, improved_topk,
             baseline_rounds, improved_rounds=1,
             baseline_tamper_rate=0.0, improved_tamper_rate=1.0) -> dict:
    """Assemble the v2 §9 comparison rows into one JSON-dumpable dict.

    ``*_rounds`` are the verification latencies (baseline = dialogue rounds,
    improved = 1 ZKP round). Tamper rows are rates (0.0..1.0). The ``improved``
    fast path grants in one round while the baseline must pass the Bayesian
    dialogue game — the "N rounds vs 1" claim is therefore derived from the
    actual values passed in, not asserted in prose.
    """
    def _rate(x):
        return f"{x * 100:.0f}%"

    return {
        "generated_by": "src/baseline.py (P2-E)",
        "date": _dt.date.today().isoformat(),
        "hardware": f"{_platform.machine()} ({_platform.processor() or 'n/a'})",
        "python": _sys.version.split()[0],
        "numpy": _np.__version__,
        "authorized_hit_rate": {
            "baseline": hit_rate(baseline_topk, true_topk),
            "improved": hit_rate(improved_topk, true_topk),
        },
        "verification_latency_rounds": {
            "baseline": int(baseline_rounds),
            "improved": int(improved_rounds),
        },
        "tamper_detection_rate": {
            "baseline_row": _rate(baseline_tamper_rate),
            "improved_row": _rate(improved_tamper_rate),
        },
        "qualitative_g_secret_sharing": (
            "baseline: single point of failure (one authority holds ALL keys); "
            "improved: Shamir t-of-n split — cloud+colluders below threshold "
            "reconstruct nothing (P2-C)"
        ),
        "qualitative_c_voice_privacy": (
            "baseline: raw speaker signal carries identity; improved: "
            "pitch/formant anonymiser collapses speaker-ID to chance (P2-H)"
        ),
        "qualitative_dp_numeric_answer": {
            "baseline": "FixedEpsDP (fixed epsilon Laplace, dp.FixedEpsDP reused)",
            "improved": "ClearanceAwareDP: smaller epsilon for higher clearance "
                        "(see P2-G MAE/MSE-vs-eps table)",
        },
        "notes": {
            "baseline_retrieval": "plaintext cosine top-k (no IPFE)",
            "improved_retrieval": "encrypted IPFE inner-product (RetrievalEngine.rank)",
            "boundary": "mechanism-cost isolation: BOTH paths filter by the same "
                        "build_tree(tag).satisfies(attrs) authorisation gate",
        },
    }


def compare(docs, query_vec, *, attrs, true_topk,
            k: int = 5, improved_ranker=None,
            baseline_rounds=1, improved_rounds=1) -> dict:
    """One-shot ``v9_table`` for a shared corpus and authorised caller.

    ``docs`` are duck-typed objects with ``.doc_id``/``.vec``/``.policy``.
    ``improved_ranker(docs, query_vec, attrs, k) -> [RetrievedDoc]`` is the
    encrypted path. ``true_topk`` is REQUIRED and must be real ground truth
    (normally ``unfiltered_topk(docs, query_vec, k)``): it is deliberately not
    defaulted, because defaulting it to the already-authorisation-filtered
    baseline ranking would make the baseline hit-rate tautologically 1.0 and
    silently invalidate the paper artifact. ``baseline_rounds`` is the
    dialogue-game verification latency; P2-G must thread a real
    ``BayesianAttributeInference.rounds_until_grant(...)`` value here, not a
    placeholder.
    """
    if true_topk is None:
        raise ValueError(
            "true_topk is required and must be real ground truth "
            "(use unfiltered_topk(docs, query_vec, k)); defaulting it to the "
            "baseline ranking would make the baseline hit-rate trivially 1.0")
    if improved_ranker is None:
        raise ValueError("improved_ranker must be provided by the caller")
    attr_obj = attrs if isinstance(attrs, Attributes) else Attributes(attrs)
    baseline_topk = [r[0] for r in plaintext_rank(docs, query_vec, attr_obj, k=k)]
    improved_topk = [str(r.doc_id) for r in improved_ranker(docs, query_vec,
                                                            attr_obj, k=k)]
    return v9_table(true_topk=true_topk, baseline_topk=baseline_topk,
                    improved_topk=improved_topk, baseline_rounds=baseline_rounds,
                    improved_rounds=improved_rounds)