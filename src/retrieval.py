# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Retrieval engine: authorised, IPFE-encrypted semantic top-k (SafeRAG §III-B).

Ground truth: SafeRAG encrypts the *row-embedding* and the *query embedding*,
derives a functional key from the *query vector*, and decrypts to the inner
product of query×document embeddings — on unit-normalised embeddings this
inner product IS the cosine similarity (Eq. 8). Authorisation is decided
earlier by the access tree (Section III-C) so only docs whose policy the
caller's attribute set satisfies ever reach the encrypted inner-product stage.

T3 wiring: ``cosine_ipfe`` now runs the real encrypted path —
``IPFE.inner_product(Enc(doc), KeyDerive(query), query)`` → quantised
``<q_doc, q_query>``; ``/B^2`` with ``B = 10**quant_bits`` recovers the cosine
on unit-normalised vectors (SafeRAG Eq. 6-8, matches the vendored FeDDH oracle
in ``tests/test_ipfe_engine.py``, including negative cosines via the BSGS
inverse branch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .attributes import AccessNode, Attributes
from .ipfe import IPFEScheme


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    group_bits: int = 0
    authorized: bool = True


def _policy_of(doc) -> AccessNode | None:
    cached = getattr(doc, "policy_cached", None)
    if callable(cached):
        return cached()
    return getattr(doc, "policy", None)


class RetrievalEngine:
    def __init__(self, ipfe: IPFEScheme, k: int = 5):
        if ipfe.msk is None:
            raise ValueError(
                "RetrievalEngine needs a full IPFEScheme holding the master "
                "secret to derive the query functional key (msk is None)")
        self.ipfe = ipfe
        self._k = k

    # -- authorisation gate (SafeRAG Algorithm 7) ---------------------------- #
    def authorized(self, policy: AccessNode, attrs: Attributes) -> bool:
        return bool(policy and policy.satisfies(attrs))

    # -- encrypted inner-product = cosine (Eqs. 5-8) ------------------------ #
    def cosine_ipfe(self, ct: list[int], q: list[float]) -> float:
        """Recover cosine via the REAL encrypted IPFE path.

        ``val = IPFE.inner_product(Enc(doc_vec), KeyDerive(q), q)`` is the
        quantised ``<q_doc, q_query>``; on unit-normalised embeddings
        ``val / B**2`` (``B = 10**quant_bits``) IS the cosine similarity.
        Negative cosines come back negative (BSGS inverse branch).
        """
        sk_fe = self.ipfe.key_derive(q)
        val = self.ipfe.inner_product(ct, sk_fe, q)
        B = 10 ** self.ipfe.quant_bits
        return val / (B * B)

    def rank(self, docs, query_vec: list[float], attrs: Attributes,
             k: Optional[int] = None) -> list[RetrievedDoc]:
        """Return top-k docs whose access tree is satisfied by attrs, scored
        by the encrypted IPFE cosine (descending)."""
        k = self._k if k is None else k
        results = []
        for d in docs:
            if not self.authorized(_policy_of(d), attrs):
                continue
            results.append(RetrievedDoc(
                doc_id=str(getattr(d, "doc_id", "")),
                score=self.cosine_ipfe(d.ct, query_vec),
                group_bits=int(getattr(d, "group_bits", 0))))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]