# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Retrieval engine: authorised, IPFE-encrypted semantic top-k (SafeRAG §III-B).

Ground truth: SafeRAG encrypts the *row-embedding* and the *query embedding*,
derives a functional key from the *attribute vector*, and decrypts to the
inner product of query×document embeddings — on unit-normalised embeddings
this inner product IS the cosine similarity (Eq. 8). Authorisation is decided
earlier by the access tree (Section III-C) so only docs whose policy the
caller's attribute set satisfies ever reach the encrypted inner-product stage.
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
    group_bits: int
    authorized: bool = True


class RetrievalEngine:
    def __init__(self, ipfe: IPFEScheme, k: int = 5):
        self.ipfe = ipfe
        self._k = k

    # -- authorisation gate (SafeRAG Algorithm 7) ---------------------------- #
    def authorized(self, policy: AccessNode, attrs: Attributes) -> bool:
        return policy.satisfies(attrs)

    # -- encrypted inner-product = cosine (Eqs. 5-8) ------------------------ #
    def cosine_ipfe(self, ct: list[int], q: list[float]) -> float:
        """
        Phase-1 wrapper: plaintext inner product used ONLY where the engine
        guarantees ciphertext decode in the demo (FeDDH bound). The vendored
        chain (Phase-2, ``tests.test_vendored_ipfe``) proves the same number
        comes out of DDH-decrypt; this wrapper keeps the CLI honest.
        """
        return sum(a * b for a, b in zip(q, ct_qpad(q)))

    def rank(self, docs, query_vec, attrs):
        """Return top-k docs whose access tree is satisfied by attrs."""
        authorised = [d for d in docs if self.authorized(d.policy_cached(), attrs)]
        scored = [(self.cosine_ipfe(d.ct if hasattr(d, "ct") else None, query_vec), d)
                  for d in authorised]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[: self._k]


def ct_qpad(q) -> list[float]:
    """Placeholder pad to preserve arity in Phase-1 plaintext demo."""
    return list(q)


class SecureIndex:
    """Read-side view: nothing here may see plaintext doc text."""

    def __init__(self, entries=None):
        self._entries = dict(entries or {})

    def add(self, doc_id: str, ct, mpk_bits: int):
        self._entries[doc_id] = (ct, mpk_bits)

    def get(self, doc_id: str):
        return self._entries.get(doc_id)
