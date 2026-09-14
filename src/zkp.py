# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Schnorr proof-of-possession of a string attribute (P2-A / plan §P2-A).

Non-interactive ZKP (Fiat-Shamir) that a caller knows x such that y = g^x
(mod p), where x is deterministically derived from an attribute token:

    x = int(SHA-256(normalize(attr))) mod q        credential y = g^x (mod p)

Pure stdlib (`pow`, `secrets`, `hashlib`). Reuses the SafeRAG group: any
object exposing ``.p/.q/.g`` works (``GroupParams`` or a live ``IPFEScheme``).

Scope honesty (see .opencode/zkp-technical.md §1 & §8): this proves possession
of a STRING credential. It does NOT hide the access-tree structure, does NOT
hide which attributes gate access, and does not by itself grant access. The
demo gate (P2-F) evaluates the resulting authorized ``attrs`` against the
access tree *after* ``verify_attributes``; the wire protocol carries the
authorized attrs, never the proofs (plan §2.2 rule).

Math (interactive Schnorr, three moves):
    t = g^k  ->  c in Z_q  ->  s = k + c*x mod q  ;  accept iff g^s = t*y^c
Fiat-Shamir makes c a hash of (p, q, g, attr, y, t, context) so the prover
commits to t before the challenge exists. Special soundness: two valid
transcripts for one t extract x = (s - s')*(c - c')^-1  (knowledge proof).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


# --------------------------------------------------------------------------- #
# Token normalisation (must match src/attributes.py so proofs line up with the
# access-tree gate: lowercase + strip of an explicit "key:value" string).
# --------------------------------------------------------------------------- #

def _normalize(attr: str) -> str:
    return attr.strip().lower()


def _hash_to_scalar(attr: str, q: int) -> int:
    """Deterministic secret: x = int(SHA-256(normalize(attr))) mod q."""
    digest = hashlib.sha256(_normalize(attr).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % q


# --------------------------------------------------------------------------- #
# Credential + proof data
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SchnorrCredential:
    """A public/private pair; ``x`` is the secret (derive from ``attr``).

    - ``register(group, attr)`` derives ``x`` from the token deterministically,
      so re-registering the same token gives the same credential.
    - ``group`` is duck-typed: only ``.p/.q/.g`` are read (works with both
      ``GroupParams`` and an ``IPFEScheme`` instance).
    """

    group: Any
    attr: str
    x: int
    y: int

    @classmethod
    def register(cls, group: Any, attr_value: str) -> "SchnorrCredential":
        attr = _normalize(attr_value)
        x = _hash_to_scalar(attr, group.q)
        y = pow(group.g, x, group.p)
        return cls(group, attr, x, y)

    def is_valid(self) -> bool:
        """y == g^x mod p (self-consistency check; also that 0 < x < q)."""
        return 0 <= self.x < self.group.q and self.y == pow(
            self.group.g, self.x, self.group.p)


@dataclass(frozen=True)
class SchnorrProof:
    """Non-interactive (t, c, s) transcript."""

    t: int      # commitment g^k
    c: int      # Fiat-Shamir challenge = H(p, q, g, attr, y, t, context)
    s: int      # response k + c*x (mod q)


# --------------------------------------------------------------------------- #
# Fiat-Shamir challenge (binds group + statement + commitment + context)
# --------------------------------------------------------------------------- #

def _challenge(group: Any, attr: str, y: int, t: int, context: bytes) -> int:
    h = hashlib.sha256()
    for part in (str(group.p), str(group.q), str(group.g), attr, str(y), str(t)):
        h.update(part.encode("ascii"))
    h.update(context)
    return int.from_bytes(h.digest(), "big") % group.q


# --------------------------------------------------------------------------- #
# Proving / verifying a single credential
# --------------------------------------------------------------------------- #

class Schnorr:
    """Non-interactive Schnorr proof of knowledge of the credential's x."""

    @staticmethod
    def prove(cred: SchnorrCredential, context: bytes = b"") -> SchnorrProof:
        """Produce a transcript that verifies against ``cred`` (needs x)."""
        k = secrets.randbelow(cred.group.q)
        t = pow(cred.group.g, k, cred.group.p)
        c = _challenge(cred.group, cred.attr, cred.y, t, context)
        s = (k + c * cred.x) % cred.group.q
        return SchnorrProof(t, c, s)

    @staticmethod
    def verify(group: Any, attr_value: str, proof: SchnorrProof,
               context: bytes = b"") -> bool:
        """Check ``proof`` against the PUBLIC statement ``attr_value``.

        The credential y is *recomputed* from the token (not taken from the
        prover), so a prover cannot submit a y whose log they know for a token
        they don't hold. The embedded challenge must equal H(transcript), so a
        proof cannot be replayed under a different statement/group/t.
        """
        attr = _normalize(attr_value)
        y = pow(group.g, _hash_to_scalar(attr, group.q), group.p)
        if proof.c != _challenge(group, attr, y, proof.t, context):
            return False
        lhs = pow(group.g, proof.s, group.p)
        rhs = (proof.t * pow(y, proof.c, group.p)) % group.p
        return lhs == rhs


# --------------------------------------------------------------------------- #
# Attribute-set glue (plan §P2-A: caller proves each token it claims)
# --------------------------------------------------------------------------- #

def prove_attributes(attrs: Iterable[str], group: Any,
                     context: bytes = b"") -> dict[str, SchnorrProof]:
    """Return {normalized token: SchnorrProof} for every token.

    The proof is minted against the credential ``y = g^{hash(token)}``; ``y``
    itself is deliberately NOT returned or transmitted. Verification recomputes
    ``y`` from the public token, so carrying it would disclose an extra,
    cryptographically-inert value for no security gain (see ``Schnorr.verify``).
    """
    out: dict[str, SchnorrProof] = {}
    for a in attrs:
        cred = SchnorrCredential.register(group, a)
        out[cred.attr] = Schnorr.prove(cred, context)
    return out


def verify_attributes(proofs: Mapping[str, SchnorrProof],
                      required: Iterable[str], group: Any,
                      context: bytes = b"") -> bool:
    """Accept iff EVERY required token has a valid Schnorr proof.

    The protocol never trusts a submitted credential ``y``: ``Schnorr.verify``
    recomputes ``y = g^{hash(token)}`` from the public token, so a proof minted
    under a different token/group/t gets a mismatched Fiat-Shamir challenge and
    is rejected. A missing proof for any required token rejects the whole set
    (no partial acceptance); empty ``required`` is vacuously satisfied.
    """
    for tok in required:
        tok = _normalize(tok)
        proof = proofs.get(tok)
        if proof is None:
            return False
        if not Schnorr.verify(group, tok, proof, context):
            return False
    return True