# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Shamir (t,n)-threshold secret sharing over GF(p) (Phase-2 P2-C / DD-4).

Defence against collusion / single-point-of-failure (v2 §3.3, milestone (G)):
the IPFE master secret ``msk`` must NEVER exist whole on one device; it lives
encrypted under ONE symmetric key ``K`` (the DD-4 envelope), and only ``K`` is
Shamir-split so ``t``-of-``n`` share holders can recover ``msk`` transiently.

FIELD INVARIANTS (documented per P2-C review-me):
  * ``msk`` is a vector; every element is sampled as ``secrets.randbelow(gp.q)``
    (ipfe.py:222), so ``0 <= msk_i < q`` and q | p-1 gives ``q < p`` — the msk
    elements are group-field scalars BELOW the IPFE group modulus ``p``. They
    are never Shamir-shared directly.
  * ``K`` is 256-bit (32 bytes) but the group modulus ``p`` is only 128-bit
    (group_bits=128), so ``K`` does NOT fit GF(p). The DD-4 resolution is a
    dedicated Shamir field: a **257-bit prime** (getPrime(257), in [2^256,
    2^257)) so every 256-bit ``K`` is an in-field secret. Used ONLY for the
    envelope key, never for the msk scalars.
  * Everything is stdlib big-int arithmetic (O(t^2) Lagrange interpolation);
    ``pow(x, -1, p)`` is the modular inverse. No floating point anywhere.

Scope honesty: Shamir gives INFORMATION-THEORETIC secrecy of the shared value
(t-1 shares carry zero info about the secret inside GF(p)) — proven in the
exhaustive small-field test, not hand-waved. It does NOT forgive a weak ``K``
(envelope strength = Fernet(K) + physical custody of the shares); an attacker who
collects t shares legitimately recovers everything, by design.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence, Tuple

from cryptography.fernet import Fernet, InvalidToken

# The envelope key's Shamir field (DD-4). A 257-bit prime is in [2^256, 2^257),
# so every 256-bit K is < prime and can be stored/shared as a plain field scalar.
SHARE_FIELD_BITS = 257
_K_BYTES = 32


def new_field_prime(bits: int = SHARE_FIELD_BITS) -> int:
    """A fresh prime for the K-Shamir field (PyCryptodome, already pinned)."""
    from Crypto.Util.number import getPrime
    return getPrime(bits)


def _modinv(a: int, prime: int) -> int:
    """Modular inverse via pow (stdlib; extension-mod for negative handled by %)."""
    return pow(a % prime, -1, prime)


# --------------------------------------------------------------------------- #
# Core Shamir split / reconstruct
# --------------------------------------------------------------------------- #

def split(secret: int, threshold: int, n: int, prime: int) -> List[Tuple[int, int]]:
    """Return n shares ``(x_i, y_i)`` of ``secret`` under a (t,n)-threshold.

    ``f(x) = secret + a1*x + ... + a_{t-1}*x^{t-1} (mod prime)`` with random
    ``a_k``; share ``i = (i, f(i))`` for ``i in 1..n``. Any t shares recreate
    ``secret = f(0)`` by Lagrange interpolation; t-1 give zero information.
    """
    if not 1 <= threshold <= n:
        raise ValueError(f"need 1 <= threshold <= n, got t={threshold}, n={n}")
    if not 0 <= secret < prime:
        raise ValueError(
            f"secret must be an in-field scalar 0 <= secret < prime "
            f"({secret} vs prime {prime.bit_length()} bits)")
    coeffs = [secret % prime] + [secrets.randbelow(prime)
                                 for _ in range(threshold - 1)]
    shares = []
    for i in range(1, n + 1):
        y = 0
        power = 1
        for c in coeffs:
            y = (y + c * power) % prime
            power = (power * i) % prime
        shares.append((i, y))
    return shares


def reconstruct(shares: Sequence[Tuple[int, int]], prime: int) -> int:
    """Lagrange-interpolate ``f(0)`` from any ``k >= threshold`` shares (O(k^2))."""
    if not shares:
        raise ValueError("reconstruct() needs at least one share")
    xs = [int(x) for x, _ in shares]
    ys = [int(y) for _, y in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("share x-coordinates must be distinct")
    k = len(xs)
    acc = 0
    for i in range(k):
        num, den = 1, 1
        xi, yi = xs[i], ys[i]
        for j in range(k):
            if i == j:
                continue
            num = (num * (-xs[j])) % prime          # prod_{j!=i} (0 - x_j)
            den = (den * (xi - xs[j])) % prime      # prod_{j!=i} (x_i - x_j)
        acc = (acc + yi * num % prime * _modinv(den, prime)) % prime
    return acc


# --------------------------------------------------------------------------- #
# DD-4 envelope: SecretManager shares ONLY K; msk rides Fernet-encrypted
# --------------------------------------------------------------------------- #

class SecretManager:
    """Envelope holder: ``threshold``-of-``shares`` holders unlock ``K``.

    ``K`` (the DD-4 envelope key) is recovered transiently; if a Fernet
    ``payload`` (the serialised ``msk`` blob) was attached, ``reconstruct_secret``
    decrypts it under ``K`` and yields the live ``msk`` list. On exit the key
    buffer and (where possible) the msk are zeroed before references drop.

    The payload rides ALONGSIDE the shares and is useless without ``K`` — an
    attacker holding every payload copy but fewer than ``threshold`` shares has
    zero information about ``msk``.
    """

    def __init__(self, threshold: int, shares: Sequence[Tuple[int, int]],
                 prime: Optional[int] = None, payload: Optional[bytes] = None):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self.shares = list(shares)
        self.prime = prime if prime is not None else new_field_prime()
        self.payload = payload

    def has_enough(self) -> bool:
        return len(self.shares) >= self.threshold

    @contextmanager
    def reconstruct_secret(self) -> Iterator[object]:
        """Context manager: yields ``K`` (bytes) or, with a payload, ``msk``.

        Zeroes the ``K`` buffer on exit; clears the yielded list (msk) if it is
        a plain Python list; zero-fills a numpy array if numpy provided one.
        """
        if not self.has_enough():
            raise ValueError(
                f"need >= {self.threshold} shares to reconstruct, have "
                f"{len(self.shares)}")
        k_int = reconstruct(self.shares, self.prime)
        if k_int >= (1 << (_K_BYTES * 8)):
            raise ValueError(
                "reconstructed value is not a valid 256-bit envelope key "
                "(foreign/wrong shares?) — payload not decryptable")
        k_buf = bytearray(k_int.to_bytes(_K_BYTES, "big"))
        msk_ref = None
        try:
            if self.payload is None:
                yield bytes(k_buf)
            else:
                fernet = Fernet(base64.urlsafe_b64encode(bytes(k_buf)))
                try:
                    data = json.loads(fernet.decrypt(self.payload).decode("utf-8"))
                except InvalidToken as exc:
                    raise ValueError(
                        "payload not decryptable under reconstructed K "
                        "(wrong/too-few shares?)") from exc
                if not isinstance(data.get("msk"), list):
                    raise ValueError("payload carries no 'msk' list")
                msk = [int(x) for x in data["msk"]]
                msk_ref = msk
                yield msk
        finally:
            for i in range(len(k_buf)):           # scrub K material
                k_buf[i] = 0
            if msk_ref is not None:
                if hasattr(msk_ref, "fill"):      # numpy array
                    msk_ref.fill(0)
                else:                              # plain python list
                    msk_ref.clear()
            del k_buf

    # -- envelope helpers (test/DEMO + P2-F ingest) --------------------- #
    @staticmethod
    def build_envelope(msk: Sequence[int],
                       threshold: int, n: int, prime: Optional[int] = None) -> "SecretManager":
        """Make a new envelope: random K -> Fernet-encrypt msk blob -> split K.

        Returns a manager holding ALL n shares (a demo/authority setup); tests /
        the real deployment hand the n shares out to distinct parties.
        """
        prime = prime or new_field_prime()
        k = os.urandom(_K_BYTES)
        fernet = Fernet(base64.urlsafe_b64encode(k))
        payload = fernet.encrypt(json.dumps({"msk": [int(x) for x in msk],
                                             "k_bytes": _K_BYTES}).encode("utf-8"))
        k_int = int.from_bytes(k, "big")
        shares = split(k_int, threshold, n, prime)
        return SecretManager(threshold, shares, prime=prime, payload=payload)