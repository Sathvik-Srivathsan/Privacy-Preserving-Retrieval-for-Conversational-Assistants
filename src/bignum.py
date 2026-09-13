"""Pure-Python IPFE (Inner-Product Functional Encryption) engine — Phase-1 core.

Implements the SafeRAG marine: IPFE-DDH (Abdalla–Bourse–De Caro–Pointcheval,
"Simple Functional Encryption Schemes for Inner Products", PKC 2015), the same
scheme the vendored `pymife` FeDDH provides; this module is the pure-stdlib
second opinion used to cross-verify the vendored engine and to enable the
project to run in environments where pip/mife wheels are unavailable (Rs 0).

Primitive (paper Eqs. 1-15, Section III-B, Algorithms 1-6):

    Setup(1^l, l')  -> (mpk, msk)
        sample msk = s in Z_p^l' ;  mpk_i = g^{s_i}
    Encrypt(q)      -> Ct = (C0 = g^r, C_i = mpk_i^r * g^{q_i})
    KeyDerive(v)    -> sk_fe = <s, v> = sum_i s_i * v_i
    Decrypt(Ct,sk)  -> g^{<q,v>} recovered via discrete log (BSGS)

On unit-normalised embeddings, <q,v> = cosine similarity (paper Eqs. 5-8):
the encrypted inner product IS the secure semantic similarity. Dimension is
l' = #attributes and vectors are quantised Q-bit fixed-point ints so the inner
product stays in a small, dlog-recoverable range => bounded BSGS.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Number theory helpers (pure stdlib).
# --------------------------------------------------------------------------- #

def _primes_le(n):
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = b"\x00" * (((n - i * i) // i) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


_SMALL_PRIMES = _primes_le(4096)


def is_probable_prime(n, rounds=24):
    """Miller-Rabin with fixed small-prime pre-filter + random bases (secrets)."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES[:64]:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def gen_prime(bits=64):
    while True:
        n = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(n):
            return n


def safe_prime(bits=64):
    """(p, q) with p = 2q+1 and both prime (safe prime pair)."""
    while True:
        q = gen_prime(bits - 1)
        p = 2 * q + 1
        if is_probable_prime(p):
            return p, q


def primitive_root(p, q):
    """Generator g of the order-q subgroup of Z_p^* (p = 2q+1)."""
    while True:
        g = secrets.randbelow(p - 3) + 2
        if pow(g, q, p) != 1 and pow(g, 2, p) != 1:
            return g


def modinv(a, m):
    return pow(a, -1, m)


# --------------------------------------------------------------------------- #
# BSGS (baby-step giant-step) discrete log: recover bounded x from g^x = h.
# --------------------------------------------------------------------------- #

class DiscreteLogNotFound(ValueError):
    pass


@dataclass
class BSGSTable:
    g: int
    p: int
    step: int
    table: dict = field(default_factory=dict)

    @classmethod
    def baby(cls, g, p, step):
        tbl = {}
        cur = 1
        for k in range(step):
            if cur not in tbl:
                tbl[cur] = k
            cur = (cur * pow(g, step, p)) % p
        return cls(g, p, step, tbl)

    def dlog(self, h, x_range):
        """Recover x in [0, x_range] with self.g^x == h (mod p)."""
        step = max(1, int(x_range ** 0.5) + 1)
        table = BSGSTable.baby(self.g, self.p, step).table if not self.table else self.table
        inv_g = pow(self.g, step, self.p)
        inv_g = modinv(inv_g, self.p)
        gamma = h
        for j in range(x_range // step + 2):
            if gamma in table:
                x = table[gamma] + j * step
                if pow(self.g, x, self.p) == h:
                    return x
            gamma = (gamma * inv_g) % self.p
        raise DiscreteLogNotFound("dlog not found in range")


_CACHE: dict = {}


def _bsgs_table(g, p, x_range):
    key = (g, p)
    st = int(x_range ** 0.5) + 1
    t = _CACHE.get(key)
    if t is None:
        h = 1
        tbl = {}
        for i in range(st):
            if h not in tbl:
                tbl[h] = i
            h = (h * pow(g, st, p)) % p
        t = BSGSTable(g, p, st, tbl)
        _CACHE[key] = t
    return t, st


def dlog_bsgs(g, h, p, x_range, table=None, step=None):
    """BSGS dlog: x=g^x==h (mod p) with 0<=x<=x_range. Tables persist per (g,p)."""
    if x_range <= 0:
        raise DiscreteLogNotFound("empty range")
    st = step or max(1, int(x_range ** 0.5) + 1)
    tbl = table.table if table else _bsgs_table(g, p, x_range)[0].table
    inv_g = modinv(pow(g, st, p), p)
    gamma = h
    for j in range(x_range // st + 2):
        if gamma in tbl:
            x = tbl[gamma] + j * st
            if pow(g, x, p) == h:
                return x
        gamma = (gamma * inv_g) % p
    raise DiscreteLogNotFound("dlog not found in range")
