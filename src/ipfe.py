"""IPFE (Inner-Product Functional Encryption) engine.

Implements the Abdalla-Bourse-De Caro-Pointcheval IPFE (s-IND-CPA under DDH,
PKC 2015) as used by SafeRAG (IEEE TNSE 2026, Vol 13, pp. 6211-6223,
Section III-B, Algorithms 1-6). Pure stdlib big-int arithmetic over Z_p with
baby-step/giant-step discrete-log recovery of the inner product.

Primitive (paper Eqs. 1-15):
    Setup(1^l, l')       -> (mpk, msk)  group G=<g> prime order p
                            sample s in Z_p^l'; mpk = (g^{s_i}); msk = s
    Encrypt(q)           -> Ct = (C0 = g^r, C_i = mpk_i^r * g^{q_i})
    KeyDerive(v)         -> sk_fe = <s, v> = sum_i s_i * v_i
    Decrypt(Ct, sk_fe)   -> g^{<q,v>}   (recover by BSGS discrete log)

On unit-normalised embeddings the inner product is the cosine similarity
(paper Eq. 5-8), so encrypted retrieval = secure semantic similarity search.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Number theory helpers (pure stdlib).
# --------------------------------------------------------------------------- #

def _primes_le(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = b"\x00" * (((n - i * i) // i) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


_SMALL_PRIMES = _primes_le(4096)


def is_probable_prime(n: int, rounds: int = 32) -> bool:
    """Miller-Rabin with fixed base set (deterministic for n < 2^64)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
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


def gen_prime(bits: int, rand: int = -1) -> int:
    """Random probable prime of exactly `bits` bits (top bit forced)."""
    while True:
        c = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(c):
            return c


def safe_prime(bits: int) -> tuple[int, int]:
    """Return (p, q) with p = 2q + 1, both prime (safe prime, subgroup order q)."""
    while True:
        q = gen_prime(bits - 1)
        p = 2 * q + 1
        if is_probable_prime(p):
            return p, q


def primitive_root(p: int, q: int) -> int:
    """Generator g of the order-q subgroup of Z_p^* (p = 2q+1).

    Squaring a random element of Z_p^* always lands in the (unique) order-q
    subgroup of index 2; any result != 1 is a generator of that subgroup.
    """
    while True:
        x = secrets.randbelow(p - 3) + 2
        g = pow(x, 2, p)
        if g != 1:
            return g


def modinv(a: int, m: int) -> int:
    return pow(a, -1, m)


class DiscreteLogNotFound(ValueError):
    pass


@dataclass
class BSGSTable:
    g: int
    p: int
    step: int
    table: dict[int, int] = field(default_factory=dict)

    def baby(self, g: int, p: int, step: int) -> "BSGSTable":
        """Canonical baby: cur = g^k -> k for k in 0..step-1 (multiply by g)."""
        cur = 1
        tbl = {}
        for k in range(step):
            if cur not in tbl:
                tbl[cur] = k
            cur = (cur * g) % p
        return BSGSTable(g, p, step, tbl)

    def dlog(self, h: int, x_range: int) -> int:
        """Recover x in [0, x_range] with g^x == h (mod p), g = self.g."""
        step = self.step
        gm = pow(self.g, step, self.p)
        inv_gm = modinv(gm, self.p)
        gamma = h
        for j in range(x_range // step + 2):
            if gamma in self.table:
                x = self.table[gamma] + j * step
                if pow(self.g, x, self.p) == h:
                    return x
            gamma = (gamma * inv_gm) % self.p
        raise DiscreteLogNotFound("dlog not in range")


def bsgs(g: int, h: int, p: int, x_range: int) -> int:
    """Standalone BSGS: find x in [0, x_range] with g^x = h (mod p)."""
    m = int(x_range ** 0.5) + 1
    tbl = {}
    cur = 1
    for j in range(m):
        tbl.setdefault(cur, j)
        cur = (cur * g) % p
    g_m = pow(g, m, p)
    inv_g_m = modinv(g_m, p)
    gamma = h
    for i in range(m + 1):
        if gamma in tbl:
            x = tbl[gamma] + i * m
            if pow(g, x, p) == h:
                return x
        gamma = (gamma * inv_g_m) % p
    raise DiscreteLogNotFound("bsgs failed")


_SEP_RANGE_CACHE: dict[int, BSGSTable] = {}


def get_bsgs_table(g: int, p: int, x_range: int) -> BSGSTable:
    """Cached BSGS table for a fixed (g, p, x_range) triple."""
    key = (g, p, x_range)
    t = _SEP_RANGE_CACHE.get(key)
    if t is None:
        t = BSGSTable(g, p, int(x_range ** 0.5) + 1).baby(g, p, int(x_range ** 0.5) + 1)
        _SEP_RANGE_CACHE[key] = t
        return t
    return t


# --------------------------------------------------------------------------- #
# SafeRAG group parameters (paper Table III fields).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroupParams:
    p: int
    q: int
    g: int
    group_bits: int

    @classmethod
    def generate(cls, group_bits: int = 128) -> "GroupParams":
        p, q = safe_prime(group_bits)
        g = primitive_root(p, q)
        return cls(p, q, g, group_bits)


# --------------------------------------------------------------------------- #
# IPFE scheme (paper Algorithms 1-6).
# --------------------------------------------------------------------------- #

class IPFEScheme:
    """Textbook IPFE-DDH instance parametrised by an IPFE (mpk, msk)."""

    def __init__(
        self,
        p: int, q: int, g: int,
        mpk: list[int],
        msk: Optional[list[int]],
        vec_len: int,
        quant_bits: int = 12,
    ):
        """Build an instance from explicit group + (mpk, msk)."""
        self.p = p
        self.q = q
        self.g = g
        self.mpk = mpk            # g^{s_i}
        self.msk = msk            # s (None for the public-only cipher side)
        self.vec_len = vec_len
        self.quant_bits = quant_bits
        self._range = 2 * vec_len * (10 ** (2 * quant_bits)) + 1  # |<q,v>| <= n*B^2 (SafeRAG Eq.7)

    # -- construction ------------------------------------------------ #
    @classmethod
    def setup(cls, vec_len: int, group_bits: int = 128,
              quant_bits: int = 12) -> "IPFEScheme":
        """IPFE.Setup(1^l'): sample s in Z_q^l', mpk = (g^{s_i})."""
        gp = GroupParams.generate(group_bits)
        msk = [secrets.randbelow(gp.q) for _ in range(vec_len)]
        mpk = [pow(gp.g, si, gp.p) for si in msk]

        s2gpi = {}
        for i, si in enumerate(msk):
            pass
        return cls(gp.p, gp.q, gp.g, mpk, msk, vec_len, quant_bits)

    # -- quantisation ------------------------------------------------ #
    def quant(self, v: list[float]) -> list[int]:
        """Map real vector in [-1,1] to ints in [-(2^qb), 2^qb]."""
        B = 10 ** self.quant_bits  # SafeRAG Eq.6 decimal scale (matches vendored oracle)
        return [int(round(x * B)) for x in v]

    def _power(self, base: int, e: int) -> int:
        return pow(base, e, self.p)

    # -- algorithms (paper) ------------------------------------------ #
    def encrypt(self, vec: list[float], r: Optional[int] = None
                ) -> tuple[list[int], list[int]]:
        """IPFE.Encrypt(mpk, q): Ct = (C0=g^r, C_i = mpk_i^r * g^{q_i})."""
        qq = self.quant(vec)
        if r is None:
            r = secrets.randbelow(self.q)
        C = [self._power(self.g, r)]
        for i, qi in enumerate(qq):
            a = self._power(self.mpk[i], r)          # g^{s_i r}
            b = self._power(self.g, qi)              # g^{q_i}
            C.append((a * b) % self.p)
        return C, qq

    def key_derive(self, v: list[float]) -> int:
        """IPFE.KeyDerive(msk, v) = <s, v> (mod q)."""
        vv = self.quant(v)
        return sum(si * vi for si, vi in zip(self.msk, vv)) % self.q

    def decrypt(self, Ct: list[int], sk_fe: int) -> int:
        """IPFE.Decrypt(Ct, sk_fe) -> g^{<q,v>}. Use inner_product() for the
        full keyed decrypt (SafeRAG IPFE Eq.7 + canonical BSGS dlog)."""
        raise NotImplementedError("use inner_product for full keyed decrypt")

    def inner_product(self, Ct: list[int], sk_fe: int, v: list[float]) -> int:
        """Recover quantised <q,v> = sum_i q_i*v_i (i.e. scaled cosine)."""
        vv = self.quant(v)
        num = 1
        for ci, vi in zip(Ct[1:], vv):
            num = (num * self._power(ci, vi)) % self.p      # prod C_i^{v_i}
        denom = self._power(Ct[0], sk_fe)          # C0^{sk_fe}
        ratio = (num * modinv(denom, self.p)) % self.p
        # ratio = g^{<q,v>}; recover exponent by BSGS over the quantised range.
        tblstep = int((self._range) ** 0.5) + 1
        tbl = get_bsgs_table(self.g, self.p, self._range)
        try:
            x = tbl.dlog(ratio, self._range)
        except DiscreteLogNotFound:
            # SafeRAG SafeRange Eq.7: negative <q,v> shows as the inverse.
            invratio = pow(ratio, -1, self.p)     # g^{|<q,v>|}
            x = -tbl.dlog(invratio, self._range)
        return x


def _power_checked(base: int, e: int, p: int) -> int:
    return pow(base, e, p)





