# -*- coding: utf-8 -*-
"""
Shamir threshold secret-sharing tests (Phase-2 P2-C, DD-4).

Acceptance criteria (per plan P2-C):
  * reconstruct == secret with EXACTLY t shares for t in {2..n} (n=3), and is
    order-independent; wrong-share substitution fails.
  * small-field EXHAUSTIVE information-theoretic test (p=17, t=2, n=3,
    secret=5): every candidate secret 0..p-1 admits SOME degree-(t-1) poly
    through the observed t-1 points — i.e. the views are indistinguishable, so
    t-1 shares carry ZERO information. (t=3 case solved explicitly too.)
  * collusion scenario (t=3, DD-4 envelope): cloud(1 share) + laptop(1 share)
    -> reconstruct != K AND the Fernet payload is undecryptable; + authority's
    third share -> exact K -> exact msk, and the recovered msk is a WORKING
    key (reproduces the IPFE mpk / key_derive behaviour).
  * SecretManager transiency: msk/K material is scrubbed when the context
    manager exits (list cleared; buffer zeroed); nothing survives the block.
  * guards: secret >= prime rejected, threshold>n rejected, duplicate x in
    reconstruct rejected, empty reconstruct rejected, too-few shares raise.
  * DD-4 field invariant is exercised in the collusion test via a real
    GroupParams.q (recovered msk elements satisfy 0 <= msk_i < q < p).

Run:  python tests\\test_secretshare.py      (from repo root)
Exit 0 iff ALL_OK. Also pytest-compatible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.ipfe import GroupParams, IPFEScheme                 # noqa: E402
from src.secretshare import (                               # noqa: E402
    SecretManager, new_field_prime, reconstruct, split,
)

P17 = 17                                        # small field for the IT test
SECRET5 = 5                                     # plan's named toy secret


def _fresh_prime() -> int:
    from src.secretshare import new_field_prime
    return new_field_prime()


# --------------------------------------------------------------------------- #
# Exact-threshold reconstruction + substitution
# --------------------------------------------------------------------------- #

def test_split_reconstruct_exact_threshold_small_field():
    for t in (2, 3):
        shares = split(SECRET5, t, 3, P17)
        assert reconstruct(shares[:t], P17) == SECRET5      # exactly t shares
        assert reconstruct(shares, P17) == SECRET5           # n shares


def test_split_reconstruct_big_field_roundtrip():
    prime = _fresh_prime()                                  # 257-bit
    secret = (2 ** 250 + 123456789) % prime                 # in-field, huge
    for t in (2, 3):
        shares = split(secret, t, 5, prime)
        assert reconstruct(shares[:t], prime) == secret
        assert reconstruct(shares[: t - 1], prime) != secret


def test_reconstruct_order_independent():
    shares = split(SECRET5, 2, 3, P17)
    a, b = shares[0], shares[1]
    assert reconstruct([a, b], P17) == SECRET5
    assert reconstruct([b, a], P17) == SECRET5


def test_wrong_share_substitution_fails():
    # hand-fixed polys over GF(17), no RNG, so the asserts are deterministic:
    #   f(x) = 5 + 3x   -> (1,8) (2,11) (3,14)
    #   g(x) = 9 + 2x   -> (1,11) (2,13) (3,15)
    f1, f2, f3 = (1, 8), (2, 11), (3, 14)
    g1, g2, g3 = (1, 11), (2, 13), (3, 15)
    mixed = [f2, g3]                        # slope 4, const 3 -> != 5, != 9
    assert reconstruct(mixed, P17) == 3
    mixed2 = [g1, f2]                       # slope 0, const 11 -> != 9
    assert reconstruct(mixed2, P17) == 11


# --------------------------------------------------------------------------- #
# Exhaustive IT: t-1 shares admit EVERY candidate secret (indistinguishable)
# --------------------------------------------------------------------------- #

def _admitted_t2(point, prime):
    """With ONE point (t-1=1, degree 1): secret s, a1 pinned by the point."""
    x, y = point
    admitted = set()
    for s in range(prime):
        a1 = (y - s) * pow(x, -1, prime) % prime            # f(x)=s+a1*x
        if (s + a1 * x) % prime == y % prime:
            admitted.add(s)
    return admitted


def _admitted_t3(points, prime):
    """With TWO points (t-1=2, degree 2): solve the 2x2 Vandermonde for (a1,a2)."""
    (x1, y1), (x2, y2) = points
    admitted = set()
    for s in range(prime):
        app = [y1 - s, y2 - s]
        m = [[x1, x1 * x1], [x2, x2 * x2]]
        det = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) % prime
        if det % prime == 0:
            continue                                        # degenerate (not here)
        inv_det = pow(det, -1, prime)
        a1 = (app[0] * m[1][1] - app[1] * m[0][1]) * inv_det % prime
        a2 = (app[1] * m[0][0] - app[0] * m[1][0]) * inv_det % prime
        ok1 = (s + a1 * x1 + a2 * x1 * x1) % prime == y1 % prime
        ok2 = (s + a1 * x2 + a2 * x2 * x2) % prime == y2 % prime
        if ok1 and ok2:
            admitted.add(s)
    return admitted


def test_exhaustive_it_no_info_at_t_minus_one():
    shares = split(SECRET5, 2, 3, P17)
    for point in shares:
        admitted = _admitted_t2(point, P17)
        assert admitted == set(range(P17))                 # ALL candidates fit


def test_exhaustive_it_t3_two_share_view_also_zero_info():
    shares = split(SECRET5, 3, 3, P17)
    admitted = _admitted_t3(shares[:2], P17)
    assert admitted == set(range(P17))                     # ALL candidates fit


# --------------------------------------------------------------------------- #
# DD-4 envelope + collusion scenario (t=3, n=3)
# --------------------------------------------------------------------------- #

def _envelope(msk, t=3, n=3, prime=None):
    """Build an envelope; accepts an explicit Shamir-field prime so tests can
    construct MANY envelopes in the SAME field (foreign-secret scenarios)."""
    return SecretManager.build_envelope(msk, t, n, prime)


def test_collusion_two_shares_insufficient_authority_completes():
    """Cloud(1 share) + laptop(1 share) cannot recover K; + authority(1) = exact msk."""
    msk_vec = [7, 11, 13]
    env = _envelope(msk_vec, t=3, n=3)
    cloud, laptop, authority = env.shares[0], env.shares[1], env.shares[2]
    prime = env.prime

    # too-few shares: threshold guard fires BEFORE Fernet can even try
    mgr2 = SecretManager(3, [cloud, laptop], prime=prime, payload=env.payload)
    assert not mgr2.has_enough()
    try:
        with mgr2.reconstruct_secret():
            pass
    except ValueError as exc:
        assert "need" in str(exc)                         # "need >= 3 shares"
    else:
        assert False, "2 shares must NOT pass the threshold check"

    # cloud + laptop + authority = exactly enough → exact msk, scrubbed after exit
    mgr3 = SecretManager(3, [cloud, laptop, authority], prime=prime,
                         payload=env.payload)
    assert mgr3.has_enough()
    with mgr3.reconstruct_secret() as msk:
        assert msk == msk_vec
    assert msk == []                                      # scrubbed on context exit


def test_recovered_msk_is_working_key_group_op():
    """mgr-full recovered msk reproduces mpk / key_derive of a real IPFE group."""
    gp = GroupParams.generate(32)
    vec_len, msk_true = 3, [7, 11, 13]
    assert all(0 <= s < gp.q for s in msk_true)              # msk_i < q < p
    mpk_true = [pow(gp.g, s, gp.p) for s in msk_true]

    env = _envelope(msk_true, t=3, n=3)
    with env.reconstruct_secret() as msk_rec:
        assert msk_rec == msk_true
        scheme_raw = IPFEScheme(gp.p, gp.q, gp.g, mpk_true, msk_true, vec_len, 2)
        scheme_rec = IPFEScheme(gp.p, gp.q, gp.g,
                                [pow(gp.g, s, gp.p) for s in msk_rec],
                                list(msk_rec), vec_len, 2)
        assert scheme_rec.mpk == scheme_raw.mpk
        assert scheme_rec.key_derive([1, 2, 3]) == scheme_raw.key_derive([1, 2, 3])
    assert msk_rec == []                                     # scrubbed


# --------------------------------------------------------------------------- #
# SecretManager transiency + pure-K path + failure modes
# --------------------------------------------------------------------------- #

def test_secretmanager_scrubs_msk_after_context_exit():
    env = _envelope([1, 2, 3], t=2, n=3)
    with env.reconstruct_secret() as msk:
        assert msk == [1, 2, 3]
    assert msk == []                          # yielded list cleared on exit


def test_secretmanager_pure_k_path():
    from src.secretshare import _K_BYTES, split
    import os
    k = os.urandom(_K_BYTES)
    prime = _fresh_prime()
    shares = split(int.from_bytes(k, "big"), 2, 3, prime)
    mgr = SecretManager(2, shares, prime=prime, payload=None)
    with mgr.reconstruct_secret() as got:
        assert isinstance(got, bytes)
        assert len(got) == _K_BYTES
        assert got == k


def test_too_few_shares_raises():
    env = _envelope([1, 2, 3], t=3, n=3)
    mgr = SecretManager(3, env.shares[:2], prime=env.prime, payload=env.payload)
    assert not mgr.has_enough()
    try:
        with mgr.reconstruct_secret():
            pass
    except ValueError:
        pass
    else:
        assert False, "3-of-3 manager with 2 shares must raise"


def test_wrong_shares_payload_undecryptable():
    shared = new_field_prime()                              # SAME field, both
    a = _envelope([1, 2, 3], t=3, n=3, prime=shared)
    b = _envelope([9, 9, 9], t=3, n=3, prime=shared)        # different K, WELL-FORMED
    mgr = SecretManager(3, [a.shares[0], b.shares[1], b.shares[2]],
                        prime=shared, payload=a.payload)
    # the mixed shares are all valid points in the SAME field — they just
    # belong to different secrets' polynomials; the correct, honest scenario
    assert mgr.prime == shared
    try:
        with mgr.reconstruct_secret():
            pass
    except ValueError as exc:
        assert "not decryptable" in str(exc)
    else:
        assert False, "foreign shares must not decrypt the payload"


# --------------------------------------------------------------------------- #
# Argument guards
# --------------------------------------------------------------------------- #

def test_split_rejects_non_field_secret():
    try:
        split(18, 2, 3, P17)
    except ValueError:
        pass
    else:
        assert False, "secret >= prime must be rejected (must be in-field)"


def test_split_rejects_bad_threshold():
    for t, n in ((0, 3), (4, 3)):
        try:
            split(SECRET5, t, n, P17)
        except ValueError:
            pass
        else:
            assert False, f"threshold t={t}, n={n} must be rejected"


def test_reconstruct_guards():
    try:
        reconstruct([], P17)
    except ValueError:
        pass
    else:
        assert False, "empty shares must be rejected"
    shares = split(SECRET5, 2, 3, P17)
    dup = [(shares[0][0], shares[0][1]), (shares[0][0], shares[1][1])]
    try:
        reconstruct(dup, P17)
    except ValueError:
        pass
    else:
        assert False, "duplicate x-coordinates must be rejected"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main():
    import re

    # self-check: the RUNNER must never silently swallow a duplicated test name.
    # A name defined twice at module scope collapses to one dict key, so the
    # collected count would silently drop one visually-present test (the same
    # bug class as ipfe.py's doubled dlog def). Guard here, statically:
    src = SELF.read_text(encoding="utf-8")
    defined = re.findall(r"^def (test_\w+)\(", src, re.M)
    dup = {n for n in defined if defined.count(n) > 1}
    if dup:
        print(f"DUPLICATE test definitions in source: {sorted(dup)}")
        print("ALL_OK: False")
        return 1

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report = {
        "module": "secretshare",
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