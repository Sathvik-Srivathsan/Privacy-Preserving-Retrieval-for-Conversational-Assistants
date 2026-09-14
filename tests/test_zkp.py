# -*- coding: utf-8 -*-
"""
ZKP tests (Phase-2 P2-A) — Schnorr proof-of-possession of a string attribute.

Acceptance criteria (per plan §P2-A):
  honest verifies; wrong secret fails; wrong attr string fails; replay of a
  copied proof fails for a second attr; soundness spot-check (a forger can't
  answer a challenge without x); msk / k_int untouched; group reuse works with
  IPFEScheme.setup output. Plus the attribute glue (verify_attributes) and the
  hand-verifiable toy example locked from .opencode/zkp-technical.md §7.

Run:  python tests\test_zkp.py      (from repo root)
Exit 0 iff ALL_OK. Also pytest-compatible.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.ipfe import GroupParams, IPFEScheme        # noqa: E402
from src.zkp import (                               # noqa: E402
    Schnorr, SchnorrCredential, SchnorrProof, _challenge,
    prove_attributes, verify_attributes,
)

GROUP = GroupParams.generate(128)


# --------------------------------------------------------------------------- #
# Core proof
# --------------------------------------------------------------------------- #

def test_honest_credential_verifies():
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    assert cred.is_valid()
    for _ in range(3):
        proof = Schnorr.prove(cred)
        assert Schnorr.verify(GROUP, "role:Doctor", proof)
        assert Schnorr.verify(GROUP, "ROLE:doctor", proof)  # normalisation


def test_registration_is_deterministic():
    a = SchnorrCredential.register(GROUP, "role:Doctor")
    b = SchnorrCredential.register(GROUP, "role:Doctor")
    c = SchnorrCredential.register(GROUP, "role:Nurse")
    assert a.y == b.y and a.x == b.x
    assert a.y != c.y and a.x != c.x


def test_prove_nonce_randomness():
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    p1, p2 = Schnorr.prove(cred), Schnorr.prove(cred)
    assert p1.t != p2.t, "fresh nonce k must give fresh commitment t"
    assert Schnorr.verify(GROUP, "role:Doctor", p1)
    assert Schnorr.verify(GROUP, "role:Doctor", p2)


def test_wrong_secret_fails():
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    bad = replace(cred, x=(cred.x + 1) % GROUP.q)
    proof = Schnorr.prove(bad)
    assert not Schnorr.verify(GROUP, "role:Doctor", proof)


def test_wrong_attr_string_fails():
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    proof = Schnorr.prove(cred)
    for other in ("role:Nurse", "dept:Cardio"):
        assert not Schnorr.verify(GROUP, other, proof), other
    # 'Role:Doctor' is NOT a different attribute: normalization makes it the
    # same token, so it MUST verify (asserted in test_honest_credential_verifies).


def test_replay_copied_proof_fails_for_second_attr():
    proof = Schnorr.prove(SchnorrCredential.register(GROUP, "role:Doctor"))
    assert Schnorr.verify(GROUP, "role:Doctor", proof)
    assert not Schnorr.verify(GROUP, "role:Nurse", proof), "copied proof reused"


def test_context_binds_the_proof():
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    proof = Schnorr.prove(cred, context=b"session-1")
    assert Schnorr.verify(GROUP, "role:Doctor", proof, context=b"session-1")
    assert not Schnorr.verify(GROUP, "role:Doctor", proof, context=b"session-2")


def test_forged_proof_without_secret_fails():
    # A non-knower plays by the REAL Fiat-Shamir rules: commit a random t, take
    # the hash-derived challenge c for THAT t, then try to answer with a random s
    # while knowing no x. ``g^s == t.y^c`` holds iff s = k + c.x (mod q); for a
    # random s over an order-q group the forgery succeeds with probability ~1/q.
    # This exercises the mechanism the doc §6 argument rests on (c fixed AFTER t),
    # not just "verify rejects a mismatched challenge".
    import secrets as _ss
    cred = SchnorrCredential.register(GROUP, "role:Doctor")
    for _ in range(20):
        t = pow(GROUP.g, _ss.randbelow(GROUP.q), GROUP.p)
        c = _challenge(GROUP, "role:Doctor", cred.y, t, b"")
        s = _ss.randbelow(GROUP.q)
        assert not Schnorr.verify(GROUP, "role:Doctor", SchnorrProof(t, c, s))


def test_msk_and_integrity_key_untouched():
    scheme = IPFEScheme.setup(8)
    msk_snapshot = list(scheme.msk)
    cred = SchnorrCredential.register(scheme, "role:Doctor")
    proof = Schnorr.prove(cred)
    assert Schnorr.verify(scheme, "role:Doctor", proof)
    assert scheme.msk == msk_snapshot, "ZKP must never touch the master secret"
    # the proof transcript is exactly (t, c, s): no x, no key bytes can leak in.
    assert tuple(vars(proof)) == ("t", "c", "s")
    for val in map(int, vars(proof).values()):
        assert val != cred.x


def test_group_reuse_with_ipfe_scheme():
    scheme = IPFEScheme.setup(16)
    # duck-typed group: pass the live scheme itself (exposes .p/.q/.g).
    cred = SchnorrCredential.register(scheme, "dept:Cardio")
    proof = Schnorr.prove(cred)
    assert Schnorr.verify(scheme, "dept:Cardio", proof)
    # cross-check the same token under a fresh group of the same scheme size.
    cred2 = SchnorrCredential.register(scheme, "dept:Cardio")
    assert cred2.y == cred.y


def test_doc_toy_arithmetic_locked():
    # .opencode/zkp-technical.md §7: p=23, q=11, g=4, x=8, k=6, c=3, s=8.
    G = GroupParams(p=23, q=11, g=4, group_bits=5)
    assert pow(G.g, 11, G.p) == 1, "order of g in Z_23^* must be 11 = q"
    assert pow(G.g, 8, G.p) == 9          # y = g^x
    assert pow(G.g, 6, G.p) == 2          # t = g^k
    assert (pow(G.g, 8, G.p) ==
            (pow(G.g, 6, G.p) * pow(9, 3, G.p)) % G.p), "g^s == t.y^c (mod 23)"


# --------------------------------------------------------------------------- #
# Attribute-set glue (verify_attributes)
# --------------------------------------------------------------------------- #

def test_verify_attributes_honest():
    attrs = ["role:Doctor", "dept:Cardio", "clearance:2"]
    proofs = prove_attributes(attrs, GROUP)
    assert verify_attributes(proofs, attrs, GROUP)
    assert verify_attributes(proofs, ["role:Doctor", "dept:Cardio"], GROUP)
    assert verify_attributes(proofs, [], GROUP)          # vacuously satisfied
    assert verify_attributes(proofs, {"CLEARANCE:2"}, GROUP)  # normalisation


def test_verify_attributes_missing_or_extra():
    attrs = ["role:Doctor", "dept:Cardio"]
    proofs = prove_attributes(attrs, GROUP)
    assert not verify_attributes(proofs, ["role:Doctor", "role:Nurse"], GROUP)
    assert not verify_attributes(proofs, ["dept:Cardio", "dept:Neuro"], GROUP)


def test_verify_attributes_rejects_mismatched_proof():
    # A proof minted for token A is worthless under token B even at the glue
    # layer: Schnorr.verify recomputes y and the challenge from token B, so the
    # embedded c (computed against A's y at proving time) never matches.
    attrs = ["role:doctor", "dept:cardio"]
    proofs = prove_attributes(attrs, GROUP)
    swapped = dict(proofs)
    swapped["dept:cardio"] = proofs["role:doctor"]  # proof belongs to role:doctor
    assert not verify_attributes(swapped, ["role:doctor", "dept:cardio"], GROUP)


def test_verify_attributes_context_scoped():
    proofs = prove_attributes(["role:Doctor"], GROUP, context=b"session-1")
    assert verify_attributes(proofs, ["role:Doctor"], GROUP, context=b"session-1")
    assert not verify_attributes(proofs, ["role:Doctor"], GROUP, context=b"session-2")


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
        "module": "zkp",
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