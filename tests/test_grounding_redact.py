# -*- coding: utf-8 -*-
"""
Redact/re-answer seam — P2 contract stub (phase2-checklist §2.5 / DD-9).

LOCKED 2026-09-13. These tests pin the idempotency/order/composition contract
for the NOT-YET-BUILT ``grounding.redact_spans`` so Phase-2 implements an
already-agreed definition instead of inventing one after the fact.

Contract (see plan §2.5):

    grounding.redact_spans(text: str, spans: tuple[tuple[int,int], ...]) -> str

    - spans are [start, end) offsets into the INPUT text of the current pass;
      pairwise non-overlapping.
    - LENGTH-PRESERVING per-char masking: every char i with start <= i < end
      is replaced by U+2588 ``█``; everything else is byte-identical.
      Length-preservation makes the properties below hold algebraically
      (offsets never shift, so repeated passes cannot drift or compound).
    - pure + deterministic: no state, no logging, no I/O.

That function does not exist on this rig yet, so the file SKIPs cleanly (both
standalone and under pytest). The moment an implementation lands these tests
MUST report real assertions — a permanently skipping contract is the failure
mode the plan explicitly forbids (P2-F is not DONE while this skips).

Run:  python tests\test_grounding_redact.py   (from repo root)
Exit 0 iff status is SKIP (contract not yet implemented) or ALL_OK (implemented + passing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from src import grounding  # noqa: E402

MARK = "\u2588"

IMPLEMENTED = hasattr(grounding, "redact_spans")
SKIP_REASON = ("P2 contract stub: grounding.redact_spans not yet implemented "
               "(phase2-checklist \u00a72.5 / DD-9)")


# -------------------------------------------------------------------- fixtures
# spans are length-agnostic by design (masking is per-char, length-preserving)
TEXT = ("The patient was started on rosuvastatin 5 mg nightly "
        "to lower her cardiac risk.")
A = (27, 39)   # "rosuvastatin"
B = (45, 51)   # "nightly"
C = (66, 72)   # "cardiac"
SPANS = (A, B, C)
assert MARK not in TEXT, "fixture must start unredacted"
assert len(set(range(A[0], A[1])) | set(range(B[0], B[1])) | set(range(C[0], C[1]))) \
    == (A[1] - A[0]) + (B[1] - B[0]) + (C[1] - C[0]), "spans must be disjoint"


def _assert_masked_at(out, s, e):
    assert all(ch == MARK for ch in out[s:e])


@pytest.mark.skipif(not IMPLEMENTED, reason=SKIP_REASON)
def test_p1_idempotent_in_span_set():
    once = grounding.redact_spans(TEXT, SPANS)
    P1 = grounding.redact_spans(once, SPANS)
    assert P1 == once


@pytest.mark.skipif(not IMPLEMENTED, reason=SKIP_REASON)
def test_p2_deterministic_and_order_invariant():
    a = grounding.redact_spans(TEXT, SPANS)
    b = grounding.redact_spans(TEXT, tuple(sorted(SPANS)))
    c = grounding.redact_spans(TEXT, tuple(reversed(SPANS)))
    assert a == b == c


@pytest.mark.skipif(not IMPLEMENTED, reason=SKIP_REASON)
def test_p3_disjoint_composition_equals_union():
    union_first = grounding.redact_spans(TEXT, (A, B, C))
    seq = grounding.redact_spans(grounding.redact_spans(TEXT, (A,)), (B, C))
    assert seq == union_first


@pytest.mark.skipif(not IMPLEMENTED, reason=SKIP_REASON)
def test_p4_span_integrity():
    out = grounding.redact_spans(TEXT, SPANS)
    assert len(out) == len(TEXT)
    for s, e in SPANS:
        _assert_masked_at(out, s, e)
    masked = {i for s, e in SPANS for i in range(s, e)}
    for i, ch in enumerate(out):
        if i not in masked:
            assert ch == TEXT[i]


# -------------------------------------------------------------------- runner
def main():
    report = {
        "module": "grounding_redact",
        "contract": {"text_len": len(TEXT), "spans": [list(sp) for sp in SPANS],
                     "mask": MARK},
    }
    if not IMPLEMENTED:
        report.update({"status": "SKIP", "reason": SKIP_REASON})
        print(json.dumps(report, indent=2))
        print("SKIP (contract implemented yet?) => NOT YET; clean exit 0")
        return 0
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report.update({
        "status": "PASS" if not fails else "FAIL",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
    })
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())