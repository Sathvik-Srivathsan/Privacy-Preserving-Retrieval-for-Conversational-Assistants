# -*- coding: utf-8 -*-
"""Redact/re-answer seam — P2-F (phase2-checklist §2.5 / DD-9).

This file LOCKED the contract as a stub (2026-09-13) while
``grounding.redact_spans`` did not yet exist. P2-F implements it, so the stub
machinery is DISSOLVED: every test below now runs real assertions and exits
SKIP-free (the plan forbids a permanently-skipping contract).

Contract (see plan §2.5):

    grounding.redact_spans(text: str, spans: tuple[tuple[int,int], ...]) -> str
    grounding.resolve_spans(response, report_spans) -> list[(int, int)]
    grounding.RedactSeam(audit_path, max_spans, max_chars)

    - spans are [start, end) offsets into the INPUT text of the current pass;
      pairwise non-overlapping (else ValueError).
    - LENGTH-PRESERVING per-char masking: every char i with start <= i < end
      is replaced by U+2588 ``█``; everything else is byte-identical.
    - pure + deterministic: no hidden state, no logging, no I/O.

Five acceptance guards are exercised here (one shared, M7):
  1. once-per-message   — the second ``RedactSeam.redact`` for the same
                          ``message_id`` is a NO-OP (guard 1).
  2. no vector-path feedback — the seam holds no store reference; after a
                          redacted re-answer the cloud store's ``list_ids()``
                          is unchanged and NO captured wire body carries a
                          redacted ``█`` byte, before or after the pass (guard 2,
                          exercised against a REAL spawned cloud server).
  3. original-offset rule — ``resolve_spans`` recomputes drifted offsets from
                          ``verify_grounded`` against the REAL response before
                          any redaction (guard 3; fixes the Phase-1 KNOWN
                          LIMITATION). Panic (ValueError) if unresolvable.
  4. idempotency / order / composition (P1-P4) — length-preservation makes
                          these hold algebraically (guard 4).
  5. caps — len(spans) > MAX_REDACT_SPANS or masked chars > MAX_REDACT_CHARS
                          is refused ATOMICALLY (output == input) and logged;
                          each executed/refused pass writes exactly ONE audit
                          entry (guard 5).

Run:  python tests\test_grounding_redact.py   (from repo root)
Exit 0 iff ALL_OK (no skips, no failures).  Also pytest-compatible.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src import grounding  # noqa: E402
from src.cloud import CloudQuery, CloudStore  # noqa: E402
from src.ipfe import IPFEScheme  # noqa: E402
from src.store import StoredChunk  # noqa: E402

from test_cloud import _spawn_server, _stop_server  # noqa: E402

MARK = "\u2588"
MARK_UTF8 = MARK.encode("utf-8")


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


def _count_audit_entries(path, message_id=None):
    lines = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    if message_id is not None:
        lines = [l for l in lines if l["message_id"] == message_id]
    return lines


# -------------------------------------------------------------------- guard 4
def test_p1_idempotent_in_span_set():
    once = grounding.redact_spans(TEXT, SPANS)
    P1 = grounding.redact_spans(once, SPANS)
    assert P1 == once


def test_p2_deterministic_and_order_invariant():
    a = grounding.redact_spans(TEXT, SPANS)
    b = grounding.redact_spans(TEXT, tuple(sorted(SPANS)))
    c = grounding.redact_spans(TEXT, tuple(reversed(SPANS)))
    assert a == b == c


def test_p3_disjoint_composition_equals_union():
    union_first = grounding.redact_spans(TEXT, (A, B, C))
    seq = grounding.redact_spans(grounding.redact_spans(TEXT, (A,)), (B, C))
    assert seq == union_first


def test_p4_span_integrity():
    out = grounding.redact_spans(TEXT, SPANS)
    assert len(out) == len(TEXT)
    for s, e in SPANS:
        _assert_masked_at(out, s, e)
    masked = {i for s, e in SPANS for i in range(s, e)}
    for i, ch in enumerate(out):
        if i not in masked:
            assert ch == TEXT[i]


def test_redact_validation_rejects_bad_input():
    for bad in [((0, 2), (1, 3)),          # overlap
                ((0, 99),),                # out of range
                ((0, 1.5),),               # non-int endpoint
                ((0, "x"),),               # non-int endpoint
                ((0,),),                   # not a pair
                (("a", "b"),)]:            # non-int pair
        try:
            grounding.redact_spans(TEXT, bad)
            raise AssertionError(f"must reject {bad!r}")
        except ValueError:
            pass
    assert grounding.redact_spans(TEXT, ()) == TEXT
    assert grounding.redact_spans(TEXT, ()) == TEXT  # empty is a legal no-op


# -------------------------------------------------------------------- guard 1
def test_g1_once_per_message_second_call_is_noop():
    with tempfile.TemporaryDirectory() as td:
        seam = grounding.RedactSeam(audit_path=os.path.join(td, "audit.jsonl"))
        out, applied, why = seam.redact("m1", TEXT, SPANS)
        assert applied and why == ""
        assert out != TEXT
        again, applied2, why2 = seam.redact("m1", TEXT, SPANS)
        assert again == TEXT, "second pass must return the text UNCHANGED"
        assert not applied2 and why2 == "already_done"
        # guard 5 audit: exactly ONE entry for this message across both calls
        entries = _count_audit_entries(os.path.join(td, "audit.jsonl"), "m1")
        assert len(entries) == 1, entries
        assert entries[0]["n_spans"] == 3 and entries[0]["refused"] is None


def test_g1_different_message_ids_may_redact():
    with tempfile.TemporaryDirectory() as td:
        seam = grounding.RedactSeam(audit_path=os.path.join(td, "audit.jsonl"))
        o1, ap1, _ = seam.redact("m1", TEXT, (A,))
        o2, ap2, _ = seam.redact("m2", TEXT, (B,))
        assert ap1 and ap2
        assert MARK in o1 and TEXT[B[0]:B[1]] in o1
        assert MARK in o2 and TEXT[A[0]:A[1]] in o2


# -------------------------------------------------------------------- guard 3
def test_g3_resolve_spans_recovers_offsets_under_irregular_whitespace():
    """verify_grounded''s per-sentence ``pos`` advances ``len(s)+1`` assuming
    ONE space at the split point; ``re.split(r"(?<=[.!?])\\s+")`` consumes the
    WHOLE run, so a double space/newline BETWEEN sentences drifts the reported
    offsets by the surplus. resolve_spans must recompute TRUE offsets into the
    real response so redaction masks exactly the hallucinated span."""
    answer = ("Heart  rhythm monitoring prevents stroke.  The fund moon landing\n"
              " was approved.")
    chunks = ["heart rhythm monitoring prevents stroke"]
    rep = grounding.verify_grounded(answer, chunks)
    assert rep.ungrounded_spans, "expected exactly one ungrounded span"
    reported_start, reported_end, span_text = rep.ungrounded_spans[0]
    true_start = answer.find(span_text)
    assert reported_start != true_start, (
        "irregular split-space must drift the reported offset",
        reported_start, true_start)
    resolved = grounding.resolve_spans(answer, rep.ungrounded_spans)
    assert resolved == [(true_start, true_start + len(span_text))], resolved
    redacted = grounding.redact_spans(answer, resolved)
    assert span_text not in redacted
    assert "Heart  rhythm monitoring prevents stroke." in redacted, \
        "grounded sentence must stay byte-identical"
    assert len(redacted) == len(answer)


def test_g3_unresolvable_span_panics():
    with tempfile.TemporaryDirectory():
        try:
            grounding.resolve_spans("heart rhythm monitoring prevents stroke",
                                    [(0, 5, "the fund moon landing")])
            raise AssertionError("unresolvable span must raise ValueError")
        except ValueError:
            pass


# -------------------------------------------------------------------- guard 5
def test_g5_span_cap_refused_atomically():
    with tempfile.TemporaryDirectory() as td:
        seam = grounding.RedactSeam(audit_path=os.path.join(td, "audit.jsonl"),
                                    max_spans=4)
        many = tuple((i, i + 1) for i in range(8))
        out, applied, why = seam.redact("m-cap", TEXT, many)
        assert out == TEXT, "cap refusal must leave output byte-identical"
        assert not applied and why == "span_cap"
        entries = _count_audit_entries(os.path.join(td, "audit.jsonl"))
        assert len(entries) == 1 and entries[0]["refused"] == "span_cap"
        assert entries[0]["n_spans"] == 8  # logged truth, not clamped


def test_g5_char_cap_refused_atomically():
    with tempfile.TemporaryDirectory() as td:
        seam = grounding.RedactSeam(audit_path=os.path.join(td, "audit.jsonl"),
                                    max_chars=10)
        big = ((0, 5), (20, 30))          # 15 masked chars > budget 10
        out, applied, why = seam.redact("m-chars", TEXT, big)
        assert out == TEXT and not applied and why == "char_cap"
        entries = _count_audit_entries(os.path.join(td, "audit.jsonl"))
        assert len(entries) == 1 and entries[0]["refused"] == "char_cap"


def test_g5_default_caps_match_plan():
    assert grounding.MAX_REDACT_SPANS == 64
    assert grounding.MAX_REDACT_CHARS == 1024


# -------------------------------------------------------------------- guard 2
def test_g2_no_vector_path_feedback_threshold_cap_roundtrip():
    """The seam holds a reference to NO store and induces NO cloud writes: after
    a redacted re-answer the spawned server's ``list_ids()`` is unchanged, no
    new PUT/DELETE reaches the wire, and NO captured body (PUT or QUERY) ever
    carries a redacted ``█`` byte — the redact path is strictly assistant-side."""
    import numpy as np

    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(32, quant_bits=2)
        rng = np.random.default_rng(3)
        docs = []
        for cid, tag in [("D0", "role:Doctor"), ("D1", "role:Nurse")]:
            v = rng.normal(size=32)
            v = list(v / float(np.linalg.norm(v)))
            docs.append((cid, tag, scheme.encrypt(v)[0]))

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            for cid, tag, ct in docs:
                cs.put(StoredChunk(cid, json.dumps(ct, separators=(",", ":")).encode("utf-8"),
                                   tag, {"dim": 32}))

            cq = CloudQuery(base, scheme)
            cq.setup_server()
            _ = cq.query([0.1] * 32, attrs=["role:Doctor"])
            before_ids = cs.list_ids()
            n_captures_at_seam = len(cs.captured)
            assert all(MARK_UTF8 not in b for b in cs.captured + cq.captured)

            # redact/re-answer pass against a hallucinated assistant answer
            answer = "The fund moon landing was approved by the board."
            seam = grounding.RedactSeam(audit_path=os.path.join(td, "audit.jsonl"))
            out, applied, why = seam.redact("wiring-demo", answer,
                                            [(0, len(answer))])
            assert applied and why == "" and MARK in out

            # guard 1: between the snapshot and this check ONLY the seam ran,
            # so an unchanged capture count proves the seam emitted no writes
            assert len(cs.captured) == n_captures_at_seam, "seam must not emit store writes"
            assert cs.list_ids() == before_ids, "redaction must not touch the store"

            # guard 2: even a fresh query AFTER the pass stays on-clean-wire
            _ = cq.query([0.1] * 32, attrs=["role:Doctor"])
            assert cs.list_ids() == before_ids
            assert all(MARK_UTF8 not in b for b in cs.captured + cq.captured), (
                "no redacted byte may ever reach the vector path")
    finally:
        _stop_server(proc, base)


# -------------------------------------------------------------------- runner
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
        "module": "grounding_redact",
        "seam": "redact_spans + resolve_spans + RedactSeam (P2-F §2.5)",
        "status": "PASS" if not fails else "FAIL",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "skipped": 0,
        "failures": [f[0] for f in fails],
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())