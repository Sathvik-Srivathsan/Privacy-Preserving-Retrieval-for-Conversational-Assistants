# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Grounding verification + redact/re-answer seam (SafeRAG Stage-3/4).

Phase-1 ships ``verify_grounded`` (the span producer) + the exact contract so
Phase-2 can drop in a full verifier. P2-F adds the REDACT side (§2.5 / DD-9):
``redact_spans`` (the pure, length-preserving span rewriter), ``resolve_spans``
(offset recomputation against the REAL response — fixes the Phase-1 known
limitation on drift), and ``RedactSeam`` (the response-path wiring that enforces
the five acceptance guards: once-per-message, no vector-path feedback,
original-offset rule, idempotency/order/composition, span cap + audit log).
"""

from __future__ import annotations

import json
import os
import re
import time

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Redact seam constants (plan §2.5 guard 5)
# --------------------------------------------------------------------------- #

REDACT_MARK = "\u2588"            # U+2588 block, per-char length-preserving mask
MAX_REDACT_SPANS = 64             # one pass masks at most this many spans
MAX_REDACT_CHARS = 1024           # ... and at most this many chars


@dataclass
class GroundingReport:
    fraction_grounded: float = 0.0
    ungrounded_spans: list[tuple[int, int, str]] = None  # (start,end,text)

    def __post_init__(self):
        if self.ungrounded_spans is None:
            self.ungrounded_spans = []


# Content words only may anchor a claim. Common English stopwords are
# excluded so two sentences do NOT register as "grounded" by sharing an
# article/preposition by chance - otherwise a hallucinated claim could fake
# its way in with stopword-only overlap ("The dog is a cat." vs
# "The stock market is a factor." would both strip to content and differ).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "it", "its", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "not",
    "no", "i", "you", "he", "she", "we", "they", "do", "does", "did",
    "will", "would", "can", "could", "should", "have", "has", "had",
    "there", "then", "than", "so", "about", "into", "over", "under",
    "after", "before", "between", "up", "out",
})


def verify_grounded(response: str, chunks: list[str]) -> GroundingReport:
    """
    Phase-2 contract: every sentence in `response` must have a lexical
    content-word overlap anchor in at least one authorised `chunk`. Full
    n-gram trace is deferred; this returns a based-on-overlap estimate.

    Stopwords are stripped before overlap counting, so "grounded" requires
    a shared content token, not a chance-shared article/preposition.

    KNOWN LIMITATION (span offsets): the ungrounded spans advance ``pos`` by
    ``len(sentence) + 1``, assuming exactly one space between sentences.
    Irregular whitespace in real LLM output (double spaces, newlines) will
    drift the reported ``(start, end)`` offsets from the true index into
    ``response``. Harmless for the fraction score, but MUST be recomputed
    against the real response before any redaction ("flag/redact/re-answer")
    uses them.
    """
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    total, grounded = 0, 0
    ungrounded = []
    pos = 0
    for s in sentences:
        if not s.strip():
            continue
        total += 1
        toks = set(re.findall(r"[a-z0-9']+", s.lower())) - _STOPWORDS
        overlap = False
        for chunk in chunks:
            ctoks = set(re.findall(r"[a-z0-9']+", chunk.lower())) - _STOPWORDS
            if len(toks & ctoks) >= 1:
                overlap = True
                break
        if overlap:
            grounded += 1
        else:
            ungrounded.append((pos, pos + len(s), s))
        pos += len(s) + 1
    frac = (grounded / total) if total else 1.0
    return GroundingReport(frac, ungrounded)


# --------------------------------------------------------------------------- #
# P2-F redact/re-answer seam (plan §2.5, DD-9)
# --------------------------------------------------------------------------- #


def redact_spans(text: str, spans: tuple[tuple[int, int], ...]) -> str:
    """
    Length-preserving redaction: replace every character inside each span with
    the U+2588 block (``REDACT_MARK``), keeping the string the exact same length
    (so punctuation/sentence structure survive for the re-answer pass).

    LOCKED CONTRACT (P2-F §2.5):
      * ``spans`` is a (possibly empty) iterable of ``(start, end)`` pairs.
      * ``start``/``end`` are indexes into **this pass's input** ``text``,
        ``0 <= start <= end <= len(text)``.
      * spans are sorted by construction (any order accepted), and must be
        pairwise non-overlapping or ``ValueError`` is raised.
      * pure + deterministic: identical inputs always give identical output;
        anything outside the spans is returned byte-for-byte (guard 4).
    """
    cleaned = []
    for sp in spans:
        if not (isinstance(sp, (tuple, list)) and len(sp) == 2):
            raise ValueError(f"span must be a (start, end) pair: {sp!r}")
        start, end = sp
        if not (isinstance(start, int) and isinstance(end, int)):
            raise ValueError(f"span endpoints must be ints: {sp!r}")
        if start < 0 or end < start or end > len(text):
            raise ValueError(f"span {sp!r} out of range for text length {len(text)}")
        cleaned.append((start, end))
    cleaned.sort()                       # order-invariant by construction
    prev_end = 0
    for start, end in cleaned:
        if start < prev_end:
            raise ValueError(f"overlapping spans: {(start, end)} after end {prev_end}")
        prev_end = end
    out = list(text)
    for start, end in cleaned:
        for i in range(start, end):
            out[i] = REDACT_MARK
    return "".join(out)


def resolve_spans(response: str, report_spans) -> list[tuple[int, int]]:
    """
    Recompute ``(start, end)`` offsets against the REAL response (guard 3).

    ``verify_grounded`` advances ``pos`` by ``len(sentence) + 1`` and strips the
    response first, so double spaces/newlines drift the reported offsets (the
    Phase-1 KNOWN LIMITATION above). Before any redaction the span TEXT is
    re-located in the true response by a whitespace-robust token-window scan,
    first match at/after the previous span's end. Returns a list of exact
    ``[start, end)`` offsets valid for THIS response string.

    Raises ``ValueError`` if any span text is not resolvable in the response
    (a redaction on unresolvable offsets would mask the wrong content).
    """
    tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", response)]
    resolved = []
    cursor = 0
    for (_rep_start, _rep_end, text) in report_spans:
        needle = text.split()
        if not needle:
            continue
        hit = None
        for i in range(len(tokens) - len(needle) + 1):
            if tokens[i][0] < cursor:
                continue
            window = [t[2] for t in tokens[i:i + len(needle)]]
            if window == needle:
                hit = (tokens[i][0], tokens[i + len(needle) - 1][1])
                break
        if hit is None:
            raise ValueError(f"ungrounded span not resolvable in response: {text!r}")
        resolved.append((hit[0], hit[1]))
        cursor = hit[1]
    return resolved


class RedactSeam:
    """
    Response-path wiring for the flag/redact/re-answer pass (P2-F §2.5).

    Holds the two guarded states the pure ``redact_spans`` cannot:
      * guard 1  — once-per-message: the second ``redact`` for the same
                   ``message_id`` is a no-op (returns the text unchanged) and
                   writes NO second audit entry.
      * guard 5  — caps: a pass whose ``len(spans) > max_spans`` or whose
                   masked char count exceeds ``max_chars`` is refused ATOMICALLY
                   (output == input, nothing partially masked) and the refusal
                   is logged; every executed/refused pass appends exactly one
                   audit line ``{message_id, ts, n_spans, n_chars, spans}``.

    The seam holds reference to no store (guard 2 lives in the wiring test /
    demo: nothing the seam emits ever re-enters the vector path).
    """

    def __init__(self, audit_path=None, max_spans=MAX_REDACT_SPANS,
                 max_chars=MAX_REDACT_CHARS):
        self.audit_path = audit_path
        self.max_spans = max_spans
        self.max_chars = max_chars
        self._done = set()

    def redact(self, message_id: str, text: str,
               spans) -> tuple[str, bool, str]:
        """
        Apply one redaction pass for `message_id`.

        Returns ``(output_text, applied, refused_reason)`` where ``applied`` is
        True only if the masked output was produced, and ``refused_reason`` is
        ``""`` (applied) or ``"already_done"`` / ``"span_cap"`` /
        ``"char_cap"`` (no-op, output == input).
        """
        if message_id in self._done:
            return text, False, "already_done"
        n_chars = sum(max(0, end - start) for start, end in spans)
        refused = ""
        if len(spans) > self.max_spans:
            refused = "span_cap"
        elif n_chars > self.max_chars:
            refused = "char_cap"
        self._audit(message_id, spans, n_chars, refused)
        if refused:
            return text, False, refused
        out = redact_spans(text, tuple(spans))
        self._done.add(message_id)
        return out, True, ""

    def _audit(self, message_id, spans, n_chars, refused):
        if self.audit_path is None:
            return
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        entry = {
            "message_id": message_id,
            "ts": time.time(),
            "n_spans": len(spans),
            "n_chars": n_chars,
            "spans": [[int(a), int(b)] for a, b in spans],
            "refused": refused or None,
        }
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
