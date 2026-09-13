# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Grounding verification (SafeRAG Stage-3/4 seam, Improved contribution target).

Guarantees "every claim traces to an authorised retrieved chunk". Phase-1
ships a scoring stub + the exact contract so Phase 2 can drop in a full
verifier; also exposes `verify_grounded` which the CLI wires to diagnostics.
"""

from __future__ import annotations

import re

from dataclasses import dataclass


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
