# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Grounding verification (SafeRAG Stage-3/4 seam, Improved contribution target).

Guarantees "every claim traces to an authorised retrieved chunk". Phase-1
ships a scoring stub + the exact contract so Phase 2 can drop in a full
verifier; also exposes `verify_grounded` which the CLI wires to diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundingReport:
    fraction_grounded: float = 0.0
    ungrounded_spans: list[tuple[int, int, str]] = None  # (start,end,text)

    def __post_init__(self):
        if self.ungrounded_spans is None:
            self.ungrounded_spans = []


def verify_grounded(response: str, chunks: list[str]) -> GroundingReport:
    """
    Phase-2 contract: every sentence in `response` must have a lexical
    overlap anchor in at least one authorised `chunk`. Full n-gram trace
    is deferred; this returns a based-on-overlap estimate.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    total, grounded = 0, 0
    ungrounded = []
    pos = 0
    for s in sentences:
        if not s.strip():
            continue
        total += 1
        toks = set(re.findall(r"[a-z0-9']+", s.lower()))
        overlap = False
        for chunk in chunks:
            ctoks = set(re.findall(r"[a-z0-9']+", chunk.lower()))
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
