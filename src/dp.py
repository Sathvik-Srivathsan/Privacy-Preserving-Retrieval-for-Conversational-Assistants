# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Differential privacy stage (SafeRAG §III-D: Laplace / adaptive-epsilon).

Baseline: fixed-knob (eps) Laplace noise on numeric answers (paper Table IV:
MAE~10/MSE~200 at eps=0.1; MAE~1/MSE~5 at eps=1.0). Improvement (SafeRAG-Improved,
Phase 3): per-query adaptive epsilon — a small bandit/context controller chooses
eps_t from a budget, so non-numeric utility-heavy queries pay *less*. Phase-1
ships the baseline Laplace + the controller seam.
"""

from __future__ import annotations

import math

from secrets import SystemRandom

_sysrand = SystemRandom()


def laplace_noise(sensitivity: float, eps: float) -> float:
    """Y ~ Laplace(0, sensitivity/eps) via inverse-CDF sampling.

    Two-sided, cancellation-free: u<0.5 -> x = b*ln(2u), u>=0.5 ->
    x = -b*ln(2(1-u)). Computed as ``2*u`` / ``2*(1-u)`` directly (NOT
    ``1+2*(u-0.5)``) so tail draws near u->0/1 keep full float precision —
    that is exactly the regime where DP noise magnitude matters. (The naive
    ``-b*sign*ln(1-2|u-0.5|)`` form both cancels catastrophically AND
    mirrors the negative tail into a folded, always-non-negative distribution.)
    """
    u = max(_sysrand.random(), 1e-12)
    b = sensitivity / max(eps, 1e-12)
    if u >= 0.5:
        return -b * math.log(2.0 * (1.0 - u))
    return b * math.log(2.0 * u)


def clamp(x: float, lo: float | None = None, hi: float | None = None) -> float:
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


class FixedEpsDP:
    """Baseline SafeRAG controller: the epsilon knob is fixed per deploy."""

    def __init__(self, eps: float = 1.0):
        self.eps = eps

    def budget(self, _query: list[float] | None = None) -> float:
        return self.eps

    def privatise(self, value: float, sensitivity: float = 1.0) -> float:
        return value + laplace_noise(sensitivity, self.budget())


class AdaptiveEpsDP(FixedEpsDP):
    """
    SafeRAG-Improved Phase-3: select epsilon per query from a small bandit
    (eps_min, eps_max]. Phase-1: deterministic heuristic driven by query
    length; Phase-3 swaps in the actual bandit update rule.
    """

    def __init__(self, eps_min: float = 0.05, eps_max: float = 1.0,
                 default: float = 0.5):
        super().__init__(default)
        self.eps_min, self.eps_max = eps_min, eps_max

    def budget(self, query: list[float] | None = None) -> float:
        if query is None or len(query) == 0:
            return self.eps
        # design intent (code follows this): a longer query carries more
        # information content => higher privacy budget (less noise);
        # saturates at eps_max once length >= 512.
        frac = min(1.0, len(query) / 512.0)
        return self.eps_min + (self.eps_max - self.eps_min) * frac
