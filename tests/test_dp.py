# -*- coding: utf-8 -*-
"""
Differential-privacy tests (Phase-1 T6) —— Laplace noise + clamp + DP budgets.

Coverage:

  - `laplace_noise` inverse-CDF: exact endpoint values (monkeypatched
    `secrets.random`) plus reproducible statistical sanity (mean ~ 0,
    var ~ 2*b^2, scale/sensitivity/eps behaviour) under a seeded PRNG.
  - `clamp` lo/hi/both/neither.
  - `FixedEpsDP`: constant budget, privatise adds bounded-mean noise.
  - `AdaptiveEpsDP`: budget within [eps_min, eps_max] for every query length,
    non-decreasing in query length, empty/None -> default.

`secrets.random` is swapped for a seeded `random.Random` so everything is
deterministic across runs (no flaky statistical tests).

Run:  python tests\test_dp.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import random
import sys
import types
import math
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

import src.dp as dp                                              # noqa: E402
from src.dp import (clamp, laplace_noise, FixedEpsDP,            # noqa: E402
                    AdaptiveEpsDP)


def _seeded_rng(seed=20260913):
    rnd = random.Random(seed)
    return rnd.random


def _patch_random(seed=20260913):
    original = dp._sysrand
    dp._sysrand = types.SimpleNamespace(random=_seeded_rng(seed))
    return original


def _restore(original):
    dp._sysrand = original


# -- inverse-CDF endpoints (deterministic, exact) --------------------------- #

def test_laplace_inverse_cdf_median_zero():
    orig = _patch_random()
    try:
        dp._sysrand = types.SimpleNamespace(random=lambda: 0.5)
        assert laplace_noise(1.0, 1.0) == 0.0
    finally:
        _restore(orig)


def test_laplace_inverse_cdf_left_tail():
    orig = _patch_random()
    try:
        dp._sysrand = types.SimpleNamespace(random=lambda: 1e-12)  # clipped min u
        x = laplace_noise(2.0, 1.0)                 # b = 2
        # cancellation-free form: x = b*ln(2u) with 2*u exact -> full precision
        assert abs(x - 2.0 * math.log(2e-12)) < 1e-9, x
        assert x < 0
    finally:
        _restore(orig)


def test_laplace_inverse_cdf_mid_left_exact():
    # u=0.25 -> x = b*ln(1+2(u-0.5)) = b*ln(0.5) = -b*ln2, exactly representable
    orig = _patch_random()
    try:
        dp._sysrand = types.SimpleNamespace(random=lambda: 0.25)
        assert abs(laplace_noise(2.0, 1.0) - (-2.0 * math.log(2.0))) < 1e-12
    finally:
        _restore(orig)


def test_laplace_inverse_cdf_right_tail():
    orig = _patch_random()
    try:
        u = 0.999999999
        dp._sysrand = types.SimpleNamespace(random=lambda: u)
        x = laplace_noise(1.0, 1.0)                 # b = 1
        # x = -b*ln(2(1-u)); use the same float expression so equality is exact
        assert abs(x - (-math.log(2.0 * (1.0 - u)))) < 1e-9
        assert x > 0
    finally:
        _restore(orig)


# -- statistical sanity (seeded, reproducible) ----------------------------- #

def test_laplace_mean_near_zero_and_variance():
    orig = _patch_random()
    try:
        b = 1.0
        n = 200_000
        xs = [laplace_noise(1.0, 1.0) for _ in range(n)]        # b = 1
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        assert abs(mean) < 0.02 * b, mean                       # E[X]=0
        assert abs(var - 2.0 * b * b) < 0.1 * (2.0 * b * b), var  # Var=2b^2
    finally:
        _restore(orig)


def test_laplace_median_near_zero():
    orig = _patch_random()
    try:
        n = 200_000
        xs = sorted(laplace_noise(1.0, 1.0) for _ in range(n))
        med = xs[n // 2]
        assert abs(med) < 0.02, med
    finally:
        _restore(orig)


def test_laplace_sensitivity_scales_variance():
    orig = _patch_random()
    try:
        n = 100_000
        xs_small = [laplace_noise(1.0, 1.0) for _ in range(n)]
        xs_dbl = [laplace_noise(2.0, 1.0) for _ in range(n)]   # b = 2
        mean_s = sum(xs_small) / n
        mean_d = sum(xs_dbl) / n
        var_s = sum((x - mean_s) ** 2 for x in xs_small) / n
        var_d = sum((x - mean_d) ** 2 for x in xs_dbl) / n
        ratio = var_d / var_s
        assert 3.0 < ratio < 5.0, ratio            # b doubled -> var x4
    finally:
        _restore(orig)


def test_laplace_epsilon_increases_precision():
    orig = _patch_random()
    try:
        n = 100_000
        xs_lo = [laplace_noise(1.0, 0.1) for _ in range(n)]    # b = 10
        xs_hi = [laplace_noise(1.0, 10.0) for _ in range(n)]   # b = 0.1
        var_lo = sum(x * x for x in xs_lo) / n
        var_hi = sum(x * x for x in xs_hi) / n
        assert var_lo > 1.0                       # big eps -> spread
        assert var_hi < 0.1                        # small eps -> tight
        assert var_lo / var_hi > 100.0, var_lo / var_hi
    finally:
        _restore(orig)


# -- clamp ------------------------------------------------------------------ #

def test_clamp_bounds():
    assert clamp(5.0) == 5.0
    assert clamp(-1.0, lo=0.0) == 0.0
    assert clamp(11.0, hi=10.0) == 10.0
    assert clamp(0.5, lo=0.0, hi=1.0) == 0.5
    assert clamp(-3.0, lo=0.0, hi=1.0) == 0.0
    assert clamp(3.0, lo=0.0, hi=1.0) == 1.0


# -- FixedEpsDP ------------------------------------------------------------- #

def test_fixed_dp_budget_constant():
    d = FixedEpsDP(eps=0.7)
    assert d.budget() == 0.7
    assert d.budget([1.0, 2.0]) == 0.7
    assert d.budget([]) == 0.7


def test_fixed_dp_privatise_mean_preserves_value():
    orig = _patch_random()
    try:
        d = FixedEpsDP(eps=1.0)
        n = 100_000
        vals = [d.privatise(10.0, sensitivity=1.0) for _ in range(n)]
        mean = sum(vals) / n
        assert abs(mean - 10.0) < 0.05, mean
    finally:
        _restore(orig)


# -- AdaptiveEpsDP ---------------------------------------------------------- #

def test_adaptive_budget_within_bounds_all_lengths():
    d = AdaptiveEpsDP(eps_min=0.05, eps_max=1.0)
    for length in range(0, 1024, 7):
        b = d.budget([1.0] * length)
        assert d.eps_min <= b <= d.eps_max, (length, b)


def test_adaptive_budget_non_decreasing_in_length():
    d = AdaptiveEpsDP(eps_min=0.05, eps_max=1.0)
    prev = None
    for length in range(1, 513):      # empty query returns default (not part of
        b = d.budget([1.0] * length)  # the monotone non-empty chain)
        if prev is not None:
            assert b >= prev - 1e-12, (length, prev, b)
        prev = b
    assert prev == d.eps_max                              # saturates at 512


def test_adaptive_budget_empty_query_default():
    d = AdaptiveEpsDP(eps_min=0.05, eps_max=1.0, default=0.5)
    assert d.budget(None) == 0.5
    assert d.budget([]) == 0.5


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
        "module": "dp",
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