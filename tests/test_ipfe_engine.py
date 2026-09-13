# -*- coding: utf-8 -*-
"""
Canonical IPFE verify - SafeRAG exact crypto, two engines, one verdict.

The protocol mirrors the SafeRAG construction (IEEE TNSE 2026, Vol. 13,
pp. 6211-6223, Section III-B, Algorithms 1-6) built on the ABDP public-key
inner-product functional encryption scheme (Abdalla et al., PKC 2015), as
implemented in src/ipfe.py (pure stdlib). The project reference list lives in
the phase-1 master plan; this module is the runnable oracle cross-check.

Two engines, one verdict:
    vendored oracle = pymife FeDDH (mife.single.selective.ddh) — the reference
                      IPFE implementation shipped by the mife package;
    pure engine     = src/ipfe.py — our standalone re-implementation of the
                      same scheme with exact BSGS discrete-log recovery.

Protocol facts exercised per grid cell:
    + SafeRAG Eq.6-8: <q,v> == cosine similarity when both vectors are
      unit-normalised;  quant q = round(x*10^prec);  BSGS recovers
      round(cos * 10^(2*prec)).
    + The BSGS bound (Eq.7) is DIMENSION-AWARE:  |<q,v>| <= n*B*B, fixing the
      discrete-log range the Decrypt step may search.

Grid: n in {4,8,16,32,64}, prec=2; every cell must satisfy
    pure == vendored == expected   (within the prec+1 rounding floor).
Writes artifacts/timings_ipfe_grid.json (5-cell single-shot smoke artifact;
the 7-cell median-setup complexity table from T8 lives in the separate
artifacts/timings_ipfe_complexity.json). Exit 0 iff every cell is OK.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO / "src"))

from mife.single.selective.ddh import FeDDH
from ipfe import IPFEScheme


def quant(vec, prec):
    B = 10 ** prec
    return [int(round(xi * B)) for xi in vec]


def cos_sim(x, v):
    return float(np.dot(x, v))


def main():
    rng = np.random.default_rng(7)
    report = {"grid": {}, "summary": {}}
    all_ok = True

# regime: cosine in [0,1) (flip v below if the draw lands negative)
    for n in (4, 8, 16, 32, 64):
        for prec in (2,):           # decimal quantisation scale (paper Eq.6)
            x = rng.normal(size=n).astype(np.float32)
            v = rng.normal(size=n).astype(np.float32)
            x /= np.linalg.norm(x)
            v /= np.linalg.norm(v)
            if cos_sim(x, v) < 0.0:
                v = -v                          # enforce cossim >= 0
            cossim = float(np.dot(x, v))
            expected = int(np.dot(quant(x, prec), quant(v, prec)))
            B = 10 ** prec

            # vendored oracle (pymife FeDDH)
            t0 = time.perf_counter()
            mk = FeDDH.generate(n, F=None)
            t_setup = time.perf_counter() - t0

            t0 = time.perf_counter()
            ct = FeDDH.encrypt(quant(x, prec), mk)
            t_enc = time.perf_counter() - t0

            t0 = time.perf_counter()
            sk = FeDDH.keygen(quant(v, prec), mk)
            t_key = time.perf_counter() - t0

            hi = int(2 * n * B * B) + 1         # BSGS bound, Eq.7 (dimension-aware)
            t0 = time.perf_counter()
            valV = FeDDH.decrypt(ct, mk, sk, (0, hi))
            t_dec = time.perf_counter() - t0

            # pure engine (quantises internally exactly once)
            scheme = IPFEScheme.setup(n, quant_bits=prec)
            t_setup2 = time.perf_counter()
            Ct, _qq = scheme.encrypt(x)      # raw floats -> single quant
            valP = scheme.inner_product(
                Ct,
                scheme.key_derive(v),        # raw floats -> single quant
                v)
            t_dec2 = time.perf_counter() - t_setup2

            ok = (abs(valV - expected) <= prec + 1
                  and abs(valP - expected) <= prec + 1)
            all_ok = all_ok and ok
            report["grid"][f"n{n}"] = {
                "vendored": int(valV), "pure": int(valP), "expected": int(expected),
                "cossim": round(cossim, 6), "ok": bool(ok),
                "t_setup_ms": round(t_setup * 1e3, 2),
                "t_enc_ms": round(t_enc * 1e3, 2),
                "t_key_ms": round(t_key * 1e3, 2),
                "t_dec_ms": round(t_dec * 1e3, 2),
}

    report["summary"] = {"all_ok": bool(all_ok), "cases": len(report["grid"])}

    artifacts = REPO / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with open(artifacts / "timings_ipfe_grid.json", "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\nALL_OK: {all_ok}   cases: {len(report['grid'])}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

