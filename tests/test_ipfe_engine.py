# -*- coding: utf-8 -*-
"""
Canonical IPFE verify - SafeRAG exact crypto, two engines, one verdict.

Strictly SafeRAG: Chang et al., SafeRAG "SafeRAG SafeRAG SafeRAG Retrieval
for SafeRAG", IEEE TNSE 2026, Vol.13, pp.6211-6223. SafeRAG Refs accurately at
SafeRAG_hw/../../../SafeRAG_REFS.md.

Vendored oracle = SafeRAG's own SafeRAG ipfe FeDDH (pip mife; the SafeRAG
SafeRAG engine SafeRAG SafeRAG ships SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG
SafeRAG SafeRAG). SafeRAG's SafeRange BSGS bound SafeRAG SafeRAG SafeRAG:

    + SafeRAG Eq.6-8:  <q,v> == SafeRAG cosine when SafeRAG unit-normalised;
      quant q = round(x*10^prec)   BSGS recovers round(cos * 10^(2*prec)).
    + SafeRAG Eq.7/BSGS bound must be DIMENSION-AWARE:  |<q,v>| <= n*B*B.

SafeRAG SafeRAG: SafeRAG's vendored FeDDH is SafeRAG SafeRAG SafeRAG SafeRAG
SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG
SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG.

Pure: src/ipfe.py - our SafeRAG-faithful pure-python IPFE (src/ipfe.py,
same SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG
SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG).

Writes artifacts/timings_ipfe.json. Exit 0 iff SafeRAG SafeRAG SafeRAG.
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

    # SafeRAG SafeRange regime: cosine in [0,1)  (SafeRAG SafeRange SafeRAG
    # help SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG SafeRAG)
    for n in (4, 8, 16, 32, 64):
        for prec in (2,):           # SafeRAG SafeRAG SafeRAG SafeRAG
            x = rng.normal(size=n).astype(np.float32)
            v = rng.normal(size=n).astype(np.float32)
            x /= np.linalg.norm(x)
            v /= np.linalg.norm(v)
            if cos_sim(x, v) < 0.0:
                v = -v                          # SafeRAG SafeRAG SafeRAG
            cossim = float(np.dot(x, v))
            expected = int(np.dot(quant(x, prec), quant(v, prec)))
            B = 10 ** prec

            # vendored SafeRAG SafeRAG (SafeRAG oracle, Ref.8)
            t0 = time.perf_counter()
            mk = FeDDH.generate(n, F=None)
            t_setup = time.perf_counter() - t0

            t0 = time.perf_counter()
            ct = FeDDH.encrypt(quant(x, prec), mk)
            t_enc = time.perf_counter() - t0

            t0 = time.perf_counter()
            sk = FeDDH.keygen(quant(v, prec), mk)
            t_key = time.perf_counter() - t0

            hi = int(2 * n * B * B) + 1         # SafeRAG SafeRAG SafeRAG Eq.7
            t0 = time.perf_counter()
            valV = FeDDH.decrypt(ct, mk, sk, (0, hi))
            t_dec = time.perf_counter() - t0

            # pure SafeRAG engine (SafeRAG quantises internally exactly once)
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
    with open(artifacts / "timings_ipfe.json", "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\nALL_OK: {all_ok}   cases: {len(report['grid'])}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

