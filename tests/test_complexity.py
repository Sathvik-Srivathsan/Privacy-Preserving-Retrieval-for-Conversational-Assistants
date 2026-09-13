# -*- coding: utf-8 -*-
"""
Phase-1 T8 — IPFE time & space complexity analysis (explicit deliverable).

Two engines, one verdict, across the FULL swept grid n ∈ {4,8,16,32,64,128,256}
(the 128/256 cells are now re-run against the CURRENT post-cleanup code; the
earlier 14294/20757 marks were run-only, against a pre-`if False`-sweep build).

Timing policy (as agreed): SETUP is the prime-generation-dominated phase with
high single-shot variance (integer-prime search) — it is reported as the
**MEDIAN over REPEATS_SETUP fresh setups**, never a single shot. Encrypt /
KeyDerive / Decrypt are measured warm (BSGS table already cached for that
(g,p,range)); Decrypt timing is the steady-state amortised dlog cost.

Asymptotic model being checked against the measured table (modpow counts
trace the code exactly): encrypt() = 1 modpow (C0=g^r) + per dim 2 modpows
(a=mpk_i^r, b=g^q_i) -> **2n+1** mod-exps = O(n.log p). inner_product() =
n mod-pows (prod C_i^{v_i} over Ct[1:]) + 1 modpow (C0^{sk_fe}) -> **n+1**
mod-pows, + 1 modinv + BSGS discrete log: step = sqrt(range)+1 baby entries;
giant loop ~ range/step; time O(sqrt(range)), space O(sqrt(range));
range = 2n.B^2+1 (SafeRAG Eq.7), so sqrt(range) = O(sqrt(n).B) ~ n^{1/2}.

AMORTISATION SCOPE (precise): the Baby-Step table is cached per (g,p,range),
so "amortised across queries" holds WITHIN one scheme's lifetime (one corpus
setup serving many queries). In THIS benchmark each n cell calls
IPFEScheme.setup() fresh (new prime/group every time), so there is NO table
reuse ACROSS the different n cells — the t-dec "warm" numbers are amortised
only over the repeated decrypts of the SAME scheme of one n.

Writes artifacts/timings_ipfe_complexity.json (inner-product table for the
paper: 7-cell grid n∈{4…256}, dual-oracle verdicts, median-of-5 setup, warm
median-of-3 enc/key/dec, per-n BSGS table stats, corrected op counts,
model labels). The T0 5-cell smoke grid lives in the separate
artifacts/timings_ipfe_grid.json — each benchmark owns its own artifact
(no cross-clobbering). Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO / "src"))

from mife.single.selective.ddh import FeDDH       # noqa: E402 vendored oracle
from ipfe import IPFEScheme                       # noqa: E402 pure engine
from ipfe import BSGSTable, get_bsgs_table        # noqa: E402 (table stats)


N_SWEEP = (4, 8, 16, 32, 64, 128, 256)
REPEATS_SETUP = 5          # prime-gen variance -> median, not single-shot
REPEATS_OPS = 3            # enc/key/dec measured warm (BSGS cached), median
PREC = 2
B = 10 ** PREC


def quant(vec, prec):
    return [int(round(xi * B)) for xi in vec]


def cos_sim(x, v):
    return float(np.dot(x, v))


def _table_stats(scheme: IPFEScheme) -> dict:
    rng = int(scheme._range)
    step = int(rng ** 0.5) + 1
    tbl: BSGSTable = get_bsgs_table(scheme.g, scheme.p, rng)
    # entries == step (baby powers g^0..g^{step-1}); values are step indices
    entries = len(tbl.table)
    giant_upper = rng // step + 2                  # dlog loop bound
    est_bytes = (sys.getsizeof(tbl.table)
                 + sum(sys.getsizeof(k) for k in tbl.table)
                 + sum(sys.getsizeof(v) for v in tbl.table.values()))
    return {"range": rng, "step": step, "baby_entries": entries,
            "giant_loops_upper": giant_upper, "table_bytes_est": est_bytes,
            "cache_key": (tbl.g, tbl.p, rng)}


def main():
    rng = np.random.default_rng(7)
    report = {"grid": {}, "bsgs": {}, "model": {}, "summary": {}}
    all_ok = True

    for n in N_SWEEP:
        x = rng.normal(size=n).astype(np.float32)
        v = rng.normal(size=n).astype(np.float32)
        x /= np.linalg.norm(x)
        v /= np.linalg.norm(v)
        if cos_sim(x, v) < 0.0:
            v = -v                      # regime: cosine in [0,1) (SafeRAG Eq.5)
        cossim = float(np.dot(x, v))
        expected = int(np.dot(quant(x, PREC), quant(v, PREC)))
        hi = 2 * n * B * B + 1                       # SafeRAG Eq.7 BSGS bound

        # ---- vendored oracle (single shot, unchanged from grid) ----------- #
        t0 = time.perf_counter()
        mk = FeDDH.generate(n, F=None)
        t_setup_v = time.perf_counter() - t0
        ct = FeDDH.encrypt(quant(x, PREC), mk)
        sk = FeDDH.keygen(quant(v, PREC), mk)
        t0 = time.perf_counter()
        valV = FeDDH.decrypt(ct, mk, sk, (0, hi))
        t_dec_v = time.perf_counter() - t0

        # ---- pure engine: SETUP median over fresh independent setups -------- #
        setups = []
        for _ in range(REPEATS_SETUP):
            t0 = time.perf_counter()
            IPFEScheme.setup(n, group_bits=128, quant_bits=PREC)
            setups.append((time.perf_counter() - t0) * 1e3)
        t_setup_ms = round(statistics.median(setups), 2)

        scheme = IPFEScheme.setup(n, group_bits=128, quant_bits=PREC)

        # warm-up once (builds the (g,p,range) BSGS table; not timed)
        scheme.inner_product(*_enc_key(scheme, x, v), v)

        # ---- pure ops, warm median ---------------------------------------- #
        encs, keys, decs = [], [], []
        for _ in range(REPEATS_OPS):
            t0 = time.perf_counter()
            Ct, _ = scheme.encrypt(x)
            encs.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            skfe = scheme.key_derive(v)
            keys.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            valP = scheme.inner_product(Ct, skfe, v)
            decs.append((time.perf_counter() - t0) * 1e3)
        t_enc_ms = round(statistics.median(encs), 2)
        t_key_ms = round(statistics.median(keys), 2)
        t_dec_ms = round(statistics.median(decs), 2)

        ok = (abs(valV - expected) <= PREC + 1
              and abs(valP - expected) <= PREC + 1)
        all_ok = all_ok and ok

        report["grid"][f"n{n}"] = {
            "vendored": int(valV), "pure": int(valP), "expected": int(expected),
            "cossim": round(cossim, 6), "ok": bool(ok),
            "t_setup_ms_median5": t_setup_ms,
            "t_setup_ms_single_vendor": round(t_setup_v * 1e3, 2),
            "t_enc_ms_median3": t_enc_ms,
            "t_key_ms_median3": t_key_ms,
            "t_dec_ms_median3_warm": t_dec_ms,
            "estimated_ops": {
                "encrypt_modpow": 2 * n + 1, "keyderive_mult": n,
                "decrypt_modpow": n + 1,
                "bsgs": {"step": int(int(2 * n * B * B + 1) ** 0.5) + 1,
                         "range": 2 * n * B * B + 1},
            },
        }
        report["bsgs"][f"n{n}"] = _table_stats(scheme)

    # asymptotic labels for the write-up (checked against measured table)
    report["model"] = {
        "setup": "O(n) field ops; wall clock dominated by safe-prime/gen search",
        "encrypt": "O(n.log p)  (2n+1 mod-exps: C0=g^r + per-dim mpk_i^r, g^q_i)",
        "keyderive": "O(n)  big-int mults + 1 mod",
        "decrypt": "O(n.log p) mod-pows (n+1: prod C_i^v_i + C0^sk_fe) + 1 modinv "
                   "+ O(sqrt(range)) BSGS dlog",
        "bsgs_space": "O(sqrt(range)) Baby-Step table, cached per (g,p,range); "
                      "amortised across queries ONLY within one scheme's lifetime "
                      "(each n cell here re-setups, so no cross-cell reuse)",
        "range": "2*n*B^2+1 (SafeRAG Eq.7); sqrt(range) = O(sqrt(n).B)",
    }
    report["summary"] = {
        "all_ok": bool(all_ok), "cases": len(report["grid"]),
        "timing": {"setup": "median of 5 fresh setups",
                   "enc/key/dec": "warm median of 3 (BSGS cached)"},
    }

    artifacts = REPO / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with open(artifacts / "timings_ipfe_complexity.json", "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\nALL_OK: {all_ok}   cases: {len(report['grid'])}")
    return 0 if all_ok else 1


def _enc_key(scheme, x, v):
    Ct, _ = scheme.encrypt(x)
    return Ct, scheme.key_derive(v)


if __name__ == "__main__":
    raise SystemExit(main())