# -*- coding: utf-8 -*-
r"""The band test on a second asset class: NASA lithium-ion cells.

The step in prediction error at the screening threshold has so far been shown on
turbofan units only.  That leaves the obvious question open -- is the edge a
property of the problem, or of C-MAPSS?

The archive to answer it was already on disk.  Paper 3 used four cells from the
NASA battery set; the set contains thirty-eight files across six aging
campaigns.  Loading all of them, deduplicating by cell identifier and keeping
the longest record for each, gives enough cells to span the onset range and run
the same test.

Nothing is re-tuned.  The onset rule keeps the paper's three constants, the
indicator keeps the form used for batteries throughout (capacity drop from the
healthy window), and the filter keeps its priors, particle count and threshold
calibration.

    python code/study_bandbattery.py           default 6 cells per bin
    python code/study_bandbattery.py 8 12345   cells per bin, seed
"""
from __future__ import annotations

import glob
import io
import hashlib
import json
import os
import statistics as st
import sys

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_bearings as sb                                     # noqa: E402


def stable_seed(*key):
    """A seed that is the same on every run, machine and Python build.

    ``hash()`` is unsuitable: PYTHONHASHSEED randomises it per process for any
    key containing a string, so a study seeded that way cannot be reproduced.
    """
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000

SRC = os.path.join(HERE, "..", "..", "data", "NASABattery")
OUT = os.path.join(HERE, "..", "results", "bearings")
PER_BIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260807
BINS = [(0.00, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 0.80), (0.80, 1.01)]
THR, PERSIST, FRAC, N_HEALTH = 3.0, 10, 0.8, 40
rng = np.random.default_rng(SEED)


def onset(hi):
    hi = np.asarray(hi, float)
    n = len(hi)
    nh = min(N_HEALTH, max(5, n // 10))
    base = np.median(hi[:nh])
    mad = np.median(np.abs(hi[:nh] - base)) * 1.4826
    above = hi > base + THR * max(mad, 1e-9)
    fa = np.cumsum(above[::-1])[::-1] / np.maximum(np.arange(n, 0, -1), 1)
    for i in range(n - PERSIST):
        if above[i:i + PERSIST].all() and fa[i] >= FRAC:
            return i
    return None


def capacities(path):
    key = os.path.splitext(os.path.basename(path))[0]
    m = sio.loadmat(path, simplify_cells=True)
    if key not in m:
        return key, None
    cyc = m[key]["cycle"]
    caps = []
    for c in np.atleast_1d(cyc):
        if c.get("type") != "discharge":
            continue
        try:
            caps.append(float(np.ravel(c["data"]["Capacity"])[0]))
        except Exception:
            continue
    return key, (np.asarray(caps, float) if len(caps) >= 60 else None)


print("  loading every cell in the archive")
best: dict[str, np.ndarray] = {}
for p in sorted(glob.glob(os.path.join(SRC, "**", "*.mat"), recursive=True)):
    try:
        key, cap = capacities(p)
    except Exception:
        continue
    if cap is None:
        continue
    #  the same cell appears in more than one campaign folder; keep the longest
    if key not in best or len(cap) > len(best[key]):
        best[key] = cap
print(f"  {len(best)} distinct cells with at least 60 discharge cycles")

records = {}
for key, cap in best.items():
    hi = cap[0] - cap                       # capacity drop, as the paper uses
    i = onset(hi)
    if i is None or i == 0:
        continue
    n = len(hi)
    records[key] = ({"t": np.arange(n, dtype=float), "hi": hi, "condition": 0,
                     "life": float(n - 1), "n": n, "dt": 1.0}, i / n)
print(f"  {len(records)} cells with a detected onset\n")

picked = {}
for lo, hi_ in BINS:
    pool = [k for k, (_r, f) in records.items() if lo <= f < hi_]
    if not pool:
        print(f"  onset {lo:.0%}-{hi_:.0%}:   0 available")
        continue
    take = list(rng.permutation(pool))[:PER_BIN]
    picked[(lo, hi_)] = take
    print(f"  onset {lo:.0%}-{hi_:.0%}: {len(pool):3d} available, {len(take)} sampled")

terminal = {k: float(np.median(records[k][0]["hi"][-3:])) for k in records}

print("\n  running the path model (BS increments, informed prior, Paris coupling)")
rows = []
for (lo, hi_), keys in picked.items():
    for key in keys:
        rec, frac = records[key]
        ref = [k for k in records if k != key]
        D = float(np.median([terminal[k] for k in ref]))
        med_life = float(np.median(
            [max(records[k][0]["n"] - (onset(records[k][0]["hi"]) or 0), 20)
             for k in ref]))
        n = rec["n"]
        for frac_pt in (0.5, 0.7, 0.85):
            k_pred = int(frac_pt * n)
            if k_pred < 20 or k_pred >= n - 1:
                continue
            true = float(rec["t"][-1] - rec["t"][k_pred])
            if true <= 0:
                continue
            r = sb.predict_one(rec, k_pred, D, "BS", informed=True,
                               coupling=True, med_life_steps=med_life,
                               seed=stable_seed(key, k_pred), n_part=600)
            if r is None:
                continue
            er = 100.0 * (true - r["pred"]) / true
            rows.append({"unit": key, "onset_frac": frac,
                         "bin": f"{lo:.2f}-{hi_:.2f}", "k": k_pred,
                         "at_life": frac_pt, "true": true,
                         "pred": float(r["pred"]), "abs_pct_err": abs(er),
                         "covered": bool(r.get("lo", 0) <= true <= r.get("hi", 0))})
    print(f"    bin {lo:.0%}-{hi_:.0%} done ({len(rows)} predictions so far)")

print("\n  prediction quality against detected onset, lithium-ion cells\n")
print(f"  {'onset bin':14s} {'cells':>5s} {'preds':>6s} "
      f"{'median |err| %':>15s} {'coverage':>9s}")
summary = {}
for (lo, hi_) in BINS:
    tag = f"{lo:.2f}-{hi_:.2f}"
    sel = [r for r in rows if r["bin"] == tag]
    if not sel:
        continue
    med = st.median([r["abs_pct_err"] for r in sel])
    cov = sum(r["covered"] for r in sel) / len(sel)
    nu = len({r["unit"] for r in sel})
    band = "  <-- decision band" if lo >= 0.70 else ""
    print(f"  {lo:.0%}-{hi_:.0%}      {nu:5d} {len(sel):6d} {med:15.1f} "
          f"{cov:9.2f}{band}")
    summary[tag] = {"units": nu, "predictions": len(sel),
                    "median_abs_pct_err": med, "coverage": cov}

json.dump({"config": {"per_bin": PER_BIN, "sampling_seed": SEED},
           "per_bin": summary, "rows": rows, "n_cells": len(records)},
          io.open(os.path.join(OUT, "bandbattery.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  wrote results/bearings/bandbattery.json")
