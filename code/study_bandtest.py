# -*- coding: utf-8 -*-
r"""Test the screening rule inside its own decision band.

The limitation Paper 3 raises first against itself is that no archive's median
detected onset falls between 70 and 90 percent of life, so the recommended
threshold sits in a gap rather than on data.  An earlier search for a marginal
archive came back empty and was reported as such.

That search asked the wrong question.  The rule judges an asset, not an archive,
and per record the band is not empty: 169 of the 740 run-to-failure records the
paper uses land in it (\\code{study_band.py}).  Two archive medians land in it as
well, C-MAPSS FD002 at 85.0 and FD004 at 87.8 percent, in the raw six-regime
form where the detector fires on only nine units in each.

So the band can be populated.  What it needs is the experiment: take units
spanning the whole onset range, run the same path model the paper runs on
bearings, and ask whether prediction quality falls off across the band the way
the rule says it should.  If it does, the threshold is calibrated.  If accuracy
is flat across the band, the rule is drawing a line where the data has no edge,
and the paper should say so.

Nothing is re-tuned.  The onset rule keeps the paper's three constants; the
filter keeps its priors, particle count and threshold calibration.  The only new
code is the adapter that presents a C-MAPSS unit in the record shape the filter
already consumes.

    python code/study_bandtest.py            default sample
    python code/study_bandtest.py 12         units per onset bin
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import study_bearings as sb                                    # noqa: E402
import study_thirdset as ts                                    # noqa: E402


def stable_seed(*key):
    """A seed that is the same on every run, machine and Python build.

    ``hash()`` is unsuitable: PYTHONHASHSEED randomises it per process for any
    key containing a string, so a study seeded that way cannot be reproduced.
    """
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000

OUT = os.path.join(HERE, "..", "results", "bearings")
PER_BIN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260805
STEM = sys.argv[3] if len(sys.argv) > 3 else "bandtest"
BINS = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
rng = np.random.default_rng(SEED)


def as_record(hi, dt=1.0):
    """A C-MAPSS unit in the shape the bearing filter consumes."""
    hi = np.asarray(hi, float)
    n = len(hi)
    t = np.arange(n, dtype=float) * dt
    return {"t": t, "hi": hi, "condition": 0, "life": float(t[-1]),
            "n": n, "dt": dt}


print("  building C-MAPSS records and their detected onsets")
units = {}
for fd, by_regime in (("FD002", True), ("FD004", True)):
    for u, arr in ts.load(fd, by_regime=by_regime).items():
        hi = ts.indicator(arr)
        k = ts.onset(hi)
        if k is None or len(hi) < 60:
            continue
        units[f"{fd}:{u}"] = (as_record(hi), k / len(hi))
print(f"  {len(units)} units with a detected onset\n")

#  sample within each onset bin, so the comparison spans the band rather than
#  being dominated by wherever the population happens to pile up
picked = {}
for lo, hi_ in BINS:
    pool = [k for k, (_r, f) in units.items() if lo <= f < hi_]
    if not pool:
        continue
    take = list(rng.permutation(pool))[:PER_BIN]
    picked[(lo, hi_)] = take
    print(f"  onset {lo:.0%}-{hi_:.0%}: {len(pool):3d} available, {len(take)} sampled")

#  the failure threshold is calibrated on OTHER units, exactly as for bearings:
#  the median terminal indicator of the units not being predicted
all_keys = [k for ks in picked.values() for k in ks]
terminal = {k: float(np.median(units[k][0]["hi"][-3:])) for k in units}

print("\n  running the path model (BS increments, informed prior, Paris coupling)")
rows = []
for (lo, hi_), keys in picked.items():
    for key in keys:
        rec, frac = units[key]
        ref = [k for k in units if k != key]
        D = float(np.median([terminal[k] for k in ref]))
        med_life = float(np.median(
            [max(units[k][0]["n"] - ts.onset(units[k][0]["hi"]) or 20, 20)
             for k in rng.permutation(ref)[:40]]))
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
                               seed=stable_seed(key, k_pred),
                               n_part=600)
            if r is None:
                continue
            er = 100.0 * (true - r["pred"]) / true
            rows.append({"unit": key, "onset_frac": frac, "bin": f"{lo:.2f}-{hi_:.2f}",
                         "k": k_pred, "at_life": frac_pt, "true": true,
                         "pred": float(r["pred"]), "abs_pct_err": abs(er),
                         "covered": bool(r.get("lo", 0) <= true <= r.get("hi", 0))})
    print(f"    bin {lo:.0%}-{hi_:.0%} done ({len(rows)} predictions so far)")

print("\n  prediction quality against detected onset\n")
print(f"  {'onset bin':14s} {'units':>5s} {'preds':>6s} "
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
    band = "  <-- decision band" if lo >= 0.70 and hi_ <= 0.90 else ""
    print(f"  {lo:.0%}-{hi_:.0%}      {nu:5d} {len(sel):6d} "
          f"{med:15.1f} {cov:9.2f}{band}")
    summary[tag] = {"units": nu, "predictions": len(sel),
                    "median_abs_pct_err": med, "coverage": cov}

#  The arguments a run was launched with have to survive in the result.  The
#  replication's sampling seed was recoverable from neither its JSON nor its
#  log, so the run could be described but not repeated.
json.dump({"config": {"per_bin": PER_BIN, "sampling_seed": SEED, "stem": STEM},
           "per_bin": summary, "rows": rows},
          io.open(os.path.join(OUT, f"{STEM}.json"), "w", encoding="utf-8"),
          indent=1)
print(f"\n  wrote results/bearings/{STEM}.json")
