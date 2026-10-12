# -*- coding: utf-8 -*-
r"""The screening rule on a fifth asset class: hard disk drives.

The step in prediction error at the screening threshold has been shown on
turbofan units, could not be tested on lithium cells (no cell in the band), and
does not appear on the two bearing archives (which lie wholly on the
non-applying side).  What is missing is an asset class whose degradation is
observable but late, which is the profile the band describes.

Disk drives have exactly that profile.  The SMART counters that matter --
reallocated sectors, pending sectors, uncorrectable errors -- sit at zero for
most of a drive's life and climb only near the end.  Backblaze publishes one
daily row per drive per day, with a failure flag, for its entire fleet.

Two design decisions make this usable, both forced by the data:

  1. LIFE IS MEASURED IN POWER-ON HOURS, not in samples.  A quarterly file shows
     only a 92-day window of a drive that may be years old, so a fraction of the
     window would not be a fraction of life.  ``smart_9_raw`` records the drive's
     own age at every sample, so the onset position is (age at onset) / (age at
     failure) regardless of when observation started.

  2. ONLY DRIVES STILL AT BASELINE WHEN THE WINDOW OPENS ARE USED.  If a drive
     already shows reallocated sectors on day one, its onset happened before the
     window and cannot be located.  Requiring a clean start is what makes the
     detected onset the true one.

The onset rule keeps the paper's three constants.

    python code/study_hdd.py                path to zip inferred
    python code/study_hdd.py <file.zip>
"""
from __future__ import annotations

import csv
import io
import json
import os
import statistics as st
import sys
import zipfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(HERE, "..", "..", "data", "bb_Q4_2023.zip"))
OUT = os.path.join(HERE, "..", "results", "bearings")

#  counters that rise as a drive degrades; all are cumulative and start at zero
SMART = ["smart_5_raw", "smart_187_raw", "smart_188_raw",
         "smart_197_raw", "smart_198_raw"]
AGE = "smart_9_raw"
THR, PERSIST, FRAC, N_HEALTH = 3.0, 10, 0.8, 40
MIN_SAMPLES = 60


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


print(f"  reading {os.path.basename(ZIP)}")
series: dict[str, list] = defaultdict(list)
failed: set[str] = set()
n_rows = 0

with zipfile.ZipFile(ZIP) as z:
    names = sorted(n for n in z.namelist() if n.endswith(".csv"))
    print(f"  {len(names)} daily files")
    for j, name in enumerate(names):
        with z.open(name) as fh:
            rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8",
                                                 errors="replace"))
            for row in rd:
                n_rows += 1
                sn = row.get("serial_number")
                if not sn:
                    continue
                try:
                    age = float(row.get(AGE) or "nan")
                except ValueError:
                    age = float("nan")
                vals = []
                for k in SMART:
                    try:
                        vals.append(float(row.get(k) or 0.0))
                    except ValueError:
                        vals.append(0.0)
                series[sn].append((row.get("date", ""), age, vals))
                if (row.get("failure") or "0").strip() == "1":
                    failed.add(sn)
        if (j + 1) % 20 == 0:
            print(f"    {j + 1}/{len(names)} files, {n_rows/1e6:.1f}M rows, "
                  f"{len(failed)} failures so far")

print(f"\n  {n_rows/1e6:.1f}M rows, {len(series)} drives, {len(failed)} failures")

rows = []
skipped_dirty = skipped_short = skipped_age = 0
for sn in failed:
    rec = sorted(series[sn], key=lambda r: r[0])
    if len(rec) < MIN_SAMPLES:
        skipped_short += 1
        continue
    #  a drive already showing counters on the first day had its onset before
    #  the window opened; its position cannot be located
    if any(v > 0 for v in rec[0][2]):
        skipped_dirty += 1
        continue
    ages = np.array([r[1] for r in rec], float)
    if not np.isfinite(ages).all() or ages[-1] <= 0 or ages[-1] <= ages[0]:
        skipped_age += 1
        continue
    #  the paper's own indicator construction: robust z-scores of each channel
    #  against the unit's healthy window, summed
    arr = np.array([r[2] for r in rec], float)
    nh = min(N_HEALTH, max(5, len(rec) // 10))
    hi = np.zeros(len(rec))
    for c in range(arr.shape[1]):
        col = arr[:, c]
        base = np.median(col[:nh])
        mad = np.median(np.abs(col[:nh] - base)) * 1.4826
        hi += (col - base) / max(mad, 1e-9)
    i = onset(hi)
    if i is None:
        rows.append({"serial": sn, "n": len(rec), "life_hours": float(ages[-1]),
                     "onset_frac": None, "deg_phase_frac": None})
        continue
    rows.append({"serial": sn, "n": len(rec), "life_hours": float(ages[-1]),
                 "onset_frac": float(ages[i] / ages[-1]),
                 "deg_phase_frac": float(1.0 - ages[i] / ages[-1])})

det = [r for r in rows if r["onset_frac"] is not None]
print(f"  usable failed drives: {len(rows)}   "
      f"(skipped: {skipped_short} too short, {skipped_dirty} already degraded "
      f"at the window start, {skipped_age} bad age)")
print(f"  with a detected onset: {len(det)}")

if det:
    fr = sorted(r["onset_frac"] for r in det)
    print(f"  median detected onset: {100 * st.median(fr):.1f}% of life")
    print("\n  distribution of onset (fraction of power-on hours at failure)")
    edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    for lo, hi_ in zip(edges, edges[1:]):
        k = sum(1 for f in fr if lo <= f < hi_)
        mark = "  <-- decision band" if lo >= .7 and hi_ <= .9 else ""
        print(f"    {lo:.0%}-{hi_:.0%}  {k:5d}  {'#' * min(k, 60)}{mark}")
    band = [r for r in det if 0.70 <= r["onset_frac"] <= 0.90]
    print(f"\n  in the 70-90% band: {len(band)} drives")

os.makedirs(OUT, exist_ok=True)
json.dump({"n_failed": len(failed), "n_usable": len(rows),
           "n_detected": len(det),
           "median_onset_pct": (100 * st.median([r["onset_frac"] for r in det])
                                if det else None),
           "drives": rows},
          io.open(os.path.join(OUT, "hdd.json"), "w", encoding="utf-8"), indent=1)
print("\n  wrote results/bearings/hdd.json")
