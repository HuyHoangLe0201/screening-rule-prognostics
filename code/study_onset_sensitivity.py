"""Is the late-onset finding a property of the bearings or of our detector?

Paper 3's central negative result -- a median detected onset at 97 % of life on
PRONOSTIA against 41 % on XJTU-SY -- is the output of one onset rule whose three
swept constants are a 3 robust-standard-deviation band, a persistence run, and an
80 % sustained-departure requirement.  A referee is entitled to say that this
measures the detector rather than the asset.

This sweeps those three, 60 settings per archive, and reports what the finding
does.  A fourth choice is NOT swept and is worth naming: the healthy window,
min(40, n/10) acquisitions, from which the baseline and the robust scale are
taken.  It is held at that rule in every cell, so what follows bounds
sensitivity to the three and not to the window.

The sweep re-implements the
rule locally, with the constants exposed, rather than importing
datasets.first_prediction_time, so that the sweep cannot silently inherit a
default the paper does not state.

    python code/study_onset_sensitivity.py

Writes results/bearings/onset_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import statistics as st

import numpy as np

import datasets as ds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "bearings")
os.makedirs(OUT, exist_ok=True)


def onset(hi, n_health=40, thr=3.0, persist=10, frac=0.8):
    """datasets.first_prediction_time with every constant exposed.

    Returns 0 when nothing is detected, matching the production rule, so that
    "no onset" and "onset at the first sample" stay distinguishable only through
    the detected flag the caller derives.
    """
    hi = np.asarray(hi, dtype=float)
    n = len(hi)
    nh = min(n_health, max(5, n // 10))
    base = np.median(hi[:nh])
    mad = np.median(np.abs(hi[:nh] - base)) * 1.4826
    above = hi > base + thr * max(mad, 1e-9)
    frac_above = np.cumsum(above[::-1])[::-1] / np.maximum(np.arange(n, 0, -1), 1)
    for i in range(n - persist):
        if above[i:i + persist].all() and frac_above[i] >= frac:
            return i
    return 0


THR = [2.0, 2.5, 3.0, 4.0, 5.0]
PERSIST = [5, 10, 20, 40]
FRAC = [0.5, 0.8, 0.95]

out = {}
for dataset in ("pronostia", "xjtu"):
    data = ds.load_dataset(dataset)
    recs = [(b, rec["hi"], rec["n"]) for b, rec in sorted(data.items())]
    rows = []
    for thr in THR:
        for persist in PERSIST:
            for frac in FRAC:
                pos = []
                for _, hi, n in recs:
                    i = onset(hi, thr=thr, persist=persist, frac=frac)
                    if i > 0:
                        pos.append(100.0 * i / n)
                rows.append({
                    "thr": thr, "persist": persist, "frac": frac,
                    "n_records": len(recs),
                    "n_detected": len(pos),
                    "median_onset_pct": st.median(pos) if pos else None,
                    "median_deg_phase_pct": (100.0 - st.median(pos)) if pos else None,
                })
    out[dataset] = rows

with open(os.path.join(OUT, "onset_sensitivity.json"), "w") as f:
    json.dump(out, f, indent=1)

for dataset, rows in out.items():
    print(f"\n=== {dataset}")
    print(f"{'thr':>5}{'persist':>9}{'frac':>7}{'detected':>10}"
          f"{'median onset %':>16}{'deg phase %':>13}")
    for r in rows:
        m = r["median_onset_pct"]
        d = r["median_deg_phase_pct"]
        star = "  <-- paper" if (r["thr"] == 3.0 and r["persist"] == 10
                                 and r["frac"] == 0.8) else ""
        print(f"{r['thr']:>5.1f}{r['persist']:>9d}{r['frac']:>7.2f}"
              f"{r['n_detected']:>6d}/{r['n_records']:<3d}"
              f"{(f'{m:.1f}' if m is not None else '--'):>16}"
              f"{(f'{d:.1f}' if d is not None else '--'):>13}{star}")

    det = [r for r in rows if r["median_onset_pct"] is not None]
    if det:
        lo = min(r["median_onset_pct"] for r in det)
        hi_ = max(r["median_onset_pct"] for r in det)
        print(f"  median onset ranges {lo:.1f}--{hi_:.1f} % of life over "
              f"{len(det)}/{len(rows)} settings that detect anything")

print("\nwrote results/bearings/onset_sensitivity.json")
