"""A marginal archive for the screening rule: NASA Li-ion battery capacity fade.

C-MAPSS validated the rule out of sample, but it separated so cleanly --- onset
in 200 units out of 200 --- that it could not have detected a badly chosen
detection band.  The test still missing is an archive that is MARGINAL under the
rule rather than clearly on one side of it.

Battery capacity fade is the natural candidate, and for reasons that are
adversarial in a different direction from every archive used so far:

  * There is no healthy plateau.  Capacity falls from the first cycle, so the
    window the rule uses to estimate a healthy baseline and scale already
    contains real degradation.  That inflates the band with signal, which should
    make onset LATER than it truly is -- the opposite of the PRONOSTIA failure.
  * Degradation is non-monotone.  Capacity partially recovers after rest
    (regeneration), so a rule requiring the departure to be sustained over
    80 % of the remaining record can be broken by recovery even though the
    asset is genuinely degrading.
  * The indicator falls rather than rises, so the direction must be handled.

If the rule reports something sensible here it is doing more than recognising
flat records; if it fails, the failure is informative and belongs in the paper.

    python code/study_battery.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("P3_DATA", os.path.join(HERE, "..", "data"))
OUT = os.path.join(HERE, "..", "results", "bearings")
os.makedirs(OUT, exist_ok=True)

CELLS = ["B0005", "B0006", "B0007", "B0018"]
RATED_AH = 2.0
EOL_FRAC = 0.7           # standard end-of-life for these cells

#  the paper's constants, unchanged
THR, PERSIST, FRAC, N_HEALTH = 3.0, 10, 0.8, 40


def onset(hi, n_health=N_HEALTH, thr=THR, persist=PERSIST, frac=FRAC):
    hi = np.asarray(hi, float)
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


def capacity(cell):
    m = sio.loadmat(os.path.join(SRC, f"{cell}.mat"), simplify_cells=True)
    cycles = m[cell]["cycle"]
    caps = [float(np.ravel(c["data"]["Capacity"])[0])
            for c in cycles if c["type"] == "discharge"]
    return np.asarray(caps, float)


rows = []
for cell in CELLS:
    cap = capacity(cell)
    n = len(cap)
    #  Indicator in the paper's own form: departure from the healthy window,
    #  signed so that degradation is positive.  Capacity FALLS, so the
    #  indicator is the drop.
    hi = cap[0] - cap
    i = onset(hi)
    eol = RATED_AH * EOL_FRAC
    crossed = np.nonzero(cap <= eol)[0]
    rows.append({
        "cell": cell,
        "n_cycles": n,
        "cap_first": float(cap[0]),
        "cap_last": float(cap[-1]),
        #  how far the capacity trace walks back up: the regeneration that a
        #  sustained-departure rule has to survive
        "max_recovery_ah": float(np.max(np.maximum.accumulate(hi) - hi)),
        "onset_idx": int(i),
        "onset_pct": (100.0 * i / n) if i > 0 else None,
        "deg_phase_pct": (100.0 - 100.0 * i / n) if i > 0 else None,
        "eol_reached": bool(len(crossed)),
        "eol_pct": (100.0 * crossed[0] / n) if len(crossed) else None,
    })

det = [r for r in rows if r["onset_pct"] is not None]
summary = {
    "cells": rows,
    "n_cells": len(rows),
    "n_detected": len(det),
    "median_onset_pct": st.median([r["onset_pct"] for r in det]) if det else None,
    "median_deg_phase_pct": (st.median([r["deg_phase_pct"] for r in det])
                             if det else None),
}
with open(os.path.join(OUT, "battery.json"), "w") as f:
    json.dump(summary, f, indent=1)

print("NASA Li-ion battery, screening rule with the paper's constants unchanged")
print(f"  (band {THR} robust s.d., persistence {PERSIST}, sustained {FRAC})\n")
print(f"{'cell':<8}{'cycles':>8}{'cap Ah':>16}{'recovery':>10}"
      f"{'onset %':>9}{'deg %':>8}{'EOL at %':>10}")
for r in rows:
    o = f"{r['onset_pct']:.1f}" if r["onset_pct"] is not None else "none"
    d = f"{r['deg_phase_pct']:.1f}" if r["deg_phase_pct"] is not None else "--"
    e = f"{r['eol_pct']:.0f}" if r["eol_pct"] is not None else "not reached"
    print(f"{r['cell']:<8}{r['n_cycles']:>8}"
          f"{r['cap_first']:>8.2f}->{r['cap_last']:<7.2f}"
          f"{r['max_recovery_ah']:>10.3f}{o:>9}{d:>8}{e:>10}")

print(f"\ndetected in {len(det)}/{len(rows)} cells; "
      f"median onset {summary['median_onset_pct']}")
print("\nfor comparison:")
print("  PRONOSTIA  12/17  onset 97 %   XJTU-SY 15/15 onset 41 %")
print("  C-MAPSS single-condition 200/200 onset 56-57 %")
print("\nwrote results/bearings/battery.json")
