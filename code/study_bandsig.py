# -*- coding: utf-8 -*-
r"""Is the step at the screening threshold real, or is it twenty units?

``study_bandtest.py`` finds median absolute RUL error roughly doubling as the
detected onset crosses 70 percent of life: 53-62 percent below, 105 percent
above.  That is where the paper's threshold sits, which is the answer the paper
wants -- and exactly the reason to check it before believing it.

Two checks, both clustered on the unit, because every prediction point inside a
unit shares that unit's trajectory and treating them as independent would
inflate any difference:

  1. a cluster bootstrap on the difference in median absolute error between the
     units below the threshold and those above it;
  2. a permutation test that shuffles the below/above label across units, which
     asks how often a split this large arises from a threshold placed at random.

A step that survives both is evidence the threshold is on an edge in the data.
A step that does not is evidence the paper should keep saying its threshold is
uncalibrated.

    python code/study_bandsig.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bearings")
CUT = 0.70
B = 4000
rng = np.random.default_rng(20260806)

STEM = sys.argv[1] if len(sys.argv) > 1 else "bandtest"
rows = json.load(io.open(os.path.join(RES, f"{STEM}.json"),
                        encoding="utf-8"))["rows"]
by_unit = {}
for r in rows:
    by_unit.setdefault(r["unit"], []).append(r)
units = sorted(by_unit)
onset = {u: by_unit[u][0]["onset_frac"] for u in units}
below = [u for u in units if onset[u] < CUT]
above = [u for u in units if onset[u] >= CUT]
print(f"  {len(units)} units, {len(rows)} predictions")
print(f"  below {CUT:.0%}: {len(below)} units   above: {len(above)} units\n")


def med_err(unit_list):
    vals = [r["abs_pct_err"] for u in unit_list for r in by_unit[u]]
    return st.median(vals) if vals else float("nan")


obs = med_err(above) - med_err(below)
print(f"  observed: below {med_err(below):6.1f}%   above {med_err(above):6.1f}%"
      f"   difference {obs:+.1f} points\n")

#  1. cluster bootstrap: resample units with replacement, within each group
diffs = []
for _ in range(B):
    b = list(rng.choice(below, len(below), replace=True))
    a = list(rng.choice(above, len(above), replace=True))
    diffs.append(med_err(a) - med_err(b))
diffs = np.asarray(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"  cluster bootstrap  95% interval on the difference: "
      f"[{lo:+.1f}, {hi:+.1f}] points")
print(f"                     excludes zero: {'yes' if lo > 0 or hi < 0 else 'NO'}")

#  2. permutation: is 70% special, or would any split do as well?
perm = []
allu = below + above
for _ in range(B):
    p = list(rng.permutation(allu))
    perm.append(med_err(p[len(below):]) - med_err(p[:len(below)]))
perm = np.asarray(perm)
pval = float((np.abs(perm) >= abs(obs)).mean())
print(f"\n  permutation over the below/above label: p = {pval:.4f}")

#  3. does the threshold sit where the step is?  scan every candidate cut
print("\n  the same difference computed at other candidate thresholds")
scan = {}
for cut in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
    b = [u for u in units if onset[u] < cut]
    a = [u for u in units if onset[u] >= cut]
    if len(b) < 5 or len(a) < 5:
        continue
    d = med_err(a) - med_err(b)
    scan[cut] = d
    mark = "  <-- the paper's threshold" if abs(cut - CUT) < 1e-9 else ""
    print(f"    cut at {cut:.0%}: below {med_err(b):6.1f}%  above {med_err(a):6.1f}%"
          f"   difference {d:+7.1f}{mark}")

json.dump({"cut": CUT, "observed_diff": obs,
           "bootstrap_ci": [float(lo), float(hi)], "permutation_p": pval,
           "scan": {f"{k:.2f}": v for k, v in scan.items()},
           "n_units_below": len(below), "n_units_above": len(above)},
          io.open(os.path.join(RES, f"{STEM.replace('test','sig')}.json"), "w", encoding="utf-8"), indent=1)
print(f"\n  wrote results/bearings/{STEM.replace('test', 'sig')}.json")
