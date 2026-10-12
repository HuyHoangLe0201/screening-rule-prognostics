# -*- coding: utf-8 -*-
r"""Does prediction error rise with detected onset on the bearings too?

The step at the screening threshold has been shown on turbofan units.  The
attempt to repeat it on lithium-ion cells failed because that archive has no
cell in the band.  A third asset class is not available to download here, but a
second one is already on disk with its predictions computed: the two bearing
archives, whose path-model runs are stored in results_both.json.

So the test can be run without any new fitting.  Join each prediction to the
detected onset of its own bearing -- the same onset the paper reports in
\Cref{tab:applicability} -- and ask whether error rises with onset, clustered on
the bearing.

The band itself is thin here: of the twenty-seven bearings with a detected
onset, three fall between 70 and 90 percent of life.  So this cannot calibrate
the threshold.  What it can do is say whether the direction holds on a second
asset class, which is the part the battery attempt could not answer either.

    python code/study_bandbearing.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bearings")
CUT = 0.70
B = 4000
rng = np.random.default_rng(20260807)

onset = {}
for r in json.load(io.open(os.path.join(RES, "applicability.json"),
                           encoding="utf-8")):
    if r.get("fpt_frac"):
        onset[(r["dataset"], r["bearing"])] = r["fpt_frac"]
print(f"  {len(onset)} bearings with a detected onset")

rows = json.load(io.open(os.path.join(RES, "results_both.json"),
                         encoding="utf-8"))["rows"]
#  the paper's own configuration: BS increments, informed prior, coupling on
sel = [r for r in rows
       if r.get("law") == "BS" and r.get("informed") and r.get("coupling")
       and (r["dataset"], r["bearing"]) in onset]
print(f"  {len(sel)} predictions from the paper's own runs\n")

by_bearing = defaultdict(list)
for r in sel:
    by_bearing[(r["dataset"], r["bearing"])].append(abs(r["er"]))

bearings = sorted(by_bearing)
below = [b for b in bearings if onset[b] < CUT]
above = [b for b in bearings if onset[b] >= CUT]
print(f"  below {CUT:.0%}: {len(below)} bearings   above: {len(above)}")
inband = [b for b in bearings if 0.70 <= onset[b] <= 0.90]
#  Print the dataset too: both archives contain a Bearing1_4, and a list of bare
#  names shows the same string twice and reads like a duplicate.
print(f"  inside the 70-90% band: {len(inband)}  "
      f"({', '.join(f'{d}/{b}' for d, b in inband) if inband else 'none'})\n")


def med(group):
    v = [e for b in group for e in by_bearing[b]]
    return st.median(v) if v else float("nan")


obs = med(above) - med(below)
print(f"  below {med(below):7.1f}%   above {med(above):7.1f}%   "
      f"difference {obs:+.1f} points")

diffs = []
for _ in range(B):
    a = list(rng.choice(range(len(above)), len(above), replace=True))
    b = list(rng.choice(range(len(below)), len(below), replace=True))
    diffs.append(med([above[i] for i in a]) - med([below[i] for i in b]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"  cluster bootstrap on the bearing: [{lo:+.1f}, {hi:+.1f}] points"
      f"   {'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES zero'}")

allb = below + above
perm = []
for _ in range(B):
    p = list(rng.permutation(len(allb)))
    perm.append(med([allb[i] for i in p[len(below):]])
                - med([allb[i] for i in p[:len(below)]]))
pval = float((np.abs(np.asarray(perm)) >= abs(obs)).mean())
print(f"  permutation over the below/above label: p = {pval:.4f}")

#  a rank correlation uses every bearing rather than a split, so it does not
#  depend on where the threshold is put
x = np.array([onset[b] for b in bearings])
y = np.array([st.median(by_bearing[b]) for b in bearings])
rx, ry = x.argsort().argsort(), y.argsort().argsort()
rho = float(np.corrcoef(rx, ry)[0, 1])
boot = []
for _ in range(B):
    idx = rng.choice(len(bearings), len(bearings), replace=True)
    if len(set(x[idx])) < 3:
        continue
    a, b = x[idx].argsort().argsort(), y[idx].argsort().argsort()
    boot.append(float(np.corrcoef(a, b)[0, 1]))
rlo, rhi = np.percentile(boot, [2.5, 97.5])
print(f"\n  Spearman correlation of onset with median error: {rho:+.3f}"
      f"   95% [{rlo:+.3f}, {rhi:+.3f}]"
      f"   {'excludes zero' if rlo > 0 or rhi < 0 else 'INCLUDES zero'}")

json.dump({"n_bearings": len(bearings), "n_in_band": len(inband),
           "median_below": med(below), "median_above": med(above),
           "observed_diff": obs, "bootstrap_ci": [float(lo), float(hi)],
           "permutation_p": pval, "spearman": rho,
           "spearman_ci": [float(rlo), float(rhi)]},
          io.open(os.path.join(RES, "bandbearing.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  wrote results/bearings/bandbearing.json")
