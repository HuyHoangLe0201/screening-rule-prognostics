"""Out-of-sample test of the screening rule on a third, held-back archive.

Paper 3's screening rule was built while looking at PRONOSTIA and XJTU-SY and
then demonstrated on the same two, which is the limitation the paper raises
first against itself.  This applies it, unchanged, to an archive that played no
part in its construction: the NASA C-MAPSS turbofan run-to-failure data.

Why C-MAPSS rather than a third bearing set.  The rule is claimed to be about
degradation-threshold prognostics in general, not about bearings, and both
existing archives are accelerated BEARING tests -- so a third bearing rig would
test the constants but not the claim.  C-MAPSS is a different asset class, a
different failure mechanism and a different sensor suite, and it is widely used
precisely because degradation IS observable in it.  That makes it a genuine
adversarial test: if the rule is measuring something real it should PASS
C-MAPSS, having failed PRONOSTIA.

What is held fixed.  The detection band, the persistence run and the sustained
departure requirement keep the values used throughout the paper.  Nothing here
is tuned; the only new choice is how to build a health indicator from a sensor
suite the paper has never seen, and that follows the paper's own construction --
robust z-scores of each channel against the unit's own healthy window.

    python code/study_thirdset.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.environ.get("P3_DATA", os.path.join(HERE, "..", "data")), "CMAPSSData")
OUT = os.path.join(HERE, "..", "results", "bearings")
os.makedirs(OUT, exist_ok=True)

#  The 14 sensors that carry trend in C-MAPSS.  This is the standard published
#  subset, not a selection made here: the other seven are constant or pure noise
#  in FD001 and would only add variance to the indicator.
TREND = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

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


def load(fd, by_regime=False):
    """{unit: (n_cycles, 14-channel array)} from one C-MAPSS training file.

    With by_regime=True the sensors are first standardised WITHIN each operating
    regime, the standard repair for the six-condition subsets: the three
    operating settings take six discrete combinations, and without removing them
    the healthy window measures regime switching rather than noise.
    """
    path = os.path.join(SRC, f"train_{fd}.txt")
    units, settings = {}, {}
    for line in io.open(path, encoding="utf-8"):
        p = line.split()
        if not p:
            continue
        u = int(p[0])
        #  columns: unit, cycle, 3 settings, then sensors 1..21
        units.setdefault(u, []).append([float(p[4 + s]) for s in TREND])
        settings.setdefault(u, []).append([round(float(p[2 + i]), 0)
                                           for i in range(3)])
    units = {u: np.asarray(v, float) for u, v in units.items()}
    if not by_regime:
        return units

    #  pool every unit to estimate each regime's own centre and scale
    regs = {u: [tuple(r) for r in settings[u]] for u in settings}
    pool = {}
    for u, arr in units.items():
        for row, r in zip(arr, regs[u]):
            pool.setdefault(r, []).append(row)
    stat = {r: (np.median(np.asarray(v), axis=0),
                np.median(np.abs(np.asarray(v) - np.median(np.asarray(v), axis=0)),
                          axis=0) * 1.4826)
            for r, v in pool.items()}
    out = {}
    for u, arr in units.items():
        z = np.empty_like(arr)
        for i, r in enumerate(regs[u]):
            med, mad = stat[r]
            z[i] = (arr[i] - med) / np.where(mad > 1e-9, mad, 1.0)
        out[u] = z
    return out


def indicator(arr):
    """One health indicator per unit, by the paper's own construction.

    Robust z-score of every channel against that unit's own healthy window, sign
    aligned so that departure is positive, then averaged.  Channels that do not
    move in the healthy window carry no weight.
    """
    n = len(arr)
    nh = min(N_HEALTH, max(5, n // 10))
    base = np.median(arr[:nh], axis=0)
    mad = np.median(np.abs(arr[:nh] - base), axis=0) * 1.4826
    ok = mad > 1e-9
    z = (arr[:, ok] - base[ok]) / mad[ok]
    #  align each channel to the direction it drifts over the record, so that a
    #  falling sensor counts as departure rather than cancelling a rising one
    drift = np.median(z[-nh:], axis=0) - np.median(z[:nh], axis=0)
    sign = np.where(drift >= 0, 1.0, -1.0)
    return (z * sign).mean(axis=1)


#  FD002/FD004 are repeated with the regime repair, as the marginal test: the
#  raw form fails because the indicator is invalid, and what the rule says once
#  the indicator is fixed is a different question.
RUNS = [("FD001", False), ("FD002", False), ("FD003", False), ("FD004", False),
        ("FD002", True), ("FD004", True)]

out = {}
for fd, by_regime in RUNS:
    units = load(fd, by_regime=by_regime)
    key = fd + ("+regime" if by_regime else "")
    pos, ndet = [], 0
    for u, arr in sorted(units.items()):
        hi = indicator(arr)
        i = onset(hi)
        if i > 0:
            ndet += 1
            pos.append(100.0 * i / len(hi))
    #  Why a subset can fail: with several operating conditions the healthy
    #  window contains regime switches, so its robust scale measures the
    #  switching rather than the noise and no departure can clear the band.
    mads = []
    for arr in list(units.values()):
        nh = min(N_HEALTH, max(5, len(arr) // 10))
        base = np.median(arr[:nh], axis=0)
        mad = np.median(np.abs(arr[:nh] - base), axis=0) * 1.4826
        mads.append(float(np.median(mad[mad > 1e-9])))

    out[key] = {
        "n_units": len(units),
        "median_healthy_mad": float(np.median(mads)),
        "n_detected": ndet,
        "median_onset_pct": st.median(pos) if pos else None,
        "median_deg_phase_pct": (100.0 - st.median(pos)) if pos else None,
        "q1_onset_pct": st.quantiles(pos, n=4)[0] if len(pos) > 3 else None,
        "q3_onset_pct": st.quantiles(pos, n=4)[2] if len(pos) > 3 else None,
    }

with open(os.path.join(OUT, "thirdset.json"), "w") as f:
    json.dump(out, f, indent=1)

print("C-MAPSS, screening rule applied with the paper's constants unchanged")
print(f"  (band {THR} robust s.d., persistence {PERSIST}, sustained {FRAC})\n")
print(f"{'subset':<8}{'units':>7}{'detected':>10}{'median onset %':>16}"
      f"{'deg phase %':>13}")
for fd, r in out.items():
    m, d = r["median_onset_pct"], r["median_deg_phase_pct"]
    print(f"{fd:<8}{r['n_units']:>7}{r['n_detected']:>10}"
          f"{(f'{m:.1f}' if m is not None else '--'):>16}"
          f"{(f'{d:.1f}' if d is not None else '--'):>13}")

print("\nfor comparison, the two archives the rule was built on:")
print(f"{'PRONOSTIA':<8}{17:>7}{12:>10}{'96.6':>16}{'3.4':>13}")
print(f"{'XJTU-SY':<8}{15:>7}{15:>10}{'40.7':>16}{'59.3':>13}")
print("\nwrote results/bearings/thirdset.json")
