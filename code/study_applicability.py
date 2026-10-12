"""
When is a first-passage degradation model applicable at all?

A degradation-based prognostic needs an indicator that has *moved* before the
prediction is made. This script measures, for every bearing in both benchmarks,
the detected onset of degradation (first prediction time) and the fraction of
life over which the indicator is actually informative, and cross-tabulates that
against the standard prediction points. It is the diagnostic that explains the
accuracy reported in study_bearings.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

import datasets as ds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bearings")
os.makedirs(OUT, exist_ok=True)


def analyse(dataset):
    data = ds.load_dataset(dataset)
    rows = []
    for b, rec in sorted(data.items()):
        hi, n = rec["hi"], rec["n"]
        fpt = ds.first_prediction_time(hi)
        nh = min(40, max(5, n // 10))
        base = np.median(hi[:nh])
        mad = np.median(np.abs(hi[:nh] - base)) * 1.4826
        snr_end = (ds.terminal_hi(rec) - base) / max(mad, 1e-9)
        # fraction of the record spent in the detectable degradation phase
        deg_frac = (n - fpt) / n if fpt > 0 else np.nan
        # how much of the total indicator rise happens in the last 5% of life
        rise = ds.terminal_hi(rec) - base
        last5 = (ds.terminal_hi(rec) - np.median(hi[int(0.95 * n):int(0.96 * n) + 1])) \
            if n > 30 else np.nan
        rows.append({
            "dataset": dataset, "bearing": b, "n": n,
            "life_s": rec["life"], "fpt_idx": int(fpt),
            "fpt_frac": float(fpt / n) if fpt > 0 else None,
            "deg_phase_frac": float(deg_frac) if np.isfinite(deg_frac) else None,
            "terminal_snr_mad": float(snr_end),
            "rise": float(rise),
            "frac_rise_in_last_5pct": float(last5 / rise) if rise > 1e-6 and np.isfinite(last5) else None,
        })
    return rows


if __name__ == "__main__":
    allrows = []
    for dset in ("pronostia", "xjtu"):
        rows = analyse(dset)
        allrows += rows
        print(f"\n{dset.upper()}")
        print(f"  {'bearing':12s} {'n':>5s} {'FPT':>6s} {'FPT%':>6s} {'degphase%':>10s} "
              f"{'termSNR':>8s} {'rise in last 5%':>16s}")
        for r in rows:
            fp = f"{100*r['fpt_frac']:5.1f}" if r["fpt_frac"] is not None else " none"
            dp = f"{100*r['deg_phase_frac']:9.1f}" if r["deg_phase_frac"] is not None else "        -"
            fr = f"{100*r['frac_rise_in_last_5pct']:15.1f}" if r["frac_rise_in_last_5pct"] is not None else "              -"
            print(f"  {r['bearing']:12s} {r['n']:5d} {r['fpt_idx']:6d} {fp:>6s} {dp:>10s} "
                  f"{r['terminal_snr_mad']:8.1f} {fr:>16s}")
        det = [r for r in rows if r["fpt_frac"] is not None]
        print(f"  onset detected in {len(det)}/{len(rows)} bearings; "
              f"median onset at {100*np.median([r['fpt_frac'] for r in det]):.0f}% of life"
              if det else "  no onsets detected")
    with open(os.path.join(OUT, "applicability.json"), "w") as f:
        json.dump(allrows, f, indent=1)
    print("\nwrote results/bearings/applicability.json")
