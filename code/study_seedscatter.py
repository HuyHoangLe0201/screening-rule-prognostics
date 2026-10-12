# -*- coding: utf-8 -*-
r"""How many digits of the headline table are real?

The tables print MAE to four significant figures and PHM score to three
decimals.  Those come from a particle filter with a finite number of particles,
run once.  Nothing in the papers says how much of the last digit is the filter
rather than the model, and a difference between two rows is only evidence if it
is larger than that.

This re-runs the headline configuration with several independent seed streams
and reports, for every dataset, law and metric, the spread across streams.  Two
things are seeded: the per-prediction filter, and the threshold calibration that
runs the filter over each training record.  Both are re-drawn together, because
that is what a reader re-running the code with a different seed would see.

Everything else -- data, indicator, prediction points, particle count -- is held
fixed.  The output is therefore Monte Carlo noise in the published pipeline, not
variability of the data or of the protocol.

Read the resulting spread as the resolution of the tables: a gap between two
methods smaller than it is not a finding.
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_bearings as sb                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "bearings", "seedscatter.json")

LAWS = ("BS", "BS-t", "Wiener", "Gamma", "IGproc")
METRICS = ("MAE", "RMSE", "MAPE_pct", "PHM_score", "coverage95", "MPIW")
STREAMS = 4                    # the published run plus three independent ones

_base_seed = sb.stable_seed
_base_terminal = sb.latent_terminal


def install(offset):
    """Re-seed the whole pipeline, filter and threshold calibration alike."""
    sb.stable_seed = lambda *key: _base_seed(offset, *key)

    def terminal(rec, law, informed, coupling, mls, D_seed, seed=0, n_part=600):
        return _base_terminal(rec, law, informed, coupling, mls, D_seed,
                              seed=7 + 1000 * offset, n_part=n_part)
    sb.latent_terminal = terminal


results = {}
for dset in ("pronostia", "xjtu"):
    per_stream = []
    for offset in range(STREAMS):
        install(offset)
        t0 = time.time()
        rows = sb.run(dset, laws=LAWS)
        per_stream.append(sb.summarise(rows))          # keyed by law name
        print(f"  {dset:10s} stream {offset}  ({time.time()-t0:.0f} s)", flush=True)

    results[dset] = {}
    for law in LAWS:
        results[dset][law] = {}
        for m in METRICS:
            v = np.array([s[law][m] for s in per_stream], float)
            centre = float(np.mean(v))
            results[dset][law][m] = {
                "values": [float(x) for x in v],
                "mean": centre,
                "sd": float(np.std(v, ddof=1)),
                "range": float(v.max() - v.min()),
                "range_pct": float(100.0 * (v.max() - v.min()) / abs(centre))
                if centre else float("nan"),
            }

io.open(OUT, "w", encoding="utf-8", newline="").write(
    json.dumps(results, indent=1))

print("\n  spread across independent seed streams, as a percentage of the mean\n")
print(f"  {'dataset':10s} {'law':10s} " + " ".join(f"{m:>11s}" for m in METRICS))
worst = 0.0
for dset in results:
    for law in LAWS:
        cells = []
        for m in METRICS:
            p = results[dset][law][m]["range_pct"]
            worst = max(worst, p if np.isfinite(p) else 0.0)
            cells.append(f"{p:10.1f}%")
        print(f"  {dset:10s} {law:10s} " + " ".join(cells))

print(f"\n  widest spread of any published statistic: {worst:.1f}% of its value")
print(f"  wrote {os.path.relpath(OUT, ROOT)}")
