# -*- coding: utf-8 -*-
r"""The pre-registered test of PREREG_p3_rul.md: does a declined cohort predict
worse?

The XJTU cells' charging regimes split into one cohort the screen declines (3C)
and four it passes.  This script predicts the remaining life of every cell at
50, 70 and 85 % of its life, with the path model and with a life table, and
scores the three predictions of the protocol.  Every constant is the protocol's;
the script refuses to run on any other version of it.

    python code/study_p3_rul.py           all regimes, then the summary
    python code/study_p3_rul.py Batch-2   one regime (for parallel runs)
    python code/study_p3_rul.py summary   the summary only
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_bearings as sb                                      # noqa: E402
import study_p3_prereg as P                                      # noqa: E402
import study_p3_review as R                                      # noqa: E402

PREREG = os.path.join(HERE, "PREREG_p3_rul.md")
FROZEN_SHA256 = "5fb38dca8accc171ca474c2731251cb118bf639753f2e4281e331a6acc3aa8aa"
FROZEN_AT = "2026-10-14T21:58:27+07:00"
BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_rul.json")
DECLINED = ("Batch-2",)
PASSED = ("Batch-1", "Batch-3", "Batch-4", "Batch-6")
UNDECIDED = ("Batch-5",)
POINTS = (0.50, 0.70, 0.85)
B_BOOT, SEED = 10_000, 20261016


def check_frozen():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {h}")
    return h


def stable_seed(*key):
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def load_regime(batch):
    """Every cell of one regime: indicator, life, and the protocol's onset."""
    onset = {r["cell"]: r for r in json.load(open(os.path.join(BE, "p3_prereg.json"),
                                                  encoding="utf-8"))["battery"]["records"]}
    cells = {}
    for f in sorted(glob.glob(os.path.join(P.BAT, batch, "*.mat"))):
        cell = os.path.basename(f)[:-4]
        cap = P.battery_capacity(f, batch)
        hi = cap[0] - cap
        n = len(hi)
        o = onset[cell]
        assert o["n"] == n, (cell, o["n"], n)
        idx = o["onset_idx"] if o["onset_frac"] is not None else n
        cells[cell] = {"t": np.arange(n, dtype=float), "hi": hi, "condition": 0,
                       "life": float(n - 1), "n": n, "dt": 1.0, "onset_idx": idx}
    return cells


def run_regime(batch):
    check_frozen()
    cells = load_regime(batch)
    rows = []
    for cell, rec in cells.items():
        ref = [c for c in cells if c != cell]
        D = float(np.median([np.median(cells[c]["hi"][-3:]) for c in ref]))
        med_life = float(np.median([max(cells[c]["n"] - cells[c]["onset_idx"], 20)
                                    for c in ref]))
        lives = [cells[c]["n"] - 1 for c in ref]
        n = rec["n"]
        for f in POINTS:
            k = int(f * n)
            if k < 20 or k >= n - 1:
                continue
            true = float((n - 1) - k)
            r = sb.predict_one(rec, k, D, "BS", informed=True, coupling=True,
                               med_life_steps=med_life, seed=stable_seed(cell, k),
                               n_part=600)
            lt = R._life_table(lives, float(k))
            row = {"cell": cell, "batch": batch, "n": n, "k": k, "at_life": f,
                   "onset_frac": rec["onset_idx"] / n, "true": true, "D": D,
                   "med_life": med_life,
                   "lt_pred": lt[0], "lt_lo": lt[1], "lt_hi": lt[2]}
            if r is not None:
                row.update({"pred": r["pred"], "lo": r["lo"], "hi": r["hi"],
                            "capped": r["capped"]})
            rows.append(row)
            print(f"  {batch} {cell} k={k} true={true:.0f} "
                  f"path={row.get('pred', float('nan')):.0f} lt={lt[0]:.0f}", flush=True)
    json.dump(rows, open(os.path.join(BE, f"p3_rul_part_{batch}.json"), "w"), indent=1)


def _ape(r, key="pred"):
    return 100.0 * abs(r[key] - r["true"]) / r["true"]


def _cov(r, lo="lo", hi="hi"):
    return float(r[lo] <= r["true"] <= r[hi])


def _cohort(rows):
    """Median APE and coverage of both predictors over a set of predictions."""
    ok = [r for r in rows if "pred" in r]
    return {"cells": len({r["cell"] for r in ok}), "predictions": len(ok),
            "medape_path": float(np.median([_ape(r) for r in ok])),
            "coverage_path": float(np.mean([_cov(r) for r in ok])),
            "medape_lt": float(np.median([_ape(r, "lt_pred") for r in ok])),
            "coverage_lt": float(np.mean([_cov(r, "lt_lo", "lt_hi") for r in ok])),
            "capped_share": float(np.mean([r["capped"] for r in ok]))}


def summary():
    h = check_frozen()
    rows = []
    for f in sorted(glob.glob(os.path.join(BE, "p3_rul_part_Batch-*.json"))):
        rows += json.load(open(f))
    by_cell = {}
    for r in rows:
        if "pred" in r:
            by_cell.setdefault(r["cell"], []).append(r)
    dec = sorted(c for c, v in by_cell.items() if v[0]["batch"] in DECLINED)
    pas = sorted(c for c, v in by_cell.items() if v[0]["batch"] in PASSED)

    def flat(cs):
        return [r for c in cs for r in by_cell[c]]

    def stats(d_cells, p_cells):
        d, p = flat(d_cells), flat(p_cells)
        return (np.median([_ape(r) for r in d]) - np.median([_ape(r) for r in p]),
                np.mean([_cov(r) for r in d]) - np.mean([_cov(r) for r in p]),
                np.median([_ape(r) for r in d]) - np.median([_ape(r, "lt_pred") for r in d]))

    obs = stats(dec, pas)
    rng = np.random.default_rng(SEED)
    boot = np.array([stats([dec[i] for i in rng.integers(0, len(dec), len(dec))],
                           [pas[i] for i in rng.integers(0, len(pas), len(pas))])
                     for _ in range(B_BOOT)])
    ci = [np.percentile(boot[:, j], [2.5, 97.5]).tolist() for j in range(3)]
    res = {"protocol_sha256": h, "protocol_frozen_at": FROZEN_AT,
           "declined": _cohort(flat(dec)), "passed": _cohort(flat(pas)),
           "by_regime": {b: _cohort([r for r in rows if r["batch"] == b])
                         for b in sorted({r["batch"] for r in rows})},
           "diff_medape_path": {"obs": float(obs[0]), "ci": ci[0]},
           "diff_coverage_path": {"obs": float(obs[1]), "ci": ci[1]},
           "diff_path_minus_lt_declined": {"obs": float(obs[2]), "ci": ci[2]},
           "rows": rows}
    res["R1"] = {"holds": bool(ci[0][0] > 0)}
    res["R2"] = {"holds": bool(ci[1][1] < 0)}
    res["R3"] = {"holds": bool(not ci[2][1] < 0)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for k in ("declined", "passed"):
        c = res[k]
        print(f"{k:9s} {c['cells']:3d} cells {c['predictions']:3d} preds  path MedAPE "
              f"{c['medape_path']:6.1f} cov {c['coverage_path']:.2f}   life table MedAPE "
              f"{c['medape_lt']:6.1f} cov {c['coverage_lt']:.2f}")
    for b, c in res["by_regime"].items():
        print(f"  {b}: {c['cells']} cells, path {c['medape_path']:.1f} / {c['coverage_path']:.2f},"
              f" life table {c['medape_lt']:.1f} / {c['coverage_lt']:.2f}")
    print(f"R1 diff MedAPE {obs[0]:+.1f} {np.round(ci[0], 1)} -> {res['R1']['holds']}")
    print(f"R2 diff coverage {obs[1]:+.2f} {np.round(ci[1], 2)} -> {res['R2']['holds']}")
    print(f"R3 path - LT on declined {obs[2]:+.1f} {np.round(ci[2], 1)} -> {res['R3']['holds']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["summary"]:
        summary()
    elif args:
        for b in args:
            run_regime(b)
    else:
        for b in DECLINED + PASSED + UNDECIDED:
            run_regime(b)
        summary()
