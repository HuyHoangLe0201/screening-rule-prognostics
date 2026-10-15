# -*- coding: utf-8 -*-
r"""A health-information comparator on the cohorts of the third and fourth
pre-registered tests.  Exploratory: computed after those tests were scored.

Gradient boosting on causal features of the indicator, with the features,
model and folds that PREREG_p3_step.md fixes for its own comparator, predicts
each unit at the points the path model and the life table were scored on.  If
it also loses to the life table on a cohort, the indicator carries little
information there; if only the path model loses, the failure is the model's.

    python code/study_p3_gbr.py cells Batch-2   one battery regime (slow: .mat files)
    python code/study_p3_gbr.py blind           the N-CMAPSS subsets of the blind test
    python code/study_p3_gbr.py summary
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from study_p3_step import _features                              # noqa: E402

BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_gbr.json")
SEED = 20261020


def _fit_predict(series, points):
    """series: unit -> indicator array; points: unit -> list of k.
    Five folds by unit; returns (unit, k) -> (median, lo, hi)."""
    from sklearn.ensemble import GradientBoostingRegressor
    keys = sorted(series)
    fold = {k: int(i) % 5 for k, i in zip(keys, np.random.default_rng(SEED).permutation(len(keys)))}
    out = {}
    for f in range(5):
        test = [k for k in keys if fold[k] == f and points.get(k)]
        if not test:
            continue
        X, y = [], []
        for k in keys:
            if fold[k] == f:
                continue
            hi = series[k]
            n = len(hi)
            for j in range(20, n - 1, 5):
                X.append(_features(hi, j))
                y.append(np.log((n - 1) - j))
        if len(X) < 10:
            continue
        models = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200,
                                               max_depth=3, learning_rate=0.05,
                                               random_state=0).fit(X, y)
                  for a in (0.5, 0.025, 0.975)}
        for k in test:
            for j in points[k]:
                x = [_features(series[k], j)]
                out[(k, j)] = tuple(float(np.exp(models[a].predict(x)[0])) for a in (0.5, 0.025, 0.975))
    return out


def _score(rows):
    ok = [r for r in rows if "gb_pred" in r and "pred" in r]
    ape = lambda r, k: 100.0 * abs(r[k] - r["true"]) / r["true"]   # noqa: E731
    cov = lambda r, a, b: float(r[a] <= r["true"] <= r[b])          # noqa: E731
    return {"units": len({r["unit"] for r in ok}), "predictions": len(ok),
            "medape_path": float(np.median([ape(r, "pred") for r in ok])),
            "medape_lt": float(np.median([ape(r, "lt_pred") for r in ok])),
            "medape_gb": float(np.median([ape(r, "gb_pred") for r in ok])),
            "coverage_path": float(np.mean([cov(r, "lo", "hi") for r in ok])),
            "coverage_lt": float(np.mean([cov(r, "lt_lo", "lt_hi") for r in ok])),
            "coverage_gb": float(np.mean([cov(r, "gb_lo", "gb_hi") for r in ok]))}


def cells(batch):
    import study_p3_rul as RU
    T = json.load(open(os.path.join(BE, "p3_rul.json"), encoding="utf-8"))
    rows = [dict(r, unit=r["cell"]) for r in T["rows"] if r["batch"] == batch]
    cs = RU.load_regime(batch)
    series = {c: np.asarray(v["hi"], float) for c, v in cs.items()}
    points = {}
    for r in rows:
        points.setdefault(r["unit"], []).append(r["k"])
    g = _fit_predict(series, points)
    for r in rows:
        if (r["unit"], r["k"]) in g:
            r["gb_pred"], r["gb_lo"], r["gb_hi"] = g[(r["unit"], r["k"])]
    json.dump(rows, open(os.path.join(BE, f"p3_gbr_part_{batch}.json"), "w"), indent=1)
    print(batch, _score(rows))


def blind():
    data = json.load(open(os.path.join(BE, "p3_blind_indicators.json")))
    S = json.load(open(os.path.join(BE, "p3_blind_rows.json")))
    rows = S["rows"]
    for tag, units in data.items():
        series = {f"{u['file']}-{u['split']}-{u['unit']}": np.asarray(u["hi"], float) for u in units}
        points = {}
        for r in rows:
            if r["cohort"] == tag:
                points.setdefault(r["unit"], []).append(r["k"])
        g = _fit_predict(series, points)
        for r in rows:
            if r["cohort"] == tag and (r["unit"], r["k"]) in g:
                r["gb_pred"], r["gb_lo"], r["gb_hi"] = g[(r["unit"], r["k"])]
        print(tag, _score([r for r in rows if r["cohort"] == tag]), flush=True)
    json.dump(rows, open(os.path.join(BE, "p3_gbr_part_blind.json"), "w"), indent=1)


def summary():
    res = {"note": "exploratory: computed after the pre-registered tests were scored"}
    cell_rows = []
    for b in ("Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5", "Batch-6"):
        p = os.path.join(BE, f"p3_gbr_part_{b}.json")
        if os.path.exists(p):
            rs = json.load(open(p))
            cell_rows += rs
            res[b] = _score(rs)
    res["cells_declined"] = _score([r for r in cell_rows if r["batch"] == "Batch-2"])
    res["cells_passed"] = _score([r for r in cell_rows if r["batch"] in
                                  ("Batch-1", "Batch-3", "Batch-4", "Batch-6")])
    br = json.load(open(os.path.join(BE, "p3_gbr_part_blind.json")))
    res["blind"] = _score(br)
    res["blind_by_cohort"] = {c: _score([r for r in br if r["cohort"] == c])
                              for c in sorted({r["cohort"] for r in br})}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "cells":
        cells(a[1])
    elif a == ["blind"]:
        blind()
    elif a == ["summary"]:
        summary()
    else:
        sys.exit(__doc__)
