# -*- coding: utf-8 -*-
r"""The pre-registered test of PREREG_p3_step.md: does prediction worsen across
the threshold, on C-MAPSS units whose remaining life was never predicted?

    python code/study_p3_step.py run       draw the cohort, predict every unit
    python code/study_p3_step.py summary   D1-D4 and the descriptive comparator

Every constant is the protocol's; the script refuses to run on any other
version of it.  Reference sets ("the other units of the same subset") are all
eligible units of that subset other than the one predicted; the band-study
exclusion decides only which units are predicted.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PREREG = os.path.join(HERE, "PREREG_p3_step.md")
FROZEN_SHA256 = "96b1b6d95c91f35e6cecdf5a8de4c7ba0bcf26f47e4e620f4013f79051901e2d"
FROZEN_AT = "2026-10-15T10:11:44+07:00"
BE = os.path.join(ROOT, "results", "bearings")
ROWS = os.path.join(BE, "p3_step_rows.json")
OUT = os.path.join(BE, "p3_step.json")
BINS = [(0.00, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
POINTS = (0.50, 0.70, 0.85)
SEED, B_BOOT, PER_BIN = 20261020, 10_000, 20


def check_frozen():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {h}")
    return h


def stable_seed(*key):
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def _features(hi, k):
    w30 = hi[max(0, k - 29):k + 1]
    w10 = hi[max(0, k - 9):k + 1]
    s30 = np.polyfit(np.arange(len(w30)), w30, 1)[0]
    s10 = np.polyfit(np.arange(len(w10)), w10, 1)[0]
    return [float(k), float(hi[k]), float(np.mean(hi[max(0, k - 4):k + 1])),
            float(s10), float(s30), float(np.max(hi[:k + 1]))]


def run():
    check_frozen()
    import study_bearings as sb
    import study_p3_review as R
    import study_thirdset as ts
    from sklearn.ensemble import GradientBoostingRegressor

    used = set()
    for f in ("bandtest.json", "bandtest_rep.json"):
        used |= {r["unit"] for r in json.load(open(os.path.join(BE, f)))["rows"]}
    units = {}
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", True), ("FD004", True)):
        for u, arr in ts.load(fd, by_regime=by).items():
            hi = np.asarray(ts.indicator(arr), float)
            k = ts.onset(hi)
            if k is None or len(hi) < 60:
                continue
            n = len(hi)
            units[f"{fd}:{u}"] = {"subset": fd, "hi": hi, "n": n, "onset": int(k),
                                  "frac": k / n}
    pool = sorted(k for k in units if k not in used)
    rng = np.random.default_rng(SEED)
    picked = []
    for lo, hi_ in BINS:
        cand = sorted(k for k in pool if lo <= units[k]["frac"] < hi_)
        take = [str(x) for x in rng.permutation(cand)[:PER_BIN]] if cand else []
        picked += take
        print(f"  onset {lo:.0%}-{hi_:.0%}: {len(cand)} available, {len(take)} drawn", flush=True)

    #  gradient boosting: five folds by unit within each subset
    gbr = {}
    rf = np.random.default_rng(SEED)
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        keys = sorted(k for k in units if units[k]["subset"] == fd)
        fold = {k: int(i) % 5 for k, i in zip(keys, rf.permutation(len(keys)))}
        for f in range(5):
            test = [k for k in picked if units[k]["subset"] == fd and fold[k] == f]
            if not test:
                continue
            X, y = [], []
            for k in keys:
                if fold[k] == f:
                    continue
                hi, n = units[k]["hi"], units[k]["n"]
                for j in range(20, n - 1, 5):
                    X.append(_features(hi, j))
                    y.append(np.log((n - 1) - j))
            models = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200,
                                                   max_depth=3, learning_rate=0.05,
                                                   random_state=0).fit(X, y)
                      for a in (0.5, 0.025, 0.975)}
            for k in test:
                hi, n = units[k]["hi"], units[k]["n"]
                for p in POINTS:
                    j = int(p * n)
                    if j < 20 or j >= n - 1:
                        continue
                    x = [_features(hi, j)]
                    gbr[(k, j)] = tuple(float(np.exp(models[a].predict(x)[0]))
                                        for a in (0.5, 0.025, 0.975))
        print(f"  gradient boosting fitted on {fd}", flush=True)

    rows = []
    for key in picked:
        u = units[key]
        ref = [k for k in units if units[k]["subset"] == u["subset"] and k != key]
        D = float(np.median([np.median(units[k]["hi"][-3:]) for k in ref]))
        med_life = float(np.median([max(units[k]["n"] - units[k]["onset"], 20) for k in ref]))
        lives = [units[k]["n"] - 1 for k in ref]
        n = u["n"]
        rec = {"t": np.arange(n, dtype=float), "hi": u["hi"], "condition": 0,
               "life": float(n - 1), "n": n, "dt": 1.0}
        for p in POINTS:
            j = int(p * n)
            if j < 20 or j >= n - 1:
                continue
            true = float((n - 1) - j)
            r = sb.predict_one(rec, j, D, "BS", informed=True, coupling=True,
                               med_life_steps=med_life, seed=stable_seed(key, j), n_part=600)
            lt = R._life_table(lives, float(j))
            g = gbr[(key, j)]
            row = {"unit": key, "subset": u["subset"], "onset_frac": u["frac"], "n": n,
                   "k": j, "at_life": p, "true": true,
                   "lt_pred": lt[0], "lt_lo": lt[1], "lt_hi": lt[2],
                   "gb_pred": g[0], "gb_lo": g[1], "gb_hi": g[2]}
            if r is not None:
                row.update({"pred": r["pred"], "lo": r["lo"], "hi": r["hi"],
                            "capped": r["capped"]})
            rows.append(row)
        print(f"  {key} onset {u['frac']:.2f}: {len(rows)} predictions so far", flush=True)
    json.dump(rows, open(ROWS, "w"), indent=1)


def _ape(r, k="pred"):
    return 100.0 * abs(r[k] - r["true"]) / r["true"]


def _cov(r, lo="lo", hi="hi"):
    return float(r[lo] <= r["true"] <= r[hi])


def summary():
    h = check_frozen()
    from scipy.stats import spearmanr
    rows = [r for r in json.load(open(ROWS)) if "pred" in r]
    by = {}
    for r in rows:
        by.setdefault(r["unit"], []).append(r)
    late = sorted(u for u in by if by[u][0]["onset_frac"] >= 0.70)
    early = sorted(u for u in by if by[u][0]["onset_frac"] < 0.70)
    rng = np.random.default_rng(SEED)

    def flat(us):
        return [r for u in us for r in by[u]]

    def stats(L, E):
        fl, fe = flat(L), flat(E)
        return (np.median([_ape(r) for r in fl]) - np.median([_ape(r) for r in fe]),
                np.mean([_cov(r) for r in fl]) - np.mean([_cov(r) for r in fe]),
                np.median([_ape(r) for r in fl]) - np.median([_ape(r, "lt_pred") for r in fl]))

    obs = stats(late, early)
    bt = np.array([stats([late[i] for i in rng.integers(0, len(late), len(late))],
                         [early[i] for i in rng.integers(0, len(early), len(early))])
                   for _ in range(B_BOOT)])
    ci = [np.percentile(bt[:, j], [2.5, 97.5]).tolist() for j in range(3)]
    us = sorted(by)
    on = np.array([by[u][0]["onset_frac"] for u in us])
    er = np.array([np.mean([_ape(r) for r in by[u]]) for u in us])
    rho = float(spearmanr(on, er).correlation)
    rb = []
    for _ in range(B_BOOT):
        i = rng.integers(0, len(us), len(us))
        if np.ptp(on[i]) > 0:
            rb.append(spearmanr(on[i], er[i]).correlation)
    rci = np.percentile(rb, [2.5, 97.5]).tolist()

    def group(us_):
        f = flat(us_)
        return {"units": len(us_), "predictions": len(f),
                "medape_path": float(np.median([_ape(r) for r in f])),
                "coverage_path": float(np.mean([_cov(r) for r in f])),
                "medape_lt": float(np.median([_ape(r, "lt_pred") for r in f])),
                "coverage_lt": float(np.mean([_cov(r, "lt_lo", "lt_hi") for r in f])),
                "medape_gb": float(np.median([_ape(r, "gb_pred") for r in f])),
                "coverage_gb": float(np.mean([_cov(r, "gb_lo", "gb_hi") for r in f]))}
    res = {"protocol_sha256": h, "protocol_frozen_at": FROZEN_AT,
           "late": group(late), "early": group(early),
           "by_bin": {f"{lo:.2f}-{hi_:.2f}": group([u for u in us if lo <= by[u][0]["onset_frac"] < hi_])
                      for lo, hi_ in BINS if any(lo <= by[u][0]["onset_frac"] < hi_ for u in us)},
           "D1": {"diff": float(obs[0]), "ci": ci[0], "holds": bool(ci[0][0] > 0)},
           "D2": {"rho": rho, "ci": rci, "holds": bool(rci[0] > 0)},
           "D3": {"diff": float(obs[1]), "ci": ci[1], "holds": bool(ci[1][1] < 0)},
           "D4": {"diff": float(obs[2]), "ci": ci[2], "holds": bool(not ci[2][1] < 0)}}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for g in ("early", "late"):
        v = res[g]
        print(f"{g:6s} {v['units']:3d} units  path {v['medape_path']:.1f}/{v['coverage_path']:.2f}  "
              f"LT {v['medape_lt']:.1f}/{v['coverage_lt']:.2f}  GB {v['medape_gb']:.1f}/{v['coverage_gb']:.2f}")
    for b, v in res["by_bin"].items():
        print(f"  {b}: {v['units']} units, path {v['medape_path']:.1f}/{v['coverage_path']:.2f}, "
              f"LT {v['medape_lt']:.1f}, GB {v['medape_gb']:.1f}")
    for d in ("D1", "D2", "D3", "D4"):
        print(d, res[d])


if __name__ == "__main__":
    {"run": run, "summary": summary}.get(sys.argv[1] if len(sys.argv) > 1 else "",
                                         lambda: sys.exit(__doc__))()
