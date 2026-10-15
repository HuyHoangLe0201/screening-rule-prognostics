# -*- coding: utf-8 -*-
r"""The pre-registered external test of PREREG_p3_filter.md: the screen on
filters, an asset class that took no part in building it.

    python code/study_p3_filter.py run        screen every cohort, predict every run
    python code/study_p3_filter.py summary    V1, F1-F3

Every constant is the protocol's; the script refuses to run on any other
version of it.  The data file is read from $P3_DATA/filter_similar_system/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PREREG = os.path.join(HERE, "PREREG_p3_filter.md")
FROZEN_SHA256 = "4e75cf4f3d39027efee6d1b247fb11d5107a872e4f61c5d447fdf70e82635442"
FROZEN_AT = "2026-10-15T12:51:29+07:00"
DATA = os.environ.get("P3_DATA", os.path.join(ROOT, "data"))
CSV = os.path.join(DATA, "filter_similar_system", "data.csv")
BE = os.path.join(ROOT, "results", "bearings")
ROWS = os.path.join(BE, "p3_filter_rows.json")
OUT = os.path.join(BE, "p3_filter.json")
POINTS = (0.50, 0.70, 0.85)
SEED, B_BOOT, GB_SEED = 20261022, 10_000, 20261020


def check_frozen():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {h}")
    return h


def stable_seed(*key):
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def load_runs():
    """Runs keyed by Data_No: time, differential pressure, RUL, orifice."""
    import pandas as pd
    df = pd.read_csv(CSV)
    runs = {}
    for no, g in df.groupby("Data_No"):
        g = g.sort_values("Time")
        t = g["Time"].to_numpy(float)
        dp = g["Differential_Pressure"].to_numpy(float)
        rul = g["RUL"].to_numpy(float)
        n = len(t)
        if n < 60:
            continue
        if n > 5000:                                     # the protocol's averaging
            b = math.ceil(n / 1000)
            m = n // b
            t = t[:m * b].reshape(m, b).mean(axis=1)
            dp = dp[:m * b].reshape(m, b).mean(axis=1)
            rul = rul[:m * b].reshape(m, b).mean(axis=1)
            n = m
        nh = min(40, max(5, n // 10))
        h = dp - np.median(dp[:nh])
        runs[int(no)] = {"cohort": float(g["Orifice"].iloc[0]), "t": t, "h": h,
                         "rul": rul, "n": n, "life": float(t[-1] + rul[-1]),
                         "dt": float(np.median(np.diff(t)))}
    return runs


def _screen(fracs, rng):
    import study_p3_review as R
    x = np.array(fracs, float)
    x = np.where(x >= 1.0, np.inf, x)
    rate = float(np.isfinite(x).mean())
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    lo, hi = np.quantile(np.median(x[idx], axis=1), [0.025, 0.975], method="inverted_cdf")

    def pct(z):
        return None if not np.isfinite(z) else 100 * float(z)
    verdict = "check indicator" if rate < 0.25 else R._verdict3(pct(lo), pct(hi), 70.0)
    return {"n": len(x), "rate": rate, "censored_median_pct": pct(np.median(x)),
            "ci": [pct(lo), pct(hi)], "verdict": verdict}


def _gbr(runs, keys_by_cohort):
    """The step protocol's comparator: features of h at index j, log RUL target."""
    from sklearn.ensemble import GradientBoostingRegressor
    from study_p3_step import _features
    out = {}
    for c, keys in keys_by_cohort.items():
        keys = sorted(keys)
        fold = {k: int(i) % 5 for k, i in zip(keys, np.random.default_rng(GB_SEED).permutation(len(keys)))}
        for f in range(5):
            test = [k for k in keys if fold[k] == f]
            if not test:
                continue
            X, y = [], []
            for k in keys:
                if fold[k] == f:
                    continue
                r = runs[k]
                for j in range(20, r["n"] - 1, 5):
                    if r["rul"][j] > 0:
                        X.append(_features(r["h"], j))
                        y.append(np.log(r["rul"][j]))
            models = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200,
                                                   max_depth=3, learning_rate=0.05,
                                                   random_state=0).fit(X, y)
                      for a in (0.5, 0.025, 0.975)}
            for k in test:
                r = runs[k]
                for p in POINTS:
                    j = int(p * r["n"])
                    if j < 20 or j >= r["n"] - 1:
                        continue
                    x = [_features(r["h"], j)]
                    out[(k, j)] = tuple(float(np.exp(models[a].predict(x)[0])) for a in (0.5, 0.025, 0.975))
    return out


def run():
    check_frozen()
    import study_bearings as sb
    import study_p3_prereg as P
    import study_p3_review as R
    runs = load_runs()
    for k, r in runs.items():
        i = P.NS["onset"](r["h"])
        r["onset_idx"] = int(i) if i > 0 else r["n"]
        r["onset_frac"] = i / r["n"] if i > 0 else 1.0
    cohorts = {}
    for k, r in runs.items():
        cohorts.setdefault(r["cohort"], []).append(k)
    rng = np.random.default_rng(SEED)
    screen = {str(c): _screen([runs[k]["onset_frac"] for k in sorted(ks)], rng)
              for c, ks in sorted(cohorts.items())}
    screen["pooled"] = _screen([runs[k]["onset_frac"] for k in sorted(runs)], rng)
    for c, s in screen.items():
        print(f"  orifice {c}: {s['n']} runs, onset {s['censored_median_pct']} {s['ci']} -> {s['verdict']}",
              flush=True)
    gb = _gbr(runs, cohorts)
    rows = []
    for c, ks in sorted(cohorts.items()):
        for k in sorted(ks):
            r = runs[k]
            ref = [j for j in ks if j != k]
            D = float(np.median([np.median(runs[j]["h"][-3:]) for j in ref]))
            med = float(np.median([max(runs[j]["n"] - runs[j]["onset_idx"], 20) for j in ref]))
            lives = [runs[j]["life"] for j in ref]
            rec = {"t": r["t"], "hi": r["h"], "condition": 0, "life": r["life"],
                   "n": r["n"], "dt": r["dt"]}
            for p in POINTS:
                j = int(p * r["n"])
                if j < 20 or j >= r["n"] - 1:
                    continue
                true = float(r["rul"][j])
                if true <= 0:
                    continue
                pr = sb.predict_one(rec, j, D, "BS", informed=True, coupling=True,
                                    med_life_steps=med, seed=stable_seed(k, j), n_part=600)
                lt = R._life_table(lives, float(r["t"][j]))
                g = gb.get((k, j))
                row = {"run": k, "cohort": c, "n": r["n"], "k": j, "at_life": p,
                       "onset_frac": r["onset_frac"], "true": true,
                       "lt_pred": lt[0], "lt_lo": lt[1], "lt_hi": lt[2]}
                if g:
                    row.update({"gb_pred": g[0], "gb_lo": g[1], "gb_hi": g[2]})
                if pr is not None:
                    row.update({"pred": pr["pred"], "lo": pr["lo"], "hi": pr["hi"]})
                rows.append(row)
            print(f"  orifice {c} run {k}: onset {r['onset_frac']:.2f}, n {r['n']}", flush=True)
    json.dump({"screen": screen, "rows": rows,
               "runs": {str(k): {kk: r[kk] for kk in ("cohort", "n", "onset_idx", "onset_frac",
                                                     "life", "dt")} for k, r in runs.items()}},
              open(ROWS, "w"), indent=1)


def _ape(r, k="pred"):
    return 100.0 * abs(r[k] - r["true"]) / r["true"]


def _cov(r, lo="lo", hi="hi"):
    return float(r[lo] <= r["true"] <= r[hi])


def summary():
    h = check_frozen()
    S = json.load(open(ROWS))
    screen = S["screen"]
    rows = [r for r in S["rows"] if "pred" in r]
    rng = np.random.default_rng(SEED)
    by = {}
    for r in rows:
        by.setdefault(r["run"], []).append(r)

    def flat(us):
        return [r for u in us for r in by[u]]

    def score(rs):
        out = {"runs": len({r["run"] for r in rs}), "predictions": len(rs),
               "medape_path": float(np.median([_ape(r) for r in rs])),
               "coverage_path": float(np.mean([_cov(r) for r in rs])),
               "medape_lt": float(np.median([_ape(r, "lt_pred") for r in rs])),
               "coverage_lt": float(np.mean([_cov(r, "lt_lo", "lt_hi") for r in rs]))}
        g = [r for r in rs if "gb_pred" in r]
        if g:
            out["medape_gb"] = float(np.median([_ape(r, "gb_pred") for r in g]))
            out["coverage_gb"] = float(np.mean([_cov(r, "gb_lo", "gb_hi") for r in g]))
        return out
    cohorts = sorted({r["cohort"] for r in rows})
    res = {"protocol_sha256": h, "protocol_frozen_at": FROZEN_AT, "screen": screen,
           "by_cohort": {str(c): score([r for r in rows if r["cohort"] == c]) for c in cohorts},
           "pooled": score(rows)}
    res["V1"] = {"holds": all(screen[str(c)]["verdict"] != "declines" for c in cohorts)}
    dec = [c for c in cohorts if screen[str(c)]["verdict"] == "declines"]
    nd = [c for c in cohorts if screen[str(c)]["verdict"] == "not declined"]
    if dec and nd:
        Ud = sorted(u for u in by if by[u][0]["cohort"] in dec)
        Un = sorted(u for u in by if by[u][0]["cohort"] in nd)

        def st(a, b):
            fa, fb = flat(a), flat(b)
            return (np.median([_ape(r) for r in fa]) - np.median([_ape(r) for r in fb]),
                    np.mean([_cov(r) for r in fa]) - np.mean([_cov(r) for r in fb]))
        obs = st(Ud, Un)
        bt = np.array([st([Ud[i] for i in rng.integers(0, len(Ud), len(Ud))],
                          [Un[i] for i in rng.integers(0, len(Un), len(Un))]) for _ in range(B_BOOT)])
        c1 = np.percentile(bt[:, 0], [2.5, 97.5]).tolist()
        c2 = np.percentile(bt[:, 1], [2.5, 97.5]).tolist()
        res["F1"] = {"testable": True, "diff": float(obs[0]), "ci": c1, "holds": bool(c1[0] > 0)}
        res["F2"] = {"testable": True, "diff": float(obs[1]), "ci": c2, "holds": bool(c2[1] < 0)}
    else:
        res["F1"] = res["F2"] = {"testable": False, "declined": dec, "not_declined": nd}
    late = sorted(u for u in by if by[u][0]["onset_frac"] >= 0.70)
    early = sorted(u for u in by if by[u][0]["onset_frac"] < 0.70)
    if len(late) >= 5 and len(early) >= 5:
        def cv(a, b):
            return np.mean([_cov(r) for r in flat(a)]) - np.mean([_cov(r) for r in flat(b)])
        obs = cv(late, early)
        bt = [cv([late[i] for i in rng.integers(0, len(late), len(late))],
                 [early[i] for i in rng.integers(0, len(early), len(early))]) for _ in range(B_BOOT)]
        c3 = np.percentile(bt, [2.5, 97.5]).tolist()
        res["F3"] = {"testable": True, "n_late": len(late), "n_early": len(early),
                     "diff": float(obs), "ci": c3, "holds": bool(c3[1] < 0)}
    else:
        res["F3"] = {"testable": False, "n_late": len(late), "n_early": len(early)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for c, v in res["by_cohort"].items():
        s = screen[c]
        print(f"orifice {c}: onset {s['censored_median_pct']} {s['ci']} {s['verdict']:13s} "
              f"path {v['medape_path']:.1f}/{v['coverage_path']:.2f}  LT {v['medape_lt']:.1f}/"
              f"{v['coverage_lt']:.2f}  GB {v.get('medape_gb', float('nan')):.1f}/"
              f"{v.get('coverage_gb', float('nan')):.2f}")
    print("pooled screen", screen["pooled"])
    for x in ("V1", "F1", "F2", "F3"):
        print(x, res[x])


if __name__ == "__main__":
    {"run": run, "summary": summary}.get(sys.argv[1] if len(sys.argv) > 1 else "",
                                         lambda: sys.exit(__doc__))()
