# -*- coding: utf-8 -*-
r"""Revision of the exact-posterior (PMMH) evidence of paper 3 (review issue G).

The stored PMMH run (p3_review.json, "exact_rows") kept only the classical
R-hat and the acceptance rate; the chains were not stored.  Everything here is
therefore a rerun, reusing study_p3_review._exact_one unchanged.  Two things
are wrapped around it:

  * starts:  study_p3_review._pmmh starts all eight chains within 0.05 prior
    s.d. of the CMA-ES mode.  Its first random draw is replaced (through a
    generator proxy) so that chains 0-3 start at mode + N(0, OD_SD^2 I) and
    chains 4-7 at independent prior draws N(0, I), in the prior-standardised
    coordinates u that _exact_one samples in.  The algorithm itself, its
    adaptation and its particle counts are untouched.
  * storage: the kept chains (S, C, d) are captured, and diagnosed with
    rank-normalised split R-hat and bulk/tail ESS (Vehtari et al. 2021,
    Bayesian Analysis 16:667), implemented below.

Variants (per evaluation point, the same k as the stored run):
  od        the baseline settings, over-dispersed starts
  sig05     the sigma_eps prior log-s.d. 0.5 instead of 0.15 (filter and PMMH)
  Dlo, Dhi  D at the 2.5 % / 97.5 % point of its bootstrap law over the
            reference records (B = 2000, as study_p3_review.part_dthr, but on the
            raw indicator the exact study uses); D enters priors and passage.

    python code/study_p3_rev_pmmh.py time   [bearing dataset]
    python code/study_p3_rev_pmmh.py run    VARIANT[,VARIANT] [early|late|all|unconv|LIST]
    python code/study_p3_rev_pmmh.py sum
        -> results/bearings/p3_rev_pmmh.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_p3_review as srp                                      # noqa: E402
import study_bearings as sb                                        # noqa: E402
import datasets as ds                                              # noqa: E402

BE = srp.BE
MAIN_OUT = os.path.join(BE, "p3_rev_pmmh.json")
#  a second pool run concurrently writes its own side file, merged by "sum"
OUT = os.path.join(BE, os.environ.get("P3_REV_OUT", "p3_rev_pmmh.json"))
SIDE_OUT = os.path.join(BE, "p3_rev_pmmh_sens.json")
#  sensitivity variants skip points costlier than this many particle-steps
#  per likelihood (steps x particles) in the stored run
SENS_MAX_UNITS = float(os.environ.get("P3_SENS_MAX_UNITS", 60_000))
OD_SD = float(os.environ.get("P3_OD_SD", 1.0))
NPROC = int(os.environ.get("P3_NPROC", 10))
ALPHA = 0.05
_ORIG_PMMH = srp._pmmh
_CAPTURE = {}


# ── rank-normalised R-hat and ESS (Vehtari, Gelman, Simpson, Carpenter, Buerkner 2021)
def _split(x):                       # (S, C) -> (S//2, 2C)
    h = x.shape[0] // 2
    return np.concatenate([x[:h], x[x.shape[0] - h:]], axis=1)


def _ranknorm(x):
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return special.ndtri((r - 3.0 / 8.0) / (x.size + 0.25))


def _rhat_basic(x):
    n = x.shape[0]
    W = x.var(axis=0, ddof=1).mean()
    B = n * x.mean(axis=0).var(ddof=1)
    return float(math.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else float("nan")


def _ess_basic(x):
    """ESS of (S, C) draws, Geyer's initial monotone sequence on the
    multi-chain autocorrelation, as in Stan."""
    n, m = x.shape
    xc = x - x.mean(axis=0)
    f = np.fft.rfft(xc, n=2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), axis=0)[:n] / n          # (n, m)
    W = (acov[0] * n / (n - 1)).mean()
    var_plus = W * (n - 1) / n + x.mean(axis=0).var(ddof=1)
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (W - acov.mean(axis=1)) / var_plus
    rho[0] = 1.0
    t, P, prev = 0, [], math.inf
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        p = min(p, prev)
        P.append(p)
        prev = p
        t += 2
    tau = -1.0 + 2.0 * sum(P)
    return float(n * m / max(tau, 1.0 / math.log10(n * m)))


def diagnose(x):
    """x (S, C): rank-normalised split R-hat (max of bulk and folded) and
    bulk / tail ESS."""
    xs = _split(np.asarray(x, float))
    z = _ranknorm(xs)
    zf = _ranknorm(np.abs(xs - np.median(xs)))
    rh = max(_rhat_basic(z), _rhat_basic(zf))
    ess_bulk = _ess_basic(z)
    q05, q95 = np.quantile(xs, [0.05, 0.95])
    ess_tail = min(_ess_basic(_split_ind(xs <= q05)), _ess_basic(_split_ind(xs <= q95)))
    return {"rhat_rank": rh, "ess_bulk": ess_bulk, "ess_tail": ess_tail}


def _split_ind(b):
    return b.astype(float)


# ── over-dispersed starts around the unchanged sampler
class _FirstDraw:
    """Proxy generator whose first standard_normal call returns given values."""

    def __init__(self, g, z):
        self.g, self.z = g, z

    def standard_normal(self, size=None):
        if self.z is not None:
            z, self.z = self.z, None
            return z
        return self.g.standard_normal(size)

    def __getattr__(self, a):
        return getattr(self.g, a)


def _pmmh_od(lpost, u0, rng):
    C, d = srp.C_MH, len(u0)
    starts = np.empty((C, d))
    h = C // 2
    starts[:h] = u0 + OD_SD * rng.standard_normal((h, d))
    starts[h:] = rng.standard_normal((C - h, d))
    #  _pmmh sets u = u0 + 0.05 * (first draw)
    K, acc, rhat = _ORIG_PMMH(lpost, u0, _FirstDraw(rng, (starts - u0) / 0.05))
    _CAPTURE.update({"K": K, "u0": u0, "starts": starts, "rhat_classic": rhat})
    return K, acc, rhat


srp._pmmh = _pmmh_od


# ── points and settings
def _stored_points():
    """(dataset, bearing, k, onset side, converged in the stored run) for the 26
    stored points, reruns replacing first runs as in part_exact_sum."""
    res = json.load(open(os.path.join(BE, "p3_review.json"), encoding="utf-8"))
    app = {(r["dataset"], r["bearing"]): r for r in
           json.load(open(os.path.join(BE, "applicability.json"), encoding="utf-8"))}
    keys = sorted(res["exact_rows"], key=lambda k_: k_.split("/")[2].startswith("rerun"))
    pool = {}
    for key in keys:
        tag, name, _ = key.split("/")
        if tag != "raw":
            continue
        for r in res["exact_rows"][key]:
            if r["law"] != "exact":
                continue
            a = app[(name, r["bearing"])]
            early = a["fpt_idx"] > 0 and a["fpt_frac"] < r["life_frac"]
            pool[(name, r["bearing"])] = {
                "dataset": name, "bearing": r["bearing"], "k": r["k"],
                "side": "early" if early else "late",
                "conv_stored": max(r["rhat"]) <= 1.1, "rhat_stored": max(r["rhat"]),
                "units": r["steps"] * r["n_mh"],
                "true": r["true"], "lo": r["lo"], "hi": r["hi"], "pred": r["pred"]}
    return [pool[k] for k in sorted(pool)]


_DATA = {}


def _data(name):
    if name not in _DATA:
        _DATA[name] = srp._hi_with_smoothing(name, 1)
    return _DATA[name]


def _D_settings(name, b):
    """Calibrated D (as _calibrated_D) and the 2.5 / 97.5 % bootstrap points."""
    data = _data(name)
    D_obs, ref = ds.threshold_for(data, b, name)
    med = float(np.median([max(data[r]["n"] - ds.first_prediction_time(data[r]["hi"]), 20)
                           for r in ref]))
    vals = np.array([v for v in (sb.latent_terminal(data[r], "BS", True, True, med, D_obs,
                                                    seed=7) for r in ref)
                     if np.isfinite(v) and v > 0])
    D0 = float(np.median(vals)) if vals.size else D_obs
    rng = np.random.default_rng(sb.stable_seed(b, "Dboot"))
    Db = np.median(vals[rng.integers(0, vals.size, size=(2000, vals.size))], axis=1)
    lo, hi = (float(v) for v in np.quantile(Db, [0.025, 0.975]))
    return D0, lo, hi, med, int(vals.size)


def interval_score(lo, hi, y, a=ALPHA):
    return (hi - lo) + 2.0 / a * max(lo - y, 0.0) + 2.0 / a * max(y - hi, 0.0)


def run_point(job):
    variant, p = job
    t0 = time.time()
    name, b, k = p["dataset"], p["bearing"], p["k"]
    rec = _data(name)[b]
    D0, Dlo, Dhi, med, nref = _D_settings(name, b)
    D = {"od": D0, "sig05": D0, "Dlo": Dlo, "Dhi": Dhi}[variant]
    keep = sb.PRIOR_SD_INFORMED["sigma_eps"]
    try:
        if variant == "sig05":
            sb.PRIOR_SD_INFORMED["sigma_eps"] = 0.5
        _CAPTURE.clear()
        r = srp._exact_one(rec, k, D, med, sb.stable_seed(b, k, "BS"))
    finally:
        sb.PRIOR_SD_INFORMED["sigma_eps"] = keep
    if r is None:
        return variant, p, None
    K = _CAPTURE["K"]                                   # (S, C, d)
    S, C, d = K.shape
    diag = [diagnose(K[:, :, j]) for j in range(d)]
    moved = np.any(np.diff(K, axis=0) != 0, axis=2)     # (S-1, C)
    true = float(rec["t"][-1] - rec["t"][k])
    row = {"dataset": name, "bearing": b, "k": k, "side": p["side"], "variant": variant,
           "D": D, "D0": D0, "D_ci": [Dlo, Dhi], "n_ref": nref, "true": true,
           "names": r["names"], "n_mh": r["n_mh"], "sd_logz": r["sd_logz"],
           "steps": r["steps"], "acc": r["acc"],
           "acc_chain": moved.mean(axis=0).tolist(),
           "rhat_classic": [float(v) for v in r["rhat"]],
           "rhat_rank": [q["rhat_rank"] for q in diag],
           "ess_bulk": [q["ess_bulk"] for q in diag],
           "ess_tail": [q["ess_tail"] for q in diag],
           "starts": _CAPTURE["starts"].tolist(), "mode": _CAPTURE["u0"].tolist(),
           "chain_means": K.mean(axis=0).tolist(),
           "chain_sds": K.std(axis=0).tolist(),
           "chains_thin10": K[::10].round(4).tolist(),
           "theta_exact": r["theta_exact"], "seconds": time.time() - t0}
    for est in ("exact", "main"):
        q = r[est]
        row[est] = {"pred": q["pred"], "lo": q["lo"], "hi": q["hi"],
                    "covered": bool(q["lo"] <= true <= q["hi"]),
                    "width": q["hi"] - q["lo"],
                    "IS": interval_score(q["lo"], q["hi"], true),
                    "capped": bool(q["pred"] == q["hi"])}
    return variant, p, row


def _load():
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def _save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, OUT)


def cmd_time():
    pts = _stored_points()
    b = sys.argv[2] if len(sys.argv) > 2 else "Bearing2_7"
    name = sys.argv[3] if len(sys.argv) > 3 else "pronostia"
    p = next(q for q in pts if q["bearing"] == b and q["dataset"] == name)
    t0 = time.time()
    _, _, row = run_point(("od", p))
    print(f"{name} {b}: {time.time() - t0:.1f} s; steps {row['steps']} N {row['n_mh']}; "
          f"rank R-hat {max(row['rhat_rank']):.3f} classic {max(row['rhat_classic']):.3f} "
          f"ESS bulk {min(row['ess_bulk']):.0f} tail {min(row['ess_tail']):.0f} "
          f"acc {row['acc']:.2f}; exact [{row['exact']['lo']:.0f},{row['exact']['hi']:.0f}] "
          f"true {row['true']:.0f}", flush=True)


def cmd_run():
    import multiprocessing as mp
    variants = sys.argv[2].split(",")
    sel = sys.argv[3] if len(sys.argv) > 3 else "all"
    pts = _stored_points()
    if sel == "early":
        pts = [p for p in pts if p["side"] == "early"]
    elif sel == "late":
        pts = [p for p in pts if p["side"] == "late"]
    elif sel == "conv":
        pts = [p for p in pts if p["conv_stored"]]
    elif sel == "unconv":
        pts = [p for p in pts if not p["conv_stored"]]
    elif sel != "all":
        want = set(sel.split(","))
        pts = [p for p in pts if p["bearing"] in want or f"{p['dataset']}/{p['bearing']}" in want]
    done = set()
    for f in (MAIN_OUT, SIDE_OUT):
        if os.path.exists(f):
            done |= {(v, f"{x['dataset']}/{x['bearing']}") for v, rows in
                     json.load(open(f, encoding="utf-8")).get("runs", {}).items()
                     for x in rows.values()}
    jobs = [(v, p) for v in variants for p in pts
            if (v, f"{p['dataset']}/{p['bearing']}") not in done
            and (v == "od" or p["units"] <= SENS_MAX_UNITS)]
    #  long records first, so the pool does not end on one straggler
    jobs.sort(key=lambda j: -j[1]["k"] * (8 if j[1]["dataset"] == "xjtu" else 1))
    print(f"{len(jobs)} jobs on {NPROC} processes", flush=True)
    with mp.Pool(NPROC) as pool:
        for variant, p, row in pool.imap_unordered(run_point, jobs):
            res = _load()
            key = f"{p['dataset']}/{p['bearing']}"
            if row is None:
                print(f"  {variant} {key}: no filter states", flush=True)
                continue
            res.setdefault("runs", {}).setdefault(variant, {})[key] = row
            _save(res)
            e, m = row["exact"], row["main"]
            print(f"  {variant:5s} {key:22s} {p['side']:5s} {row['seconds']:6.0f}s "
                  f"Rrank {max(row['rhat_rank']):6.3f} Rcl {max(row['rhat_classic']):6.3f} "
                  f"ESSb {min(row['ess_bulk']):5.0f} ESSt {min(row['ess_tail']):5.0f} "
                  f"acc {row['acc']:.2f} true {row['true']:8.0f} exact [{e['lo']:.0f},{e['hi']:.0f}]"
                  f" cov {int(e['covered'])} | filter [{m['lo']:.0f},{m['hi']:.0f}] "
                  f"cov {int(m['covered'])}", flush=True)


# ── tests of the early / late contrast
def _strat_exact(tables):
    """Exact conditional test of a common odds ratio across 2x2 strata
    (statistic: covered early points summed over strata), one-sided
    for more coverage among early points; returns p and the support."""
    dists = []
    for (a, n1, c, n0) in tables:              # a of n1 early covered, c of n0 late
        m = a + c
        lo, hi = max(0, m - n0), min(n1, m)
        xs = np.arange(lo, hi + 1)
        pr = stats.hypergeom.pmf(xs, n1 + n0, m, n1)
        dists.append((xs, pr))
    tot = {0: 1.0}
    for xs, pr in dists:
        new = {}
        for s, q in tot.items():
            for x, r in zip(xs, pr):
                new[s + int(x)] = new.get(s + int(x), 0.0) + q * r
        tot = new
    obs = sum(t[0] for t in tables)
    return float(sum(q for s, q in tot.items() if s >= obs))


def contrast(rows_by_ds):
    """rows_by_ds: {dataset: [(side, covered)]} -> Fisher, stratified exact,
    Bayesian Beta(1,1) comparison."""
    tab = []
    for name, rs in sorted(rows_by_ds.items()):
        e = [c for s, c in rs if s == "early"]
        l_ = [c for s, c in rs if s == "late"]
        tab.append((sum(e), len(e), sum(l_), len(l_)))
    a, n1 = sum(t[0] for t in tab), sum(t[1] for t in tab)
    c, n0 = sum(t[2] for t in tab), sum(t[3] for t in tab)
    out = {"early": [a, n1], "late": [c, n0], "by_dataset": tab}
    if n1 and n0:
        out["fisher_p_two_sided"] = float(stats.fisher_exact([[a, n1 - a], [c, n0 - c]])[1])
        out["fisher_p_one_sided"] = float(stats.fisher_exact([[a, n1 - a], [c, n0 - c]],
                                                             alternative="greater")[1])
        out["stratified_exact_p_one_sided"] = _strat_exact([t for t in tab if t[1] and t[3]])
        rng = np.random.default_rng(1)
        pe = rng.beta(1 + a, 1 + n1 - a, 200_000)
        pl = rng.beta(1 + c, 1 + n0 - c, 200_000)
        out["bayes_P_early_gt_late"] = float(np.mean(pe > pl))
        out["bayes_diff_95"] = np.quantile(pe - pl, [0.025, 0.5, 0.975]).tolist()
    return out


def cmd_sum():
    res = _load()
    if OUT == MAIN_OUT and os.path.exists(SIDE_OUT):
        for v, rows in json.load(open(SIDE_OUT, encoding="utf-8")).get("runs", {}).items():
            res.setdefault("runs", {}).setdefault(v, {}).update(rows)
    pts = _stored_points()
    res["stored_diagnostics"] = {
        "chains_stored": False,
        "note": ("p3_review.json exact_rows keep the classical R-hat per parameter and "
                 "the pooled acceptance only; no draws, so rank-normalised R-hat and ESS "
                 "need the rerun under 'runs'."),
        "points": [{k_: p[k_] for k_ in ("dataset", "bearing", "k", "side", "conv_stored",
                                         "rhat_stored")} for p in pts]}
    by = {}
    for p in pts:
        if p["conv_stored"]:
            by.setdefault(p["dataset"], []).append(
                (p["side"], p["lo"] <= p["true"] <= p["hi"]))
    summ = {"stored_converged": contrast(by)}
    for variant, rows in res.get("runs", {}).items():
        rs = list(rows.values())
        crit = {"stored_conv": lambda r: (r["dataset"], r["bearing"]) in
                {(p["dataset"], p["bearing"]) for p in pts if p["conv_stored"]},
                "rank101_ess400": lambda r: max(r["rhat_rank"]) <= 1.01
                and min(r["ess_bulk"]) >= 400 and min(r["ess_tail"]) >= 400,
                "rank105": lambda r: max(r["rhat_rank"]) <= 1.05,
                "rank110": lambda r: max(r["rhat_rank"]) <= 1.10,
                "all": lambda r: True}
        sv = {"n_points": len(rs)}
        for cname, f in crit.items():
            sub = [r for r in rs if f(r)]
            byd = {}
            for r in sub:
                byd.setdefault(r["dataset"], []).append((r["side"], r["exact"]["covered"]))
            byd_m = {}
            for r in sub:
                byd_m.setdefault(r["dataset"], []).append((r["side"], r["main"]["covered"]))
            ent = {"n": len(sub), "exact": contrast(byd), "filter": contrast(byd_m)}
            for side in ("early", "late"):
                ss = [r for r in sub if r["side"] == side]
                if not ss:
                    continue
                ent[side] = {
                    est: {"n": len(ss),
                          "covered": int(sum(r[est]["covered"] for r in ss)),
                          "median_width": float(np.median([r[est]["width"] for r in ss])),
                          "median_IS": float(np.median([r[est]["IS"] for r in ss])),
                          "mean_IS": float(np.mean([r[est]["IS"] for r in ss])),
                          "median_rel_width": float(np.median([r[est]["width"] / r["true"]
                                                               for r in ss])),
                          "n_capped": int(sum(r[est]["capped"] for r in ss))}
                    for est in ("exact", "main")}
            sv[cname] = ent
        sv["rhat_rank_max_median"] = float(np.median([max(r["rhat_rank"]) for r in rs]))
        sv["ess_bulk_min_median"] = float(np.median([min(r["ess_bulk"]) for r in rs]))
        sv["ess_tail_min_median"] = float(np.median([min(r["ess_tail"]) for r in rs]))
        sv["acc_median"] = float(np.median([r["acc"] for r in rs]))
        sv["per_point"] = [{"point": f"{r['dataset']}/{r['bearing']}", "side": r["side"],
                            "rhat_rank_max": max(r["rhat_rank"]),
                            "rhat_classic_max": max(r["rhat_classic"]),
                            "ess_bulk_min": min(r["ess_bulk"]), "ess_tail_min": min(r["ess_tail"]),
                            "acc": r["acc"], "acc_chain_min": min(r["acc_chain"]),
                            "D": r["D"], "true": r["true"],
                            "exact": [r["exact"]["lo"], r["exact"]["hi"], r["exact"]["covered"]],
                            "filter": [r["main"]["lo"], r["main"]["hi"], r["main"]["covered"]]}
                           for r in sorted(rs, key=lambda r: (r["side"], r["dataset"], r["bearing"]))]
        summ[variant] = sv
    res["summary"] = summ
    _save(res)
    print(json.dumps({k_: {kk: vv for kk, vv in v.items() if kk != "per_point"}
                      for k_, v in summ.items()}, indent=1)[:12000])


if __name__ == "__main__":
    {"time": cmd_time, "run": cmd_run, "sum": cmd_sum}[sys.argv[1]]()
