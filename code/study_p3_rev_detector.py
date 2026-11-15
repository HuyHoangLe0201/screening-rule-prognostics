# -*- coding: utf-8 -*-
r"""Revision analyses of the onset detector and the screen (review issues E, H, I).

sim    Operating characteristics of the paper's detector
       (datasets.first_prediction_time: 3 robust s.d. band, persistence 10,
       sustained share 0.8, healthy window min(40, max(5, n // 10))).
       Null records: stationary noise of unit marginal s.d., AR(1) with lag-1
       autocorrelation 0, 0.5, 0.85, and white noise passed through the paper's
       causal smoother (datasets._smooth, windows 9 and 15).  Lengths 50, 200,
       1000, 3000; 2000 records per cell.  -> false-onset rate.
       Alternatives: the same noise plus a flat-then-rising signal starting at
       true onset fraction 0.3, 0.6, 0.9, linear or exponential (k = 5), with a
       terminal rise of 10, 30, 100 noise s.d.; 1000 records per cell.
       -> detection rate and bias of the detected onset fraction.
       The detector is vectorised over records; the vectorised form is checked
       against datasets.first_prediction_time on 3000 random records first.

cens   Every Table 8 archive under three conventions for records without a
       detected onset: (a) right-censored at end of life (the paper), (b) dropped,
       (c) onset at end of life.  Beside the percentile bootstrap, a BCa
       bootstrap and the exact distribution-free order-statistic interval for
       the median.  Verdict by study_p3_review._verdict3 at 70 %.

thr    Verdict of each archive at thresholds 50-95 % (step 1) under each
       interval, and the thresholds at which each verdict flips.

cusum  A structurally different detector: a one-sided Page CUSUM, sequential
       and causal, on the same indicator standardised by its healthy window
       (median, robust s.d.), reference value k = 0.5, decision interval h set
       per record as the 95th percentile of the maximum CUSUM statistic over
       2000 simulated in-control AR(1) records of the same length, with the
       archive's pooled healthy-window lag-1 autocorrelation and the baseline
       estimated from the same healthy window: a nominal 5 % false-alarm
       probability per record.  Onset = the last zero of the statistic before
       the alarm.  Verdicts as in the paper (censored median, percentile
       bootstrap, indicator check below 25 % detected).  With CUSUM_CAL=white
       the decision interval is calibrated for independent noise instead
       (key "cusum_white"), to bracket the effect of the calibration.

hiq    Coble & Hines (2009) indicator-quality metrics, on the indicator as the
       screen uses it:
         monotonicity  M = mean_j |#(dx_j > 0) - #(dx_j < 0)| / (n_j - 1)
         prognosability P = exp(-sd_j(x_j(N_j)) / mean_j |x_j(N_j) - x_j(1)|)
         trendability  T = min_{j,k} |corr(x_j, x_k)|, records resampled to a
                       common grid of 101 life fractions (median pairwise
                       |corr| also given)
       for the five bearing indicators of Table 6 and every archive; and, per
       record, Spearman correlation of each record's monotonicity and of its
       onset fraction with the path model's error on the C-MAPSS band-study units.

    python code/study_p3_rev_detector.py [sim cens thr cusum hiq]
        ->  results/bearings/p3_rev_detector.json (+ .log)
Indicator series come from study_p3_rev_cache.py ($P3_REV_CACHE).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import datasets as ds                                             # noqa: E402
import study_p3_review as R                                       # noqa: E402
import study_p3_rev_cache as C                                    # noqa: E402

BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_rev_detector.json")
LOG = os.path.join(BE, "p3_rev_detector.log")
THR = 70.0
B_BOOT = 10_000

TABLE8 = ["PRONOSTIA", "XJTU-SY", "C-MAPSS FD001", "C-MAPSS FD003",
          "C-MAPSS FD002+regime", "C-MAPSS FD004+regime", "NASA cells",
          "Backblaze Q4_2023", "XJTU Li-ion", "N-CMAPSS", "Backblaze Q1_2024",
          "Backblaze Q2_2024"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def _save(key, val):
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res[key] = val
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


def _f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ══════════════════════════════════════════════════════════════════════════
# the paper's detector, vectorised over records (checked against the original)
# ══════════════════════════════════════════════════════════════════════════
def onset_batch(X, n_health=40, thr=3.0, persist=10, frac=0.8):
    """Row-wise datasets.first_prediction_time; 0 = no onset."""
    X = np.asarray(X, float)
    m, n = X.shape
    nh = min(n_health, max(5, n // 10))
    base = np.median(X[:, :nh], axis=1, keepdims=True)
    mad = np.median(np.abs(X[:, :nh] - base), axis=1, keepdims=True) * 1.4826
    above = X > base + thr * np.maximum(mad, 1e-9)
    fa = np.cumsum(above[:, ::-1], axis=1)[:, ::-1] / np.arange(n, 0, -1)
    cs = np.concatenate([np.zeros((m, 1)), np.cumsum(above, axis=1)], axis=1)
    run = (cs[:, persist:] - cs[:, :-persist]) == persist      # run[i]: above[i:i+persist]
    ok = run[:, :n - persist] & (fa[:, :n - persist] >= frac)
    idx = np.argmax(ok, axis=1)
    return np.where(ok.any(axis=1), idx, 0)


def _check_batch():
    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(3000):
        n = int(rng.integers(30, 400))
        x = rng.standard_normal(n).cumsum() * rng.uniform(0, 0.3) + rng.standard_normal(n)
        if rng.random() < 0.5:
            x[int(rng.uniform(0.2, 0.9) * n):] += rng.uniform(0, 8)
        bad += int(onset_batch(x[None])[0] != ds.first_prediction_time(x))
    assert bad == 0, bad
    return 3000


# ══════════════════════════════════════════════════════════════════════════
# sim
# ══════════════════════════════════════════════════════════════════════════
NOISES = ("ar0", "ar0.5", "ar0.85", "smooth9", "smooth15")
_SM_SD = {}


def noise(kind, m, n, rng):
    if kind.startswith("ar"):
        phi = float(kind[2:])
        z = rng.standard_normal((m, n))
        z[:, 0] /= math.sqrt(1 - phi ** 2) if phi > 0 else 1.0
        return signal.lfilter([math.sqrt(1 - phi ** 2)], [1, -phi], z, axis=1)
    w = int(kind[6:])
    if w not in _SM_SD:                       # stationary s.d. of the smoothed noise
        _SM_SD[w] = float(np.std(ds._smooth(np.random.default_rng(0).standard_normal(200_000), w)[w:]))
    z = rng.standard_normal((n, m))
    s = pd.DataFrame(z).rolling(w, min_periods=1).median()
    s = s.rolling(w, min_periods=1).mean().to_numpy().T
    return s / _SM_SD[w]


def shape(kind, n, f0, snr):
    t = np.arange(n)
    t0 = int(round(f0 * n))
    u = np.clip((t - t0) / max(n - 1 - t0, 1), 0, None)
    if kind == "linear":
        s = u
    else:
        s = (np.exp(5 * u) - 1) / (np.exp(5) - 1)
    return snr * s, t0


def part_sim():
    n_checked = _check_batch()
    log(f"vectorised detector == datasets.first_prediction_time on {n_checked} records")
    rng = np.random.default_rng(20261114)
    out = {"checked_records": n_checked, "null": {}, "alt": {}}
    for kind in NOISES:
        for n in (50, 200, 1000, 3000):
            X = noise(kind, 2000, n, rng)
            i = onset_batch(X)
            k = int((i > 0).sum())
            lo, hi = R.wilson(k, 2000)
            out["null"][f"{kind}|n{n}"] = {
                "noise": kind, "n": n, "healthy_window": min(40, max(5, n // 10)), "reps": 2000,
                "false_onset_rate": k / 2000, "ci": [lo, hi],
                "median_false_onset_frac": _f(np.median(i[i > 0] / n)) if k else None}
            log(f"null {kind:8s} n={n:5d} nh={min(40, max(5, n // 10)):3d}  false onset "
                f"{k / 2000:.3f} [{lo:.3f},{hi:.3f}]")
    for kind in ("ar0", "ar0.5", "ar0.85", "smooth15"):
        for n in (50, 200, 1000, 3000):
            for sh in ("linear", "exponential"):
                for f0 in (0.3, 0.6, 0.9):
                    for snr in (10, 30, 100):
                        s, t0 = shape(sh, n, f0, snr)
                        X = noise(kind, 1000, n, rng) + s
                        i = onset_batch(X)
                        d = i > 0
                        fr = i[d] / n
                        out["alt"][f"{kind}|n{n}|{sh}|f{f0}|snr{snr}"] = {
                            "noise": kind, "n": n, "shape": sh, "f0": f0, "snr": snr,
                            "detection_rate": float(d.mean()),
                            "bias_median": _f(np.median(fr - t0 / n)) if d.any() else None,
                            "bias_q10_q90": [_f(q) for q in np.quantile(fr - t0 / n, [.1, .9])]
                            if d.any() else None,
                            "share_before_true": _f(np.mean(fr < t0 / n)) if d.any() else None}
    #  compact summaries
    alt = out["alt"]

    def agg(sel):
        v = [r for r in alt.values() if sel(r)]
        b = [r["bias_median"] for r in v if r["bias_median"] is not None]
        return {"cells": len(v), "mean_detection": _f(np.mean([r["detection_rate"] for r in v])),
                "median_bias": _f(np.median(b)) if b else None,
                "max_bias": _f(max(b)) if b else None}
    summ = {}
    for sh in ("linear", "exponential"):
        for snr in (10, 30, 100):
            for f0 in (0.3, 0.6, 0.9):
                summ[f"{sh}|snr{snr}|f{f0}"] = agg(lambda r: r["shape"] == sh and r["snr"] == snr
                                                   and r["f0"] == f0)
    out["alt_summary"] = summ
    for k, v in summ.items():
        log(f"alt {k:26s} detection {v['mean_detection']:.3f}  median bias {v['median_bias']}  "
            f"max {v['max_bias']}")
    for n in (50, 200, 1000, 3000):
        v = agg(lambda r: r["n"] == n)
        out["alt_summary"][f"n{n}"] = v
        log(f"alt n={n}: {v}")
    for kind in ("ar0", "ar0.5", "ar0.85", "smooth15"):
        v = agg(lambda r: r["noise"] == kind)
        out["alt_summary"][kind] = v
        log(f"alt {kind}: {v}")
    _save("sim", out)


# ══════════════════════════════════════════════════════════════════════════
# per-record onsets of every Table 8 archive (stored results, as the paper)
# ══════════════════════════════════════════════════════════════════════════
def onsets_table8():
    arch = R._onset_records()
    arch["Backblaze Q4_2023"] = arch.pop("Backblaze drives")
    pre = json.load(open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))
    arch["XJTU Li-ion"] = [r["onset_frac"] for r in pre["battery"]["records"]]
    arch["N-CMAPSS"] = [r["onset_frac"] for r in pre["ncmapss"]["records"]]
    for q in ("Q1_2024", "Q2_2024"):
        d = json.load(open(os.path.join(BE, f"p3_decline_part_{q}.json"), encoding="utf-8"))
        arch["Backblaze " + q] = [r["onset_frac"] for r in d["drives"]]
    return {k: arch[k] for k in TABLE8}


def order_stat_ci(x, level=0.95):
    """Exact distribution-free interval [x_(l), x_(n+1-l)] for the median."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    l = int(stats.binom.ppf((1 - level) / 2, n, 0.5))      # P(B <= l-1) <= alpha/2
    while l >= 1 and stats.binom.cdf(l - 1, n, 0.5) > (1 - level) / 2:
        l -= 1
    l = max(l, 1)
    cov = 1 - 2 * stats.binom.cdf(l - 1, n, 0.5)
    return float(x[l - 1]), float(x[n - l]), float(cov)


def bca_ci(x, seed):
    """BCa interval of the median; undetected records enter as 2.0 (beyond the
    end of life, which leaves every order statistic below them unchanged)."""
    x = np.asarray(x, float)
    if np.all(x == x[0]):
        return float(x[0]), float(x[0])
    r = stats.bootstrap((x,), np.median, method="BCa", n_resamples=B_BOOT,
                        random_state=np.random.default_rng(seed), vectorized=True, axis=-1)
    lo, hi = r.confidence_interval
    return float(lo), float(hi)


def _pct(z):
    """fraction -> % of life; beyond the end of life -> None (not reached)."""
    return None if (z is None or not np.isfinite(z) or z > 1.0 + 1e-12) else 100 * float(z)


def summarise(v, conv, seed):
    """v: onset fractions with None for undetected; conv a/b/c."""
    det = np.array([f for f in v if f is not None], float)
    if conv == "a":
        x = np.array([np.inf if f is None else f for f in v], float)
    elif conv == "b":
        x = det
    else:
        x = np.array([1.0 if f is None else f for f in v], float)
    rng = np.random.default_rng(seed)
    med = float(np.median(x))
    meds = np.median(x[rng.integers(0, len(x), size=(B_BOOT, len(x)))], axis=1)
    plo, phi_ = np.quantile(meds, [0.025, 0.975], method="inverted_cdf")
    olo, ohi, cov = order_stat_ci(x)
    xb = np.where(np.isfinite(x), x, 2.0)
    try:
        blo, bhi = bca_ci(xb, seed)
    except Exception as e:                    # degenerate jackknife: reported, not hidden
        blo = bhi = float("nan")
        log("   BCa failed:", repr(e))
    ci = {"percentile": [_pct(plo), _pct(phi_)], "bca": [_pct(blo), _pct(bhi)],
          "order_stat": [_pct(olo), _pct(ohi)]}
    check = det.size < 0.25 * len(v)
    ver = {k: ("check the indicator" if check else R._verdict3(c[0], c[1], THR))
           for k, c in ci.items()}
    return {"n": len(x), "median_pct": _pct(med), "ci": ci, "order_stat_coverage": round(cov, 4),
            "verdict": ver}


def part_cens():
    arch = onsets_table8()
    rev = json.load(open(os.path.join(BE, "p3_review.json"), encoding="utf-8"))["cens"]
    paper = {"PRONOSTIA": "declines", "XJTU-SY": "undecided", "Backblaze Q4_2023": "declines",
             "Backblaze Q1_2024": "declines", "Backblaze Q2_2024": "declines"}
    out = {}
    for j, (k, v) in enumerate(arch.items()):
        r = {c: summarise(v, c, 20261114 + j) for c in "abc"}
        r["n_detected"] = sum(f is not None for f in v)
        r["paper_verdict"] = paper.get(k, "not declined")
        r["verdict_changes"] = sorted({f"{c}/{m}" for c in "abc" for m, vv in r[c]["verdict"].items()
                                       if vv != r["paper_verdict"]})
        #  convention (a) with the percentile bootstrap must be the paper's own
        assert r["a"]["verdict"]["percentile"] == r["paper_verdict"], (k, r["a"])
        if k in rev:
            assert abs((r["a"]["median_pct"] or -1) - (rev[k]["censored_median_pct"] or -1)) < 1e-9
        out[k] = r
        log(f"{k:22s} n={r['a']['n']:3d} det={r['n_detected']:3d}  paper {r['paper_verdict']:12s}")
        for c in "abc":
            s = r[c]
            log(f"    ({c}) med {s['median_pct']}  pct {s['ci']['percentile']}  "
                f"BCa {s['ci']['bca']}  exact {s['ci']['order_stat']} (cov {s['order_stat_coverage']})"
                f"  -> {s['verdict']}")
        log(f"    changes: {r['verdict_changes'] or 'none'}")
    _save("cens", out)


def part_thr():
    cens = json.load(open(OUT, encoding="utf-8"))["cens"]
    grid = list(range(50, 96))
    fine = np.round(np.arange(50, 99.0001, 0.1), 1)
    out = {"grid": grid, "by_interval": {}}
    for m in ("percentile", "bca", "order_stat"):
        res = {}
        for k, r in cens.items():
            lo, hi = r["a"]["ci"][m]
            if r["a"]["verdict"][m] == "check the indicator":
                continue
            vs = [R._verdict3(lo, hi, t) for t in grid]
            v70 = R._verdict3(lo, hi, THR)
            same = [float(t) for t in fine if R._verdict3(lo, hi, t) == v70]
            flips = [float(fine[i]) for i in range(1, len(fine))
                     if R._verdict3(lo, hi, fine[i]) != R._verdict3(lo, hi, fine[i - 1])]
            res[k] = {"ci": [lo, hi], "verdicts": vs, "at70": v70,
                      "same_as_70": [min(same), max(same)], "flips_at": flips}
        keep = [float(t) for t in fine if all(R._verdict3(*res[k]["ci"], t) == res[k]["at70"]
                                              for k in res)]
        near = {k: v["flips_at"] for k, v in res.items() if any(60 <= f <= 80 for f in v["flips_at"])}
        counts = {str(t): {s: sum(res[k]["verdicts"][i] == s for k in res)
                           for s in ("declines", "undecided", "not declined")}
                  for i, t in enumerate(grid)}
        out["by_interval"][m] = {"archives": res, "stable_range": [min(keep), max(keep)] if keep else None,
                                 "flip_within_60_80": near, "counts": counts}
        log(f"[{m}] every verdict as at 70 % for thresholds {out['by_interval'][m]['stable_range']}")
        for k, v in res.items():
            log(f"   {k:22s} ci {v['ci']}  at70 {v['at70']:12s} same for {v['same_as_70']}  flips {v['flips_at']}")
    _save("thr", out)


# ══════════════════════════════════════════════════════════════════════════
# cusum
# ══════════════════════════════════════════════════════════════════════════
K_REF = 0.5
ALPHA = 0.05
N_CAL = 2000
_HCACHE = {}


def _standardise(x, nh):
    base = np.median(x[..., :nh], axis=-1, keepdims=True)
    sd = np.median(np.abs(x[..., :nh] - base), axis=-1, keepdims=True) * 1.4826
    sd2 = np.std(x[..., :nh], axis=-1, keepdims=True)
    sd = np.where(sd > 1e-9, sd, np.maximum(sd2, 1e-9))
    return (x - base) / sd


def cusum_path(z, nh):
    """S_t over t >= nh, vectorised over rows."""
    z = np.atleast_2d(z)
    S = np.zeros(z.shape)
    s = np.zeros(z.shape[0])
    for t in range(nh, z.shape[1]):
        s = np.maximum(0.0, s + z[:, t] - K_REF)
        S[:, t] = s
    return S


def h_for(n, rho):
    key = (n, round(rho, 2))
    if key not in _HCACHE:
        nh = min(40, max(5, n // 10))
        rng = np.random.default_rng(hash(key) % 2 ** 32)
        X = noise(f"ar{round(rho, 2)}" if rho > 0 else "ar0", N_CAL, n, rng)
        S = cusum_path(_standardise(X, nh), nh)
        _HCACHE[key] = float(np.quantile(S.max(axis=1), 1 - ALPHA))
    return _HCACHE[key]


def cusum_onset(x, rho):
    x = np.asarray(x, float)
    n = len(x)
    nh = min(40, max(5, n // 10))
    S = cusum_path(_standardise(x, nh), nh)[0]
    h = h_for(n, rho)
    hit = np.nonzero(S > h)[0]
    if hit.size == 0:
        return None, h
    a = hit[0]
    zeros = np.nonzero(S[nh:a] == 0)[0]
    tau = nh + (zeros[-1] + 1 if zeros.size else 0)
    return int(tau), h


def pooled_rho(recs):
    r = []
    for rec in recs:
        x = np.asarray(rec["hi"], float)
        nh = min(40, max(5, len(x) // 10))
        w = x[:nh]
        if nh >= 10 and np.std(w) > 1e-12:
            r.append(np.corrcoef(w[:-1], w[1:])[0, 1])
    return float(np.clip(np.nanmedian(r), 0, 0.95)) if r else 0.0


def _frac(rec, i):
    """onset index -> fraction of life as the archive's own script defines it"""
    if i is None or i == 0:
        return None
    if rec.get("frac_of") is not None:
        return float(rec["frac_of"][i])
    return i / len(rec["hi"])


def part_cusum():
    names = [k for k in TABLE8 if os.path.exists(C.cache_path(k))]
    log("archives with cached indicators:", names)
    out = {"k": K_REF, "alpha": ALPHA, "n_cal": N_CAL, "archives": {}}
    for j, k in enumerate(names):
        recs = C.load(k)
        #  the cached indicators reproduce the paper's onsets
        mism = 0
        for rec in recs:
            i = onset_batch(np.asarray(rec["hi"], float)[None])[0]
            f = _frac(rec, i)
            s = rec["stored"]
            mism += int((f is None) != (s is None) or (f is not None and abs(f - s) > 1e-9))
        #  CUSUM_CAL=white: h calibrated for independent noise (rho = 0), the
        #  textbook setting, which ignores the indicator's autocorrelation
        rho = 0.0 if os.environ.get("CUSUM_CAL") == "white" else pooled_rho(recs)
        t0 = time.time()
        cu = []
        for rec in recs:
            tau, _ = cusum_onset(rec["hi"], rho)
            cu.append(_frac(rec, tau))
        paper = [rec["stored"] for rec in recs]
        sa = summarise(cu, "a", 777 + j)
        sp = summarise(paper, "a", 777 + j)
        both = [(a, b) for a, b in zip(paper, cu) if a is not None and b is not None]
        rho_s = (float(stats.spearmanr(*zip(*both))[0]) if len(both) > 3 else None)
        agree_det = float(np.mean([(a is None) == (b is None) for a, b in zip(paper, cu)]))
        r = {"n": len(recs), "rho_healthy": rho, "paper_onsets_reproduced": mism == 0,
             "n_mismatch": mism,
             "cusum_detected": sum(c is not None for c in cu),
             "paper_detected": sum(p is not None for p in paper),
             "cusum_censored_median_pct": sa["median_pct"],
             "cusum_ci_percentile": sa["ci"]["percentile"],
             "cusum_verdict": sa["verdict"]["percentile"],
             #  the interval read against 70 % without the step-4 detection check
             "cusum_verdict_interval_only": R._verdict3(*sa["ci"]["percentile"], THR),
             "cusum_ci_order_stat": sa["ci"]["order_stat"],
             "paper_verdict": sp["verdict"]["percentile"],
             "same_verdict": sa["verdict"]["percentile"] == sp["verdict"]["percentile"],
             "detection_agreement": agree_det,
             "spearman_onsets_both_detected": rho_s,
             "median_abs_diff_pct": _f(100 * np.median([abs(a - b) for a, b in both])) if both else None,
             "median_diff_cusum_minus_paper_pct": _f(100 * np.median([b - a for a, b in both])) if both else None,
             "share_cusum_earlier": _f(np.mean([b < a for a, b in both])) if both else None}
        out["archives"][k] = r
        log(f"{k:22s} rho {rho:.2f} reproduced {mism == 0} | CUSUM det {r['cusum_detected']}/{len(recs)} "
            f"med {sa['median_pct']} {sa['ci']['percentile']} -> {r['cusum_verdict']} | paper "
            f"{r['paper_verdict']} | agree {agree_det:.2f} spearman {rho_s} "
            f"diff {r['median_diff_cusum_minus_paper_pct']} ({time.time() - t0:.0f}s)")
    out["n_same"] = sum(v["same_verdict"] for v in out["archives"].values())
    out["n_archives"] = len(out["archives"])
    #  realised false-alarm probability of the calibrated CUSUM on smoothed white noise
    rng = np.random.default_rng(5)
    fa = {}
    for kind, n in (("smooth15", 2800), ("smooth9", 150)):
        X = noise(kind, 500, n, rng)
        rho = float(np.median([np.corrcoef(x[:40][:-1], x[:40][1:])[0, 1] for x in X[:200]]))
        nh = min(40, max(5, n // 10))
        S = cusum_path(_standardise(X, nh), nh)
        fa[f"{kind}|n{n}"] = {"rho_used": _f(np.clip(rho, 0, .95)),
                              "false_alarm": _f(np.mean(S.max(axis=1) > h_for(n, float(np.clip(rho, 0, .95)))))}
    out["false_alarm_check_smoothed_noise"] = fa
    log("CUSUM false alarm on smoothed white noise:", fa)
    log(f"CUSUM verdict = paper verdict on {out['n_same']} of {out['n_archives']} archives")
    _save("cusum_white" if os.environ.get("CUSUM_CAL") == "white" else "cusum", out)


# ══════════════════════════════════════════════════════════════════════════
# hiq
# ══════════════════════════════════════════════════════════════════════════
def monotonicity(x):
    d = np.diff(np.asarray(x, float))
    return abs(int((d > 0).sum()) - int((d < 0).sum())) / max(len(x) - 1, 1)


def hi_quality(series):
    xs = [np.asarray(s, float) for s in series if len(s) > 2]
    M = float(np.mean([monotonicity(x) for x in xs]))
    end = np.array([x[-1] for x in xs])
    start = np.array([x[0] for x in xs])
    den = np.mean(np.abs(end - start))
    P = float(np.exp(-np.std(end) / den)) if den > 0 else 0.0
    g = np.linspace(0, 1, 101)
    Y = np.array([np.interp(g, np.linspace(0, 1, len(x)), x) for x in xs])
    ok = Y.std(axis=1) > 1e-12
    Y = Y[ok]
    if len(Y) >= 2:
        Cm = np.abs(np.corrcoef(Y))
        iu = np.triu_indices(len(Y), 1)
        T, Tmed = float(Cm[iu].min()), float(np.median(Cm[iu]))
    else:
        T = Tmed = None
    return {"monotonicity": _f(M), "prognosability": _f(P), "trendability_min": _f(T),
            "trendability_median": _f(Tmed), "n": len(xs), "n_flat_excluded_from_T": int((~ok).sum())}


def bearing_indicators(name):
    """The five indicators of Table 6, built as study_p3_review.part_altind
    builds them (the paper's log RMS is checked against datasets.load_dataset)."""
    cfg = ds.DATASETS[name]
    files = sorted(glob.glob(os.path.join(cfg["dir"], "*.csv")))
    ref = ds.load_dataset(name)
    out = {}
    for ind, ((a, b), how) in R.ALT_IND.items():
        out[ind] = []
        for f in files:
            d = pd.read_csv(f)
            x, y = d[a].to_numpy(float), d[b].to_numpy(float)
            v = np.sqrt(x ** 2 + y ** 2) if how == "norm" else 0.5 * (x + y)
            v = ds._smooth(np.maximum(v, 1e-12), cfg["smooth"])
            nh = min(cfg["n_health"], max(5, int(0.05 * len(v))))
            hi = np.log(v / np.median(v[:nh]))
            if ind.startswith("log RMS"):
                assert np.allclose(hi, ref[os.path.basename(f)[:-4]]["hi"])
            out[ind].append(hi)
    return out


def part_hiq():
    out = {"bearing_indicators": {}, "archives": {}}
    altind = json.load(open(os.path.join(BE, "p3_review.json"), encoding="utf-8"))["altind"]
    for name in ("pronostia", "xjtu"):
        out["bearing_indicators"][name] = {}
        for ind, ser in bearing_indicators(name).items():
            q = hi_quality(ser)
            q["screen_verdict"] = altind[name][ind]["verdict"]
            out["bearing_indicators"][name][ind] = q
            log(f"{name:10s} {ind:22s} {q}")
    cens = json.load(open(OUT, encoding="utf-8")).get("cens", {})
    for k in TABLE8:
        if not os.path.exists(C.cache_path(k)):
            continue
        q = hi_quality([r["hi"] for r in C.load(k)])
        q["verdict"] = cens[k]["paper_verdict"] if k in cens else None
        out["archives"][k] = q
        log(f"{k:22s} {q}")
    #  does any metric separate declined from passed archives?
    sep = {}
    for m in ("monotonicity", "prognosability", "trendability_min", "trendability_median"):
        dec = [v[m] for v in out["archives"].values() if v["verdict"] == "declines" and v[m] is not None]
        ok = [v[m] for v in out["archives"].values() if v["verdict"] == "not declined" and v[m] is not None]
        if dec and ok:
            auc = float(np.mean([(o > d) + 0.5 * (o == d) for o in ok for d in dec]))
            sep[m] = {"declined": dec, "passed": ok, "auc_passed_higher": _f(auc),
                      "perfect_separation": auc in (0.0, 1.0)}
    out["separation"] = sep
    log("separation:", json.dumps(sep))
    #  per unit: path-model error on the C-MAPSS band-study units against the
    #  unit's monotonicity and its onset fraction
    import study_p3_review as RR
    ns = RR._thirdset_namespace()
    series = {}
    for fd in ("FD002", "FD004"):
        for u, arr in ns["load"](fd, by_regime=True).items():
            series[f"{fd}:{u}"] = ns["indicator"](arr)
    pu = {}
    for stem in ("bandtest", "bandtest_rep"):
        rows = json.load(open(os.path.join(BE, f"{stem}.json"), encoding="utf-8"))["rows"]
        by = {}
        for r in rows:
            by.setdefault(r["unit"], []).append(r)
        units = sorted(by)
        err = np.array([np.median([r["abs_pct_err"] for r in by[u]]) for u in units])
        cov = np.array([np.mean([r["covered"] for r in by[u]]) for u in units])
        mono = np.array([monotonicity(series[u]) for u in units])
        ons = np.array([by[u][0]["onset_frac"] for u in units])
        pu[stem] = {"n_units": len(units),
                    "spearman_err_onset": _f(stats.spearmanr(ons, err)[0]),
                    "spearman_err_monotonicity": _f(stats.spearmanr(mono, err)[0]),
                    "spearman_cov_onset": _f(stats.spearmanr(ons, cov)[0]),
                    "spearman_cov_monotonicity": _f(stats.spearmanr(mono, cov)[0]),
                    "spearman_onset_monotonicity": _f(stats.spearmanr(ons, mono)[0])}
        log(f"per unit {stem}: {pu[stem]}")
    out["per_unit_cmapss_band"] = pu
    #  bearing archives: path-model coverage (BS, informed, coupled) for reference
    rows = json.load(open(os.path.join(BE, "results_both.json"), encoding="utf-8"))["tables"]
    out["bearing_path_coverage95"] = {n: rows[f"{n}/process"]["BS"]["coverage95"]
                                      for n in ("pronostia", "xjtu") if f"{n}/process" in rows}
    log("bearing path coverage:", out["bearing_path_coverage95"])
    _save("hiq", out)


def part_seed():
    """How much the percentile-bootstrap bounds (convention a) move with the
    bootstrap seed alone: 200 seeds of 10,000 resamples each."""
    arch = onsets_table8()
    out = {}
    for k, v in arch.items():
        x = np.array([np.inf if f is None else f for f in v], float)
        lo, hi, ver = [], [], []
        for s in range(200):
            rng = np.random.default_rng(s)
            m = np.median(x[rng.integers(0, len(x), size=(B_BOOT, len(x)))], axis=1)
            a, b = np.quantile(m, [0.025, 0.975], method="inverted_cdf")
            lo.append(_pct(a))
            hi.append(_pct(b))
            ver.append(R._verdict3(_pct(a), _pct(b), THR))
        vals = lambda z: sorted({None if q is None else round(q, 1) for q in z},
                                key=lambda q: (q is None, q))
        cnt = {u: ver.count(u) / 200 for u in set(ver)}
        hs = [None if q is None else round(q, 1) for q in hi]
        #  the ideal (B -> infinity) percentile bound: P(resampled median >= x_(j))
        #  is a binomial tail, so the 97.5 % bound is the largest x_(j) whose
        #  tail exceeds 0.025
        xs = np.sort(x)
        n = len(xs)
        tail = [1 - stats.binom.cdf((n - 1) // 2, n, (n - j) / n) for j in range(n)]
        ub = [xs[j] for j in range(n) if tail[j] > 0.025]
        out_ub = _pct(max(ub)) if ub else None
        out[k] = {"lower_values": vals(lo), "upper_values": vals(hi), "verdict_shares": cnt,
                  "upper_value_shares": {str(u): hs.count(u) / 200 for u in set(hs)},
                  "upper_bound_B_infinity": out_ub,
                  "share_resampled_median_ge_70": _f(np.mean(np.median(
                      x[np.random.default_rng(0).integers(0, len(x), size=(B_BOOT, len(x)))], axis=1) >= 0.7))}
        log(f"{k:22s} lower {out[k]['lower_values']}  upper {out[k]['upper_values']}  verdicts {cnt}")
    _save("seed", out)


PARTS = {"sim": part_sim, "cens": part_cens, "thr": part_thr, "cusum": part_cusum, "hiq": part_hiq,
         "seed": part_seed}

if __name__ == "__main__":
    for p in sys.argv[1:] or list(PARTS):
        log(f"══ {p} ══ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        PARTS[p]()
