# -*- coding: utf-8 -*-
"""A two-phase (change-point) Wiener comparator for the late-onset claim.

Review issue D / R2 #5: "late onset -> no inference can predict" was shown only
for a single-phase, trend-extrapolating model fitted per record.  This script
fits the standard alternative, a two-phase degradation model with a population
prior on the change time, and scores it on exactly the prediction points of
Table 2 (PRONOSTIA n = 77, XJTU-SY n = 89) and on the C-MAPSS units of the band
test (Table 9, results/bearings/bandtest.json).

Model (on the paper's indicator y = log RMS ratio, smoothed as in Table 2)
------------------------------------------------------------------------
    phase 1, t <= tau :  y_t = b0 + e_t
    phase 2, t >  tau :  y_t = b0 + X(t - tau) + e_t,
                         X(s) = mu s + sigma_B W(s)      (Wiener with drift)
    e_t iid N(0, sigma^2)                                (measurement noise)
    failure: first passage of b0 + X to the threshold D.

A Wiener process on log RMS is the log-linear form of the exponential
degradation model (Gebraeel et al. 2005; Li et al. 2015), and the flat first
phase with a random change point is the two-phase / FPT construction of those
papers and of the two-phase Wiener literature (e.g. Wang et al. 2018).

Priors (all learned from the reference records only, which are the same ones
the paper uses for its threshold: the six PRONOSTIA learning bearings; the other
bearings of the same XJTU-SY condition; all other C-MAPSS units):
  * b0: median of the target's own healthy window (the first
    min(n_health, max(5, (k+1)//10)) points, as the onset rule).
  * sigma, primary ("two_phase"): max(1.4826 * MAD of that window, median over
    the references of the robust scale of their thinned PRE-ONSET segment
    about the healthy level).  Variant "two_phase_ownnoise": the window MAD
    only, the textbook choice.  The variant was run first; it detects a change
    almost everywhere because the healthy phase of these indicators wanders
    5-50x more than the short healthy window shows (both are reported).
  * tau ~ lognormal, fitted (mean and sd of log) to the reference records'
    retrospective onset TIMES (paper's onset rule on the complete record);
    a reference record with no detected onset is given tau at its 10th-last
    acquisition; the log-sd is floored at 0.25.
  * mu ~ N(m_mu, v_mu), m_mu = mean over references of
    (terminal level - b0) / (life - tau), sd = max(sample sd, 0.5 m_mu).
    Posterior draws of mu are truncated to mu > 0.
  * sigma_B^2 fixed at the median over references of the phase-2 quadratic
    variation net of measurement noise.
  * D: the paper's observed-scale threshold (datasets.threshold_for: median
    terminal indicator of the reference records; for C-MAPSS the band test's).

Inference at a prediction point k (exact, no MCMC)
--------------------------------------------------
The series up to k is thinned by the archive's smoothing window (15 / 9 / 1)
so the smoothed noise is close to independent.  For every candidate change
time on the thinned grid, mu is integrated out analytically (Gaussian
conjugacy: phase-2 data ~ N(m_mu s, sigma_B^2 min(s,s') + sigma^2 I
+ v_mu s s')), which gives the marginal likelihood of that tau and the
Gaussian posterior of mu and of X(t_k).  One further hypothesis, "tau > t_k"
(no change yet), has likelihood = all points flat, and prior mass = lognormal
survival at t_k.  The predictive RUL integrates over all of them by Monte
Carlo (4000 draws): if tau < t_k, RUL = inverse-Gaussian first passage from
X(t_k); if tau > t_k, RUL = (tau - t_k) + first passage from 0, tau drawn
from the prior truncated to tau > t_k.  So the model can predict failure
before any change is visible.  Point prediction = predictive median; interval
= 2.5 / 97.5 % quantiles; both capped at ten times the record length as the
path model is.

    python code/study_p3_rev_twophase.py [bearings] [cmapss]
        ->  results/bearings/p3_rev_twophase.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import datasets as ds                                             # noqa: E402
import study_p3_review as pr                                      # noqa: E402

BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_rev_twophase.json")
N_DRAW = 4000
LOGSD_FLOOR = 0.25
NO_ONSET_TAIL = 10


# ══════════════════════════════════════════════════════════════════════════
# population prior from reference records
# ══════════════════════════════════════════════════════════════════════════
def _healthy(hi, n_health, n_upto):
    nh = min(n_health, max(5, n_upto // 10))
    b0 = float(np.median(hi[:nh]))
    sig = float(1.4826 * np.median(np.abs(hi[:nh] - b0)))
    return b0, max(sig, 1e-3), nh


def ref_stats(rec, onset_idx, n_health, thin):
    """Onset time, phase-2 drift and diffusion of one completed record."""
    t, hi, n = rec["t"], rec["hi"], rec["n"]
    b0, sig, _ = _healthy(hi, n_health, n)
    i0 = onset_idx if onset_idx > 0 else max(n - NO_ONSET_TAIL, 1)
    tau = float(t[i0])
    L2 = float(t[-1] - tau)
    rise = float(np.median(hi[-3:]) - b0)
    mu = rise / max(L2, rec["dt"])
    idx = np.arange(i0, n, thin)
    y, tt = hi[idx], t[idx]
    if len(y) >= 3:
        dy, dtt = np.diff(y), np.diff(tt)
        qv = float(np.sum((dy - mu * dtt) ** 2) - 2 * sig ** 2 * len(dy)) / float(np.sum(dtt))
    else:
        qv = 0.0
    #  robust scale of the pre-onset (healthy) segment about the healthy level,
    #  at the thinned resolution: it includes the slow wander and level steps of
    #  the healthy phase, which the short healthy window does not see
    _, _, nh = _healthy(hi, n_health, n)
    pre = hi[nh:i0:thin] - b0
    pre_sig = float(1.4826 * np.median(np.abs(pre))) if len(pre) >= 3 else float("nan")
    return {"tau": tau, "mu": mu, "qv": max(qv, 1e-12), "detected": onset_idx > 0,
            "pre_sig": pre_sig}


def population_prior(stats_list):
    taus = np.log([max(s["tau"], 1.0) for s in stats_list])
    mus = np.array([s["mu"] for s in stats_list])
    m_mu = float(np.mean(mus))
    sd_mu = float(np.std(mus, ddof=1)) if len(mus) > 1 else 0.0
    return {"log_tau_m": float(np.mean(taus)),
            "log_tau_s": max(float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0, LOGSD_FLOOR),
            "m_mu": m_mu, "v_mu": max(sd_mu, 0.5 * abs(m_mu)) ** 2,
            "sB2": float(np.median([s["qv"] for s in stats_list])),
            "sig_pre": float(np.nanmedian([s["pre_sig"] for s in stats_list])),
            "n_ref": len(stats_list),
            "n_ref_detected": int(sum(s["detected"] for s in stats_list))}


# ══════════════════════════════════════════════════════════════════════════
# prediction at one point
# ══════════════════════════════════════════════════════════════════════════
def _tau_cdf(tau, P):
    return stats.norm.cdf((math.log(max(tau, 1e-9)) - P["log_tau_m"]) / P["log_tau_s"])


def predict(rec, k, D, P, n_health, thin, seed, noise="pooled"):
    """noise="own": sigma from the target's healthy window only (textbook);
    noise="pooled": sigma = max(own, median pre-onset scale of the references)."""
    rng = np.random.default_rng(seed)
    t, hi = rec["t"], rec["hi"]
    b0, sig, nh = _healthy(hi, n_health, k + 1)
    if noise == "pooled" and np.isfinite(P["sig_pre"]):
        sig = max(sig, P["sig_pre"])
    Dp = D - b0
    idx = np.unique(np.r_[np.arange(0, k + 1, thin), k])
    y, tt = hi[idx] - b0, t[idx]
    tk = float(t[k])
    m, v, sB2 = P["m_mu"], P["v_mu"], P["sB2"]
    ll_flat_each = stats.norm.logpdf(y, 0.0, sig)
    cum_flat = np.cumsum(ll_flat_each)

    #  candidate change times: thinned times from the healthy window to t_k
    j0 = int(np.searchsorted(tt, t[min(nh, k)]))
    cands = list(range(max(j0, 0), len(tt) - 1))
    logpost, cache = [], []
    edges = [_tau_cdf(tt[j], P) for j in cands] + [_tau_cdf(tk, P)]
    for c, j in enumerate(cands):
        mass = (edges[c + 1] - edges[c]) if c > 0 else edges[1]   # lump tau < first cand
        if mass <= 0:
            logpost.append(-np.inf)
            cache.append(None)
            continue
        s = tt[j + 1:] - tt[j]
        y2 = y[j + 1:]
        K = sB2 * np.minimum.outer(s, s) + sig ** 2 * np.eye(len(s))
        cf = cho_factor(K, lower=True)
        Ki_s, Ki_y = cho_solve(cf, s), cho_solve(cf, y2)
        sKs, sKy = float(s @ Ki_s), float(s @ Ki_y)
        r = y2 - m * s
        rKr = float(r @ cho_solve(cf, r))
        sKr = sKy - m * sKs
        logdet = 2 * np.sum(np.log(np.diag(cf[0]))) + math.log1p(v * sKs)
        quad = rKr - v * sKr ** 2 / (1 + v * sKs)
        ll2 = -0.5 * (logdet + quad + len(s) * math.log(2 * math.pi))
        ll = ll2 + cum_flat[j]
        prec = 1 / v + sKs
        mu_mean, mu_sd = (m / v + sKy) / prec, 1 / math.sqrt(prec)
        #  X(t_k) | mu: c = sB2 * s (s_k is the largest s), K^-1 c = sB2 Ki_s
        sk = s[-1]
        a = sk - sB2 * float(s @ Ki_s)            # coefficient of mu
        b = sB2 * float(s @ Ki_y)                 # constant
        var_x = max(sB2 * sk - sB2 ** 2 * float(s @ Ki_s), 1e-12)
        logpost.append(math.log(mass) + ll)
        cache.append((mu_mean, mu_sd, a, b, var_x, tt[j]))
    S_k = max(1.0 - edges[-1], 0.0)
    log_after = (math.log(S_k) if S_k > 0 else
                 stats.norm.logsf((math.log(tk) - P["log_tau_m"]) / P["log_tau_s"])) + cum_flat[-1]
    logpost.append(log_after)
    lp = np.array(logpost)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    p_after = float(w[-1])

    counts = rng.multinomial(N_DRAW, w)
    draws = []
    for c, nc in enumerate(counts):
        if nc == 0:
            continue
        if c == len(cache):          # tau > t_k: onset not yet happened
            u = rng.uniform(size=nc)
            z0 = (math.log(tk) - P["log_tau_m"]) / P["log_tau_s"]
            z = stats.norm.isf(u * stats.norm.sf(z0))
            z = np.maximum(z, z0)
            tau = np.exp(P["log_tau_m"] + P["log_tau_s"] * z)
            mu = _trunc_pos(m, math.sqrt(v), nc, rng)
            draws.append((tau - tk) + _fpt(np.zeros(nc), Dp, mu, sB2, rng))
        else:
            mu_mean, mu_sd, a, b, var_x, _ = cache[c]
            mu = _trunc_pos(mu_mean, mu_sd, nc, rng)
            x = a * mu + b + math.sqrt(var_x) * rng.standard_normal(nc)
            draws.append(_fpt(x, Dp, mu, sB2, rng))
    r = np.concatenate(draws)
    cap = 10.0 * float(t[-1])
    r = np.clip(r, 0.0, cap)
    lo, med, hi_ = np.quantile(r, [0.025, 0.5, 0.975])
    #  posterior mean change time among the "already changed" hypotheses
    tau_c = [cache[c][5] for c in range(len(cache)) if cache[c] is not None]
    w_c = w[:-1][[cache[c] is not None for c in range(len(cache))]]
    tau_mean = float(np.sum(np.array(tau_c) * w_c) / w_c.sum()) if w_c.sum() > 0 else None
    return {"pred": float(med), "mean": float(np.mean(r)), "lo": float(lo),
            "hi": float(hi_), "p_no_change": p_after,
            "tau_post_mean_if_changed": tau_mean, "capped": bool(med >= cap)}


def _trunc_pos(mean, sd, n, rng):
    a = (1e-9 - mean) / sd
    if a > 30:                      # posterior entirely negative: smallest positive drift
        return np.full(n, max(mean, 0) + 1e-12)
    return stats.truncnorm.rvs(a, np.inf, loc=mean, scale=sd, size=n, random_state=rng)


def _fpt(x, Dp, mu, sB2, rng):
    gap = Dp - x
    out = np.zeros(len(x))
    ok = gap > 0
    if ok.any():
        out[ok] = rng.wald(gap[ok] / mu[ok], gap[ok] ** 2 / sB2)
    return out


# ══════════════════════════════════════════════════════════════════════════
# bearings: exactly Table 2's points
# ══════════════════════════════════════════════════════════════════════════
def _summ(rows):
    s = pr._score(rows) if rows else {"n": 0}
    return s


def _split(rows):
    return {"all": _summ(rows),
            "onset_before_point": _summ([r for r in rows if r["onset_before"]]),
            "onset_after_point": _summ([r for r in rows if not r["onset_before"]])}


def part_bearings(out):
    both = json.load(open(os.path.join(BE, "results_both.json"), encoding="utf-8"))["rows"]
    app_all = json.load(open(os.path.join(BE, "applicability.json"), encoding="utf-8"))
    for name in ("pronostia", "xjtu"):
        t0 = time.time()
        cfg = ds.DATASETS[name]
        thin, n_health = cfg["smooth"], cfg["n_health"]
        data = ds.load_dataset(name)
        app = {r["bearing"]: r for r in app_all if r["dataset"] == name}
        onset = {b: ds.first_prediction_time(r["hi"]) for b, r in data.items()}
        lives = {b: float(r["t"][-1]) for b, r in data.items()}
        rs = {b: ref_stats(data[b], onset[b], n_health, thin) for b in data}
        #  path model (Table 2 row 1): BS, informed, coupling, every point
        path = {}
        for r in both:
            if (r["dataset"] == name and r["law"] == "BS" and r["informed"]
                    and r["coupling"]):
                #  the first occurrence is the main run (later ones are the
                #  ablation/grid reruns); this reproduces Table 2 MAE 5090 / 13355
                path.setdefault((r["bearing"], r["k"]), r)
        rows = {"two_phase": [], "two_phase_ownnoise": [], "path_BS": [], "life_table": []}
        priors = {}
        for b, pts in ds.evaluation_bearings(data, name):
            rec = data[b]
            D, ref = ds.threshold_for(data, b, name)
            P = population_prior([rs[r] for r in ref])
            priors[b] = {**P, "D": D, "ref": ref}
            for k in pts:
                if k < 20 or k >= rec["n"] - 1:
                    continue
                true = float(rec["t"][-1] - rec["t"][k])
                if true <= 0:
                    continue
                lf = float(rec["t"][k] / rec["life"])
                before = bool(app[b]["fpt_idx"] > 0 and app[b]["fpt_frac"] < lf)
                base = {"bearing": b, "k": k, "true": true, "life_frac": lf,
                        "onset_before": before}
                q = predict(rec, k, D, P, n_health, thin, seed=k * 7919 + len(b))
                rows["two_phase"].append({**base, **q})
                q0 = predict(rec, k, D, P, n_health, thin, seed=k * 7919 + len(b), noise="own")
                rows["two_phase_ownnoise"].append({**base, **q0})
                pm = path[(b, k)]
                rows["path_BS"].append({**base, "pred": pm["pred"], "lo": pm["lo"], "hi": pm["hi"]})
                p, l, h = pr._life_table([lives[r] for r in ref], float(rec["t"][k]))
                rows["life_table"].append({**base, "pred": p, "lo": l, "hi": h})
                print(f"  {name} {b} k={k} lf={lf:.2f} {'B' if before else 'A'} "
                      f"true={true:8.0f} 2ph={q['pred']:8.0f} [{q['lo']:.0f},{q['hi']:.0f}] "
                      f"P(no change)={q['p_no_change']:.2f}  life={p:8.0f}", flush=True)
        res = {m: _split(v) for m, v in rows.items()}
        sides = (("onset_before_point", True), ("onset_after_point", False))
        for var in ("two_phase", "two_phase_ownnoise"):
            tp = rows[var]
            res[var + "_extra"] = {
                "median_p_no_change": {
                    side: float(np.median([r["p_no_change"] for r in tp if r["onset_before"] == sb_]))
                    for side, sb_ in sides if any(r["onset_before"] == sb_ for r in tp)},
                #  where the interval misses, per side: true below lo = life
                #  predicted too long; true above hi = too short
                "share_true_below_lo": {
                    side: float(np.mean([r["true"] < r["lo"] for r in tp if r["onset_before"] == sb_]))
                    for side, sb_ in sides if any(r["onset_before"] == sb_ for r in tp)},
                "share_true_above_hi": {
                    side: float(np.mean([r["true"] > r["hi"] for r in tp if r["onset_before"] == sb_]))
                    for side, sb_ in sides if any(r["onset_before"] == sb_ for r in tp)},
                "share_capped": float(np.mean([r["capped"] for r in tp])),
                "MAE_using_mean": float(np.mean([abs(r["mean"] - r["true"]) for r in tp])),
                #  per-point agreement with the life table on late-onset points
                "late_corr_with_life_table": float(np.corrcoef(
                    [r["pred"] for r in tp if not r["onset_before"]],
                    [r["pred"] for r in rows["life_table"] if not r["onset_before"]])[0, 1])}
        out[name] = {"scores": res, "priors": priors, "rows": rows,
                     "seconds": time.time() - t0}
        for m, v in res.items():
            if m.endswith("_extra"):
                continue
            for side, s in v.items():
                if s.get("n"):
                    print(f"{name:9s} {m:18s} {side:19s} n={s['n']:3d} MAE={s['MAE']:8.0f} "
                          f"MedAPE={s['MedAPE_pct']:6.1f} cov={s['coverage95']:.2f}")
        print(json.dumps({m: v for m, v in res.items() if m.endswith("_extra")}, indent=1))
        _save(out)


# ══════════════════════════════════════════════════════════════════════════
# C-MAPSS: the band-test units and points (bandtest.json rows)
# ══════════════════════════════════════════════════════════════════════════
def part_cmapss(out):
    t0 = time.time()
    ns = pr._thirdset_namespace()           # avoids study_thirdset's import-time run
    bt = json.load(open(os.path.join(BE, "bandtest.json"), encoding="utf-8"))
    units = {}
    for fd in ("FD002", "FD004"):
        for u, arr in ns["load"](fd, by_regime=True).items():
            hi = ns["indicator"](arr)
            k = ns["onset"](hi)
            if k is None or len(hi) < 60:
                continue
            n = len(hi)
            units[f"{fd}:{u}"] = ({"t": np.arange(n, dtype=float), "hi": hi, "condition": 0,
                                   "life": float(n - 1), "n": n, "dt": 1.0}, k)
    n_health = ns["N_HEALTH"]
    rs = {key: ref_stats(rec, k, n_health, 1) for key, (rec, k) in units.items()}
    terminal = {key: float(np.median(rec["hi"][-3:])) for key, (rec, _) in units.items()}
    lives = {key: rec["t"][-1] for key, (rec, _) in units.items()}
    rows = []
    for br in bt["rows"]:
        key, k = br["unit"], br["k"]
        rec, _ = units[key]
        ref = [u for u in units if u != key]
        D = float(np.median([terminal[u] for u in ref]))
        P = population_prior([rs[u] for u in ref])
        true = float(rec["t"][-1] - rec["t"][k])
        assert abs(true - br["true"]) < 1e-9, (key, k)
        q = predict(rec, k, D, P, n_health, 1, seed=k * 7919 + len(key))
        q0 = predict(rec, k, D, P, n_health, 1, seed=k * 7919 + len(key), noise="own")
        p, l, h = pr._life_table([lives[u] for u in ref], float(rec["t"][k]))
        rows.append({"unit": key, "bin": br["bin"], "onset_frac": br["onset_frac"],
                     "at_life": br["at_life"], "k": k, "true": true,
                     "onset_before": bool(br["onset_frac"] < br["at_life"]),
                     "two_phase": q, "two_phase_ownnoise": q0,
                     "life_table": {"pred": p, "lo": l, "hi": h},
                     "path_BS": {"pred": br["pred"], "covered": br["covered"]}})
    def agg(sel):
        if not sel:
            return {"n": 0}
        o = {"n": len(sel)}
        for m in ("two_phase", "two_phase_ownnoise", "life_table", "path_BS"):
            ape = [100 * abs(r[m]["pred"] - r["true"]) / r["true"] for r in sel]
            cov = [r[m]["covered"] if m == "path_BS" else r[m]["lo"] <= r["true"] <= r[m]["hi"]
                   for r in sel]
            o[m] = {"MedAPE_pct": float(np.median(ape)), "coverage95": float(np.mean(cov)),
                    "MAE": float(np.mean([abs(r[m]["pred"] - r["true"]) for r in sel]))}
        for m in ("two_phase", "two_phase_ownnoise"):
            o[m]["median_p_no_change"] = float(np.median([r[m]["p_no_change"] for r in sel]))
        return o
    summ = {"by_bin": {b: agg([r for r in rows if r["bin"] == b])
                       for b in sorted({r["bin"] for r in rows})},
            "onset_before_point": agg([r for r in rows if r["onset_before"]]),
            "onset_after_point": agg([r for r in rows if not r["onset_before"]]),
            "all": agg(rows)}
    out["cmapss_bandtest"] = {"summary": summ, "rows": rows,
                              "n_units_pool": len(units), "seconds": time.time() - t0}
    for b, v in list(summ["by_bin"].items()) + [("before", summ["onset_before_point"]),
                                                 ("after", summ["onset_after_point"])]:
        print(f"  C-MAPSS {b:10s} n={v['n']:3d} " + "  ".join(
            f"{m}: MedAPE {v[m]['MedAPE_pct']:6.1f} cov {v[m]['coverage95']:.2f}"
            for m in ("two_phase", "two_phase_ownnoise", "life_table", "path_BS")))
    _save(out)


def part_width(out):
    """Median relative interval width (hi - lo) / true, per method and onset
    side, from the stored rows (no recomputation).  Coverage bought by width
    is not prediction, so the two are reported together."""
    W = {}
    for name in ("pronostia", "xjtu"):
        if name not in out:
            continue
        rows = out[name]["rows"]
        W[name] = {m: {side: float(np.median([(r["hi"] - r["lo"]) / r["true"] for r in v
                                              if r["onset_before"] == sb_]))
                       for side, sb_ in (("onset_before_point", True), ("onset_after_point", False))}
                   for m, v in rows.items()}
    if "cmapss_bandtest" in out:
        rows = out["cmapss_bandtest"]["rows"]
        W["cmapss_bandtest"] = {m: {side: float(np.median([(r[m]["hi"] - r[m]["lo"]) / r["true"]
                                                           for r in rows if r["onset_before"] == sb_]))
                                    for side, sb_ in (("onset_before_point", True),
                                                      ("onset_after_point", False))}
                                for m in ("two_phase", "two_phase_ownnoise", "life_table")}
    out["median_rel_interval_width"] = W
    print(json.dumps(W, indent=1))
    _save(out)


def _save(out):
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    out["config"] = {"N_DRAW": N_DRAW, "LOGSD_FLOOR": LOGSD_FLOOR,
                     "NO_ONSET_TAIL": NO_ONSET_TAIL, "point": "predictive median",
                     "interval": "2.5-97.5 % predictive quantiles", "cap": "10 x record length"}
    todo = [a for a in sys.argv[1:] if a in ("bearings", "cmapss", "width")] \
        or ["bearings", "cmapss", "width"]
    if "bearings" in todo:
        part_bearings(out)
    if "cmapss" in todo:
        part_cmapss(out)
    if "width" in todo:
        part_width(out)
