# -*- coding: utf-8 -*-
r"""Analyses added in response to the review of paper 3.

ci    Uncertainty of the screen's inputs.  A percentile bootstrap (10,000
      resamples) of the median detected onset, and a Wilson interval for the
      detection rate, for every archive in Table tab:archives.  The C-MAPSS
      per-unit onsets are recomputed with study_thirdset's own functions and
      checked against thirdset.json.

acf   Is the observation model the one the filter assumes?  Under
      Y_k = X_k + eps_k with independent eps, the first differences of Y have
      lag-1 autocorrelation -s2 / (Var dX + 2 s2), which lies in (-1/2, 0) and
      approaches -1/2 where the state is flat.  Measured on the pre-onset window
      of every bearing record, for the smoothed indicator the paper uses and for
      the raw one.

ppc   One-step posterior predictive check of the path model.  The filter is run
      exactly as in study_bearings (BS law, informed prior, coupling, N_p 1200,
      start at the causal onset) up to 90 % of each evaluation record's life.
      Before each observation is assimilated, the cloud is propagated one step
      through the transition and the predictive CDF evaluated at the new
      observation, u_k = sum_i w_i Phi((y_k - x_i - dX_i) / sigma_i).  Under a
      correct model u_k is uniform and independent.  The same check on records
      simulated from the model, raw and after the paper's causal smoothing,
      separates what the smoothing alone does.

cal   Coverage on data from the model after the indicator's causal smoothing:
      the calibration study of study_theory.t5 repeated with smoothed
      observations, to bound how much of the benchmark shortfall smoothing alone
      can explain.

reg   The decision the screen recommends.  A gradient-boosting regression of
      log RUL on the 32 causal features of rul_dataset, and a naive life-table
      baseline, scored at the path model's own prediction points with its own
      training sets (PRONOSTIA learning -> test; XJTU-SY same-regime
      leave-one-out).

sens  Path-model sensitivity, BS law only, every prediction point, the main
      study's seeds: (i) the baseline re-run, (ii) the raw unsmoothed indicator,
      (iii) four times the particles.

cens  Records without a detected onset counted as right-censored at the end of
      life: the censored (Kaplan--Meier) median onset and the share of records
      with an onset before 70 % of life, per archive.

hw    The detector's healthy window opened and narrowed (cap 10-160, a fifth to
      a twentieth of the record) on the bearing and turbofan archives.

boot  Path model minus each no-threshold predictor, in mean absolute error and
      coverage, with a bootstrap clustered on the bearing.

dec   The epistemic ratio with inverse Gaussian moments for every particle, as
      the code computes it, against simulated-path moments for the particles
      whose predictive law is simulated (n_d < 10).

sens2 The inference choices that could still explain the benchmark failure:
      the noise prior loosened (log-scale s.d. 0.5 and 1.5 for 0.15), the
      roughening floor at c = 0.01 and 0.20 for 0.05, and AR(1) observation
      noise for the independent noise of eq. obs.

exact A reference that targets the exact posterior of the BS path model, on a
      tractable subset: per evaluation record, the prediction point nearest 70 %
      of life.  The Liu--West filter approximates the posterior of the static
      parameters; this samples it by particle marginal Metropolis--Hastings,
      whose bootstrap-filter likelihood estimate is unbiased, so the chains
      target the exact posterior for any number of particles.  Eight chains
      start at a mode located by CMA-ES (the posterior can sit many prior
      standard deviations from the main filter's cloud), the particle count is
      set per point so that log Z has a standard deviation of at most 1, and
      the random-walk proposal adapts during burn-in.  256 posterior draws, each
      with its own fixed-theta filter, then go through gbs.predict_rul exactly
      as in study_bearings.predict_one.  Run on the raw indicator, whose noise
      the independent observation model describes, so that the observation
      model is right as well as the inference.  On the smoothed indicator,
      nearly noiseless, the chains do not mix: R-hat 16 and 28 on the first
      XJTU-SY and PRONOSTIA points tried, at up to 4096 particles, so that
      variant is available ("smoothed") but not reported.  "exact_sum" pools
      the shards.

lead  Time from the detected onset to failure in each archive's own unit.

    python code/study_p3_review.py [ci acf ppc cal reg sens cens hw boot dec sens2 exact exact_sum lead]
        ->  results/bearings/p3_review.json
"""
from __future__ import annotations

import io
import json
import math
import os
import statistics as st
import sys

import numpy as np
from scipy import special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import datasets as ds                                             # noqa: E402
import gbs                                                        # noqa: E402
import study_bearings as sb                                       # noqa: E402

BE = os.path.join(ROOT, "results", "bearings")
#  parts run as separate processes write separate files, merged with "merge"
OUT = os.path.join(BE, os.environ.get("P3_REVIEW_OUT", "p3_review.json"))
B_BOOT = 10_000


def _save(key, val):
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res[key] = val
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


# ══════════════════════════════════════════════════════════════════════════
# ci
# ══════════════════════════════════════════════════════════════════════════
def _thirdset_namespace():
    """study_thirdset's constants and functions, without running its study.

    The module runs its loop at import and rewrites thirdset.json, so only the
    part above the loop is executed here.
    """
    src = io.open(os.path.join(HERE, "study_thirdset.py"), encoding="utf-8").read()
    head = src[:src.index("RUNS = [")]
    ns = {"__file__": os.path.join(HERE, "study_thirdset.py")}
    exec(compile(head, "study_thirdset_head", "exec"), ns)
    return ns


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def boot_median(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


def part_ci():
    rng = np.random.default_rng(20261012)
    arch = {}
    app = json.load(open(os.path.join(BE, "applicability.json"), encoding="utf-8"))
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        rows = [r for r in app if r["dataset"] == name]
        pos = [100 * r["fpt_frac"] for r in rows if r["fpt_idx"] > 0]
        arch[label] = (pos, len(rows))
    ns = _thirdset_namespace()
    ref = json.load(open(os.path.join(BE, "thirdset.json"), encoding="utf-8"))
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", False),
                   ("FD004", False), ("FD002", True), ("FD004", True)):
        units = ns["load"](fd, by_regime=by)
        key = fd + ("+regime" if by else "")
        pos = []
        for u, arr in sorted(units.items()):
            hi = ns["indicator"](arr)
            i = ns["onset"](hi)
            if i > 0:
                pos.append(100.0 * i / len(hi))
        assert len(pos) == ref[key]["n_detected"], key
        assert abs(st.median(pos) - ref[key]["median_onset_pct"]) < 1e-9, key
        arch["C-MAPSS " + key] = (pos, len(units))
    bat = json.load(open(os.path.join(BE, "battery.json"), encoding="utf-8"))
    arch["NASA cells"] = ([c["onset_pct"] for c in bat["cells"]
                           if c["onset_idx"] and c["onset_idx"] > 0], bat["n_cells"])
    hdd = json.load(open(os.path.join(BE, "hdd.json"), encoding="utf-8"))
    arch["Backblaze drives"] = ([100 * r["onset_frac"] for r in hdd["drives"]
                                 if r["onset_frac"] is not None], hdd["n_usable"])

    out = {}
    print("archive                 n   det   rate [95%]          median onset [95%]")
    for k, (pos, n) in arch.items():
        lo, hi = wilson(len(pos), n)
        mlo, mhi = boot_median(pos, rng)
        frac_band = float(np.mean([70 <= p < 90 for p in pos]))
        out[k] = {"n": n, "n_detected": len(pos), "rate": len(pos) / n,
                  "rate_ci": [lo, hi], "median_onset_pct": float(np.median(pos)),
                  "median_ci": [mlo, mhi], "frac_detected_in_band": frac_band,
                  "frac_detected_ge70": float(np.mean([p >= 70 for p in pos]))}
        print(f"{k:<22}{n:>4}{len(pos):>6}   {len(pos) / n:5.2f} [{lo:.2f},{hi:.2f}]"
              f"     {np.median(pos):5.1f} [{mlo:5.1f},{mhi:5.1f}]")
    _save("ci", out)


# ══════════════════════════════════════════════════════════════════════════
# acf
# ══════════════════════════════════════════════════════════════════════════
def _lag1(v):
    v = np.asarray(v, float) - np.mean(v)
    d = np.dot(v, v)
    return float(np.dot(v[1:], v[:-1]) / d) if d > 0 else float("nan")


def _hi_with_smoothing(name, w):
    keep = ds.DATASETS[name]["smooth"]
    ds.DATASETS[name]["smooth"] = w
    try:
        return ds.load_dataset(name)
    finally:
        ds.DATASETS[name]["smooth"] = keep


def part_acf():
    out = {}
    for name in ("pronostia", "xjtu"):
        sm = ds.load_dataset(name)
        raw = _hi_with_smoothing(name, 1)
        rows = []
        for b, rec in sorted(sm.items()):
            n = rec["n"]
            fpt = ds.first_prediction_time(rec["hi"])
            end = fpt if fpt > 0 else n
            nh = min(40, max(5, n // 10))
            lo = max(nh, ds.DATASETS[name]["smooth"])
            if end - lo < 20:
                continue
            r_sm = _lag1(np.diff(rec["hi"][lo:end]))
            r_raw = _lag1(np.diff(raw[b]["hi"][lo:end]))
            rows.append({"bearing": b, "n_window": int(end - lo),
                         "lag1_smoothed": r_sm, "lag1_raw": r_raw})
        out[name] = {"records": rows,
                     "median_lag1_smoothed": float(np.median([r["lag1_smoothed"] for r in rows])),
                     "median_lag1_raw": float(np.median([r["lag1_raw"] for r in rows])),
                     "n_records": len(rows),
                     "n_raw_in_model_range": int(sum(-0.5 <= r["lag1_raw"] <= 0 for r in rows)),
                     "n_smoothed_in_model_range": int(sum(-0.5 <= r["lag1_smoothed"] <= 0
                                                          for r in rows))}
        print(f"{name}: {len(rows)} records; median lag-1 autocorrelation of dY "
              f"smoothed {out[name]['median_lag1_smoothed']:+.3f}, raw "
              f"{out[name]['median_lag1_raw']:+.3f}; in (-1/2,0): smoothed "
              f"{out[name]['n_smoothed_in_model_range']}, raw {out[name]['n_raw_in_model_range']}")
    _save("acf", out)


# ══════════════════════════════════════════════════════════════════════════
# ppc
# ══════════════════════════════════════════════════════════════════════════
ECDF_GRID = np.linspace(0.0, 1.0, 101)


def _ecdf(u):
    """The empirical distribution of one record's PIT values on a fixed grid,
    so that records can be averaged with equal weight."""
    u = np.sort(np.asarray(u, float))
    return np.searchsorted(u, ECDF_GRID, side="right") / len(u)


def _pit_summary(u):
    u = np.clip(np.asarray(u, float), 1e-9, 1 - 1e-9)
    z = special.ndtri(u)
    ks = stats.kstest(u, "uniform").statistic
    return {"n": int(len(u)), "ks": float(ks), "ks_sqrtn": float(ks * math.sqrt(len(u))),
            "tail95": float(np.mean((u < 0.025) | (u > 0.975))),
            "lo025": float(np.mean(u < 0.025)), "hi975": float(np.mean(u > 0.975)),
            "lo001": float(np.mean(u < 0.001)), "u_median": float(np.median(u)),
            "z_mean": float(np.mean(z)), "z_sd": float(np.std(z)),
            "z_lag1": _lag1(z)}


def _filter_pit(pf, st0, y, t, dt_ref, rng, k_from=1):
    """Run the filter over y, recording the one-step predictive PIT at each step."""
    stt, pits = st0, []
    for k in range(1, len(y)):
        dt = t[k] - t[k - 1]
        if k >= k_from:
            w = np.exp(stt.logw - stt.logw.max())
            w /= w.sum()
            th = pf._theta(stt.logth)
            s = pf._scale(stt.x, dt, dt_ref, th)
            dx = pf.law.sample(th, s, rng)
            dx = np.where(np.isfinite(dx), dx, 0.0)
            xp = np.clip(stt.x + dx, -np.inf, pf.cfg.x_max)
            sig = np.maximum(th["sigma_eps"], pf.cfg.sigma_floor)
            pits.append(float(w @ special.ndtr((y[k] - xp) / sig)))
        stt = pf.step(stt, y[k], dt, dt_ref)
    return pits


def _calibrated_D(data, tag, b, cache, name):
    """The latent-scale threshold exactly as study_bearings.run computes it."""
    D_obs, ref = ds.threshold_for(data, b, name)
    med = float(np.median([max(data[r]["n"] - ds.first_prediction_time(data[r]["hi"]), 20)
                           for r in ref]))
    key = (tag, tuple(ref))
    if key not in cache:
        vals = [sb.latent_terminal(data[r], "BS", True, True, med, D_obs, seed=7)
                for r in ref]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        cache[key] = float(np.median(vals)) if vals else D_obs
    return cache[key], med


SUMMARY = ("ks", "ks_sqrtn", "tail95", "lo025", "hi975", "lo001", "u_median",
           "z_mean", "z_sd", "z_lag1")


def part_ppc():
    out, cache = {}, {}
    for name, tag, w in (("pronostia", "pronostia", None), ("xjtu", "xjtu", None),
                         ("pronostia", "pronostia_raw", 1), ("xjtu", "xjtu_raw", 1)):
        data = ds.load_dataset(name) if w is None else _hi_with_smoothing(name, w)
        recs, ecdfs = [], []
        for b, _ in ds.evaluation_bearings(data, name):
            rec = data[b]
            D, med = _calibrated_D(data, tag, b, cache, name)
            k_pred = int(0.9 * (rec["n"] - 1))
            # the set-up of study_bearings._filter, line for line
            law = sb.LAWS["BS"]()
            cfg = gbs.PFConfig(n_particles=sb.N_PART, delta=sb.DELTA,
                               seed=sb.stable_seed(b, k_pred, "BS"), gamma_coupling=True,
                               x_ref=max(0.25 * D, 1e-3), x_max=50.0 * max(D, 1e-3),
                               x_couple_max=max(D, 1e-3))
            pf = gbs.ParticleFilter(law, cfg)
            y_all = rec["hi"][:k_pred + 1]
            nh = max(5, min(40, len(y_all) // 10))
            sigma0 = max(float(np.std(y_all[:nh])), 1e-3)
            fpt = ds.first_prediction_time(y_all)
            k0 = int(np.clip(fpt, 0, max(0, k_pred - 15)))
            y = y_all[k0:]
            t = rec["t"][k0:k_pred + 1]
            m, s = sb.make_priors(D, med, True, True, sigma0)
            m, s = sb.adapt_prior(law, m, s, D, med)
            x0 = float(np.median(y[:max(3, len(y) // 20)]))
            st0 = pf.init(m, s, x0)
            pits = _filter_pit(pf, st0, y, t, rec["dt"], np.random.default_rng(1),
                               k_from=min(10, len(y) - 1))
            if len(pits) < 10:
                continue
            r = _pit_summary(pits)
            r.update({"bearing": b, "k0": k0, "k_pred": k_pred})
            recs.append(r)
            ecdfs.append(_ecdf(pits))
            print(f"  {tag} {b}: n={r['n']:5d} KS={r['ks']:.3f} tail={r['tail95']:.3f} "
                  f"z mean={r['z_mean']:+.2f} sd={r['z_sd']:.2f} lag1={r['z_lag1']:+.2f}")
        out[tag] = {"records": recs, "ecdf": np.mean(ecdfs, axis=0).tolist(),
                    **{f"median_{k}": float(np.nanmedian([r[k] for r in recs]))
                       for k in SUMMARY}}
        print(tag, {k: round(v, 3) for k, v in out[tag].items() if k.startswith("median")})

    # the same check on data from the model, raw and smoothed as the indicator is
    import study_theory as T
    for tag, w in (("synthetic", 1), ("synthetic_smoothed9", 9), ("synthetic_smoothed15", 15)):
        recs, ecdfs = [], []
        for rep in range(24):
            r_ = np.random.default_rng(500 + rep)
            x, y, t = T._synth(r_, K=320)
            if w > 1:
                y = ds._smooth(y, w)
            pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=sb.N_PART, seed=rep))
            st0 = pf.init(T.PRIOR_M, T.PRIOR_S, y[0])
            pits = _filter_pit(pf, st0, y, t, 1.0, np.random.default_rng(1), k_from=10)
            recs.append(_pit_summary(pits))
            ecdfs.append(_ecdf(pits))
        out[tag] = {"records": recs, "ecdf": np.mean(ecdfs, axis=0).tolist(),
                    **{f"median_{k}": float(np.nanmedian([r[k] for r in recs]))
                       for k in SUMMARY}}
        print(tag, {k: round(v, 3) for k, v in out[tag].items() if k.startswith("median")})
    _save("ppc", out)


# ══════════════════════════════════════════════════════════════════════════
# cal
# ══════════════════════════════════════════════════════════════════════════
def part_cal():
    import study_theory as T
    out = {}
    for w in (1, 9, 15):
        rows = []
        for rep in range(60):
            r = np.random.default_rng(200 + rep)
            x, y, t = T._synth(r, K=320)
            if w > 1:
                y = ds._smooth(y, w)
            D = 0.75 * x[-1]
            kf = int(np.argmax(x >= D))
            if kf < 60:
                continue
            pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=1500, seed=rep))
            states = pf.run(y[:kf], t[:kf], 1.0, T.PRIOR_M, T.PRIOR_S, x0=y[0])
            for frac in (0.4, 0.6, 0.8):
                k = int(frac * kf)
                if k < 25:
                    continue
                p = gbs.predict_rul(states[k - 1], pf, D, 1.0)
                lo, hi = np.quantile(p["samples"], [0.025, 0.975])
                true = t[kf] - t[k]
                rows.append({"frac": frac, "cov": bool(lo <= true <= hi),
                             "ape": float(abs(p["mean"] - true) / true)})
        out[str(w)] = {str(f): {"n": len([r for r in rows if r["frac"] == f]),
                                "coverage": float(np.mean([r["cov"] for r in rows if r["frac"] == f])),
                                "medape": float(100 * np.median([r["ape"] for r in rows
                                                                 if r["frac"] == f]))}
                       for f in (0.4, 0.6, 0.8)}
        print(f"smoothing window {w:2d}: " + "  ".join(
            f"{f}: cov {v['coverage']:.3f} MedAPE {v['medape']:.1f}"
            for f, v in out[str(w)].items()))
    _save("cal", out)


# ══════════════════════════════════════════════════════════════════════════
# reg
# ══════════════════════════════════════════════════════════════════════════
def _life_table(lives, t):
    """Median and 95 % range of residual life among training records alive at t."""
    res = np.array([L - t for L in lives if L > t])
    if len(res) == 0:
        return 0.0, 0.0, 0.0
    return float(np.median(res)), float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def _score(rows):
    e = np.array([r["pred"] - r["true"] for r in rows])
    tr = np.array([r["true"] for r in rows])
    return {"n": len(rows), "MAE": float(np.mean(np.abs(e))),
            "MedAPE_pct": float(np.median(np.abs(e / tr)) * 100),
            "coverage95": float(np.mean([r["lo"] <= r["true"] <= r["hi"] for r in rows])),
            "PHM_score": float(np.mean(ds.phm_score(100 * (tr - np.array([r["pred"] for r in rows])) / tr)))}


def _reg_rows(name):
    """Per-point predictions of the two no-threshold predictors on one archive."""
    import rul_dataset as R
    from sklearn.ensemble import GradientBoostingRegressor
    data = ds.load_dataset(name)
    evals = ds.evaluation_bearings(data, name)
    lives = {b: float(r["t"][-1]) for b, r in data.items()}
    rows = {"gbr": [], "life_table": []}
    fitted = {}
    for b, pts in evals:
        _, ref = ds.threshold_for(data, b, name)
        key = tuple(ref)
        t_scale = float(np.median([lives[r] for r in ref]))
        if key not in fitted:
            Xs, ys, ws = [], [], []
            for r in ref:
                X, rul, t, n, nh = R.build_bearing(r, name, t_scale)
                k0 = max(20, nh, int(0.05 * n))
                idx = np.arange(k0, n - 1)
                Xs.append(X[idx])
                ys.append(np.log(rul[idx]))
                ws.append(np.full(len(idx), 1.0 / len(idx)))
            X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
            kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, random_state=0)
            fitted[key] = {q: GradientBoostingRegressor(
                loss="squared_error" if q is None else "quantile",
                **({} if q is None else {"alpha": q}), **kw).fit(X, y, sample_weight=w)
                for q in (None, 0.025, 0.975)}
        mdl = fitted[key]
        Xb, rul_b, tb, nb, _ = R.build_bearing(b, name, t_scale)
        rec = data[b]
        assert nb == rec["n"], (b, nb, rec["n"])
        for k in pts:
            if k < 20 or k >= rec["n"] - 1:
                continue
            true = float(rec["t"][-1] - rec["t"][k])
            xk = Xb[k:k + 1]
            pred = float(np.exp(mdl[None].predict(xk)[0]))
            lo = float(np.exp(mdl[0.025].predict(xk)[0]))
            hi = float(np.exp(mdl[0.975].predict(xk)[0]))
            rows["gbr"].append({"bearing": b, "k": k, "true": true, "pred": pred,
                                "lo": min(lo, hi), "hi": max(lo, hi)})
            p, l, h = _life_table([lives[r] for r in ref], float(rec["t"][k]))
            rows["life_table"].append({"bearing": b, "k": k, "true": true,
                                       "pred": p, "lo": l, "hi": h})
    return rows


def part_reg():
    out = {}
    for name in ("pronostia", "xjtu"):
        rows = _reg_rows(name)
        out[name] = {m: _score(v) for m, v in rows.items()}
        for m, v in out[name].items():
            print(f"{name:10s} {m:11s} n={v['n']:3d} MAE={v['MAE']:8.0f} "
                  f"MedAPE={v['MedAPE_pct']:6.1f} cov={v['coverage95']:.2f} "
                  f"PHM={v['PHM_score']:.3f}")
    _save("reg", out)


# ══════════════════════════════════════════════════════════════════════════
# sens
# ══════════════════════════════════════════════════════════════════════════
def part_sens():
    out = {}
    variants = (("baseline", 1, sb.N_PART), ("raw_indicator", 0, sb.N_PART),
                ("particles_x4", 1, 4 * sb.N_PART))
    which = [v for v in variants if v[0] in sys.argv] or variants
    prev = json.load(open(OUT, encoding="utf-8")).get("sens", {}) if os.path.exists(OUT) else {}
    out.update(prev)
    for tag, smooth, n_part in which:
        out[tag] = {}
        for name in ("pronostia", "xjtu"):
            keep = ds.DATASETS[name]["smooth"]
            if not smooth:
                ds.DATASETS[name]["smooth"] = 1
            try:
                rows = sb.run(name, laws=("BS",), n_part=n_part, tag=tag)
            finally:
                ds.DATASETS[name]["smooth"] = keep
            s = sb.summarise(rows)["BS"]
            out[tag][name] = s
            print(f"{tag:14s} {name:10s} n={s['n']:3d} MAE={s['MAE']:8.0f} "
                  f"MedAPE={s['MedAPE_pct']:6.1f} cov={s['coverage95']:.2f} "
                  f"PHM={s['PHM_score']:.3f}", flush=True)
        _save("sens", out)


# ══════════════════════════════════════════════════════════════════════════
# cens   records without a detected onset, as right-censored observations
# ══════════════════════════════════════════════════════════════════════════
def _onset_records():
    """{archive: [onset fraction, or None where no onset is detected]}."""
    arch = {}
    app = json.load(open(os.path.join(BE, "applicability.json"), encoding="utf-8"))
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        arch[label] = [r["fpt_frac"] if r["fpt_idx"] > 0 else None
                       for r in app if r["dataset"] == name]
    ns = _thirdset_namespace()
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", False),
                   ("FD004", False), ("FD002", True), ("FD004", True)):
        vals = []
        for u, arr in sorted(ns["load"](fd, by_regime=by).items()):
            hi = ns["indicator"](arr)
            i = ns["onset"](hi)
            vals.append(i / len(hi) if i > 0 else None)
        arch["C-MAPSS " + fd + ("+regime" if by else "")] = vals
    bat = json.load(open(os.path.join(BE, "battery.json"), encoding="utf-8"))
    arch["NASA cells"] = [c["onset_pct"] / 100 if c["onset_idx"] and c["onset_idx"] > 0
                          else None for c in bat["cells"]]
    hdd = json.load(open(os.path.join(BE, "hdd.json"), encoding="utf-8"))
    arch["Backblaze drives"] = [r["onset_frac"] for r in hdd["drives"]]
    return arch


def _censored_median(v):
    """Median onset with undetected records right-censored at the end of life.

    Every record is followed to failure, so every censoring time equals the
    record's whole life, fraction 1.  The Kaplan--Meier estimate is then the
    empirical distribution below 1, and its median is the plain median with an
    undetected record counted as an onset beyond the end of life (inf here).
    """
    x = np.array([np.inf if f is None else f for f in v], float)
    #  np.median averages the two middle values as the conditional median does;
    #  with an inf among them the result is inf, "not reached"
    return float(np.median(x))


def part_cens():
    rng = np.random.default_rng(20261013)
    out = {}
    print("archive                 n  det   censored median [95%]      share < 70% [95%]")
    for k, v in _onset_records().items():
        x = np.array([np.inf if f is None else f for f in v], float)
        med = _censored_median(v)
        idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
        meds = np.median(x[idx], axis=1)
        lo, hi = np.quantile(meds, [0.025, 0.975], method="inverted_cdf")
        k70 = int(np.sum(x < 0.70))
        plo, phi = wilson(k70, len(x))

        def pct(z):
            return None if not np.isfinite(z) else 100 * float(z)
        out[k] = {"n": len(x), "n_detected": int(np.isfinite(x).sum()),
                  "censored_median_pct": pct(med), "censored_median_ci": [pct(lo), pct(hi)],
                  "share_before_70": k70 / len(x), "share_before_70_ci": [plo, phi]}

        def fmt(z):
            return "  n.r." if z is None else f"{z:6.1f}"
        print(f"{k:<22}{len(x):>4}{out[k]['n_detected']:>5}   {fmt(pct(med))} "
              f"[{fmt(pct(lo))},{fmt(pct(hi))}]     {k70 / len(x):.2f} [{plo:.2f},{phi:.2f}]")
    _save("cens", out)


# ══════════════════════════════════════════════════════════════════════════
# hw     sensitivity to the healthy window
# ══════════════════════════════════════════════════════════════════════════
def _onset_hw(hi, cap, div, thr=3.0, persist=10, frac=0.8):
    """The detector of datasets.first_prediction_time with its window opened up."""
    hi = np.asarray(hi, float)
    n = len(hi)
    nh = min(cap, max(5, n // div))
    base = np.median(hi[:nh])
    mad = np.median(np.abs(hi[:nh] - base)) * 1.4826
    above = hi > base + thr * max(mad, 1e-9)
    frac_above = np.cumsum(above[::-1])[::-1] / np.maximum(np.arange(n, 0, -1), 1)
    for i in range(n - persist):
        if above[i:i + persist].all() and frac_above[i] >= frac:
            return i
    return 0


def part_hw():
    series = {}
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        series[label] = [r["hi"] for _, r in sorted(ds.load_dataset(name).items())]
    ns = _thirdset_namespace()
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", True), ("FD004", True)):
        series["C-MAPSS " + fd + ("+regime" if by else "")] = [
            ns["indicator"](arr) for _, arr in sorted(ns["load"](fd, by_regime=by).items())]
    ref = _onset_records()
    out = {}
    for k, hs in series.items():
        #  the paper's setting reproduces the stored onsets exactly
        base = [_onset_hw(h, 40, 10) for h in hs]
        assert [b / len(h) if b > 0 else None for b, h in zip(base, hs)] == ref[k], k
        rows = []
        for cap in (10, 20, 40, 80, 160):
            for div in (5, 10, 20):
                f = [_onset_hw(h, cap, div) for h in hs]
                v = [i / len(h) if i > 0 else None for i, h in zip(f, hs)]
                det = [x for x in v if x is not None]
                cm = _censored_median(v)
                rows.append({"cap": cap, "div": div, "n_detected": len(det),
                             "median_pct": 100 * float(np.median(det)) if det else None,
                             "censored_median_pct": 100 * cm if np.isfinite(cm) else None})
        meds = [r["median_pct"] for r in rows if r["median_pct"] is not None]
        cms = [r["censored_median_pct"] for r in rows]
        verdicts = {(r["median_pct"] or 0) >= 70 for r in rows}
        cverdicts = {(r["censored_median_pct"] is None) or r["censored_median_pct"] >= 70
                     for r in rows}
        out[k] = {"settings": rows, "median_range": [min(meds), max(meds)],
                  "censored_median_range": [min(c for c in cms if c is not None)
                                            if any(c is not None for c in cms) else None,
                                            None if any(c is None for c in cms)
                                            else max(cms)],
                  "detected_range": [min(r["n_detected"] for r in rows),
                                     max(r["n_detected"] for r in rows)],
                  "n_records": len(hs), "verdict_changes": len(verdicts) > 1,
                  "censored_verdict_changes": len(cverdicts) > 1}
        print(f"{k:<22} median {min(meds):5.1f}-{max(meds):5.1f}  censored "
              f"{out[k]['censored_median_range']}  detected {out[k]['detected_range']} "
              f"of {len(hs)}  verdict changes: {len(verdicts) > 1} / {len(cverdicts) > 1}")
    _save("hw", out)


# ══════════════════════════════════════════════════════════════════════════
# boot   path model against the no-threshold predictors, clustered on bearing
# ══════════════════════════════════════════════════════════════════════════
def part_boot():
    from collections import defaultdict
    rng = np.random.default_rng(20261014)
    out = {}
    for name in ("pronostia", "xjtu"):
        path = {(r["bearing"], r["k"]): r for r in sb.run(name, laws=("BS",))}
        reg = _reg_rows(name)
        out[name] = {}
        for m in ("gbr", "life_table"):
            alt = {(r["bearing"], r["k"]): r for r in reg[m]}
            assert set(alt) == set(path), (name, m)
            by_b = defaultdict(list)
            for key in sorted(path):
                p, a = path[key], alt[key]
                by_b[key[0]].append((abs(p["pred"] - p["true"]), abs(a["pred"] - a["true"]),
                                     float(p["lo"] <= p["true"] <= p["hi"]),
                                     float(a["lo"] <= a["true"] <= a["hi"])))
            bs = sorted(by_b)

            def stat(sel):
                v = np.array([x for b in sel for x in by_b[b]])
                return v[:, 0].mean() - v[:, 1].mean(), v[:, 2].mean() - v[:, 3].mean()
            obs = stat(bs)
            boot = np.array([stat([bs[i] for i in rng.integers(0, len(bs), len(bs))])
                             for _ in range(B_BOOT)])
            ci_mae = np.percentile(boot[:, 0], [2.5, 97.5])
            ci_cov = np.percentile(boot[:, 1], [2.5, 97.5])
            out[name][m] = {"n_bearings": len(bs), "mae_diff": float(obs[0]),
                            "mae_diff_ci": [float(c) for c in ci_mae],
                            "cov_diff": float(obs[1]), "cov_diff_ci": [float(c) for c in ci_cov]}
            print(f"{name:10s} path - {m:10s}: MAE {obs[0]:+8.0f} s [{ci_mae[0]:+.0f},{ci_mae[1]:+.0f}]"
                  f"   coverage {obs[1]:+.2f} [{ci_cov[0]:+.2f},{ci_cov[1]:+.2f}]"
                  f"   ({len(bs)} bearings)", flush=True)
    _save("boot", out)


# ══════════════════════════════════════════════════════════════════════════
# dec    the decomposition where the predictive law is simulated (n_d < 10)
# ══════════════════════════════════════════════════════════════════════════
def _decomp(st, pf, D, dt, nd_min=10.0, n_mc=32, n_max=200):
    """rho as the code computes it (IG moments for every particle) and with the
    simulated paths' own mean and variance for the particles that use them."""
    th, mu, lam, n_d = gbs._rul_ig(st, pf, D, dt)
    w = np.exp(st.logw - st.logw.max())
    w /= w.sum()
    var_ig = mu ** 3 / np.maximum(lam, 1e-300)
    v_al = float(np.sum(w * var_ig))
    v_ep = max(float(np.sum(w * mu ** 2)) - float(np.sum(w * mu)) ** 2, 0.0)
    rho_ig = v_ep / max(v_ep + v_al, 1e-300)
    mc = np.flatnonzero(~((n_d >= nd_min) & np.isfinite(mu)))
    m_i, v_i = mu.copy(), var_ig.copy()
    if mc.size:
        sims, _ = gbs.mc_first_passage(st, pf, mc, D, dt, n_mc=n_mc, n_max=n_max)
        m_i[mc], v_i[mc] = sims.mean(axis=1), sims.var(axis=1)
    m_i = np.where(np.isfinite(m_i), m_i, 0.0)
    v_i = np.where(np.isfinite(v_i), v_i, 0.0)
    v_al2 = float(np.sum(w * v_i))
    v_ep2 = max(float(np.sum(w * m_i ** 2)) - float(np.sum(w * m_i)) ** 2, 0.0)
    rho_h = v_ep2 / max(v_ep2 + v_al2, 1e-300)
    return {"rho_ig": float(rho_ig), "rho_sim": float(rho_h),
            "mc_share": float(w[mc].sum()) if mc.size else 0.0}


def _dec_summary(rows):
    d = np.array([abs(r["rho_ig"] - r["rho_sim"]) for r in rows])
    with_mc = [r for r in rows if r["mc_share"] > 0]
    dm = np.array([abs(r["rho_ig"] - r["rho_sim"]) for r in with_mc]) if with_mc else np.zeros(1)
    return {"n": len(rows), "n_with_mc": len(with_mc),
            "median_mc_share": float(np.median([r["mc_share"] for r in rows])),
            "median_abs_drho": float(np.median(d)), "p90_abs_drho": float(np.quantile(d, 0.9)),
            "max_abs_drho": float(d.max()),
            "median_abs_drho_with_mc": float(np.median(dm)),
            "decision_flips": int(sum((r["rho_ig"] > 0.7) != (r["rho_sim"] > 0.7) for r in rows))}


def part_dec():
    import study_theory as T
    out, cache = {}, {}
    rows = []
    for rep in range(60):
        r = np.random.default_rng(200 + rep)
        x, y, t = T._synth(r, K=320)
        D = 0.75 * x[-1]
        kf = int(np.argmax(x >= D))
        if kf < 60:
            continue
        pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=1500, seed=rep))
        states = pf.run(y[:kf], t[:kf], 1.0, T.PRIOR_M, T.PRIOR_S, x0=y[0])
        for frac in (0.4, 0.6, 0.8, 0.9, 0.95):
            k = int(frac * kf)
            if k < 25:
                continue
            rows.append({"frac": frac, **_decomp(states[k - 1], pf, D, 1.0)})
    out["synthetic"] = {str(f): _dec_summary([r for r in rows if r["frac"] == f])
                        for f in (0.4, 0.6, 0.8, 0.9, 0.95)}
    for f, s in out["synthetic"].items():
        print(f"  synthetic {f}: {s}")
    for name in ("pronostia", "xjtu"):
        data = ds.load_dataset(name)
        rows = []
        for b, pts in ds.evaluation_bearings(data, name):
            rec = data[b]
            D, med = _calibrated_D(data, name, b, cache, name)
            for k in pts:
                if k < 20 or k >= rec["n"] - 1:
                    continue
                pf, states = sb._filter(rec, k, D, "BS", True, True, med,
                                        sb.stable_seed(b, k, "BS"), sb.N_PART)
                rows.append({"bearing": b, "k": k, **_decomp(states[-1], pf, D, rec["dt"])})
        out[name] = _dec_summary(rows)
        print(f"  {name}: {out[name]}", flush=True)
    _save("dec", out)


# ══════════════════════════════════════════════════════════════════════════
# sens2  the inference choices that could still explain the benchmark failure
# ══════════════════════════════════════════════════════════════════════════
class ARPF(gbs.ParticleFilter):
    """The filter with AR(1) observation noise, eps_k = phi eps_{k-1} + e_k.

    Given a particle, eps_{k-1} = Y_{k-1} - X_{k-1} is known, so the
    observation density of Y_k is that of e_k evaluated at
    Y_k - X_k - phi (Y_{k-1} - X_{k-1}).  Feeding the parent step the
    particle-wise target y_eff = Y_k - phi (Y_{k-1} - X_{k-1}) gives exactly that,
    in the guided proposal as well as in the weight; sigma_eps is then the
    innovation scale.
    """

    def __init__(self, law, cfg, phi):
        super().__init__(law, cfg)
        self.phi = float(phi)
        self._yprev = None

    def run(self, y, t, dt_ref, prior_logmean, prior_logsd, x0=None):
        self._yprev = float(y[0])
        return super().run(y, t, dt_ref, prior_logmean, prior_logsd, x0)

    def step(self, st, y, dt, dt_ref):
        y_eff = y - self.phi * (self._yprev - st.x)
        self._yprev = float(y)
        return super().step(st, y_eff, dt, dt_ref)


def _filter_ar(rec, k_pred, D, law_name, informed, coupling, med_life_steps,
               seed, n_part):
    """study_bearings._filter with AR(1) observation noise.

    phi is the lag-one autocorrelation of the bearing's own healthy window, the
    same causal window that sets sigma_0, and the noise prior is centred on the
    innovation scale sigma_0 sqrt(1 - phi^2) rather than on sigma_0.
    """
    law = sb.LAWS[law_name]()
    cfg = gbs.PFConfig(n_particles=n_part, delta=sb.DELTA, seed=seed,
                       gamma_coupling=coupling,
                       x_ref=max(0.25 * D, 1e-3), x_max=50.0 * max(D, 1e-3),
                       x_couple_max=max(D, 1e-3))
    y_all = rec["hi"][:k_pred + 1]
    nh = max(5, min(40, len(y_all) // 10))
    h = y_all[:nh] - np.polyval(np.polyfit(np.arange(nh), y_all[:nh], 1), np.arange(nh))
    phi = float(np.clip(_lag1(h), 0.0, 0.95))
    sigma0 = max(float(np.std(y_all[:nh])) * math.sqrt(1.0 - phi ** 2), 1e-3)
    pf = ARPF(law, cfg, phi)
    fpt = ds.first_prediction_time(y_all)
    k0 = int(np.clip(fpt, 0, max(0, k_pred - 15)))
    y = y_all[k0:]
    t = rec["t"][k0:k_pred + 1]
    pf.fpt = int(fpt)
    m, s = sb.make_priors(D, med_life_steps, informed, cfg.gamma_coupling, sigma0)
    m, s = sb.adapt_prior(law, m, s, D, med_life_steps)
    x0 = float(np.median(y[:max(3, len(y) // 20)]))
    return pf, pf.run(y, t, rec["dt"], m, s, x0=x0)


def part_sens2():
    import functools
    variants = {
        "noise_prior_sd_0.5": ("sigma", 0.5), "noise_prior_sd_1.5": ("sigma", 1.5),
        "floor_c_0.01": ("floor", 0.01), "floor_c_0.20": ("floor", 0.20),
        "ar1_observation": ("ar", None),
    }
    which = [v for v in variants if v in sys.argv] or list(variants)
    out = json.load(open(OUT, encoding="utf-8")).get("sens2", {}) if os.path.exists(OUT) else {}
    for tag in which:
        kind, val = variants[tag]
        out[tag] = {}
        for name in ("pronostia", "xjtu"):
            keep_sd = sb.PRIOR_SD_INFORMED["sigma_eps"]
            keep_cfg, keep_filter = gbs.PFConfig, sb._filter
            try:
                if kind == "sigma":
                    sb.PRIOR_SD_INFORMED["sigma_eps"] = val
                elif kind == "floor":
                    gbs.PFConfig = functools.partial(keep_cfg, jitter_floor=val)
                else:
                    sb._filter = _filter_ar
                rows = sb.run(name, laws=("BS",), tag=tag)
            finally:
                sb.PRIOR_SD_INFORMED["sigma_eps"] = keep_sd
                gbs.PFConfig, sb._filter = keep_cfg, keep_filter
            s = sb.summarise(rows)["BS"]
            out[tag][name] = s
            print(f"{tag:20s} {name:10s} n={s['n']:3d} MAE={s['MAE']:8.0f} "
                  f"MedAPE={s['MedAPE_pct']:6.1f} cov={s['coverage95']:.2f} "
                  f"PHM={s['PHM_score']:.3f}", flush=True)
        _save("sens2", out)


# ══════════════════════════════════════════════════════════════════════════
# exact   the exact posterior by particle marginal Metropolis--Hastings
# ══════════════════════════════════════════════════════════════════════════
LAM_ES, N_ES, G_ES = 48, 64, 40    # CMA-ES: candidates, particles, generations
C_MH, B_MH, S_MH = 8, 400, 800      # PMMH: chains, burn-in (adaptive), kept iterations
#  Particles per likelihood, per prediction point: doubled from 128 until the
#  standard deviation of log Z at the mode is at most 1.0, where pseudo-marginal
#  chains mix, or until 4096 particles or 4e5 particle-steps per chain.  On
#  smooth, nearly noiseless XJTU-SY records it is 6.5 at 128 particles and 1.6
#  at 2048; on PRONOSTIA it is 0.7 at 128.
N_MH0, SD_MH, N_MH_MAX, NSTEP_MAX = 128, 1.0, 4096, 4e5
M_PRED, N_PRED = 256, 256           # posterior draws, particles each, to predict
LIFE_AT = 0.70                      # the subset: one point per record, nearest here


def _cmaes(f, x0, sigma0, rng, lam=LAM_ES, gens=G_ES, tol=0.02):
    """Minimise f (batch of rows -> values) by CMA-ES with rank-mu update and
    cumulative step-size control.  Returns the final mean and sigma^2 C."""
    n = len(x0)
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1.0 / np.sum(wts ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    xm, sigma = np.asarray(x0, float).copy(), float(sigma0)
    C, B, Dg = np.eye(n), np.eye(n), np.ones(n)
    pc, pss = np.zeros(n), np.zeros(n)
    g = 0
    for g in range(1, gens + 1):
        ary = rng.standard_normal((lam, n)) @ (B * Dg).T
        arx = xm + sigma * ary
        idx = np.argsort(f(arx))[:mu]
        xold = xm
        xm = wts @ arx[idx]
        yw = (xm - xold) / sigma
        pss = (1 - cs) * pss + math.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / Dg))
        hsig = (np.linalg.norm(pss) / math.sqrt(1 - (1 - cs) ** (2 * g)) / chin
                < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        art = (arx[idx] - xold) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (art.T * wts) @ art)
        sigma *= math.exp((cs / damps) * (np.linalg.norm(pss) / chin - 1))
        C = np.triu(C) + np.triu(C, 1).T
        d2, B = np.linalg.eigh(C)
        Dg = np.sqrt(np.maximum(d2, 1e-20))
        if sigma * Dg.max() < tol:
            break
    return xm, sigma ** 2 * C, g


def _bank(pf, logth, y, t, dt_ref, x0, n, rng, proposal="bootstrap"):
    """One particle filter per row of logth, theta fixed, n particles each.

    The bootstrap proposal by default.  The main filter's guided proposal is
    also exact, but near the posterior mode its likelihood estimate is far
    noisier: on PRONOSTIA Bearing1_4 the standard deviation of log Z is 2.4 at
    128 particles and 1.5 at 1024, against 0.7 and 0.4 for the bootstrap.
    Resampling stays inside each filter.  Returns the final states (M, n),
    normalised log weights (M, n) and log Z (M,).
    """
    M = len(logth)
    th = pf._theta(np.repeat(logth, n, axis=0))
    sig = np.maximum(th["sigma_eps"], pf.cfg.sigma_floor)
    x = np.full(M * n, float(x0))
    logw = np.full((M, n), -math.log(n))
    logz = np.zeros(M)
    off = np.arange(M)[:, None]
    for k in range(1, len(y)):
        s = pf._scale(x, t[k] - t[k - 1], dt_ref, th)
        if proposal == "guided":
            dx, log_q = pf._guided(x, y[k], th, s, sig)
        else:
            dx, log_q = pf.law.sample(th, s, rng), None
        dx = np.where(np.isfinite(dx), dx, 0.0)
        x = np.clip(x + dx, -np.inf, pf.cfg.x_max)
        ll = -0.5 * ((y[k] - x) / sig) ** 2 - np.log(sig) - gbs.LOG_SQRT2PI
        if log_q is not None:
            lp = pf.law.logpdf(dx, th, s)
            ll = ll + np.where(np.isfinite(lp), lp, -1e10) - log_q
        a = logw + ll.reshape(M, n)
        lse = special.logsumexp(a, axis=1)
        logz += lse
        logw = a - lse[:, None]
        w = np.exp(logw)
        rows = np.flatnonzero(1.0 / np.sum(w ** 2, axis=1) < 0.5 * n)
        if rows.size:
            #  systematic resampling of every such row at once: row r's
            #  cumulative weights are shifted into (r, r+1], which keeps the
            #  concatenation sorted for a single searchsorted
            cw = np.cumsum(w[rows], axis=1)
            cw[:, -1] = 1.0
            r_off = off[:rows.size]
            u = (rng.random((rows.size, 1)) + np.arange(n)) / n
            idx = np.searchsorted((cw + r_off).ravel(), (u + r_off).ravel())
            idx = np.clip(idx.reshape(rows.size, n) - r_off * n, 0, n - 1)
            x2 = x.reshape(M, n)
            x2[rows] = np.take_along_axis(x2[rows], idx, axis=1)
            x = x2.ravel()
            logw[rows] = -math.log(n)
    return x.reshape(M, n), logw, logz


def _pmmh(lpost, u0, rng):
    """Pseudo-marginal random-walk Metropolis, C_MH chains vectorised.

    lpost maps a (C, d) batch to noisy log-posterior estimates.  The proposal
    covariance adapts during burn-in, to 2.38^2/d times the pooled covariance
    of the recent half of the chains; a chain's current estimate is kept, never
    refreshed, as the pseudo-marginal algorithm requires.  Returns the kept
    draws (S, C, d), the acceptance rate over them and the Gelman--Rubin R-hat
    of each parameter.
    """
    d = len(u0)
    u = u0 + 0.05 * rng.standard_normal((C_MH, d))
    lp = lpost(u)
    shape, scale = 0.1 ** 2 * np.eye(d), 1.0
    srw = shape
    hist, keep, acc, rate = [], [], 0, []
    for it in range(B_MH + S_MH):
        if 25 <= it < B_MH and it % 25 == 0:
            #  a global scale steered towards 20 % acceptance, and from the
            #  100th iteration the shape of the recent half of the chains
            scale *= math.exp(2.0 * (np.mean(rate[-25:]) - 0.2))
            if it >= 100:
                h = np.concatenate(hist[it // 2:], axis=0)
                shape = 2.38 ** 2 / d * np.cov(h.T) + 1e-6 * np.eye(d)
            srw = scale * shape
        prop = u + rng.standard_normal((C_MH, d)) @ np.linalg.cholesky(srw).T
        lq = np.nan_to_num(lpost(prop), nan=-np.inf)
        a = np.log(rng.random(C_MH)) < lq - lp
        u = np.where(a[:, None], prop, u)
        lp = np.where(a, lq, lp)
        rate.append(float(a.mean()))
        hist.append(u.copy())
        if it >= B_MH:
            keep.append(u.copy())
            acc += int(a.sum())
    K = np.array(keep)
    n = K.shape[0]
    W = K.var(axis=0, ddof=1).mean(axis=0)
    Bv = n * K.mean(axis=0).var(axis=0, ddof=1)
    rhat = np.sqrt(((n - 1) / n * W + Bv / n) / np.maximum(W, 1e-300))
    return K, acc / (S_MH * C_MH), rhat


def _exact_one(rec, k, D, med, seed):
    """The exact-posterior reference at one prediction point, and the main
    filter's own prediction there, which reproduces study_bearings."""
    law = sb.LAWS["BS"]()
    pf_main, states = sb._filter(rec, k, D, "BS", True, True, med, seed, sb.N_PART)
    if not states:
        return None
    #  set-up shared with study_bearings._filter, line for line
    y_all = rec["hi"][:k + 1]
    nh = max(5, min(40, len(y_all) // 10))
    sigma0 = max(float(np.std(y_all[:nh])), 1e-3)
    fpt = ds.first_prediction_time(y_all)
    k0 = int(np.clip(fpt, 0, max(0, k - 15)))
    y, t = y_all[k0:], rec["t"][k0:k + 1]
    m, s = sb.make_priors(D, med, True, True, sigma0)
    m, s = sb.adapt_prior(law, m, s, D, med)
    x0 = float(np.median(y[:max(3, len(y) // 20)]))
    names = pf_main.names
    pm = np.array([m[nm] for nm in names])
    ps = np.array([s[nm] for nm in names])
    rng = np.random.default_rng(seed)
    pf = gbs.ParticleFilter(law, gbs.PFConfig(
        n_particles=N_PRED, delta=1.0, seed=seed, gamma_coupling=True,
        x_ref=pf_main.cfg.x_ref, x_max=pf_main.cfg.x_max,
        x_couple_max=pf_main.cfg.x_couple_max))

    #  in prior-sd units u, theta = pm + u ps, so the prior is standard normal
    def lpost(u, n):
        _, _, lz = _bank(pf, pm + u * ps, y, t, rec["dt"], x0, n, rng)
        return lz - 0.5 * np.sum(u ** 2, axis=1)

    #  The posterior can sit many prior standard deviations from the prior
    #  centre and from the main filter's cloud, so the chains start at a mode
    #  located by CMA-ES rather than at either.
    um, _, gens = _cmaes(lambda u: -lpost(u, N_ES), np.zeros(len(pm)), 1.0, rng)
    n_mh = N_MH0
    while True:
        _, _, lz = _bank(pf, np.repeat((pm + um * ps)[None, :], 16, axis=0), y, t,
                         rec["dt"], x0, n_mh, rng)
        sd_lz = float(lz.std())
        if (sd_lz <= SD_MH or 2 * n_mh > N_MH_MAX
                or 2 * n_mh * len(y) > NSTEP_MAX):
            break
        n_mh *= 2
    K, acc, rhat = _pmmh(lambda u: lpost(u, n_mh), um, rng)
    flat = K.reshape(-1, len(pm))
    z = pm + flat[rng.choice(len(flat), M_PRED, replace=False)] * ps
    xb, lwb, _ = _bank(pf, z, y, t, rec["dt"], x0, N_PRED, rng)
    pooled = gbs.PFState(x=xb.ravel(), logth=np.repeat(z, N_PRED, axis=0),
                         logw=(lwb - math.log(M_PRED)).ravel(), names=names)
    st = states[-1]
    w0 = np.exp(st.logw - st.logw.max())
    w0 /= w0.sum()
    cap = 10.0 * float(rec["t"][-1])
    out = {}
    for tag, stt, pff in (("exact", pooled, pf), ("main", st, pf_main)):
        p = gbs.predict_rul(stt, pff, D, rec["dt"], nd_min=10.0, n_mc=32, n_draw=8,
                            n_max=200)
        lo, hi_ = np.quantile(p["samples"], [0.025, 0.975])
        out[tag] = {"pred": float(np.clip(p["mean"], 0.0, cap)),
                    "lo": float(np.clip(lo, 0.0, cap)), "hi": float(np.clip(hi_, 0.0, cap)),
                    "rho": p["rho"]}
    zbar = pm + flat.mean(axis=0) * ps
    out.update({"acc": float(acc), "rhat": rhat.tolist(), "generations": gens,
                "n_mh": n_mh, "sd_logz": sd_lz,
                "steps": int(len(y)), "life_frac": float(rec["t"][k] / rec["life"]),
                "theta_exact": np.exp(zbar).tolist(),
                "theta_main": np.exp(w0 @ st.logth).tolist(),
                #  posterior mean against the main filter's cloud, in prior sd
                "shift_sd": ((zbar - w0 @ st.logth) / ps).tolist(),
                "names": list(names)})
    return out


def part_exact():
    """Rows of the exact-posterior reference, one shard at a time.

    python study_p3_review.py exact [smoothed|raw] [pronostia|xjtu]
    with P3_EXACT_SHARD=i/n taking every n-th evaluation record from the i-th,
    and P3_REVIEW_OUT naming a separate file; "merge" then "exact_sum" pool them.
    """
    variants = {"smoothed": None, "raw": 1}
    which = [v for v in variants if v in sys.argv] or ["raw"]
    names = [n for n in ("pronostia", "xjtu") if n in sys.argv] or ["pronostia", "xjtu"]
    i, n_sh = (int(v) for v in os.environ.get("P3_EXACT_SHARD", "0/1").split("/"))
    #  A rerun of named records whose chains did not converge: P3_EXACT_ONLY
    #  lists them, and P3_MH_B, P3_MH_S, P3_NMH_MAX, P3_NSTEP_MAX lengthen the
    #  chains or raise the particle cap.  Its rows are stored under a "rerun"
    #  key and replace the first run's rows for the same points in exact_sum.
    only = [b for b in os.environ.get("P3_EXACT_ONLY", "").split(",") if b]
    global B_MH, S_MH, N_MH_MAX, NSTEP_MAX
    B_MH = int(os.environ.get("P3_MH_B", B_MH))
    S_MH = int(os.environ.get("P3_MH_S", S_MH))
    N_MH_MAX = int(os.environ.get("P3_NMH_MAX", N_MH_MAX))
    NSTEP_MAX = float(os.environ.get("P3_NSTEP_MAX", NSTEP_MAX))
    out = json.load(open(OUT, encoding="utf-8")).get("exact_rows", {}) if os.path.exists(OUT) else {}
    for tag in which:
        for name in names:
            w = variants[tag]
            data = ds.load_dataset(name) if w is None else _hi_with_smoothing(name, w)
            rows, cache = [], {}
            for j, (b, pts) in enumerate(ds.evaluation_bearings(data, name)):
                if (b not in only) if only else (j % n_sh != i):
                    continue
                rec = data[b]
                D, med = _calibrated_D(data, f"{name}_{tag}", b, cache, name)
                ok = [k for k in pts if 20 <= k < rec["n"] - 1
                      and rec["t"][-1] - rec["t"][k] > 0]
                if not ok:
                    continue
                #  the subset: per record, the prediction point nearest LIFE_AT
                for k in (min(ok, key=lambda k_: abs(rec["t"][k_] / rec["life"] - LIFE_AT)),):
                    true = float(rec["t"][-1] - rec["t"][k])
                    r = _exact_one(rec, k, D, med, sb.stable_seed(b, k, "BS"))
                    if r is None:
                        continue
                    for est in ("exact", "main"):
                        er = 100.0 * (true - r[est]["pred"]) / true
                        rows.append({"bearing": b, "k": k, "law": est, "true": true,
                                     "er": er, "score": float(ds.phm_score(er)), **r[est],
                                     **({k_: r[k_] for k_ in ("acc", "rhat", "n_mh", "sd_logz", "generations", "steps",
                                                               "life_frac", "theta_exact",
                                                               "theta_main", "shift_sd", "names")}
                                        if est == "exact" else {}),
                                     **({"rerun": True, "iterations": B_MH + S_MH} if only else {})})
                    print(f"  {tag} {name} {b} k={k}: true={true:8.0f} "
                          f"exact={r['exact']['pred']:8.0f} [{r['exact']['lo']:.0f},"
                          f"{r['exact']['hi']:.0f}] main={r['main']['pred']:8.0f} "
                          f"acc={r['acc']:.2f} rhat={max(r['rhat']):.2f} steps={r['steps']} "
                          f"N={r['n_mh']} sd={r['sd_logz']:.1f}", flush=True)
                out[f"{tag}/{name}/" + (f"rerun-{'+'.join(only)}" if only else f"{i}of{n_sh}")] = rows
                _save("exact_rows", out)


def _by_onset(ex, name):
    """The exact-posterior points split by whether the record's detected onset
    (applicability.json; none detected counts as later) precedes the point."""
    app = {r["bearing"]: r for r in json.load(open(os.path.join(BE, "applicability.json"),
                                                   encoding="utf-8")) if r["dataset"] == name}
    out = {}
    for side in ("before", "after"):
        pts = [r for r in ex
               if (app[r["bearing"]]["fpt_idx"] > 0
                   and app[r["bearing"]]["fpt_frac"] < r["life_frac"]) == (side == "before")]
        conv = [r for r in pts if max(r["rhat"]) <= 1.1]
        out[side] = {"n": len(pts),
                     "covered": int(sum(r["lo"] <= r["true"] <= r["hi"] for r in pts)),
                     "n_converged": len(conv),
                     "covered_converged": int(sum(r["lo"] <= r["true"] <= r["hi"] for r in conv)),
                     "capped": int(sum(r["pred"] == r["hi"] for r in pts))}
    return out


def part_exact_sum():
    """Pool the shards of part exact and summarise them as tab:path does."""
    res = json.load(open(OUT, encoding="utf-8"))
    pool = {}
    #  first runs, then reruns, which replace the first run's rows point by point
    keys = sorted(res["exact_rows"], key=lambda k_: k_.split("/")[2].startswith("rerun"))
    for key in keys:
        tag, name, _ = key.split("/")
        d_ = pool.setdefault((tag, name), {})
        for r in res["exact_rows"][key]:
            d_[(r["bearing"], r["k"], r["law"])] = r
    pool = {k_: list(v.values()) for k_, v in pool.items()}
    out = {}
    for (tag, name), rows in sorted(pool.items()):
        s = sb.summarise(rows)
        ex = [r for r in rows if r["law"] == "exact"]
        out.setdefault(tag, {})[name] = {
            "exact": s["exact"], "main": s["main"],
            "acc_median": float(np.median([r["acc"] for r in ex])),
            "rhat_max": float(max(max(r["rhat"]) for r in ex)),
            "rhat_median": float(np.median([max(r["rhat"]) for r in ex])),
            #  points whose chains converged (every R-hat at most 1.1), and the
            #  coverage of both estimates over those points alone
            "n_converged": int(sum(max(r["rhat"]) <= 1.1 for r in ex)),
            "cov_converged": {est: float(np.mean([
                q["lo"] <= q["true"] <= q["hi"] for q in rows
                if q["law"] == est and max(next(x["rhat"] for x in ex if x["bearing"] == q["bearing"]
                                                and x["k"] == q["k"])) <= 1.1]))
                for est in ("exact", "main")},
            #  predictions at the cap of ten times the record length, where the
            #  point and the upper end are both clipped to the same value: the
            #  posterior sees no failure coming within that horizon
            "share_capped": float(np.mean([r["pred"] == r["hi"] for r in ex])),
            "by_onset": _by_onset(ex, name),
            "life_frac_median": float(np.median([r["life_frac"] for r in ex])),
            #  where the interval misses: the share of points whose true RUL lies
            #  below the exact interval, i.e. life predicted too long
            "share_true_below": float(np.mean([r["true"] < r["lo"] for r in ex])),
            "share_true_above": float(np.mean([r["true"] > r["hi"] for r in ex])),
            "beta_ratio_median": float(np.median([r["theta_exact"][1] / r["theta_main"][1]
                                                  for r in ex])),
            "shift_sd_median_abs": np.median(np.abs([r["shift_sd"] for r in ex]),
                                             axis=0).tolist(),
            "names": ex[0]["names"]}
        for est in ("exact", "main"):
            q = s[est]
            print(f"{tag:9s} {name:10s} {est:6s} n={q['n']:3d} MAE={q['MAE']:8.0f} "
                  f"MedAPE={q['MedAPE_pct']:6.1f} cov={q['coverage95']:.3f}")
        print(f"          acceptance {out[tag][name]['acc_median']:.2f}, "
              f"R-hat median {out[tag][name]['rhat_median']:.2f} max {out[tag][name]['rhat_max']:.2f}, "
              f"true below interval {out[tag][name]['share_true_below']:.2f}, "
              f"above {out[tag][name]['share_true_above']:.2f}")
    _save("exact", out)


# ══════════════════════════════════════════════════════════════════════════
# dthr   the uncertainty of the calibrated failure threshold D
# ══════════════════════════════════════════════════════════════════════════
K_D, B_D = 20, 2000          # threshold values in the mixture; bootstrap resamples


def part_dthr():
    """D is the median terminal latent state of the reference records, so it is
    uncertain.  Per evaluation bearing its bootstrap law over the reference
    records is summarised by K_D quantiles.  At every prediction point of the
    main study the filter runs exactly as there (so the D-fixed prediction
    reproduces tab:path), and the predictive law is then mixed over the K_D
    values with equal weight.  D enters the passage; its effect through the
    priors and the coupling reference, set from the calibrated D, is held fixed.
    The predictions at the 2.5 % and 97.5 % points of D's law are kept as well.
    """
    names = [n for n in ("pronostia", "xjtu") if n in sys.argv] or ["pronostia", "xjtu"]
    out = json.load(open(OUT, encoding="utf-8")).get("dthr", {}) if os.path.exists(OUT) else {}
    rng = np.random.default_rng(20261017)
    for name in names:
        data = ds.load_dataset(name)
        rows, spread, vcache = [], [], {}
        for b, pts in ds.evaluation_bearings(data, name):
            rec = data[b]
            D_obs, ref = ds.threshold_for(data, b, name)
            med = float(np.median([max(data[r]["n"] - ds.first_prediction_time(data[r]["hi"]), 20)
                                   for r in ref]))
            key = tuple(ref)
            if key not in vcache:
                vals = [sb.latent_terminal(data[r], "BS", True, True, med, D_obs, seed=7)
                        for r in ref]
                vcache[key] = np.array([v for v in vals if np.isfinite(v) and v > 0])
            vals = vcache[key]
            D0 = float(np.median(vals)) if vals.size else D_obs
            Db = np.median(vals[rng.integers(0, vals.size, size=(B_D, vals.size))], axis=1)
            Dq = np.quantile(Db, (np.arange(K_D) + 0.5) / K_D)
            D_lo, D_hi = (float(v) for v in np.quantile(Db, [0.025, 0.975]))
            spread.append({"bearing": b, "n_ref": int(vals.size), "D": D0,
                           "D_ci": [D_lo, D_hi], "rel_width": (D_hi - D_lo) / D0})
            cap = 10.0 * float(rec["t"][-1])
            for k in pts:
                if k < 20 or k >= rec["n"] - 1:
                    continue
                true = float(rec["t"][-1] - rec["t"][k])
                if true <= 0:
                    continue
                pf, states = sb._filter(rec, k, D0, "BS", True, True, med,
                                        sb.stable_seed(b, k, "BS"), sb.N_PART)
                if not states:
                    continue
                st_ = states[-1]

                def pred(D):
                    p = gbs.predict_rul(st_, pf, D, rec["dt"], nd_min=10.0, n_mc=32,
                                        n_draw=8, n_max=200)
                    return float(p["mean"]), p["samples"], p["rho"]
                #  first, exactly as study_bearings.predict_one
                m0, s0, r0 = pred(D0)
                mix = [pred(float(D)) for D in Dq]
                ml, sl, rl = pred(D_lo)
                mh, sh, rh = pred(D_hi)
                ests = {"fixed": (m0, s0, r0),
                        "mixture": (float(np.mean([m for m, _, _ in mix])),
                                    np.concatenate([s for _, s, _ in mix]),
                                    float(np.mean([r for _, _, r in mix]))),
                        "D_low": (ml, sl, rl), "D_high": (mh, sh, rh)}
                for est, (m, s, r) in ests.items():
                    lo, hi_ = np.quantile(s, [0.025, 0.975])
                    pr = float(np.clip(m, 0.0, cap))
                    er = 100.0 * (true - pr) / true
                    rows.append({"bearing": b, "k": k, "law": est, "true": true,
                                 "pred": pr, "lo": float(np.clip(lo, 0.0, cap)),
                                 "hi": float(np.clip(hi_, 0.0, cap)), "rho": r,
                                 "er": er, "score": float(ds.phm_score(er))})
            print(f"  {name} {b}: D {D0:.3f} [{D_lo:.3f}, {D_hi:.3f}] from {vals.size} records",
                  flush=True)
        s = sb.summarise(rows)
        out[name] = {"summary": s, "D_spread": spread,
                     "rel_width_median": float(np.median([x["rel_width"] for x in spread])),
                     "rows": rows}
        for est in ("fixed", "mixture", "D_low", "D_high"):
            q = s[est]
            print(f"{name:10s} {est:8s} n={q['n']:3d} MAE={q['MAE']:8.0f} "
                  f"MedAPE={q['MedAPE_pct']:6.1f} cov={q['coverage95']:.3f}", flush=True)
        _save("dthr", out)


# ══════════════════════════════════════════════════════════════════════════
# altind  the screen on other indicators of the same vibration records
# ══════════════════════════════════════════════════════════════════════════
ALT_IND = {
    #  name: (channel pair, how the pair is combined)
    "log RMS (the paper's)": (("h_rms", "v_rms"), "norm"),
    "kurtosis": (("h_kurtosis", "v_kurtosis"), "mean"),
    "peak to peak": (("h_p2p", "v_p2p"), "norm"),
    "envelope RMS": (("h_hilbert_rms", "v_hilbert_rms"), "norm"),
    "high-band energy": (("h_band_hi", "v_band_hi"), "norm"),
}


def part_altind():
    """Each indicator is built as the paper builds log RMS: the two channels
    combined, causally smoothed with the archive's window, divided by the
    healthy-window median and logged.  The detector, its constants and the
    three-state rule are unchanged; undetected records are censored."""
    import glob
    import pandas as pd
    rng = np.random.default_rng(20261018)
    out = {}
    for name in ("pronostia", "xjtu"):
        cfg = ds.DATASETS[name]
        out[name] = {}
        files = sorted(glob.glob(os.path.join(cfg["dir"], "*.csv")))
        for ind, ((a, b), how) in ALT_IND.items():
            fr = []
            for f in files:
                d = pd.read_csv(f)
                x, y = d[a].to_numpy(float), d[b].to_numpy(float)
                v = np.sqrt(x ** 2 + y ** 2) if how == "norm" else 0.5 * (x + y)
                v = ds._smooth(np.maximum(v, 1e-12), cfg["smooth"])
                nh = min(cfg["n_health"], max(5, int(0.05 * len(v))))
                hi = np.log(v / np.median(v[:nh]))
                i = ds.first_prediction_time(hi)
                fr.append(i / len(hi) if i > 0 else np.inf)
            x = np.array(fr)
            med = float(np.median(x))
            meds = np.median(x[rng.integers(0, len(x), size=(B_BOOT, len(x)))], axis=1)
            lo, hi_ = (float(q) for q in np.quantile(meds, [0.025, 0.975], method="inverted_cdf"))

            def pct(z):
                return None if not np.isfinite(z) else 100 * z
            det = x[np.isfinite(x)]
            #  Algorithm 3 in order: the indicator check (step 4) before the rule
            v = ("check the indicator" if det.size < 0.25 * len(x)
                 else _verdict3(pct(lo), pct(hi_), 70))
            r = {"n": len(x), "n_detected": int(det.size),
                 "median_detected_pct": pct(float(np.median(det))) if det.size else None,
                 "censored_median_pct": pct(med), "censored_median_ci": [pct(lo), pct(hi_)],
                 "verdict": v}
            out[name][ind] = r
            print(f"  {name:10s} {ind:22s} detected {det.size:2d}/{len(x)}  censored median "
                  f"{r['censored_median_pct']} {r['censored_median_ci']}  -> {r['verdict']}")
    _save("altind", out)


# ══════════════════════════════════════════════════════════════════════════
# thr    how much the verdicts, and the step, depend on the threshold
# ══════════════════════════════════════════════════════════════════════════
def _verdict3(lo, hi, t):
    lo = np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return "declines" if lo >= t else ("not declined" if hi < t else "undecided")


def part_thr():
    """(1) The three-state verdict of every archive at thresholds 50-95 %, and
    the range over which every verdict stays as at 70 %.  (2) Where the step in
    the C-MAPSS band study peaks: a bootstrap over units of the cut, among those
    study_bandsig.py scans, at which the difference in median absolute error
    (above minus below) is largest."""
    cens = json.load(open(OUT, encoding="utf-8"))["cens"]
    ci = {k: cens[k]["censored_median_ci"] for k in
          ("PRONOSTIA", "XJTU-SY", "C-MAPSS FD001", "C-MAPSS FD003",
           "C-MAPSS FD002+regime", "C-MAPSS FD004+regime", "NASA cells",
           "Backblaze drives")}
    pre = json.load(open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))
    ci["XJTU Li-ion"] = pre["battery"]["censored_median_ci"]
    ci["N-CMAPSS"] = pre["ncmapss"]["censored_median_ci"]
    dec = json.load(open(os.path.join(BE, "p3_decline.json"), encoding="utf-8"))
    for q, c in dec["cohorts"].items():
        ci["Backblaze " + q] = c["censored_median_ci"]
    grid = list(range(50, 100, 5))
    out = {"grid": grid, "verdicts": {k: [_verdict3(*v, t) for t in grid] for k, v in ci.items()}}
    at70 = {k: _verdict3(*v, 70) for k, v in ci.items()}
    fine = np.round(np.arange(50, 99.0001, 0.1), 1)
    keep = [float(t) for t in fine if all(_verdict3(*ci[k], t) == at70[k] for k in ci)]
    out["stable_range"] = [min(keep), max(keep)]
    #  per archive, the thresholds (on the fine grid) at which its verdict is
    #  the one it has at 70 %
    out["same_as_70"] = {k: [float(min(f)), float(max(f))] for k in ci
                         for f in [[t for t in fine if _verdict3(*ci[k], t) == at70[k]]]}
    print("every verdict as at 70 % for thresholds in", out["stable_range"])

    rng = np.random.default_rng(20261016)
    cuts = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    draws = {}
    for stem in ("bandtest", "bandtest_rep"):
        draws[stem] = json.load(open(os.path.join(BE, f"{stem}.json"), encoding="utf-8"))["rows"]
    #  pooled: the union of the two draws' units, each unit once
    seen, pooled = set(), []
    for stem in ("bandtest", "bandtest_rep"):
        for u in sorted({r["unit"] for r in draws[stem]}):
            if u not in seen:
                seen.add(u)
                pooled += [r for r in draws[stem] if r["unit"] == u]
    draws["pooled"] = pooled
    band = {}
    for name, rows in draws.items():
        by = {}
        for r in rows:
            by.setdefault(r["unit"], []).append(r)
        units = sorted(by)
        ons = np.array([by[u][0]["onset_frac"] for u in units])
        errs = [np.array([r["abs_pct_err"] for r in by[u]]) for u in units]

        def scan(idx):
            d = {}
            for c in cuts:
                b = [i for i in idx if ons[i] < c]
                a = [i for i in idx if ons[i] >= c]
                if len(b) >= 5 and len(a) >= 5:
                    d[c] = (np.median(np.concatenate([errs[i] for i in a]))
                            - np.median(np.concatenate([errs[i] for i in b])))
            return d
        obs = scan(range(len(units)))
        arg, pos70 = [], []
        for _ in range(2000):
            d = scan(rng.integers(0, len(units), len(units)))
            if d:
                arg.append(max(d, key=d.get))
                pos70.append(d.get(0.70, np.nan) > 0)
        arg = np.array(arg)
        band[name] = {"n_units": len(units),
                      "argmax_observed": float(max(obs, key=obs.get)),
                      "argmax_share": {f"{c:.2f}": float(np.mean(arg == c)) for c in cuts},
                      "argmax_q": [float(v) for v in np.percentile(arg, [2.5, 50, 97.5])],
                      "share_positive_at_70": float(np.nanmean(pos70)),
                      }
        print(f"  {name}: {len(units)} units, argmax observed {band[name]['argmax_observed']:.2f}, "
              f"bootstrap 2.5/50/97.5 % {band[name]['argmax_q']}, "
              f"difference > 0 at 70 % in {band[name]['share_positive_at_70']:.3f} of resamples")
    out["band"] = band
    _save("thr", out)


# ══════════════════════════════════════════════════════════════════════════
# lead   the warning a detected onset gives, in each archive's own time unit
# ══════════════════════════════════════════════════════════════════════════
def part_lead():
    """Time from the detected onset to failure, for every detected unit.

    The rule reads onset as a fraction of life, which compares assets of very
    different lives; a maintainer needs the time it leaves.  Seconds for the
    bearing rigs, power-on hours for the drives, cycles otherwise.
    """
    lead = {}
    app = json.load(open(os.path.join(BE, "applicability.json"), encoding="utf-8"))
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        lead[label] = ("s", [r["life_s"] * (1 - r["fpt_frac"]) for r in app
                             if r["dataset"] == name and r["fpt_idx"] > 0])
    ns = _thirdset_namespace()
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", True), ("FD004", True)):
        v = []
        for u, arr in sorted(ns["load"](fd, by_regime=by).items()):
            hi = ns["indicator"](arr)
            i = ns["onset"](hi)
            if i > 0:
                v.append(len(hi) - i)
        lead["C-MAPSS " + fd + ("+regime" if by else "")] = ("cycles", v)
    bat = json.load(open(os.path.join(BE, "battery.json"), encoding="utf-8"))
    lead["NASA cells"] = ("cycles", [c["n_cycles"] - c["onset_idx"] for c in bat["cells"]
                                     if c["onset_idx"] and c["onset_idx"] > 0])
    hdd = json.load(open(os.path.join(BE, "hdd.json"), encoding="utf-8"))
    lead["Backblaze drives"] = ("hours", [r["life_hours"] * (1 - r["onset_frac"])
                                          for r in hdd["drives"] if r["onset_frac"] is not None])
    pre = json.load(open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))
    for key, label in (("battery", "XJTU Li-ion"), ("ncmapss", "N-CMAPSS")):
        lead[label] = ("cycles", [r["n"] - r["onset_idx"] for r in pre[key]["records"]
                                  if r.get("onset_idx") is not None and r["onset_idx"] > 0])
    ci = json.load(open(OUT, encoding="utf-8"))["ci"]
    out = {}
    for label, (unit, v) in lead.items():
        if label in ci:
            assert len(v) == ci[label]["n_detected"], label
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        out[label] = {"unit": unit, "n": len(v), "median": float(q2),
                      "q1": float(q1), "q3": float(q3)}
        print(f"{label:24s} n={len(v):4d}  median {q2:10.1f} {unit}  IQR {q1:.1f}-{q3:.1f}")
    _save("lead", out)


# ══════════════════════════════════════════════════════════════════════════
# regime    the screen applied to each charging regime of the XJTU cells
# ══════════════════════════════════════════════════════════════════════════
def part_regime():
    """Algorithm 3 on each regime of the pre-registered battery archive alone.
    Exploratory, not part of the protocol: it reads the stored onsets of
    p3_prereg.json and applies the censored median, its bootstrap interval and
    the detection-rate check exactly as the pre-registered summary does."""
    pre = json.load(open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))
    recs = [r for r in pre["battery"]["records"] if "load_error" not in r]
    rng = np.random.default_rng(20261014)
    out = {}
    for b in sorted({r["batch"] for r in recs}):
        rs = [r for r in recs if r["batch"] == b]
        x = np.array([np.inf if r["onset_frac"] is None else r["onset_frac"]
                      for r in rs], float)
        idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
        lo, hi = np.quantile(np.median(x[idx], axis=1), [0.025, 0.975],
                             method="inverted_cdf")

        def pct(z):
            return None if not np.isfinite(z) else 100 * float(z)
        rate = float(np.isfinite(x).mean())
        out[b] = {"regime": rs[0]["cell"].rsplit("_battery", 1)[0], "n": len(x),
                  "n_detected": int(np.isfinite(x).sum()),
                  "censored_median_pct": pct(np.median(x)),
                  "ci": [pct(lo), pct(hi)],
                  "verdict": ("check indicator" if rate < 0.25 else
                              _verdict3(pct(lo), pct(hi), 70.0))}
        print(b, out[b])
    _save("regime", out)


def part_merge():
    """Fold p3_review_*.json, written by parallel runs, into p3_review.json."""
    import glob
    main = os.path.join(BE, "p3_review.json")
    res = json.load(open(main, encoding="utf-8")) if os.path.exists(main) else {}
    for f in sorted(glob.glob(os.path.join(BE, "p3_review_*.json"))):
        #  parallel runs of one part each hold some of its variants, so a part
        #  present in several files is merged variant by variant
        for k, v in json.load(open(f, encoding="utf-8")).items():
            if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                res[k].update(v)
            else:
                res[k] = v
        os.remove(f)
    json.dump(res, open(main, "w", encoding="utf-8"), indent=1)
    print("parts:", sorted(res))


PARTS = {"ci": part_ci, "acf": part_acf, "ppc": part_ppc, "cal": part_cal,
         "reg": part_reg, "sens": part_sens, "cens": part_cens, "hw": part_hw,
         "boot": part_boot, "dec": part_dec, "sens2": part_sens2, "exact": part_exact,
         "exact_sum": part_exact_sum, "lead": part_lead, "thr": part_thr,
         "dthr": part_dthr, "altind": part_altind, "regime": part_regime,
         "merge": part_merge}

if __name__ == "__main__":
    todo = [a for a in sys.argv[1:] if a in PARTS] or list(PARTS)
    for p in todo:
        print(f"== {p}", flush=True)
        PARTS[p]()
