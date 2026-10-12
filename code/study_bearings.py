"""
Real-data validation on PRONOSTIA (IEEE PHM 2012) and XJTU-SY.

Every degradation process is driven by the SAME Liu-West particle filter, the
SAME health indicator and the SAME threshold calibration; only the increment law
changes.  Differences in the tables are therefore attributable to the increment
law and not to feature engineering or tuning.

Usage:  python study_bearings.py [pronostia|xjtu|both]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

import datasets as ds
import gbs


def stable_seed(*key):
    """A seed that is the same on every run, machine and Python build.

    ``hash()`` is unsuitable: PYTHONHASHSEED randomises it per process for any
    key containing a string, so a study seeded that way cannot be reproduced.
    """
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bearings")
os.makedirs(RES, exist_ok=True)

N_PART = 1200
DELTA = 0.98

# ── priors ────────────────────────────────────────────────────────────────
# What is physics-derived and what is data-derived is kept explicit.
#
# alpha  PHYSICS.  alpha is the coefficient of variation of cycle-to-cycle crack
#        growth; fatigue data for AISI 52100 bearing steel give CV in [0.2,0.5],
#        so the prior is centred at 0.4 with a spread covering that band.
# gamma  PHYSICS SETS THE FORM AND THE CENTRE, DATA SET THE VALUE.  Paris'
#        da/dN = C (Y d_sigma sqrt(pi a))^m implies growth rate proportional to
#        a^{m/2}, i.e. a power-law coupling with gamma = m/2 = 1.5 for m = 3.
#        The functional form transfers to any indicator that is a smooth
#        increasing function of crack size, but the *effective* exponent does
#        not, because the crack-to-vibration map is unknown and non-linear.
#        The prior is therefore centred at 1.5 but deliberately wide: the
#        exponents estimated from the benchmark trajectories have medians 0.70
#        (XJTU-SY) and 1.47 (PRONOSTIA) with a broad inter-quartile range.
#        Claiming a tight N(1.5, 0.15^2) prior on the indicator scale would not
#        be defensible.
# beta   DATA.  The indicator-increment scale cannot be fixed from material data
#        without a calibrated crack-to-RMS map, so its prior mean comes from the
#        TRAINING bearings' median life (empirical Bayes) with a wide spread.
# sigma  DATA.  Set from the bearing's own healthy window, which is causal.
# sigma_eps is given a tight prior. A strictly increasing latent state fitted to
# a flat indicator can trade drift against measurement noise: the filter inflates
# sigma_eps, lets the latent run away, and still explains the data. Pinning
# sigma_eps near the noise level measured on the healthy window -- a property of
# the sensing chain, not of the damage -- removes that degeneracy and cuts the
# worst-case latent overshoot at failure from 45x the observed indicator to 9x.
PRIOR_SD_INFORMED = {"alpha": 0.45, "beta": 0.60, "sigma_eps": 0.15, "gamma": 0.35}
PRIOR_SD_DIFFUSE = {"alpha": 1.60, "beta": 2.50, "sigma_eps": 1.50, "gamma": 0.90}


def make_priors(D, med_life_steps, informed=True, coupling=True, sigma0=0.1):
    beta0 = D / max(med_life_steps, 1.0)
    m = {"alpha": np.log(0.40), "beta": np.log(beta0),
         "sigma_eps": np.log(max(sigma0, 1e-3))}
    if coupling:
        m["gamma"] = np.log(1.0 + 1.5)          # Paris: gamma = m/2, m = 3
    s = dict(PRIOR_SD_INFORMED if informed else PRIOR_SD_DIFFUSE)
    if not informed:
        m = dict(m)
        m["alpha"] = np.log(0.60)
        if coupling:
            m["gamma"] = np.log(1.0 + 0.5)      # uninformed centre
    return m, {k: s[k] for k in m}


LAWS = {
    "BS":       lambda: gbs.BSLaw(consistent=True),
    "BS-naive": lambda: gbs.BSLaw(consistent=False),
    "BS-t":     lambda: gbs.BSTLaw(nu=4.0, consistent=True),
    "Wiener":   lambda: gbs.WienerLaw(),
    "Gamma":    lambda: gbs.GammaLaw(),
    "IGproc":   lambda: gbs.IGLaw(),
}
# parameter names differ per law; map the shared prior onto each
ALIAS = {
    "Wiener": {"mu": "beta", "sigma": "alpha"},
    "Gamma":  {"eta": "alpha", "lam": "beta"},
    "IGproc": {"mu": "beta", "lam": "alpha"},
}


def adapt_prior(law, m, s, D, med_life_steps):
    """Translate the shared (alpha, beta) prior into each law's own parameters."""
    beta0 = D / max(med_life_steps, 1.0)
    out_m, out_s = {}, {}
    for nm in law.param_names:
        if law.name in ALIAS:
            src = ALIAS[law.name][nm]
            if law.name == "Wiener":
                out_m[nm] = np.log(beta0) if nm == "mu" else np.log(0.4 * beta0)
            elif law.name == "Gamma":
                # E[dX] = eta/lam = beta0 ; Var = eta/lam^2 = (0.4*beta0)^2
                lam0 = beta0 / (0.4 * beta0) ** 2
                out_m[nm] = np.log(lam0 * beta0) if nm == "eta" else np.log(lam0)
            else:   # IG process: E = mu, Var = mu^3/lam
                lam0 = beta0 ** 3 / (0.4 * beta0) ** 2
                out_m[nm] = np.log(beta0) if nm == "mu" else np.log(lam0)
            out_s[nm] = s[src]
        else:
            out_m[nm], out_s[nm] = m[nm], s[nm]
    for extra in ("sigma_eps", "gamma"):
        if extra in m:
            out_m[extra], out_s[extra] = m[extra], s[extra]
    return out_m, out_s


def _filter(rec, k_pred, D, law_name, informed, coupling, med_life_steps,
            seed, n_part):
    law = LAWS[law_name]()
    # the Paris coupling is a structural feature of the *drift*, independent of
    # the increment law: it is applied identically to every competing process so
    # that the comparison isolates the increment distribution
    cfg = gbs.PFConfig(n_particles=n_part, delta=DELTA, seed=seed,
                       gamma_coupling=coupling,
                       x_ref=max(0.25 * D, 1e-3), x_max=50.0 * max(D, 1e-3),
                       x_couple_max=max(D, 1e-3))
    pf = gbs.ParticleFilter(law, cfg)
    y_all = rec["hi"][:k_pred + 1]
    nh = max(5, min(40, len(y_all) // 10))
    sigma0 = max(float(np.std(y_all[:nh])), 1e-3)      # causal noise scale
    # start the degradation model at the detected onset, keeping at least 15
    # observations; the FPT rule uses only data up to k_pred
    fpt = ds.first_prediction_time(y_all)
    k0 = int(np.clip(fpt, 0, max(0, k_pred - 15)))
    y = y_all[k0:]
    t = rec["t"][k0:k_pred + 1]
    pf.fpt = int(fpt)
    # `med_life_steps` is the median number of DEGRADATION steps (life minus
    # onset) over the training bearings, so beta0 = D / med_life_steps is the
    # increment that traverses the threshold over a typical degradation phase.
    # It must not be reduced by this bearing's own k0: when no onset has been
    # detected, k0 is simply k_pred - 15 and subtracting it would tell the model
    # that failure is due within fifteen inspections, which is what produced
    # near-zero RUL predictions at every prediction point.
    m, s = make_priors(D, med_life_steps, informed, cfg.gamma_coupling, sigma0)
    m, s = adapt_prior(law, m, s, D, med_life_steps)
    x0 = float(np.median(y[:max(3, len(y) // 20)]))
    return pf, pf.run(y, t, rec["dt"], m, s, x0=x0)


def latent_terminal(rec, law_name, informed, coupling, med_life_steps, D_seed,
                    seed=0, n_part=600):
    """Posterior-mean latent state at the end of a completed run-to-failure record."""
    pf, states = _filter(rec, rec["n"] - 1, D_seed, law_name, informed, coupling,
                         med_life_steps, seed, n_part)
    if not states:
        return np.nan
    st = states[-1]
    w = np.exp(st.logw - st.logw.max())
    w /= w.sum()
    return float(w @ st.x)


def predict_one(rec, k_pred, D, law_name, informed=True, coupling=True,
                med_life_steps=100.0, seed=0, n_part=N_PART):
    """Filter up to index k_pred using only Y_{1:k_pred}, then predict the RUL."""
    pf, states = _filter(rec, k_pred, D, law_name, informed, coupling,
                         med_life_steps, seed, n_part)
    if not states:
        return None
    # The Monte Carlo branch is used only for particles with n_d < 10, so a
    # horizon of 200 steps is twenty times the expected crossing; paths that
    # survive it are genuinely non-crossing and are reported as censored rather
    # than simulated indefinitely.
    p = gbs.predict_rul(states[-1], pf, D, rec["dt"], nd_min=10.0,
                        n_mc=32, n_draw=8, n_max=200)
    lo, hi_ = np.quantile(p["samples"], [0.025, 0.975])
    # A prognostic that answers "ten times longer than the whole test" is
    # reporting "no imminent failure", not a number. Predictions are censored at
    # that cap so that a handful of diverging particles cannot dominate an MAE,
    # and the censoring rate is reported alongside the accuracy.
    cap = 10.0 * float(rec["t"][-1])
    pred = float(np.clip(p["mean"], 0.0, cap))
    return {"pred": pred, "lo": float(np.clip(lo, 0.0, cap)),
            "hi": float(np.clip(hi_, 0.0, cap)),
            "capped": bool(p["mean"] > cap),
            "rho": p["rho"], "v_al": p["v_aleatory"], "v_ep": p["v_epistemic"],
            "ess_min": float(np.min(states[-1].ess_hist)) if states[-1].ess_hist else np.nan,
            "frac_mc": p["frac_mc"], "frac_censored": p["frac_censored"],
            "fpt": int(getattr(pf, "fpt", 0)), "n_d": p["n_d_mean"]}


def run(dataset, laws=("BS", "Wiener", "Gamma", "IGproc"), informed=True,
        coupling=True, tag="main", n_part=N_PART, stride=1):
    """Evaluate one configuration.

    The failure threshold is calibrated ON THE LATENT SCALE: for each reference
    (training) bearing the filter is run over its complete run-to-failure record
    and the posterior-mean latent state at failure recorded; D is the median of
    those values.  Calibrating instead on the raw indicator is inconsistent,
    because a monotone latent state tracks the upper envelope of a noisy
    non-monotone indicator and therefore reaches any indicator-scale threshold
    early.  The calibration uses training bearings only and is recomputed for
    every model, so no model is advantaged.
    """
    data = ds.load_dataset(dataset)
    evals = ds.evaluation_bearings(data, dataset)
    rows = []
    dcache: dict = {}
    for b, pts in evals:
        rec = data[b]
        D_obs, ref = ds.threshold_for(data, b, dataset)
        # expected length of the DEGRADATION phase on the training bearings
        med_life_steps = float(np.median(
            [max(data[r]["n"] - ds.first_prediction_time(data[r]["hi"]), 20)
             for r in ref]))
        for lw in laws:
            ck = (lw, informed, coupling, tuple(ref))
            if ck not in dcache:
                vals = [latent_terminal(data[r], lw, informed, coupling,
                                        med_life_steps, D_obs, seed=7)
                        for r in ref]
                vals = [v for v in vals if np.isfinite(v) and v > 0]
                dcache[ck] = float(np.median(vals)) if vals else D_obs
        for k in pts[::stride]:
            if k < 20 or k >= rec["n"] - 1:
                continue
            # the records run to failure, so the actual RUL at index k is
            # t_end - t_k; at the official PRONOSTIA truncation point this
            # reproduces the published value for ten of the eleven bearings
            true = float(rec["t"][-1] - rec["t"][k])
            if true <= 0:
                continue
            is_official = ds.official_point(dataset, b, k)
            for lw in laws:
                D = dcache[(lw, informed, coupling, tuple(ref))]
                r = predict_one(rec, k, D, lw, informed, coupling,
                                med_life_steps, seed=stable_seed(b, k, lw),
                                n_part=n_part)
                if r is None:
                    continue
                er = 100.0 * (true - r["pred"]) / true
                rows.append({"dataset": dataset, "bearing": b, "k": k,
                             "life_frac": rec["t"][k] / rec["life"],
                             "law": lw, "informed": informed,
                             "coupling": coupling, "true": true,
                             "post_fpt": bool(k > r["fpt"] > 0),
                             "official": is_official,
                             "score": float(ds.phm_score(er)), "er": er, **r})
    return rows


def summarise(rows, by="law", subset=None):
    """Aggregate accuracy. `subset` optionally restricts to a row predicate."""
    out = {}
    if subset is not None:
        rows = [r for r in rows if subset(r)]
    keys = sorted({r[by] for r in rows})
    for kk in keys:
        s = [r for r in rows if r[by] == kk]
        if not s:
            continue
        e = np.array([r["pred"] - r["true"] for r in s])
        tr = np.array([r["true"] for r in s])
        cov = np.mean([r["lo"] <= r["true"] <= r["hi"] for r in s])
        out[kk] = {
            "n": len(s),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_pct": float(np.mean(np.abs(e / tr)) * 100),
            "MedAPE_pct": float(np.median(np.abs(e / tr)) * 100),
            "mean_Er_pct": float(np.mean([r["er"] for r in s])),
            "PHM_score": float(np.mean([r["score"] for r in s])),
            "coverage95": float(cov),
            "MPIW": float(np.mean([r["hi"] - r["lo"] for r in s])),
            "rho": float(np.mean([r["rho"] for r in s])),
            "capped_pct": float(100 * np.mean([r.get("capped", False) for r in s])),
        }
    return out


def show(title, tbl, unit="s"):
    print(f"\n{title}")
    hdr = ["n", "MAE", "RMSE", "MAPE_pct", "MedAPE_pct", "PHM_score",
           "coverage95", "MPIW", "rho"]
    print(f"  {'method':10s} " + " ".join(f"{h:>10s}" for h in hdr))
    for k, v in tbl.items():
        print(f"  {k:10s} " + " ".join(
            f"{v[h]:10.0f}" if h in ("n",) else f"{v[h]:10.3f}" for h in hdr))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    todo = ["pronostia", "xjtu"] if which == "both" else [which]
    allrows, tables = [], {}
    for dset in todo:
        t0 = time.time()
        print(f"\n{'='*78}\n{dset.upper()}\n{'='*78}")

        rows = run(dset, laws=("BS", "BS-t", "Wiener", "Gamma", "IGproc"))
        tables[f"{dset}/process"] = summarise(rows)
        show(f"[{dset}] increment law (physics-informed prior, Paris coupling on)",
             tables[f"{dset}/process"])
        tables[f"{dset}/process_postfpt"] = summarise(rows, subset=lambda r: r["post_fpt"])
        show(f"[{dset}] same, restricted to prediction points AFTER the detected onset",
             tables[f"{dset}/process_postfpt"])
        if dset == "pronostia":
            tables["pronostia/official"] = summarise(
                rows, subset=lambda r: r["official"])
            show("[pronostia] official IEEE PHM 2012 truncation points only",
                 tables["pronostia/official"])
        allrows += rows

        # The ablation and the grid comparison use every second prediction point
        # and 600 particles: they compare variants of one model against each
        # other, so they need less resolution than the headline table.
        abl = []
        for inf in (True, False):
            for cp in (True, False):
                r2 = run(dset, laws=("BS",), informed=inf, coupling=cp,
                         n_part=600, stride=2)
                for r in r2:
                    r["variant"] = f"prior={'informed' if inf else 'diffuse'}, " \
                                   f"coupling={'on' if cp else 'off'}"
                abl += r2
        tables[f"{dset}/ablation"] = summarise(abl, by="variant")
        show(f"[{dset}] ablation of the BS model", tables[f"{dset}/ablation"])
        allrows += abl

        grid = run(dset, laws=("BS", "BS-naive"), n_part=600, stride=2)
        tables[f"{dset}/grid"] = summarise(grid)
        show(f"[{dset}] grid-consistent vs naive BS scaling", tables[f"{dset}/grid"])
        allrows += grid
        print(f"\n  ({time.time()-t0:.0f} s)", flush=True)

        # checkpoint after each dataset so a long run is never lost
        with open(os.path.join(RES, f"results_{which}.json"), "w") as f:
            json.dump({"tables": tables, "rows": allrows}, f, indent=1, default=float)
        print(f"  checkpoint written after {dset}", flush=True)

    with open(os.path.join(RES, f"results_{which}.json"), "w") as f:
        json.dump({"tables": tables, "rows": allrows}, f, indent=1, default=float)
    print(f"\nwrote results/bearings/results_{which}.json")
