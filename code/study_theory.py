"""
Numerical validation of every analytical claim in the paper.

Produces  results/theory/*.json  (numbers quoted in the text and tables) and
results/figures/fig_theory_*.pdf|png.

  T1  first-passage of iid increments -> BS, against the Berry-Esseen bound
  T2  IG approximation to the multi-step RUL: continuous vs lattice crossing
  T3  particle-filter Monte-Carlo error ~ N^{-1/2}
  T4  epistemic-variance decay and the Liu-West variance floor
  T5  credible-interval calibration of the predictive RUL
  T6  BS vs BS-t under impulsive contamination
  T7  value of the physics-informed prior early in life
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from scipy import stats

import gbs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "theory")
FIG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

figstyle.apply()
C, HEADC, HEADL = figstyle.C, figstyle.HEADC, figstyle.HEADL
OUT: dict = {}


def save(fig, name):
    figstyle.save(fig, name, FIG)
    print(f"  -> {name}")


# ══════════════════════════════════════════════════════════════════════════
def t1_derivation(rng):
    """sup-norm distance between the exact first-passage law and BS(alpha,beta)."""
    print("T1  BS derivation / Berry-Esseen")
    mu_y, sd_y = 1.0, 0.35
    shape_g = (mu_y / sd_y) ** 2
    rho3 = stats.gamma.expect(lambda x: abs(x - shape_g) ** 3,
                              args=(shape_g,)) / shape_g ** 1.5
    omegas = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    rows = []
    for omega in omegas:
        alpha = sd_y / np.sqrt(mu_y * omega)
        beta = omega / mu_y
        nmax = int(2.2 * beta + 60)
        n_sim, chunk = 200_000, max(1, int(3e7 // nmax))
        Ns = []
        for lo in range(0, n_sim, chunk):
            m = min(chunk, n_sim - lo)
            Y = rng.gamma(shape_g, sd_y ** 2 / mu_y, size=(m, nmax)).astype(np.float32)
            S = np.cumsum(Y, axis=1)
            Ns.append(np.argmax(S >= omega, axis=1) + 1.0)
            del Y, S
        N = np.concatenate(Ns)
        ns = np.unique(np.linspace(0.55 * beta, 1.8 * beta, 90).astype(int))
        ns = ns[(ns >= 1) & (ns < nmax)]
        emp = np.array([(N <= n).mean() for n in ns])
        th = stats.norm.cdf(gbs.xi(ns / beta) / alpha)
        sup = float(np.max(np.abs(emp - th)))
        be = float(0.4748 * rho3 / np.sqrt(omega / mu_y))
        rows.append({"omega_over_mu": omega / mu_y, "alpha": alpha,
                     "sup_error": sup, "be_bound": be, "holds": sup <= be})
        print(f"    w/mu={omega/mu_y:6.0f}  alpha={alpha:.4f}  sup={sup:.4f}  BE={be:.4f}")
    OUT["T1"] = {"rho3_over_sigma3": rho3, "rows": rows}

    fig, ax = plt.subplots(figsize=figstyle.size("fig_theory_derivation", 0.8))
    x = [r["omega_over_mu"] for r in rows]
    ax.loglog(x, [r["sup_error"] for r in rows], "o-", color=C["gbs"],
              label=r"measured $\sup_n|F_N-F_{\rm BS}|$")
    ax.loglog(x, [r["be_bound"] for r in rows], "s--", color=C["ig"],
              label=r"Berry--Esseen bound")
    ax.loglog(x, 0.03 * np.array(x, float) ** -0.5, ":", color=C["ref"],
              label=r"$\propto(\omega/\mu)^{-1/2}$")
    ax.set_xlabel(r"expected cycles to failure $\omega/\mu$")
    ax.set_ylabel("sup-norm error")
    ax.legend(frameon=False)
    save(fig, "fig_theory_derivation")


# ══════════════════════════════════════════════════════════════════════════
def t2_ig_approx(rng):
    """Accuracy of RUL ~ IG(mu,lam): continuous crossing time vs step index."""
    print("T2  IG approximation to the multi-step RUL")
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    nds = [5, 10, 25, 50, 100]
    tab_c = np.zeros((len(alphas), len(nds)))
    tab_l = np.zeros_like(tab_c)
    for i, a in enumerate(alphas):
        for j, nd in enumerate(nds):
            b = 1.0
            d = nd * gbs.bs_mean(a, b)
            nmax = int(4 * nd + 40)
            n_sim = 60_000
            dx = gbs.bs_rvs(a, b, size=(n_sim, nmax), rng=rng)
            S = np.cumsum(dx, axis=1)
            N = np.argmax(S >= d, axis=1)
            Sprev = np.where(N > 0, S[np.arange(n_sim), np.maximum(N - 1, 0)], 0.0)
            tau = N + (d - Sprev) / dx[np.arange(n_sim), N]
            mu_r, lam_r = d / gbs.bs_mean(a, b), d ** 2 / gbs.bs_var(a, b)
            grid = np.linspace(0.25 * mu_r, 2.6 * mu_r, 300)
            F = gbs.ig_cdf(grid, mu_r, lam_r)
            tab_c[i, j] = np.max(np.abs(np.array([(tau <= g).mean() for g in grid]) - F))
            tab_l[i, j] = np.max(np.abs(np.array([(N + 1.0 <= g).mean() for g in grid]) - F))
        print(f"    alpha={a}: continuous " +
              " ".join(f"{v:.4f}" for v in tab_c[i]))
    OUT["T2"] = {"alphas": alphas, "n_d": nds,
                 "sup_continuous": tab_c.tolist(), "sup_lattice": tab_l.tolist()}

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size("fig_theory_igapprox", 0.42))
    for i, a in enumerate(alphas):
        axes[0].loglog(nds, tab_c[i], "o-", label=fr"$\alpha={a}$")
        axes[1].loglog(nds, tab_l[i], "o-", label=fr"$\alpha={a}$")
    axes[0].loglog(nds, 0.09 * np.array(nds, float) ** -0.5, "k:", label=r"$\propto n_d^{-1/2}$")
    for ax, ttl in zip(axes, ["(a) continuous crossing time",
                              "(b) step index (lattice)"]):
        ax.set_xlabel(r"expected remaining increments $n_d$")
        ax.set_title(ttl)
    axes[0].set_ylabel(r"$\sup_\tau|F_{\rm RUL}-F_{\rm IG}|$")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(0.05, color=C["ref"], ls="--", lw=0.8)
    axes[1].annotate("$0.05$, the switching criterion", xy=(6, 0.05),
                     xytext=(0, 4), textcoords="offset points",
                     fontsize=6.5, color=C["ref"])
    save(fig, "fig_theory_igapprox")


# ══════════════════════════════════════════════════════════════════════════
def _synth(rng, K=300, a=0.40, b=0.012, se=0.010, dt=1.0, nu_contam=None,
           contam_frac=0.0):
    x = np.zeros(K + 1)
    for k in range(1, K + 1):
        x[k] = x[k - 1] + gbs.bs_rvs(a, b, size=1, rng=rng)[0]
    eps = se * rng.standard_normal(K + 1)
    if contam_frac > 0:
        m = rng.random(K + 1) < contam_frac
        eps[m] = se * 8.0 * rng.standard_t(nu_contam or 3, size=m.sum())
    return x, x + eps, np.arange(K + 1) * dt


PRIOR_M = {"alpha": np.log(0.45), "beta": np.log(0.015), "sigma_eps": np.log(0.015)}
PRIOR_S = {"alpha": 0.5, "beta": 0.7, "sigma_eps": 0.5}


def t3_pf_convergence(rng):
    """Monte-Carlo error of the particle approximation vs N_p."""
    print("T3  particle-filter convergence")
    x, y, t = _synth(np.random.default_rng(11), K=200)
    Ns = [50, 100, 200, 400, 800, 1600, 3200]
    reps = 24
    sds = []
    for N in Ns:
        est = []
        for r in range(reps):
            pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=N, seed=1000 * r + N))
            st = pf.run(y, t, 1.0, PRIOR_M, PRIOR_S, x0=y[0])[-1]
            w = np.exp(st.logw - st.logw.max()); w /= w.sum()
            est.append(float(w @ pf._theta(st.logth)["beta"]))
        sds.append(float(np.std(est)))
        print(f"    N={N:5d}  sd(E[beta|Y])={sds[-1]:.3e}")
    sl, ic = np.polyfit(np.log(Ns), np.log(sds), 1)
    #  Store the whole fit, not just the point estimate.  The manuscript quotes
    #  the standard error, the interval, R^2 and the test against -1/2, and a
    #  number that lives only in the prose is a number nobody can re-check.
    #  Seven design points and two parameters leave five degrees of freedom, so
    #  the interval takes a t quantile, matching the test reported beside it --
    #  a normal quantile here would understate the width by a quarter.
    fit = stats.linregress(np.log(Ns), np.log(sds))
    df = len(Ns) - 2
    tq = float(stats.t.ppf(0.975, df))
    t_stat = float((fit.slope - (-0.5)) / fit.stderr)
    OUT["T3"] = {"N": Ns, "sd": sds, "slope": float(sl), "target_slope": -0.5,
                 "slope_se": float(fit.stderr), "df": df, "r2": float(fit.rvalue ** 2),
                 "t_vs_target": t_stat,
                 "p_vs_target": float(2 * stats.t.sf(abs(t_stat), df)),
                 "ci95": [float(fit.slope - tq * fit.stderr),
                          float(fit.slope + tq * fit.stderr)]}
    print(f"    fitted log-log slope = {sl:+.3f}  (theory -0.5), "
          f"se {fit.stderr:.3f}, 95% CI "
          f"[{OUT['T3']['ci95'][0]:+.3f}, {OUT['T3']['ci95'][1]:+.3f}]")

    fig, ax = plt.subplots(figsize=figstyle.size("fig_theory_pfconv", 0.92))
    ax.loglog(Ns, sds, "o", color=C["gbs"], label="measured MC s.d.")
    ax.loglog(Ns, np.exp(ic) * np.array(Ns, float) ** sl, "-", color=C["gbs"],
              label=fr"fit: slope $={sl:.2f}$")
    ax.loglog(Ns, sds[0] * (np.array(Ns, float) / Ns[0]) ** -0.5, "--",
              color=C["ref"], label=r"theory $N_p^{-1/2}$")
    ax.set_xlabel(r"number of particles $N_p$")
    ax.set_ylabel(r"s.d. of $\hat{\mathbb{E}}[\beta\mid Y_{1:K}]$")
    ax.legend(frameon=False)
    save(fig, "fig_theory_pfconv")


def t4_epistemic(rng):
    """Posterior contraction of the parameter functional h(theta)=1/[beta(1+a^2/2)].

    The Bernstein-von Mises prediction is O(1/k).  delta = 1 switches the
    Liu-West jitter off entirely: the parameter particles are then never
    refreshed, resampling collapses them onto a handful of distinct values and
    the reported variance falls to ~0 -- degeneracy masquerading as certainty,
    not an exact posterior.  The number of distinct parameter particles is
    tracked to make that visible.
    """
    print("T4  epistemic-variance decay and the role of the Liu-West jitter")
    x, y, t = _synth(np.random.default_rng(5), K=600)
    deltas = [0.95, 0.98, 0.99, 0.995, 1.0]
    curves, uniq = {}, {}
    for dl in deltas:
        pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=4000, delta=dl, seed=3))
        states = pf.run(y, t, 1.0, PRIOR_M, PRIOR_S, x0=y[0])
        v, u = [], []
        for st in states:
            w = np.exp(st.logw - st.logw.max()); w /= w.sum()
            th = pf._theta(st.logth)
            h = 1.0 / (th["beta"] * (1 + th["alpha"] ** 2 / 2))
            v.append(float(w @ h ** 2 - (w @ h) ** 2))
            u.append(int(len(np.unique(np.round(th["beta"], 12)))))
        curves[dl], uniq[dl] = v, u
        va = np.maximum(np.array(v), 1e-300)
        # asymptotic window only: the early transient is prior-dominated
        ks = np.arange(150, len(v))
        sl = np.polyfit(np.log(ks), np.log(va[150:]), 1)[0]
        print(f"    delta={dl:5.3f}  slope(k>=150)={sl:+.3f}   "
              f"V(600)/V(150)={v[-1]/v[149]:.4f}   distinct theta at k=600: {u[-1]:5d}/4000")
        OUT.setdefault("T4", {})[str(dl)] = {
            "slope_asymptotic": float(sl), "ratio_600_150": float(v[-1] / v[149]),
            "distinct_theta_end": u[-1]}
    OUT["T4"]["curves"] = {str(k): v for k, v in curves.items()}
    OUT["T4"]["distinct"] = {str(k): v for k, v in uniq.items()}

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size("fig_theory_epistemic", 0.53))
    #  The diversity count jumps between about 1500 and the 4000 ceiling from one
    #  resampling to the next. A smoothed level would sit at the ceiling and so
    #  contradict the text, which quotes the count at the last step; the raw
    #  series is drawn instead and that last value is marked, which is the
    #  quantity the claim is about.
    for dl, v in curves.items():
        lbl = fr"$\delta={dl}$" + (" (no jitter)" if dl == 1.0 else "")
        axes[0].loglog(np.arange(1, len(v) + 1), v, label=lbl)
        u = np.asarray(uniq[dl], float)
        k = np.arange(1, len(u) + 1)
        col = f"C{list(curves).index(dl)}"
        axes[1].semilogx(k, u, lw=0.6, alpha=0.45 if dl < 1.0 else 1.0, color=col)
        axes[1].plot([k[-1]], [u[-1]], "o", ms=3.5, color=col, zorder=4)
    fin = [uniq[d][-1] for d in curves if d < 1.0]
    axes[1].annotate(f"{min(fin):.0f}--{max(fin):.0f} still distinct",
                     xy=(len(u), max(fin)), xytext=(-3, 9),
                     textcoords="offset points", ha="right", fontsize=6.5,
                     color="#444444")
    axes[1].annotate("collapsed to 1", xy=(len(u), 1), xytext=(-3, 7),
                     textcoords="offset points", ha="right", fontsize=6.5,
                     color="C4")
    k = np.arange(150, 600)
    axes[0].loglog(k, curves[0.98][149] * (k / 150.0) ** -1.0, "k:", label=r"$O(1/k)$")
    axes[0].set_ylabel(r"$\mathrm{Var}_{\theta\mid Y}[h(\theta)]$")
    axes[1].set_ylabel("distinct parameter particles")
    for ax in axes:
        ax.set_xlabel("observation index $k$")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig_theory_epistemic")


def t5_calibration(rng):
    """Coverage of the 95% predictive RUL interval on synthetic truth."""
    print("T5  predictive-interval calibration")
    rows = []
    for rep in range(60):
        r = np.random.default_rng(200 + rep)
        x, y, t = _synth(r, K=320)
        D = 0.75 * x[-1]
        kf = int(np.argmax(x >= D))
        if kf < 60:
            continue
        pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=1500, seed=rep))
        states = pf.run(y[:kf], t[:kf], 1.0, PRIOR_M, PRIOR_S, x0=y[0])
        for frac in (0.4, 0.6, 0.8):
            k = int(frac * kf)
            if k < 25:
                continue
            p = gbs.predict_rul(states[k - 1], pf, D, 1.0)
            lo, hi = np.quantile(p["samples"], [0.025, 0.975])
            true = t[kf] - t[k]
            rows.append({"frac": frac, "true": float(true), "mean": p["mean"],
                         "lo": float(lo), "hi": float(hi),
                         "cov": bool(lo <= true <= hi), "rho": p["rho"]})
    for frac in (0.4, 0.6, 0.8):
        s = [r for r in rows if r["frac"] == frac]
        cov = np.mean([r["cov"] for r in s])
        mpiw = np.mean([r["hi"] - r["lo"] for r in s])
        err = np.array([r["mean"] - r["true"] for r in s])
        rel = err / np.array([r["true"] for r in s])
        print(f"    life fraction {frac:.0%}: n={len(s):3d} coverage={cov:.3f} "
              f"MPIW={mpiw:6.1f} MAPE={np.mean(np.abs(rel)):.3f} rho={np.mean([r['rho'] for r in s]):.3f}")
        OUT.setdefault("T5", {})[str(frac)] = {
            "n": len(s), "coverage": float(cov), "mpiw": float(mpiw),
            "mape": float(np.mean(np.abs(rel))), "bias": float(np.mean(err)),
            "rho": float(np.mean([r["rho"] for r in s]))}


def t6_robustness(rng):
    """Where do heavy tails have to go?

    Contamination here is impulsive MEASUREMENT noise.  A heavy-tailed
    *transition* (BS-t) does not defend against it -- only a heavy-tailed
    *observation* density down-weights outlying measurements.  Both variants are
    run so the distinction is measured rather than asserted.
    """
    print("T6  heavy-tailed robustness: transition vs observation")
    variants = {
        "BS / Gauss":  (gbs.BSLaw(), "gauss"),
        "BS-t / Gauss": (gbs.BSTLaw(nu=4.0), "gauss"),
        "BS / t":      (gbs.BSLaw(), "t"),
        "BS-t / t":    (gbs.BSTLaw(nu=4.0), "t"),
    }
    res = {}
    for cf in (0.0, 0.05, 0.10, 0.20):
        errs = {k: [] for k in variants}
        for rep in range(30):
            r = np.random.default_rng(900 + rep)
            x, y, t = _synth(r, K=300, contam_frac=cf, nu_contam=2.5)
            for nm, (law, obs) in variants.items():
                pf = gbs.ParticleFilter(law, gbs.PFConfig(
                    n_particles=1200, seed=rep, obs=obs, nu_obs=4.0))
                st = pf.run(y, t, 1.0, PRIOR_M, PRIOR_S, x0=y[0])[-1]
                w = np.exp(st.logw - st.logw.max()); w /= w.sum()
                errs[nm].append(abs(float(w @ st.x) - x[-1]) / x[-1])
        res[str(cf)] = {k: float(np.median(v)) for k, v in errs.items()}
        print(f"    contamination {cf:4.0%}: " +
              "  ".join(f"{k}={res[str(cf)][k]:.4f}" for k in variants))
    OUT["T6"] = res

    fig, ax = plt.subplots(figsize=figstyle.size("fig_theory_robust", 0.8))
    cfs = [0.0, 0.05, 0.10, 0.20]
    for nm, mk in zip(variants, ["o-", "s--", "^-", "v--"]):
        ax.plot([100 * c for c in cfs], [1000 * res[str(c)][nm] for c in cfs], mk, label=nm)
    ax.set_xlabel("impulsive contamination [%]")
    ax.set_ylabel(r"median $|\hat{x}_K-x_K|/x_K\ \times 10^{3}$")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_theory_robust")


def t7_prior_value(rng):
    """Physics-informed vs diffuse prior: RUL error as a function of life fraction."""
    print("T7  value of the physics-informed prior")
    informed_s = {"alpha": 0.35, "beta": 0.50, "sigma_eps": 0.5}
    diffuse_s = {"alpha": 1.60, "beta": 2.50, "sigma_eps": 1.5}
    diffuse_m = {"alpha": np.log(0.6), "beta": np.log(0.05), "sigma_eps": np.log(0.03)}
    out = {"informed": {}, "diffuse": {}}
    for rep in range(45):
        r = np.random.default_rng(400 + rep)
        x, y, t = _synth(r, K=320)
        D = 0.75 * x[-1]
        kf = int(np.argmax(x >= D))
        if kf < 60:
            continue
        for nm, (pm, ps) in (("informed", (PRIOR_M, informed_s)),
                             ("diffuse", (diffuse_m, diffuse_s))):
            pf = gbs.ParticleFilter(gbs.BSLaw(), gbs.PFConfig(n_particles=1500, seed=rep))
            states = pf.run(y[:kf], t[:kf], 1.0, pm, ps, x0=y[0])
            for frac in (0.2, 0.4, 0.6, 0.8):
                k = int(frac * kf)
                if k < 12:
                    continue
                p = gbs.predict_rul(states[k - 1], pf, D, 1.0)
                true = t[kf] - t[k]
                out[nm].setdefault(str(frac), []).append(abs(p["mean"] - true) / true)
    OUT["T7"] = {nm: {f: float(np.median(v)) for f, v in d.items()} for nm, d in out.items()}
    for f in ("0.2", "0.4", "0.6", "0.8"):
        print(f"    life {float(f):.0%}: informed MAPE={OUT['T7']['informed'][f]:.3f}  "
              f"diffuse MAPE={OUT['T7']['diffuse'][f]:.3f}")

    fig, ax = plt.subplots(figsize=figstyle.size("fig_theory_prior", 0.8))
    fr = [0.2, 0.4, 0.6, 0.8]
    ax.plot(fr, [OUT["T7"]["informed"][str(f)] for f in fr], "o-", color=C["gbs"],
            label="physics-informed prior")
    ax.plot(fr, [OUT["T7"]["diffuse"][str(f)] for f in fr], "s--", color=C["ig"],
            label="diffuse prior")
    ax.set_xlabel("fraction of life elapsed at prediction")
    ax.set_ylabel("median relative RUL error")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    save(fig, "fig_theory_prior")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    t1_derivation(rng)
    t2_ig_approx(rng)
    t3_pf_convergence(rng)
    t4_epistemic(rng)
    t5_calibration(rng)
    t6_robustness(rng)
    t7_prior_value(rng)
    with open(os.path.join(RES, "theory_results.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print("\nwrote results/theory/theory_results.json")
