"""
Grid consistency of the Birnbaum-Saunders increment process.

Claim.  Under the naive scaling dX ~ BS(alpha, beta*s) the accumulated variance
over a fixed horizon T is proportional to the inspection interval s, so the
process -- and every first-passage quantity derived from it -- is an artefact of
how often the asset happens to be measured.  The moment-consistent
parameterisation (alpha_s, beta_s) removes this dependence, and the resulting
first-passage law converges to a single grid-invariant limit.
"""
import json
import os

import numpy as np

import gbs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "theory")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(17)

ALPHA, BETA, T, D = 0.4, 1.0, 20.0, 12.0
NS = [1, 2, 4, 8, 16, 32, 64]
res = {"alpha": ALPHA, "beta": BETA, "T": T, "D": D, "naive": [], "consistent": []}

print("Accumulated moments of X(T) over a FIXED horizon T = %.0f, split into n steps" % T)
print(f"{'n':>4} {'s=T/n':>7} | {'naive E':>9} {'naive Var':>11} | "
      f"{'cons. E':>9} {'cons. Var':>11} {'alpha_s':>8} {'beta_s':>8}")
print("-" * 82)
for n in NS:
    s = T / n
    th = {"alpha": np.array([ALPHA]), "beta": np.array([BETA])}
    naive, cons = gbs.BSLaw(consistent=False), gbs.BSLaw(consistent=True)
    mn, vn = float(naive.mean(th, s)) * n, float(naive.var(th, s)) * n
    mc, vc = float(cons.mean(th, s)) * n, float(cons.var(th, s)) * n
    a_s, b_s = gbs.bs_consistent_params(ALPHA, BETA, s)
    print(f"{n:4d} {s:7.3f} | {mn:9.4f} {vn:11.5f} | {mc:9.4f} {vc:11.5f} "
          f"{float(a_s):8.4f} {float(b_s):8.4f}")
    res["naive"].append({"n": n, "mean": mn, "var": vn})
    res["consistent"].append({"n": n, "mean": mc, "var": vc,
                              "alpha_s": float(a_s), "beta_s": float(b_s)})

sp = lambda k, w: max(r[w] for r in res[k]) / min(r[w] for r in res[k])
print(f"\n  variance spread over the grids:  naive x{sp('naive','var'):.1f}   "
      f"consistent x{sp('consistent','var'):.4f}")
res["var_spread_naive"] = sp("naive", "var")
res["var_spread_consistent"] = sp("consistent", "var")

# ── first-passage law under grid refinement ────────────────────────────────
print("\nFirst-passage time to D = %.0f, simulated on each grid (continuous crossing):" % D)
print(f"{'n steps/unit':>13} | {'naive E[tau]':>13} {'naive sd':>9} | "
      f"{'cons. E[tau]':>13} {'cons. sd':>9}")
print("-" * 68)
fp = {"naive": [], "consistent": []}
for n in [1, 2, 4, 8, 16]:
    s = 1.0 / n
    for tag, law in (("naive", gbs.BSLaw(consistent=False)),
                     ("consistent", gbs.BSLaw(consistent=True))):
        n_sim, nmax = 40_000, int(400 * n)
        th = {"alpha": np.full(n_sim, ALPHA), "beta": np.full(n_sim, BETA)}
        x = np.zeros(n_sim); tau = np.zeros(n_sim); alive = np.ones(n_sim, bool)
        for _ in range(nmax):
            if not alive.any():
                break
            dx = law.sample(th, s, rng)
            xn = x + dx
            cr = alive & (xn >= D)
            fr = np.where(cr, (D - x) / np.maximum(dx, 1e-300), 1.0)
            tau = np.where(alive, tau + s * np.clip(fr, 0, 1), tau)
            x = np.where(alive, xn, x); alive &= ~cr
        fp[tag].append({"steps_per_unit": n, "mean": float(tau.mean()),
                        "sd": float(tau.std())})
    print(f"{n:13d} | {fp['naive'][-1]['mean']:13.3f} {fp['naive'][-1]['sd']:9.3f} | "
          f"{fp['consistent'][-1]['mean']:13.3f} {fp['consistent'][-1]['sd']:9.3f}")
res["first_passage"] = fp
sdn = [r["sd"] for r in fp["naive"]]
sdc = [r["sd"] for r in fp["consistent"]]
print(f"\n  s.d. of the RUL spread over grids: naive x{max(sdn)/min(sdn):.1f}   "
      f"consistent x{max(sdc)/min(sdc):.3f}")
res["rul_sd_spread_naive"] = max(sdn) / min(sdn)
res["rul_sd_spread_consistent"] = max(sdc) / min(sdc)

#  The consistent parameterisation does not leave the crossing law grid-free,
#  and cannot: matching two moments says nothing about the third, and BS is not
#  closed under convolution. What is left is small but it is systematic, and
#  calling it Monte Carlo error would be wrong -- so measure it rather than
#  characterise it. The s.e. of a sample s.d. is sd/sqrt(2n).
_lg = [np.log2(r["steps_per_unit"]) for r in fp["consistent"]]
_b, _a = np.polyfit(_lg, sdc, 1)
_resid = np.array(sdc) - (_b * np.array(_lg) + _a)
_se_slope = float(np.sqrt((_resid ** 2).sum() / (len(sdc) - 2)
                          / ((np.array(_lg) - np.mean(_lg)) ** 2).sum()))
_se_sd = float(np.mean(sdc) / np.sqrt(2 * n_sim))
res["rul_sd_residual"] = {
    "monotone_in_rate": bool(all(b > a for a, b in zip(sdc, sdc[1:]))),
    "slope_per_doubling": float(_b),
    "slope_se": _se_slope,
    "t": float(_b / _se_slope) if _se_slope else None,
    "mc_se_of_each_sd": _se_sd,
    "drift_in_mc_se": float((max(sdc) - min(sdc)) / _se_sd),
    "n_paths_per_cell": n_sim,
}
print(f"  residual drift: {'monotone' if res['rul_sd_residual']['monotone_in_rate'] else 'not monotone'}"
      f", {_b:+.4f} per doubling (t={res['rul_sd_residual']['t']:.1f}), "
      f"{res['rul_sd_residual']['drift_in_mc_se']:.1f} MC standard errors")

with open(os.path.join(OUT, "grid_consistency.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote results/theory/grid_consistency.json")
