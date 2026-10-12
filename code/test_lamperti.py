"""
Validation of the Lamperti reduction for the Paris-coupled GBS process.

Claim.  If  dX_k ~ BS(alpha_s, beta_s)  with  s = (dt/dt_ref) q(X_{k-1})^gamma,
q(x) = 1+x/x_ref  and  (alpha_s, beta_s)  the GRID-CONSISTENT pair of
bs_consistent_params -- this is Definition "Paris-coupled BS process" -- then
with  psi(x) = int_0^x q(u)^{-gamma} du  the first-passage time of X to D
is approximately IG(mu, lam) with the homogeneous formulas evaluated at
d_Z = psi(D) - psi(x_0):

Note on what is being simulated.  This sweep used to draw
BS(alpha, beta*q^gamma) -- scaling beta with alpha held fixed.  That is the
scale-family shortcut bs_consistent_params exists to rule out: it gives
Var[dX] proportional to s^2 instead of s, so it simulates the naive process
rather than the defined one.  The manuscript's proof of the second-order
reduction contained the matching error (both moments given q^{2 gamma}), so the
two cancelled and the table looked clean.  Both are now fixed.

    mu  = d_Z * dt_ref / [beta (1+alpha^2/2)],
    lam = d_Z^2 * dt_ref / [beta^2 alpha^2 (1 + 5 alpha^2/4)].

The reduction is exact to first order in the increment; the residual error is
controlled by the per-step relative increment  beta/x_ref  (step coarseness).
"""
import json
import os

import numpy as np

import gbs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "theory")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(4)
rows = []
print(f"{'gam/xref':>9}{'alpha':>6} {'n_d':>7} {'sup 1st':>8}{'sup 2nd':>9} "
      f"{'mean%1st':>8}{'mean%2nd':>9} {'sd%2nd':>8}")
print("-" * 68)

for gamma, x_ref in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0),
                     (1.5, 0.4), (1.5, 2.5), (1.0, 0.25)]:
    for alpha in [0.2, 0.4]:
        beta, dt = 0.02, 1.0
        x0, D = 0.0, 3.0
        n_sim, nmax = 40_000, 4000

        x = np.full(n_sim, x0)
        tau = np.zeros(n_sim)
        alive = np.ones(n_sim, dtype=bool)
        for _ in range(nmax):
            if not alive.any():
                break
            s = np.power(1.0 + np.maximum(x, 0.0) / x_ref, gamma)
            a_s, b_s = gbs.bs_consistent_params(alpha, beta, s)
            dx = gbs.bs_rvs(a_s, b_s, rng=rng)
            xn = x + dx
            cross = alive & (xn >= D)
            frac = np.where(cross, (D - x) / np.maximum(dx, 1e-300), 1.0)
            tau = np.where(alive, tau + dt * np.clip(frac, 0, 1), tau)
            x = np.where(alive, xn, x)
            alive &= ~cross
        assert not alive.any(), "some paths did not cross"

        # d_V is the effective distance for the SPREAD.  The first-order column
        # keeps the homogeneous choice d_V = d_Z, which is what a treatment that
        # stops at the drift produces; the second-order column uses both.
        sig2 = beta ** 2 * alpha ** 2 * (1 + 1.25 * alpha ** 2)
        dV = float(gbs.lamperti_var_distance(x0, D, gamma, x_ref))
        res = {}
        for order, tag in ((False, "1st"), (True, "2nd")):
            dZ = float(gbs.lamperti_distance(
                x0, D, gbs.bs_mean(alpha, beta), gbs.bs_var(alpha, beta),
                gamma, x_ref, second_order=order))
            mu = dZ * dt / (beta * (1 + alpha ** 2 / 2))
            lam = dZ ** 3 * dt / (sig2 * (dV if order else dZ))
            grid = np.linspace(0.3 * mu, 2.2 * mu, 300)
            res[tag] = (
                float(np.max(np.abs(np.array([(tau <= g).mean() for g in grid])
                                    - gbs.ig_cdf(grid, mu, lam)))),
                100 * (tau.mean() / mu - 1),
                100 * (tau.std() / np.sqrt(mu ** 3 / lam) - 1),
                mu)
        # The manuscript's C2 claim needs a THIRD variant, which nothing here
        # used to compute: keep kappa in the mean but accumulate the spread over
        # d_Z instead of d_V.  That is what a treatment which stops at the drift
        # produces, and the quoted over-prediction of the RUL standard deviation
        # was not traceable to any stored number.  Store it, in the direction the
        # sentence states it: predicted over simulated, minus one.
        dZ2 = float(gbs.lamperti_distance(
            x0, D, gbs.bs_mean(alpha, beta), gbs.bs_var(alpha, beta),
            gamma, x_ref, second_order=True))
        mu2 = dZ2 * dt / (beta * (1 + alpha ** 2 / 2))
        lam_dZ = dZ2 ** 3 * dt / (sig2 * float(gbs.lamperti_distance(
            x0, D, gbs.bs_mean(alpha, beta), gbs.bs_var(alpha, beta),
            gamma, x_ref, second_order=False)))
        sd_overpred = 100 * (np.sqrt(mu2 ** 3 / lam_dZ) / tau.std() - 1)

        n_d = res["2nd"][3]
        print(f"{gamma:4.1f}/{x_ref:<4.2f}{alpha:6.2f} {n_d:7.0f} "
              f"{res['1st'][0]:8.4f}{res['2nd'][0]:9.4f} "
              f"{res['1st'][1]:+8.2f}{res['2nd'][1]:+9.2f} {res['2nd'][2]:+8.2f}")
        rows.append({"gamma": gamma, "x_ref": x_ref, "alpha": alpha, "n_d": n_d,
                     "sup_1st": res["1st"][0], "sup_2nd": res["2nd"][0],
                     "mean_err_pct_1st": float(res["1st"][1]),
                     "mean_err_pct_2nd": float(res["2nd"][1]),
                     "sd_err_pct_2nd": float(res["2nd"][2]),
                     "sd_overpred_pct_dZ": float(sd_overpred)})

# step-coarseness sensitivity: the reduction degrades as beta/x_ref grows
print(f"\n{'beta/x_ref':>11} {'n_d':>7} {'sup|F-F_IG|':>12} {'E[tau] err%':>12}")
print("-" * 46)
coarse = []
# the sweep runs past the usable range on purpose: the last two rows are
# where a single step moves the state by a third of x_ref and then by more
# than half of it, so the reader can see where the expansion stops working
for beta in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60]:
    gamma, alpha, x_ref, dt = 1.5, 0.3, 1.0, 1.0
    x0, D, n_sim, nmax = 0.0, 3.0, 40_000, 6000
    x = np.full(n_sim, x0); tau = np.zeros(n_sim); alive = np.ones(n_sim, bool)
    for _ in range(nmax):
        if not alive.any():
            break
        s = np.power(1.0 + np.maximum(x, 0.0) / x_ref, gamma)
        a_s, b_s = gbs.bs_consistent_params(alpha, beta, s)
        dx = gbs.bs_rvs(a_s, b_s, rng=rng)
        xn = x + dx
        cross = alive & (xn >= D)
        frac = np.where(cross, (D - x) / np.maximum(dx, 1e-300), 1.0)
        tau = np.where(alive, tau + dt * np.clip(frac, 0, 1), tau)
        x = np.where(alive, xn, x)
        alive &= ~cross
    dZ = float(gbs.lamperti_distance(x0, D, gbs.bs_mean(alpha, beta),
                                     gbs.bs_var(alpha, beta), gamma, x_ref))
    dV = float(gbs.lamperti_var_distance(x0, D, gamma, x_ref))
    mu = dZ * dt / (beta * (1 + alpha ** 2 / 2))
    lam = dZ ** 3 * dt / (beta ** 2 * alpha ** 2 * (1 + 1.25 * alpha ** 2) * dV)
    grid = np.linspace(0.3 * mu, 2.2 * mu, 300)
    sup = float(np.max(np.abs(np.array([(tau <= g).mean() for g in grid])
                              - gbs.ig_cdf(grid, mu, lam))))
    e_mu = 100 * (tau.mean() / mu - 1)
    print(f"{beta/1.0:11.3f} {mu:7.0f} {sup:12.4f} {e_mu:+12.2f}")
    coarse.append({"beta_over_xref": beta, "n_d": mu, "sup": sup, "mean_err_pct": float(e_mu)})

with open(os.path.join(OUT, "lamperti.json"), "w") as f:
    json.dump({"grid": rows, "coarseness": coarse}, f, indent=1)
print("\nwrote results/theory/lamperti.json")
