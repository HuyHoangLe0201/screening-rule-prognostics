"""Moment consistency on irregular inspection schedules.

The homogeneous BS process of the grid study (alpha = 0.4, beta = 1,
dt_ref = 1) is observed over the horizon T = 20 on four kinds of schedule:
  equal      16 equal steps;
  unequal    a fixed unequal partition, steps cycling through 0.25, 4, 0.5, 2, 1;
  random     a new partition for every path, steps drawn from
             {0.25, 0.5, 1, 2, 4} until T is reached (a last remainder under
             0.25 is merged into the previous step);
  coarse     one step of length T (the reference against which the others are
             compared).
Each step is drawn from the moment-consistent law (eq. consistent_root) or from
interval scaling.  Reported: E[X(T)] and Var[X(T)] relative to the targets
T*mu and T*sigma^2, and the mean and s.d. of the interpolated first passage to
D = 12 relative to those on the equal grid.  Writes results/theory/irregular.json
and results/tables/tab_irregular.tex.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_J = os.path.join(ROOT, "results", "theory", "irregular.json")
OUT_T = os.path.join(ROOT, "results", "tables", "tab_irregular.tex")

ALPHA, BETA, T, D = 0.4, 1.0, 20.0, 12.0
STEPS = (0.25, 0.5, 1.0, 2.0, 4.0)
N, SEED = 40000, 3
MU = BETA * (1 + ALPHA ** 2 / 2)
VAR = BETA ** 2 * ALPHA ** 2 * (1 + 1.25 * ALPHA ** 2)


def bs_draw(mean, var, rng):
    """BS variates with the given means and variances (arrays, closed form)."""
    r = var / mean ** 2
    assert np.all(r < 5), "BS infeasible on this grid"
    u = np.sqrt(1 + 3 * r)
    A = 2 * r * (u + 4) / ((5 - r) * (u + 1))
    b = mean / (1 + A / 2)
    h = np.sqrt(A) * rng.standard_normal(mean.shape) / 2
    return b * (h + np.sqrt(h * h + 1)) ** 2


def bs_interval(s, rng):
    """Interval scaling: shape alpha fixed, scale beta*s."""
    h = ALPHA * rng.standard_normal(s.shape) / 2
    return BETA * s * (h + np.sqrt(h * h + 1)) ** 2


def random_partition(rng):
    steps, tot = [], 0.0
    while tot < T - 1e-12:
        d = rng.choice(STEPS)
        steps.append(min(d, T - tot))
        tot += d
    if len(steps) > 1 and steps[-1] < 0.25:
        steps[-2] += steps.pop()
    return steps


def schedules(rng):
    """(name, N x K matrix of step lengths, zero-padded)."""
    out = {"equal": np.full((N, 16), T / 16), "coarse": np.full((N, 1), T)}
    cyc, unequal, tot = (0.25, 4.0, 0.5, 2.0, 1.0), [], 0.0
    while tot < T - 1e-12:
        d = cyc[len(unequal) % len(cyc)]
        unequal.append(min(d, T - tot))
        tot += d
    out["unequal"] = np.tile(unequal, (N, 1))
    parts = [random_partition(rng) for _ in range(N)]
    K = max(len(p) for p in parts)
    M = np.zeros((N, K))
    for i, p in enumerate(parts):
        M[i, :len(p)] = p
    out["random"] = M
    return out


def run(dts, consistent, rng):
    """X(T) and the interpolated first passage to D for paths with steps dts."""
    s = dts                                      # dt_ref = 1
    on = s > 0
    if consistent:
        dx = np.zeros_like(s)
        dx[on] = bs_draw(MU * s[on], VAR * s[on], rng)
    else:
        dx = np.where(on, bs_interval(np.where(on, s, 1.0), rng), 0.0)
    X = np.cumsum(dx, axis=1)
    t = np.cumsum(s, axis=1)
    hit = X >= D
    k = np.argmax(hit, axis=1)
    ok = hit.any(axis=1)
    x_prev = np.where(k > 0, X[np.arange(N), k - 1], 0.0)
    t_prev = np.where(k > 0, t[np.arange(N), k - 1], 0.0)
    tau = t_prev + (D - x_prev) / dx[np.arange(N), k] * s[np.arange(N), k]
    tau = np.where(ok, tau, np.nan)
    return X[:, -1], tau


def main():
    rng = np.random.default_rng(SEED)
    sch = schedules(rng)
    res = []
    for name in ("coarse", "equal", "unequal", "random"):
        for cons in (True, False):
            XT, tau = run(sch[name], cons, rng)
            res.append({"schedule": name, "model": "consistent" if cons else "interval",
                        "steps": float(np.mean((sch[name] > 0).sum(1))),
                        "mean_ratio": float(XT.mean() / (T * MU)),
                        "var_ratio": float(XT.var() / (T * VAR)),
                        "tau_mean": float(np.nanmean(tau)), "tau_sd": float(np.nanstd(tau)),
                        "uncrossed": float(np.mean(np.isnan(tau)))})
    ref = {r["model"]: r for r in res if r["schedule"] == "equal"}
    for r in res:
        r["tau_mean_rel"] = r["tau_mean"] / ref[r["model"]]["tau_mean"]
        r["tau_sd_rel"] = r["tau_sd"] / ref["consistent"]["tau_sd"]
        print(f"{r['schedule']:8s} {r['model']:10s} steps {r['steps']:5.1f}"
              f"  E {r['mean_ratio']:.4f}  Var {r['var_ratio']:.4f}"
              f"  tau mean {r['tau_mean']:.3f}  sd {r['tau_sd']:.3f}  uncrossed {r['uncrossed']:.4f}")
    os.makedirs(os.path.dirname(OUT_J), exist_ok=True)
    json.dump({"alpha": ALPHA, "beta": BETA, "T": T, "D": D, "n": N, "rows": res},
              open(OUT_J, "w"), indent=1)
    write_table(res)


def write_table(res):
    names = {"coarse": "one step", "equal": "16 equal", "unequal": "unequal, fixed",
             "random": "random per path"}
    body = []
    for name in ("coarse", "equal", "unequal", "random"):
        c = next(r for r in res if r["schedule"] == name and r["model"] == "consistent")
        i = next(r for r in res if r["schedule"] == name and r["model"] == "interval")
        #  one step spans the whole horizon: its crossing time is only a linear
        #  interpolation inside that step, so its s.d. is not comparable
        sc = "--" if name == "coarse" else f"{c['tau_sd']:.3f}"
        si = "--" if name == "coarse" else f"{i['tau_sd']:.3f}"
        body.append(f"{names[name]} & {c['steps']:.1f} & {c['mean_ratio']:.3f} & {c['var_ratio']:.3f}"
                    f" & {sc} & {i['mean_ratio']:.3f} & {i['var_ratio']:.3f} & {si}" + r"\\")
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Irregular inspection schedules over the horizon $T=20$ of"
           r" \Cref{tab:grid} ($\alpha=0.4$, $\beta=1$). Mean and variance of $X(T)$"
           r" relative to $T\mu_\Delta$ and $T\sigma_\Delta^2$; s.d.\ of the first"
           r" passage to $D=12$. Random: a new partition for every path, steps from"
           r" $\{0.25,0.5,1,2,4\}$. $4\times10^4$ paths per row.}",
           r"\label{tab:irregular}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}lr rrr rrr@{}}", r"\toprule",
           r"& & \multicolumn{3}{c}{moment-consistent} & \multicolumn{3}{c}{interval scaling}\\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"schedule & steps & mean & variance & s.d.\ $\tau$ & mean & variance & s.d.\ $\tau$\\",
           r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(OUT_T, "w", encoding="utf-8", newline="\n").write("\n".join(tex))


if __name__ == "__main__":
    main()
