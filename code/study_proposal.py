"""Does the proposal have to be observation-aware?

Paper 3 claims, in its abstract, in contribution C1 and in its first conclusion,
that "a bootstrap proposal loses a strictly increasing latent state against a
flat indicator, so the proposal must be observation-aware". No experiment in this
repository tested that. This script does.

The comparison is paired: the same simulated record, the same seed, the same
prior, the same particle count, changing only PFConfig.proposal. Reported per
cell:

  MAE(E[X_k|Y] , truth)      how well the latent path is tracked
  drift                       mean of (posterior mean - observation), signed;
                              the failure the claim describes is a positive
                              drift that does not come back
  ESS/N at the end            whether the weights have degenerated

Usage:  python study_proposal.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import gbs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "theory")
os.makedirs(OUT, exist_ok=True)

ALPHA, BETA = 0.4, 0.05
K = 400
SEEDS = range(12)

REGIMES = {
    "flat": lambda t: np.full_like(t, 1.0),
    "flat then cliff": lambda t: 1.0 + 1.5 * (t / K) ** 8,
    "steady growth": lambda t: 1.0 + 0.004 * t,
}


def run_one(prop, y, t, sig, seed):
    pf = gbs.ParticleFilter(
        gbs.BSLaw(),
        gbs.PFConfig(n_particles=1500, seed=seed, proposal=prop))
    sts = pf.run(y, t, 1.0,
                 {"alpha": np.log(ALPHA), "beta": np.log(BETA),
                  "sigma_eps": np.log(sig)},
                 {"alpha": 0.5, "beta": 0.8, "sigma_eps": 0.5},
                 x0=float(y[0]))
    xh = []
    for st in sts:
        w = np.exp(st.logw - st.logw.max())
        w /= w.sum()
        xh.append(float(w @ st.x))
    last = sts[-1]
    w = np.exp(last.logw - last.logw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2) / len(w))
    return np.array(xh), ess


def main():
    rows = []
    t = np.arange(K + 1, dtype=float)
    for rname, fn in REGIMES.items():
        truth = fn(t)
        for sig in (0.01, 0.04, 0.10):
            acc = {p: {"mae": [], "drift": [], "ess": []}
                   for p in ("bootstrap", "guided")}
            for sd in SEEDS:
                rng = np.random.default_rng(1000 + sd)
                y = truth + rng.normal(0, sig, K + 1)
                for prop in ("bootstrap", "guided"):
                    xh, ess = run_one(prop, y, t, sig, seed=11 + sd)
                    m = min(len(xh), len(truth))
                    acc[prop]["mae"].append(float(np.mean(np.abs(xh[:m] - truth[:m]))))
                    acc[prop]["drift"].append(float(np.mean(xh[:m] - y[:m])))
                    acc[prop]["ess"].append(ess)
            row = {"regime": rname, "sigma": sig}
            for prop in ("bootstrap", "guided"):
                for k in ("mae", "drift", "ess"):
                    row[f"{prop}_{k}"] = float(np.mean(acc[prop][k]))
                row[f"{prop}_mae_sd"] = float(np.std(acc[prop]["mae"]))
            rows.append(row)
            print(f"  {rname:16s} sigma={sig:.2f} | "
                  f"bootstrap MAE={row['bootstrap_mae']:.4f}"
                  f"({row['bootstrap_mae_sd']:.4f}) drift={row['bootstrap_drift']:+.4f} "
                  f"ESS={row['bootstrap_ess']:.2f} | "
                  f"guided MAE={row['guided_mae']:.4f}"
                  f"({row['guided_mae_sd']:.4f}) drift={row['guided_drift']:+.4f} "
                  f"ESS={row['guided_ess']:.2f}")

    wins = sum(1 for r in rows if r["guided_mae"] < r["bootstrap_mae"])
    print(f"\n  guided beats bootstrap on MAE in {wins}/{len(rows)} cells")
    with open(os.path.join(OUT, "proposal.json"), "w") as f:
        json.dump({"n_seeds": len(list(SEEDS)), "K": K, "alpha": ALPHA,
                   "beta": BETA, "rows": rows, "guided_wins": wins}, f, indent=1)
    print("  -> results/theory/proposal.json")


if __name__ == "__main__":
    main()
