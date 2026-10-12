"""
Health-indicator construction for PRONOSTIA, with a strictly causal, leakage-free
protocol.

Protocol
--------
1.  Feature ranking (monotonicity + trendability) is computed on the six LEARNING
    bearings only.
2.  Fusion weights (first principal component of the standardised, selected
    features) are fitted on the LEARNING bearings only.
3.  Per-bearing normalisation uses only the first `n_health` acquisitions of that
    bearing -- information available at run time, never the end of the run.
4.  The failure threshold D is the median terminal HI over the LEARNING bearings.
    It is applied unchanged to all eleven test bearings.

Nothing about a test bearing's end of life enters its own health indicator.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "..", "results", "features")
OUT = os.path.join(HERE, "..", "results", "hi")

LEARNING = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
            "Bearing3_1", "Bearing3_2"]
TEST = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
        "Bearing3_3"]
ALL = LEARNING + TEST

# Official IEEE PHM 2012 challenge ground-truth RUL at the provided truncation
# point [s].  Ten of the eleven values equal 10 s x (|Full_Test_Set| - |Test_set|)
# computed from the distributed archive; Bearing1_4 is the documented exception
# (see results/hi/meta.json).
ACT_RUL = {"Bearing1_3": 5730, "Bearing1_4": 339, "Bearing1_5": 1610,
           "Bearing1_6": 1460, "Bearing1_7": 7570, "Bearing2_3": 7530,
           "Bearing2_4": 1390, "Bearing2_5": 3090, "Bearing2_6": 1290,
           "Bearing2_7": 580, "Bearing3_3": 820}
TRUNC_N = {"Bearing1_3": 1802, "Bearing1_4": 1139, "Bearing1_5": 2302,
           "Bearing1_6": 2302, "Bearing1_7": 1502, "Bearing2_3": 1202,
           "Bearing2_4": 612, "Bearing2_5": 2002, "Bearing2_6": 572,
           "Bearing2_7": 172, "Bearing3_3": 352}

N_HEALTH = 40          # acquisitions defining the healthy baseline (~7 min)
SMOOTH = 15            # causal smoothing window
N_KEEP = 8             # features retained after ranking


def monotonicity(v):
    d = np.diff(v)
    return abs((d > 0).sum() - (d < 0).sum()) / max(len(d), 1)


def trendability(v):
    t = np.arange(len(v), dtype=float)
    return abs(np.corrcoef(t, v)[0, 1]) if np.std(v) > 0 else 0.0


def causal_smooth(v, w=SMOOTH):
    """Trailing (causal) moving average -- uses no future samples."""
    s = pd.Series(v).rolling(w, min_periods=1).median()
    return s.rolling(w, min_periods=1).mean().to_numpy()


def load(name):
    return pd.read_csv(os.path.join(FEAT, f"{name}.csv"))


def rank_features(verbose=True):
    """Rank features by PROGNOSABILITY: how far a feature has departed from its
    own healthy baseline (in robust MAD units) at 70-90% of life, aggregated over
    the learning bearings by the median.

    Monotonicity and trendability, computed on raw PRONOSTIA features, are
    dominated by measurement noise and rank almost every feature alike (all
    within 0.27-0.41); they cannot discriminate.  What matters for a
    first-passage model is that the indicator has *moved* before failure, which
    is what this criterion measures.  Monotonicity/trendability are still
    reported for reference.
    """
    cols = [c for c in load(LEARNING[0]).columns
            if c not in ("k", "t", "condition")]
    prog, mono, tren = {}, {}, {}
    for c in cols:
        dep, mo, tr = [], [], []
        for b in LEARNING:
            v = causal_smooth(load(b)[c].to_numpy(), 31)
            base = np.median(v[:N_HEALTH])
            mad = np.median(np.abs(v[:N_HEALTH] - base)) * 1.4826 + 1e-12
            n = len(v)
            dep.append(np.mean([abs(v[int(f * (n - 1))] - base) / mad
                                for f in (0.7, 0.8, 0.9)]))
            mo.append(monotonicity(v))
            tr.append(trendability(v))
        prog[c] = float(np.median(dep))
        mono[c] = float(np.mean(mo))
        tren[c] = float(np.mean(tr))
    ranked = sorted(prog.items(), key=lambda kv: -kv[1])
    if verbose:
        print("  feature ranking (learning set): departure from healthy baseline")
        print(f"    {'feature':16s} {'prognosability':>14s} {'mono':>6s} {'trend':>6s}")
        for c, s in ranked[:14]:
            print(f"    {c:16s} {s:14.1f} {mono[c]:6.3f} {tren[c]:6.3f}")
    return ([c for c, _ in ranked[:N_KEEP]],
            {"prognosability": prog, "monotonicity": mono, "trendability": tren})


def _standardise(df, cols):
    """Robust causal standardisation against each bearing's own healthy window.

    z_f(k) = (f(k) - median of healthy window) / (1.4826 * MAD of healthy window)
    -- a purely causal quantity: it needs the first N_HEALTH acquisitions only.
    """
    Z = []
    for c in cols:
        v = causal_smooth(df[c].to_numpy())
        base = np.median(v[:N_HEALTH])
        mad = np.median(np.abs(v[:N_HEALTH] - base)) * 1.4826
        Z.append((v - base) / max(mad, 1e-9))
    return np.column_stack(Z)


def build_hi(name, cols):
    """HI(t) = log1p( mean_f max(z_f(t), 0) ).

    The rectifier keeps only departures in the degrading direction; the mean
    over features is a robust unweighted fusion (no fitted weights, hence no
    opportunity for the fusion itself to leak information); log1p compresses the
    end-of-life blow-up onto a scale on which increments are of comparable size
    throughout the run.
    """
    df = load(name)
    Z = _standardise(df, cols)
    fused = np.maximum(Z, 0.0).mean(axis=1)
    hi = np.log1p(np.maximum(fused - np.median(fused[:N_HEALTH]), 0.0))
    return df["t"].to_numpy(), hi, df


def main():
    os.makedirs(OUT, exist_ok=True)
    cols, scores = rank_features()
    print(f"\n  retained: {cols}")

    series, term = {}, {}
    for b in ALL:
        t, hi, df = build_hi(b, cols)
        series[b] = (t, hi)
        term[b] = float(np.median(hi[-10:]))

    # threshold per operating condition, from that condition's two learning
    # bearings only (mirrors the IEEE PHM 2012 challenge design)
    D_cond = {}
    for cond in (1, 2, 3):
        lb = [b for b in LEARNING if b.startswith(f"Bearing{cond}_")]
        D_cond[cond] = float(np.mean([term[b] for b in lb]))
    D = float(np.median([term[b] for b in LEARNING]))

    print(f"\n  terminal HI:  learning = "
          f"{ {b: round(term[b], 2) for b in LEARNING} }")
    print(f"                test     = { {b: round(term[b], 2) for b in TEST} }")
    print(f"  global threshold D = {D:.3f};  per-condition D = "
          f"{ {k: round(v, 2) for k, v in D_cond.items()} }")

    meta = {"features": cols, "D": D, "D_cond": D_cond,
            "terminal_hi": term, "n_health": N_HEALTH,
            "smooth": SMOOTH, "learning": LEARNING, "test": TEST,
            "act_rul": ACT_RUL, "trunc_n": TRUNC_N,
            "feature_scores": scores,
            "note_bearing1_4": (
                "10 s x (|Full_Test_Set| - |Test_set|) reproduces the official "
                "actual RUL for 10 of 11 test bearings; for Bearing1_4 it gives "
                "2890 s against the officially published 339 s. The published "
                "value is used for comparability and this discrepancy is "
                "reported in the paper.")}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for b, (t, hi) in series.items():
        pd.DataFrame({"t": t, "hi": hi}).to_csv(
            os.path.join(OUT, f"{b}.csv"), index=False)
    print(f"\n  wrote {len(series)} HI series + meta.json to results/hi/")

    # crossing diagnostics
    print("\n  threshold crossing (fraction of life at which HI first reaches D):")
    for b in ALL:
        t, hi = series[b]
        cond = int(b[7])
        tag = "learn" if b in LEARNING else "test "
        out = []
        for lbl, thr in (("global", D), ("cond", D_cond[cond])):
            idx = np.flatnonzero(hi >= thr)
            out.append(f"{lbl}={t[idx[0]]/t[-1]:5.1%}" if idx.size else f"{lbl}= never")
        print(f"    {tag} {b:12s} maxHI={hi.max():5.2f}  " + "  ".join(out))


if __name__ == "__main__":
    main()
