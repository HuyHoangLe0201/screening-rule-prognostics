"""
Supervised dataset for direct RUL regression.

At every inspection index k a feature vector is built from acquisitions
1..k only, and paired with the target RUL = t_end - t_k. Nothing derived from
the end of a record enters that record's own features, and no feature depends on
the record's total length -- which would leak the answer.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

FEATDIR = {"pronostia": os.path.join(HERE, "..", "results", "features"),
           "xjtu": os.path.join(HERE, "..", "results", "features_xjtu")}
DT = {"pronostia": 10.0, "xjtu": 60.0}
SMOOTH = {"pronostia": 15, "xjtu": 9}
N_HEALTH = {"pronostia": 40, "xjtu": 10}
#  GBS_LEGACY_HEALTH=1 reproduces the first-round protocol, whose healthy
#  window depended on record length and whose training began inside it.
LEGACY_HEALTH = os.environ.get("GBS_LEGACY_HEALTH", "0") == "1"

# Physically motivated feature subset, fixed a priori. Amplitude/energy features
# track damage severity; impulsive features respond earliest to defect
# initiation; spectral features track the shift of energy into the defect band.
BASE = ["h_rms", "v_rms", "h_peak", "h_kurtosis", "v_kurtosis", "h_crest",
        "h_impulse", "h_hilbert_rms", "v_hilbert_rms", "h_band_hi",
        "h_energy_op", "h_spec_centroid"]

PRONOSTIA_LEARN = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
                   "Bearing3_1", "Bearing3_2"]
PRONOSTIA_TEST = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6",
                  "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5",
                  "Bearing2_6", "Bearing2_7", "Bearing3_3"]
# Published reference only: the regression target is t_end - t below, so this
# table is not what anything is scored against. See datasets.py for the one
# entry where the two disagree.
PHM12_ACT_RUL = {"Bearing1_3": 5730, "Bearing1_4": 339, "Bearing1_5": 1610,
                 "Bearing1_6": 1460, "Bearing1_7": 7570, "Bearing2_3": 7530,
                 "Bearing2_4": 1390, "Bearing2_5": 3090, "Bearing2_6": 1290,
                 "Bearing2_7": 580, "Bearing3_3": 820}
PHM12_TRUNC_N = {"Bearing1_3": 1802, "Bearing1_4": 1139, "Bearing1_5": 2302,
                 "Bearing1_6": 2302, "Bearing1_7": 1502, "Bearing2_3": 1202,
                 "Bearing2_4": 612, "Bearing2_5": 2002, "Bearing2_6": 572,
                 "Bearing2_7": 172, "Bearing3_3": 352}


def _smooth(v, w):
    s = pd.Series(v).rolling(w, min_periods=1).median()
    return s.rolling(w, min_periods=1).mean().to_numpy()


def _causal_slope(v, w):
    """Slope of a trailing window of length w, at every index (uses past only)."""
    n = len(v)
    out = np.zeros(n)
    x = np.arange(w, dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    for k in range(n):
        lo = max(0, k - w + 1)
        seg = v[lo:k + 1]
        if len(seg) < 3:
            continue
        xx = np.arange(len(seg), dtype=float)
        xx -= xx.mean()
        d = (xx ** 2).sum()
        out[k] = float((xx * (seg - seg.mean())).sum() / d) if d > 0 else 0.0
    return out


FEATNAMES = ([f"z_{c}" for c in BASE] + [f"dz_{c}" for c in BASE]
             + ["hi", "hi_cummax", "hi_mean20", "hi_slope50", "hi_slope200",
                "t_elapsed", "frac_alarm", "log1p_t"])


def build_bearing(name, dataset, t_scale, n_max=None):
    """Return (features (n,d), rul (n,), t (n,), n) for one bearing."""
    df = pd.read_csv(os.path.join(FEATDIR[dataset], f"{name}.csv"))
    if n_max is not None:          # truncated record, for the causality check only
        df = df.iloc[:n_max]
    dt, w = DT[dataset], SMOOTH[dataset]
    n = len(df)
    #  The healthy window is a fixed number of acquisitions. The legacy rule
    #  shrank it to n // 10 on short records, which made every feature of
    #  those records depend on their total length -- information from the
    #  future (Bearing2_7 on PRONOSTIA; Bearing1_5 and Bearing2_4 on XJTU-SY).
    nh = min(N_HEALTH[dataset], max(5, n // 10)) if LEGACY_HEALTH else N_HEALTH[dataset]

    cols = []
    for c in BASE:
        v = _smooth(df[c].to_numpy(), w)
        base = np.median(v[:nh])
        mad = np.median(np.abs(v[:nh] - base)) * 1.4826
        cols.append((v - base) / max(mad, 1e-9))
    Z = np.column_stack(cols)                                    # (n, |BASE|)
    dZ = np.column_stack([np.concatenate([np.zeros(10), Z[10:, j] - Z[:-10, j]])
                          for j in range(Z.shape[1])])

    rms = _smooth(np.sqrt(df["h_rms"].to_numpy() ** 2 + df["v_rms"].to_numpy() ** 2), w)
    hi = np.log(rms / np.median(rms[:nh]))
    hi_cummax = np.maximum.accumulate(hi)
    hi_m20 = pd.Series(hi).rolling(20, min_periods=1).mean().to_numpy()
    sl50 = _causal_slope(hi, 50)
    sl200 = _causal_slope(hi, 200)
    z_rms = Z[:, BASE.index("h_rms")]
    frac_alarm = np.cumsum(z_rms > 3.0) / np.arange(1, n + 1)

    t = np.arange(n, dtype=float) * dt
    X = np.column_stack([Z, dZ, hi, hi_cummax, hi_m20, sl50, sl200,
                         t / t_scale, frac_alarm, np.log1p(t / t_scale)])
    rul = t[-1] - t
    return X, rul, t, n, nh


def load(dataset, t_scale=None):
    """Load every bearing of a dataset. `t_scale` normalises elapsed time and
    must come from TRAINING records only."""
    names = sorted(f[:-4] for f in os.listdir(FEATDIR[dataset]) if f.endswith(".csv"))
    if t_scale is None:
        t_scale = 1.0
    out = {}
    for b in names:
        X, rul, t, n, nh = build_bearing(b, dataset, t_scale)
        out[b] = {"X": X, "rul": rul, "t": t, "n": n, "nh": nh}
    return out


def stack(data, names, warmup=0.05, stride=1, normalised=True, target="fraction"):
    """Concatenate training bearings, weighting each record equally.

    With `normalised=True` the target is the fraction of life remaining,
    RUL_k / T_total in (0,1]. This is the protocol under which the PRONOSTIA
    literature reports MAE/RMSE, and it is the only one under which the problem
    is well posed here: bearing lifetimes on this rig span a factor of twelve
    (2290 s to 28020 s) and no causal feature identifies which regime a bearing
    is in, so absolute-seconds RUL is not determined by the observable state.
    All support-(0,inf) heads apply unchanged; the data simply concentrate below
    one. The relative error Er% -- and hence the IEEE PHM 2012 score -- is
    invariant to this rescaling, so those metrics remain directly comparable
    with the challenge protocol.
    """
    Xs, ys, ws, bs = [], [], [], []
    for b in names:
        d = data[b]
        #  Training starts after the healthy window: a point inside it would be
        #  standardised by a baseline that extends past it.
        k0 = max(20, int(warmup * d["n"])) if LEGACY_HEALTH else max(20, d["nh"], int(warmup * d["n"]))
        idx = np.arange(k0, d["n"] - 1, stride)
        if len(idx) == 0:
            continue
        y = d["rul"][idx]
        if target == "ratio":
            #  remaining-to-elapsed ratio w = R/t, on (0, inf): known t, no bound
            y = y / np.maximum(d["t"][idx], 1e-9)
        elif normalised:
            y = y / max(d["t"][-1], 1e-9)
        Xs.append(d["X"][idx])
        ys.append(y)
        ws.append(np.full(len(idx), 1.0 / len(idx)))
        bs += [b] * len(idx)
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.array(bs))


def standardise(Xtr, *others):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others] + [(mu, sd)]


def phm_score(er_pct):
    e = np.asarray(er_pct, dtype=float)
    return np.where(e <= 0, np.exp(-np.log(0.5) * e / 5.0),
                    np.exp(np.log(0.5) * e / 20.0))
