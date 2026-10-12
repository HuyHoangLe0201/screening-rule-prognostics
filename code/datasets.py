"""
Unified health-indicator layer for the two bearing benchmarks.

HI(t) = log( RMS(t) / RMS_healthy ),  RMS = sqrt(RMS_h^2 + RMS_v^2)

Rationale.  A first-passage degradation model needs an indicator that measures
damage *severity* and grows with it.  Vibration RMS is the classical severity
proxy for rolling-element bearings; the logarithm both stabilises the terminal
blow-up and makes the indicator comparable across bearings and rigs.  Impulsive
features (kurtosis, crest, clearance) detect a defect earlier but peak at
initiation and recede as the spall spreads, so they are unsuitable as the state
of a monotone degradation process -- this is quantified in hi_select.py.

The healthy baseline uses only the first acquisitions of each bearing, so the
indicator is computable online.  The failure threshold is calibrated on training
bearings only and never on the bearing being predicted.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "pronostia": {
        "dir": os.path.join(HERE, "..", "results", "features"),
        "dt": 10.0, "smooth": 15, "n_health": 40,
        "learning": ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
                     "Bearing3_1", "Bearing3_2"],
        "test": ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6",
                 "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5",
                 "Bearing2_6", "Bearing2_7", "Bearing3_3"],
    },
    "xjtu": {
        "dir": os.path.join(HERE, "..", "results", "features_xjtu"),
        "dt": 60.0, "smooth": 9, "n_health": 10,
        "learning": [], "test": [],          # leave-one-out within condition
    },
}

# Official IEEE PHM 2012 actual RUL at the challenge truncation point [s].
# 10 s x (|Full_Test_Set| - |Test_set|) reproduces these for ten of eleven
# bearings; Bearing1_4 is the documented exception (2890 s from the archive
# against the published 339 s).
#
# NOTE: nothing scores against this table. Every pipeline computes the target
# as t_end - t_k straight from the record, so Bearing1_4 is scored at the
# archive-derived 2890 s, not at the 339 s below. The table is kept because it
# is the published reference, and because the difference is worth knowing: it
# moves the official-subset error by up to a fifth and changes no ordering in
# either paper. Do not read it as the value in use.
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


def load_dataset(name):
    """Return {bearing: dict(t, hi, condition, life)} for one benchmark."""
    cfg = DATASETS[name]
    out = {}
    for f in sorted(glob.glob(os.path.join(cfg["dir"], "*.csv"))):
        b = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        rms = np.sqrt(d["h_rms"].to_numpy() ** 2 + d["v_rms"].to_numpy() ** 2)
        rms = _smooth(rms, cfg["smooth"])
        nh = min(cfg["n_health"], max(5, int(0.05 * len(rms))))
        hi = np.log(rms / np.median(rms[:nh]))
        out[b] = {"t": d["t"].to_numpy(), "hi": hi,
                  "condition": int(d["condition"].iloc[0]),
                  "life": float(d["t"].iloc[-1]), "n": len(hi), "dt": cfg["dt"]}
    return out


def terminal_hi(rec, k=3):
    return float(np.median(rec["hi"][-k:]))


def first_prediction_time(hi, n_health=40, thr=3.0, persist=10):
    """Index of the detected onset of degradation (first prediction time, FPT).

    The first index after which the indicator stays `thr` robust standard
    deviations above its healthy median for `persist` consecutive acquisitions.
    A degradation process fitted through the healthy plateau spends its
    likelihood explaining measurement noise and biases the growth parameters
    towards zero, so the filter is started at the FPT instead.  The rule is
    causal only when `hi` stops at the prediction time, which is how the path
    model calls it (study_bearings._filter passes hi[:k+1]).  Called on a
    complete record it is retrospective: the 80 % sustained-departure test reads
    the rest of the record and the healthy window n // 10 depends on its total
    length.  The applicability screen uses it that way, offline, on completed
    run-to-failure records.  Returns 0 if no onset is detected, so the model
    still sees the whole record.
    """
    hi = np.asarray(hi, dtype=float)
    n = len(hi)
    nh = min(n_health, max(5, n // 10))
    base = np.median(hi[:nh])
    mad = np.median(np.abs(hi[:nh] - base)) * 1.4826
    above = hi > base + thr * max(mad, 1e-9)
    # Require the departure to be SUSTAINED: from the onset onwards at least
    # 80 % of the remaining acquisitions must stay above the detection band, and
    # the first `persist` of them must all be above it.  A plain run-length rule
    # fires on the first noise excursion, which on records that stay flat for
    # thousands of acquisitions puts the onset near t=0 and leaves the
    # degradation model fitting pure measurement noise.
    frac_above = np.cumsum(above[::-1])[::-1] / np.maximum(np.arange(n, 0, -1), 1)
    for i in range(n - persist):
        if above[i:i + persist].all() and frac_above[i] >= 0.8:
            return i
    return 0


def threshold_for(data, bearing, name):
    """Failure threshold for one bearing, calibrated WITHOUT that bearing.

    pronostia -- median terminal HI of the six learning bearings (fixed split).
    xjtu      -- median terminal HI of the other bearings of the same operating
                 condition (leave-one-out), since XJTU-SY defines no split.
    """
    if name == "pronostia":
        ref = DATASETS["pronostia"]["learning"]
    else:
        c = data[bearing]["condition"]
        ref = [b for b, r in data.items() if r["condition"] == c and b != bearing]
    return float(np.median([terminal_hi(data[b]) for b in ref])), ref


def evaluation_bearings(data, name):
    """Bearings to report on, and their prediction points (index into the series).

    For PRONOSTIA the official IEEE PHM 2012 truncation point is always included
    (so the headline comparison is the standard one) together with a sweep of
    life fractions, which is needed to see how accuracy depends on how much
    degradation has been observed. Because the Full_Test_Set records run to
    failure, the actual RUL at any index is t_end - t_k.
    """
    fracs = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    out = []
    for b, r in sorted(data.items()):
        if name == "pronostia":
            if b not in DATASETS["pronostia"]["test"]:
                continue
            pts = {PHM12_TRUNC_N[b] - 1}
        else:
            pts = set()
        pts |= {int(f * (r["n"] - 1)) for f in fracs}
        out.append((b, sorted(pts)))
    return out


def official_point(name, bearing, k):
    return name == "pronostia" and k == PHM12_TRUNC_N.get(bearing, -99) - 1


def phm_score(er_pct):
    """IEEE PHM 2012 asymmetric score; er_pct = 100*(ActRUL-PredRUL)/ActRUL."""
    e = np.asarray(er_pct, dtype=float)
    return np.where(e <= 0, np.exp(-np.log(0.5) * e / 5.0),
                    np.exp(np.log(0.5) * e / 20.0))
