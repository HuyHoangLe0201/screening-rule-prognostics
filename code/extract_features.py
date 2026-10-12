"""
Feature extraction for the PRONOSTIA / IEEE PHM 2012 bearing dataset.

Each acquisition file `acc_*.csv` holds 2560 samples (0.1 s at 25.6 kHz) recorded
every 10 s.  Columns: hour, minute, second, microsecond, horizontal acc [g],
vertical acc [g].

For every acquisition we compute a fixed bank of time- and frequency-domain
features on both channels and write one row per acquisition to
`results/features/<bearing>.csv`.

Usage:  python extract_features.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import signal, stats

#  Folder of the IEEE PHM 2012 challenge data (Learning_set/, Full_Test_Set/):
#  set PRONOSTIA_DIR, or pass it as the first argument.
ROOT = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("PRONOSTIA_DIR", "data/phm-ieee-2012-data-challenge-dataset-master"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "features")

FS = 25600.0          # sampling frequency [Hz]
DT_ACQ = 10.0         # seconds between successive acquisitions

BEARINGS = {
    # condition 1: 1800 rpm / 4000 N
    "Bearing1_1": ("Learning_set", 1), "Bearing1_2": ("Learning_set", 1),
    "Bearing1_3": ("Full_Test_Set", 1), "Bearing1_4": ("Full_Test_Set", 1),
    "Bearing1_5": ("Full_Test_Set", 1), "Bearing1_6": ("Full_Test_Set", 1),
    "Bearing1_7": ("Full_Test_Set", 1),
    # condition 2: 1650 rpm / 4200 N
    "Bearing2_1": ("Learning_set", 2), "Bearing2_2": ("Learning_set", 2),
    "Bearing2_3": ("Full_Test_Set", 2), "Bearing2_4": ("Full_Test_Set", 2),
    "Bearing2_5": ("Full_Test_Set", 2), "Bearing2_6": ("Full_Test_Set", 2),
    "Bearing2_7": ("Full_Test_Set", 2),
    # condition 3: 1500 rpm / 5000 N
    "Bearing3_1": ("Learning_set", 3), "Bearing3_2": ("Learning_set", 3),
    "Bearing3_3": ("Full_Test_Set", 3),
}

# ── feature bank ────────────────────────────────────────────────────────────
FEAT_NAMES = [
    "rms", "peak", "p2p", "kurtosis", "skewness", "crest", "shape", "impulse",
    "clearance", "energy_op", "sk_max", "spec_centroid", "spec_spread",
    "spec_entropy", "band_hi", "hilbert_rms",
]


def _channel_features(x: np.ndarray, sos_bp, sos_hi) -> np.ndarray:
    """16 features for one vibration channel."""
    x = x - x.mean()
    xb = signal.sosfilt(sos_bp, x)               # 100 Hz - 10 kHz band

    absx = np.abs(xb)
    rms = float(np.sqrt(np.mean(xb ** 2))) + 1e-12
    peak = float(absx.max())
    p2p = float(xb.max() - xb.min())
    mean_abs = float(absx.mean()) + 1e-12
    sqrt_mean = float(np.mean(np.sqrt(absx))) ** 2 + 1e-12

    kur = float(stats.kurtosis(xb, fisher=False))
    skw = float(stats.skew(xb))
    crest = peak / rms
    shape = rms / mean_abs
    impulse = peak / mean_abs
    clearance = peak / sqrt_mean

    # spectral quantities on the band-passed signal
    f, P = signal.welch(xb, fs=FS, nperseg=512, noverlap=256)
    Ptot = float(P.sum()) + 1e-20
    p = P / Ptot
    centroid = float((f * p).sum())
    spread = float(np.sqrt(((f - centroid) ** 2 * p).sum()))
    entropy = float(-(p * np.log(p + 1e-20)).sum() / np.log(len(p)))

    # operating-band energy and high-frequency (defect) band energy
    energy_op = float(np.log10(Ptot + 1e-20))
    band_hi = float(np.log10(P[f > 5000].sum() + 1e-20))

    # spectral kurtosis proxy: kurtosis of the high-pass envelope
    xh = signal.sosfilt(sos_hi, x)
    env = np.abs(signal.hilbert(xh))
    sk_max = float(stats.kurtosis(xh, fisher=False))
    hilbert_rms = float(np.sqrt(np.mean(env ** 2)))

    return np.array([rms, peak, p2p, kur, skw, crest, shape, impulse,
                     clearance, energy_op, sk_max, centroid, spread,
                     entropy, band_hi, hilbert_rms], dtype=np.float64)


def _process_one(args):
    path, sos_bp_c, sos_hi_c = args
    sos_bp = np.asarray(sos_bp_c)
    sos_hi = np.asarray(sos_hi_c)
    try:
        # a handful of folders (e.g. Bearing1_4) are semicolon-delimited
        with open(path, "r") as fh:
            probe = fh.readline()
        sep = ";" if probe.count(";") > probe.count(",") else ","
        arr = pd.read_csv(path, header=None, usecols=[4, 5], sep=sep,
                          dtype=np.float64, engine="c").to_numpy()
    except Exception:                                    # pragma: no cover
        return None
    if arr.shape[0] < 256:
        return None
    h = _channel_features(arr[:, 0], sos_bp, sos_hi)
    v = _channel_features(arr[:, 1], sos_bp, sos_hi)
    return np.concatenate([h, v])


def process_bearing(name: str, folder: str, workers: int = 8) -> pd.DataFrame:
    d = os.path.join(ROOT, folder, name)
    files = sorted(f for f in os.listdir(d) if f.startswith("acc_") and f.endswith(".csv"))
    sos_bp = signal.butter(4, [100.0, 10000.0], btype="bandpass", fs=FS, output="sos")
    sos_hi = signal.butter(4, 2000.0, btype="highpass", fs=FS, output="sos")

    tasks = [(os.path.join(d, f), sos_bp.tolist(), sos_hi.tolist()) for f in files]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_process_one, tasks, chunksize=16):
            rows.append(r)

    keep = [(i, r) for i, r in enumerate(rows) if r is not None]
    idx = [i for i, _ in keep]
    mat = np.vstack([r for _, r in keep])
    cols = [f"h_{c}" for c in FEAT_NAMES] + [f"v_{c}" for c in FEAT_NAMES]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "t", np.asarray(idx, dtype=float) * DT_ACQ)
    df.insert(0, "k", np.asarray(idx, dtype=int))
    return df


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    todo = sys.argv[1:] or list(BEARINGS)
    for name in todo:
        folder, cond = BEARINGS[name]
        out = os.path.join(OUT, f"{name}.csv")
        if os.path.exists(out):
            print(f"[skip] {name}")
            continue
        t0 = time.time()
        df = process_bearing(name, folder)
        df["condition"] = cond
        df.to_csv(out, index=False)
        print(f"[ok]   {name:<12} n={len(df):5d}  life={df['t'].iloc[-1]:8.0f}s  "
              f"({time.time() - t0:5.1f}s)", flush=True)


if __name__ == "__main__":
    main()
