"""
Feature extraction for the XJTU-SY run-to-failure bearing dataset.

15 bearings, 3 operating conditions.  Each acquisition file holds 32768 samples
(1.28 s at 25.6 kHz) recorded once per minute, with a two-column header
(horizontal, vertical vibration).  The same feature bank as the PRONOSTIA
pipeline is used so that the two datasets are directly comparable.
"""
from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import signal

from extract_features import FEAT_NAMES, _channel_features

#  Folder holding the XJTU-SY condition folders (35Hz12kN/, 37.5Hz11kN/, 40Hz10kN/):
#  set XJTU_DIR, or pass it as the first argument.
ROOT = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("XJTU_DIR", "data/XJTU-SY_Bearing_Datasets"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "features_xjtu")

FS = 25600.0
DT_ACQ = 60.0        # one acquisition per minute

CONDITIONS = {"35Hz12kN": 1, "37.5Hz11kN": 2, "40Hz10kN": 3}


def _one(args):
    path, sos_bp_c, sos_hi_c = args
    sos_bp, sos_hi = np.asarray(sos_bp_c), np.asarray(sos_hi_c)
    try:
        arr = pd.read_csv(path, dtype=np.float64, engine="c").to_numpy()
    except Exception:                                        # pragma: no cover
        return None
    if arr.shape[0] < 256 or arr.shape[1] < 2:
        return None
    return np.concatenate([_channel_features(arr[:, 0], sos_bp, sos_hi),
                           _channel_features(arr[:, 1], sos_bp, sos_hi)])


def process(cond_dir, bearing, workers=8):
    d = os.path.join(ROOT, cond_dir, bearing)
    files = [f for f in os.listdir(d) if f.endswith(".csv")]
    files.sort(key=lambda f: int(re.match(r"(\d+)", f).group(1)))
    sos_bp = signal.butter(4, [100.0, 10000.0], btype="bandpass", fs=FS, output="sos")
    sos_hi = signal.butter(4, 2000.0, btype="highpass", fs=FS, output="sos")
    tasks = [(os.path.join(d, f), sos_bp.tolist(), sos_hi.tolist()) for f in files]

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_one, tasks, chunksize=4):
            rows.append(r)
    keep = [(i, r) for i, r in enumerate(rows) if r is not None]
    cols = [f"h_{c}" for c in FEAT_NAMES] + [f"v_{c}" for c in FEAT_NAMES]
    df = pd.DataFrame(np.vstack([r for _, r in keep]), columns=cols)
    df.insert(0, "t", np.array([i for i, _ in keep], float) * DT_ACQ)
    df.insert(0, "k", np.array([i for i, _ in keep], int))
    df["condition"] = CONDITIONS[cond_dir]
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    want = set(sys.argv[1:])
    for cond_dir in CONDITIONS:
        for bearing in sorted(os.listdir(os.path.join(ROOT, cond_dir))):
            if not os.path.isdir(os.path.join(ROOT, cond_dir, bearing)):
                continue
            if want and bearing not in want:
                continue
            out = os.path.join(OUT, f"{bearing}.csv")
            if os.path.exists(out):
                print(f"[skip] {bearing}")
                continue
            t0 = time.time()
            df = process(cond_dir, bearing)
            df.to_csv(out, index=False)
            print(f"[ok]   {bearing:<12s} cond={CONDITIONS[cond_dir]} n={len(df):5d} "
                  f"life={df['t'].iloc[-1]/60:6.0f} min  ({time.time()-t0:6.1f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
