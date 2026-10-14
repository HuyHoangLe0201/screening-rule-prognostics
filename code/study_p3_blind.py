# -*- coding: utf-8 -*-
r"""The pre-registered blind test of PREREG_p3_blind.md, on the N-CMAPSS subsets
that no earlier analysis opened.

    python code/study_p3_blind.py extract   stream each subset out of the archive,
                                            reduce it to per-unit indicators, delete it
    python code/study_p3_blind.py rul       onsets, screen and remaining-life predictions
    python code/study_p3_blind.py summary   B1-B4

Every constant is the protocol's; the script refuses to run on any other
version of it.  The archive holds a zip inside a zip, too large to unpack on
this machine, so the inner zip is read as a stream in a single pass: each wanted
member is written out, reduced, and deleted before the stream moves on.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zipfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_bearings as sb                                      # noqa: E402
import study_p3_prereg as P                                      # noqa: E402
import study_p3_review as R                                      # noqa: E402

PREREG = os.path.join(HERE, "PREREG_p3_blind.md")
FROZEN_SHA256 = "f85f510eab49d57bb32119e3e34b5e3513aa5b6459e6752bc1a70d4c55206a3f"
FROZEN_AT = "2026-10-14T23:07:43+07:00"
NCM_DIR = os.path.join(P.DATA, "ncmapss")
OUTER = os.path.join(NCM_DIR, "ncmapss.zip")
INNER = "17. Turbofan Engine Degradation Simulation Data Set 2/data_set.zip"
BE = os.path.join(ROOT, "results", "bearings")
REDUCED = os.path.join(BE, "p3_blind_indicators.json")
ROWS = os.path.join(BE, "p3_blind_rows.json")
OUT = os.path.join(BE, "p3_blind.json")
POINTS = (0.50, 0.70, 0.85)
B_BOOT, SEED = 10_000, 20261018


def check_frozen():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {h}")
    return h


def wanted(name):
    base = os.path.basename(name)
    return (base.endswith(".h5") and not base.startswith("N-CMAPSS_DS01")
            and not base.startswith("N-CMAPSS_DS02"))


# ══════════════════════════════════════════════════════════════════════════
# extract: a single streaming pass over the inner zip
# ══════════════════════════════════════════════════════════════════════════
def _stream_members(fobj):
    """Yield (name, write_to) for each member, writing wanted ones to disk.

    A minimal sequential zip reader: local headers, stored or deflated data,
    data descriptors and zip64 sizes.  The caller processes and deletes a
    written member before asking for the next."""
    buf = bytearray()
    CH = 1 << 24

    def need(n):
        while len(buf) < n:
            chunk = fobj.read(CH)
            if not chunk:
                raise EOFError("archive ended inside a header")
            buf.extend(chunk)

    while True:
        need(4)
        if bytes(buf[:4]) != b"PK\x03\x04":
            return                                   # central directory: done
        need(30)
        (_, _, flag, method, _, _, _, csize, usize, nlen,
         xlen) = struct.unpack("<IHHHHHIIIHH", bytes(buf[:30]))
        need(30 + nlen + xlen)
        name = bytes(buf[30:30 + nlen]).decode("utf-8", "replace")
        extra = bytes(buf[30 + nlen:30 + nlen + xlen])
        del buf[:30 + nlen + xlen]
        i = 0
        while i + 4 <= len(extra):                   # zip64 sizes, if present
            hid, hsz = struct.unpack("<HH", extra[i:i + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hsz // 8), extra[i + 4:i + 4 + 8 * (hsz // 8)]))
                if usize == 0xFFFFFFFF and vals:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF and vals:
                    csize = vals.pop(0)
            i += 4 + hsz
        dest = os.path.join(NCM_DIR, os.path.basename(name)) if wanted(name) else None
        out = open(dest, "wb") if dest else None
        if method == 8:
            d = zlib.decompressobj(-15)
            while True:
                if not buf:
                    need(1)
                data = d.decompress(bytes(buf))
                if out is not None and data:
                    out.write(data)
                if d.eof:
                    buf[:] = d.unused_data
                    break
                buf.clear()
        elif method == 0:
            if flag & 8 and csize == 0:
                raise ValueError(f"stored member with unknown size: {name}")
            left = csize
            while left:
                need(1)
                take = min(left, len(buf))
                if out is not None:
                    out.write(bytes(buf[:take]))
                del buf[:take]
                left -= take
        else:
            raise ValueError(f"compression method {method} in {name}")
        if out is not None:
            out.close()
        if flag & 8:                                 # data descriptor
            need(24)
            if bytes(buf[:4]) == b"PK\x07\x08":
                nxt = bytes(buf[16:20])
                skip = 16 if nxt in (b"PK\x03\x04", b"PK\x01\x02") else 24
            else:
                skip = 12
            del buf[:skip]
        print(f"  {'kept' if dest else 'skipped'} {name} ({usize / 1e6:.0f} MB)", flush=True)
        yield name, dest


def _reduce(path):
    """Per-unit indicators of one subset, exactly as run_ncmapss computes them."""
    import h5py
    tag = os.path.basename(path)[:-3]
    rows = []
    for split in ("dev", "test"):
        with h5py.File(path, "r") as f:
            A = f[f"A_{split}"][:]
            W = f[f"W_{split}"][:]
            X = f[f"X_s_{split}"][:]
        for u in np.unique(A[:, 0]).astype(int):
            sel = A[:, 0] == u
            cyc, hs = A[sel, 1].astype(int), A[sel, 3]
            w, x = W[sel], X[sel]
            cycles = np.unique(cyc)
            n = len(cycles)
            nh = min(40, max(5, n // 10))
            healthy = cyc <= cycles[nh - 1]
            mu, sd = w[healthy].mean(axis=0), w[healthy].std(axis=0) + 1e-12
            Phi = P._quadratic((w - mu) / sd)
            beta, *_ = np.linalg.lstsq(Phi[healthy], x[healthy], rcond=None)
            res = x - Phi @ beta
            pos = np.searchsorted(cycles, cyc)
            cnt = np.bincount(pos, minlength=n)
            M = np.column_stack([np.bincount(pos, weights=res[:, j], minlength=n) / cnt
                                 for j in range(res.shape[1])])
            hi = P.NS["indicator"](M)
            hs_c = np.array([hs[pos == k].min() for k in range(n)])
            lab = int(np.argmax(hs_c == 0)) if (hs_c == 0).any() else None
            rows.append({"file": tag, "split": split, "unit": int(u), "n": int(n),
                         "hi": [float(v) for v in hi], "label_idx": lab})
    return rows


def extract():
    check_frozen()
    done = json.load(open(REDUCED)) if os.path.exists(REDUCED) else {}
    with zipfile.ZipFile(OUTER) as z, z.open(INNER) as inner:
        for name, dest in _stream_members(inner):
            if dest is None:
                continue
            tag = os.path.basename(dest)[:-3]
            if tag not in done:
                done[tag] = _reduce(dest)
                json.dump(done, open(REDUCED, "w"))
                print(f"  reduced {tag}: {len(done[tag])} units", flush=True)
            os.remove(dest)


def extract_member(base):
    """Re-extract one member through zipfile's seeking reader, CRC-checked.

    Slower than the stream (each backward seek re-inflates the outer member
    from its start), and used only for a member the stream wrote short."""
    check_frozen()
    done = json.load(open(REDUCED)) if os.path.exists(REDUCED) else {}
    dest = os.path.join(NCM_DIR, base)
    if os.path.exists(dest):
        os.remove(dest)                      # the short copy this script wrote
    with zipfile.ZipFile(OUTER) as z, z.open(INNER) as inner:
        zz = zipfile.ZipFile(inner)
        info = next(i for i in zz.infolist() if os.path.basename(i.filename) == base)
        print(f"  {info.filename}: {info.file_size} bytes, method {info.compress_type}",
              flush=True)
        with zz.open(info) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
    tag = base[:-3]
    done[tag] = _reduce(dest)
    json.dump(done, open(REDUCED, "w"))
    print(f"  reduced {tag}: {len(done[tag])} units", flush=True)
    os.remove(dest)


# ══════════════════════════════════════════════════════════════════════════
# rul: onsets, the screen, and the predictions
# ══════════════════════════════════════════════════════════════════════════
def stable_seed(*key):
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def _screen(fracs, rng):
    x = np.array(fracs, float)                     # 1.0 = censored at end of life
    x = np.where(x >= 1.0, np.inf, x)
    rate = float(np.isfinite(x).mean())
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    lo, hi = np.quantile(np.median(x[idx], axis=1), [0.025, 0.975], method="inverted_cdf")

    def pct(z):
        return None if not np.isfinite(z) else 100 * float(z)
    verdict = ("check indicator" if rate < 0.25 else
               R._verdict3(pct(lo), pct(hi), 70.0))
    return {"n": len(x), "rate": rate, "censored_median_pct": pct(np.median(x)),
            "ci": [pct(lo), pct(hi)], "verdict": verdict}


def rul():
    check_frozen()
    data = json.load(open(REDUCED))
    rng = np.random.default_rng(SEED)
    screen, rows = {}, []
    for tag in sorted(data):
        units = data[tag]
        for u in units:
            hi = np.asarray(u["hi"])
            i = P.NS["onset"](hi)
            u["onset_idx"] = int(i) if i > 0 else u["n"]
            u["onset_frac"] = i / u["n"] if i > 0 else 1.0
        screen[tag] = _screen([u["onset_frac"] for u in units], rng)
        print(f"  {tag}: {len(units)} units, verdict {screen[tag]['verdict']} "
              f"{screen[tag]['censored_median_pct']} {screen[tag]['ci']}", flush=True)
        for j, u in enumerate(units):
            key = f"{u['file']}-{u['split']}-{u['unit']}"
            ref = [v for k, v in enumerate(units) if k != j]
            D = float(np.median([np.median(v["hi"][-3:]) for v in ref]))
            med_life = float(np.median([max(v["n"] - v["onset_idx"], 20) for v in ref]))
            lives = [v["n"] - 1 for v in ref]
            n = u["n"]
            rec = {"t": np.arange(n, dtype=float), "hi": np.asarray(u["hi"]),
                   "condition": 0, "life": float(n - 1), "n": n, "dt": 1.0}
            for f in POINTS:
                k = int(f * n)
                if k < 20 or k >= n - 1:
                    continue
                true = float((n - 1) - k)
                r = sb.predict_one(rec, k, D, "BS", informed=True, coupling=True,
                                   med_life_steps=med_life, seed=stable_seed(key, k),
                                   n_part=600)
                lt = R._life_table(lives, float(k))
                row = {"unit": key, "cohort": tag, "n": n, "k": k, "at_life": f,
                       "onset_frac": u["onset_frac"], "true": true,
                       "lt_pred": lt[0], "lt_lo": lt[1], "lt_hi": lt[2]}
                if r is not None:
                    row.update({"pred": r["pred"], "lo": r["lo"], "hi": r["hi"],
                                "capped": r["capped"]})
                rows.append(row)
        print(f"    {tag}: {len(rows)} predictions so far", flush=True)
    json.dump({"screen": screen, "rows": rows,
               "units": {t: [{k: u[k] for k in ("file", "split", "unit", "n", "onset_idx",
                                                 "onset_frac", "label_idx")} for u in data[t]]
                         for t in data}},
              open(ROWS, "w"), indent=1)


# ══════════════════════════════════════════════════════════════════════════
# summary: B1-B4
# ══════════════════════════════════════════════════════════════════════════
def _ape(r, key="pred"):
    return 100.0 * abs(r[key] - r["true"]) / r["true"]


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def summary():
    h = check_frozen()
    S = json.load(open(ROWS))
    screen, rows = S["screen"], [r for r in S["rows"] if "pred" in r]
    rng = np.random.default_rng(SEED)
    by_unit = {}
    for r in rows:
        by_unit.setdefault(r["unit"], []).append(r)
    units = sorted(by_unit)
    on = np.array([by_unit[u][0]["onset_frac"] for u in units])
    err = np.array([np.mean([_ape(r) for r in by_unit[u]]) for u in units])
    res = {"protocol_sha256": h, "protocol_frozen_at": FROZEN_AT, "screen": screen,
           "n_units": len(units), "n_predictions": len(rows),
           "deviations": ["N-CMAPSS_DS08d-010.h5 excluded: the archive's copy is "
                          "2885034848 bytes, 32 short of the length its HDF5 superblock "
                          "records, and the file cannot be opened.  Excluded before any "
                          "of its data was read; the CRC-checked re-extraction gives the "
                          "same length."]}

    # B1
    res["B1"] = {"holds": all(s["verdict"] != "declines" for s in screen.values())}
    # B2
    rho = _spearman(on, err)
    bs = []
    for _ in range(B_BOOT):
        i = rng.integers(0, len(units), len(units))
        if np.ptp(on[i]) > 0:
            bs.append(_spearman(on[i], err[i]))
    ci = np.percentile(bs, [2.5, 97.5]).tolist()
    res["B2"] = {"rho": rho, "ci": ci, "holds": bool(ci[0] > 0)}
    # B3
    late = [u for u in units if by_unit[u][0]["onset_frac"] >= 0.70]
    early = [u for u in units if by_unit[u][0]["onset_frac"] < 0.70]
    if len(late) >= 5 and len(early) >= 5:
        def d(L, E):
            return (np.median([_ape(r) for u in L for r in by_unit[u]])
                    - np.median([_ape(r) for u in E for r in by_unit[u]]))
        obs = d(late, early)
        bt = [d([late[i] for i in rng.integers(0, len(late), len(late))],
                [early[i] for i in rng.integers(0, len(early), len(early))])
              for _ in range(B_BOOT)]
        c3 = np.percentile(bt, [2.5, 97.5]).tolist()
        res["B3"] = {"testable": True, "n_late": len(late), "n_early": len(early),
                     "diff": float(obs), "ci": c3, "holds": bool(c3[0] > 0)}
    else:
        res["B3"] = {"testable": False, "n_late": len(late), "n_early": len(early)}
    # B4
    dec = [c for c, s in screen.items() if s["verdict"] == "declines"]
    nd = [c for c, s in screen.items() if s["verdict"] == "not declined"]
    if dec and nd:
        Ud = [u for u in units if by_unit[u][0]["cohort"] in dec]
        Un = [u for u in units if by_unit[u][0]["cohort"] in nd]
        Ua = Ud + Un

        def stats(ud, un):
            pd = [_ape(r) for u in ud for r in by_unit[u]]
            pn = [_ape(r) for u in un for r in by_unit[u]]
            follow = [_ape(r, "lt_pred") for u in ud for r in by_unit[u]] + pn
            return np.median(pd) - np.median(pn), np.median(pd + pn) - np.median(follow)
        obs = stats(Ud, Un)
        bt = np.array([stats([Ud[i] for i in rng.integers(0, len(Ud), len(Ud))],
                             [Un[i] for i in rng.integers(0, len(Un), len(Un))])
                       for _ in range(B_BOOT)])
        c_a = np.percentile(bt[:, 0], [2.5, 97.5]).tolist()
        c_b = np.percentile(bt[:, 1], [2.5, 97.5]).tolist()
        res["B4"] = {"testable": True, "declined": dec, "not_declined": nd,
                     "diff_declined_minus_passed": float(obs[0]), "ci_a": c_a,
                     "gain_of_following_screen": float(obs[1]), "ci_b": c_b,
                     "holds": bool(c_a[0] > 0 and c_b[0] > 0), "n_units": len(Ua)}
    else:
        res["B4"] = {"testable": False, "declined": dec, "not_declined": nd}
    res["pooled"] = {
        "medape_path": float(np.median([_ape(r) for r in rows])),
        "coverage_path": float(np.mean([r["lo"] <= r["true"] <= r["hi"] for r in rows])),
        "medape_lt": float(np.median([_ape(r, "lt_pred") for r in rows])),
        "coverage_lt": float(np.mean([r["lt_lo"] <= r["true"] <= r["lt_hi"] for r in rows]))}
    res["by_cohort"] = {}
    for c in sorted(screen):
        rc = [r for r in rows if r["cohort"] == c]
        res["by_cohort"][c] = {
            **screen[c], "units": len({r["unit"] for r in rc}), "predictions": len(rc),
            "medape_path": float(np.median([_ape(r) for r in rc])) if rc else None,
            "coverage_path": float(np.mean([r["lo"] <= r["true"] <= r["hi"] for r in rc])) if rc else None,
            "medape_lt": float(np.median([_ape(r, "lt_pred") for r in rc])) if rc else None,
            "coverage_lt": float(np.mean([r["lt_lo"] <= r["true"] <= r["lt_hi"] for r in rc])) if rc else None}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for c, v in res["by_cohort"].items():
        print(f"{c:22s} {v['units']:3d} units  onset {v['censored_median_pct']} {v['ci']}  "
              f"{v['verdict']:13s} path {v['medape_path']:.1f}/{v['coverage_path']:.2f}  "
              f"LT {v['medape_lt']:.1f}/{v['coverage_lt']:.2f}")
    for b in ("B1", "B2", "B3", "B4"):
        print(b, res[b])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "extract_member":
        extract_member(sys.argv[2])
    else:
        {"extract": extract, "rul": rul, "summary": summary}.get(
            step, lambda: sys.exit(__doc__))()
