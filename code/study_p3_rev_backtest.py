"""Paper 3 revision: the screen against a direct back-test, and warning time.

Review issues B and C (submission/paper3_simulated_review_20261114.md).
Reads stored per-point predictions only; no path model is refitted except one
timed fit used to cost a back-test.

Part 1  screen vs back-test vs life table, by subsampling n records per cohort
        on the 14 cohorts of Table 15a (p3_review.json["decisions"]).
Part 2a onset-to-failure lead time per record, and the share of records whose
        lead time exceeds illustrative response times.
Part 2b one-sided miss rates P(true < lo_2.5%) and early-call rates
        P(true > hi_97.5%) of path model and life table, by screen verdict.

Sources
  results_both.json   PRONOSTIA / XJTU-SY path-model rows (law BS, the
                      'process' table: first 77 / 89 BS rows without variant)
  p3_rul.json         XJTU cell rows (path + life table, 2.5/97.5 % quantiles)
  p3_blind_rows.json  blind N-CMAPSS rows (path + life table)
  life table for the two bearing archives rebuilt with study_p3_review's
  _life_table on the same reference records as part_reg (checked against the
  stored MedAPE).

    python code/study_p3_rev_backtest.py   ->  results/bearings/p3_rev_backtest.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import datasets as ds                                             # noqa: E402
import study_bearings as sb                                       # noqa: E402
import study_p3_review as R                                       # noqa: E402

BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_rev_backtest.json")
N_SUB = 1000          # subsamples per cohort and n
N_BOOT = 1000         # bootstrap draws inside each screen (as part_subsample)
THR = 70.0
NS = (3, 5, 10)


def load(f):
    return json.load(open(os.path.join(BE, f), encoding="utf-8"))


def ape(p, t):
    return 100.0 * abs(p - t) / t


# ─────────────────────────────────────────────────────────────────────────
# cohorts: {name: {"records": {rec: {"onset": frac|None, "life": L,
#                                    "pts": [dict(true, k, path, plo, phi, lt, llo, lhi)]}}}}
# ─────────────────────────────────────────────────────────────────────────
def bearing_cohorts():
    rows = load("results_both.json")["rows"]
    app = load("applicability.json")
    out = {}
    for name, label, npts in (("pronostia", "PRONOSTIA", 77), ("xjtu", "XJTU-SY", 89)):
        bs = [r for r in rows if r["dataset"] == name and r["law"] == "BS"
              and "variant" not in r][:npts]
        data = ds.load_dataset(name)
        lives = {b: float(r["t"][-1]) for b, r in data.items()}
        onset = {r["bearing"]: (r["fpt_frac"] if r["fpt_idx"] > 0 else None)
                 for r in app if r["dataset"] == name}
        recs = {}
        for r in bs:
            b = r["bearing"]
            _, ref = ds.threshold_for(data, b, name)
            lt = R._life_table([lives[x] for x in ref], float(data[b]["t"][r["k"]]))
            recs.setdefault(b, {"onset": onset[b], "life": lives[b], "pts": []})
            recs[b]["pts"].append({"k": r["k"], "t": float(data[b]["t"][r["k"]]),
                                   "true": r["true"], "path": r["pred"], "plo": r["lo"],
                                   "phi": r["hi"], "capped": r.get("capped", False),
                                   "lt": lt[0], "llo": lt[1], "lhi": lt[2]})
        out[label] = {"records": recs, "all_lives": lives}
    return out


def cell_cohorts(include_rw=False):
    U = load("p3_rul.json")
    pre = load("p3_prereg.json")
    onset = {r["cell"]: r["onset_frac"] for r in pre["battery"]["records"]}
    reg = load("p3_review.json")["regime"]
    out = {}
    for r in U["rows"]:
        if "pred" not in r or (r["batch"] == "Batch-5" and not include_rw):
            continue
        lab = "XJTU cells " + reg[r["batch"]]["regime"]
        c = out.setdefault(lab, {"records": {}})["records"]
        c.setdefault(r["cell"], {"onset": onset[r["cell"]], "life": r["n"] - 1, "pts": []})
        c[r["cell"]]["pts"].append({"k": r["k"], "t": float(r["k"]), "true": r["true"],
                                    "at_life": r["at_life"],
                                    "path": r["pred"], "plo": r["lo"], "phi": r["hi"],
                                    "capped": r["capped"], "lt": r["lt_pred"],
                                    "llo": r["lt_lo"], "lhi": r["lt_hi"]})
    return out


def blind_cohorts():
    B = load("p3_blind_rows.json")
    out = {}
    for r in B["rows"]:
        if "pred" not in r:
            continue
        lab = r["cohort"].replace("N-CMAPSS_", "N-CMAPSS ")
        c = out.setdefault(lab, {"records": {}})["records"]
        c.setdefault(r["unit"], {"onset": r["onset_frac"], "life": r["n"] - 1, "pts": []})
        c[r["unit"]]["pts"].append({"k": r["k"], "t": float(r["k"]), "true": r["true"],
                                    "at_life": r["at_life"],
                                    "path": r["pred"], "plo": r["lo"], "phi": r["hi"],
                                    "capped": r["capped"], "lt": r["lt_pred"],
                                    "llo": r["lt_lo"], "lhi": r["lt_hi"]})
    return out


def metrics(pts, m):
    if not pts:
        return None, None
    a = [ape(p[m], p["true"]) for p in pts]
    lo, hi = ("plo", "phi") if m == "path" else ("llo", "lhi")
    c = [p[lo] <= p["true"] <= p[hi] for p in pts]
    return float(np.median(a)), float(np.mean(c))


def screen_verdict(onsets, rng, ci=None):
    x = np.array([np.inf if f is None else f for f in onsets], float)
    if np.isfinite(x).mean() < 0.25:
        return "check indicator"
    if ci is None:
        n = len(x)
        med = np.median(x[rng.integers(0, n, size=(N_BOOT, n))], axis=1)
        lo, hi = np.quantile(med, [0.025, 0.975], method="inverted_cdf")
        pct = (lambda z: None if not np.isfinite(z) else 100 * float(z))
        ci = (pct(lo), pct(hi))
    return R._verdict3(ci[0], ci[1], THR)


def lt_refit(pts, lives_other):
    """Life-table predictions rebuilt from the other subsampled records only."""
    out = []
    for p in pts:
        m, l, h = R._life_table(lives_other, p["t"])
        q = dict(p)
        q.update({"lt": m, "llo": l, "lhi": h})
        out.append(q)
    return out


# ═════════════════════════════════════════════════════════════════════════
def part1(coh, stored_ci):
    rng = np.random.default_rng(20261114)
    res = {"cohorts": {}, "pooled": {}}
    for name, c in coh.items():
        recs = c["records"]
        ids = sorted(recs)
        N = len(ids)
        allp = [p for i in ids for p in recs[i]["pts"]]
        mp, cp = metrics(allp, "path")
        ml, cl = metrics(allp, "lt")
        usable = bool(mp <= ml)
        #  the full-cohort verdict is Table 15a's, from the stored interval
        sc_full = R._verdict3(stored_ci[name][0], stored_ci[name][1], THR)
        cres = {"N_records": N, "N_points": len(allp), "medape_path": mp, "cov_path": cp,
                "medape_lt": ml, "cov_lt": cl, "usable": usable,
                "screen_full_cohort": sc_full, "by_n": {}}
        for n in list(NS) + ["all"]:
            if n != "all" and n >= N:
                continue
            nn = N if n == "all" else n
            reps = 1 if n == "all" else N_SUB
            tally = {k: 0 for k in ("screen_declines", "screen_passes", "screen_undecided",
                                    "screen_check", "screen_agree", "bt_agree",
                                    "bt_refit_agree", "bt_choose_path", "bt_refit_choose_path")}
            real = {k: [] for k in ("screen", "backtest", "backtest_refit", "life_table",
                                    "path", "oracle")}
            for _ in range(reps):
                sub = ids if n == "all" else list(rng.choice(ids, size=nn, replace=False))
                hold = [i for i in ids if i not in sub] or ids
                # (a) screen on the onsets of the n records
                v = (sc_full if n == "all" else
                     screen_verdict([recs[i]["onset"] for i in sub], rng))
                key = {"declines": "screen_declines", "not declined": "screen_passes",
                       "undecided": "screen_undecided", "check indicator": "screen_check"}[v]
                tally[key] += 1
                s_choice = "path" if v == "not declined" else "lt"
                tally["screen_agree"] += int((s_choice == "path") == usable
                                             if v in ("declines", "not declined") else 0)
                # (b) back-test on the n records with the stored predictions
                sp = [p for i in sub for p in recs[i]["pts"]]
                b_choice = "path" if metrics(sp, "path")[0] <= metrics(sp, "lt")[0] else "lt"
                tally["bt_choose_path"] += int(b_choice == "path")
                tally["bt_agree"] += int((b_choice == "path") == usable)
                # (b') back-test with the life table rebuilt from the other n-1 records
                spr = []
                for i in sub:
                    spr += lt_refit(recs[i]["pts"], [recs[j]["life"] for j in sub if j != i])
                r_choice = "path" if metrics(spr, "path")[0] <= metrics(spr, "lt")[0] else "lt"
                tally["bt_refit_choose_path"] += int(r_choice == "path")
                tally["bt_refit_agree"] += int((r_choice == "path") == usable)
                # realised accuracy of each policy on the held-out records
                hp = [p for i in hold for p in recs[i]["pts"]]
                hm = {m: metrics(hp, m) for m in ("path", "lt")}
                for pol, ch in (("screen", s_choice), ("backtest", b_choice),
                                ("backtest_refit", r_choice), ("life_table", "lt"),
                                ("path", "path"),
                                ("oracle", "path" if hm["path"][0] <= hm["lt"][0] else "lt")):
                    real[pol].append(hm[ch])
            row = {k: v / reps for k, v in tally.items()}
            row["realised_holdout"] = {pol: {"medape": float(np.mean([x[0] for x in v])),
                                             "coverage": float(np.mean([x[1] for x in v]))}
                                       for pol, v in real.items()}
            row["note"] = ("n=all: realised metrics are in-sample (whole cohort)"
                           if n == "all" else f"realised on the {N - nn} held-out records")
            cres["by_n"][str(n)] = row
        res["cohorts"][name] = cres
        print(f"{name:26s} N={N:2d} usable={usable} path {mp:6.1f} lt {ml:6.1f} "
              f"screen={sc_full}  " + "  ".join(
                  f"n={k}: scr {v['screen_agree']:.2f} bt {v['bt_agree']:.2f} "
                  f"btr {v['bt_refit_agree']:.2f}" for k, v in cres["by_n"].items()),
              flush=True)
    # pooled over cohorts (equal weight per cohort; cohorts lacking n are skipped)
    for n in [str(x) for x in NS] + ["all"]:
        cs = [c["by_n"][n] for c in res["cohorts"].values() if n in c["by_n"]]
        us = [c["usable"] for c in res["cohorts"].values() if n in c["by_n"]]
        pooled = {"n_cohorts": len(cs), "n_usable": int(sum(us))}
        for k in ("screen_agree", "screen_declines", "screen_passes", "screen_undecided",
                  "screen_check", "bt_agree", "bt_refit_agree"):
            pooled[k] = float(np.mean([c[k] for c in cs]))
        pooled["life_table_agree"] = float(np.mean([not u for u in us]))
        pooled["realised_holdout"] = {
            pol: {"medape_mean_over_cohorts": float(np.mean([c["realised_holdout"][pol]["medape"]
                                                             for c in cs])),
                  "coverage_mean_over_cohorts": float(np.mean([c["realised_holdout"][pol]["coverage"]
                                                               for c in cs]))}
            for pol in ("screen", "backtest", "backtest_refit", "life_table", "path", "oracle")}
        res["pooled"][n] = pooled
        print("pooled", n, json.dumps(pooled), flush=True)
    return res


def part1_cost():
    """Time one path-model prediction and one threshold calibration on XJTU-SY."""
    data = ds.load_dataset("xjtu")
    b = sorted(data)[0]
    rec = data[b]
    D_obs, ref = ds.threshold_for(data, b, "xjtu")
    mls = float(np.median([max(data[r]["n"] - ds.first_prediction_time(data[r]["hi"]), 20)
                           for r in ref]))
    out = {"record": b, "n_acq": rec["n"]}
    for npart in (600, sb.N_PART):
        t0 = time.perf_counter()
        sb.predict_one(rec, int(0.7 * (rec["n"] - 1)), D_obs, "BS", True, True, mls,
                       seed=1, n_part=npart)
        out[f"predict_s_{npart}"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sb.latent_terminal(data[ref[0]], "BS", True, True, mls, D_obs, seed=7)
    out["calibrate_one_ref_s_600"] = time.perf_counter() - t0
    # the screen: onset detection on n records plus a 1000-draw bootstrap
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    on = [ds.first_prediction_time(data[r]["hi"]) / data[r]["n"] for r in data]
    screen_verdict(on, rng)
    out["screen_all_records_s"] = time.perf_counter() - t0
    out["n_records_screen"] = len(data)
    out["backtest_fits_formula"] = ("LOO on n records with m points each: n*m filter "
                                    "predictions, plus n*(n-1) latent calibrations where "
                                    "the threshold is set on the latent scale (bearings); "
                                    "cells/N-CMAPSS set D on the indicator, no calibration fit")
    print("cost", out, flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════
def part2a():
    app = load("applicability.json")
    lead = {}
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        rs = [r for r in app if r["dataset"] == name]
        lead[label] = ("h", [r["life_s"] * (1 - r["fpt_frac"]) / 3600 if r["fpt_idx"] > 0
                             else 0.0 for r in rs], [r["fpt_idx"] > 0 for r in rs])
    ns = R._thirdset_namespace()
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", True), ("FD004", True)):
        v, d = [], []
        for u, arr in sorted(ns["load"](fd, by_regime=by).items()):
            hi = ns["indicator"](arr)
            i = ns["onset"](hi)
            v.append(float(len(hi) - i) if i > 0 else 0.0)
            d.append(i > 0)
        lead["C-MAPSS " + fd + ("+regime" if by else "")] = ("cycles", v, d)
    bat = load("battery.json")
    lead["NASA cells"] = ("cycles", [float(c["n_cycles"] - c["onset_idx"]) if c["onset_idx"]
                                     and c["onset_idx"] > 0 else 0.0 for c in bat["cells"]],
                          [bool(c["onset_idx"] and c["onset_idx"] > 0) for c in bat["cells"]])
    hdd = load("hdd.json")
    lead["Backblaze drives"] = ("h", [r["life_hours"] * (1 - r["onset_frac"])
                                      if r["onset_frac"] is not None else 0.0
                                      for r in hdd["drives"]],
                                [r["onset_frac"] is not None for r in hdd["drives"]])
    pre = load("p3_prereg.json")
    for key, label in (("battery", "XJTU Li-ion"), ("ncmapss", "N-CMAPSS")):
        rs = [r for r in pre[key]["records"] if "load_error" not in r]
        det = [r.get("onset_idx") is not None and r["onset_idx"] > 0 for r in rs]
        lead[label] = ("cycles", [float(r["n"] - r["onset_idx"]) if dd else 0.0
                                  for r, dd in zip(rs, det)], det)
    stored = load("p3_review.json")["lead"]
    resp = {"h": [("10 min", 1 / 6), ("1 h", 1.0), ("24 h", 24.0), ("1 week", 168.0)],
            "cycles": [(f"{c} cycles", float(c)) for c in (10, 25, 50, 100)]}
    out = {}
    for label, (unit, v, det) in lead.items():
        v = np.array(v)
        det = np.array(det, bool)
        vd = v[det]
        q1, q2, q3 = np.percentile(vd, [25, 50, 75])
        sk = label if label in stored else label
        if sk in stored:      # same numbers as Table 8's onset-to-failure column
            f = 3600.0 if stored[sk]["unit"] == "s" else 1.0
            assert abs(stored[sk]["median"] / f - q2) < 1e-6 * max(1, q2), (label, q2)
        out[label] = {"unit": unit, "n_units": int(len(v)), "n_detected": int(det.sum()),
                      "median_detected": float(q2), "q1_detected": float(q1),
                      "q3_detected": float(q3),
                      "median_all_units_undetected_as_zero": float(np.median(v)),
                      "share_exceeding": {
                          nm: {"of_detected": float(np.mean(vd > x)),
                               "of_all_units": float(np.mean(v > x))}
                          for nm, x in resp[unit]}}
        print(label, json.dumps(out[label]), flush=True)
    return out


def part2b(bear, cells, blind):
    reg = load("p3_review.json")["regime"]
    verdict = {"PRONOSTIA": "declined", "XJTU-SY": "undecided"}
    for b, v in reg.items():
        verdict["XJTU cells " + v["regime"]] = {"declines": "declined",
                                                 "not declined": "passed"}.get(v["verdict"],
                                                                               "undecided")
    for k in blind:
        verdict[k] = "passed"
    allc = {**bear, **cells, **blind}

    def rates(pts):
        out = {"n_points": len(pts)}
        for m, lo, hi in (("path", "plo", "phi"), ("life_table", "llo", "lhi")):
            out[m] = {"miss_true_below_lo": float(np.mean([p["true"] < p[lo] for p in pts])),
                      "early_true_above_hi": float(np.mean([p["true"] > p[hi] for p in pts])),
                      "coverage": float(np.mean([p[lo] <= p["true"] <= p[hi] for p in pts]))}
        out["path"]["share_capped"] = float(np.mean([bool(p["capped"]) for p in pts]))
        out["path"]["share_hi_zero"] = float(np.mean([p["phi"] <= 0 for p in pts]))
        return out

    res = {"by_cohort": {}, "by_verdict": {}}
    groups = {}
    for name, c in allc.items():
        pts = [p for r in c["records"].values() for p in r["pts"]]
        res["by_cohort"][name] = {"verdict70": verdict[name], **rates(pts)}
        groups.setdefault(verdict[name], []).extend(pts)
        groups.setdefault("all", []).extend(pts)
    for g, pts in groups.items():
        res["by_verdict"][g] = rates(pts)
        # cohort-weighted too, so the 15-cell 3C regime or the bearings do not dominate
        cs = [v for v in res["by_cohort"].values() if g == "all" or v["verdict70"] == g]
        res["by_verdict"][g]["cohort_mean"] = {
            m: {k: float(np.mean([c[m][k] for c in cs]))
                for k in ("miss_true_below_lo", "early_true_above_hi", "coverage")}
            for m in ("path", "life_table")}
        res["by_verdict"][g]["n_cohorts"] = len(cs)
        print(g, json.dumps(res["by_verdict"][g]), flush=True)
    res["note"] = ("lo/hi are the 2.5 and 97.5 % predictive quantiles for both predictors "
                   "(study_bearings.predict_one; study_p3_review._life_table). Includes the "
                   "XJTU RW regime (3 cells, undecided), which Table 15a omits.")
    return res


if __name__ == "__main__":
    t0 = time.time()
    bear = bearing_cohorts()
    # stored-MedAPE checks against p3_review.json["decisions"]
    dec = {c["cohort"]: c for c in load("p3_review.json")["decisions"]["cohorts"]}
    cells, blind = cell_cohorts(), blind_cohorts()
    coh = {k: {"records": v["records"]} for k, v in bear.items()}
    coh.update(cells)
    coh.update(blind)
    assert set(coh) == set(dec), (set(coh) ^ set(dec))
    for k, c in coh.items():
        pts = [p for r in c["records"].values() for p in r["pts"]]
        mp, ml = metrics(pts, "path")[0], metrics(pts, "lt")[0]
        assert abs(mp - dec[k]["path"]) < 1e-6 and abs(ml - dec[k]["lt"]) < 1e-6, (k, mp, ml)
    stored_ci = {k: (v["lo"], v["hi"]) for k, v in dec.items()}
    out = {"part1": part1(coh, stored_ci)}
    out["part1_cost"] = part1_cost()
    out["part2a_lead"] = part2a()
    out["part2b_miss"] = part2b({k: {"records": v["records"]} for k, v in bear.items()},
                                cell_cohorts(include_rw=True), blind)
    out["settings"] = {"N_SUB": N_SUB, "N_BOOT": N_BOOT, "threshold": THR, "ns": NS,
                       "seed": 20261114, "runtime_s": time.time() - t0}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)
