# -*- coding: utf-8 -*-
r"""Emit results/tables/tab_archives.tex: every archive the rule has been run on.

The paper now applies the screening rule to eight archives across four asset
classes, and until this table the evidence was spread over four sections and a
sentence listing eight medians.  A reader deciding whether the rule applies to
their own asset wants one place that says, for each archive: how many units, how
many register an onset at all, where that onset sits, and what verdict the rule
returns.

Generated from the result files rather than typed, like every other table here.

    python code/table_archives.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bearings")
OUT = os.path.join(HERE, "..", "results", "tables", "tab_archives.tex")


def load(name):
    p = os.path.join(RES, name)
    return json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None


rows = []

app = load("applicability.json") or []
for tag, label, cls in (("pronostia", "PRONOSTIA", "rolling bearing"),
                        ("xjtu", "XJTU-SY", "rolling bearing")):
    rec = [r for r in app if r["dataset"] == tag]
    det = [r for r in rec if r.get("fpt_frac")]
    rows.append((label, cls, len(rec), len(det),
                 100 * st.median([r["fpt_frac"] for r in det]) if det else None))

third = load("thirdset.json") or {}
for key, label in (("FD001", "C-MAPSS FD001"), ("FD003", "C-MAPSS FD003"),
                   ("FD002+regime", "C-MAPSS FD002"),
                   ("FD004+regime", "C-MAPSS FD004")):
    b = third.get(key)
    if b:
        rows.append((label, "turbofan engine", b["n_units"], b["n_detected"],
                     b["median_onset_pct"]))

bat = load("battery.json")
if bat:
    rows.append(("NASA Li-ion", "lithium-ion cell", bat["n_cells"],
                 bat["n_detected"], bat["median_onset_pct"]))

hdd = load("hdd.json")
if hdd:
    rows.append(("Backblaze Q4 2023", "hard disk drive", hdd["n_usable"],
                 hdd["n_detected"], hdd["median_onset_pct"]))


#  intervals from study_p3_review.py (ci): Wilson for the detection rate,
#  percentile bootstrap for the median onset
CI = (load("p3_review.json") or {}).get("ci", {})
CI_KEY = {"PRONOSTIA": "PRONOSTIA", "XJTU-SY": "XJTU-SY",
          "C-MAPSS FD001": "C-MAPSS FD001", "C-MAPSS FD003": "C-MAPSS FD003",
          "C-MAPSS FD002": "C-MAPSS FD002+regime", "C-MAPSS FD004": "C-MAPSS FD004+regime",
          "NASA Li-ion": "NASA cells", "Backblaze Q4 2023": "Backblaze drives"}


def point_verdict(med):
    #  the rule as first stated, on the point estimate of the median
    return r"declines" if med >= 70 else r"not declined"


def verdict(lo, hi):
    """Three outcomes, read off the bootstrap interval of the censored median.

    An interval wholly at or above the threshold declines; one wholly below
    does not; one that straddles it is undecided, and the cohort needs more
    records before the screen can say either.  "Not declined" is deliberately
    not "applies": XJTU-SY's onset is early and every increment law still errs
    there by order 100 % (tab:path).  None stands for "not reached", beyond
    the end of life.
    """
    lo = 1e9 if lo is None else lo
    hi = 1e9 if hi is None else hi
    if lo >= 70:
        return r"declines"
    return r"not declined" if hi < 70 else r"undecided"


#  time from the detected onset to failure (study_p3_review.py lead)
LEAD = (load("p3_review.json") or {}).get("lead", {})
#  thresholds, within 50-99 %, at which each verdict is the one at 70 % (thr)
SAME = (load("p3_review.json") or {}).get("thr", {}).get("same_as_70", {})


def fmt_same(key):
    lo, hi = SAME[key]
    if lo <= 50 and hi >= 99:
        return "all"
    if lo <= 50:
        return f"$\\le{hi:.0f}$"
    if hi >= 99:
        return f"$\\ge{lo:.0f}$"
    return f"{lo:.0f}--{hi:.0f}"


def fmt_lead(key):
    d = LEAD[key]
    v, u = d["median"], d["unit"]
    if u == "s":
        return f"{v / 60:.0f} min" if v < 3600 else f"{v / 3600:.1f} h"
    return f"{v:.0f} h" if u == "hours" else f"{v:.0f} cycles"


#  the rule's own statistic: undetected records right-censored at end of life
CENS = (load("p3_review.json") or {}).get("cens", {})


def fmt(v):
    return "n.r." if v is None else f"{v:.1f}"


body = []
for label, cls, n, det, med in rows:
    c = CI[CI_KEY[label]]
    assert c["n"] == n and c["n_detected"] == det, label
    assert abs(c["median_onset_pct"] - med) < 1e-9, label
    lo, hi = c["median_ci"]
    rlo, rhi = c["rate_ci"]
    k = CENS[CI_KEY[label]]
    assert k["n"] == n and k["n_detected"] == det, label
    cm, (clo, chi) = k["censored_median_pct"], k["censored_median_ci"]
    #  "not reached" is beyond the end of life, so it declines
    cv = 1e9 if cm is None else cm
    assert point_verdict(cv) == point_verdict(med), label    # the two statistics agree
    v3 = verdict(clo, chi)
    #  the interval reading never contradicts the point one; it only withholds
    assert v3 == "undecided" or v3 == point_verdict(cv), label
    body.append(f"{label} & {cls} & {n} & {det} & {det / n:.2f} [{rlo:.2f}, {rhi:.2f}] & "
                f"{med:.1f} [{lo:.1f}, {hi:.1f}] & "
                f"{fmt(cm)} [{fmt(clo)}, {fmt(chi)}] & {fmt_lead(CI_KEY[label])} & {v3} & {fmt_same(CI_KEY[label])}\\\\")

#  the pre-registered test (study_p3_prereg.py): protocol frozen before the
#  analysis, verdict taken as the script computed it
PRE = load("p3_prereg.json")
if PRE:
    body.append(r"\midrule")
    for key, label, cls, lkey in (("battery", "XJTU Li-ion", "lithium-ion cell", "XJTU Li-ion"),
                                  ("ncmapss", "N-CMAPSS DS01--02", "turbofan engine", "N-CMAPSS")):
        s = PRE[key]
        n, det = s["n"], s["n_detected"]
        rlo, rhi = s["rate_ci"]
        cm, (clo, chi) = s["censored_median_pct"], s["censored_median_ci"]
        dm = (f"{s['median_pct']:.1f} [{s['median_ci'][0]:.1f}, {s['median_ci'][1]:.1f}]"
              if "median_pct" in s else "---")
        v3 = verdict(clo, chi)
        #  the protocol fixed the point rule; the interval reading agrees here
        assert v3 == s["verdict"], label
        body.append(f"{label}$^\\ddagger$ & {cls} & {n} & {det} & {det / n:.2f} "
                    f"[{rlo:.2f}, {rhi:.2f}] & {dm} & {fmt(cm)} [{fmt(clo)}, {fmt(chi)}] & "
                    f"{fmt_lead(lkey)} & {v3} & {fmt_same(lkey)}\\\\")

#  the pre-registered test of a decline (study_p3_decline.py): two quarters of
#  Backblaze drives, verdict as the script computed it
DEC = load("p3_decline.json")
if DEC:
    for q, c in DEC["cohorts"].items():
        if not c.get("tested"):
            continue
        cm, (clo, chi) = c["censored_median_pct"], c["censored_median_ci"]
        assert verdict(clo, chi) == c["verdict"], q
        dm = c["median_detected_pct"]
        dci = c.get("median_detected_ci")
        dms = f"{dm:.1f} [{dci[0]:.1f}, {dci[1]:.1f}]" if dci else f"{dm:.1f}"
        lead = f"{c['lead_hours_median']:.0f} h"
        body.append(f"Backblaze {q.replace('_', ' ')}$^\\S$ & hard disk drive & {c['n_usable']} & "
                    f"{c['n_detected']} & {c['rate']:.2f} [{c['rate_ci'][0]:.2f}, "
                    f"{c['rate_ci'][1]:.2f}] & {dms} & {fmt(cm)} [{fmt(clo)}, {fmt(chi)}] & "
                    f"{lead} & {c['verdict']} & {fmt_same('Backblaze ' + q)}\\\\")

PRE_NOTE = (r" $^\ddagger$, the pre-registered test of \Cref{sec:prereg}, whose protocol"
            r" was fixed before the analysis." if PRE else "")
if DEC:
    PRE_NOTE += r" $^\S$, the pre-registered test of a decline, likewise."

tex = r"""%  generated by code/table_archives.py -- do not edit
\begin{table}[htbp]
\centering
\caption{The screening rule on every archive in the study, its constants unchanged.}
\label{tab:archives}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{@{}llrrcccrlc@{}}
\toprule
& & & \multicolumn{2}{c}{detected} & \multicolumn{2}{c}{median onset, \% of life} & onset to & & holds for\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
archive & asset class & units & $n$ & rate & detected units & all units, censored & failure & verdict & thresholds\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\vspace{4pt}
\begin{minipage}{0.95\linewidth}\footnotesize
Brackets are $95\,\%$ intervals: Wilson for the rate, percentile bootstrap
($10\,000$ resamples) for the medians. The censored median counts a unit without
a detected onset as right-censored at the end of its life; n.r., not reached,
means that fewer than half the units show an onset before failure. The verdict
reads the censored interval against the $70\,\%$ threshold: declines if it lies
wholly at or above it, not declined if wholly below, undecided if it straddles
it, and the last column gives the thresholds, within $50$--$99\,\%$, at which
the verdict is the same. Onset to failure is the median over detected units, in the archive's own
time unit. For the disk drives life is the drive's own power-on hours;
the C-MAPSS multi-regime subsets are normalised within operating regime
(\Cref{sec:band}).""" + PRE_NOTE + r"""
\end{minipage}
\end{table}
"""

io.open(OUT, "w", encoding="utf-8", newline="").write(tex)
print(f"  wrote {os.path.relpath(OUT, os.path.join(HERE, '..'))}  ({len(rows)} archives)")
