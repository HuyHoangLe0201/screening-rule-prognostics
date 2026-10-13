"""LaTeX table for the held-back archives: C-MAPSS and NASA battery."""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CM = os.path.join(HERE, "..", "results", "bearings", "thirdset.json")
BAT = os.path.join(HERE, "..", "results", "bearings", "battery.json")

COND = {"FD001": "one", "FD002": "six", "FD003": "one", "FD004": "six",
        "FD002+regime": "six, repaired", "FD004+regime": "six, repaired"}


#  censored median intervals of every archive (study_p3_review.py cens)
CENS = json.load(io.open(os.path.join(HERE, "..", "results", "bearings", "p3_review.json"),
                         encoding="utf-8"))["cens"]


def build():
    d = json.load(io.open(CM, encoding="utf-8"))
    b = json.load(io.open(BAT, encoding="utf-8"))

    def verdict(n_det, n, key):
        #  Algorithm 3: an onset in fewer than a quarter of the records sends the
        #  user back to the indicator; otherwise the bootstrap interval of the
        #  censored median is read against 70 %, with three outcomes.  A pass is
        #  "not declined", never "applies", which the study does not show.
        if n_det / n < 0.25:
            return "check indicator"
        lo, hi = CENS[key]["censored_median_ci"]
        lo = 1e9 if lo is None else lo
        hi = 1e9 if hi is None else hi
        if lo >= 70:
            return "declines"
        return "not declined" if hi < 70 else "undecided"

    rows = ""
    for fd in ("FD001", "FD003", "FD002", "FD004", "FD002+regime", "FD004+regime"):
        r = d[fd]
        m, p = r["median_onset_pct"], r["median_deg_phase_pct"]
        rows += (f"C-MAPSS {fd.replace('+regime', '')} & {COND[fd]} & "
                 f"{r['n_units']} & {r['n_detected']}/{r['n_units']} & "
                 f"{r['median_healthy_mad']:.2f} & {m:.0f} & {p:.0f} & "
                 + verdict(r["n_detected"], r["n_units"], "C-MAPSS " + fd) + "\\\\\n")
    rows += (f"NASA battery & --- & {b['n_cells']} & "
             f"{b['n_detected']}/{b['n_cells']} & --- & "
             f"{b['median_onset_pct']:.0f} & {b['median_deg_phase_pct']:.0f} & "
             + verdict(b["n_detected"], b["n_cells"], "NASA cells") + "\\\\\n")
    v_p = verdict(12, 17, "PRONOSTIA")
    v_x = verdict(15, 15, "XJTU-SY")

    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{The screening rule on the two bearing archives and, unchanged, on archives
that took no part in its construction.}}
\\label{{tab:thirdset}}
\\begin{{adjustbox}}{{max width=\\linewidth}}
\\begin{{tabular}}{{@{{}}l l r r r r r l@{{}}}}
\\toprule
archive & op.\\ cond. & units & onset found & healthy scale
        & median onset [\\%] & deg.\\ phase [\\%] & verdict\\\\
\\midrule
PRONOSTIA & one & 17 & 12/17 & --- & 97 & 3 & {v_p}\\\\
XJTU-SY & one & 15 & 15/15 & --- & 41 & 59 & {v_x}\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}

\\vspace{{4pt}}
\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize
The first two rows are the archives on which the rule was built, repeated for
comparison. The healthy-scale column is on each archive's own indicator scale and
is comparable only within C-MAPSS.
\\end{{minipage}}
\\end{{table}}
"""


if __name__ == "__main__":
    out = os.path.join(HERE, "..", "results", "tables", "tab_thirdset.tex")
    io.open(out, "w", encoding="utf-8").write(build())
    print("  -> results/tables/tab_thirdset.tex")
