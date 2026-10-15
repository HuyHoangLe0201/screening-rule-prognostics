"""Tables of paper 3 from the stored results: tab_applicability and tab_path,
then the four table scripts.  python code/make_tables_p3.py"""
import json
import os
import re
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
BE = os.path.join(HERE, "..", "results", "bearings")
OUT = os.path.join(HERE, "..", "results", "tables")


def jload(p):
    with open(p) as f:
        return json.load(f)


def _fit(s):
    """Wrap every tabular so it shrinks to the text width instead of overrunning.

    elsarticle's measure is narrower than the article class these tables were
    first laid out for, and several of them ran up to 70pt into the margin.
    adjustbox only scales when a table is actually too wide, so tables that
    already fit are left at their natural size and unscaled.
    """
    return re.sub(
        r"(\\begin\{tabular\}.*?\\end\{tabular\})",
        r"\\begin{adjustbox}{max width=\\linewidth}\n\1\n\\end{adjustbox}",
        s, flags=re.S)


def write(name, s):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(_fit(s))
    print(f"  -> results/tables/{name}")


def t_applicability():
    rows = jload(os.path.join(BE, "applicability.json"))
    out = ""
    for dset, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        rs = [r for r in rows if r["dataset"] == dset]
        body = ""
        for r in rs:
            fp = f"{100*r['fpt_frac']:.1f}" if r["fpt_frac"] is not None else "none"
            dp = f"{100*r['deg_phase_frac']:.1f}" if r["deg_phase_frac"] is not None else "--"
            fr = (f"{100*r['frac_rise_in_last_5pct']:.0f}"
                  if r["frac_rise_in_last_5pct"] is not None else "--")
            body += (f"{r['bearing'].replace('_', chr(92)+'_')} & {r['n']} & {fp} & {dp} & "
                     f"{r['terminal_snr_mad']:.0f} & {fr}\\\\\n")
        det = [r for r in rs if r["fpt_frac"] is not None]
        med = 100 * sorted(r["fpt_frac"] for r in det)[len(det) // 2] if det else float("nan")
        out += f"""\\begin{{minipage}}[t]{{0.49\\linewidth}}\\centering\\scriptsize
{label}\\\\[2pt]
\\begin{{tabular}}{{@{{}}l rrrrr@{{}}}}
\\toprule
bearing & $n$ & onset & deg. & SNR & last 5\\%\\\\
\\midrule
{body}\\midrule
\\multicolumn{{6}}{{@{{}}l@{{}}}}{{onset in {len(det)}/{len(rs)}, median at {med:.0f}\\,\\% of life}}\\\\
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}\\hfill
"""
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Applicability of a first-passage model to each benchmark record, with the detected
onset as a fraction of life.}}
\\label{{tab:applicability}}
{out}
\\end{{table}}
"""


def t_path_compact():
    """One compact table for the whole path-model study on real bearings.

    The per-split tables are large and say one thing; this carries the four
    numbers the text relies on. Full tables remain in
    results/bearings/results_both.json.
    """
    d = jload(os.path.join(BE, "results_both.json"))["tables"]
    cols = ["MAE", "MedAPE_pct", "PHM_score", "coverage95"]

    def cells(key, row):
        t = d.get(key, {})
        if row not in t:
            return " & --- & --- & --- & ---"
        v = t[row]
        return (f" & {v['MAE']:.0f} & {v['MedAPE_pct']:.0f} & "
                f"{v['PHM_score']:.3f} & {v['coverage95']:.2f}")

    body = "\\multicolumn{9}{@{}l@{}}{\\itshape increment law "
    body += "($n=77$ / $n=89$ prediction points, $N_p=1200$)}\\\\[1pt]\n"
    for r, nm in (("BS", "BS"), ("BS-t", "BS-$t$"), ("Wiener", "Wiener"),
                  ("Gamma", "gamma"), ("IGproc", "IG process")):
        body += nm + cells("pronostia/process", r) + cells("xjtu/process", r) + "\\\\\n"

    body += ("\\midrule\n\\multicolumn{9}{@{}l@{}}{\\itshape ablation and scaling, "
             "BS only (every second point, $N_p=600$; row 1 is BS above)}"
             "\\\\[1pt]\n")
    for r, nm in (("prior=informed, coupling=on", "informed prior, coupling"),
                  ("prior=informed, coupling=off", "informed prior, no coupling"),
                  ("prior=diffuse, coupling=on", "diffuse prior, coupling"),
                  ("prior=diffuse, coupling=off", "diffuse prior, no coupling")):
        body += nm + cells("pronostia/ablation", r) + cells("xjtu/ablation", r) + "\\\\\n"
    body += ("naive interval scaling" + cells("pronostia/grid", "BS-naive")
             + cells("xjtu/grid", "BS-naive") + "\\\\\n")

    #  the review analyses of study_p3_review.py: the same model on other
    #  inputs, and the two decisions the screen recommends instead of it
    rv = jload(os.path.join(BE, "p3_review.json")) if os.path.exists(
        os.path.join(BE, "p3_review.json")) else {}

    def rcells(v):
        return (f" & {v['MAE']:.0f} & {v['MedAPE_pct']:.0f} & "
                f"{v['PHM_score']:.3f} & {v['coverage95']:.2f}")

    sens = rv.get("sens", {})
    if "raw_indicator" in sens and "particles_x4" in sens:
        body += ("\\midrule\n\\multicolumn{9}{@{}l@{}}{\\itshape sensitivity runs "
                 "(BS, every point, main-study seeds)}\\\\[1pt]\n")
        for key, nm in (("raw_indicator", "unsmoothed indicator"),
                        ("particles_x4", "$N_p=4800$")):
            body += nm + rcells(sens[key]["pronostia"]) + rcells(sens[key]["xjtu"]) + "\\\\\n"
        s2 = rv.get("sens2", {})
        for key, nm in (("noise_prior_sd_0.5", "noise prior s.d.\\ $0.5$"),
                        ("noise_prior_sd_1.5", "noise prior s.d.\\ $1.5$"),
                        ("floor_c_0.01", "floor $c=0.01$"),
                        ("floor_c_0.20", "floor $c=0.20$"),
                        ("ar1_observation", "autoregressive noise")):
            if key in s2:
                body += nm + rcells(s2[key]["pronostia"]) + rcells(s2[key]["xjtu"]) + "\\\\\n"
    reg = rv.get("reg", {})
    if reg:
        body += ("\\midrule\n\\multicolumn{9}{@{}l@{}}{\\itshape no threshold "
                 "(same points and training records)}\\\\[1pt]\n")
        for key, nm in (("gbr", "gradient boosting, $\\log\\RUL$"),
                        ("life_table", "life table")):
            body += nm + rcells(reg["pronostia"][key]) + rcells(reg["xjtu"][key]) + "\\\\\n"
    #  the calibrated failure threshold made uncertain (dthr): every point, the
    #  main study's filter; the D-fixed prediction is the BS row above
    dt = rv.get("dthr", {})
    if {"pronostia", "xjtu"} <= set(dt):
        sp, sx = dt["pronostia"]["summary"], dt["xjtu"]["summary"]
        assert abs(sp["fixed"]["MAE"] - d["pronostia/process"]["BS"]["MAE"]) < 1e-6
        body += ("\\midrule\n\\multicolumn{9}{@{}l@{}}{\\itshape the failure threshold $D$, "
                 "uncertain over its reference records (BS, every point)}\\\\[1pt]\n")
        for key, nm in (("mixture", "mixed over the law of $D$"),
                        ("D_low", "$D$ at its $2.5\\,\\%$ point"),
                        ("D_high", "$D$ at its $97.5\\,\\%$ point")):
            body += nm + rcells(sp[key]) + rcells(sx[key]) + "\\\\\n"
    #  the exact-posterior reference, last because it uses a subset of points:
    #  the main filter is shown again on exactly that subset rather than
    #  compared with row 1 (on the unsmoothed indicator, whose noise the
    #  observation model describes)
    ex = rv.get("exact", {})
    if "raw" in ex and {"pronostia", "xjtu"} <= set(ex["raw"]):
        e = ex["raw"]
        body += ("\\midrule\n\\multicolumn{9}{@{}l@{}}{\\itshape unsmoothed indicator, "
                 "one point per record near $70\\,\\%$ of life ("
                 + f"$n={e['pronostia']['exact']['n']}$ / $n={e['xjtu']['exact']['n']}$"
                 + ")}\\\\[1pt]\n")
        body += ("Liu--West filter" + rcells(e["pronostia"]["main"])
                 + rcells(e["xjtu"]["main"]) + "\\\\\n")
        body += ("exact posterior" + rcells(e["pronostia"]["exact"])
                 + rcells(e["xjtu"]["exact"]) + "\\\\\n")

    return f"""\\begin{{table}}[htbp]
\\centering\\footnotesize
\\caption{{The first-passage path model on real bearings, with ablations,
sensitivity runs, an uncertain failure threshold, two predictors that need no
threshold and an exact-posterior reference.}}
\\label{{tab:path}}
\\begin{{tabular}}{{@{{}}l rrrr rrrr@{{}}}}
\\toprule
& \\multicolumn{{4}}{{c}}{{PRONOSTIA}} & \\multicolumn{{4}}{{c}}{{XJTU-SY}}\\\\
\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}
& MAE & MedAPE & PHM & cov & MAE & MedAPE & PHM & cov\\\\
\\midrule
{body}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


if __name__ == "__main__":
    write("tab_applicability.tex", t_applicability())
    write("tab_path.tex", t_path_compact())
    for name in ("table_archives", "table_altind", "table_ladder", "table_rultest", "table_blind", "table_step", "table_decide", "table_filter", "table_band", "table_onset_sens", "table_thirdset"):
        runpy.run_path(os.path.join(HERE, name + ".py"), run_name="__main__")
