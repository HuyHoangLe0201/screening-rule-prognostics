# screening-rule-prognostics

Code and stored results for

> H. H. Le and K.-A. Nguyen, *A Screening Rule for Degradation-Threshold
> Prognostics* (submitted).

A degradation-threshold prognostic fits a stochastic process to a health
indicator and predicts remaining useful life (RUL) as a first-passage time.
When it predicts badly, either the inference failed or the model does not
apply, and the two call for opposite remedies. The paper separates them. One
particle filter, for a grid-consistent Birnbaum–Saunders (BS) process with a
Paris-type coupling and an epistemic/aleatory split of the predictive variance,
is tested on data simulated from the model and on the PRONOSTIA and XJTU-SY
bearing benchmarks. An exact-posterior reference (particle marginal
Metropolis–Hastings) shows that the benchmark failure divides by degradation
onset. The paper turns that into an offline screen: from completed
run-to-failure records, it declines a first-passage model on an indicator when
the bootstrap interval of the median detected onset lies at 70 % of life or
later, and is undecided when the interval straddles that value.

## Data

No archive is redistributed here, neither raw nor as derived features or
health indicators. Download them from their publishers.

| archive | used for | where the code looks |
|---|---|---|
| PRONOSTIA / IEEE PHM 2012 | bearings, onset, path model | `PRONOSTIA_DIR`, or the first argument of `extract_features.py` |
| XJTU-SY | bearings, onset, path model | `XJTU_DIR`, or the first argument of `extract_xjtu.py` |
| NASA C-MAPSS (`train_FD00x.txt`) | the rule on turbofan engines, the band study | `$P3_DATA/CMAPSSData/` |
| NASA Li-ion cells B0005, B0006, B0007, B0018 (`.mat`) | the rule on cells | `$P3_DATA/` |
| Backblaze drive stats, Q4 2023 (`.zip`) | the rule on disk drives | the argument of `study_hdd.py` |
| Backblaze drive stats, Q1 and Q2 2024 | pre-registered test of a decline | `$P3_DATA/bb_Q1_2024.zip`, `$P3_DATA/bb_Q2_2024.zip` |
| XJTU battery dataset | pre-registered test | `$P3_DATA/NASABattery/Battery Dataset/` |
| N-CMAPSS DS01-005 and DS02-006 (`.h5`) | pre-registered test | `$P3_DATA/ncmapss/` |
| N-CMAPSS, the full archive (`ncmapss.zip`) | blind pre-registered test | `$P3_DATA/ncmapss/ncmapss.zip` |

`P3_DATA` defaults to `data/` in this repository. The bearing features are
extracted once:

```bash
python code/extract_features.py /path/to/phm-ieee-2012-data-challenge-dataset-master
python code/extract_xjtu.py /path/to/XJTU-SY_Bearing_Datasets
python code/hi.py
```

## What is stored

* `results/bearings/`: every result the paper's tables and figures are built
  from. `applicability.json` (detected onset per bearing), `results_both.json`
  (the path-model study, all laws and ablations), `seedscatter.json`,
  `onset_sensitivity.json`, `thirdset.json`, `battery.json`, `hdd.json`, the
  band study (`band*.json`), `p3_review.json` (predictive checks, sensitivity
  runs, the exact-posterior reference, censored medians, bootstrap intervals,
  the no-threshold predictors, the verdicts across thresholds, the failure
  threshold made uncertain, the screen on other indicators, the screen per
  battery regime, the coupled-model checks, the policy comparison),
  `p3_prereg.json`, `p3_decline.json`, `p3_rul.json` and `p3_blind.json` with
  `p3_blind_rows.json` (the four pre-registered tests).
* `results/theory/`: the simulation studies on data from the model
  (`theory_results.json`, `proposal.json`) and the checks of the model itself
  (`grid_consistency.json`, `lamperti.json`, `irregular.json` and others).
* `results/hi/meta.json`: the indicator's construction, without the per-bearing
  series.
* `results/tables/`: the paper's generated LaTeX tables.
* `results/figures/`: the paper's figures as pgfplots code;
  `preview.tex` compiles them all.

## Reproducing

From the stored results, in seconds and without any data:

```bash
cd code
python make_tables_p3.py    # every table; byte-identical to results/tables/
python -c "import make_fig_p3 as M; [f() for f in (M.fig_diagnoses, M.fig_decomp,
  M.fig_ssm, M.fig_grid, M.fig_ppc, M.fig_filter, M.fig_guided, M.fig_robustprior)]"
```

`python make_fig_p3.py` writes all ten figures; the two onset figures also need
the extracted PRONOSTIA features.

The full pipeline, in order (laptop CPU; times are rough):

```bash
cd code
# data from the model (about an hour)
python study_theory.py; python study_proposal.py
python test_gridconsistency.py; python test_lamperti.py; python study_irregular.py
# the bearings (several hours)
python study_applicability.py
python study_bearings.py both
python study_seedscatter.py; python study_onset_sensitivity.py
# other archives and the band study
python study_thirdset.py; python study_battery.py
python study_hdd.py /path/to/bb_Q4_2023.zip
python study_band.py
python study_bandtest.py 20 20260805 bandtest
python study_bandtest.py 20 20260806 bandtest_rep
python study_bandsig.py bandtest; python study_bandsig.py bandtest_rep
python study_bandbattery.py; python study_bandbearing.py
# review analyses: every part, or name the ones to run
python study_p3_review.py ci acf ppc cal reg sens cens hw boot dec sens2 lead altind
python study_p3_review.py thr                  # after study_p3_decline.py summary
python study_p3_review.py dthr                 # the uncertain threshold D (an hour or so)
python study_p3_review.py regime igshort calgamma calgrid policy
# the exact-posterior reference takes about two hours on eight processes:
# shard it, then merge and summarise
P3_EXACT_SHARD=0/5 P3_REVIEW_OUT=p3_review_ex0.json python study_p3_review.py exact raw pronostia
#   ... shards 1/5 to 4/5, and likewise for xjtu ...
python study_p3_review.py merge; python study_p3_review.py exact_sum
python make_tables_p3.py; python make_fig_p3.py
```

All studies are seeded. `study_bearings.py` seeds every prediction from a hash
of the bearing, the index and the law, so a rerun reproduces
`results_both.json`; `study_p3_review.py sens` reproduces its BS row exactly.

## The pre-registered tests

`code/PREREG_p3_new_archives.md` fixed the archives, the indicator for each,
the detector, the rule and four predictions before any onset on those archives
was computed. It was frozen on 12 October 2026 with SHA-256 digest
`c6116aeba39294d6cc3d6af0e62ceda65f00d065941a47a75936df24b2e2d0b1`, and
`study_p3_prereg.py` refuses to run if the file differs:

```bash
python study_p3_prereg.py battery Batch-1     # one process per battery batch
python study_p3_prereg.py ncmapss
python study_p3_prereg.py summary             # merges, tests P1-P4
```

Both of its archives passed, so a second protocol tested a decline.
`code/PREREG_p3_decline.md` was frozen on 13 October 2026, before either data
file was downloaded, with SHA-256 digest
`0e6a6801c6667061fd8edb9486f25c71952fad297237264ab2126f3e88acf890`. It predicts
a decline, and three properties of the onsets, on two unseen quarters of
Backblaze drives, analysed by `study_hdd.py` unchanged (its digest is recorded
in the protocol; `.gitattributes` keeps these files byte-exact). All eight
predictions held (`results/bearings/p3_decline.json`).

```bash
python study_p3_decline.py run Q1_2024        # reads $P3_DATA/bb_Q1_2024.zip
python study_p3_decline.py run Q2_2024
python study_p3_decline.py summary            # tests P1-P4 per quarter
```

Neither protocol measured prediction error. `code/PREREG_p3_rul.md` was frozen
on 14 October 2026, after the screen had been applied to each XJTU battery
regime but before any remaining life was predicted on those cells, with SHA-256
digest `5fb38dca8accc171ca474c2731251cb118bf639753f2e4281e331a6acc3aa8aa`. It
predicts that the path model does worse on the declined 3C cohort than on the
four regimes the screen passes, and no better than a life table on the declined
cohort. All three predictions held (`results/bearings/p3_rul.json`).

```bash
python study_p3_rul.py Batch-2                # one process per regime
python study_p3_rul.py summary                # tests R1-R3
```

The cells' verdicts were known when that protocol was written, so a fourth was
frozen before any of its data were read. `code/PREREG_p3_blind.md` (14 October
2026, SHA-256 `f85f510eab49d57bb32119e3e34b5e3513aa5b6459e6752bc1a70d4c55206a3f`)
takes every N-CMAPSS subset that no earlier analysis opened, one cohort per
subset. B1 (no cohort declined) held; B2, the primary prediction that error
rises with onset, failed; B3 and B4 were not testable. DS08d is truncated in
the archive (32 bytes short of its HDF5 length) and was excluded unread, a
reported deviation. `extract` streams the nested archive in one pass and keeps
only per-unit indicators, which are not redistributed here.

```bash
python study_p3_blind.py extract              # reads $P3_DATA/ncmapss/ncmapss.zip
python study_p3_blind.py rul
python study_p3_blind.py summary              # tests B1-B4
```

## Requirements

Python 3.10 or later with the packages in `requirements.txt`. No GPU; every
result was produced on CPU.

## Licence

MIT, see `LICENSE`. The licence covers the code and the stored results, not the
third-party data.
