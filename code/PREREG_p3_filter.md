# Pre-registered external test on filters, a new asset class (paper 3)

Written and frozen before the data file below was downloaded.

## Why this test

Every earlier test of the screen used bearings, turbofan engines, lithium-ion
cells or disk drives.  The reviewers ask for the screen's verdict, fixed in
advance, on an asset class that took no part in building it, and for a check
that the verdict marks how well a first-passage model predicts there.

## Data

* Kaggle, user "prognosticshse": "Similar System Data Set for Condition
  Prognosis" (Braig and Zeiler, IEEE Access 13, 2025; CC BY 4.0), file
  `data.csv`.
* From its published description only (the data file has not been opened):
  clogging of non-woven filter media by ISO 12103-1 A3 dust under flow
  control; ten run-to-failure runs for each of four orifice sizes (100, 87.5,
  62.5 and 37.5 % of the open filtration area); failure when the differential
  pressure exceeds 600 Pa; columns Time (s), Differential_Pressure (Pa),
  Flow_Rate (l/min), RUL (s), Orifice and Data_No.

## Units, cohorts and indicator

* A unit is one run (`Data_No`).  A cohort is one orifice size, giving four
  cohorts.  The pooled cohort of all runs is reported descriptively.
* A run with fewer than 60 samples is excluded.  A run with more than 5000
  samples is averaged over consecutive non-overlapping blocks of
  ceil(n / 1000) samples before anything else is computed.
* Indicator: h_k = DP_k minus the median of DP over the run's first
  min(40, max(5, n // 10)) samples, where DP is the differential pressure.
* Life is the time of the run's last sample plus its RUL there.  True RUL at
  sample k is the value in the RUL column (after block averaging, the block
  mean).

## Screen

* The paper's `onset()` (study_thirdset) with its three constants, applied to
  each complete run.  The onset fraction is onset_idx / n, and 1 (censored at
  the end of life) when no onset is detected.
* Algorithm 3 per cohort.  If an onset is detected in fewer than a quarter of
  the runs, the verdict is "check indicator".  Otherwise the verdict comes from
  the censored median onset and its percentile bootstrap interval over runs
  (10 000 resamples, seed 20261022), read against 70 %: "declines",
  "not declined" or "undecided".

## Predictors

As in `PREREG_p3_rul.md`, with the cohort in place of the regime and seconds in
place of cycles:

* Path model: `study_bearings.predict_one`, law "BS", informed prior, Paris
  coupling, 600 particles.  t is the Time column, and dt is the run's median
  sampling interval.  The seed is the blake2b hash of (run, k) modulo 10 000.
* D: the median, over the cohort's other runs, of the median of each run's
  last three indicator values.
* `med_life_steps`: the median, over the cohort's other runs, of
  max(n - onset_idx, 20), with onset_idx = n when no onset is detected.
* Life table: `study_p3_review._life_table` over the cohort's other runs, with
  their lives in seconds.
* Gradient boosting: the comparator of `PREREG_p3_step.md`, with its features,
  model and seed, in five folds by run within each cohort.  It is reported
  descriptively.
* Prediction points: k = int(f n) for f in {0.50, 0.70, 0.85}, skipping
  k < 20 or k >= n - 1.

## Predictions

* V1: no cohort is declined.  A depth filter's pressure drop rises from the
  start of loading, so onsets are expected early.  This is the only
  prediction about verdicts.
* F1 (primary, conditional): if at least one cohort is declined and at least
  one is not declined, the path model's median absolute percentage error is
  higher on the declined cohorts.  The bootstrap interval of the difference,
  clustered on the run (10 000 resamples, seed 20261022), must lie wholly above
  0.  Otherwise F1 is reported as not testable.
* F2 (conditional, as F1): coverage of the 95 % interval is lower on the
  declined cohorts, with the interval of the difference wholly below 0.
* F3 (unit level): if at least five runs have onset at 0.70 or later and at
  least five earlier, the path model's coverage is lower in the late group,
  with the interval of the difference wholly below 0.  Otherwise F3 is not
  testable.

## Reporting

Every outcome is reported whatever it is, with per-cohort verdicts, onsets and
errors for all three predictors.  If no cohort is declined, the test does not
validate a decline, and the paper says so.  Any deviation is reported as such.
`study_p3_filter.py` refuses to run if this file's SHA-256 differs from the
value recorded in it.
