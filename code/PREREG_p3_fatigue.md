# Pre-registered external test on fatigue specimens, a new asset class (paper 3)

Written and frozen before any data file below was downloaded.

## Why this test

The six earlier tests passed every cohort that took no part in building the
screen, so a decline has been reproduced only on unseen units of a class that
already declined (disk drives).  This test predicts a decline, in advance, on
an asset class that took no part in building the screen, and checks whether a
lifetime baseline then does at least as well as the path model.

## Data

* Harvard Dataverse, doi:10.7910/DVN/EXS3F5, "Strain-Based Fatigue Data for
  Ti-6Al-4V ELI under Fully-Reversed and Mean Strain Loads" (Data in Brief,
  2016; CC0 1.0).  The original workbooks `Peak-Valley R=-1.xlsx`,
  `Peak-Valley R=-0.5.xlsx`, `Peak-Valley R=0.xlsx`, `Peak-Valley R=0.5 Part
  1.xlsx` and `Peak-Valley R=0.5 Part 2.xlsx`, placed in
  `$P3_DATA/fatigue_ti64/`.  The `Cyclic data` workbooks are not used.
* From the published description only (no data file has been opened): smooth
  specimens of Ti-6Al-4V ELI in strain-controlled constant-amplitude tests at
  strain ratios R = -1, -0.5, 0 and 0.5, with strain amplitudes from 0.0015 to
  0.012; the maximum and minimum stress and strain of every cycle; tests that
  passed 10^6 cycles were run-outs; some long-life tests in the elastic region
  were switched to load control.  The article's figures were not viewed.
* The workbook layout is not described.  Before any indicator value is
  computed, the loader identifies each specimen and its cycle, maximum and
  minimum stress, and maximum and minimum strain columns from sheet names and
  column headers alone, and records that mapping in the output.

## Units, cohorts and indicator

* A unit is one specimen.  A cohort is one strain ratio, giving four cohorts.
  The pooled cohort of all specimens is reported descriptively.
* Rows with a non-finite value, or with a stress range or strain range that is
  not positive, are dropped first.
* A specimen is excluded if its last recorded cycle is 10^6 or more (a
  run-out) or if fewer than 60 rows remain.  A cohort left with fewer than
  four specimens is reported as "too few" and is not screened.
* A specimen with more than 5000 rows is averaged over consecutive
  non-overlapping blocks of ceil(n / 1000) rows before anything else is
  computed (cycle number, stresses and strains alike).
* Indicator: the secant stiffness E_k = (s_max - s_min) / (e_max - e_min), and
  h_k = 1 - E_k / E_ref, where E_ref is the median of E over the specimen's
  first min(40, max(5, n // 10)) samples.  h is the relative stiffness loss.
  The stiffness is chosen because it reads the same under strain and load
  control, so no specimen has to be excluded for a change of control mode.
* Life is the specimen's last recorded cycle, read before any averaging.  t_k
  is the (block-mean) cycle
  number, and true remaining life at sample k is life - t_k.
* The strain amplitude of a specimen is the median of (e_max - e_min) / 2
  over its samples.

## Screen

* The paper's `onset()` (study_thirdset) with its three constants, unchanged,
  applied to each complete specimen.  The onset fraction is onset_idx / n, and
  1 (censored at the end of life) when no onset is detected.
* Algorithm 3 per cohort.  If an onset is detected in fewer than a quarter of
  the specimens, the verdict is "check indicator".  Otherwise the verdict comes
  from the censored median onset and its percentile bootstrap interval over
  specimens (10 000 resamples, seed 20261023), read against 70 %: "declines",
  "not declined" or "undecided".

## Predictors

As in `PREREG_p3_filter.md`, with the strain-ratio cohort in place of the
orifice and cycles in place of seconds, except for the lifetime baseline:

* Path model: `study_bearings.predict_one`, law "BS", informed prior, Paris
  coupling, 600 particles.  dt is the specimen's median sampling interval in
  cycles.  The seed is the blake2b hash of (specimen, k) modulo 10 000.
* D: the median, over the cohort's other specimens, of the median of each
  specimen's last three indicator values.
* `med_life_steps`: the median, over the cohort's other specimens, of
  max(n - onset_idx, 20), with onset_idx = n when no onset is detected.
* Lifetime baseline: lives in a cohort span orders of magnitude with strain
  amplitude, so a plain life table would be a straw man.  The baseline is the
  life table (`study_p3_review._life_table`) over the cohort's other
  specimens with their lives rescaled to this specimen's strain amplitude:
  log life = a + b log(strain amplitude) is fitted by least squares to the
  other specimens, and each other specimen's life L_j becomes
  L_j * L_hat(this) / L_hat(j).  If the other specimens have fewer than three
  distinct strain amplitudes (amplitudes within 1 % of the next smaller one
  count as the same), the unscaled life table is used.
* Gradient boosting: the comparator of `PREREG_p3_step.md`, with its features,
  model and seed, in five folds by specimen within each cohort.  It is
  reported descriptively.
* Prediction points: k = int(f n) for f in {0.50, 0.70, 0.85}, skipping
  k < 20 or k >= n - 1.

## Predictions

* W1 (primary): at least one cohort is declined, and no cohort is "not
  declined".  In strain-controlled tests on smooth specimens, crack initiation
  takes most of the life and the stiffness is expected to hold until a
  macrocrack grows, late in life.  The known risk is cyclic softening: if the
  detector reads it as a departure, onsets come early and W1 fails.
* W2: in every screened cohort an onset is detected in at least a quarter of
  the specimens (the indicator check passes).
* W3 (conditional): if at least one cohort is declined, on the declined
  cohorts the path model does not beat the lifetime baseline: the bootstrap
  interval of the difference in median absolute percentage error (path model
  minus baseline), clustered on the specimen (10 000 resamples, seed
  20261023), does not lie wholly below 0.  Otherwise W3 is not testable.
* W4 (unit level): if at least five specimens have onset at 0.70 or later and
  at least five earlier, the path model's coverage of its 95 % interval is
  lower in the late group, with the interval of the difference wholly below 0.
  Otherwise W4 is not testable.

## Reporting

Every outcome is reported whatever it is, with per-cohort verdicts, onsets and
errors for all three predictors.  If no cohort is declined, the test does not
validate a decline on a new asset class, and the paper says so.  Any deviation
is reported as such.  `study_p3_fatigue.py` refuses to run if this file's
SHA-256 differs from the value recorded in it.
