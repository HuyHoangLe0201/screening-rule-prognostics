# Pre-registered blind test on unseen N-CMAPSS subsets (paper 3)

Written and frozen before any of the data below was extracted, opened or
analysed.

## Why this test

The third protocol (`PREREG_p3_rul.md`) tested prediction error on cohorts
whose screening verdicts were already known.  This protocol fixes the cohorts,
the screen, the predictors and the predictions before either the onsets or the
remaining-life predictions of any unit exist.

## Status at freezing

* `data/ncmapss/ncmapss.zip` holds `data_set.zip`, which holds every N-CMAPSS
  subset.  Only `N-CMAPSS_DS01-005.h5` and `N-CMAPSS_DS02-006.h5` have been
  extracted or analysed, under `PREREG_p3_new_archives.md`.
* No other subset has been extracted, opened or analysed.  Not even the list of
  file names inside `data_set.zip` has been read: one attempt to list it timed
  out without output.

## Data and cohorts

* Every `.h5` file inside `data_set.zip` whose name does not begin with
  `N-CMAPSS_DS01` or `N-CMAPSS_DS02`.
* Each file is one cohort: one subset, with its development and test units
  pooled.  A unit is keyed by file, split and unit number.
* Each file is streamed out of the archive, reduced to the per-cycle indicator
  below, and deleted before the next is read.

## Indicator and onset

Exactly as for N-CMAPSS under `PREREG_p3_new_archives.md`, that is, as in
`study_p3_prereg.run_ncmapss`:

* per unit, a quadratic regression of the sensors on the four flight
  descriptors, fitted on the unit's healthy window of min(40, max(5, n // 10))
  cycles;
* the per-cycle means of the residuals;
* the paper's `indicator()` and `onset()`.

The onset fraction is onset_idx / n, and 1 (censored at the end of life) when
no onset is detected.

## Screen

Algorithm 3 on each cohort:

* If an onset is detected in fewer than a quarter of the units, the verdict is
  "check indicator".
* Otherwise the verdict comes from the censored median onset and its percentile
  bootstrap interval over units (10 000 resamples, seed 20261018), read against
  70 %: "declines" if the interval lies at or above 70 %, "not declined" if
  wholly below, "undecided" otherwise.

## Remaining life

As in `PREREG_p3_rul.md`, with the cohort in place of the regime:

* Path model: `study_bearings.predict_one`, law "BS", informed prior, Paris
  coupling, 600 particles.  The seed is the blake2b hash of (unit key, k)
  modulo 10 000.
* D is the median, over the cohort's other units, of the median of each unit's
  last three indicator values.
* `med_life_steps` is the median, over the other units, of
  max(n - onset_idx, 20), with onset_idx = n when no onset is detected.
* Life table: `study_p3_review._life_table` over the cohort's other units.
* Prediction points: k = int(f n) for f in {0.50, 0.70, 0.85}.  A point is
  skipped if k < 20 or k >= n - 1.  True RUL = (n - 1) - k cycles.

## Predictions

* B1: no cohort is declined.
* B2 (primary): across units, the path model's mean absolute percentage error
  over the unit's prediction points rises with the unit's onset fraction.  The
  Spearman correlation is positive, and its percentile bootstrap 95 % interval
  over units (10 000 resamples, seed 20261018) lies wholly above 0.
* B3: units whose onset fraction is 0.70 or more have a higher path-model
  median absolute percentage error than units below 0.70.  The bootstrap
  interval of the difference, clustered on the unit, must lie wholly above 0.
  B3 is tested only if each group has at least five units; otherwise it is
  reported as not testable.
* B4 (conditional): if at least one cohort is declined and at least one is not
  declined, then two things must hold.  The path model's median absolute
  percentage error over the declined cohorts exceeds that over the not-declined
  ones.  And following the screen (life table on declined cohorts, path model
  elsewhere) gives a lower pooled median absolute percentage error than always
  using the path model.  Both intervals, clustered on the unit, must lie wholly
  on the predicted side of 0.  Otherwise B4 is reported as not testable.

## Reporting

Every outcome is reported whatever it is, with per-cohort verdicts, onsets and
errors.  A failed B2 counts against the paper's fourth claim.  Any deviation
from this protocol is reported as a deviation.  The analysis script
`study_p3_blind.py` refuses to run if this file's SHA-256 differs from the value
recorded in the script.
