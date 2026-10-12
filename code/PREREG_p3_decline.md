# Pre-registered test of a decline: the screen on two unseen quarters of Backblaze drives

Protocol for paper 3, "A Screening Rule for Degradation-Threshold Prognostics".
Written and frozen before either data file was downloaded. Once frozen, its
SHA-256 digest is recorded in `code/study_p3_decline.py`, which refuses to run
on any other version of this file.

## Why this test

The first pre-registered test (`PREREG_p3_new_archives.md`) confirmed two
verdicts of "not declined" but could not test a decline. The one asset class
the screen has declined outside the bearings is the hard disk drive: on
Backblaze's fourth quarter of 2023 the censored median onset was not reached,
with a bootstrap interval starting at 99.3 % of life. This test asks whether
that decline replicates on new drives, under a protocol fixed in advance. It
tests a decline on new units of one asset class, not on a new asset class.

## What had been seen when this protocol was frozen

No file of either quarter below had been downloaded or opened. The only
Backblaze data analysed before is the fourth quarter of 2023 (`bb_Q4_2023.zip`),
whose results set the expectations below.

## Data

Two quarterly archives of Backblaze's public drive statistics, each one cohort,
downloaded from Backblaze's own server and checked against these sizes:

| cohort | file | source | size (bytes) |
|---|---|---|---|
| Q1_2024 | `data_Q1_2024.zip`, saved as `bb_Q1_2024.zip` | `https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2024.zip` | 1006502769 |
| Q2_2024 | `data_Q2_2024.zip`, saved as `bb_Q2_2024.zip` | `https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2024.zip` | 1020665037 |

## Analysis, fixed

Everything up to the per-drive onsets is `code/study_hdd.py`, unchanged, whose
SHA-256 digest at freezing is

    b4c4e520467882a241d5dbc7c16e232a13ba5b7e3c62d1836467c97c127af74f

`study_p3_decline.py run <cohort>` executes that source with only its output
file name replaced. In words:

* Drives: every drive with a failure flag in the quarter that has at least 60
  daily samples, all five counters at zero on its first sample, and valid,
  increasing power-on hours.
* Indicator: SMART 5, 187, 188, 197 and 198 (raw), each as a robust z-score
  against the drive's own healthy window (the first max(5, min(40, n/10))
  samples), summed.
* Onset detector: the paper's, constants unchanged: band 3 robust standard
  deviations, a run of 10 consecutive samples, and at least 80 % of the rest of
  the record above the band.
* Life: power-on hours (SMART 9); onset fraction = age at onset / age at the
  last sample.

Per cohort, `study_p3_decline.py summary` then computes:

* the detection rate, onsets detected / usable drives;
* the censored median onset, a drive without a detected onset counted as an
  onset beyond the end of its life ("not reached" when fewer than half show
  one);
* its 95 % percentile bootstrap interval over drives, 10 000 resamples, numpy
  `default_rng(20261014)`, cohorts in the order Q1_2024 then Q2_2024, quantiles
  by the `inverted_cdf` method;
* the verdict of Algorithm 3: declines if the interval lies at or above 70 % of
  life, not declined if it lies wholly below, undecided otherwise.

## Predictions

For each cohort:

* **P1 (the decline).** The screen declines the first-passage model: the lower
  end of the censored interval is at or above 70 % of life.
* **P2 (the indicator check passes).** An onset is detected in at least a
  quarter of the usable drives, so step 4 of the screen does not stop it.
* **P3 (onsets are late).** The median detected onset is at or above 90 % of
  life.
* **P4 (none in the decision band).** Fewer than 10 % of the detected onsets
  fall before 70 % of life.

The test of a decline is passed if P1 holds in both cohorts. P2 to P4 are
reported for each cohort whatever their outcome.

## Contingencies, fixed in advance

* A download whose size differs from the table is repeated once; a second
  failure is reported, and that cohort is not tested.
* A cohort with fewer than 30 usable drives is reported as not tested, not as
  a verdict.
* No constant of the detector, the indicator, the selection or the rule is
  changed after freezing. Any deviation from this protocol is reported with the
  result.
