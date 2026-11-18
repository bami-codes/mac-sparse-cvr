# Study B: auxiliary attribution label vs a same-label control (development data)

Run `runs/p2_train00_02/20261117T131443Z` (train-00..02, V1 train-03, V2 train-04); seeds [2026, 2027, 2028]. Design: docs/DESIGN.md §15.2. All variants after (ii) global calibration fitted on V1; metric V2 G1 macro LogLoss. M2both was chosen on this V2, so comparisons with M2both are optimistic.

Checks: refit of the saved mta auxiliary model on the rebuilt training rows: {'refit_rounds': 174, 'saved_rounds': 174, 'max_abs_pred_diff': 0.0}; 12 stored V2 values recomputed within 1e-09; group-rate inputs equal up to 2.0e-10 (saved run's floating-point drift).

| variant | inputs added to M2 | V2 G1 macro LogLoss (ii), seed mean |
|---|---|---:|
| M2 | - | 0.064346 |
| M2aux | out-of-fold prediction of mta > 0 | 0.064257 |
| M2te | group rates | 0.064264 |
| M2both | mta prediction + group rates | 0.064134 |
| M2auxlast | out-of-fold prediction of last > 0 (control) | 0.064332 |
| M2bothlast | last prediction + group rates (control) | 0.064204 |

| comparison | role | difference A - B [95% CI] | per seed pair |
|---|---|---|---|
| M2aux - M2auxlast | core: same procedure, label mta vs last | -0.000075 [-0.000169, +0.000021] | -0.000041, -0.000104, -0.000079 |
| M2both - M2bothlast | core: same procedure, label mta vs last (with group rates) | -0.000070 [-0.000165, +0.000026] | -0.000080, -0.000052, -0.000078 |
| M2auxlast - M2 | reference: adding a same-label out-of-fold prediction | -0.000014 [-0.000090, +0.000054] | -0.000044, +0.000050, -0.000049 |
| M2aux - M2 | reference: the second study's mta input | -0.000089 [-0.000182, +0.000002] | -0.000085, -0.000054, -0.000128 |

Intervals resample V2 users with the fitted models held fixed (seed-paired); they do not include training variation.
