# Agent test error rates on synthetic data with a known answer

`scripts/agent_bootstrap_check.py`, seed 20261118, 2000 simulations per scale, alpha 0.05. Pass: null rejection <= 0.065 and 95% CI coverage >= 0.935 (decision rule in the script).
Clicks per user from the development menu (522,922 users, mean 12.86). True alternative difference -0.002367 (Monte Carlo s.e. 7.4e-06).

| rows | tool | computable | median positives (smallest arm) | rejection rate [Wilson 95%] | CI coverage | pass |
|---:|---|---:|---:|---|---:|---|
| 700 | ablation_effect (null) | 1292 | 11 | 0.074 [0.061, 0.090] | 0.924 | False |
| 700 | interaction (null) | 36 | 4 | 0.250 [0.138, 0.411] | 0.750 | False |
| 700 | association (null) | 12 | 3 | 0.250 [0.089, 0.532] | 0.750 | False |
| 700 | ablation_effect (alternative) | 1292 | 11 | 0.190 [0.169, 0.212] | 0.953 | True |
| 2,000 | ablation_effect (null) | 2000 | 31 | 0.072 [0.061, 0.084] | 0.928 | False |
| 2,000 | interaction (null) | 1746 | 13 | 0.078 [0.067, 0.092] | 0.921 | False |
| 2,000 | association (null) | 921 | 9 | 0.045 [0.033, 0.060] | 0.954 | True |
| 2,000 | ablation_effect (alternative) | 2000 | 31 | 0.344 [0.323, 0.365] | 0.925 | False |
| 7,000 | ablation_effect (null) | 2000 | 109 | 0.066 [0.056, 0.078] | 0.932 | False |
| 7,000 | interaction (null) | 2000 | 51 | 0.061 [0.051, 0.072] | 0.938 | True |
| 7,000 | association (null) | 2000 | 33 | 0.057 [0.047, 0.067] | 0.942 | True |
| 7,000 | ablation_effect (alternative) | 2000 | 109 | 0.857 [0.840, 0.871] | 0.943 | True |
| 20,000 | ablation_effect (null) | 2000 | 312 | 0.053 [0.044, 0.064] | 0.945 | True |
| 20,000 | interaction (null) | 2000 | 149 | 0.049 [0.040, 0.059] | 0.949 | True |
| 20,000 | association (null) | 2000 | 94 | 0.059 [0.049, 0.070] | 0.941 | True |
| 20,000 | ablation_effect (alternative) | 2000 | 312 | 1.000 [0.997, 1.000] | 0.941 | True |
| 100,000 | ablation_effect (null) | 2000 | 1557 | 0.057 [0.048, 0.068] | 0.941 | True |
| 100,000 | interaction (null) | 2000 | 763 | 0.054 [0.045, 0.065] | 0.944 | True |
| 100,000 | association (null) | 2000 | 466 | 0.053 [0.044, 0.064] | 0.946 | True |
| 100,000 | ablation_effect (alternative) | 2000 | 1557 | 1.000 [0.998, 1.000] | 0.935 | True |

For the alternative row, the rejection rate is the power.

**Decision**: every tool passes from 20,000 rows on (medians of the smallest arm there: ablation_effect (null) 312, interaction (null) 149, association (null) 94, ablation_effect (alternative) 312). 5 results failed above the old minimum of 10 positives, so the minimum per compared arm becomes **312 positives** (largest median, one constant for every arm).
