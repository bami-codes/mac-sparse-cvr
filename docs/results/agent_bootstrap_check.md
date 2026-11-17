# Agent test error rates on synthetic data with a known answer

`scripts/agent_bootstrap_check.py --gate diagnostic`, gate per compared arm {'min_rows': 200, 'min_positives': 10, 'min_users': 1}, seed 20261118, 2000 simulations per scale, alpha 0.05. Pass: null rejection <= 0.065 and 95% CI coverage >= 0.935 (decision rule in the script).
Clicks per user from the development menu (522,922 users, mean 12.86). True alternative difference -0.002367 (Monte Carlo s.e. 7.4e-06).

| rows | tool | computable | median positives / users / converting users (smallest arm) | rejection rate [Wilson 95%] | CI coverage | pass |
|---:|---|---:|---|---|---:|---|
| 700 | ablation_effect (null) | 1275 | 11 / 57 / 8 | 0.087 [0.073, 0.104] | 0.910 | False |
| 700 | interaction (null) | 25 | 4 / 50 / 3 | 0.040 [0.007, 0.195] | 0.960 | True |
| 700 | association (null) | 9 | 3 / 39 / 3 | 0.111 [0.020, 0.435] | 0.778 | False |
| 700 | ablation_effect (alternative) | 1275 | 11 / 57 / 8 | 0.171 [0.151, 0.193] | 0.965 | True |
| 2,000 | ablation_effect (null) | 2000 | 31 / 159 / 22 | 0.079 [0.068, 0.091] | 0.919 | False |
| 2,000 | interaction (null) | 1763 | 14 / 139 / 11 | 0.062 [0.052, 0.074] | 0.936 | True |
| 2,000 | association (null) | 955 | 9 / 106 / 8 | 0.050 [0.038, 0.066] | 0.948 | True |
| 2,000 | ablation_effect (alternative) | 2000 | 31 / 159 / 22 | 0.336 [0.316, 0.357] | 0.922 | False |
| 7,000 | ablation_effect (null) | 2000 | 109 / 548 / 76 | 0.073 [0.063, 0.086] | 0.925 | False |
| 7,000 | interaction (null) | 2000 | 50 / 478 / 41 | 0.051 [0.042, 0.061] | 0.949 | True |
| 7,000 | association (null) | 2000 | 33 / 366 / 27 | 0.057 [0.047, 0.067] | 0.941 | True |
| 7,000 | ablation_effect (alternative) | 2000 | 109 / 548 / 76 | 0.850 [0.834, 0.865] | 0.952 | True |
| 20,000 | ablation_effect (null) | 2000 | 312 / 1561 / 218 | 0.051 [0.042, 0.062] | 0.947 | True |
| 20,000 | interaction (null) | 2000 | 149 / 1361 / 120 | 0.058 [0.049, 0.069] | 0.939 | True |
| 20,000 | association (null) | 2000 | 93 / 1042 / 77 | 0.058 [0.049, 0.069] | 0.940 | True |
| 20,000 | ablation_effect (alternative) | 2000 | 312 / 1561 / 218 | 1.000 [0.997, 1.000] | 0.946 | True |
| 100,000 | ablation_effect (null) | 2000 | 1561 / 7776 / 1090 | 0.061 [0.052, 0.073] | 0.936 | True |
| 100,000 | interaction (null) | 2000 | 764 / 6786 / 613 | 0.050 [0.041, 0.060] | 0.950 | True |
| 100,000 | association (null) | 2000 | 468 / 5189 / 384 | 0.058 [0.049, 0.069] | 0.942 | True |
| 100,000 | ablation_effect (alternative) | 2000 | 1561 / 7776 / 1090 | 1.000 [0.998, 1.000] | 0.954 | True |

For the alternative row, the rejection rate is the power. Medians are over all simulations, computable or not.

**Decision**: every tool passes from 20,000 rows on (medians of the smallest arm there: ablation_effect (null) 312, interaction (null) 149, association (null) 93, ablation_effect (alternative) 312). 4 results failed above the old minimum of 10 positives, so the minimum per compared arm becomes **312 positives** (largest median, one constant for every arm).
