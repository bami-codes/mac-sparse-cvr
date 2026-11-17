# Agent test error rates under the operating gate (synthetic data, known answer)

`scripts/agent_bootstrap_check.py --gate operating`, gate per compared arm {'min_rows': 200, 'min_positives': 312, 'min_users': 2}, seed 20261118, 2000 simulations per scale, alpha 0.05. Pass: null rejection <= 0.065 and 95% CI coverage >= 0.935 (decision rule in the script).
Clicks per user from the development menu (522,922 users, mean 12.86). True alternative difference -0.002367 (Monte Carlo s.e. 7.4e-06).

| rows | tool | computable | median positives / users / converting users (smallest arm) | rejection rate [Wilson 95%] | CI coverage | pass |
|---:|---|---:|---|---|---:|---|
| 700 | ablation_effect (null) | 0 | 11 / 57 / 8 | - | - | - |
| 700 | interaction (null) | 0 | 4 / 50 / 3 | - | - | - |
| 700 | association (null) | 0 | 3 / 39 / 3 | - | - | - |
| 700 | ablation_effect (alternative) | 0 | 11 / 57 / 8 | - | - | - |
| 2,000 | ablation_effect (null) | 0 | 31 / 159 / 22 | - | - | - |
| 2,000 | interaction (null) | 0 | 14 / 139 / 11 | - | - | - |
| 2,000 | association (null) | 0 | 9 / 106 / 8 | - | - | - |
| 2,000 | ablation_effect (alternative) | 0 | 31 / 159 / 22 | - | - | - |
| 7,000 | ablation_effect (null) | 0 | 109 / 548 / 76 | - | - | - |
| 7,000 | interaction (null) | 0 | 50 / 478 / 41 | - | - | - |
| 7,000 | association (null) | 0 | 33 / 366 / 27 | - | - | - |
| 7,000 | ablation_effect (alternative) | 0 | 109 / 548 / 76 | - | - | - |
| 20,000 | ablation_effect (null) | 1003 | 312 / 1561 / 218 | 0.050 [0.038, 0.065] | 0.947 | True |
| 20,000 | interaction (null) | 0 | 149 / 1361 / 120 | - | - | - |
| 20,000 | association (null) | 0 | 93 / 1042 / 77 | - | - | - |
| 20,000 | ablation_effect (alternative) | 1003 | 312 / 1561 / 218 | 1.000 [0.996, 1.000] | 0.967 | True |
| 100,000 | ablation_effect (null) | 2000 | 1561 / 7776 / 1090 | 0.061 [0.052, 0.073] | 0.936 | True |
| 100,000 | interaction (null) | 2000 | 764 / 6786 / 613 | 0.050 [0.041, 0.060] | 0.950 | True |
| 100,000 | association (null) | 2000 | 468 / 5189 / 384 | 0.058 [0.049, 0.069] | 0.942 | True |
| 100,000 | ablation_effect (alternative) | 2000 | 1561 / 7776 / 1090 | 1.000 [0.998, 1.000] | 0.954 | True |

For the alternative row, the rejection rate is the power. Medians are over all simulations, computable or not.

Error rates here are among simulations that pass the operating gate (computable column). This describes the rule actually used on this synthetic configuration; it is not a general sample-size guarantee, and no threshold is chosen from it.
