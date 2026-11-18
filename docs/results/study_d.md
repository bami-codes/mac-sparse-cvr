# Study D: intercept-only group shrinkage (iii-0) vs intercept+slope (iii-d) vs (ii)

Backtest on files opened before (docs/DESIGN.md §17); not a confirmation. Differences are A - B; seed-paired user bootstrap, models and calibrators fixed.

- verdict (setting C, G-int, train-13 + 14): {'logloss_gain_retained': False, 'group_count_error_reduced': True, 'rule': 'retain: dLL 95% CI upper < 5e-05; reduce: dPD < 0, CI upper < 0, all 3 seed pairs < 0'}

## Setting C, G-int

- lambda (main rule = V2 macro LogLoss): {'iiid': 1.0, 'iii0': 100.0}; aggregate rule (V2 Poisson deviance): {'iiid': 10000.0, 'iii0': 100.0}

V2 (selection data) scores of (ii) and of the lambdas chosen by each rule:

| rule | method | lambda | V2 macro LogLoss | V2 click LogLoss | V2 per-group Poisson deviance | V2 pooled O/E |
|---|---|---:|---:|---:|---:|---:|
| – | ii | – | 0.040239 | 0.034595 | 1.1069 | 1.038 |
| main | iiid | 1 | 0.040134 | 0.034574 | 1.7344 | 0.980 |
| main | iii0 | 100 | 0.040236 | 0.034593 | 1.0644 | 1.016 |
| aggregate | iiid | 10000 | 0.040238 | 0.034592 | 1.0621 | 1.028 |
| aggregate | iii0 | 100 | 0.040236 | 0.034593 | 1.0644 | 1.016 |

Evaluation files:

| rule | method | macro LogLoss | click LogLoss | per-group Poisson deviance | pooled O/E |
|---|---|---:|---:|---:|---:|
| main | ii | 0.038094 | 0.033268 | 1.9584 | 1.011 |
| main | iiid | 0.037947 | 0.033265 | 3.7550 | 0.950 |
| main | iii0 | 0.038083 | 0.033267 | 1.9124 | 0.988 |
| aggregate | ii | 0.038094 | 0.033268 | 1.9584 | 1.011 |
| aggregate | iiid | 0.038092 | 0.033266 | 1.9027 | 1.001 |
| aggregate | iii0 | 0.038083 | 0.033267 | 1.9124 | 0.988 |

| comparison (main lambda) | macro [95% CI] | click [95% CI] | Poisson deviance [95% CI] | per seed pair (Poisson) |
|---|---|---|---|---|
| iii0 - iiid | +0.000137 [+0.000050, +0.000238] | +0.000002 [-0.000024, +0.000030] | -1.8425 [-2.8719, -0.7636] | [-1.9288, -1.784, -1.8147] |
| iii0 - ii | -0.000010 [-0.000020, +0.000001] | -0.000001 [-0.000009, +0.000008] | -0.0460 [-0.4603, +0.4024] | [0.0362, -0.1806, 0.0066] |
| iiid - ii | -0.000147 [-0.000253, -0.000054] | -0.000003 [-0.000036, +0.000028] | +1.7966 [+0.4095, +3.1648] | [1.965, 1.6033, 1.8213] |
| iii0 - iiid, file 0 | +0.000102 [-0.000025, +0.000223] | -0.000014 | -1.2659 [-1.9762, -0.5217] | |
| iii0 - iiid, file 1 | +0.000170 [+0.000030, +0.000327] | +0.000016 | -0.5841 [-1.3190, +0.2043] | |

## Setting C, G-shared

- lambda (main rule = V2 macro LogLoss): {'iiid': 1.0, 'iii0': 10000.0}; aggregate rule (V2 Poisson deviance): {'iiid': 10000.0, 'iii0': 10000.0}

V2 (selection data) scores of (ii) and of the lambdas chosen by each rule:

| rule | method | lambda | V2 macro LogLoss | V2 click LogLoss | V2 per-group Poisson deviance | V2 pooled O/E |
|---|---|---:|---:|---:|---:|---:|
| – | ii | – | 0.040173 | 0.034558 | 0.9876 | 1.011 |
| main | iiid | 1 | 0.040116 | 0.034547 | 1.7286 | 0.980 |
| main | iii0 | 10000 | 0.040173 | 0.034558 | 0.9876 | 1.011 |
| aggregate | iiid | 10000 | 0.040173 | 0.034558 | 0.9914 | 1.005 |
| aggregate | iii0 | 10000 | 0.040173 | 0.034558 | 0.9876 | 1.011 |

Evaluation files:

| rule | method | macro LogLoss | click LogLoss | per-group Poisson deviance | pooled O/E |
|---|---|---:|---:|---:|---:|
| main | ii | 0.038062 | 0.033295 | 1.7957 | 0.983 |
| main | iiid | 0.037974 | 0.033303 | 3.8256 | 0.950 |
| main | iii0 | 0.038062 | 0.033295 | 1.7961 | 0.982 |
| aggregate | ii | 0.038062 | 0.033295 | 1.7957 | 0.983 |
| aggregate | iiid | 0.038062 | 0.033296 | 1.8481 | 0.977 |
| aggregate | iii0 | 0.038062 | 0.033295 | 1.7961 | 0.982 |

| comparison (main lambda) | macro [95% CI] | click [95% CI] | Poisson deviance [95% CI] | per seed pair (Poisson) |
|---|---|---|---|---|
| iii0 - iiid | +0.000088 [-0.000015, +0.000202] | -0.000008 [-0.000037, +0.000022] | -2.0295 [-3.2214, -0.7903] | [-2.3486, -2.0309, -1.709] |
| iii0 - ii | -0.000000 [-0.000000, +0.000000] | +0.000000 [-0.000000, +0.000000] | +0.0003 [-0.0067, +0.0079] | [0.0017, -0.0006, -0.0001] |
| iiid - ii | -0.000088 [-0.000202, +0.000015] | +0.000008 [-0.000022, +0.000037] | +2.0298 [+0.7884, +3.2236] | [2.3503, 2.0303, 1.7089] |
| iii0 - iiid, file 0 | +0.000032 [-0.000101, +0.000151] | -0.000026 | -1.3676 [-2.1662, -0.5480] | |
| iii0 - iiid, file 1 | +0.000143 [-0.000019, +0.000327] | +0.000010 | -0.6688 [-1.5159, +0.2282] | |

## Setting 1, M2

- lambda (main rule = V2 macro LogLoss): {'iiid': 100.0, 'iii0': 10.0}; aggregate rule (V2 Poisson deviance): {'iiid': 100.0, 'iii0': 10.0}

V2 (selection data) scores of (ii) and of the lambdas chosen by each rule:

| rule | method | lambda | V2 macro LogLoss | V2 click LogLoss | V2 per-group Poisson deviance | V2 pooled O/E |
|---|---|---:|---:|---:|---:|---:|
| – | ii | – | 0.063198 | 0.053379 | 1.2908 | 0.989 |
| main | iiid | 100 | 0.063083 | 0.053295 | 1.2255 | 0.983 |
| main | iii0 | 10 | 0.063097 | 0.053290 | 1.2213 | 0.985 |
| aggregate | iiid | 100 | 0.063083 | 0.053295 | 1.2255 | 0.983 |
| aggregate | iii0 | 10 | 0.063097 | 0.053290 | 1.2213 | 0.985 |

Evaluation files:

| rule | method | macro LogLoss | click LogLoss | per-group Poisson deviance | pooled O/E |
|---|---|---:|---:|---:|---:|
| main | ii | 0.064480 | 0.055406 | 1.3081 | 1.046 |
| main | iiid | 0.064380 | 0.055325 | 1.2401 | 1.039 |
| main | iii0 | 0.064386 | 0.055318 | 1.2399 | 1.041 |
| aggregate | ii | 0.064480 | 0.055406 | 1.3081 | 1.046 |
| aggregate | iiid | 0.064380 | 0.055325 | 1.2401 | 1.039 |
| aggregate | iii0 | 0.064386 | 0.055318 | 1.2399 | 1.041 |

| comparison (main lambda) | macro [95% CI] | click [95% CI] | Poisson deviance [95% CI] | per seed pair (Poisson) |
|---|---|---|---|---|
| iii0 - iiid | +0.000005 [-0.000018, +0.000029] | -0.000006 [-0.000026, +0.000015] | -0.0002 [-0.0144, +0.0152] | [-0.0005, 0.0001, -0.0003] |
| iii0 - ii | -0.000094 [-0.000137, -0.000057] | -0.000088 [-0.000122, -0.000055] | -0.0682 [-0.0950, -0.0423] | [-0.0669, -0.0682, -0.0695] |
| iiid - ii | -0.000100 [-0.000157, -0.000050] | -0.000081 [-0.000128, -0.000035] | -0.0680 [-0.1064, -0.0327] | [-0.0664, -0.0683, -0.0692] |

