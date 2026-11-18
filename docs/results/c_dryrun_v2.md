# Study C final evaluation (dryrun): train-12.parquet

- evaluation code `4e03dea`
- freeze.json `32011a9a6b716a91`; files {'train-12.parquet': 'bc88690125f6defd5208b6106d473fceb16c613b3da2f9049a7c8e2527bbecb1'}; rows 3544371 [3544371]
- primary: G-shared (iii-d) - G-int (iii-d), lambda chosen on V2 per family on small-domain macro LogLoss (equal weight over small domains), lower is better
- **difference -0.000018, 95% CI [-0.000145, +0.000105], per seed pair [-4.9e-05, 3.7e-05, -4.2e-05] -> tier: not_confirmed**
- conditions: {"effect_size": {"relative_improvement": 0.0004514405419930894, "needed": 0.003, "met": false}, "large_domains": {"ci95_upper": 1.0101125099339864e-05, "needed_at_most": 0.0001, "met": true}, "latency": {"p99_ms_max_over_seeds": 0.10529239370953293, "p99_ms_by_seed": {"2026": 0.10529239370953293, "2027": 0.08833759668050344, "2028": 0.09641949320212012}, "needed_at_most": 1.0, "met": true}}
- large domains (click-weighted) -0.000008 [-0.000026, +0.000010]; small within-domain AUC -0.00746 [-0.01705, +0.00184] (22 domains)
- leave one small domain out: range [-0.000056, +0.000017], sign flips 4

| family | small macro LogLoss | small click LogLoss | small Poisson deviance (group totals) | large click LogLoss | small within-domain AUC | p99 ms, max over seeds (uniform traffic) |
|---|---|---|---|---|---|---|
| G-int | 0.040134 | 0.034574 | 1.7344 | 0.069611 | 0.7182 | 0.21 |
| G-shared | 0.040116 | 0.034547 | 1.7286 | 0.069603 | 0.7107 | 0.105 |
| N-int | 0.040359 | 0.034663 | 1.7062 | 0.069850 | 0.7028 | 0.176 |
| N-shared | 0.040352 | 0.034665 | 1.7080 | 0.069817 | 0.7039 | 0.185 |
| N-head | 0.040328 | 0.034691 | 1.7427 | 0.069821 | 0.7068 | 0.314 |
| N-star | 0.040875 | 0.034991 | 1.6328 | 0.069857 | 0.6816 | 0.776 |
| N-star-noPN | 0.040222 | 0.034556 | 1.7408 | 0.069761 | 0.7002 | – |

| secondary (A - B) | small macro [95% CI] | large click [95% CI] |
|---|---|---|
| N-star - N-shared | +0.000523 [+0.000203, +0.000859] | +0.000040 [-0.000004, +0.000082] |
| N-head - N-shared | -0.000024 [-0.000107, +0.000063] | +0.000004 [-0.000007, +0.000014] |
| N-star-noPN - N-star | -0.000653 [-0.000972, -0.000389] | -0.000096 [-0.000122, -0.000071] |
| N-head - G-shared | +0.000213 [-0.000048, +0.000475] | +0.000218 [+0.000167, +0.000267] |
| G-shared: iiid - ii | -0.000057 [-0.000197, +0.000082] | -0.000014 [-0.000022, -0.000007] |
| G-int: iiid - ii | -0.000106 [-0.000233, +0.000028] | -0.000018 [-0.000026, -0.000009] |

Intervals hold the fitted models fixed (evaluation-user uncertainty only); users seen on both files are one cluster.
