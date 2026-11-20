# Study C final evaluation (test): train-13.parquet, train-14.parquet

- evaluation code `e08e980`
- freeze.json `32011a9a6b716a91`; files {'train-13.parquet': '2a0fc1c9dbda6d0dbb818d769fade5c3b7a9a43267daa9f7bb714dce6b74c081', 'train-14.parquet': 'db1e0768f76c3480bce1823f48c43b259229aced671b49810d25b1b92bba11a8'}; rows 7002260 [3482362, 3519898]
- primary: G-shared (iii-d) - G-int (iii-d), lambda chosen on V2 per family on small-domain macro LogLoss (equal weight over small domains), lower is better
- **difference +0.000027, 95% CI [-0.000050, +0.000107], per seed pair [2.7e-05, 7e-06, 4.9e-05] -> tier: not_confirmed**
- conditions: {"effect_size": {"relative_improvement": -0.0007218990568200757, "needed": 0.003, "met": false}, "large_domains": {"ci95_upper": 7.3444400595885745e-06, "needed_at_most": 0.0001, "met": true}, "latency": {"p99_ms_max_over_seeds": 0.10529239370953293, "p99_ms_by_seed": {"2026": 0.10529239370953293, "2027": 0.08833759668050344, "2028": 0.09641949320212012}, "needed_at_most": 1.0, "met": true}}
- large domains (click-weighted) -0.000005 [-0.000019, +0.000007]; small within-domain AUC -0.00373 [-0.01152, +0.00945] (22 domains)
- leave one small domain out: range [+0.000000, +0.000059], sign flips 0
- train-13.parquet: +0.000053 [-0.000061, +0.000162]
- train-14.parquet: +0.000004 [-0.000124, +0.000132]

| family | small macro LogLoss | small click LogLoss | small Poisson deviance (group totals) | large click LogLoss | small within-domain AUC | p99 ms, max over seeds (uniform traffic) |
|---|---|---|---|---|---|---|
| G-int | 0.037947 | 0.033265 | 3.7550 | 0.068859 | 0.7169 | 0.21 |
| G-shared | 0.037974 | 0.033303 | 3.8256 | 0.068854 | 0.7132 | 0.105 |
| N-int | 0.038132 | 0.033374 | 4.3615 | 0.069081 | 0.7103 | 0.176 |
| N-shared | 0.038071 | 0.033337 | 4.4318 | 0.069054 | 0.7138 | 0.185 |
| N-head | 0.038118 | 0.033381 | 4.9241 | 0.069065 | 0.7119 | 0.314 |
| N-star | 0.038329 | 0.033556 | 3.7621 | 0.069090 | 0.6974 | 0.776 |
| N-star-noPN | 0.037893 | 0.033207 | 4.1602 | 0.068992 | 0.7222 | – |

| secondary (A - B) | small macro [95% CI] | large click [95% CI] |
|---|---|---|
| N-star - N-shared | +0.000258 [+0.000084, +0.000418] | +0.000035 [+0.000006, +0.000064] |
| N-head - N-shared | +0.000046 [-0.000016, +0.000115] | +0.000011 [+0.000004, +0.000019] |
| N-star-noPN - N-star | -0.000436 [-0.000589, -0.000273] | -0.000098 [-0.000116, -0.000081] |
| N-head - G-shared | +0.000144 [-0.000041, +0.000322] | +0.000212 [+0.000176, +0.000245] |
| G-shared: iiid - ii | -0.000088 [-0.000207, +0.000029] | -0.000006 [-0.000012, -0.000001] |
| G-int: iiid - ii | -0.000147 [-0.000256, -0.000052] | -0.000009 [-0.000015, -0.000003] |

Intervals hold the fitted models fixed (evaluation-user uncertainty only); users seen on both files are one cluster.
