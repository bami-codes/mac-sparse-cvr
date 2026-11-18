# Study C latency (protocol v2, accepted attempts)

- run `runs/c_dev_train08_10/20261118T061335Z`; {'cpu': 'Apple M4', 'python': '3.11.9', 'threads': 1, 'process': 'one subprocess per family'}
- protocol {'version': 2, 'warmup': 50, 'requests': 5000, 'traffic_rows': 10000, 'traffic_seed': 20261119}
- acceptance: no other python process of this virtualenv at the start and end of each family's timing, and other processes use at most 400% CPU at both points

| family | attempts (accepted?) | p50 ms per seed | p99 ms per seed | batch=100 p99 ms per seed | max serving diff |
|---|---|---|---|---|---|
| G-int | [True] | [0.078, 0.07, 0.073] | [0.21, 0.087, 0.125] | [3.42, 2.82, 3.23] | 0.0e+00 |
| G-shared | [True] | [0.072, 0.067, 0.067] | [0.105, 0.088, 0.096] | [3.46, 3.02, 3.11] | 0.0e+00 |
| N-int | [True] | [0.151, 0.15, 0.15] | [0.176, 0.167, 0.163] | [1.38, 1.32, 1.36] | 5.7e-08 |
| N-shared | [True] | [0.146, 0.148, 0.143] | [0.185, 0.171, 0.159] | [1.46, 1.33, 1.51] | 5.2e-08 |
| N-head | [False, True] | [0.161, 0.162, 0.16] | [0.242, 0.314, 0.196] | [1.47, 1.43, 1.33] | 7.1e-08 |
| N-star | [False, False, True] | [0.287, 0.284, 0.276] | [0.776, 0.313, 0.311] | [6.43, 3.84, 3.8] | 4.2e-08 |
