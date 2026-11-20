# Latency (second study)

Apple M4, local, single thread; raw row -> mapping -> history features -> [auxiliary model] -> main model -> calibration; network and feature-store lookups excluded; one process per model.

## calibration (ii), as in the primary comparison

| model | batch=1 p50 | batch=1 p99 | batch=100 p50 | batch=100 p99 |
|---|---|---|---|---|
| M2_s2026 | 0.074 | 0.124 | 3.79 | 4.11 |
| M2_s2027 | 0.072 | 0.112 | 3.82 | 4.74 |
| M2_s2028 | 0.072 | 0.118 | 3.52 | 3.96 |
| M2aux_s2026 | 0.105 | 0.146 | 4.61 | 5.36 |
| M2aux_s2027 | 0.113 | 0.357 | 4.28 | 5.33 |
| M2aux_s2028 | 0.105 | 0.169 | 4.55 | 5.35 |
| M2both_s2026 | 0.103 | 0.149 | 3.76 | 4.06 |
| M2both_s2027 | 0.101 | 0.132 | 3.79 | 4.13 |
| M2both_s2028 | 0.097 | 0.132 | 3.73 | 4.06 |
| M2te_s2026 | 0.118 | 0.147 | 5.98 | 6.35 |
| M2te_s2027 | 0.120 | 0.179 | 5.83 | 6.38 |
| M2te_s2028 | 0.117 | 0.151 | 5.46 | 5.86 |

## calibration (iii), the run's selected setting

| model | batch=1 p50 | batch=1 p99 | batch=100 p50 | batch=100 p99 |
|---|---|---|---|---|
| M2_s2026 | 0.081 | 0.152 | 3.77 | 4.40 |
| M2_s2027 | 0.080 | 0.105 | 3.79 | 7.57 |
| M2_s2028 | 0.079 | 0.132 | 3.55 | 4.04 |
| M2aux_s2026 | 0.114 | 0.150 | 4.95 | 19.94 |
| M2aux_s2027 | 0.113 | 0.171 | 4.43 | 5.33 |
| M2aux_s2028 | 0.122 | 0.197 | 5.75 | 7.67 |
| M2both_s2026 | 0.112 | 0.175 | 4.02 | 8.32 |
| M2both_s2027 | 0.112 | 0.173 | 4.15 | 8.50 |
| M2both_s2028 | 0.109 | 0.147 | 3.84 | 4.76 |
| M2te_s2026 | 0.136 | 0.198 | 6.03 | 6.64 |
| M2te_s2027 | 0.124 | 0.165 | 5.61 | 6.18 |
| M2te_s2028 | 0.128 | 0.198 | 5.78 | 6.34 |
