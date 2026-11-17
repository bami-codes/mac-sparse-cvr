# Latency (main run models)

Apple M4, local, single thread; raw row -> mapping -> purchase-history features -> model -> calibration (iii_lam100); network and feature-store lookups excluded. One process per model. Serving and evaluation paths were checked to agree before timing.

| model | batch=1 p50 (ms) | batch=1 p99 (ms) | batch=100 p50 (ms) | batch=100 p99 (ms) | max pred diff |
|---|---|---|---|---|---|
| M1 | 0.041 | 0.049 | 0.83 | 0.87 | 1.9e-16 |
| M2_s2026 | 0.066 | 0.085 | 2.86 | 3.16 | 0.0e+00 |
| M2_s2027 | 0.065 | 0.085 | 2.92 | 3.13 | 0.0e+00 |
| M2_s2028 | 0.071 | 0.292 | 3.42 | 9.01 | 0.0e+00 |
| M3_attn_s2026 | 0.227 | 0.349 | 2.46 | 3.66 | 0.0e+00 |
| M3_attn_s2027 | 0.222 | 0.262 | 2.39 | 2.71 | 0.0e+00 |
| M3_attn_s2028 | 0.237 | 0.342 | 2.75 | 3.45 | 0.0e+00 |
| M3_mean_s2026 | 0.207 | 0.247 | 2.26 | 2.90 | 0.0e+00 |
| M3_mean_s2027 | 0.209 | 0.250 | 2.19 | 2.53 | 0.0e+00 |
| M3_mean_s2028 | 0.221 | 0.360 | 2.27 | 2.95 | 0.0e+00 |
