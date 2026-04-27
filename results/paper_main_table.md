# neuro-arm-control Main Results Table

Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]`

## 2A Static Holding

| Condition | static_mae [mrad] | Cohen's d vs PD |
|---|---:|---:|
| PD | 158.4 ± 0.0 | — |
| PD+CfC | 114.4 ± 30.3 | 2.05 |
| Full | 111.7 ± 25.4 | 2.60 |

## 2B Disturbance Rejection

| Disturbance | PD+CfC peak_err [rad] | Full peak_err [rad] | Cohen's d |
|---|---:|---:|---:|
| light_30Nm | 0.337 ± 0.148 | 0.310 ± 0.157 | 0.18 |
| medium_60Nm | 0.355 ± 0.178 | 0.348 ± 0.182 | 0.04 |
| heavy_87Nm | 0.600 ± 0.291 | 0.562 ± 0.284 | 0.13 |

## 2C Cyclic Motion

| Condition | ep_err_post [mrad] | mae_post [mrad] | Cohen's d |
|---|---:|---:|---:|
| CPG+CfC | 354.5 ± 87.5 | 133.4 ± 17.8 | — |
| CPG+CfC+LIF_FB | 331.7 ± 77.5 | 130.4 ± 18.6 | 0.28 |

## 2D Integrated Evaluation

| Condition | static_mae [mrad] | peak_err [rad] | recovery_time [s] |
|---|---:|---:|---:|
| PD | 160.5 ± 0.0 | 1.011 ± 0.000 | NA |
| PD+CfC | 115.8 ± 30.3 | 0.412 ± 0.282 | 0.271 ± 0.297 |
| Full | 113.1 ± 25.7 | 0.415 ± 0.281 | 0.319 ± 0.297 |

## Notes

- This table contains only main metrics.
- `tau_sys` and Cartesian error are treated as secondary analyses.
