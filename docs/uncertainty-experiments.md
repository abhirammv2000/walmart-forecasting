# Uncertainty Experiment Log

Bottom-level (item x store) weighted scaled pinball loss over the nine M5 quantiles. Lower is better.

| timestamp | name | fold | backend | train_start_day | n_estimators | bottom_spl | crossing_rate | best_iters | runtime_s | desc |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-20 20:50 | q_fast | final | xgboost | 1750 | 150 | 0.2618 | 0.16918 | 149/149/149/149/149/149/149/149/149 | 1056 | XGB multi-quantile, final window, ts1750 fast config |
