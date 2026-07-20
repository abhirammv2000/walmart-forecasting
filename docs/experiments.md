# Experiment Log

WRMSSE on rolling dev folds (cv1: d1830-1857, cv2: d1858-1885, cv3: d1886-1913) and the held-out final test (d1914-1941). Lower is better. We steer on **cv_mean**; the final test is touched only at milestones.

| timestamp | name | train_start_day | n_estimators | cv1 | cv2 | cv3 | cv_mean | final_test | best_iters | runtime_s | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-08 21:42 | ref_ts1300 | 1300 | 1500 | 0.7589 | 0.7059 | 0.5573 | 0.674 |  | 306/257/185 | 850 | Reference: release-filter recipe, train_start=1300, 3-fold |
| 2026-07-08 22:21 | hist_ts700 | 700 | 1500 | 0.7282 | 0.6654 | 0.5463 | 0.6466 |  | 962/225/226 | 2279 | More history: train_start=700 |
| 2026-07-11 01:23 | hist_ts300 | 300 | 1500 | 0.6975 | 0.6655 | 0.5483 | 0.6371 |  | 761/220/225 | 1814 | Near-full history: train_start=300 (release-filter, day-300 cache) |
| 2026-07-11 03:07 | b_roll | 700 | 1500 | 0.7194 | 0.6936 | 0.5376 | 0.6502 |  | 240/202/276 | 940 | Milestone B: base + roll_ext (60/180 mean, 60 std, 28 max/min) @ts700 |
| 2026-07-11 03:39 | b_price | 700 | 1500 | 0.7284 | 0.6731 | 0.5463 | 0.6493 |  | 794/372/280 | 1627 | Milestone B: base + price_ext (norm, max/min, nunique, wow change) @ts700 |
| 2026-07-11 04:05 | b_cal | 700 | 1500 | 0.7203 | 0.6719 | 0.5492 | 0.6472 |  | 798/222/207 | 1344 | Milestone B: base + cal_ext (is_weekend, week_of_month) @ts700 |
| 2026-07-11 04:27 | b_all | 700 | 1500 | 0.7075 | 0.6878 | 0.5515 | 0.6489 |  | 298/215/203 | 1015 | Milestone B: base + all groups @ts700 |
| 2026-07-20 01:09 | best_final | 300 | 1500 | 0.7081 | 0.673 | 0.5391 | 0.6401 | 0.6475 | 748/231/245/1369 | 4118 | Best config (base features, train_start=300) + frozen final test; corrected WRMSSE |
