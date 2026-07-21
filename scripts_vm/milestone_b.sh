#!/bin/bash
# Milestone B - feature-group screening, run as one VM session.
# Each optional group is added to the base recipe and scored on the 3 dev folds
# at train_start=700 (a faster proxy for the train_start=300 operating point).
# Reference to beat = hist_700 base recipe = CV mean 0.6466.
# Pushes the experiment log to GCS after each run (preemption-safe).
set -euo pipefail
BUCKET=gs://final-project-478101-m5
export DEBIAN_FRONTEND=noninteractive

echo "=== setup ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3.10-venv libgomp1
cd ~ && mkdir -p m5/data m5/outputs/cache && cd m5
gsutil -q cp "$BUCKET/code/m5_code.tgz" . && tar xzf m5_code.tgz
[ -f data/calendar.csv ] || gsutil -q -m cp "$BUCKET/data/*.csv" data/
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements.txt
python -c "import lightgbm,pandas; print('libs', lightgbm.__version__, pandas.__version__)"

mkdir -p outputs docs
gsutil -q cp "$BUCKET/results/experiments.csv" outputs/experiments.csv 2>/dev/null || true
gsutil -q cp "$BUCKET/results/experiments.md"  docs/experiments.md      2>/dev/null || true

TS="--train-start-day 700 --day-floor 700 --n-estimators 1500 --rebuild-features"
run() {  # name, desc, feature-groups
  echo "=== RUN: $1 (features=$3) ==="
  PYTHONPATH=. python -u -m src.cv --name "$1" --desc "$2" --features "$3" $TS
  gsutil -q cp outputs/experiments.csv "$BUCKET/results/experiments.csv" || true
  gsutil -q cp docs/experiments.md     "$BUCKET/results/experiments.md"  || true
}

run "b_roll"  "Milestone B: base + roll_ext (60/180 mean, 60 std, 28 max/min) @ts700"  "roll_ext"
run "b_price" "Milestone B: base + price_ext (norm, max/min, nunique, wow change) @ts700" "price_ext"
run "b_cal"   "Milestone B: base + cal_ext (is_weekend, week_of_month) @ts700"          "cal_ext"
run "b_all"   "Milestone B: base + all groups @ts700"                                    "roll_ext,price_ext,cal_ext"

echo "=== ALL DONE ==="
