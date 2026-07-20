#!/bin/bash
# Milestone A - "how much history?" sweep, run as one VM session.
# Builds a low-floor (day 300) release-filtered feature cache once, then scores
# three train-start sizes on the 3 dev folds. Pushes the experiment log to GCS
# after each run so a preemption never loses a completed experiment.
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
# Fresh 3-fold log: drop any tarball/old copy, then pull the canonical bucket
# copy if one exists (it won't on the first milestone-A run).
rm -f outputs/experiments.csv docs/experiments.md
gsutil -q cp "$BUCKET/results/experiments.csv" outputs/experiments.csv 2>/dev/null || true
gsutil -q cp "$BUCKET/results/experiments.md"  docs/experiments.md      2>/dev/null || true

run() {  # name, desc, extra-flags
  echo "=== RUN: $1 | $2 ==="
  PYTHONPATH=. python -u -m src.cv --name "$1" --desc "$2" $3
  gsutil -q cp outputs/experiments.csv "$BUCKET/results/experiments.csv" || true
  gsutil -q cp docs/experiments.md     "$BUCKET/results/experiments.md"  || true
}

# 1) reference: current recipe (release-filter) at train_start=1300, 3-fold,
#    rebuilding the cache at day_floor=300 so the later runs can reuse it.
run "ref_ts1300" "Reference: release-filter recipe, train_start=1300, 3-fold" \
    "--train-start-day 1300 --day-floor 300 --rebuild-features"

# 2) more history
run "hist_ts700" "More history: train_start=700" \
    "--train-start-day 700 --day-floor 300"

# 3) near-full history
run "hist_ts300" "Near-full history: train_start=300" \
    "--train-start-day 300 --day-floor 300"

echo "=== ALL DONE ==="
