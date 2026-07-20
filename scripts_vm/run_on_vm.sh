#!/bin/bash
# Run one CV experiment on the VM.
#
# Usage (env vars drive the experiment; kept separate so the spaced --desc is
# quoted correctly - shell does not re-parse quotes inside a single variable):
#   EXP_NAME="release_filter" \
#   EXP_DESC="Phase1: drop pre-release rows" \
#   EXP_EXTRA="--train-start-day 1300 --n-estimators 1500 --rebuild-features" \
#   ~/run_on_vm.sh
#
# Persists across runs on the VM's disk:
#   ~/m5/data/                     raw CSVs (fetched once)
#   ~/m5/.venv/                    python env (created once)
#   ~/m5/outputs/cache/            feature cache (reused unless --rebuild-features)
# The canonical, accumulating experiment log lives in gs://.../results/ and is
# pulled in before the run and pushed back after, so every run appends to it.
set -euo pipefail
BUCKET=gs://final-project-478101-m5
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/5] system deps (idempotent) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3.10-venv libgomp1

echo "=== [2/5] refresh code; keep data/cache ==="
cd ~
mkdir -p m5/data m5/outputs/cache && cd m5
gsutil -q cp "$BUCKET/code/m5_code.tgz" .
tar xzf m5_code.tgz                       # overwrites src/, docs/, requirements
[ -f data/calendar.csv ] || gsutil -q -m cp "$BUCKET/data/*.csv" data/
# Restore any previously-built feature caches (VMs are deleted between runs).
gsutil -q -m cp "$BUCKET/results/features_*.parquet" outputs/cache/ 2>/dev/null || true

echo "=== [3/5] python env (create once) ==="
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements.txt
python -c "import lightgbm,pandas,numpy; print('libs:', lightgbm.__version__, pandas.__version__, numpy.__version__)"

echo "=== [4/5] run experiment ==="
# Pull the canonical log so this run appends to the full history.
mkdir -p outputs docs
gsutil -q cp "$BUCKET/results/experiments.csv" outputs/experiments.csv 2>/dev/null || true
gsutil -q cp "$BUCKET/results/experiments.md"  docs/experiments.md      2>/dev/null || true
EXP_MODULE="${EXP_MODULE:-src.cv}"     # src.cv (point) or src.run_uncertainty
echo "module=$EXP_MODULE | name=$EXP_NAME | desc=$EXP_DESC | extra=$EXP_EXTRA"
PYTHONPATH=. python -u -m "$EXP_MODULE" --name "$EXP_NAME" --desc "$EXP_DESC" $EXP_EXTRA

echo "=== [5/5] push results ==="
# Experiment ledgers (point track and uncertainty track).
for f in outputs/*.csv docs/*experiments*.md; do
  [ -e "$f" ] && gsutil -q cp "$f" "$BUCKET/results/$(basename "$f")" || true
done
# Saved forecasts (e.g. the quantile forecasts Project 2 consumes).
for f in outputs/predictions/*.parquet; do
  [ -e "$f" ] && gsutil -q cp "$f" "$BUCKET/results/$(basename "$f")" || true
done
# Persist per-config feature caches to GCS. VMs get deleted (not just stopped)
# to avoid disk charges, so the cache must survive off-box or every new VM pays
# the ~6 min rebuild. Names encode the recipe: features_<groups>_f<floor>.parquet
for f in outputs/cache/features_*.parquet; do
  [ -e "$f" ] || continue
  gsutil -q cp "$f" "$BUCKET/results/$(basename "$f")" || true
done
echo "=== DONE ==="
