"""Central configuration for the M5 forecasting project.

All paths are resolved relative to the repository root so the code runs the
same way regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

for _d in (MODEL_DIR, SUBMISSION_DIR, PREDICTION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw competition files
SALES_VALIDATION_CSV = DATA_DIR / "sales_train_validation.csv"   # d_1 .. d_1913
SALES_EVALUATION_CSV = DATA_DIR / "sales_train_evaluation.csv"   # d_1 .. d_1941
CALENDAR_CSV = DATA_DIR / "calendar.csv"
PRICES_CSV = DATA_DIR / "sell_prices.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# --- Competition constants -------------------------------------------------
# Identifier columns present in the wide sales files.
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Forecast horizon: 28 days (F1 .. F28).
HORIZON = 28

# Day boundaries (a "day" is the integer N in the column name "d_N").
#  - The public-LB "validation" window is d_1914 .. d_1941 (ground truth lives
#    in sales_train_evaluation.csv, so we can score it locally).
#  - The private-LB "evaluation" window is d_1942 .. d_1969 (never released;
#    we can only submit predictions for it, not score them locally).
LAST_TRAIN_DAY_VALIDATION = 1913   # last day used to predict the validation window
LAST_TRAIN_DAY_EVALUATION = 1941   # last day used to predict the evaluation window

VALIDATION_DAYS = list(range(1914, 1914 + HORIZON))   # 1914 .. 1941
EVALUATION_DAYS = list(range(1942, 1942 + HORIZON))   # 1942 .. 1969

# The 10 stores let us run feature engineering one store at a time, which keeps
# peak memory bounded (one store = ~3,049 series instead of all 30,490).
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

SEED = 42
