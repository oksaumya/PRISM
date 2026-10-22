#!/usr/bin/env bash
# Download the M5 Forecasting - Accuracy data into data/raw/.
#
# Option 1 (official): accept the competition rules at
#   https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
# then this script downloads the competition files.
# Option 2 (fallback): a public Kaggle dataset mirror of the same five CSV files.
# Both need a Kaggle API token (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

if [[ -f "$RAW/sales_train_evaluation.csv" ]]; then
  echo "M5 data already present in $RAW"; exit 0
fi

if kaggle competitions download -c m5-forecasting-accuracy -p "$RAW" 2>/dev/null; then
  unzip -o -q "$RAW/m5-forecasting-accuracy.zip" -d "$RAW" && rm "$RAW/m5-forecasting-accuracy.zip"
else
  echo "Competition download refused (rules not accepted?) - using the dataset mirror."
  kaggle datasets download aryayadav0513/m5-forecasting-accuracy -p "$RAW" --unzip
  if [[ -d "$RAW/m5-forecasting-accuracy" ]]; then
    mv "$RAW"/m5-forecasting-accuracy/* "$RAW"/ && rmdir "$RAW/m5-forecasting-accuracy"
  fi
fi

# sanity check: official file has 30,490 series and 1,941 day columns
python3 - <<'PY'
import csv
with open("data/raw/sales_train_evaluation.csv") as f:
    r = csv.reader(f); header = next(r); rows = sum(1 for _ in r)
days = sum(c.startswith("d_") for c in header)
assert (rows, days) == (30490, 1941), (rows, days)
print(f"OK: {rows} series x {days} days")
PY
