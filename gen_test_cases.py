"""
gen_test_cases.py
Generate 5 real test cases from the trained CNN+LSTM model
using actual data from the test split (2006–2015).
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import (
    load_rainfall, load_climate_indices, wide_to_long,
    merge_climate, add_features, apply_scaler, load_scaler,
    FEATURE_COLS, MONTH_COLS, SEQ_LEN
)
from model import RainSightCNNLSTM, get_device
from train import BEST_MODEL_PATH
import torch

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── 1. Load model ───────────────────────────────────────────────────
device = get_device()
ckpt   = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
model  = RainSightCNNLSTM(
    n_features = ckpt.get("n_features", len(FEATURE_COLS)),
    seq_len    = ckpt.get("seq_len", SEQ_LEN),
)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()
print(f"[OK] Model loaded — epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")

# ── 2. Build full pipeline ──────────────────────────────────────────
df_wide = load_rainfall()
try:
    df_cli = load_climate_indices()
except FileNotFoundError:
    years  = range(1901, 2016)
    months = range(1, 13)
    df_cli = pd.DataFrame([{"YEAR": y, "MONTH": m, "ENSO": 0.0, "IOD": 0.0}
                            for y in years for m in months])

df_long = wide_to_long(df_wide)
df      = merge_climate(df_long, df_cli)
df      = add_features(df)
scaler  = load_scaler()
df_scaled = apply_scaler(df, scaler)

rf_mean  = scaler.mean_[0]
rf_scale = scaler.scale_[0]

# ── 3. Pick 5 test cases: year > 2005, high-rainfall months ─────────
test_cases = [
    ("COASTAL KARNATAKA",   2010, 7),   # Monsoon peak
    ("KERALA",              2011, 6),   # Southwest monsoon
    ("KONKAN & GOA",        2012, 7),   # Heavy monsoon
    ("TAMIL NADU",          2013, 11),  # Northeast monsoon
    ("ASSAM & MEGHALAYA",   2014, 8),   # Peak monsoon
]

results = []
with torch.no_grad():
    for subdivision, target_year, target_month in test_cases:
        sub_df = df_scaled[df_scaled["SUBDIVISION"] == subdivision].sort_values(["YEAR","MONTH"])

        # Find actual rainfall for this target month
        row = df[(df["SUBDIVISION"] == subdivision) &
                 (df["YEAR"] == target_year) &
                 (df["MONTH"] == target_month)]
        actual_mm = float(row["RAINFALL"].values[0]) if len(row) > 0 else None

        # Get the 24-month window ending just BEFORE the target
        mask = (sub_df["YEAR"] < target_year) | (
            (sub_df["YEAR"] == target_year) & (sub_df["MONTH"] < target_month)
        )
        seed = sub_df[mask].tail(SEQ_LEN)

        if len(seed) < SEQ_LEN:
            seed = sub_df.head(SEQ_LEN)

        seq = seed[FEATURE_COLS].values.astype(np.float32)[-SEQ_LEN:]
        x   = torch.from_numpy(seq[np.newaxis]).to(device)
        pred_norm = model(x).item()
        pred_mm   = max(pred_norm * rf_scale + rf_mean, 0.0)

        if actual_mm is not None:
            accuracy = max(0.0, 100.0 - abs(pred_mm - actual_mm) / max(actual_mm, 1.0) * 100)
        else:
            accuracy = None

        results.append({
            "subdivision": subdivision,
            "year": target_year,
            "month": target_month,
            "month_name": MONTH_NAMES[target_month - 1],
            "actual_mm": round(actual_mm, 1) if actual_mm is not None else "N/A",
            "predicted_mm": round(pred_mm, 1),
            "accuracy": round(accuracy, 1) if accuracy is not None else "N/A",
        })

# ── 4. Print results ────────────────────────────────────────────────
print("\n" + "="*70)
print("  ML MODEL VALIDATION — RainSight India (CNN+LSTM)")
print("="*70)
for i, r in enumerate(results, 1):
    print(f"\n  Test Case {i}: {r['subdivision']}")
    print(f"    Period   : {r['month_name']} {r['year']}")
    print(f"    Actual   : {r['actual_mm']} mm")
    print(f"    Predicted: {r['predicted_mm']} mm")
    print(f"    Accuracy : {r['accuracy']}%")

# ── 5. Save as JSON ─────────────────────────────────────────────────
out_path = ROOT / "outputs" / "test_cases.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[OK] Saved → {out_path}")
