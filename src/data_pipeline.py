"""
data_pipeline.py
Full data pipeline for RainSight India:
  1. Load & clean IMD rainfall CSV (forward-fill NaNs per subdivision)
  2. Load climate indices (ENSO, IOD)
  3. Merge, normalise per feature
  4. Build sliding-window sequences for CNN-LSTM
  5. Train / val / test split by year
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import pickle
import json


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAINFALL_CSV = DATA_DIR / "rainfall_in_india_1901-2015.csv"
CLIMATE_CSV  = DATA_DIR / "climate_indices.csv"
SCALER_PATH  = ROOT / "checkpoints" / "scalers.pkl"

SEQ_LEN = 24      # look-back window: 24 months


# ─────────────────────────────────────────────────────────────────
# 1. Load & clean rainfall data
# ─────────────────────────────────────────────────────────────────
def load_rainfall(path: Path = RAINFALL_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = [c.strip().upper().replace("-", "_").replace(" ", "_")
                  for c in df.columns]
    df.rename(columns={"SUBDIVISION": "SUBDIVISION", "YEAR": "YEAR"}, inplace=True)

    # Keep only the 12 monthly columns + meta
    keep = ["SUBDIVISION", "YEAR"] + MONTH_COLS
    df = df[[c for c in keep if c in df.columns]].copy()

    # Replace 'NA' strings with actual NaN
    df.replace("NA", np.nan, inplace=True)
    for col in MONTH_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill NaN per subdivision (sort by year first)
    df.sort_values(["SUBDIVISION", "YEAR"], inplace=True)
    for col in MONTH_COLS:
        df[col] = df.groupby("SUBDIVISION")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Drop any rows that still have NaN after ffill/bfill
    df.dropna(subset=MONTH_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 2. Load climate indices
# ─────────────────────────────────────────────────────────────────
def load_climate_indices(path: Path = CLIMATE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Climate indices not found at {path}.\n"
            "Run: python src/generate_climate_indices.py"
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────
# 3. Build long-format monthly dataframe
# ─────────────────────────────────────────────────────────────────
def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert year-wide format to one row per (subdivision, year, month)."""
    records = []
    for _, row in df_wide.iterrows():
        sub  = row["SUBDIVISION"]
        year = int(row["YEAR"])
        for m_idx, col in enumerate(MONTH_COLS, start=1):
            records.append({
                "SUBDIVISION": sub,
                "YEAR":        year,
                "MONTH":       m_idx,
                "RAINFALL":    float(row[col]),
            })
    df_long = pd.DataFrame(records)
    df_long.sort_values(["SUBDIVISION", "YEAR", "MONTH"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    return df_long


# ─────────────────────────────────────────────────────────────────
# 4. Merge climate indices
# ─────────────────────────────────────────────────────────────────
def merge_climate(df_long: pd.DataFrame, df_cli: pd.DataFrame) -> pd.DataFrame:
    df = df_long.merge(df_cli[["YEAR", "MONTH", "ENSO", "IOD"]],
                       on=["YEAR", "MONTH"], how="left")
    # Fill any remaining NaN in climate cols with 0
    df["ENSO"] = df["ENSO"].fillna(0.0)
    df["IOD"]  = df["IOD"].fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────
# 5. Feature engineering
# ─────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cyclic month encoding
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    # Year normalised (helps LSTM sense trends)
    df["YEAR_NORM"] = (df["YEAR"] - 1901) / (2015 - 1901)
    return df


FEATURE_COLS = ["RAINFALL", "ENSO", "IOD", "MONTH_SIN", "MONTH_COS", "YEAR_NORM"]
N_FEATURES   = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────
# 6. Normalise
# ─────────────────────────────────────────────────────────────────
def fit_scalers(df: pd.DataFrame) -> StandardScaler:
    """Fit a single StandardScaler on all FEATURE_COLS."""
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].values)
    return df


def save_scaler(scaler: StandardScaler, path: Path = SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[✓] Scaler saved → {path}")


def load_scaler(path: Path = SCALER_PATH) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
# 7. Sequence builder
# ─────────────────────────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    feature_cols: List[str] = FEATURE_COLS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X      : (N, seq_len, n_features)
        y      : (N,)  — next-step RAINFALL
        years  : (N,)  — year of the target
        subdivs: (N,)  — subdivision name strings
    """
    X_list, y_list, yr_list, sub_list = [], [], [], []

    for sub, grp in df.groupby("SUBDIVISION"):
        grp = grp.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)
        values = grp[feature_cols].values  # (T, n_features)
        years  = grp["YEAR"].values

        for i in range(seq_len, len(values)):
            X_list.append(values[i - seq_len : i])
            y_list.append(values[i, feature_cols.index("RAINFALL")])
            yr_list.append(years[i])
            sub_list.append(sub)

    X      = np.array(X_list,  dtype=np.float32)
    y      = np.array(y_list,  dtype=np.float32)
    years  = np.array(yr_list, dtype=np.int32)
    subdivs= np.array(sub_list)
    return X, y, years, subdivs


# ─────────────────────────────────────────────────────────────────
# 8. Train / Val / Test split (by year)
# ─────────────────────────────────────────────────────────────────
def split_by_year(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    subdivs: np.ndarray,
    train_until: int = 1995,
    val_until:   int = 2005,
) -> Dict[str, Tuple]:
    train_mask = years <= train_until
    val_mask   = (years > train_until) & (years <= val_until)
    test_mask  = years > val_until

    return {
        "train": (X[train_mask], y[train_mask], years[train_mask], subdivs[train_mask]),
        "val":   (X[val_mask],   y[val_mask],   years[val_mask],   subdivs[val_mask]),
        "test":  (X[test_mask],  y[test_mask],  years[test_mask],  subdivs[test_mask]),
    }


def split_by_ratio_years(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    subdivs: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio_within_train: float = 0.2,
) -> Tuple[Dict[str, Tuple], Dict[str, int]]:
    """
    Time-aware split by year ratio.

    Example:
      train_ratio=0.8 and val_ratio_within_train=0.2
      -> 64% train, 16% val, 20% test by year buckets.
    """
    unique_years = np.array(sorted(np.unique(years)))
    n_years = len(unique_years)
    if n_years < 3:
        raise ValueError("Need at least 3 unique years to create train/val/test splits.")

    train_year_count = int(round(n_years * train_ratio))
    train_year_count = max(2, min(train_year_count, n_years - 1))

    val_year_count = int(round(train_year_count * val_ratio_within_train))
    val_year_count = max(1, min(val_year_count, train_year_count - 1))

    train_until = int(unique_years[train_year_count - 1])
    val_start_year = int(unique_years[train_year_count - val_year_count])

    train_mask = years < val_start_year
    val_mask = (years >= val_start_year) & (years <= train_until)
    test_mask = years > train_until

    splits = {
        "train": (X[train_mask], y[train_mask], years[train_mask], subdivs[train_mask]),
        "val":   (X[val_mask],   y[val_mask],   years[val_mask],   subdivs[val_mask]),
        "test":  (X[test_mask],  y[test_mask],  years[test_mask],  subdivs[test_mask]),
    }
    boundaries = {
        "val_start_year": val_start_year,
        "train_until": train_until,
        "year_count": int(n_years),
    }
    return splits, boundaries


# ─────────────────────────────────────────────────────────────────
# 9. Master pipeline function
# ─────────────────────────────────────────────────────────────────
def build_dataset(
    seq_len:     int = SEQ_LEN,
    train_until: int = 1995,
    val_until:   int = 2005,
    fit_new_scaler: bool = True,
    train_test_ratio: str = "",
    val_ratio_within_train: float = 0.2,
) -> Tuple[Dict, StandardScaler, pd.DataFrame]:
    """
    Full pipeline. Returns splits dict, fitted scaler, and raw long df.
    """
    print("[·] Loading rainfall data...")
    df_wide = load_rainfall()
    print(f"    {len(df_wide)} subdivision-year rows | "
          f"{df_wide['SUBDIVISION'].nunique()} subdivisions")

    print("[·] Loading climate indices...")
    df_cli = load_climate_indices()

    print("[·] Reshaping to long format...")
    df_long = wide_to_long(df_wide)

    print("[·] Merging climate indices...")
    df = merge_climate(df_long, df_cli)

    print("[·] Adding features...")
    df = add_features(df)

    if fit_new_scaler:
        print("[·] Fitting scaler...")
        scaler = fit_scalers(df)
        save_scaler(scaler)
    else:
        print("[·] Loading existing scaler...")
        scaler = load_scaler()

    print("[·] Applying scaler...")
    df_scaled = apply_scaler(df, scaler)

    print(f"[·] Building sequences (seq_len={seq_len})...")
    X, y, years, subdivs = build_sequences(df_scaled, seq_len=seq_len)
    print(f"    X shape: {X.shape}  y shape: {y.shape}")

    print("[·] Splitting train / val / test...")
    if train_test_ratio:
        try:
            train_str, test_str = train_test_ratio.split(":")
            train_ratio = float(train_str) / (float(train_str) + float(test_str))
        except Exception as e:
            raise ValueError(
                f"Invalid train_test_ratio '{train_test_ratio}'. Expected format like '80:20'."
            ) from e
        splits, boundaries = split_by_ratio_years(
            X, y, years, subdivs,
            train_ratio=train_ratio,
            val_ratio_within_train=val_ratio_within_train,
        )
        print(
            f"    Ratio mode {train_test_ratio} | "
            f"val starts {boundaries['val_start_year']} | "
            f"train/test boundary {boundaries['train_until']}"
        )
    else:
        splits = split_by_year(X, y, years, subdivs,
                               train_until=train_until, val_until=val_until)
    for name, (Xs, ys, *_) in splits.items():
        print(f"    {name:6s}: {len(Xs):6d} samples")

    # Save subdivision list for API use
    sub_list_path = ROOT / "checkpoints" / "subdivisions.json"
    sub_list_path.parent.mkdir(parents=True, exist_ok=True)
    subdivisions = sorted(df["SUBDIVISION"].unique().tolist())
    with open(sub_list_path, "w") as f:
        json.dump(subdivisions, f, indent=2)
    print(f"[✓] Subdivision list saved → {sub_list_path}")

    return splits, scaler, df


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    splits, scaler, df = build_dataset()
    print("\n[✓] Pipeline complete.")
    print(f"   Features : {FEATURE_COLS}")
    print(f"   Train X  : {splits['train'][0].shape}")
    print(f"   Val   X  : {splits['val'][0].shape}")
    print(f"   Test  X  : {splits['test'][0].shape}")
