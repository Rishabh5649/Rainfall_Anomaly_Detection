"""
predict.py
Inference engine for RainSight India.
Provides a stateful predictor that:
  - Loads model checkpoint only once
  - Accepts (subdivision, last_year, last_month, horizon) and
    returns predicted rainfall for the next `horizon` months
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import (
    load_rainfall, load_climate_indices,
    wide_to_long, merge_climate, add_features,
    apply_scaler, load_scaler,
    FEATURE_COLS, MONTH_COLS, SEQ_LEN,
)
from model import RainSightCNNLSTM, get_device
from train import BEST_MODEL_PATH


SUBDIVISIONS_PATH = ROOT / "checkpoints" / "subdivisions.json"


class RainPredictor:
    """
    Stateful inference engine. Initialise once; call .predict() many times.
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len   = seq_len
        self.device    = get_device()
        self.model     = self._load_model()
        self.scaler    = load_scaler()
        self._raw_df   = self._build_long_df()  # normalised long-format df
        self.subdivisions = self._load_subdivisions()

    # ── Private helpers ───────────────────────────────────────────
    def _load_model(self) -> RainSightCNNLSTM:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint at {BEST_MODEL_PATH}.\n"
                "Run: python src/train.py"
            )
        ckpt  = torch.load(BEST_MODEL_PATH, map_location=self.device, weights_only=False)
        model = RainSightCNNLSTM(
            n_features = ckpt.get("n_features", len(FEATURE_COLS)),
            seq_len    = ckpt.get("seq_len", self.seq_len),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        print(f"[✓] Predictor: model loaded (epoch={ckpt['epoch']})")
        return model

    def _build_long_df(self):
        import pandas as pd
        df_wide = load_rainfall()
        try:
            df_cli = load_climate_indices()
        except FileNotFoundError:
            # Fall back to zero climate indices if file missing
            import pandas as pd
            years  = range(1901, 2016)
            months = range(1, 13)
            records = [{"YEAR": y, "MONTH": m, "ENSO": 0.0, "IOD": 0.0}
                       for y in years for m in months]
            df_cli = pd.DataFrame(records)

        df_long = wide_to_long(df_wide)
        df = merge_climate(df_long, df_cli)
        df = add_features(df)
        df = apply_scaler(df, self.scaler)
        return df

    def _load_subdivisions(self) -> List[str]:
        if SUBDIVISIONS_PATH.exists():
            with open(SUBDIVISIONS_PATH) as f:
                return json.load(f)
        return sorted(self._raw_df["SUBDIVISION"].unique().tolist())

    # ── Public API ────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        subdivision: str,
        start_year:  int,
        start_month: int,
        horizon:     int = 12,
    ) -> List[Dict]:
        """
        Predict `horizon` months of rainfall starting from (start_year, start_month).

        Returns list of dicts:
          [{"year": 2016, "month": 1, "predicted_mm": 45.2,
            "lower_mm": 30.1, "upper_mm": 60.3}, ...]
        """
        # ── Get seed window from historical data ──────────────────
        sub_df = self._raw_df[
            self._raw_df["SUBDIVISION"] == subdivision
        ].sort_values(["YEAR", "MONTH"]).reset_index(drop=True)

        if len(sub_df) == 0:
            raise ValueError(f"Subdivision '{subdivision}' not found.")

        # Find the last available data point before (start_year, start_month)
        mask = (sub_df["YEAR"] < start_year) | (
            (sub_df["YEAR"] == start_year) & (sub_df["MONTH"] < start_month)
        )
        seed_df = sub_df[mask].tail(self.seq_len)

        if len(seed_df) < self.seq_len:
            # Pad with earliest available if not enough history
            pad_len = self.seq_len - len(seed_df)
            seed_df = sub_df.head(self.seq_len)

        seq = seed_df[FEATURE_COLS].values.astype(np.float32)  # (seq_len, n_feat)
        seq = seq[-self.seq_len:]   # ensure exactly seq_len

        rf_mean  = self.scaler.mean_[0]
        rf_scale = self.scaler.scale_[0]

        results = []
        cur_year, cur_month = start_year, start_month

        for step in range(horizon):
            x_tensor = torch.from_numpy(seq[np.newaxis]).to(self.device)  # (1, T, F)
            pred_norm = self.model(x_tensor).item()
            pred_mm   = pred_norm * rf_scale + rf_mean
            pred_mm   = max(pred_mm, 0.0)   # rainfall cannot be negative

            # Uncertainty: simple ±15% relative + 5 mm absolute
            error_mm  = max(pred_mm * 0.15, 5.0)
            results.append({
                "year":         cur_year,
                "month":        cur_month,
                "predicted_mm": round(pred_mm, 2),
                "lower_mm":     round(max(pred_mm - error_mm, 0.0), 2),
                "upper_mm":     round(pred_mm + error_mm, 2),
            })

            # ── Advance sequence ──────────────────────────────────
            # Build next feature row (using predicted rainfall)
            next_month = cur_month % 12 + 1
            next_year  = cur_year + (1 if cur_month == 12 else 0)

            month_sin = np.sin(2 * np.pi * next_month / 12)
            month_cos = np.cos(2 * np.pi * next_month / 12)
            year_norm = (next_year - 1901) / (2015 - 1901)

            new_row = np.array([
                pred_norm,   # RAINFALL (normalised)
                0.0,         # ENSO (unknown future: use 0)
                0.0,         # IOD
                month_sin,
                month_cos,
                year_norm,
            ], dtype=np.float32)

            seq = np.vstack([seq[1:], new_row])   # slide window forward
            cur_year, cur_month = next_year, next_month

        return results

    def get_history(self, subdivision: str) -> List[Dict]:
        """Return full historical (unscaled) rainfall for a subdivision."""
        sub_df = self._raw_df[
            self._raw_df["SUBDIVISION"] == subdivision
        ].sort_values(["YEAR", "MONTH"]).copy()

        rf_mean  = self.scaler.mean_[0]
        rf_scale = self.scaler.scale_[0]

        records = []
        for _, row in sub_df.iterrows():
            records.append({
                "year":       int(row["YEAR"]),
                "month":      int(row["MONTH"]),
                "rainfall_mm": round(row["RAINFALL"] * rf_scale + rf_mean, 2),
            })
        return records


# ── Singleton (loaded once when imported by API) ──────────────────
_predictor: Optional[RainPredictor] = None

def get_predictor() -> RainPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RainPredictor()
    return _predictor


# ── CLI test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = RainPredictor()
    preds = predictor.predict(
        subdivision = predictor.subdivisions[0],
        start_year  = 2010,
        start_month = 1,
        horizon     = 12,
    )
    print(f"\nForecast for {predictor.subdivisions[0]}:")
    for p in preds:
        print(f"  {p['year']}-{p['month']:02d}  {p['predicted_mm']:7.2f} mm  "
              f"[{p['lower_mm']:.2f} – {p['upper_mm']:.2f}]")
