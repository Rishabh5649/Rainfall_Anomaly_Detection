"""
generate_climate_indices.py
Generates synthetic but realistic ENSO and IOD climate index data
for each month from 1901 to 2015, saved to data/climate_indices.csv.

ENSO (El Niño-Southern Oscillation): ~3-7 year quasi-periodic cycle
IOD  (Indian Ocean Dipole):          ~3 year quasi-periodic cycle

Both are scaled to realistic ranges and include noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_climate_indices(
    start_year: int = 1901,
    end_year: int = 2015,
    output_path: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_years = end_year - start_year + 1
    n_months = n_years * 12

    # --- Build a time array (in months from Jan 1901) ---
    t = np.arange(n_months, dtype=float)

    # --- ENSO signal ---
    # Primary quasi-periodic cycle ~54 months  (≈4.5 years)
    # Secondary cycle ~36 months for realism
    enso_period_primary = 54.0
    enso_period_secondary = 36.0
    enso_amplitude = 1.5        # typical Niño 3.4 range ≈ -2 to +2
    enso = (
        enso_amplitude * np.sin(2 * np.pi * t / enso_period_primary + rng.uniform(0, 2 * np.pi))
        + 0.4 * enso_amplitude * np.sin(2 * np.pi * t / enso_period_secondary + rng.uniform(0, 2 * np.pi))
        + rng.normal(0, 0.15, n_months)   # observation noise
    )

    # --- IOD signal ---
    # Quasi-periodic ~36 months, peaks June-November
    iod_period = 36.0
    iod_amplitude = 0.6         # typical range ≈ -1 to +1
    # Annual modulation: strongest in boreal summer-autumn (months 6-11)
    month_of_year = (t % 12).astype(int)
    seasonal_mask = np.sin(2 * np.pi * month_of_year / 12.0 + np.pi / 6)
    iod = (
        iod_amplitude * np.sin(2 * np.pi * t / iod_period + rng.uniform(0, 2 * np.pi))
        * (0.5 + 0.5 * seasonal_mask)
        + rng.normal(0, 0.08, n_months)
    )

    # Clip to realistic bounds
    enso = np.clip(enso, -3.0, 3.0)
    iod  = np.clip(iod,  -1.5, 1.5)

    # --- Build year/month index ---
    years  = np.repeat(np.arange(start_year, end_year + 1), 12)
    months = np.tile(np.arange(1, 13), n_years)

    df = pd.DataFrame({
        "YEAR":  years,
        "MONTH": months,
        "ENSO":  np.round(enso, 4),
        "IOD":   np.round(iod,  4),
    })

    if output_path is None:
        # Resolve relative to this file's location
        root = Path(__file__).resolve().parent.parent
        output_path = root / "data" / "climate_indices.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] climate_indices.csv written -> {output_path}")
    print(f"    Shape : {df.shape}")
    print(f"    ENSO  : min={df['ENSO'].min():.3f}  max={df['ENSO'].max():.3f}")
    print(f"    IOD   : min={df['IOD'].min():.3f}  max={df['IOD'].max():.3f}")
    return df


if __name__ == "__main__":
    generate_climate_indices()
