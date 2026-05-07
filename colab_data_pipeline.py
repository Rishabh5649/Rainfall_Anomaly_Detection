"""
Colab-Compatible Data Pipeline for CNN-LSTM Rainfall Prediction
Works on both Google Colab and local machines
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.preprocessing import StandardScaler
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# COLAB DETECTION & PATH SETUP
# ============================================================================
try:
    import google.colab
    IN_COLAB = True
    print("[✓] Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("[·] Running locally")

# Set base directory
if IN_COLAB:
    BASE_DIR = '/content/cnn_lstm_rainfall_system'
    if not os.path.exists(BASE_DIR):
        print("[!] Project not found. Cloning from GitHub...")
        os.system('git clone https://github.com/yourusername/cnn_lstm_rainfall_system.git /content/cnn_lstm_rainfall_system')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

# Define paths
RAINFALL_DATA = os.path.join(BASE_DIR, 'data', 'rainfall_in_india_1901-2015.csv')
CLIMATE_INDICES = os.path.join(BASE_DIR, 'data', 'climate_indices.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'checkpoints', 'scaler.pkl')
SUBDIVISIONS_PATH = os.path.join(BASE_DIR, 'checkpoints', 'subdivisions.json')

print(f"[·] Base directory: {BASE_DIR}")
print(f"[·] Rainfall data: {RAINFALL_DATA}")

# ============================================================================
# CORE DATA PIPELINE FUNCTIONS
# ============================================================================

def load_rainfall_data():
    """Load rainfall data from CSV"""
    if not os.path.exists(RAINFALL_DATA):
        raise FileNotFoundError(f"Rainfall data not found at {RAINFALL_DATA}")
    
    df = pd.read_csv(RAINFALL_DATA)
    print(f"[✓] Loaded rainfall data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_climate_indices():
    """Load climate indices (ENSO, IOD)"""
    if not os.path.exists(CLIMATE_INDICES):
        print(f"[!] Climate indices not found at {CLIMATE_INDICES}")
        return None
    
    df = pd.read_csv(CLIMATE_INDICES)
    print(f"[✓] Loaded climate indices: {df.shape[0]} rows")
    return df

def add_seasonal_features(df):
    """Add seasonal features (month sine/cosine)"""
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    
    # Normalize year
    min_year = df['YEAR'].min()
    max_year = df['YEAR'].max()
    df['YEAR_NORM'] = (df['YEAR'] - min_year) / (max_year - min_year)
    
    return df

def create_sequences(X, y, seq_len):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    
    return np.array(X_seq), np.array(y_seq)

def build_dataset(seq_len=24, train_until=1970, val_until=1990, 
                  fit_new_scaler=False, train_test_ratio="80:20", 
                  val_ratio_within_train=0.2):
    """
    Build train/val/test datasets
    
    Parameters:
    - seq_len: Sequence length for LSTM
    - train_until: Year to split train/val
    - val_until: Year to split val/test
    - fit_new_scaler: If True, fit scaler on training data
    - train_test_ratio: Ratio for train:test split (e.g., "80:20")
    - val_ratio_within_train: Validation ratio within training data
    """
    
    print(f"\n[·] Building dataset with split_ratio={train_test_ratio}...")
    
    # Load data
    rainfall_df = load_rainfall_data()
    climate_df = load_climate_indices()
    
    # Merge climate indices
    if climate_df is not None:
        rainfall_df = rainfall_df.merge(climate_df, on=['YEAR', 'MONTH'], how='left')
        rainfall_df.fillna(rainfall_df.mean(), inplace=True)
    
    # Add features
    rainfall_df = add_seasonal_features(rainfall_df)
    
    # Sort by year and month
    rainfall_df = rainfall_df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
    
    # Define feature columns
    feature_cols = ['RAINFALL', 'ENSO', 'IOD', 'MONTH_SIN', 'MONTH_COS', 'YEAR_NORM']
    
    # Extract features and target
    X = rainfall_df[feature_cols].values
    y = rainfall_df['RAINFALL'].values
    years = rainfall_df['YEAR'].values
    subdivisions = rainfall_df['SUBDIVISION'].unique().tolist()
    
    # Fit or load scaler
    if fit_new_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # Save scaler
        import pickle
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[✓] Fit and saved scaler to {SCALER_PATH}")
    else:
        try:
            import pickle
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)
            print(f"[✓] Loaded scaler from {SCALER_PATH}")
        except:
            print("[!] Could not load scaler, fitting new one...")
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, seq_len)
    years_seq = years[seq_len-1:]
    
    print(f"[✓] Created sequences: X shape {X_seq.shape}, y shape {y_seq.shape}")
    
    # Split by ratio
    train_ratio, test_ratio = map(int, train_test_ratio.split(':'))
    total_ratio = train_ratio + test_ratio
    train_frac = train_ratio / total_ratio
    
    split_idx = int(len(X_seq) * train_frac)
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Split training into train/val
    val_idx = int(len(X_train) * (1 - val_ratio_within_train))
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_train, y_train = X_train[:val_idx], y_train[:val_idx]
    
    print(f"[✓] Split complete:")
    print(f"    Training:   {len(X_train)} samples")
    print(f"    Validation: {len(X_val)} samples")
    print(f"    Test:       {len(X_test)} samples")
    
    # Calculate threshold (75th percentile of training data)
    threshold = np.percentile(y_train, 75)
    print(f"[✓] Event threshold (75th percentile): {threshold:.2f} mm")
    
    # Save subdivisions
    with open(SUBDIVISIONS_PATH, 'w') as f:
        json.dump(subdivisions, f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions

# ============================================================================
# TEST THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  COLAB DATA PIPELINE TEST")
    print("="*80)
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions = \
            build_dataset(seq_len=24, train_test_ratio="80:20", fit_new_scaler=False)
        
        print("\n[✓] Pipeline test successful!")
        print(f"[✓] Subdivisions: {len(subdivisions)}")
        
    except Exception as e:
        print(f"\n[✗] Error: {str(e)}")
        import traceback
        traceback.print_exc()
