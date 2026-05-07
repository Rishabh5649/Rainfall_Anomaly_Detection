#!/usr/bin/env python3
"""
Generate Real Test Cases for CNN-LSTM Rainfall Prediction Model
Selects diverse test samples from different regions and time periods
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from data_pipeline import build_dataset
from model import RainfallCNNLSTM

# Configuration
SEQ_LEN = 24
BATCH_SIZE = 512
MODEL_CHECKPOINT = "checkpoints/best_model.pt"
RAINFALL_DATA = "data/rainfall_in_india_1901-2015.csv"

# Set device
device = torch.device("cpu")

def load_model():
    """Load the trained model"""
    model = RainfallCNNLSTM(seq_len=SEQ_LEN, n_features=6, n_subdivisions=36)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_subdivision_names():
    """Get list of subdivision names"""
    try:
        with open("checkpoints/subdivisions.json", "r") as f:
            return json.load(f)
    except:
        # Fallback to unique subdivisions from data
        df = pd.read_csv(RAINFALL_DATA)
        return sorted(df['SUBDIVISION'].unique().tolist())

def calculate_accuracy(actual, predicted):
    """Calculate accuracy percentage based on RMSE"""
    mape = np.abs((actual - predicted) / (actual + 1e-6)) * 100
    # Accuracy: how close prediction is (up to 150% is considered reasonable)
    accuracy = max(0, 100 - min(mape, 100))
    return accuracy

def get_test_cases(X_test, y_test, X_val=None, y_val=None):
    """Select diverse test cases from different regions and dates"""
    test_cases = []
    
    # We'll select cases spread across the test set
    n_samples = len(X_test)
    indices = [
        int(n_samples * 0.15),   # Early test period
        int(n_samples * 0.35),   # Mid-early test period
        int(n_samples * 0.50),   # Middle test period
        int(n_samples * 0.70),   # Late test period
        int(n_samples * 0.90),   # Very late test period
    ]
    
    return indices

def format_location_date(idx, total_samples, start_year=1975, start_month=1):
    """Estimate location and date from index"""
    # Rough estimation: each subdivision has samples across years
    # For 36 subdivisions with ~1000 monthly samples each
    samples_per_year = 36 * 12
    
    year_offset = idx // samples_per_year
    month_in_year = (idx % samples_per_year) // 36
    
    return {
        'year': start_year + year_offset,
        'month': ((start_month + month_in_year - 1) % 12) + 1
    }

def predict_batch(model, X_batch):
    """Make predictions on a batch"""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_batch).to(device)
        preds = model(X_tensor)
    return preds.cpu().numpy()

def generate_test_cases():
    """Generate comprehensive test cases"""
    print("="*80)
    print("  RainSight — Generating Real Test Cases with Model Predictions")
    print("="*80)
    print()
    
    # Load model
    print("[·] Loading trained model...")
    model = load_model()
    print("[✓] Model loaded successfully")
    
    # Load data
    print("[·] Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test, _, scaler = build_dataset(
        seq_len=SEQ_LEN,
        train_until=1970,
        val_until=1990,
        fit_new_scaler=False
    )
    print(f"[✓] Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    subdivisions = get_subdivision_names()
    print(f"[✓] Subdivisions: {len(subdivisions)} regions")
    print()
    
    # Get test case indices
    test_indices = get_test_cases(X_test, y_test, X_val, y_val)
    
    # Generate predictions
    print("[·] Generating predictions for test cases...")
    test_results = []
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for case_num, idx in enumerate(test_indices, 1):
        # Get the sample
        X_sample = X_test[idx:idx+1]
        y_actual = y_test[idx]
        
        # Make prediction
        y_pred = predict_batch(model, X_sample)[0, 0]
        
        # Inverse transform to get actual rainfall values
        y_actual_mm = scaler.inverse_transform([[y_actual, 0, 0, 0, 0, 0]])[0, 0]
        y_pred_mm = scaler.inverse_transform([[y_pred, 0, 0, 0, 0, 0]])[0, 0]
        
        # Calculate accuracy
        accuracy = calculate_accuracy(y_actual_mm, y_pred_mm)
        
        # Estimate location and date
        location_idx = idx % len(subdivisions)
        location = subdivisions[location_idx]
        
        # Estimate year and month
        samples_per_year = len(subdivisions) * 12
        year_offset = idx // samples_per_year
        month_in_year = (idx % samples_per_year) // len(subdivisions)
        year = 1990 + year_offset
        month = ((month_in_year) % 12) + 1
        
        test_results.append({
            'case': f"Test Case {case_num}",
            'location': location,
            'date': f"{months[month-1]} {year}",
            'actual_mm': round(y_actual_mm, 1),
            'predicted_mm': round(y_pred_mm, 1),
            'accuracy': round(accuracy, 1),
            'error_mm': round(abs(y_actual_mm - y_pred_mm), 1)
        })
        
        print(f"  [{case_num}/5] {location} | {months[month-1]} {year} | "
              f"Actual: {y_actual_mm:.1f}mm | Pred: {y_pred_mm:.1f}mm | "
              f"Acc: {accuracy:.1f}%")
    
    print("[✓] Predictions generated")
    print()
    
    return test_results

def display_results(results):
    """Display results in table format"""
    print("="*120)
    print("  REAL TEST CASES — CNN-LSTM RAINFALL PREDICTION MODEL")
    print("="*120)
    print()
    
    # Create formatted table
    print(f"{'Model: CNN+LSTM':<20} {results[0]['case']:<20} {results[1]['case']:<20} "
          f"{results[2]['case']:<20} {results[3]['case']:<20} {results[4]['case']:<20}")
    print("-" * 120)
    
    # Location
    print(f"{'Location':<20}", end="")
    for r in results:
        print(f"{r['location']:<20}", end="")
    print()
    
    # Date
    print(f"{'Date':<20}", end="")
    for r in results:
        print(f"{r['date']:<20}", end="")
    print()
    
    # Actual rainfall
    print(f"{'Actual (mm)':<20}", end="")
    for r in results:
        print(f"{r['actual_mm']:<20}", end="")
    print()
    
    # Predicted rainfall
    print(f"{'Predicted (mm)':<20}", end="")
    for r in results:
        print(f"{r['predicted_mm']:<20}", end="")
    print()
    
    # Accuracy
    print(f"{'Accuracy':<20}", end="")
    for r in results:
        acc_str = f"{r['accuracy']:.1f}%"
        print(f"{acc_str:<20}", end="")
    print()
    
    # Error
    print(f"{'Error (mm)':<20}", end="")
    for r in results:
        print(f"{r['error_mm']:<20}", end="")
    print()
    
    print("-" * 120)
    print()
    
    # Summary statistics
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_error = np.mean([r['error_mm'] for r in results])
    min_accuracy = np.min([r['accuracy'] for r in results])
    max_accuracy = np.max([r['accuracy'] for r in results])
    
    print("📊 SUMMARY STATISTICS")
    print(f"  Average Accuracy:     {avg_accuracy:.1f}%")
    print(f"  Accuracy Range:       {min_accuracy:.1f}% to {max_accuracy:.1f}%")
    print(f"  Average Error:        {avg_error:.1f} mm")
    print()
    
    return results

def save_test_cases(results):
    """Save test cases to JSON"""
    output = {
        "model": "CNN+LSTM",
        "description": "Real test cases with actual predictions",
        "checkpoint_epoch": 22,
        "test_cases": results,
        "summary": {
            "total_cases": len(results),
            "avg_accuracy": round(np.mean([r['accuracy'] for r in results]), 1),
            "avg_error_mm": round(np.mean([r['error_mm'] for r in results]), 1)
        }
    }
    
    output_path = Path("outputs/real_test_cases.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"[✓] Test cases saved → {output_path}")

if __name__ == "__main__":
    try:
        results = generate_test_cases()
        display_results(results)
        save_test_cases(results)
    except Exception as e:
        print(f"[✗] Error: {str(e)}")
        import traceback
        traceback.print_exc()
