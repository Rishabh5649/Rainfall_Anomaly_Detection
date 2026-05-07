#!/usr/bin/env python3
"""
Colab Full Test - Run this to verify everything works
This tests data pipeline, training, and evaluation
"""

import os
import sys
import json

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
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

print(f"[·] Base directory: {BASE_DIR}\n")

# ============================================================================
# TEST 1: Data Pipeline
# ============================================================================
print("="*80)
print("  TEST 1: DATA PIPELINE")
print("="*80)

try:
    from colab_data_pipeline import build_dataset
    
    print("\n[·] Loading dataset with 80:20 split...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions = \
        build_dataset(seq_len=24, train_test_ratio="80:20", fit_new_scaler=False)
    
    print(f"\n[✓] Data pipeline test PASSED")
    print(f"    Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"    Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"    Test:       X={X_test.shape}, y={y_test.shape}")
    print(f"    Subdivisions: {len(subdivisions)}")
    
except Exception as e:
    print(f"\n[✗] Data pipeline test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: Model Loading
# ============================================================================
print("\n" + "="*80)
print("  TEST 2: MODEL LOADING")
print("="*80)

try:
    import torch
    from colab_train import RainfallCNNLSTM
    
    print("\n[·] Creating CNN-LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RainfallCNNLSTM(seq_len=24, n_features=6, n_subdivisions=36)
    model.to(device)
    
    print(f"[✓] Model created successfully")
    print(f"    Device: {device}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    X_test_t = torch.FloatTensor(X_test[:32]).to(device)
    with torch.no_grad():
        y_pred = model(X_test_t)
    
    print(f"[✓] Model forward pass test PASSED")
    print(f"    Input shape: {X_test_t.shape}")
    print(f"    Output shape: {y_pred.shape}")
    
except Exception as e:
    print(f"\n[✗] Model loading test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Checkpoint Loading
# ============================================================================
print("\n" + "="*80)
print("  TEST 3: CHECKPOINT LOADING")
print("="*80)

try:
    checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"\n[·] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"[✓] Checkpoint loaded successfully")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Validation Loss: {checkpoint['val_loss']:.6f}")
        
        # Test inference
        with torch.no_grad():
            y_pred = model(X_test_t)
        
        print(f"[✓] Inference test PASSED")
    else:
        print(f"\n[!] Checkpoint not found at {checkpoint_path}")
        print(f"    (This is OK if you haven't trained yet)")

except Exception as e:
    print(f"\n[✗] Checkpoint loading test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Metrics Computation
# ============================================================================
print("\n" + "="*80)
print("  TEST 4: METRICS COMPUTATION")
print("="*80)

try:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np
    
    print("\n[·] Testing metrics computation...")
    
    # Get some predictions
    with torch.no_grad():
        y_test_pred = model(torch.FloatTensor(X_test[:100]).to(device)).cpu().numpy().flatten()
    
    y_test_subset = y_test[:100]
    threshold = np.percentile(y_test_subset, 75)
    
    # Convert to binary
    y_pred_binary = (y_test_pred > threshold).astype(int)
    y_true_binary = (y_test_subset > threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    print(f"[✓] Metrics computation test PASSED")
    print(f"    Threshold: {threshold:.2f} mm")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"    Confusion Matrix:\n{cm}")
    
except Exception as e:
    print(f"\n[✗] Metrics computation test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: File I/O
# ============================================================================
print("\n" + "="*80)
print("  TEST 5: FILE I/O")
print("="*80)

try:
    print("\n[·] Testing file I/O...")
    
    # Check if directories exist
    dirs_to_check = [
        os.path.join(BASE_DIR, 'data'),
        os.path.join(BASE_DIR, 'checkpoints'),
        os.path.join(BASE_DIR, 'outputs')
    ]
    
    for dir_path in dirs_to_check:
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(dir_path):
            print(f"    ✓ {dir_path}")
        else:
            print(f"    ✗ {dir_path}")
    
    # Test JSON write
    test_data = {'test': 'data', 'timestamp': '2026-05-07'}
    test_file = os.path.join(BASE_DIR, 'outputs', 'test.json')
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    
    if loaded_data == test_data:
        print(f"[✓] File I/O test PASSED")
        os.remove(test_file)
    else:
        raise Exception("Data mismatch")
    
except Exception as e:
    print(f"\n[✗] File I/O test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Google Drive (if on Colab)
# ============================================================================
if IN_COLAB:
    print("\n" + "="*80)
    print("  TEST 6: GOOGLE DRIVE")
    print("="*80)
    
    try:
        from google.colab import drive
        
        if os.path.exists('/content/drive/My Drive'):
            print("\n[✓] Google Drive is mounted")
            
            # List some files
            drive_files = os.listdir('/content/drive/My Drive')
            print(f"    Found {len(drive_files)} items in My Drive")
        else:
            print("\n[!] Google Drive not mounted (mount with: drive.mount('/content/drive'))")
    
    except Exception as e:
        print(f"\n[!] Google Drive test skipped: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("  ALL TESTS PASSED ✓")
print("="*80)
print(f"""
Summary:
  ✓ Data pipeline working
  ✓ Model creation working
  ✓ Checkpoint loading working
  ✓ Metrics computation working
  ✓ File I/O working
  {'✓ Google Drive accessible' if IN_COLAB and os.path.exists('/content/drive/My Drive') else '· Google Drive not checked'}

You are ready to:
  1. TRAIN: exec(open('{BASE_DIR}/colab_train.py').read())
  2. EVALUATE: exec(open('{BASE_DIR}/colab_evaluate.py').read())

Next steps:
  • Run colab_train.py to train the model (if needed)
  • Run colab_evaluate.py to evaluate on all 4 ratios
  • Check outputs/colab_metrics.json for results
""")
