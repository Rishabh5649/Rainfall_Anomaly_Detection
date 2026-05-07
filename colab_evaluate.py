"""
Colab-Compatible Evaluation Script for CNN-LSTM Rainfall Prediction
Tests model on all 4 split ratios and computes 8 classification metrics
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
import warnings

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
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL ARCHITECTURE (SAME AS TRAINING)
# ============================================================================

class RainfallCNNLSTM(nn.Module):
    """CNN-LSTM model for rainfall prediction"""
    
    def __init__(self, seq_len=24, n_features=6, n_subdivisions=36):
        super(RainfallCNNLSTM, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, 
                           batch_first=True, dropout=0.3)
        
        # Fully connected
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_metrics(y_true, y_pred_proba, threshold):
    """Compute all 8 classification metrics"""
    
    # Convert to binary
    y_pred = (y_pred_proba > threshold).astype(int)
    y_true_binary = (y_true > threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Metrics
    accuracy = accuracy_score(y_true_binary, y_pred)
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    sensitivity = recall  # Same as recall
    
    # AUC
    try:
        auc = roc_auc_score(y_true_binary, y_pred_proba)
    except:
        auc = 0.0
    
    # Error rate
    error_rate = 1 - accuracy
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'f1_score': float(f1),
        'auc': float(auc),
        'error_rate': float(error_rate),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'n': int(len(y_true))
    }

def evaluate_all_ratios(seq_len=24, batch_size=512, split_ratios=None):
    """Evaluate model on all split ratios"""
    
    if split_ratios is None:
        split_ratios = ["60:40", "70:30", "80:20", "90:10"]
    
    print("\n" + "="*80)
    print("  COLAB EVALUATION - All Split Ratios")
    print("="*80)
    
    # Load model
    print("\n[·] Loading trained model...")
    model_path = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first!")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = RainfallCNNLSTM(seq_len=seq_len, n_features=6, n_subdivisions=36)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")
    
    # Import data pipeline
    from colab_data_pipeline import build_dataset
    
    all_results = {
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_val_loss': float(checkpoint['val_loss']),
        'default_ratio': '80:20',
        'available_ratios': split_ratios,
        'ratios': {}
    }
    
    # Evaluate each ratio
    for ratio in split_ratios:
        print(f"\n[·] Evaluating split ratio {ratio}")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions = \
            build_dataset(seq_len=seq_len, train_test_ratio=ratio, fit_new_scaler=False)
        
        # Calculate threshold (75th percentile of training data)
        threshold = np.percentile(y_train, 75)
        print(f"    Event threshold (75th %ile): {threshold:.2f} mm")
        
        # Make predictions
        with torch.no_grad():
            # Training
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_pred = model(X_train_t).cpu().numpy().flatten()
            
            # Validation
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_pred = model(X_val_t).cpu().numpy().flatten()
            
            # Test
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_test_pred = model(X_test_t).cpu().numpy().flatten()
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred, threshold)
        val_metrics = compute_metrics(y_val, y_val_pred, threshold)
        test_metrics = compute_metrics(y_test, y_test_pred, threshold)
        
        all_results['ratios'][ratio] = {
            'threshold_mm': float(threshold),
            'splits': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
        }
        
        # Print results
        print(f"    Train | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"    Val   | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"    Test  | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_score']:.4f} | AUC: {test_metrics['auc']:.4f}")
    
    # Save results
    results_path = os.path.join(BASE_DIR, 'outputs', 'colab_metrics.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[✓] Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY TABLE - TEST SPLIT")
    print("="*80)
    print(f"{'Ratio':<10} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'AUC':<10}")
    print("-" * 70)
    
    for ratio in split_ratios:
        metrics = all_results['ratios'][ratio]['splits']['test']
        print(f"{ratio:<10} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['auc']:<10.4f}")
    
    return all_results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CNN-LSTM model on Colab')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--ratios', type=str, default='60:40,70:30,80:20,90:10',
                       help='Split ratios to evaluate')
    
    args = parser.parse_args()
    split_ratios = args.ratios.split(',')
    
    try:
        results = evaluate_all_ratios(batch_size=args.batch_size, split_ratios=split_ratios)
        print("\n[✓] Evaluation complete!")
        
        # Save for Google Drive if in Colab
        if IN_COLAB:
            try:
                import shutil
                shutil.copy(
                    os.path.join(BASE_DIR, 'outputs', 'colab_metrics.json'),
                    '/content/drive/My Drive/colab_metrics.json'
                )
                print("[✓] Results saved to Google Drive!")
            except:
                print("[!] Could not save to Google Drive (not mounted?)")
    
    except Exception as e:
        print(f"\n[✗] Error: {str(e)}")
        import traceback
        traceback.print_exc()
