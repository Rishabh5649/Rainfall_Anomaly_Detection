#!/usr/bin/env python3
"""
🌧️ RainSight - Complete Colab Notebook Setup
Copy & paste this entire script into a Google Colab cell
Or use the individual commands below
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')
print("[✓] Google Drive mounted!")
"""

# ============================================================================
# CELL 2: Clone Repository or Upload Files
# ============================================================================
"""
# Option A: Clone from GitHub (recommended)
!git clone https://github.com/yourusername/cnn_lstm_rainfall_system.git /content/cnn_lstm_rainfall_system

# Option B: Or use files from Google Drive
import shutil
shutil.copy('/content/drive/My Drive/cnn_lstm_rainfall_system.zip', '/content/')
!unzip /content/cnn_lstm_rainfall_system.zip -d /content/

print("[✓] Files ready!")
"""

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================
"""
!pip install -q torch scikit-learn pandas numpy matplotlib

import torch
import sys
print(f"[✓] PyTorch version: {torch.__version__}")
print(f"[✓] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[✓] GPU: {torch.cuda.get_device_name(0)}")
"""

# ============================================================================
# CELL 4: Test Data Pipeline
# ============================================================================
"""
import os
os.chdir('/content/cnn_lstm_rainfall_system')
sys.path.insert(0, '/content/cnn_lstm_rainfall_system')

from colab_data_pipeline import build_dataset

print("[·] Testing data pipeline...")
X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions = \
    build_dataset(seq_len=24, train_test_ratio="80:20", fit_new_scaler=False)

print(f"[✓] Data shapes:")
print(f"    Training:   X={X_train.shape}, y={y_train.shape}")
print(f"    Validation: X={X_val.shape}, y={y_val.shape}")
print(f"    Test:       X={X_test.shape}, y={y_test.shape}")
print(f"[✓] Subdivisions: {len(subdivisions)}")
"""

# ============================================================================
# CELL 5: Train Model (OPTIONAL - if you have new data)
# ============================================================================
"""
from colab_train import train

print("[·] Starting training...")
model, history = train(
    epochs=80,
    batch_size=256,
    lr=3e-3,
    split_ratio="80:20"
)

print("[✓] Training complete!")
print(f"[✓] Model saved to: /content/cnn_lstm_rainfall_system/checkpoints/best_model.pt")

# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history['train_losses'], label='Training Loss')
plt.plot(history['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training History')
plt.legend()
plt.grid()
plt.show()
"""

# ============================================================================
# CELL 6: Evaluate Model on All 4 Ratios
# ============================================================================
"""
from colab_evaluate import evaluate_all_ratios

print("[·] Evaluating model on all split ratios...")
results = evaluate_all_ratios(
    batch_size=512,
    split_ratios=["60:40", "70:30", "80:20", "90:10"]
)

print("[✓] Evaluation complete!")
"""

# ============================================================================
# CELL 7: View Results
# ============================================================================
"""
import json

# Load results
with open('/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json', 'r') as f:
    metrics = json.load(f)

# Display summary
print("\n" + "="*80)
print("EVALUATION RESULTS - TEST SPLIT")
print("="*80)
print(f"{'Ratio':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'AUC':<10}")
print("-" * 80)

for ratio in ["60:40", "70:30", "80:20", "90:10"]:
    test = metrics['ratios'][ratio]['splits']['test']
    print(f"{ratio:<10} {test['accuracy']:.4f}{' '*7} {test['precision']:.4f}{' '*7} "
          f"{test['recall']:.4f}{' '*7} {test['f1_score']:.4f}{' '*5} {test['auc']:.4f}")

# Detailed metrics for 80:20 (default)
print("\n" + "="*80)
print("DETAILED METRICS - 80:20 SPLIT (TEST)")
print("="*80)
test_metrics = metrics['ratios']['80:20']['splits']['test']
print(f"✓ Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
print(f"✓ Precision:   {test_metrics['precision']:.4f}")
print(f"✓ Recall:      {test_metrics['recall']:.4f}")
print(f"✓ Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"✓ F1-Score:    {test_metrics['f1_score']:.4f}")
print(f"✓ AUC:         {test_metrics['auc']:.4f}")
print(f"✓ Error Rate:  {test_metrics['error_rate']:.4f} ({test_metrics['error_rate']*100:.2f}%)")
print(f"\n✓ Confusion Matrix (Test):")
cm = test_metrics['confusion_matrix']
print(f"    Predicted Neg  Predicted Pos")
print(f"Actual Neg    {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Pos    {cm[1][0]:<14} {cm[1][1]}")
"""

# ============================================================================
# CELL 8: Save Results to Google Drive
# ============================================================================
"""
import shutil
import os

# Save metrics
shutil.copy(
    '/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json',
    '/content/drive/My Drive/colab_metrics.json'
)

# Save model checkpoint
shutil.copy(
    '/content/cnn_lstm_rainfall_system/checkpoints/best_model.pt',
    '/content/drive/My Drive/best_model.pt'
)

# Save training history
shutil.copy(
    '/content/cnn_lstm_rainfall_system/checkpoints/train_history.json',
    '/content/drive/My Drive/train_history.json'
)

print("[✓] All files saved to Google Drive!")
print("[✓] Check 'My Drive/colab_metrics.json' for results")
"""

# ============================================================================
# CELL 9: Compare All Ratios Visually
# ============================================================================
"""
import matplotlib.pyplot as plt
import json

with open('/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract data
ratios = ["60:40", "70:30", "80:20", "90:10"]
accuracies = []
f1_scores = []
aucs = []

for ratio in ratios:
    test = metrics['ratios'][ratio]['splits']['test']
    accuracies.append(test['accuracy'])
    f1_scores.append(test['f1_score'])
    aucs.append(test['auc'])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
axes[0].bar(ratios, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy by Split Ratio')
axes[0].set_ylim([0.7, 0.85])
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center')

# F1-Score
axes[1].bar(ratios, f1_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
axes[1].set_ylabel('F1-Score')
axes[1].set_title('F1-Score by Split Ratio')
axes[1].set_ylim([0.7, 0.85])
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center')

# AUC
axes[2].bar(ratios, aucs, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
axes[2].set_ylabel('AUC')
axes[2].set_title('AUC by Split Ratio')
axes[2].set_ylim([0.8, 0.9])
for i, v in enumerate(aucs):
    axes[2].text(i, v + 0.005, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('/content/cnn_lstm_rainfall_system/outputs/colab_results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("[✓] Comparison plot saved!")
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================
"""
QUICK START - Just run these commands in order:

1. MOUNT DRIVE:
   from google.colab import drive; drive.mount('/content/drive')

2. CLONE CODE:
   !git clone https://github.com/yourusername/cnn_lstm_rainfall_system.git /content/cnn_lstm_rainfall_system

3. INSTALL PACKAGES:
   !pip install -q torch scikit-learn pandas numpy matplotlib

4. TEST PIPELINE:
   exec(open('/content/cnn_lstm_rainfall_system/colab_test_all.py').read())

5. EVALUATE MODEL:
   exec(open('/content/cnn_lstm_rainfall_system/colab_evaluate.py').read())

6. VIEW RESULTS:
   with open('/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json') as f:
       import json; print(json.dumps(json.load(f), indent=2))

TRAINING (Optional):
   exec(open('/content/cnn_lstm_rainfall_system/colab_train.py').read())
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║                    🌧️  RAINSIGHT COLAB SETUP                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Copy and paste the cells above into Google Colab.               ║
║  Each cell is marked with a number.                              ║
║  Run them in order!                                              ║
║                                                                    ║
║  Files created:                                                   ║
║  ✓ colab_data_pipeline.py  - Data loading & splitting            ║
║  ✓ colab_train.py          - Training with validation            ║
║  ✓ colab_evaluate.py       - Testing on all 4 ratios             ║
║  ✓ colab_setup.py          - This file (contains all cells)      ║
║                                                                    ║
║  Key Differences from Local:                                      ║
║  • Automatic path handling for Colab                             ║
║  • GPU support enabled by default                                ║
║  • Auto-save to Google Drive option                              ║
║  • No virtual environment needed                                 ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
