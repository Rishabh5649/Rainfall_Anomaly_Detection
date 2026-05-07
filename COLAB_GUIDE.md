# Google Colab Setup Guide - CNN-LSTM Rainfall Prediction

This guide shows you how to run the complete CNN-LSTM rainfall prediction pipeline on Google Colab.

## 📋 Files Created for Colab

| File | Purpose |
|------|---------|
| **colab_data_pipeline.py** | Data loading, feature engineering, train/val/test splitting for all 4 ratios |
| **colab_train.py** | Model training with validation and early stopping (GPU enabled) |
| **colab_evaluate.py** | Testing on all 4 split ratios and computing 8 classification metrics |
| **colab_setup.py** | Complete setup guide with cell-by-cell instructions |
| **colab_test_all.py** | Verification script to test all components work |

## 🚀 Quick Start (Copy-Paste)

### Step 1: Create a new Google Colab notebook
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### Step 2: Mount Google Drive
In the first cell, paste and run:
```python
from google.colab import drive
drive.mount('/content/drive')
print("[✓] Google Drive mounted!")
```

### Step 3: Clone the repository
In the next cell:
```python
!git clone https://github.com/yourusername/cnn_lstm_rainfall_system.git /content/cnn_lstm_rainfall_system
```

### Step 4: Install dependencies
```python
!pip install -q torch scikit-learn pandas numpy matplotlib
```

### Step 5: Run verification
```python
exec(open('/content/cnn_lstm_rainfall_system/colab_test_all.py').read())
```

This will test all components and show you if everything is ready.

### Step 6: Train (optional)
```python
exec(open('/content/cnn_lstm_rainfall_system/colab_train.py').read())
```

Run with custom parameters:
```python
import sys
sys.argv = ['colab_train.py', '--epochs', '50', '--split_ratio', '80:20']
exec(open('/content/cnn_lstm_rainfall_system/colab_train.py').read())
```

### Step 7: Evaluate on all 4 ratios
```python
exec(open('/content/cnn_lstm_rainfall_system/colab_evaluate.py').read())
```

### Step 8: View results
```python
import json

with open('/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json', 'r') as f:
    results = json.load(f)

# Print summary
for ratio in ["60:40", "70:30", "80:20", "90:10"]:
    test = results['ratios'][ratio]['splits']['test']
    print(f"{ratio}: Accuracy={test['accuracy']:.4f}, F1={test['f1_score']:.4f}, AUC={test['auc']:.4f}")
```

## 📊 Data Flow

```
Google Drive / GitHub
        ↓
   colab_data_pipeline.py
   • Load rainfall data
   • Load climate indices
   • Add seasonal features
   • Scale with StandardScaler
   • Create 24-month sequences
   • Split by ratio (60:40 / 70:30 / 80:20 / 90:10)
        ↓
   colab_train.py
   • Create CNN-LSTM model
   • Training loop with validation
   • Early stopping (patience=15)
   • Save best checkpoint
        ↓
   colab_evaluate.py
   • Load trained model
   • Predict on test sets
   • Convert regression → binary classification
   • Compute 8 metrics per ratio:
     - Accuracy, Precision, Recall
     - Sensitivity, F1-Score, AUC
     - Error Rate, Confusion Matrix
   • Save to outputs/colab_metrics.json
        ↓
   Save to Google Drive
```

## 🔧 Configuration Options

### Training Parameters
```python
# Edit colab_train.py main section:
train(
    epochs=80,           # Number of training epochs
    batch_size=256,      # Batch size for training
    lr=3e-3,            # Learning rate
    weight_decay=1e-4,  # L2 regularization
    patience=15,        # Early stopping patience
    seq_len=24,         # Sequence length (24 months)
    split_ratio="80:20" # Train:test ratio
)
```

### Evaluation Parameters
```python
# Edit colab_evaluate.py main section:
evaluate_all_ratios(
    batch_size=512,
    split_ratios=["60:40", "70:30", "80:20", "90:10"]
)
```

## 📁 Output Files

After running evaluation, you'll get:
- **outputs/colab_metrics.json** - All metrics for all 4 ratios
- **checkpoints/best_model.pt** - Trained model checkpoint
- **checkpoints/train_history.json** - Training history (loss curves)

### Metrics JSON Structure
```json
{
  "checkpoint_epoch": 22,
  "checkpoint_val_loss": 0.16450,
  "default_ratio": "80:20",
  "available_ratios": ["60:40", "70:30", "80:20", "90:10"],
  "ratios": {
    "80:20": {
      "threshold_mm": 123.45,
      "splits": {
        "train": { "accuracy": 0.8234, "precision": 0.7891, ... },
        "val": { ... },
        "test": { "accuracy": 0.7981, "f1_score": 0.7800, ... }
      }
    },
    ...
  }
}
```

## ⚙️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'colab_data_pipeline'"
**Solution**: Make sure you've cloned the repository and added it to sys.path:
```python
import sys
sys.path.insert(0, '/content/cnn_lstm_rainfall_system')
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch_size:
```python
train(batch_size=128, ...)  # Instead of 256
evaluate_all_ratios(batch_size=256)  # Instead of 512
```

### Issue: "Model not found at checkpoints/best_model.pt"
**Solution**: You need to train the model first using colab_train.py

### Issue: "Google Drive not mounted"
**Solution**: Mount it first:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 💾 Saving to Google Drive

### Option 1: Auto-save (easier)
```python
# At the end of evaluation, results auto-save:
if IN_COLAB:
    import shutil
    shutil.copy(
        '/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json',
        '/content/drive/My Drive/colab_metrics.json'
    )
```

### Option 2: Download files manually
Click the folder icon in Colab → select files → download

## 🎯 Performance Expectations

### Expected Runtimes (with GPU)
- **Data loading**: ~2-3 seconds
- **Model initialization**: ~1 second
- **Training (80 epochs)**: ~15-20 minutes
- **Evaluation (4 ratios)**: ~2-3 minutes
- **Total pipeline**: ~20-30 minutes

### Expected Accuracies
Based on the 80:20 split:
- **Test Accuracy**: ~79-80%
- **Test F1-Score**: ~78-79%
- **Test AUC**: ~86%

## 📈 Visualization Example

```python
import matplotlib.pyplot as plt
import json

with open('/content/cnn_lstm_rainfall_system/outputs/colab_metrics.json') as f:
    metrics = json.load(f)

# Plot 8 metrics for 80:20 split
test = metrics['ratios']['80:20']['splits']['test']

fig, axes = plt.subplots(2, 4, figsize=(15, 6))

metrics_to_plot = [
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1_score', 'F1-Score'),
    ('auc', 'AUC'),
    ('sensitivity', 'Sensitivity'),
    ('error_rate', 'Error Rate'),
]

for idx, (key, label) in enumerate(metrics_to_plot):
    ax = axes[idx // 4, idx % 4]
    value = test[key]
    ax.bar(['80:20'], [value], color='#3498db')
    ax.set_ylim([0, 1])
    ax.set_ylabel(label)
    ax.text(0, value + 0.05, f'{value:.4f}', ha='center')

plt.suptitle('Classification Metrics - 80:20 Split (Test Set)')
plt.tight_layout()
plt.show()
```

## 🔗 File Dependencies

```
colab_data_pipeline.py
  ↓ imports
  → pandas, numpy, sklearn.preprocessing, pickle

colab_train.py
  ↓ imports
  → torch, torch.nn, torch.optim
  → colab_data_pipeline (build_dataset)

colab_evaluate.py
  ↓ imports
  → torch, torch.nn
  → sklearn.metrics
  → colab_data_pipeline (build_dataset)
```

## 📝 Notes

1. **Colab Path Handling**: All scripts automatically detect if running on Colab and set paths accordingly
2. **GPU Support**: CUDA is auto-detected; trains on GPU if available (much faster)
3. **Checkpoint**: Best model is saved after each epoch if validation loss improves
4. **Split Ratios**: All 4 ratios use the same pre-trained checkpoint (from 80:20 split)
5. **Data**: Uses your local `data/` folder; make sure it's synced to GitHub or uploaded to Drive

## 🚨 Important

- **Before running**: Ensure your GitHub repo is public or provide credentials
- **GPU Quota**: Colab has limits on free GPU usage (usually ~12 hours/week)
- **Large Files**: If model checkpoint is >100MB, you may need Colab Pro
- **Data Storage**: Keep your `data/` folder accessible (GitHub or Google Drive)

## ✅ Verification Checklist

After setup, verify:
- [ ] Repository cloned successfully
- [ ] All dependencies installed
- [ ] colab_test_all.py runs without errors
- [ ] Data pipeline loads all splits successfully
- [ ] Model creates and runs forward pass
- [ ] Checkpoint loads correctly
- [ ] Metrics compute without errors
- [ ] Results save to Google Drive

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Run `colab_test_all.py` to identify which component failed
3. Check file paths in BASE_DIR
4. Ensure all required packages are installed
5. Verify Google Drive is mounted (if needed)

---

**Created**: 2026-05-07  
**Last Updated**: 2026-05-07  
**Status**: Ready for Google Colab
