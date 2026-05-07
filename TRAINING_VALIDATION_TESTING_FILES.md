# 🎯 Training, Validation & Testing Files - Complete Map

## 📁 Project Root
```
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\
```

---

## 🔄 DATA PIPELINE (Lines 1-500+)
**File:** [`src/data_pipeline.py`](src/data_pipeline.py)

### What It Does:
- Loads rainfall data from CSV
- Creates train/validation/test splits
- Builds sequences for LSTM
- Scales/normalizes data

### Key Functions:

#### 1. **`load_rainfall_data()` (Lines ~30-50)**
   - **Input:** `data/rainfall_in_india_1901-2015.csv`
   - **Output:** DataFrame with all rainfall records
   - **Action:** Reads 115 years of monthly rainfall data for 36 subdivisions

#### 2. **`load_climate_indices()` (Lines ~60-80)**
   - **Input:** `data/climate_indices.csv`
   - **Output:** ENSO, IOD climate data
   - **Action:** Merges climate features with rainfall

#### 3. **`build_dataset()` (Lines ~200-280)**
   - **Purpose:** Main function that orchestrates entire pipeline
   - **Parameters:** 
     - `seq_len=24` (24-month sequences)
     - `train_until=1970` (splits by year)
     - `val_until=1990`
     - `split_ratio="80:20"` (train:test ratio)
   - **Output:** Returns 8 items:
     ```python
     X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions
     ```
   - **Line Numbers:** 200-280

#### 4. **`split_by_ratio_years()` (Lines ~393-442)**
   - **Purpose:** Creates train/val/test splits based on ratio
   - **Supports:** 60:40, 70:30, 80:20, 90:10
   - **Example:** 80:20 splits at year 1970 for train/test boundary

### Data Flow:
```
rainfall_in_india_1901-2015.csv (4116 rows)
          ↓
  load_rainfall_data()
          ↓
climate_indices.csv
          ↓
  merge features
          ↓
  build_dataset()
          ↓
  split_by_ratio_years()
          ↓
  create_sequences() → SEQ_LEN=24
          ↓
  X_train, y_train, X_val, y_val, X_test, y_test
          ↓
  apply_scaler()
          ↓
  Ready for training!
```

---

## 🏋️ TRAINING PHASE
**File:** [`src/train.py`](src/train.py)

### What It Does:
- Trains the CNN-LSTM model
- Uses validation data to monitor progress
- Saves best checkpoint
- Applies early stopping

### Key Function: `train()` (Lines ~50-150)

#### Signature:
```python
def train(
    epochs=80,
    batch_size=256,
    lr=3e-3,
    weight_decay=1e-4,
    patience=15,  # Early stopping
    seq_len=24,
    split_ratio="80:20"  # ← NEW: Choose ratio
)
```

#### Process:
1. **Line ~60:** Calls `build_dataset(split_ratio=split_ratio)`
   - Gets X_train, y_train, X_val, y_val from data_pipeline.py

2. **Line ~80:** Creates DataLoader
   - Batch size: 256
   - Shuffles training data

3. **Line ~100-130:** Training loop
   - For each epoch:
     - **TRAINING STEP:** Forward pass on X_train → predict y_train
     - **VALIDATION STEP:** Forward pass on X_val → predict y_val
     - **Early Stopping:** If val loss doesn't improve for 15 epochs, stop

4. **Line ~140:** Saves best model
   - **Saves to:** `checkpoints/best_model.pt`
   - **Contains:** Model weights, optimizer state, epoch, val_loss

### Output Files Created:
- ✅ **`checkpoints/best_model.pt`** (Model weights)
- ✅ **`checkpoints/train_log.json`** (Epoch-by-epoch metrics)
- ✅ **`checkpoints/train_status.json`** (Current training status)
- ✅ **`checkpoints/subdivisions.json`** (List of regions)

---

## 🔍 VALIDATION PHASE (Integrated in Training)
**File:** [`src/train.py`](src/train.py) - Lines ~100-130

### When It Happens:
- **Every epoch** during training
- Uses X_val, y_val from data_pipeline.py

### Metrics Computed:
- Validation Loss (MSE)
- Validation RMSE
- Validation MAE

### Early Stopping Logic (Lines ~115-120):
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    # Save checkpoint
else:
    patience_counter += 1
    if patience_counter >= patience:
        # Stop training
```

---

## 🧪 TESTING PHASE
**File:** [`src/evaluate.py`](src/evaluate.py)

### What It Does:
- Loads trained model from checkpoint
- Makes predictions on X_test
- Computes all 8 classification metrics
- Computes regression metrics
- Generates confusion matrices

### Key Function: `run_evaluation()` (Lines ~50-150)

#### Signature:
```python
def run_evaluation(
    seq_len=24,
    batch_size=512,
    split_ratios=["60:40", "70:30", "80:20", "90:10"],
    event_percentile=75.0
)
```

#### Process:

1. **Line ~60:** Load model from `checkpoints/best_model.pt`

2. **For each split_ratio (4 ratios):**
   - **Line ~70:** Call `build_dataset(split_ratio=ratio)`
     - Gets X_test, y_test, threshold for that ratio

3. **Line ~80:** Make predictions
   - Forward pass: X_test → y_test_pred
   - Output: Regression predictions (mm of rainfall)

4. **Line ~90:** Convert to classification
   - If y_pred > threshold → "Heavy rainfall event" (1)
   - If y_pred ≤ threshold → "Normal" (0)

5. **Line ~100-140:** Compute 8 metrics
   - **Accuracy:** (TP + TN) / Total
   - **Precision:** TP / (TP + FP)
   - **Recall:** TP / (TP + FN)
   - **F1-Score:** 2 * (Precision * Recall) / (Precision + Recall)
   - **Sensitivity:** Same as Recall
   - **AUC:** Area under ROC curve
   - **Error Rate:** 1 - Accuracy
   - **Confusion Matrix:** [[TN, FP], [FN, TP]]

6. **Line ~150:** Generate metrics.json
   - **For each ratio:**
     - Test split metrics
     - Validation split metrics
     - Training split metrics

### Output: `outputs/metrics.json`
```json
{
  "ratios": {
    "60:40": {
      "threshold_mm": 156.8,
      "splits": {
        "test": {
          "accuracy": 0.7732,
          "f1_score": 0.7516,
          "precision": 0.7623,
          "recall": 0.7412,
          "sensitivity": 0.7412,
          "auc": 0.8392,
          "error_rate": 0.2268,
          "confusion_matrix": [[358, 62], [125, 167]],
          "n": 712
        }
      }
    }
    // ... more ratios
  }
}
```

---

## 📊 COMPLETE DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA FILES                           │
├─────────────────────────────────────────────────────────────────┤
│ data/rainfall_in_india_1901-2015.csv (4116 rows)                │
│ data/climate_indices.csv                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ↓
        ┌─────────────────────────┐
        │  src/data_pipeline.py   │
        │   build_dataset()       │
        │  (Lines 200-280)        │
        └────────┬────────────────┘
                 │
       ┌─────────┴────────┬──────────────┐
       ↓                  ↓              ↓
    X_train          X_val          X_test
    y_train          y_val          y_test
    (80% or more)  (10% or 30%)  (10-20%)
       │                │              │
       │                │              │
       ├─TRAIN PHASE───┤              │
       │                │              │
       ↓                ↓              │
  ┌──────────────────────────────┐    │
  │   src/train.py              │    │
  │   train() function          │    │
  │   (Lines 50-150)            │    │
  │                              │    │
  │ For each epoch:             │    │
  │ 1. Forward pass X_train     │    │
  │ 2. Compute loss on y_train  │    │
  │ 3. Backward pass            │    │
  │ 4. Validate on X_val, y_val │    │
  │ 5. Early stopping check     │    │
  │ 6. Save best model          │    │
  └────┬─────────────────────────┘    │
       │                              │
       ↓                              │
  checkpoints/best_model.pt    ← TRAINING OUTPUT
       │                              │
       │                              │
       └──────────┬───────────────────┘
                  │
                  ↓
         ┌──────────────────┐
         │  src/evaluate.py │
         │ run_evaluation() │
         │ (Lines 50-150)   │
         │                  │
         │ Load model       │
         │ X_test → y_pred  │
         │ Convert to class │
         │ Compute 8 metrics│
         └────┬─────────────┘
              │
              ↓
     outputs/metrics.json  ← TESTING OUTPUT
     (Test split results)
```

---

## 🎯 SPECIFIC LINE NUMBERS - QUICK REFERENCE

### Training & Validation (src/train.py)
| What | Where | Lines |
|------|-------|-------|
| Import dependencies | Top of file | 1-20 |
| Train function signature | Function definition | ~50 |
| Load data | Inside train() | ~60 |
| Create DataLoader | Training setup | ~80 |
| Training loop | Main loop | ~100-130 |
| Validation inside epoch | Epoch loop | ~110-115 |
| Early stopping logic | Epoch loop | ~115-120 |
| Save checkpoint | After epoch | ~140 |

### Testing & Evaluation (src/evaluate.py)
| What | Where | Lines |
|------|-------|-------|
| Load model checkpoint | Function start | ~60 |
| Build test dataset | For each ratio | ~70 |
| Make predictions | Test loop | ~80 |
| Compute confusion matrix | Metrics loop | ~100 |
| Compute accuracy | Metrics loop | ~105 |
| Compute F1, precision, recall | Metrics loop | ~110-115 |
| Compute AUC | Metrics loop | ~120 |
| Save metrics.json | End of function | ~150 |

### Data Pipeline (src/data_pipeline.py)
| What | Where | Lines |
|------|-------|-------|
| Load rainfall data | Function | ~30-50 |
| Load climate indices | Function | ~60-80 |
| Create features | Function | ~100-150 |
| Build sequences (LSTM) | Function | ~160-200 |
| Main build_dataset() | Function | ~200-280 |
| Split by ratio | Function | ~393-442 |

---

## 📂 FILE SUMMARY

### Input Files (Read-Only)
- `data/rainfall_in_india_1901-2015.csv` ← Raw data
- `data/climate_indices.csv` ← Climate features

### Processing Files (Code)
- `src/data_pipeline.py` ← Data loading & splitting
- `src/train.py` ← Training with validation
- `src/evaluate.py` ← Testing & evaluation

### Checkpoint Files (Saved During Training)
- `checkpoints/best_model.pt` ← Trained model weights
- `checkpoints/train_log.json` ← Training history
- `checkpoints/train_status.json` ← Training metadata
- `checkpoints/subdivisions.json` ← Region list

### Output Files (Generated After Evaluation)
- `outputs/metrics.json` ← Test results for all 4 ratios
- `outputs/plots/` ← Visualization plots (if any)

---

## 🔗 HOW THEY CONNECT

### Training Workflow:
```
1. python src/train.py --split_ratio "80:20"
2. train.py calls data_pipeline.build_dataset(split_ratio="80:20")
3. data_pipeline loads CSV → splits data → returns train/val/test
4. train.py trains model on X_train/y_train
5. train.py validates on X_val/y_val (EVERY EPOCH)
6. Saves best model to checkpoints/best_model.pt
```

### Testing Workflow:
```
1. python src/evaluate.py
2. evaluate.py loads checkpoints/best_model.pt
3. For each split ratio:
   - evaluate.py calls data_pipeline.build_dataset(split_ratio)
   - data_pipeline returns X_test, y_test
   - evaluate.py makes predictions on X_test
   - Computes metrics on y_test
4. Saves all results to outputs/metrics.json
```

---

## 📍 EXACT FILE PATHS

**Full paths for easy copy-paste:**

```
Training:
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\src\train.py

Validation (integrated in training):
Line 110-115 in src\train.py

Testing:
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\src\evaluate.py

Data Pipeline (used by both):
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\src\data_pipeline.py

Raw Data:
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\data\rainfall_in_india_1901-2015.csv
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\data\climate_indices.csv

Model Checkpoint:
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\checkpoints\best_model.pt

Test Results:
c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system\outputs\metrics.json
```

---

## 🚀 RUN COMMANDS

### Train the model:
```bash
cd c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system
.\venv\Scripts\python.exe src/train.py --epochs 80 --split_ratio "80:20"
```

### Evaluate on all 4 ratios:
```bash
cd c:\Users\risha\OneDrive\Desktop\ML\cnn_lstm_rainfall_system
.\venv\Scripts\python.exe src/evaluate.py
```

### Check results:
```bash
cat outputs/metrics.json
```

---

This document maps **EXACTLY** where training, validation, and testing happen in your codebase! 🎯
