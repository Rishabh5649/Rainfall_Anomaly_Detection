# Training, Testing & Validation Files — Complete Reference

## 📍 Where Train/Test/Validation Happens

### **1. DATA SPLITTING** 
**File:** [src/data_pipeline.py](src/data_pipeline.py)

#### Functions:
- `split_by_year()` (lines ~378-391) — Original year-based split (1901-2015)
  - Train: years ≤ 1995
  - Val: 1995 < years ≤ 2005
  - Test: years > 2005

- `split_by_ratio_years()` (lines ~393-442) — **NEW** ratio-based split
  - Train: First N% of unique years (minus validation portion)
  - Val: 20% of training years
  - Test: Remaining years
  - Supports: 60:40, 70:30, 80:20, 90:10

#### Key Parameters:
```python
build_dataset(
    seq_len=24,
    train_until=1995,
    val_until=2005,
    fit_new_scaler=True,
    train_test_ratio="",        # "60:40", "70:30", "80:20", "90:10"
    val_ratio_within_train=0.2  # Fixed at 20%
)
```

---

### **2. TRAINING LOOP**
**File:** [src/train.py](src/train.py)

#### Main Entry Point:
- `train()` function (lines 92-253)
  - Loads splits via `build_dataset()`
  - Creates TensorDataset for train & val splits
  - DataLoader batch feeding
  - Training loop: epochs 1-N
  - Validation after each epoch

#### Key Parameters:
```python
train(
    epochs=80,
    batch_size=256,
    lr=3e-3,
    weight_decay=1e-4,
    patience=15,
    seq_len=24,
    split_ratio="80:20"  # NEW PARAMETER
)
```

#### Training Loop Pseudocode:
```
for epoch in 1..N:
    train_loss = 0
    for X_batch, y_batch in train_loader:
        forward_pass()
        loss.backward()
        optimiser.step()
    
    val_metrics = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        save_checkpoint()
    else:
        patience_count++
        
    if patience_count >= 15:
        break
```

#### Outputs:
- Best model checkpoint → `checkpoints/best_model.pt`
- Training log → `checkpoints/train_log.json` (per-epoch metrics)
- Training status → `checkpoints/train_status.json` (current progress)

---

### **3. VALIDATION DURING TRAINING**
**File:** [src/train.py](src/train.py#L65-L82)

#### `evaluate()` Function:
```python
@torch.no_grad()
def evaluate(model, loader, device):
    # Called after each training epoch
    # Computes:
    # - MSE Loss
    # - RMSE (root mean squared error)
    # - MAE (mean absolute error)
    # Returns dict with metrics
```

#### Validation Loop:
- Runs on full validation set
- No gradient computation
- Early stopping: patience=15 epochs with no improvement
- Saves best model when val_loss decreases

---

### **4. TEST SET EVALUATION**
**File:** [src/evaluate.py](src/evaluate.py)

#### Main Entry Point:
- `run_evaluation()` function (lines 72-189)
  - Loads trained model from checkpoint
  - Evaluates on train/val/test splits
  - Computes all 8 classification metrics
  - Generates plots

#### Metric Computation:
- `compute_split_metrics()` (lines 53-95)
  - Input: predictions, targets, threshold
  - Binarizes: rainfall ≥ threshold = "event" (1), else "non-event" (0)
  - Computes:
    - RMSE, MAE, R² (regression metrics)
    - Confusion Matrix: TN, FP, FN, TP
    - Precision, Recall, Sensitivity, F1, Accuracy, AUC, Error Rate

#### Processing Per Ratio:
```python
for ratio in ["60:40", "70:30", "80:20", "90:10"]:
    # Load dataset with ratio-based split
    splits = build_dataset(train_test_ratio=ratio)
    
    # Compute threshold (75th percentile of train set)
    threshold_mm = np.percentile(train_y, 75)
    
    # Evaluate on train/val/test
    for split_name in ("train", "val", "test"):
        metrics = compute_split_metrics(...)
    
    # Save results
```

#### Outputs:
- `outputs/metrics.json` — All ratios with all metrics
- `outputs/plots/` — Scatter plots, time-series, training curves

---

### **5. API ENDPOINTS**
**File:** [api/routes/train.py](api/routes/train.py) & [api/routes/data.py](api/routes/data.py)

#### Training Endpoint:
```
POST /api/train
Content-Type: application/json

{
  "epochs": 80,
  "batch_size": 256,
  "lr": 0.003,
  "patience": 15,
  "split_ratio": "70:30"
}

Response:
{
  "status": "training",
  "epoch": 0,
  "total_epochs": 80,
  "message": "Training started."
}
```

#### Metrics Endpoint:
```
GET /api/metrics

Response:
{
  "checkpoint_epoch": 45,
  "checkpoint_val_loss": 0.0234,
  "selected_ratio": "80:20",
  "available_ratios": ["60:40", "70:30", "80:20", "90:10"],
  "threshold_mm": 157.5,
  
  "test_accuracy": 0.7981,
  "test_f1_score": 0.78,
  "test_precision": 0.7891,
  "test_recall": 0.7712,
  "test_sensitivity": 0.7712,
  "test_auc": 0.8634,
  "test_error_rate": 0.2019,
  "test_confusion_matrix": [[447, 42], [152, 260]],
  
  "ratios": {
    "80:20": {
      "threshold_mm": 157.5,
      "splits": {
        "train": {...},
        "val": {...},
        "test": {...}
      }
    },
    ...
  }
}
```

---

## 🎯 KEY METRICS COMPUTED

### **During Training (validation)**
- MSE Loss
- RMSE
- MAE

### **During Evaluation (test)**
1. **Confusion Matrix** — TN, FP, FN, TP
2. **F1-Score** — Harmonic mean (Precision × Recall)
3. **Precision** — TP / (TP + FP)
4. **Recall** — TP / (TP + FN)
5. **Sensitivity** — Same as Recall in binary classification
6. **Accuracy** — (TP + TN) / Total
7. **AUC (Area Under Curve)** — ROC-AUC for event detection
8. **Error Rate** — 1 - Accuracy

### **Regression Metrics** (all splits)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

---

## 🔄 DATA FLOW DIAGRAM

```
DATA LOADING
    ↓
build_dataset()
    ├─ load_rainfall()
    ├─ load_climate_indices()
    ├─ merge_climate()
    ├─ add_features()
    ├─ fit_scalers() or load_scaler()
    ├─ build_sequences()
    └─ split_by_ratio_years() ← NEW FUNCTION
        ├─ Train (60-90%)
        ├─ Val (20% of train)
        └─ Test (10-40%)

TRAINING
    train() function
    ├─ Forward pass on train batches
    ├─ Backward pass + optimizer step
    └─ After each epoch:
        ├─ evaluate() on val set
        ├─ Early stopping check
        └─ Save checkpoint if improved

TEST EVALUATION
    run_evaluation() function
    ├─ Load checkpoint
    ├─ For each ratio (60:40, 70:30, 80:20, 90:10):
    │   ├─ Reload dataset with ratio split
    │   ├─ Compute threshold (75th percentile)
    │   └─ For train/val/test splits:
    │       └─ compute_split_metrics()
    │           ├─ Predictions (regression)
    │           ├─ Binarization (threshold)
    │           └─ Compute 8 metrics
    └─ Save to outputs/metrics.json

API SERVING
    /api/train → triggers training()
    /api/metrics → reads outputs/metrics.json
```

---

## 🚀 HOW TO RUN EACH STAGE

### **Stage 1: Train with Specific Ratio**
```powershell
python src/train.py --split_ratio 70:30 --epochs 80 --batch_size 256
```
- Loads data with 70:30 train:test split
- Trains model
- Saves best checkpoint

### **Stage 2: Evaluate All Ratios**
```powershell
python src/evaluate.py
```
- Loads the best trained model
- Evaluates on all 4 ratios
- Computes all 8 metrics
- Saves to outputs/metrics.json

### **Stage 3: View Results**
```powershell
# Start API
python -m uvicorn api.main:app --port 8000

# In browser: http://localhost:5173/analytics
```
- Fetches /api/metrics
- Displays all ratios in Analytics page
- User can switch between ratios with dropdown

---

## 📊 SAMPLE WORKFLOW

```
1. User goes to Dashboard
   └─ Clicks "Train Model" button

2. TrainModal appears
   ├─ Sets epochs: 80
   ├─ Sets split_ratio: "70:30" ← USER CHOICE
   └─ Clicks "Start Training"

3. Backend
   ├─ POST /api/train with split_ratio: "70:30"
   ├─ train() loads data with 70:30 split
   ├─ Training loop (30 mins)
   ├─ Saves best checkpoint
   └─ Returns status

4. User navigates to Analytics

5. Frontend
   ├─ Fetches GET /api/metrics
   ├─ Receives all 4 ratios' results
   ├─ Shows "80:20" by default
   ├─ User can switch ratio
   ├─ Displays:
   │   ├─ 8/8 checklist ✅
   │   ├─ Confusion matrix
   │   ├─ Classification metrics
   │   ├─ Comparison table
   │   └─ RMSE/MAE bar chart
   └─ All populated with real data
```

---

## 🎓 SUMMARY TABLE

| Stage | File | Function | Input | Output |
|-------|------|----------|-------|--------|
| **Load** | `data_pipeline.py` | `build_dataset()` | CSV files, ratio string | Splits dict with X,y arrays |
| **Train** | `train.py` | `train()` | Splits, hyperparams, ratio | Checkpoint, log, status |
| **Validate** | `train.py` | `evaluate()` | Model, val_loader | RMSE, MAE, R² |
| **Test** | `evaluate.py` | `run_evaluation()` | Model, all ratios | JSON with all 8 metrics |
| **Serve** | `api/routes/data.py` | `get_metrics()` | HTTP GET | JSON response |
| **Display** | `Analytics.jsx` | Component render | JSON data | HTML table/charts |

---

**Everything is integrated and ready to use!** You can train with any ratio, evaluate all ratios, and display all 8 metrics on the website. 🎉
