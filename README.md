# RainSight India 🌧️

A production-grade **CNN+LSTM hybrid rainfall forecasting platform** for 36 Indian meteorological subdivisions, built on real IMD data (1901–2015).

## Architecture

```
Input sequences (24 months × 6 features)
  → Conv1D Residual Blocks (64 → 128 → 256 channels)
  → Bidirectional LSTM (256 hidden × 2 layers)
  → Attention Pooling
  → FC Head → Predicted rainfall (mm)
```

**Features:** Monthly rainfall (IMD), ENSO index, IOD index, cyclic month encoding, normalised year

## Quick Start

### 1. Setup
```powershell
python -m venv venv
.\venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu -q
.\venv\Scripts\pip install -r requirements.txt -q
```

### 2. Generate Climate Indices
```powershell
$env:PYTHONUTF8=1; .\venv\Scripts\python src\generate_climate_indices.py
```

### 3. Train Model
```powershell
$env:PYTHONUTF8=1; .\venv\Scripts\python src\train.py --epochs 80 --batch_size 256
```

### 4. Evaluate
```powershell
$env:PYTHONUTF8=1; .\venv\Scripts\python src\evaluate.py
```

### 5. Start API
```powershell
$env:PYTHONUTF8=1; .\venv\Scripts\python -m uvicorn api.main:app --port 8000
```

### 6. Start Frontend
```powershell
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/subdivisions` | List all 36 subdivisions |
| `GET`  | `/api/history/{sub}` | Historical rainfall data |
| `GET`  | `/api/metrics` | Model RMSE / MAE / R² |
| `GET`  | `/api/train/status` | Training progress |
| `POST` | `/api/train` | Start background training |
| `POST` | `/api/predict` | Generate forecast |

### Predict example
```json
POST /api/predict
{
  "subdivision": "KERALA",
  "start_year": 2010,
  "start_month": 6,
  "horizon": 12
}
```

## Dataset

- **Source:** India Meteorological Department (IMD) via OGD Platform
- **File:** `data/rainfall_in_india_1901-2015.csv`
- **Coverage:** 36 meteorological subdivisions × ~115 years
- **Columns:** SUBDIVISION, YEAR, JAN–DEC monthly rainfall (mm)

## Project Structure

```
cnn_lstm_rainfall_system/
├── data/           ← IMD CSV + generated climate indices
├── src/            ← ML pipeline (data, model, train, evaluate, predict)
├── api/            ← FastAPI backend
├── frontend/       ← Vite + React dashboard
├── checkpoints/    ← Saved model weights
└── outputs/        ← Evaluation metrics + plots
```
