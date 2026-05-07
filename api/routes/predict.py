"""
api/routes/predict.py  — POST /api/predict
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.schemas import PredictRequest, PredictResponse, MonthlyPrediction

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_rainfall(req: PredictRequest):
    from predict import get_predictor

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. {str(e)}"
        )

    if req.subdivision not in predictor.subdivisions:
        raise HTTPException(
            status_code=404,
            detail=f"Subdivision '{req.subdivision}' not found. "
                   f"Available: {predictor.subdivisions[:5]}..."
        )

    try:
        raw = predictor.predict(
            subdivision = req.subdivision,
            start_year  = req.start_year,
            start_month = req.start_month,
            horizon     = req.horizon,
        )
        predictions = [MonthlyPrediction(**p) for p in raw]
        return PredictResponse(subdivision=req.subdivision, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
