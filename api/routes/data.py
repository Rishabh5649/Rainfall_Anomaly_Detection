"""
api/routes/data.py  — Data / metrics endpoints
"""

import sys
import json
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.schemas import HistoryResponse, HistoryPoint, MetricsResponse

router = APIRouter()

METRICS_PATH      = ROOT / "outputs" / "metrics.json"
SUBDIVISIONS_PATH = ROOT / "checkpoints" / "subdivisions.json"
TRAIN_LOG_PATH    = ROOT / "checkpoints" / "train_log.json"


@router.get("/subdivisions", response_model=List[str])
def list_subdivisions():
    if SUBDIVISIONS_PATH.exists():
        with open(SUBDIVISIONS_PATH) as f:
            return json.load(f)
    # fallback: read from data pipeline
    try:
        from data_pipeline import load_rainfall
        df = load_rainfall()
        return sorted(df["SUBDIVISION"].unique().tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{subdivision}", response_model=HistoryResponse)
def get_history(subdivision: str):
    try:
        from predict import get_predictor
        predictor = get_predictor()
    except FileNotFoundError:
        # Model not trained yet — still serve historical data
        sys.path.insert(0, str(ROOT / "src"))
        from data_pipeline import load_rainfall, wide_to_long
        df_wide = load_rainfall()
        df_long = wide_to_long(df_wide)

        sub_df = df_long[df_long["SUBDIVISION"] == subdivision]
        if sub_df.empty:
            raise HTTPException(status_code=404,
                detail=f"Subdivision '{subdivision}' not found.")
        data = [
            HistoryPoint(year=int(r.YEAR), month=int(r.MONTH),
                         rainfall_mm=round(float(r.RAINFALL), 2))
            for r in sub_df.itertuples()
        ]
        return HistoryResponse(subdivision=subdivision, data=data)

    if subdivision not in predictor.subdivisions:
        raise HTTPException(status_code=404,
            detail=f"Subdivision '{subdivision}' not found.")

    raw = predictor.get_history(subdivision)
    data = [HistoryPoint(**r) for r in raw]
    return HistoryResponse(subdivision=subdivision, data=data)


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    if not METRICS_PATH.exists():
        return MetricsResponse()
    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        ratios = m.get("ratios", {})
        available_ratios = m.get("available_ratios", list(ratios.keys()))
        selected_ratio = m.get("default_ratio", "80:20")
        selected = ratios.get(selected_ratio, {})
        splits = selected.get("splits", m.get("splits", {}))
        test_metrics = splits.get("test", {})

        return MetricsResponse(
            checkpoint_epoch    = m.get("checkpoint_epoch"),
            checkpoint_val_loss = m.get("checkpoint_val_loss"),
            selected_ratio      = selected_ratio,
            available_ratios    = available_ratios,
            threshold_mm        = selected.get("threshold_mm"),
            train_rmse = splits.get("train", {}).get("rmse"),
            train_mae  = splits.get("train", {}).get("mae"),
            train_r2   = splits.get("train", {}).get("r2"),
            val_rmse   = splits.get("val",   {}).get("rmse"),
            val_mae    = splits.get("val",   {}).get("mae"),
            val_r2     = splits.get("val",   {}).get("r2"),
            test_rmse  = splits.get("test",  {}).get("rmse"),
            test_mae   = splits.get("test",  {}).get("mae"),
            test_r2    = splits.get("test",  {}).get("r2"),
            test_precision = test_metrics.get("precision"),
            test_recall = test_metrics.get("recall"),
            test_sensitivity = test_metrics.get("sensitivity"),
            test_f1_score = test_metrics.get("f1_score"),
            test_accuracy = test_metrics.get("accuracy"),
            test_auc = test_metrics.get("auc"),
            test_error_rate = test_metrics.get("error_rate"),
            test_confusion_matrix = test_metrics.get("confusion_matrix"),
            ratios = ratios,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/history")
def get_train_history():
    """Return per-epoch loss history for charting."""
    if not TRAIN_LOG_PATH.exists():
        return []
    with open(TRAIN_LOG_PATH) as f:
        return json.load(f)
