"""
schemas.py  — Pydantic v2 request/response models for RainSight API
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    subdivision: str = Field(..., description="Name of the meteorological subdivision")
    start_year:  int = Field(2010, ge=1901, le=2030)
    start_month: int = Field(1,    ge=1,    le=12)
    horizon:     int = Field(12,   ge=1,    le=60, description="Months to forecast")


class MonthlyPrediction(BaseModel):
    year:         int
    month:        int
    predicted_mm: float
    lower_mm:     float
    upper_mm:     float


class PredictResponse(BaseModel):
    subdivision: str
    predictions: List[MonthlyPrediction]


class TrainRequest(BaseModel):
    epochs:     int   = Field(80,  ge=1,   le=500)
    batch_size: int   = Field(256, ge=16,  le=1024)
    lr:         float = Field(3e-3, ge=1e-5, le=1e-1)
    patience:   int   = Field(15,  ge=3,   le=100)
    split_ratio: str  = Field("80:20", pattern=r"^(60:40|70:30|80:20|90:10)$")


class TrainStatus(BaseModel):
    status:       str               # "idle" | "loading_data" | "training" | "complete" | "error"
    epoch:        Optional[int]     = None
    total_epochs: Optional[int]     = None
    loss:         Optional[float]   = None
    val_loss:     Optional[float]   = None
    val_rmse:     Optional[float]   = None
    message:      Optional[str]     = None


class HistoryPoint(BaseModel):
    year:        int
    month:       int
    rainfall_mm: float


class HistoryResponse(BaseModel):
    subdivision: str
    data:        List[HistoryPoint]


class MetricsResponse(BaseModel):
    checkpoint_epoch:    Optional[int]   = None
    checkpoint_val_loss: Optional[float] = None
    selected_ratio:      Optional[str]   = None
    available_ratios:    Optional[List[str]] = None
    threshold_mm:        Optional[float] = None

    train_rmse: Optional[float] = None
    train_mae:  Optional[float] = None
    train_r2:   Optional[float] = None
    val_rmse:   Optional[float] = None
    val_mae:    Optional[float] = None
    val_r2:     Optional[float] = None
    test_rmse:  Optional[float] = None
    test_mae:   Optional[float] = None
    test_r2:    Optional[float] = None
    test_precision:   Optional[float] = None
    test_recall:      Optional[float] = None
    test_sensitivity: Optional[float] = None
    test_f1_score:    Optional[float] = None
    test_accuracy:    Optional[float] = None
    test_auc:         Optional[float] = None
    test_error_rate:  Optional[float] = None
    test_confusion_matrix: Optional[List[List[int]]] = None

    ratios: Optional[Dict[str, Any]] = None
