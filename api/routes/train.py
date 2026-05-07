"""
api/routes/train.py  — POST /api/train  |  GET /api/train/status
"""

import sys
import json
import threading
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.schemas import TrainRequest, TrainStatus

router = APIRouter()

TRAIN_STATUS_PATH = ROOT / "checkpoints" / "train_status.json"
_training_lock = threading.Lock()


def _run_training(req: TrainRequest):
    """Run in background thread."""
    from train import train as run_train
    try:
        run_train(
            epochs     = req.epochs,
            batch_size = req.batch_size,
            lr         = req.lr,
            patience   = req.patience,
            split_ratio= req.split_ratio,
        )
        # Reload predictor singleton so next /predict uses new weights
        import predict as predict_module
        predict_module._predictor = None
    except Exception as e:
        status = {
            "status":  "error",
            "message": str(e),
        }
        with open(TRAIN_STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2)


@router.post("/train", response_model=TrainStatus)
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress.")

    def run_and_release(req):
        try:
            _run_training(req)
        finally:
            _training_lock.release()

    background_tasks.add_task(run_and_release, req)
    return TrainStatus(status="training", epoch=0,
                       total_epochs=req.epochs, message="Training started.")


@router.get("/train/status", response_model=TrainStatus)
def get_train_status():
    if not TRAIN_STATUS_PATH.exists():
        return TrainStatus(status="idle", message="No training has been run yet.")
    try:
        with open(TRAIN_STATUS_PATH) as f:
            data = json.load(f)
        return TrainStatus(**data)
    except Exception as e:
        return TrainStatus(status="error", message=str(e))
