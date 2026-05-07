"""
train.py
Training loop for RainSight India CNN+LSTM model.

Features:
  - AdamW optimizer with OneCycleLR scheduler
  - Early stopping (patience=15)
  - Best model checkpointing
  - Per-epoch metrics logged to JSON
  - GPU auto-detection
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

# ── Project imports ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import build_dataset, N_FEATURES, SEQ_LEN
from model import build_model, get_device

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH   = CHECKPOINT_DIR / "best_model.pt"
TRAIN_LOG_PATH    = CHECKPOINT_DIR / "train_log.json"
TRAIN_STATUS_PATH = CHECKPOINT_DIR / "train_status.json"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def write_status(status: dict):
    with open(TRAIN_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ─────────────────────────────────────────────────────────────────
# Evaluation pass
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        out = model(X_batch)
        loss = mse_loss(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds.extend(out.cpu().numpy())
        targets.extend(y_batch.cpu().numpy())
    preds   = np.array(preds)
    targets = np.array(targets)
    return {
        "loss": total_loss / len(targets),
        "rmse": rmse(preds, targets),
        "mae":  mae(preds, targets),
    }


# ─────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────
def train(
    epochs:      int   = 80,
    batch_size:  int   = 256,
    lr:          float = 3e-3,
    weight_decay: float = 1e-4,
    patience:    int   = 15,
    seq_len:     int   = SEQ_LEN,
    split_ratio: str   = "80:20",
):
    write_status({"status": "loading_data", "epoch": 0, "total_epochs": epochs,
                  "loss": None, "val_loss": None, "val_rmse": None})

    # ── Data ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  RainSight India — Training")
    print("=" * 60)
    splits, scaler, _ = build_dataset(
        seq_len=seq_len,
        train_test_ratio=split_ratio,
        val_ratio_within_train=0.2,
    )

    X_tr, y_tr = splits["train"][0], splits["train"][1]
    X_va, y_va = splits["val"][0],   splits["val"][1]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    model  = build_model(n_features=N_FEATURES, seq_len=seq_len, device=device)

    # ── Optimiser & scheduler ─────────────────────────────────────
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimiser,
        max_lr        = lr,
        epochs        = epochs,
        steps_per_epoch= len(train_loader),
        pct_start     = 0.3,
        anneal_strategy= "cos",
    )

    # ── Early stopping state ──────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    write_status({"status": "training", "epoch": 0, "total_epochs": epochs,
                  "loss": None, "val_loss": None, "val_rmse": None})

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            out  = model(X_batch)
            loss = mse_loss(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # ── Validate ──────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        val_loss    = val_metrics["loss"]
        val_rmse    = val_metrics["rmse"]
        val_mae     = val_metrics["mae"]

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  train_loss={train_loss:.5f}"
            f"  val_loss={val_loss:.5f}"
            f"  val_rmse={val_rmse:.4f}"
            f"  val_mae={val_mae:.4f}"
            f"  [{elapsed:.0f}s]"
        )

        rec = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_rmse":   round(val_rmse, 6),
            "val_mae":    round(val_mae, 6),
        }
        history.append(rec)
        write_status({"status": "training", "epoch": epoch, "total_epochs": epochs,
                      "loss": round(train_loss, 6),
                      "val_loss": round(val_loss, 6),
                      "val_rmse": round(val_rmse, 6)})

        # ── Checkpoint best ───────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_rmse":    val_rmse,
                "n_features":  N_FEATURES,
                "seq_len":     seq_len,
                "history":     history,
            }, BEST_MODEL_PATH)
            print(f"    [✓] Checkpoint saved (val_loss={val_loss:.5f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Save full training log ─────────────────────────────────────
    with open(TRAIN_LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[✓] Training log saved → {TRAIN_LOG_PATH}")

    write_status({"status": "complete", "epoch": epoch, "total_epochs": epochs,
                  "loss": history[-1]["train_loss"],
                  "val_loss": history[-1]["val_loss"],
                  "val_rmse": history[-1]["val_rmse"]})
    print(f"[✓] Training complete. Best val_loss = {best_val_loss:.5f}")
    return history


# ─────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RainSight CNN+LSTM")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=3e-3)
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--seq_len",     type=int,   default=SEQ_LEN)
    parser.add_argument("--split_ratio", type=str,   default="80:20")
    args = parser.parse_args()

    train(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience,
        seq_len    = args.seq_len,
        split_ratio= args.split_ratio,
    )
