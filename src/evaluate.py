"""Evaluate RainSight model across multiple train:test split ratios."""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import build_dataset, N_FEATURES, SEQ_LEN
from model import RainSightCNNLSTM, get_device
from train import BEST_MODEL_PATH

OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
METRICS_PATH= OUTPUTS_DIR / "metrics.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def load_trained_model(device: torch.device, seq_len: int = SEQ_LEN) -> RainSightCNNLSTM:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL_PATH}.\n"
            "Run: python src/train.py"
        )
    ckpt  = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model = RainSightCNNLSTM(
        n_features = ckpt.get("n_features", N_FEATURES),
        seq_len    = ckpt.get("seq_len", seq_len),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"[✓] Loaded checkpoint (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
    return model, ckpt


@torch.no_grad()
def predict_all(model: RainSightCNNLSTM, X: np.ndarray,
                device: torch.device, batch_size: int = 512) -> np.ndarray:
    ds  = TensorDataset(torch.from_numpy(X))
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    for (xb,) in ldr:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def _round_or_none(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def compute_split_metrics(preds_mm: np.ndarray, y_mm: np.ndarray, threshold_mm: float) -> dict:
    rmse_val = float(np.sqrt(np.mean((preds_mm - y_mm) ** 2)))
    mae_val = float(np.mean(np.abs(preds_mm - y_mm)))
    r2_val = r2_score(preds_mm, y_mm)

    y_true_bin = (y_mm >= threshold_mm).astype(int)
    y_pred_bin = (preds_mm >= threshold_mm).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    sensitivity = recall
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    error_rate = 1.0 - accuracy

    unique_classes = np.unique(y_true_bin)
    if len(unique_classes) == 2:
        auc_val = float(roc_auc_score(y_true_bin, preds_mm))
    else:
        auc_val = None

    return {
        "n": int(len(y_mm)),
        "rmse": _round_or_none(rmse_val),
        "mae": _round_or_none(mae_val),
        "r2": _round_or_none(r2_val),
        "precision": _round_or_none(precision),
        "recall": _round_or_none(recall),
        "sensitivity": _round_or_none(sensitivity),
        "f1_score": _round_or_none(f1),
        "accuracy": _round_or_none(accuracy),
        "auc": _round_or_none(auc_val),
        "error_rate": _round_or_none(error_rate),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_evaluation(
    seq_len: int = SEQ_LEN,
    batch_size: int = 512,
    split_ratios=None,
    event_percentile: float = 75.0,
):
    if split_ratios is None:
        split_ratios = ["60:40", "70:30", "80:20", "90:10"]

    device = get_device()
    model, ckpt = load_trained_model(device, seq_len)

    all_ratio_results = {}
    ratio_metadata = {}

    for ratio in split_ratios:
        print(f"\n[·] Evaluating split ratio {ratio}")
        splits, scaler, _ = build_dataset(
            seq_len=seq_len,
            fit_new_scaler=False,
            train_test_ratio=ratio,
            val_ratio_within_train=0.2,
        )

        rf_mean = scaler.mean_[0]
        rf_scale = scaler.scale_[0]

        train_y = splits["train"][1] * rf_scale + rf_mean
        threshold_mm = float(np.percentile(train_y, event_percentile))

        split_metrics = {}
        for split_name in ("train", "val", "test"):
            X, y, years, subdivs = splits[split_name]
            preds = predict_all(model, X, device, batch_size)
            preds_mm = preds * rf_scale + rf_mean
            y_mm = y * rf_scale + rf_mean

            metrics = compute_split_metrics(preds_mm, y_mm, threshold_mm)
            split_metrics[split_name] = metrics
            print(
                f"  [{split_name:5s}] RMSE={metrics['rmse']:.2f}  "
                f"F1={metrics['f1_score']:.3f}  "
                f"Precision={metrics['precision']:.3f}  "
                f"Recall={metrics['recall']:.3f}"
            )

            if ratio == "80:20" and split_name == "test":
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.scatter(y_mm[:2000], preds_mm[:2000],
                               alpha=0.3, s=8, color="#00D4FF", edgecolors="none")
                    ax.plot([y_mm.min(), y_mm.max()],
                            [y_mm.min(), y_mm.max()], "r--", linewidth=1.5, label="Perfect")
                    ax.set_xlabel("Actual Rainfall (mm)", fontsize=12)
                    ax.set_ylabel("Predicted Rainfall (mm)", fontsize=12)
                    ax.set_title(
                        f"80:20 Test — Predicted vs Actual\n"
                        f"RMSE={metrics['rmse']:.2f} mm  R²={metrics['r2']:.4f}"
                    )
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(PLOTS_DIR / "test_scatter_80_20.png", dpi=120)
                    plt.close(fig)
                except Exception as e:
                    print(f"[!] Plot skipped: {e}")

        all_ratio_results[ratio] = {
            "threshold_mm": _round_or_none(threshold_mm),
            "splits": split_metrics,
        }
        ratio_metadata[ratio] = {
            "event_percentile": event_percentile,
        }

    if "history" in ckpt:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            hist = ckpt["history"]
            epochs = [h["epoch"] for h in hist]
            tr_loss = [h["train_loss"] for h in hist]
            va_loss = [h["val_loss"] for h in hist]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epochs, tr_loss, label="Train Loss", color="#00D4FF")
            ax.plot(epochs, va_loss, label="Val Loss", color="#FF6B6B")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss (normalised)")
            ax.set_title("Training & Validation Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "training_curves.png", dpi=120)
            plt.close(fig)
            print("[✓] Training curves saved")
        except Exception as e:
            print(f"[!] Plot skipped: {e}")

    print(f"\n[·] Saving metrics → {METRICS_PATH}")
    default_ratio = "80:20" if "80:20" in all_ratio_results else split_ratios[0]
    metrics_out = {
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": ckpt["val_loss"],
        "default_ratio": default_ratio,
        "available_ratios": split_ratios,
        "event_definition": {
            "type": "heavy_rainfall_event",
            "threshold_basis": f"train_percentile_{event_percentile}",
        },
        "ratio_metadata": ratio_metadata,
        "ratios": all_ratio_results,
        # Backward-compatible convenience field for existing consumers.
        "splits": all_ratio_results[default_ratio]["splits"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[✓] Metrics saved.")
    return metrics_out


if __name__ == "__main__":
    run_evaluation()
