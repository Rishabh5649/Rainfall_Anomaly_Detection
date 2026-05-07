#!/usr/bin/env python
"""
Run training and evaluation for all 4 split ratios.
Results saved to outputs/metrics_all_ratios.json
"""

import sys
import json
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train import train
from evaluate import run_evaluation

print("=" * 70)
print("  RainSight — Testing All Split Ratios")
print("=" * 70)

RATIOS = ["60:40", "70:30", "80:20", "90:10"]
all_results = {}

for ratio in RATIOS:
    print(f"\n{'='*70}")
    print(f"  RATIO: {ratio}")
    print(f"{'='*70}\n")
    
    try:
        # Train with this ratio
        print(f"[1/2] Training with {ratio} split...")
        train(
            epochs=20,  # Shorter for testing
            batch_size=256,
            lr=3e-3,
            patience=5,
            seq_len=24,
            split_ratio=ratio,
        )
        print(f"[✓] Training complete for {ratio}\n")
    except Exception as e:
        print(f"[✗] Training failed for {ratio}: {e}\n")
        continue

# Evaluate all ratios at once
print(f"\n{'='*70}")
print(f"  EVALUATING ALL RATIOS")
print(f"{'='*70}\n")

try:
    results = run_evaluation(
        seq_len=24,
        batch_size=512,
        split_ratios=RATIOS,
        event_percentile=75.0,
    )
    print(f"\n[✓] Evaluation complete!\n")
    
    # Pretty print results
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    if "ratios" in results:
        for ratio, ratio_data in results["ratios"].items():
            print(f"\n{ratio}:")
            test_metrics = ratio_data["splits"]["test"]
            print(f"  Threshold: {ratio_data['threshold_mm']:.2f} mm")
            print(f"  Accuracy:  {test_metrics.get('accuracy', 'N/A')}")
            print(f"  F1-Score:  {test_metrics.get('f1_score', 'N/A')}")
            print(f"  Precision: {test_metrics.get('precision', 'N/A')}")
            print(f"  Recall:    {test_metrics.get('recall', 'N/A')}")
            print(f"  AUC:       {test_metrics.get('auc', 'N/A')}")
            print(f"  Error Rate:{test_metrics.get('error_rate', 'N/A')}")
            print(f"  RMSE:      {test_metrics.get('rmse', 'N/A')} mm")
            print(f"  CM:        TN={test_metrics.get('tn', '?')}, FP={test_metrics.get('fp', '?')}, FN={test_metrics.get('fn', '?')}, TP={test_metrics.get('tp', '?')}")
    
    print("\n" + "=" * 70)
    print("  [✓] All ratios tested successfully!")
    print("=" * 70)
    
except Exception as e:
    print(f"[✗] Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nResults saved to: outputs/metrics.json")
