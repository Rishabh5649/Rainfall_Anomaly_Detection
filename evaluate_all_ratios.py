#!/usr/bin/env python
"""
Evaluate the existing trained model for all 4 split ratios.
"""

import sys
import json
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

print("=" * 80)
print("  RainSight — Evaluating Existing Model on All Split Ratios")
print("=" * 80)

try:
    from evaluate import run_evaluation
    
    print("\nRunning evaluation for all 4 ratios...\n")
    
    results = run_evaluation(
        seq_len=24,
        batch_size=512,
        split_ratios=["60:40", "70:30", "80:20", "90:10"],
        event_percentile=75.0,
    )
    
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY - ALL METRICS FOR ALL RATIOS")
    print("=" * 80)
    
    if "ratios" in results:
        for ratio in ["60:40", "70:30", "80:20", "90:10"]:
            if ratio not in results["ratios"]:
                continue
                
            ratio_data = results["ratios"][ratio]
            threshold = ratio_data.get("threshold_mm", "N/A")
            
            print(f"\n{'─' * 80}")
            print(f"  RATIO: {ratio} | Event Threshold: {threshold} mm")
            print(f"{'─' * 80}")
            
            for split_name in ["train", "val", "test"]:
                split_data = ratio_data["splits"].get(split_name, {})
                if not split_data:
                    continue
                
                print(f"\n  {split_name.upper()}:")
                print(f"    Samples:      {split_data.get('n', 'N/A')}")
                print(f"    ✓ Accuracy:    {split_data.get('accuracy', 'N/A')}")
                print(f"    ✓ F1-Score:    {split_data.get('f1_score', 'N/A')}")
                print(f"    ✓ Precision:   {split_data.get('precision', 'N/A')}")
                print(f"    ✓ Recall:      {split_data.get('recall', 'N/A')}")
                print(f"    ✓ Sensitivity: {split_data.get('sensitivity', 'N/A')}")
                print(f"    ✓ AUC:         {split_data.get('auc', 'N/A')}")
                print(f"    ✓ Error Rate:  {split_data.get('error_rate', 'N/A')}")
                
                # Regression metrics
                print(f"\n    Regression Metrics:")
                print(f"      RMSE: {split_data.get('rmse', 'N/A')} mm")
                print(f"      MAE:  {split_data.get('mae', 'N/A')} mm")
                print(f"      R²:   {split_data.get('r2', 'N/A')}")
                
                # Confusion matrix
                cm = split_data.get('confusion_matrix', [[0,0],[0,0]])
                tn, fp = cm[0]
                fn, tp = cm[1]
                print(f"\n    Confusion Matrix:")
                print(f"      TN (True Neg):  {tn}")
                print(f"      FP (False Pos): {fp}")
                print(f"      FN (False Neg): {fn}")
                print(f"      TP (True Pos):  {tp}")
    
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE - TEST SPLIT ACROSS ALL RATIOS")
    print("=" * 80)
    
    print("\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Ratio", "Accuracy", "F1-Score", "Precision", "Recall", "Sensitivity", "AUC", "Error Rate", "RMSE (mm)"
    ))
    print("─" * 112)
    
    for ratio in ["60:40", "70:30", "80:20", "90:10"]:
        if ratio not in results["ratios"]:
            continue
        test = results["ratios"][ratio]["splits"]["test"]
        print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            ratio,
            f"{test.get('accuracy', 'N/A')}",
            f"{test.get('f1_score', 'N/A')}",
            f"{test.get('precision', 'N/A')}",
            f"{test.get('recall', 'N/A')}",
            f"{test.get('sensitivity', 'N/A')}",
            f"{test.get('auc', 'N/A')}",
            f"{test.get('error_rate', 'N/A')}",
            f"{test.get('rmse', 'N/A')}"
        ))
    
    print("\n" + "=" * 80)
    print("  ✓ ALL RATIOS EVALUATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFull results saved to: outputs/metrics.json")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
