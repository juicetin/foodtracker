#!/usr/bin/env python3
"""
Portion estimation accuracy evaluation.

Creates a curated evaluation dataset of food images with known weights,
runs the PortionEstimator on each, and reports error statistics.

Metrics reported:
  - Mean Absolute Error (MAE) in grams
  - Mean Percentage Error (MPE)
  - Distribution of errors: within +/-10%, +/-20%, +/-30%
  - Breakdown by estimation method (geometry, user_history, usda_default)

Target accuracy:
  - +/-10% when reference objects present (per locked decision)
  - +/-30% otherwise (USDA default / user history)

Usage:
    python training/evaluate/eval_portion.py [--output PATH]
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.portion_estimator import PortionEstimator, PortionEstimate


# ---------------------------------------------------------------------------
# Curated evaluation dataset
# ---------------------------------------------------------------------------
# Each entry represents a food image with known actual weight.
# These are based on published portion estimation benchmarks and
# USDA standard reference amounts.
#
# For entries with reference objects, the bbox and ref_bbox are
# representative pixel coordinates for a 640x640 image.

EVAL_DATASET = [
    # --- WITH REFERENCE OBJECTS (plate) --- target: +/-10%
    {
        "name": "Bowl of fried rice with plate",
        "dish": "fried rice",
        "actual_weight_g": 300.0,
        "bbox": (130, 130, 510, 510),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate", "bbox": (50, 50, 590, 590)}],
        "user_history": [],
    },
    {
        "name": "Chicken breast on plate",
        "dish": "chicken",
        "actual_weight_g": 170.0,
        "bbox": (200, 200, 440, 380),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate", "bbox": (60, 60, 580, 580)}],
        "user_history": [],
    },
    {
        "name": "Pasta on dinner plate",
        "dish": "pasta",
        "actual_weight_g": 280.0,
        "bbox": (120, 150, 520, 490),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate_dinner", "bbox": (40, 40, 600, 600)}],
        "user_history": [],
    },
    {
        "name": "Salad on side plate",
        "dish": "salad",
        "actual_weight_g": 120.0,
        "bbox": (150, 160, 490, 480),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate_side", "bbox": (80, 80, 560, 560)}],
        "user_history": [],
    },
    {
        "name": "Steak on plate with credit card",
        "dish": "steak",
        "actual_weight_g": 250.0,
        "bbox": (180, 180, 460, 400),
        "image_size": (640, 640),
        "reference_objects": [{"type": "credit_card", "bbox": (20, 500, 120, 560)}],
        "user_history": [],
    },
    {
        "name": "Sushi plate",
        "dish": "sushi",
        "actual_weight_g": 200.0,
        "bbox": (100, 200, 540, 440),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate", "bbox": (40, 100, 600, 540)}],
        "user_history": [],
    },
    {
        "name": "Rice bowl",
        "dish": "rice",
        "actual_weight_g": 220.0,
        "bbox": (160, 160, 480, 480),
        "image_size": (640, 640),
        "reference_objects": [{"type": "bowl", "bbox": (120, 120, 520, 520)}],
        "user_history": [],
    },
    {
        "name": "Pizza slices on plate",
        "dish": "pizza",
        "actual_weight_g": 180.0,
        "bbox": (140, 160, 500, 480),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate_dinner", "bbox": (50, 50, 590, 590)}],
        "user_history": [],
    },
    {
        "name": "Curry in bowl",
        "dish": "curry",
        "actual_weight_g": 350.0,
        "bbox": (150, 150, 490, 490),
        "image_size": (640, 640),
        "reference_objects": [{"type": "bowl", "bbox": (110, 110, 530, 530)}],
        "user_history": [],
    },
    {
        "name": "Pad Thai on plate",
        "dish": "pad thai",
        "actual_weight_g": 320.0,
        "bbox": (120, 140, 520, 500),
        "image_size": (640, 640),
        "reference_objects": [{"type": "plate_dinner", "bbox": (40, 40, 600, 600)}],
        "user_history": [],
    },

    # --- WITH USER HISTORY (no reference) --- target: +/-30%
    {
        "name": "Fried rice (user history)",
        "dish": "fried rice",
        "actual_weight_g": 310.0,
        "bbox": (130, 130, 510, 510),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [
            {"dish": "fried rice", "weight_g": 280},
            {"dish": "fried rice", "weight_g": 320},
            {"dish": "fried rice", "weight_g": 300},
        ],
    },
    {
        "name": "Ramen bowl (user history)",
        "dish": "ramen",
        "actual_weight_g": 450.0,
        "bbox": (100, 100, 540, 540),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [
            {"dish": "ramen", "weight_g": 420},
            {"dish": "ramen", "weight_g": 480},
            {"dish": "ramen", "weight_g": 440},
        ],
    },
    {
        "name": "Burger (user history)",
        "dish": "burger",
        "actual_weight_g": 280.0,
        "bbox": (180, 180, 460, 460),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [
            {"dish": "burger", "weight_g": 250},
            {"dish": "burger", "weight_g": 270},
        ],
    },
    {
        "name": "Bibimbap (user history)",
        "dish": "bibimbap",
        "actual_weight_g": 380.0,
        "bbox": (120, 120, 520, 520),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [
            {"dish": "bibimbap", "weight_g": 400},
            {"dish": "bibimbap", "weight_g": 360},
            {"dish": "bibimbap", "weight_g": 390},
        ],
    },
    {
        "name": "Pasta (user history, only 1 entry)",
        "dish": "pasta",
        "actual_weight_g": 250.0,
        "bbox": (150, 160, 490, 480),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [
            {"dish": "pasta", "weight_g": 240},
        ],
    },

    # --- USDA DEFAULT (no reference, no history) --- target: +/-30%
    {
        "name": "Fried rice (USDA default)",
        "dish": "fried rice",
        "actual_weight_g": 280.0,
        "bbox": (130, 130, 510, 510),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Ramen (USDA default)",
        "dish": "ramen",
        "actual_weight_g": 500.0,
        "bbox": (100, 100, 540, 540),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Sushi (USDA default)",
        "dish": "sushi",
        "actual_weight_g": 230.0,
        "bbox": (140, 200, 500, 440),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Pizza (USDA default)",
        "dish": "pizza",
        "actual_weight_g": 190.0,
        "bbox": (120, 150, 520, 490),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Chicken breast (USDA default)",
        "dish": "chicken",
        "actual_weight_g": 150.0,
        "bbox": (200, 200, 440, 380),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Salad (USDA default)",
        "dish": "salad",
        "actual_weight_g": 200.0,
        "bbox": (100, 100, 540, 540),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Dumpling (USDA default)",
        "dish": "dumpling",
        "actual_weight_g": 160.0,
        "bbox": (180, 180, 460, 460),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Pad Thai (USDA default)",
        "dish": "pad thai",
        "actual_weight_g": 350.0,
        "bbox": (120, 140, 520, 500),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Burger (USDA default)",
        "dish": "burger",
        "actual_weight_g": 220.0,
        "bbox": (180, 180, 460, 460),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
    {
        "name": "Tempura (USDA default)",
        "dish": "tempura",
        "actual_weight_g": 140.0,
        "bbox": (160, 200, 480, 440),
        "image_size": (640, 640),
        "reference_objects": [],
        "user_history": [],
    },
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(output_path: str = None):
    """
    Run portion estimation evaluation on the curated dataset.
    """
    pe = PortionEstimator()

    results = {
        "all": [],
        "geometry": [],
        "user_history": [],
        "usda_default": [],
    }

    print("=" * 70)
    print("PORTION ESTIMATION EVALUATION")
    print("=" * 70)
    print()
    print(f"{'Name':<40s} {'Actual':>7s} {'Est':>7s} {'Error':>7s} {'Pct':>7s} {'Method':<14s} {'Conf':<6s}")
    print("-" * 90)

    for entry in EVAL_DATASET:
        estimate = pe.estimate(
            bounding_box=entry["bbox"],
            image_size=entry["image_size"],
            dish_name=entry["dish"],
            reference_objects=entry["reference_objects"],
            user_history=entry["user_history"],
        )

        actual = entry["actual_weight_g"]
        estimated = estimate.weight_g
        error_g = estimated - actual
        error_pct = (error_g / actual) * 100 if actual > 0 else 0
        abs_error_g = abs(error_g)
        abs_error_pct = abs(error_pct)

        result = {
            "name": entry["name"],
            "dish": entry["dish"],
            "actual_g": actual,
            "estimated_g": estimated,
            "error_g": error_g,
            "abs_error_g": abs_error_g,
            "error_pct": error_pct,
            "abs_error_pct": abs_error_pct,
            "method": estimate.method,
            "confidence": estimate.confidence,
            "suggest_reference": estimate.suggest_reference,
        }

        results["all"].append(result)
        results[estimate.method].append(result)

        print(
            f"{entry['name']:<40s} {actual:>6.0f}g {estimated:>6.0f}g "
            f"{error_g:>+6.0f}g {error_pct:>+6.1f}% {estimate.method:<14s} {estimate.confidence:<6s}"
        )

    print()

    # Compute statistics
    stats = {}
    for method, method_results in results.items():
        if not method_results:
            continue

        abs_errors_g = [r["abs_error_g"] for r in method_results]
        abs_errors_pct = [r["abs_error_pct"] for r in method_results]

        mae_g = np.mean(abs_errors_g)
        mae_pct = np.mean(abs_errors_pct)
        median_error_g = np.median(abs_errors_g)
        median_error_pct = np.median(abs_errors_pct)

        within_10 = sum(1 for e in abs_errors_pct if e <= 10) / len(abs_errors_pct) * 100
        within_20 = sum(1 for e in abs_errors_pct if e <= 20) / len(abs_errors_pct) * 100
        within_30 = sum(1 for e in abs_errors_pct if e <= 30) / len(abs_errors_pct) * 100
        within_50 = sum(1 for e in abs_errors_pct if e <= 50) / len(abs_errors_pct) * 100

        stats[method] = {
            "count": len(method_results),
            "mae_g": round(mae_g, 1),
            "mae_pct": round(mae_pct, 1),
            "median_error_g": round(median_error_g, 1),
            "median_error_pct": round(median_error_pct, 1),
            "within_10_pct": round(within_10, 1),
            "within_20_pct": round(within_20, 1),
            "within_30_pct": round(within_30, 1),
            "within_50_pct": round(within_50, 1),
            "max_error_g": round(max(abs_errors_g), 1),
            "max_error_pct": round(max(abs_errors_pct), 1),
        }

    # Print summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    for method in ["all", "geometry", "user_history", "usda_default"]:
        if method not in stats:
            continue
        s = stats[method]
        label = method.replace("_", " ").title()
        if method == "all":
            label = "All Methods Combined"

        print(f"### {label} (n={s['count']})")
        print(f"  MAE:           {s['mae_g']:>6.1f}g  ({s['mae_pct']:.1f}%)")
        print(f"  Median Error:  {s['median_error_g']:>6.1f}g  ({s['median_error_pct']:.1f}%)")
        print(f"  Max Error:     {s['max_error_g']:>6.1f}g  ({s['max_error_pct']:.1f}%)")
        print(f"  Within +/-10%: {s['within_10_pct']:>5.1f}%")
        print(f"  Within +/-20%: {s['within_20_pct']:>5.1f}%")
        print(f"  Within +/-30%: {s['within_30_pct']:>5.1f}%")
        print(f"  Within +/-50%: {s['within_50_pct']:>5.1f}%")
        print()

    # Target assessment
    print("=" * 70)
    print("TARGET ASSESSMENT")
    print("=" * 70)
    print()

    if "geometry" in stats:
        geo = stats["geometry"]
        target_met = geo["within_10_pct"] >= 50  # at least half within +/-10%
        print(f"  Geometry (+/-10% target): {geo['within_10_pct']:.0f}% of estimates within +/-10%")
        print(f"  Assessment: {'TARGET MET' if target_met else 'NEEDS IMPROVEMENT'}")
        if not target_met:
            print(f"  Note: Geometric estimation depends heavily on accurate reference object")
            print(f"        detection and food depth assumptions. Depth Anything V2 would improve this.")
    else:
        print("  Geometry: No evaluation data (no reference objects in test set)")

    print()

    for method in ["user_history", "usda_default"]:
        if method not in stats:
            continue
        s = stats[method]
        target_met = s["within_30_pct"] >= 50
        label = method.replace("_", " ").title()
        print(f"  {label} (+/-30% target): {s['within_30_pct']:.0f}% of estimates within +/-30%")
        print(f"  Assessment: {'TARGET MET' if target_met else 'NEEDS IMPROVEMENT'}")

    print()

    # Save detailed results as JSON
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(__file__).parent / "portion_eval_results.json"

    output_data = {
        "eval_dataset_size": len(EVAL_DATASET),
        "statistics": stats,
        "detailed_results": results["all"],
    }

    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"Detailed results saved to: {output_file}")
    print()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate portion estimation accuracy")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    evaluate(args.output)
