#!/usr/bin/env python3
"""
DSPy prompt optimization for Gemini Nano food identification.

Runs BootstrapFewShot or COPRO against the actual on-device Gemini Nano
using labeled food images as ground truth.

Usage:
    # First: build & install the app with VlmEvalReceiver registered
    # Then:
    python optimize.py                          # Run optimization
    python optimize.py --eval-only              # Just evaluate current prompt
    python optimize.py --optimizer copro        # Use COPRO instead of bootstrap
    python optimize.py --serial 2A281FDH3002TN  # Specify device serial
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add dspy to path
sys.path.insert(0, str(Path.home() / "repos" / "dspy"))

import dspy
from dspy import Example, Image

from gemini_nano_lm import GeminiNanoLM
from food_metric import food_identification_metric, score_detailed


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).parent / "dataset"


def load_dataset() -> list[Example]:
    """
    Load labeled food images from dataset/ directory.

    Each example is a JSON file:
    {
        "image": "path/to/image.jpg",  (relative to dataset/)
        "dishes": [
            {
                "name": "pad thai",
                "cuisine": "Thai",
                "ingredients": [
                    {"name": "rice noodles", "amount_g": 200},
                    {"name": "shrimp", "amount_g": 100}
                ]
            }
        ]
    }
    """
    examples = []
    label_files = sorted(DATASET_DIR.glob("*.json"))

    if not label_files:
        print(f"No label files found in {DATASET_DIR}/")
        print("Create .json files with 'image' and 'dishes' fields.")
        print("See dataset/EXAMPLE.json for the format.")
        sys.exit(1)

    for lf in label_files:
        data = json.loads(lf.read_text())
        image_path = DATASET_DIR / data["image"]
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}, skipping")
            continue

        ex = Example(
            image=Image(str(image_path)),
            dishes=data["dishes"],
        ).with_inputs("image")
        examples.append(ex)

    print(f"Loaded {len(examples)} labeled examples")
    return examples


# ---------------------------------------------------------------------------
# DSPy Signature & Module
# ---------------------------------------------------------------------------

class FoodIdentification(dspy.Signature):
    """Identify all food dishes in the image with their ingredients and estimated gram weights.
Return only valid JSON with this schema:
{"dishes":[{"name":string,"cuisine":string,"recipe_name":string,"ingredients":[{"name":string,"amount_g":number}]}]}
Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) as size references;
fall back to a typical restaurant serving size if no reference objects are visible.
Be specific with ingredient names (e.g. "basmati rice" not "rice")."""

    image: dspy.Image = dspy.InputField(desc="Photo of a meal")
    output: str = dspy.OutputField(
        desc="JSON object with dishes array containing name, cuisine, recipe_name, and ingredients with amount_g"
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def run_bootstrap(trainset: list[Example], valset: list[Example], lm) -> dspy.Module:
    """Run BootstrapFewShot optimization."""
    program = dspy.Predict(FoodIdentification)

    optimizer = dspy.BootstrapFewShot(
        metric=food_identification_metric,
        max_bootstrapped_demos=0,  # No few-shot demos (256 token output limit)
        max_labeled_demos=0,       # Keep prompt compact for Nano
        max_rounds=1,
    )

    print("\n=== Running BootstrapFewShot ===")
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val:   {len(valset)} examples")

    optimized = optimizer.compile(program, trainset=trainset)
    return optimized


def run_copro(trainset: list[Example], valset: list[Example], lm) -> dspy.Module:
    """
    Run COPRO optimization — iterates on instruction text.
    Requires a separate "prompt model" (cloud LM) to generate instruction candidates.
    """
    program = dspy.Predict(FoodIdentification)

    # COPRO needs a prompt_model to generate instruction candidates.
    # We use a cheap cloud model for this — Gemini Nano can't self-reflect.
    try:
        prompt_model = dspy.LM("openai/gpt-4o-mini", max_tokens=1000)
    except Exception:
        print("COPRO requires an OpenAI API key for instruction generation.")
        print("Set OPENAI_API_KEY env var, or use --optimizer bootstrap")
        sys.exit(1)

    optimizer = dspy.COPRO(
        metric=food_identification_metric,
        breadth=5,    # 5 instruction candidates per round
        depth=2,      # 2 rounds of refinement
        init_temperature=1.0,
    )

    print("\n=== Running COPRO ===")
    print(f"  Train: {len(trainset)} examples, Val: {len(valset)} examples")
    print(f"  Breadth: 5, Depth: 2 (~10 instruction candidates)")
    print(f"  Prompt model: gpt-4o-mini (for candidate generation)")
    print(f"  Task model: Gemini Nano on-device (for evaluation)")

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        eval_kwargs={"num_threads": 1},  # Serial — one device
    )
    return optimized


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(program: dspy.Module, dataset: list[Example], label: str = "Eval"):
    """Run program on dataset and print detailed scores."""
    print(f"\n=== {label} ({len(dataset)} examples) ===")

    scores = []
    for i, ex in enumerate(dataset):
        try:
            pred = program(image=ex.image)
            raw = pred.output if hasattr(pred, "output") else str(pred)
        except Exception as e:
            raw = f"ERROR:{e}"

        detail = score_detailed(raw, ex.dishes)
        scores.append(detail)

        gt_names = [d["name"] for d in ex.dishes]
        print(
            f"  [{i+1}/{len(dataset)}] "
            f"GT: {gt_names} | "
            f"Score: {detail['composite']:.2f} | "
            f"Names: {detail['dish_name_f1']:.2f} | "
            f"Recall: {detail['ingredient_recall']:.2f} | "
            f"Precision: {detail['ingredient_precision']:.2f} | "
            f"Weights: {detail['weight_mae_score']:.2f} | "
            f"Parsed: {detail['json_parsed']}"
        )

    # Aggregate
    avg = lambda key: sum(s[key] for s in scores) / len(scores) if scores else 0
    print(f"\n  --- Aggregated ---")
    print(f"  Composite:   {avg('composite'):.3f}")
    print(f"  Dish name:   {avg('dish_name_f1'):.3f}")
    print(f"  Ing recall:  {avg('ingredient_recall'):.3f}")
    print(f"  Ing prec:    {avg('ingredient_precision'):.3f}")
    print(f"  Weight MAE:  {avg('weight_mae_score'):.3f}")
    print(f"  JSON parse:  {sum(1 for s in scores if s['json_parsed'])}/{len(scores)}")

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSPy prompt optimization for Gemini Nano")
    parser.add_argument("--serial", default="48181FDAP00A1U", help="ADB device serial")
    parser.add_argument("--optimizer", choices=["bootstrap", "copro"], default="bootstrap")
    parser.add_argument("--eval-only", action="store_true", help="Just evaluate, don't optimize")
    parser.add_argument("--val-split", type=float, default=0.3, help="Validation split ratio")
    args = parser.parse_args()

    # Initialize on-device LM
    print(f"Connecting to device {args.serial}...")
    lm = GeminiNanoLM(serial=args.serial)
    dspy.configure(lm=lm)
    print(f"Device ready. Stats: {lm.stats()}")

    # Load dataset
    dataset = load_dataset()

    if args.eval_only:
        program = dspy.Predict(FoodIdentification)
        evaluate(program, dataset, label="Baseline")
        print(f"\nDevice stats: {lm.stats()}")
        return

    # Split train/val
    split_idx = max(1, int(len(dataset) * (1 - args.val_split)))
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:] if split_idx < len(dataset) else dataset

    # Run optimization
    start = time.time()
    if args.optimizer == "copro":
        optimized = run_copro(trainset, valset, lm)
    else:
        optimized = run_bootstrap(trainset, valset, lm)

    elapsed = time.time() - start
    print(f"\nOptimization took {elapsed:.1f}s")

    # Evaluate baseline vs optimized
    baseline = dspy.Predict(FoodIdentification)
    evaluate(baseline, valset, label="Baseline")
    evaluate(optimized, valset, label="Optimized")

    # Save optimized program
    output_path = Path(__file__).parent / "optimized_program.json"
    optimized.save(str(output_path))
    print(f"\nOptimized program saved to {output_path}")

    # Extract the optimized instruction
    if hasattr(optimized, "signature"):
        print(f"\n=== Optimized Instruction ===")
        print(optimized.signature.instructions)

    print(f"\nDevice stats: {lm.stats()}")


if __name__ == "__main__":
    main()
