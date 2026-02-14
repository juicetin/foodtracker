#!/usr/bin/env python3
"""
Per-cuisine classification evaluation for trained YOLO dish classifier.

Loads a trained classification model and evaluates predictions on the test set,
grouping results by cuisine using the cuisine mapping. Reports per-cuisine
Top-1 and Top-5 accuracy, generates a confusion matrix for the top-20 most
confused class pairs, and flags cuisines with Top-1 accuracy below 50%.

Usage:
    python training/evaluate/eval_classification.py
    python training/evaluate/eval_classification.py --model-path path/to/best.pt
    python training/evaluate/eval_classification.py --test-dir path/to/test/

Prerequisites:
    - Trained classification model (run train_classify.py first)
    - Test images organized by class subdirectories (ImageFolder format)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = TRAINING_DIR / "runs"
EVALUATE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = TRAINING_DIR / "datasets"
DEFAULT_MODEL_PATH = RUNS_DIR / "classify" / "food-dish" / "weights" / "best.pt"
DEFAULT_TEST_DIR = DATASETS_DIR / "food-classification" / "test"

# Priority cuisines
PRIORITY_CUISINES = ["Western", "Chinese", "Japanese", "Korean", "Vietnamese", "Thai"]

# Cuisine mapping (same as audit_cuisines.py for standalone usage)
CUISINE_MAP = {
    # -- Western --
    "hamburger": "Western", "hot-dog": "Western", "pizza": "Western",
    "french-fries": "Western", "steak": "Western", "grilled-steak": "Western",
    "filet-mignon": "Western", "prime-rib": "Western", "club-sandwich": "Western",
    "caesar-salad": "Western", "greek-salad": "Western",
    "pulled-pork-sandwich": "Western", "blt": "Western", "nachos": "Western",
    "tacos": "Western", "burrito": "Western", "enchilada": "Western",
    "quesadilla": "Western", "mac-and-cheese": "Western",
    "macaroni-and-cheese": "Western", "grilled-cheese-sandwich": "Western",
    "pancakes": "Western", "waffles": "Western", "french-toast": "Western",
    "bacon": "Western", "eggs-benedict": "Western", "omelette": "Western",
    "fish-and-chips": "Western", "chicken-wings": "Western",
    "fried-chicken": "Western", "grilled-chicken": "Western",
    "roast-chicken": "Western", "meatloaf": "Western",
    "baby-back-ribs": "Western", "pork-chop": "Western",
    "clam-chowder": "Western", "onion-soup": "Western",
    "french-onion-soup": "Western", "tomato-soup": "Western",
    "lobster-bisque": "Western", "lobster-roll-sandwich": "Western",
    "crab-cakes": "Western", "apple-pie": "Western",
    "cheesecake": "Western", "chocolate-cake": "Western",
    "carrot-cake": "Western", "ice-cream": "Western",
    "donuts": "Western", "cupcakes": "Western", "brownies": "Western",
    "churros": "Western", "creme-brulee": "Western",
    "tiramisu": "Western", "panna-cotta": "Western",
    "beignets": "Western", "cannoli": "Western", "scallops": "Western",
    "oysters": "Western", "garlic-bread": "Western",
    "bruschetta": "Western", "caprese-salad": "Western",
    "risotto": "Western", "gnocchi": "Western", "ravioli": "Western",
    "lasagna": "Western", "spaghetti-bolognese": "Western",
    "spaghetti-carbonara": "Western", "beef-carpaccio": "Western",
    "beef-tartare": "Western", "foie-gras": "Western",
    "escargots": "Western", "croque-madame": "Western", "crepes": "Western",
    "frozen-yogurt": "Western", "strawberry-shortcake": "Western",
    # -- Vietnamese --
    "pho": "Vietnamese", "banh-mi": "Vietnamese", "bun-cha": "Vietnamese",
    "spring-rolls": "Vietnamese", "goi-cuon": "Vietnamese",
    "com-tam": "Vietnamese", "bun-bo-hue": "Vietnamese",
    "cao-lau": "Vietnamese", "banh-xeo": "Vietnamese",
    # -- Chinese --
    "peking-duck": "Chinese", "fried-rice": "Chinese", "dumplings": "Chinese",
    "dim-sum": "Chinese", "kung-pao-chicken": "Chinese",
    "sweet-and-sour-chicken": "Chinese", "sweet-and-sour-pork": "Chinese",
    "general-tso-chicken": "Chinese", "orange-chicken": "Chinese",
    "mapo-tofu": "Chinese", "hot-pot": "Chinese",
    "hot-and-sour-soup": "Chinese", "wonton-soup": "Chinese",
    "char-siu": "Chinese", "chow-mein": "Chinese", "lo-mein": "Chinese",
    "dan-dan-noodles": "Chinese", "congee": "Chinese",
    "scallion-pancakes": "Chinese", "bao-bun": "Chinese",
    "steamed-buns": "Chinese", "xiao-long-bao": "Chinese",
    "shumai": "Chinese", "egg-tart": "Chinese", "moon-cake": "Chinese",
    "stir-fry": "Chinese", "beef-and-broccoli": "Chinese",
    "mongolian-beef": "Chinese", "crispy-duck": "Chinese",
    "nasi-goreng": "Chinese",
    # -- Japanese --
    "sushi": "Japanese", "sashimi": "Japanese", "ramen": "Japanese",
    "udon": "Japanese", "soba": "Japanese", "tempura": "Japanese",
    "teriyaki": "Japanese", "tonkatsu": "Japanese", "katsu": "Japanese",
    "onigiri": "Japanese", "takoyaki": "Japanese", "okonomiyaki": "Japanese",
    "gyoza": "Japanese", "edamame": "Japanese", "miso-soup": "Japanese",
    "matcha": "Japanese", "mochi": "Japanese", "yakitori": "Japanese",
    "donburi": "Japanese", "oyakodon": "Japanese", "katsudon": "Japanese",
    "curry-rice": "Japanese", "japanese-curry": "Japanese",
    "tamagoyaki": "Japanese", "natto": "Japanese", "chirashi": "Japanese",
    "wagyu": "Japanese", "teppanyaki": "Japanese",
    # -- Korean --
    "bibimbap": "Korean", "kimchi": "Korean", "bulgogi": "Korean",
    "japchae": "Korean", "tteokbokki": "Korean", "korean-bbq": "Korean",
    "kimbap": "Korean", "samgyeopsal": "Korean",
    "sundubu-jjigae": "Korean", "kimchi-jjigae": "Korean",
    "budae-jjigae": "Korean", "korean-fried-chicken": "Korean",
    "pajeon": "Korean", "galbi": "Korean", "naengmyeon": "Korean",
    "bingsu": "Korean", "hotteok": "Korean", "dakgalbi": "Korean",
    "jjajangmyeon": "Korean",
    # -- Thai --
    "pad-thai": "Thai", "green-curry": "Thai", "red-curry": "Thai",
    "massaman-curry": "Thai", "tom-yum": "Thai", "tom-yum-soup": "Thai",
    "tom-kha-gai": "Thai", "som-tum": "Thai", "papaya-salad": "Thai",
    "thai-iced-tea": "Thai", "mango-sticky-rice": "Thai",
    "satay": "Thai", "pad-see-ew": "Thai", "pad-kra-pao": "Thai",
    "khao-pad": "Thai", "larb": "Thai", "thai-basil-chicken": "Thai",
    "panang-curry": "Thai", "sticky-rice": "Thai", "kai-jeow": "Thai",
    # -- Indian --
    "samosa": "Indian", "biryani": "Indian", "butter-chicken": "Indian",
    "tikka-masala": "Indian", "chicken-tikka-masala": "Indian",
    "naan": "Indian", "tandoori": "Indian", "tandoori-chicken": "Indian",
    "dal": "Indian", "palak-paneer": "Indian", "pakora": "Indian",
    "masala-dosa": "Indian", "gulab-jamun": "Indian", "curry": "Indian",
    # -- Other --
    "hummus": "Other", "falafel": "Other", "shawarma": "Other",
    "kebab": "Other", "paella": "Other", "poutine": "Other",
    "baklava": "Other", "ceviche": "Other", "empanada": "Other",
}

KEYWORD_CUISINE = {
    "sushi": "Japanese", "ramen": "Japanese", "miso": "Japanese",
    "tempura": "Japanese", "teriyaki": "Japanese", "katsu": "Japanese",
    "udon": "Japanese", "soba": "Japanese", "mochi": "Japanese",
    "kimchi": "Korean", "bibim": "Korean", "bulgogi": "Korean",
    "korean": "Korean", "tteok": "Korean",
    "thai": "Thai", "pad": "Thai", "tom-yum": "Thai",
    "curry": "Indian", "masala": "Indian", "tandoori": "Indian",
    "naan": "Indian",
    "pho": "Vietnamese", "banh": "Vietnamese", "bun": "Vietnamese",
    "vietnamese": "Vietnamese",
    "chinese": "Chinese", "wonton": "Chinese", "dumpling": "Chinese",
    "dim-sum": "Chinese", "chow": "Chinese", "peking": "Chinese",
    "szechuan": "Chinese", "sichuan": "Chinese",
    "pizza": "Western", "burger": "Western", "sandwich": "Western",
    "steak": "Western", "pasta": "Western", "spaghetti": "Western",
    "lasagna": "Western", "fries": "Western", "cake": "Western",
    "pie": "Western", "ice-cream": "Western", "pancake": "Western",
    "waffle": "Western", "bread": "Western", "cheese": "Western",
    "bacon": "Western", "lobster": "Western",
}


def classify_cuisine(class_name: str) -> str:
    """Map a normalized class name to a cuisine group."""
    if class_name in CUISINE_MAP:
        return CUISINE_MAP[class_name]

    for keyword, cuisine in KEYWORD_CUISINE.items():
        if keyword in class_name:
            return cuisine

    return "Other"


def evaluate_classification(
    model_path: Path,
    test_dir: Path,
) -> dict:
    """Run classification evaluation and compute per-cuisine metrics.

    Expects test_dir to be in ImageFolder format:
        test_dir/
            class_name_1/
                img1.jpg
                img2.jpg
            class_name_2/
                ...

    Returns evaluation results dict.
    """
    from ultralytics import YOLO

    log.info("Loading model from %s", model_path)
    model = YOLO(str(model_path))

    # Get model class names
    try:
        model_classes = list(model.names.values())
    except Exception:
        model_classes = []
    log.info("Model has %d classes", len(model_classes))

    # Discover test classes from directory structure
    test_classes = sorted([
        d.name for d in test_dir.iterdir() if d.is_dir()
    ])
    if not test_classes:
        log.error("No class subdirectories found in %s", test_dir)
        return {}

    log.info("Found %d test classes", len(test_classes))

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Track per-cuisine and overall metrics
    cuisine_correct_top1: dict[str, int] = defaultdict(int)
    cuisine_correct_top5: dict[str, int] = defaultdict(int)
    cuisine_total: dict[str, int] = defaultdict(int)

    # Confusion matrix: (true_class, predicted_class) -> count
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    total_images = 0
    total_correct_top1 = 0
    total_correct_top5 = 0

    for cls_name in test_classes:
        cls_dir = test_dir / cls_name
        normalized_name = cls_name.lower().replace("_", "-").replace(" ", "-")
        cuisine = classify_cuisine(normalized_name)

        images = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in image_exts
        ]

        if not images:
            continue

        log.info("  Evaluating class '%s' (%s) -- %d images", cls_name, cuisine, len(images))

        for img_path in images:
            results = model.predict(str(img_path), verbose=False)

            if not results:
                cuisine_total[cuisine] += 1
                total_images += 1
                continue

            result = results[0]

            # Get top-5 predictions
            probs = result.probs
            if probs is None:
                cuisine_total[cuisine] += 1
                total_images += 1
                continue

            top5_indices = probs.top5
            top5_classes = [
                model_classes[i].lower().replace("_", "-").replace(" ", "-")
                if i < len(model_classes) else str(i)
                for i in top5_indices
            ]
            top1_class = top5_classes[0] if top5_classes else ""

            # Check correctness
            is_top1 = normalized_name == top1_class
            is_top5 = normalized_name in top5_classes

            if is_top1:
                total_correct_top1 += 1
                cuisine_correct_top1[cuisine] += 1
            if is_top5:
                total_correct_top5 += 1
                cuisine_correct_top5[cuisine] += 1

            cuisine_total[cuisine] += 1
            total_images += 1

            # Track confusion
            confusion[(normalized_name, top1_class)] += 1

    # Compute overall accuracy
    overall_top1 = total_correct_top1 / total_images if total_images > 0 else 0.0
    overall_top5 = total_correct_top5 / total_images if total_images > 0 else 0.0

    # Compute per-cuisine accuracy
    cuisine_acc = {}
    for cuisine in sorted(set(list(cuisine_total.keys()) + PRIORITY_CUISINES)):
        total = cuisine_total.get(cuisine, 0)
        if total > 0:
            top1_acc = cuisine_correct_top1.get(cuisine, 0) / total
            top5_acc = cuisine_correct_top5.get(cuisine, 0) / total
        else:
            top1_acc = 0.0
            top5_acc = 0.0
        cuisine_acc[cuisine] = {
            "images": total,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "top1_correct": cuisine_correct_top1.get(cuisine, 0),
            "top5_correct": cuisine_correct_top5.get(cuisine, 0),
        }

    # Find top-20 most confused class pairs (excluding correct predictions)
    confused_pairs = [
        (pair, count)
        for pair, count in confusion.items()
        if pair[0] != pair[1]
    ]
    confused_pairs.sort(key=lambda x: -x[1])
    top_confused = confused_pairs[:20]

    # Build confusion matrix for top confused pairs
    confusion_matrix = {
        "top_confused_pairs": [
            {
                "true_class": pair[0],
                "predicted_class": pair[1],
                "count": count,
                "true_cuisine": classify_cuisine(pair[0]),
                "predicted_cuisine": classify_cuisine(pair[1]),
            }
            for pair, count in top_confused
        ],
        "total_predictions": total_images,
        "total_unique_pairs": len(confused_pairs),
    }

    # Build report
    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("CLASSIFICATION MODEL -- PER-CUISINE EVALUATION REPORT")
    report_lines.append("=" * 75)
    report_lines.append(f"Model: {model_path}")
    report_lines.append(f"Test images: {total_images}")
    report_lines.append(f"Test classes: {len(test_classes)}")
    report_lines.append(f"Overall Top-1 Accuracy: {overall_top1:.3f} ({overall_top1:.1%})")
    report_lines.append(f"Overall Top-5 Accuracy: {overall_top5:.3f} ({overall_top5:.1%})")
    report_lines.append("")
    report_lines.append("-" * 75)
    report_lines.append(
        f"{'Cuisine':<15} | {'Images':>7} | {'Top-1':>7} | {'Top-5':>7} | {'Top-1 Acc':>9} | {'Top-5 Acc':>9}"
    )
    report_lines.append("-" * 75)

    warnings = []
    for cuisine in sorted(cuisine_acc.keys(), key=lambda c: -cuisine_acc[c]["top1_accuracy"]):
        stats = cuisine_acc[cuisine]
        t1 = stats["top1_accuracy"]
        t5 = stats["top5_accuracy"]

        flag = ""
        if cuisine in PRIORITY_CUISINES and t1 < 0.50:
            flag = " *** BELOW THRESHOLD ***"
            warnings.append(
                f"WARNING: {cuisine} Top-1={t1:.1%} < 50% threshold -- needs more training data"
            )

        report_lines.append(
            f"{cuisine:<15} | {stats['images']:>7} | "
            f"{stats['top1_correct']:>7} | {stats['top5_correct']:>7} | "
            f"{t1:>8.1%} | {t5:>8.1%}{flag}"
        )

    report_lines.append("-" * 75)
    report_lines.append("")

    # Priority cuisine summary
    report_lines.append("PRIORITY CUISINE STATUS:")
    report_lines.append("-" * 55)
    for cuisine in PRIORITY_CUISINES:
        stats = cuisine_acc.get(cuisine, {"images": 0, "top1_accuracy": 0.0, "top5_accuracy": 0.0})
        n_images = stats["images"]
        t1 = stats["top1_accuracy"]
        t5 = stats["top5_accuracy"]
        status = "OK" if t1 >= 0.50 else ("LOW" if n_images > 0 else "NONE")
        report_lines.append(
            f"  [{status:>4}] {cuisine:<15} Top-1={t1:.1%}  Top-5={t5:.1%}  ({n_images} images)"
        )
    report_lines.append("")

    # Top confused pairs
    report_lines.append("TOP-20 MOST CONFUSED CLASS PAIRS:")
    report_lines.append("-" * 75)
    report_lines.append(
        f"  {'True Class':<25} {'Predicted As':<25} {'Count':>6} {'Cuisines'}"
    )
    report_lines.append("  " + "-" * 73)
    for pair, count in top_confused:
        true_cuisine = classify_cuisine(pair[0])
        pred_cuisine = classify_cuisine(pair[1])
        cross = " (cross-cuisine)" if true_cuisine != pred_cuisine else ""
        report_lines.append(
            f"  {pair[0]:<25} {pair[1]:<25} {count:>6} {true_cuisine}->{pred_cuisine}{cross}"
        )
    report_lines.append("")

    if warnings:
        report_lines.append("WARNINGS:")
        for w in warnings:
            report_lines.append(f"  {w}")
        report_lines.append("")
    else:
        report_lines.append("All priority cuisines meet Top-1 >= 50% threshold.")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Print to stdout
    print(report_text)

    # Save report to file
    report_path = EVALUATE_DIR / "classification_cuisine_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info("Report saved to %s", report_path)

    # Save confusion matrix to JSON
    confusion_path = EVALUATE_DIR / "confusion_matrix.json"
    with open(confusion_path, "w") as f:
        json.dump(confusion_matrix, f, indent=2)
    log.info("Confusion matrix saved to %s", confusion_path)

    # Return structured results
    results = {
        "model_path": str(model_path),
        "test_images": total_images,
        "test_classes": len(test_classes),
        "overall_top1": overall_top1,
        "overall_top5": overall_top5,
        "per_cuisine": cuisine_acc,
        "warnings": warnings,
        "confusion_matrix_path": str(confusion_path),
    }

    # Save JSON results
    json_path = EVALUATE_DIR / "classification_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("JSON results saved to %s", json_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classification model with per-cuisine Top-1/Top-5 accuracy"
    )
    parser.add_argument(
        "--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to trained model weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-dir", type=str, default=str(DEFAULT_TEST_DIR),
        help=f"Directory with test images in ImageFolder format (default: {DEFAULT_TEST_DIR})",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)

    # Check model exists
    if not model_path.exists():
        print(
            f"Trained classification model not found at {model_path}.\n"
            f"Please run train_classify.py first to train the model.\n\n"
            f"Usage:\n"
            f"  python training/train_classify.py --prepare-data\n"
            f"  python training/evaluate/eval_classification.py"
        )
        sys.exit(0)

    # Check test directory exists
    if not test_dir.exists():
        print(
            f"Test images directory not found at {test_dir}.\n"
            f"Please prepare the dataset first or specify --test-dir.\n\n"
            f"Usage:\n"
            f"  python training/train_classify.py --prepare-data  # creates test split\n"
            f"  python training/evaluate/eval_classification.py --test-dir path/to/test/"
        )
        sys.exit(0)

    evaluate_classification(model_path, test_dir)


if __name__ == "__main__":
    main()
