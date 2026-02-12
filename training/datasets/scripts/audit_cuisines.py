#!/usr/bin/env python3
"""
Audit cuisine coverage in the food classification dataset.

Categorizes each class into cuisine groups and reports coverage.
Priority cuisines: Australian/Western, Chinese, Japanese, Korean, Vietnamese, Thai.

Usage:
    python training/datasets/scripts/audit_cuisines.py [--dataset-dir DIR]
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent.parent
DATASETS_DIR = TRAINING_DIR / "datasets"
CLS_DIR = DATASETS_DIR / "food-classification"

# Priority cuisines that must have adequate representation
PRIORITY_CUISINES = ["Western", "Chinese", "Japanese", "Korean", "Vietnamese", "Thai"]
MIN_COVERAGE_PCT = 5.0  # Flag if any priority cuisine has < 5% representation

# Hardcoded cuisine mapping for well-known dishes
# Keys are normalized class names (lowercase-hyphenated)
CUISINE_MAP = {
    # -- Western / Australian --
    "hamburger": "Western",
    "hot-dog": "Western",
    "pizza": "Western",
    "french-fries": "Western",
    "steak": "Western",
    "grilled-steak": "Western",
    "filet-mignon": "Western",
    "prime-rib": "Western",
    "club-sandwich": "Western",
    "caesar-salad": "Western",
    "greek-salad": "Western",
    "pulled-pork-sandwich": "Western",
    "blt": "Western",
    "nachos": "Western",
    "tacos": "Western",
    "burrito": "Western",
    "enchilada": "Western",
    "quesadilla": "Western",
    "mac-and-cheese": "Western",
    "macaroni-and-cheese": "Western",
    "grilled-cheese-sandwich": "Western",
    "pancakes": "Western",
    "waffles": "Western",
    "french-toast": "Western",
    "bacon": "Western",
    "eggs-benedict": "Western",
    "omelette": "Western",
    "scrambled-eggs": "Western",
    "fried-egg": "Western",
    "deviled-eggs": "Western",
    "fish-and-chips": "Western",
    "chicken-wings": "Western",
    "chicken-nuggets": "Western",
    "fried-chicken": "Western",
    "grilled-chicken": "Western",
    "roast-chicken": "Western",
    "meatloaf": "Western",
    "pot-roast": "Western",
    "baby-back-ribs": "Western",
    "bbq-ribs": "Western",
    "pork-chop": "Western",
    "lamb-chop": "Western",
    "clam-chowder": "Western",
    "onion-soup": "Western",
    "french-onion-soup": "Western",
    "tomato-soup": "Western",
    "lobster-bisque": "Western",
    "lobster-roll-sandwich": "Western",
    "crab-cakes": "Western",
    "shrimp-and-grits": "Western",
    "bread-pudding": "Western",
    "apple-pie": "Western",
    "cheesecake": "Western",
    "chocolate-cake": "Western",
    "carrot-cake": "Western",
    "red-velvet-cake": "Western",
    "ice-cream": "Western",
    "donuts": "Western",
    "doughnut": "Western",
    "cupcakes": "Western",
    "brownies": "Western",
    "churros": "Western",
    "creme-brulee": "Western",
    "tiramisu": "Western",
    "panna-cotta": "Western",
    "beignets": "Western",
    "cannoli": "Western",
    "scallops": "Western",
    "oysters": "Western",
    "garlic-bread": "Western",
    "bruschetta": "Western",
    "caprese-salad": "Western",
    "risotto": "Western",
    "gnocchi": "Western",
    "ravioli": "Western",
    "lasagna": "Western",
    "spaghetti-bolognese": "Western",
    "spaghetti-carbonara": "Western",
    "penne-arrabiata": "Western",
    "fettuccine-alfredo": "Western",
    "beef-carpaccio": "Western",
    "beef-tartare": "Western",
    "foie-gras": "Western",
    "escargots": "Western",
    "croque-madame": "Western",
    "crepes": "Western",
    "frozen-yogurt": "Western",
    "strawberry-shortcake": "Western",
    "peking-duck": "Chinese",
    "pho": "Vietnamese",
    "banh-mi": "Vietnamese",
    "bun-cha": "Vietnamese",
    "spring-rolls": "Vietnamese",
    "goi-cuon": "Vietnamese",
    "com-tam": "Vietnamese",
    "bun-bo-hue": "Vietnamese",
    "cao-lau": "Vietnamese",
    "banh-xeo": "Vietnamese",
    "vietnamese-spring-rolls": "Vietnamese",
    # -- Chinese --
    "fried-rice": "Chinese",
    "dumplings": "Chinese",
    "dim-sum": "Chinese",
    "kung-pao-chicken": "Chinese",
    "sweet-and-sour-chicken": "Chinese",
    "sweet-and-sour-pork": "Chinese",
    "general-tso-chicken": "Chinese",
    "orange-chicken": "Chinese",
    "mapo-tofu": "Chinese",
    "hot-pot": "Chinese",
    "hot-and-sour-soup": "Chinese",
    "wonton-soup": "Chinese",
    "egg-drop-soup": "Chinese",
    "char-siu": "Chinese",
    "char-siu-pork": "Chinese",
    "chow-mein": "Chinese",
    "lo-mein": "Chinese",
    "dan-dan-noodles": "Chinese",
    "zhajiangmian": "Chinese",
    "congee": "Chinese",
    "scallion-pancakes": "Chinese",
    "bao-bun": "Chinese",
    "steamed-buns": "Chinese",
    "xiao-long-bao": "Chinese",
    "shumai": "Chinese",
    "egg-tart": "Chinese",
    "moon-cake": "Chinese",
    "mooncake": "Chinese",
    "tanghulu": "Chinese",
    "chinese-broccoli": "Chinese",
    "stir-fry": "Chinese",
    "beef-and-broccoli": "Chinese",
    "mongolian-beef": "Chinese",
    "crispy-duck": "Chinese",
    "ma-la": "Chinese",
    "szechuan": "Chinese",
    "nasi-goreng": "Chinese",
    # -- Japanese --
    "sushi": "Japanese",
    "sashimi": "Japanese",
    "ramen": "Japanese",
    "udon": "Japanese",
    "soba": "Japanese",
    "tempura": "Japanese",
    "teriyaki": "Japanese",
    "tonkatsu": "Japanese",
    "katsu": "Japanese",
    "onigiri": "Japanese",
    "takoyaki": "Japanese",
    "okonomiyaki": "Japanese",
    "gyoza": "Japanese",
    "edamame": "Japanese",
    "miso-soup": "Japanese",
    "matcha": "Japanese",
    "mochi": "Japanese",
    "yakitori": "Japanese",
    "donburi": "Japanese",
    "oyakodon": "Japanese",
    "katsudon": "Japanese",
    "curry-rice": "Japanese",
    "japanese-curry": "Japanese",
    "tamagoyaki": "Japanese",
    "natto": "Japanese",
    "chirashi": "Japanese",
    "kaiseki": "Japanese",
    "wagyu": "Japanese",
    "teppanyaki": "Japanese",
    # -- Korean --
    "bibimbap": "Korean",
    "kimchi": "Korean",
    "bulgogi": "Korean",
    "japchae": "Korean",
    "tteokbokki": "Korean",
    "korean-bbq": "Korean",
    "kimbap": "Korean",
    "samgyeopsal": "Korean",
    "sundubu-jjigae": "Korean",
    "kimchi-jjigae": "Korean",
    "budae-jjigae": "Korean",
    "korean-fried-chicken": "Korean",
    "pajeon": "Korean",
    "galbi": "Korean",
    "naengmyeon": "Korean",
    "bingsu": "Korean",
    "hotteok": "Korean",
    "gochujang": "Korean",
    "dakgalbi": "Korean",
    "jjajangmyeon": "Korean",
    # -- Thai --
    "pad-thai": "Thai",
    "green-curry": "Thai",
    "red-curry": "Thai",
    "massaman-curry": "Thai",
    "tom-yum": "Thai",
    "tom-yum-soup": "Thai",
    "tom-kha-gai": "Thai",
    "som-tum": "Thai",
    "papaya-salad": "Thai",
    "thai-iced-tea": "Thai",
    "mango-sticky-rice": "Thai",
    "satay": "Thai",
    "pad-see-ew": "Thai",
    "pad-kra-pao": "Thai",
    "khao-pad": "Thai",
    "larb": "Thai",
    "thai-basil-chicken": "Thai",
    "panang-curry": "Thai",
    "sticky-rice": "Thai",
    "kai-jeow": "Thai",
    # -- Indian --
    "samosa": "Indian",
    "biryani": "Indian",
    "butter-chicken": "Indian",
    "tikka-masala": "Indian",
    "chicken-tikka-masala": "Indian",
    "naan": "Indian",
    "tandoori": "Indian",
    "tandoori-chicken": "Indian",
    "dal": "Indian",
    "daal": "Indian",
    "palak-paneer": "Indian",
    "paneer-tikka": "Indian",
    "pakora": "Indian",
    "chaat": "Indian",
    "masala-dosa": "Indian",
    "idli": "Indian",
    "gulab-jamun": "Indian",
    "jalebi": "Indian",
    "vindaloo": "Indian",
    "korma": "Indian",
    "raita": "Indian",
    "roti": "Indian",
    "paratha": "Indian",
    "chapati": "Indian",
    "curry": "Indian",
    # -- Other (cuisines not in the priority list) --
    "hummus": "Other",
    "falafel": "Other",
    "shawarma": "Other",
    "kebab": "Other",
    "paella": "Other",
    "poutine": "Other",
    "baklava": "Other",
    "couscous": "Other",
    "ceviche": "Other",
    "empanada": "Other",
    "arepas": "Other",
    "pierogi": "Other",
    "borscht": "Other",
}


def classify_cuisine(class_name: str) -> str:
    """Map a normalized class name to a cuisine group."""
    # Direct lookup first
    if class_name in CUISINE_MAP:
        return CUISINE_MAP[class_name]

    # Keyword-based heuristics for unmapped classes
    keywords_to_cuisine = {
        "sushi": "Japanese",
        "ramen": "Japanese",
        "miso": "Japanese",
        "tempura": "Japanese",
        "teriyaki": "Japanese",
        "katsu": "Japanese",
        "udon": "Japanese",
        "soba": "Japanese",
        "mochi": "Japanese",
        "kimchi": "Korean",
        "bibim": "Korean",
        "bulgogi": "Korean",
        "korean": "Korean",
        "tteok": "Korean",
        "thai": "Thai",
        "pad": "Thai",
        "tom-yum": "Thai",
        "curry": "Indian",
        "masala": "Indian",
        "tandoori": "Indian",
        "naan": "Indian",
        "pho": "Vietnamese",
        "banh": "Vietnamese",
        "bun": "Vietnamese",
        "vietnamese": "Vietnamese",
        "chinese": "Chinese",
        "wonton": "Chinese",
        "dumpling": "Chinese",
        "dim-sum": "Chinese",
        "chow": "Chinese",
        "peking": "Chinese",
        "szechuan": "Chinese",
        "sichuan": "Chinese",
        "pizza": "Western",
        "burger": "Western",
        "sandwich": "Western",
        "steak": "Western",
        "pasta": "Western",
        "spaghetti": "Western",
        "lasagna": "Western",
        "fries": "Western",
        "cake": "Western",
        "pie": "Western",
        "ice-cream": "Western",
        "pancake": "Western",
        "waffle": "Western",
        "bread": "Western",
        "cheese": "Western",
        "bacon": "Western",
        "lobster": "Western",
        "clam": "Western",
    }

    for keyword, cuisine in keywords_to_cuisine.items():
        if keyword in class_name:
            return cuisine

    return "Other"


def count_images_in_class(cls_dir: Path) -> int:
    """Count images in a class directory across all splits."""
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        count += len(list(cls_dir.glob(ext)))
    return count


def audit_cuisines(dataset_dir: Path) -> dict:
    """Run cuisine audit on the classification dataset.

    Returns dict with per-cuisine statistics.
    """
    classes_file = dataset_dir / "classes.txt"
    if not classes_file.exists():
        log.error("classes.txt not found at %s", classes_file)
        return {}

    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]

    log.info("Auditing %d classes for cuisine coverage...", len(classes))

    # Classify each class
    cuisine_classes: dict[str, list[str]] = {}
    cuisine_images: dict[str, int] = {}
    class_cuisine_map: dict[str, str] = {}

    for cls_name in classes:
        cuisine = classify_cuisine(cls_name)
        class_cuisine_map[cls_name] = cuisine

        if cuisine not in cuisine_classes:
            cuisine_classes[cuisine] = []
            cuisine_images[cuisine] = 0
        cuisine_classes[cuisine].append(cls_name)

        # Count images across all splits
        for split in ["train", "val", "test"]:
            cls_split_dir = dataset_dir / split / cls_name
            if cls_split_dir.exists():
                cuisine_images[cuisine] += count_images_in_class(cls_split_dir)

    total_images = sum(cuisine_images.values())
    total_classes = len(classes)

    # Build report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CUISINE AUDIT REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Dataset: {dataset_dir}")
    report_lines.append(f"Total classes: {total_classes}")
    report_lines.append(f"Total images: {total_images}")
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Cuisine':<15} {'Classes':>8} {'% Classes':>10} {'Images':>10} {'% Images':>10}")
    report_lines.append("-" * 70)

    warnings = []
    cuisine_stats = {}

    for cuisine in sorted(cuisine_classes.keys(), key=lambda c: -cuisine_images.get(c, 0)):
        n_classes = len(cuisine_classes[cuisine])
        n_images = cuisine_images[cuisine]
        pct_classes = (n_classes / total_classes * 100) if total_classes > 0 else 0
        pct_images = (n_images / total_images * 100) if total_images > 0 else 0

        flag = ""
        if cuisine in PRIORITY_CUISINES and pct_images < MIN_COVERAGE_PCT:
            flag = " *** LOW COVERAGE ***"
            warnings.append(
                f"WARNING: {cuisine} has only {pct_images:.1f}% image coverage "
                f"(threshold: {MIN_COVERAGE_PCT}%)"
            )

        report_lines.append(
            f"{cuisine:<15} {n_classes:>8} {pct_classes:>9.1f}% {n_images:>10} {pct_images:>9.1f}%{flag}"
        )

        cuisine_stats[cuisine] = {
            "classes": n_classes,
            "images": n_images,
            "pct_classes": round(pct_classes, 1),
            "pct_images": round(pct_images, 1),
            "class_names": cuisine_classes[cuisine],
        }

    report_lines.append("-" * 70)
    report_lines.append("")

    # Priority cuisine summary
    report_lines.append("PRIORITY CUISINE COVERAGE:")
    report_lines.append("-" * 40)
    for cuisine in PRIORITY_CUISINES:
        stats = cuisine_stats.get(cuisine, {"classes": 0, "images": 0, "pct_images": 0})
        status = "OK" if stats["pct_images"] >= MIN_COVERAGE_PCT else "LOW"
        report_lines.append(
            f"  [{status:>3}] {cuisine:<15} {stats['classes']:>4} classes, {stats['images']:>8} images ({stats['pct_images']:.1f}%)"
        )
    report_lines.append("")

    if warnings:
        report_lines.append("WARNINGS:")
        for w in warnings:
            report_lines.append(f"  {w}")
        report_lines.append("")
    else:
        report_lines.append("All priority cuisines have adequate coverage (>= 5%).")
        report_lines.append("")

    # Per-class mapping
    report_lines.append("PER-CLASS CUISINE MAPPING:")
    report_lines.append("-" * 40)
    for cls_name in sorted(class_cuisine_map.keys()):
        report_lines.append(f"  {cls_name:<40} -> {class_cuisine_map[cls_name]}")

    report_text = "\n".join(report_lines)

    # Save report
    report_path = DATASETS_DIR / "cuisine_audit_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info("Cuisine audit report saved to %s", report_path)

    # Also save structured data as JSON
    json_report = {
        "total_classes": total_classes,
        "total_images": total_images,
        "cuisine_stats": cuisine_stats,
        "class_cuisine_map": class_cuisine_map,
        "warnings": warnings,
    }
    json_path = DATASETS_DIR / "cuisine_audit_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Print report
    print(report_text)

    return json_report


def main():
    parser = argparse.ArgumentParser(description="Audit cuisine coverage in classification dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(CLS_DIR),
        help=f"Classification dataset directory (default: {CLS_DIR})",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        log.error(
            "Classification dataset not found at %s. Run merge_datasets.py first.",
            dataset_dir,
        )
        return

    result = audit_cuisines(dataset_dir)

    if result.get("warnings"):
        log.warning(
            "%d cuisine coverage warnings. See report for details.",
            len(result["warnings"]),
        )


if __name__ == "__main__":
    main()
