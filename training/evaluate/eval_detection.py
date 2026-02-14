#!/usr/bin/env python3
"""
Per-cuisine detection evaluation for trained YOLO food detector.

Loads a trained detection model and evaluates predictions on the test set,
grouping results by cuisine using the cuisine mapping from audit_cuisines.py.
Reports per-cuisine mAP@0.5 and mAP@0.5:0.95, flagging any cuisine with
mAP@0.5 below 0.50.

Usage:
    python training/evaluate/eval_detection.py
    python training/evaluate/eval_detection.py --model-path path/to/best.pt
    python training/evaluate/eval_detection.py --test-dir path/to/test/images

Prerequisites:
    - Trained detection model (run train_detect.py first)
    - Test images organized by class or with YOLO-format labels
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
DEFAULT_MODEL_PATH = RUNS_DIR / "detect" / "food-detect" / "weights" / "best.pt"
DEFAULT_TEST_DIR = DATASETS_DIR / "food-detection-merged" / "images" / "test"
DEFAULT_LABELS_DIR = DATASETS_DIR / "food-detection-merged" / "labels" / "test"

# Import cuisine mapping -- replicate from audit_cuisines.py for standalone usage
# Priority cuisines that must have adequate representation
PRIORITY_CUISINES = ["Western", "Chinese", "Japanese", "Korean", "Vietnamese", "Thai"]

# Hardcoded cuisine mapping for well-known dishes
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

# Keyword-based heuristics for unmapped classes
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


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int = 1, img_h: int = 1) -> list[float]:
    """Convert YOLO format (center_x, center_y, width, height) to [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_ground_truth_labels(labels_dir: Path) -> dict:
    """Load YOLO-format ground truth labels.

    Returns dict mapping image stem -> list of (class_id, x1, y1, x2, y2).
    """
    gt = {}
    if not labels_dir.exists():
        return gt

    for label_file in labels_dir.glob("*.txt"):
        boxes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    box = yolo_to_xyxy(cx, cy, w, h)
                    boxes.append((cls_id, box))
        gt[label_file.stem] = boxes

    return gt


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision from precision-recall curve (11-point interpolation)."""
    if not precisions or not recalls:
        return 0.0

    # 11-point interpolation
    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Find max precision at recall >= t
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        if p_at_r:
            ap += max(p_at_r)
    return ap / 11.0


def evaluate_detections(
    model_path: Path,
    test_dir: Path,
    labels_dir: Path,
    class_names: list[str] | None = None,
) -> dict:
    """Run detection evaluation and compute per-cuisine metrics.

    Returns evaluation results dict.
    """
    from ultralytics import YOLO

    log.info("Loading model from %s", model_path)
    model = YOLO(str(model_path))

    # Get class names from model if not provided
    if class_names is None:
        try:
            class_names = list(model.names.values())
        except Exception:
            class_names = ["food"]

    log.info("Model classes: %s", class_names[:20])

    # Load ground truth labels
    gt_labels = load_ground_truth_labels(labels_dir)
    log.info("Loaded ground truth for %d images", len(gt_labels))

    # Collect test images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_images = sorted([
        f for f in test_dir.iterdir()
        if f.suffix.lower() in image_exts
    ])

    if not test_images:
        log.error("No test images found in %s", test_dir)
        return {}

    log.info("Running detection on %d test images...", len(test_images))

    # Per-class tracking for mAP computation
    # For each class, collect (confidence, is_tp) tuples and total gt count
    class_detections: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    class_gt_count: dict[int, int] = defaultdict(int)

    # Per-cuisine image tracking
    cuisine_images: dict[str, int] = defaultdict(int)
    cuisine_detections_count: dict[str, int] = defaultdict(int)

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Track per-cuisine AP accumulators for multi-threshold mAP
    cuisine_class_det: dict[str, dict[int, list[tuple[float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cuisine_class_det_multi: dict[str, dict[float, dict[int, list[tuple[float, bool]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    cuisine_class_gt: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for img_path in test_images:
        # Run prediction
        results = model.predict(str(img_path), verbose=False)

        if not results:
            continue

        result = results[0]
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append((cls_id, conf, xyxy))

        # Determine cuisine from the first detected class (or from ground truth)
        cuisine = "Other"
        gt_for_image = gt_labels.get(img_path.stem, [])

        # Count ground truth per class
        for gt_cls, gt_box in gt_for_image:
            if gt_cls < len(class_names):
                cls_name = class_names[gt_cls].lower().replace("_", "-").replace(" ", "-")
                cuisine = classify_cuisine(cls_name)
            class_gt_count[gt_cls] += 1
            cuisine_class_gt[cuisine][gt_cls] += 1

        if not gt_for_image and pred_boxes:
            # No ground truth; use predicted class for cuisine
            cls_id = pred_boxes[0][0]
            if cls_id < len(class_names):
                cls_name = class_names[cls_id].lower().replace("_", "-").replace(" ", "-")
                cuisine = classify_cuisine(cls_name)

        cuisine_images[cuisine] += 1
        cuisine_detections_count[cuisine] += len(pred_boxes)

        # Match predictions to ground truth at IoU=0.5
        gt_matched = [False] * len(gt_for_image)
        sorted_preds = sorted(pred_boxes, key=lambda x: -x[1])  # sort by confidence

        for cls_id, conf, pred_xyxy in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, (gt_cls, gt_box) in enumerate(gt_for_image):
                if gt_matched[gt_idx] or gt_cls != cls_id:
                    continue
                iou = compute_iou(pred_xyxy, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = best_iou >= 0.5 and best_gt_idx >= 0
            if is_tp:
                gt_matched[best_gt_idx] = True

            class_detections[cls_id].append((conf, is_tp))
            cuisine_class_det[cuisine][cls_id].append((conf, is_tp))

            # Multi-threshold matching for mAP@0.5:0.95
            for iou_thresh in iou_thresholds:
                is_tp_t = best_iou >= iou_thresh and best_gt_idx >= 0
                cuisine_class_det_multi[cuisine][iou_thresh][cls_id].append((conf, is_tp_t))

    # Compute per-class AP at IoU=0.5
    class_aps = {}
    for cls_id in sorted(set(list(class_detections.keys()) + list(class_gt_count.keys()))):
        dets = sorted(class_detections.get(cls_id, []), key=lambda x: -x[0])
        total_gt = class_gt_count.get(cls_id, 0)

        if total_gt == 0:
            class_aps[cls_id] = 0.0
            continue

        tp_cumsum = 0
        precisions = []
        recalls = []
        for i, (conf, is_tp) in enumerate(dets):
            if is_tp:
                tp_cumsum += 1
            prec = tp_cumsum / (i + 1)
            rec = tp_cumsum / total_gt
            precisions.append(prec)
            recalls.append(rec)

        class_aps[cls_id] = compute_ap(precisions, recalls)

    # Compute overall mAP
    if class_aps:
        overall_map50 = sum(class_aps.values()) / len(class_aps)
    else:
        overall_map50 = 0.0

    # Compute per-cuisine mAP@0.5
    cuisine_map50 = {}
    for cuisine in sorted(set(list(cuisine_class_det.keys()) + list(cuisine_class_gt.keys()))):
        cls_dets = cuisine_class_det.get(cuisine, {})
        cls_gts = cuisine_class_gt.get(cuisine, {})
        all_cls = set(list(cls_dets.keys()) + list(cls_gts.keys()))

        aps = []
        for cls_id in all_cls:
            dets = sorted(cls_dets.get(cls_id, []), key=lambda x: -x[0])
            total_gt = cls_gts.get(cls_id, 0)
            if total_gt == 0:
                continue

            tp_cumsum = 0
            precisions = []
            recalls = []
            for i, (conf, is_tp) in enumerate(dets):
                if is_tp:
                    tp_cumsum += 1
                prec = tp_cumsum / (i + 1)
                rec = tp_cumsum / total_gt
                precisions.append(prec)
                recalls.append(rec)

            aps.append(compute_ap(precisions, recalls))

        cuisine_map50[cuisine] = sum(aps) / len(aps) if aps else 0.0

    # Compute per-cuisine mAP@0.5:0.95 (average across thresholds)
    cuisine_map5095 = {}
    for cuisine in cuisine_map50:
        threshold_maps = []
        for iou_thresh in iou_thresholds:
            cls_dets = cuisine_class_det_multi.get(cuisine, {}).get(iou_thresh, {})
            cls_gts = cuisine_class_gt.get(cuisine, {})
            all_cls = set(list(cls_dets.keys()) + list(cls_gts.keys()))

            aps = []
            for cls_id in all_cls:
                dets = sorted(cls_dets.get(cls_id, []), key=lambda x: -x[0])
                total_gt = cls_gts.get(cls_id, 0)
                if total_gt == 0:
                    continue

                tp_cumsum = 0
                precisions = []
                recalls = []
                for i, (conf, is_tp) in enumerate(dets):
                    if is_tp:
                        tp_cumsum += 1
                    prec = tp_cumsum / (i + 1)
                    rec = tp_cumsum / total_gt
                    precisions.append(prec)
                    recalls.append(rec)

                aps.append(compute_ap(precisions, recalls))

            threshold_maps.append(sum(aps) / len(aps) if aps else 0.0)

        cuisine_map5095[cuisine] = sum(threshold_maps) / len(threshold_maps) if threshold_maps else 0.0

    # Compute overall mAP@0.5:0.95
    overall_map5095 = sum(cuisine_map5095.values()) / len(cuisine_map5095) if cuisine_map5095 else 0.0

    # Build report
    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("DETECTION MODEL -- PER-CUISINE EVALUATION REPORT")
    report_lines.append("=" * 75)
    report_lines.append(f"Model: {model_path}")
    report_lines.append(f"Test images: {len(test_images)}")
    report_lines.append(f"Overall mAP@0.5: {overall_map50:.3f}")
    report_lines.append(f"Overall mAP@0.5:0.95: {overall_map5095:.3f}")
    report_lines.append("")
    report_lines.append("-" * 75)
    report_lines.append(
        f"{'Cuisine':<15} | {'Images':>7} | {'Detections':>10} | {'mAP@0.5':>8} | {'mAP@0.5:0.95':>13}"
    )
    report_lines.append("-" * 75)

    warnings = []
    all_cuisines = sorted(
        set(list(cuisine_map50.keys()) + PRIORITY_CUISINES),
        key=lambda c: -cuisine_map50.get(c, 0),
    )

    for cuisine in all_cuisines:
        n_images = cuisine_images.get(cuisine, 0)
        n_dets = cuisine_detections_count.get(cuisine, 0)
        m50 = cuisine_map50.get(cuisine, 0.0)
        m5095 = cuisine_map5095.get(cuisine, 0.0)

        flag = ""
        if cuisine in PRIORITY_CUISINES and m50 < 0.50:
            flag = " *** BELOW THRESHOLD ***"
            warnings.append(
                f"WARNING: {cuisine} mAP@0.5={m50:.3f} < 0.50 threshold -- needs more training data"
            )

        report_lines.append(
            f"{cuisine:<15} | {n_images:>7} | {n_dets:>10} | {m50:>8.3f} | {m5095:>13.3f}{flag}"
        )

    report_lines.append("-" * 75)
    report_lines.append("")

    # Priority cuisine summary
    report_lines.append("PRIORITY CUISINE STATUS:")
    report_lines.append("-" * 45)
    for cuisine in PRIORITY_CUISINES:
        m50 = cuisine_map50.get(cuisine, 0.0)
        n_images = cuisine_images.get(cuisine, 0)
        status = "OK" if m50 >= 0.50 else ("LOW" if n_images > 0 else "NONE")
        report_lines.append(
            f"  [{status:>4}] {cuisine:<15} mAP@0.5={m50:.3f}  ({n_images} images)"
        )
    report_lines.append("")

    if warnings:
        report_lines.append("WARNINGS:")
        for w in warnings:
            report_lines.append(f"  {w}")
        report_lines.append("")
    else:
        report_lines.append("All priority cuisines meet mAP@0.5 >= 0.50 threshold.")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Print to stdout
    print(report_text)

    # Save report to file
    report_path = EVALUATE_DIR / "detection_cuisine_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info("Report saved to %s", report_path)

    # Return structured results
    results = {
        "model_path": str(model_path),
        "test_images": len(test_images),
        "overall_map50": overall_map50,
        "overall_map5095": overall_map5095,
        "per_cuisine": {
            cuisine: {
                "images": cuisine_images.get(cuisine, 0),
                "detections": cuisine_detections_count.get(cuisine, 0),
                "map50": cuisine_map50.get(cuisine, 0.0),
                "map5095": cuisine_map5095.get(cuisine, 0.0),
            }
            for cuisine in all_cuisines
        },
        "warnings": warnings,
    }

    # Save JSON results
    json_path = EVALUATE_DIR / "detection_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("JSON results saved to %s", json_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection model with per-cuisine mAP breakdown"
    )
    parser.add_argument(
        "--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to trained model weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-dir", type=str, default=str(DEFAULT_TEST_DIR),
        help=f"Directory with test images (default: {DEFAULT_TEST_DIR})",
    )
    parser.add_argument(
        "--labels-dir", type=str, default=str(DEFAULT_LABELS_DIR),
        help=f"Directory with ground truth labels (default: {DEFAULT_LABELS_DIR})",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    labels_dir = Path(args.labels_dir)

    # Check model exists
    if not model_path.exists():
        print(
            f"Trained detection model not found at {model_path}.\n"
            f"Please run train_detect.py first to train the model.\n\n"
            f"Usage:\n"
            f"  python training/train_detect.py --prepare-data\n"
            f"  python training/evaluate/eval_detection.py"
        )
        sys.exit(0)

    # Check test directory exists
    if not test_dir.exists():
        print(
            f"Test images directory not found at {test_dir}.\n"
            f"Please prepare the dataset first or specify --test-dir.\n\n"
            f"Usage:\n"
            f"  python training/train_detect.py --prepare-data  # creates test split\n"
            f"  python training/evaluate/eval_detection.py --test-dir path/to/test/images"
        )
        sys.exit(0)

    evaluate_detections(model_path, test_dir, labels_dir)


if __name__ == "__main__":
    main()
