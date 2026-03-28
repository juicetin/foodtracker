"""
Scoring metrics for food identification accuracy.

Compares Gemini Nano output against labeled ground truth across:
  - Dish name matching (fuzzy)
  - Ingredient recall & precision
  - Weight estimation accuracy (MAE)
  - JSON parse success rate
"""

import json
import re
from difflib import SequenceMatcher


def _normalize(s) -> str:
    """Lowercase, strip, collapse whitespace. Handles non-string inputs."""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s.lower().strip())


def _to_grams(val) -> float:
    """Coerce amount_g to float. Handles strings like '100', '100g', etc."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.]", "", val)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return 0.0


def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Check if two strings are fuzzy-equal."""
    a, b = _normalize(a), _normalize(b)
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _best_match(name: str, candidates: list[str]) -> str | None:
    """Find the best fuzzy match for name in candidates."""
    name_n = _normalize(name)
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, name_n, _normalize(c)).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


def parse_prediction(text: str) -> list[dict] | None:
    """
    Parse Gemini Nano's raw text into a list of dish dicts.
    Returns None if JSON parsing fails entirely.
    """
    if not text or text.startswith("ERROR:"):
        return None

    # Strip markdown fences
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
    if s.endswith("```"):
        s = s[:-3].strip()

    # Try direct parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "dishes" in parsed:
            return parsed["dishes"]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        pass

    # Salvage truncated JSON — find last complete object
    for i in range(len(s) - 1, 0, -1):
        if s[i] in ("}",):
            try:
                candidate = s[: i + 1]
                # Auto-close arrays/objects
                opens = candidate.count("[") - candidate.count("]")
                candidate += "]" * max(opens, 0)
                opens = candidate.count("{") - candidate.count("}")
                candidate += "}" * max(opens, 0)
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "dishes" in parsed:
                    return parsed["dishes"]
                return [parsed] if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue

    return None


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------

def dish_name_f1(pred_dishes: list[dict], gt_dishes: list[dict]) -> float:
    """F1 score for dish name matching (fuzzy)."""
    if not gt_dishes:
        return 1.0 if not pred_dishes else 0.0
    if not pred_dishes:
        return 0.0

    pred_names = [d.get("name", "") for d in pred_dishes]
    gt_names = [d.get("name", "") for d in gt_dishes]

    # Count matches
    matched_gt = set()
    tp = 0
    for pn in pred_names:
        for i, gn in enumerate(gt_names):
            if i not in matched_gt and _fuzzy_match(pn, gn):
                tp += 1
                matched_gt.add(i)
                break

    precision = tp / len(pred_names) if pred_names else 0
    recall = tp / len(gt_names) if gt_names else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ingredient_recall(pred_dishes: list[dict], gt_dishes: list[dict]) -> float:
    """What fraction of ground truth ingredients were found?"""
    if not gt_dishes:
        return 1.0

    # Flatten all GT ingredients
    gt_ingredients = []
    for d in gt_dishes:
        for ing in d.get("ingredients", []):
            name = ing.get("name", ing) if isinstance(ing, dict) else str(ing)
            gt_ingredients.append(_normalize(name))

    if not gt_ingredients:
        return 1.0

    # Flatten all predicted ingredients
    pred_ingredients = []
    for d in pred_dishes:
        for ing in d.get("ingredients", []):
            name = ing.get("name", ing) if isinstance(ing, dict) else str(ing)
            pred_ingredients.append(_normalize(name))

    # Count how many GT ingredients have a fuzzy match in predictions
    found = 0
    used = set()
    for gt_name in gt_ingredients:
        for i, pred_name in enumerate(pred_ingredients):
            if i not in used and _fuzzy_match(gt_name, pred_name, threshold=0.55):
                found += 1
                used.add(i)
                break

    return found / len(gt_ingredients)


def ingredient_precision(pred_dishes: list[dict], gt_dishes: list[dict]) -> float:
    """What fraction of predicted ingredients are real (not hallucinated)?"""
    pred_ingredients = []
    for d in pred_dishes:
        for ing in d.get("ingredients", []):
            name = ing.get("name", ing) if isinstance(ing, dict) else str(ing)
            pred_ingredients.append(_normalize(name))

    if not pred_ingredients:
        return 1.0 if not gt_dishes else 0.0

    gt_ingredients = []
    for d in gt_dishes:
        for ing in d.get("ingredients", []):
            name = ing.get("name", ing) if isinstance(ing, dict) else str(ing)
            gt_ingredients.append(_normalize(name))

    if not gt_ingredients:
        return 0.0

    correct = 0
    used = set()
    for pred_name in pred_ingredients:
        for i, gt_name in enumerate(gt_ingredients):
            if i not in used and _fuzzy_match(pred_name, gt_name, threshold=0.55):
                correct += 1
                used.add(i)
                break

    return correct / len(pred_ingredients)


def weight_mae_score(pred_dishes: list[dict], gt_dishes: list[dict]) -> float:
    """
    Score based on mean absolute error of weight estimates.
    Returns 1.0 for perfect weights, 0.0 for >100% average error.
    Matches ingredients by fuzzy name across all dishes.
    """
    gt_weights = {}  # normalized_name → amount_g
    for d in gt_dishes:
        for ing in d.get("ingredients", []):
            if isinstance(ing, dict) and "amount_g" in ing:
                gt_weights[_normalize(ing["name"])] = _to_grams(ing["amount_g"])

    if not gt_weights:
        return 1.0  # No weights to compare

    pred_weights = {}
    for d in pred_dishes:
        for ing in d.get("ingredients", []):
            if isinstance(ing, dict) and "amount_g" in ing:
                pred_weights[_normalize(ing["name"])] = _to_grams(ing["amount_g"])

    if not pred_weights:
        return 0.0

    errors = []
    for gt_name, gt_g in gt_weights.items():
        best = _best_match(gt_name, list(pred_weights.keys()))
        if best is not None:
            pred_g = pred_weights[best]
            # Relative error capped at 1.0
            rel_error = min(abs(pred_g - gt_g) / max(gt_g, 1), 1.0)
            errors.append(rel_error)
        else:
            errors.append(1.0)  # Missing ingredient = 100% error

    avg_error = sum(errors) / len(errors)
    return max(0.0, 1.0 - avg_error)


# ---------------------------------------------------------------------------
# Reward shaping — non-linear penalty for hallucinated weights
# ---------------------------------------------------------------------------

def _weight_hallucination_penalty(pred_dishes: list[dict], gt_dishes: list[dict]) -> float:
    """
    Returns a multiplier in (0, 1] that penalizes wildly wrong weight estimates.

    If any matched ingredient's predicted weight is >3x or <0.2x the ground truth,
    the entire example gets penalized. This preserves the high-recall signal for
    examples where weights are reasonable, while tanking examples where the model
    invents absurd numbers (e.g., 500g butter in a stir-fry).

    Returns:
        1.0 = no penalty (all weights within 3x)
        0.3 = moderate penalty (one ingredient 3-5x off)
        0.1 = severe penalty (one ingredient >5x off)
    """
    gt_weights = {}
    for d in gt_dishes:
        for ing in d.get("ingredients", []):
            if isinstance(ing, dict) and "amount_g" in ing:
                gt_weights[_normalize(ing["name"])] = _to_grams(ing["amount_g"])

    if not gt_weights:
        return 1.0

    pred_weights = {}
    for d in pred_dishes:
        for ing in d.get("ingredients", []):
            if isinstance(ing, dict) and "amount_g" in ing:
                pred_weights[_normalize(ing["name"])] = _to_grams(ing["amount_g"])

    if not pred_weights:
        return 1.0  # No predicted weights = no hallucination to penalize

    worst_ratio = 1.0
    for gt_name, gt_g in gt_weights.items():
        if gt_g <= 0:
            continue
        best = _best_match(gt_name, list(pred_weights.keys()))
        if best is not None:
            pred_g = pred_weights[best]
            ratio = max(pred_g / gt_g, gt_g / pred_g)  # Symmetric ratio
            worst_ratio = max(worst_ratio, ratio)

    if worst_ratio > 5:
        return 0.1  # Severe: >5x off on any ingredient
    if worst_ratio > 3:
        return 0.3  # Moderate: 3-5x off
    return 1.0


# ---------------------------------------------------------------------------
# Composite metric for DSPy
# ---------------------------------------------------------------------------

def food_identification_metric(example, prediction, trace=None) -> float:
    """
    DSPy-compatible metric function.

    Args:
        example: dspy.Example with 'dishes' field (ground truth)
        prediction: dspy.Prediction with 'output' field (raw Gemini Nano text)
        trace: Optional trace object (unused)

    Returns:
        float in [0, 1] — weighted composite score
    """
    raw_output = prediction.get("output", "") or ""
    gt_dishes = example.get("dishes", [])

    # Parse prediction
    pred_dishes = parse_prediction(raw_output)

    # JSON parse failure → partial credit only if we got some text
    if pred_dishes is None:
        return 0.05 if raw_output and not raw_output.startswith("ERROR:") else 0.0

    # Component scores
    name_score = dish_name_f1(pred_dishes, gt_dishes)
    recall_score = ingredient_recall(pred_dishes, gt_dishes)
    precision_score = ingredient_precision(pred_dishes, gt_dishes)
    weight_score = weight_mae_score(pred_dishes, gt_dishes)
    parse_bonus = 0.1  # Successfully parsed JSON

    # F2-weighted ingredient score (recall 4x more important than precision)
    # F_beta = (1 + beta^2) * P * R / (beta^2 * P + R), beta=2
    if recall_score + precision_score > 0:
        beta_sq = 4  # beta=2
        ingredient_f2 = (1 + beta_sq) * precision_score * recall_score / (
            beta_sq * precision_score + recall_score
        )
    else:
        ingredient_f2 = 0.0

    # Reward shaping: non-linear penalty for hallucinated weights
    hallucination_penalty = _weight_hallucination_penalty(pred_dishes, gt_dishes)

    # Weighted composite — high-recall HITL strategy:
    # Users can delete extras but won't notice missing ingredients
    score = (
        name_score * 0.20
        + ingredient_f2 * 0.35
        + weight_score * 0.20
        + parse_bonus
        + recall_score * 0.15  # Extra recall bonus on top of F2
    ) * hallucination_penalty

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Detailed scoring (for analysis, not DSPy)
# ---------------------------------------------------------------------------

def score_detailed(raw_output: str, gt_dishes: list[dict]) -> dict:
    """Return a breakdown dict for manual analysis."""
    pred_dishes = parse_prediction(raw_output)
    if pred_dishes is None:
        return {
            "composite": 0.0,
            "json_parsed": False,
            "dish_name_f1": 0.0,
            "ingredient_recall": 0.0,
            "ingredient_precision": 0.0,
            "weight_mae_score": 0.0,
            "pred_dishes": None,
            "raw_output": raw_output[:500],
        }

    return {
        "composite": food_identification_metric(
            {"dishes": gt_dishes}, {"output": raw_output}
        ),
        "json_parsed": True,
        "dish_name_f1": dish_name_f1(pred_dishes, gt_dishes),
        "ingredient_recall": ingredient_recall(pred_dishes, gt_dishes),
        "ingredient_precision": ingredient_precision(pred_dishes, gt_dishes),
        "weight_mae_score": weight_mae_score(pred_dishes, gt_dishes),
        "weight_hallucination_penalty": _weight_hallucination_penalty(pred_dishes, gt_dishes),
        "pred_dishes": pred_dishes,
        "raw_output": raw_output[:500],
    }
