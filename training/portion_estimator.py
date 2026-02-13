#!/usr/bin/env python3
"""
Portion estimation module with smart fallback chain.

Estimates food weight (grams) from visual cues with a three-tier fallback:
  Level 1: Reference object geometry (HIGH confidence)
  Level 2: User history extrapolation (MEDIUM confidence)
  Level 3: USDA standard serving size (LOW confidence)

Per locked decision: Target +/-10% when reference objects present.
When confidence is LOW, suggests reference objects for next time.

Usage:
    from training.portion_estimator import PortionEstimator

    pe = PortionEstimator()
    estimate = pe.estimate(
        bounding_box=(100, 100, 400, 400),
        image_size=(640, 640),
        dish_name="fried rice",
        reference_objects=[],
    )
    print(estimate)
    # PortionEstimate(weight_g=250.0, confidence='low', method='usda_default', ...)
"""

import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Knowledge graph path
KG_DIR = Path(__file__).parent.parent / "knowledge-graph"
DEFAULT_KG_DB = KG_DIR / "food-knowledge.db"


# ---------------------------------------------------------------------------
# Food density table (g/cm^3)
# ---------------------------------------------------------------------------
# Source: USDA food composition data, various food science references.
# These are approximate densities for common food categories.
FOOD_DENSITY_TABLE = {
    # Grains & starches
    "rice": 0.90,
    "fried rice": 0.85,
    "noodle": 0.80,
    "pasta": 0.85,
    "bread": 0.35,
    "cereal": 0.30,
    "oatmeal": 0.95,
    "pancake": 0.70,
    "waffle": 0.55,
    "pizza": 0.65,
    "tortilla": 0.70,
    "couscous": 0.85,
    "quinoa": 0.88,

    # Proteins
    "meat": 1.05,
    "chicken": 1.04,
    "beef": 1.08,
    "pork": 1.06,
    "fish": 1.02,
    "shrimp": 1.00,
    "egg": 1.03,
    "tofu": 0.95,
    "tempeh": 0.98,

    # Vegetables
    "vegetable": 0.60,
    "salad": 0.25,
    "broccoli": 0.55,
    "carrot": 0.65,
    "potato": 0.75,
    "sweet potato": 0.72,
    "corn": 0.70,
    "beans": 0.80,
    "peas": 0.75,
    "mushroom": 0.45,
    "spinach": 0.30,
    "cabbage": 0.35,
    "lettuce": 0.20,
    "tomato": 0.65,
    "onion": 0.60,
    "pepper": 0.55,
    "eggplant": 0.50,
    "zucchini": 0.55,

    # Fruits
    "fruit": 0.70,
    "apple": 0.80,
    "banana": 0.75,
    "orange": 0.85,
    "berry": 0.60,
    "mango": 0.78,
    "watermelon": 0.60,
    "grape": 0.85,

    # Dairy
    "cheese": 0.95,
    "yogurt": 1.05,
    "milk": 1.03,
    "ice cream": 0.65,
    "cream": 0.98,
    "butter": 0.91,

    # Soups & liquids
    "soup": 1.00,
    "broth": 1.00,
    "curry": 0.95,
    "stew": 0.98,
    "sauce": 1.05,
    "gravy": 1.02,

    # Mixed dishes (higher density = more packed)
    "stir fry": 0.75,
    "casserole": 0.85,
    "burrito": 0.80,
    "sandwich": 0.55,
    "burger": 0.70,
    "wrap": 0.65,
    "sushi": 0.95,
    "dumpling": 0.90,
    "spring roll": 0.70,
    "pie": 0.75,

    # Snacks & desserts
    "cookie": 0.60,
    "cake": 0.50,
    "brownie": 0.65,
    "chips": 0.25,
    "nuts": 0.65,
    "popcorn": 0.10,
    "chocolate": 0.95,
    "candy": 0.80,

    # Default categories
    "liquid": 1.00,
    "solid": 0.75,
    "default": 0.70,
}

# Reference object dimensions (cm)
REFERENCE_OBJECTS = {
    "plate_dinner": {"type": "circle", "diameter_cm": 26.0},
    "plate_side": {"type": "circle", "diameter_cm": 20.0},
    "plate": {"type": "circle", "diameter_cm": 26.0},  # default plate
    "bowl": {"type": "circle", "diameter_cm": 16.0},
    "credit_card": {"type": "rectangle", "width_cm": 8.56, "height_cm": 5.4},
    "coin_quarter": {"type": "circle", "diameter_cm": 2.426},
    "coin_dollar": {"type": "circle", "diameter_cm": 2.67},
    "coin_50c_aud": {"type": "circle", "diameter_cm": 3.17},  # Australian 50c
    "hand": {"type": "rectangle", "width_cm": 8.5, "height_cm": 19.0},  # average adult
    "fork": {"type": "rectangle", "width_cm": 2.5, "height_cm": 19.0},
    "knife": {"type": "rectangle", "width_cm": 2.0, "height_cm": 23.0},
    "spoon": {"type": "rectangle", "width_cm": 4.0, "height_cm": 18.0},
    "chopstick": {"type": "rectangle", "width_cm": 0.8, "height_cm": 24.0},
    "can_330ml": {"type": "circle", "diameter_cm": 6.6},  # standard soda can
    "phone": {"type": "rectangle", "width_cm": 7.15, "height_cm": 14.67},  # avg smartphone
}

# Default assumed food depth (cm) as fraction of bounding box shorter side
# This is a rough heuristic; depth estimation would improve this significantly
DEFAULT_DEPTH_RATIO = 0.25  # 25% of shorter side as depth estimate

# Packing factor: how much of the bounding box is actually filled with food
# Most food doesn't fill the entire bbox (it's an approximation)
BBOX_FILL_FACTOR = 0.55  # food typically fills ~55% of the bounding box area


# ---------------------------------------------------------------------------
# PortionEstimate dataclass
# ---------------------------------------------------------------------------

@dataclass
class PortionEstimate:
    """Result of a portion estimation."""
    weight_g: float
    confidence: str  # "high", "medium", "low"
    method: str  # "geometry", "user_history", "usda_default"
    suggest_reference: bool  # True if confidence is low
    details: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"PortionEstimate(weight_g={self.weight_g:.1f}, confidence='{self.confidence}', "
            f"method='{self.method}', suggest_reference={self.suggest_reference})"
        )


# ---------------------------------------------------------------------------
# PortionEstimator class
# ---------------------------------------------------------------------------

class PortionEstimator:
    """
    Estimates food portion weight from visual cues using a three-tier fallback chain.

    Fallback chain:
      1. Reference object geometry (HIGH confidence)
      2. User history extrapolation (MEDIUM confidence)
      3. USDA standard serving / knowledge graph lookup (LOW confidence)
    """

    def __init__(self, kg_db_path: str = None):
        """
        Args:
            kg_db_path: Path to the food knowledge graph SQLite database.
                        Default: knowledge-graph/food-knowledge.db
        """
        self.kg_db_path = kg_db_path or str(DEFAULT_KG_DB)
        self._kg_available = os.path.exists(self.kg_db_path)
        if not self._kg_available:
            import logging
            logging.getLogger(__name__).warning(
                f"Knowledge graph not found at {self.kg_db_path}. "
                "USDA default fallback will use built-in serving sizes."
            )

    def estimate(
        self,
        bounding_box: tuple,
        image_size: tuple,
        dish_name: str,
        reference_objects: list = None,
        user_history: list = None,
        exif_data: dict = None,
    ) -> PortionEstimate:
        """
        Estimate portion weight using the smart fallback chain.

        Args:
            bounding_box: (x1, y1, x2, y2) in pixels
            image_size: (width, height) in pixels
            dish_name: Identified dish name (e.g., "fried rice")
            reference_objects: Detected reference objects, each a dict with
                             keys: type (str), bbox (tuple of x1,y1,x2,y2)
            user_history: Past servings for this dish, each a dict with
                         keys: dish (str), weight_g (float), timestamp (str, optional)
            exif_data: Camera EXIF metadata (focal_length, sensor_size, etc.)

        Returns:
            PortionEstimate with weight, confidence, method, and details.
        """
        reference_objects = reference_objects or []
        user_history = user_history or []

        # Level 1: Reference object geometry
        if reference_objects:
            result = self._estimate_from_geometry(
                bounding_box, image_size, dish_name, reference_objects, exif_data
            )
            if result is not None:
                return result

        # Level 2: User history extrapolation
        if user_history:
            result = self._estimate_from_history(
                bounding_box, image_size, dish_name, user_history
            )
            if result is not None:
                return result

        # Level 3: USDA standard serving / knowledge graph
        return self._estimate_from_usda_default(dish_name)

    def _estimate_from_geometry(
        self,
        bbox: tuple,
        image_size: tuple,
        dish_name: str,
        reference_objects: list,
        exif_data: dict = None,
    ) -> Optional[PortionEstimate]:
        """
        Level 1: Estimate portion using reference object for pixel-to-cm conversion.
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        food_px_w = x2 - x1
        food_px_h = y2 - y1

        if food_px_w <= 0 or food_px_h <= 0:
            return None

        # Find best reference object
        best_ref = None
        best_scale = None  # cm per pixel

        for ref_obj in reference_objects:
            ref_type = ref_obj.get("type", "").lower().replace(" ", "_")
            ref_bbox = ref_obj.get("bbox")

            if not ref_bbox or ref_type not in REFERENCE_OBJECTS:
                continue

            ref_info = REFERENCE_OBJECTS[ref_type]
            rx1, ry1, rx2, ry2 = ref_bbox
            ref_px_w = rx2 - rx1
            ref_px_h = ry2 - ry1

            if ref_px_w <= 0 or ref_px_h <= 0:
                continue

            # Calculate cm/pixel from reference object
            if ref_info["type"] == "circle":
                # Use diameter
                ref_px_diameter = max(ref_px_w, ref_px_h)
                scale = ref_info["diameter_cm"] / ref_px_diameter
            else:
                # Use average of width and height scales
                scale_w = ref_info["width_cm"] / ref_px_w
                scale_h = ref_info["height_cm"] / ref_px_h
                scale = (scale_w + scale_h) / 2

            if best_scale is None or ref_px_w * ref_px_h > (best_ref or {}).get("_area", 0):
                best_ref = {**ref_obj, "_area": ref_px_w * ref_px_h}
                best_scale = scale

        if best_scale is None:
            return None

        # Convert food bounding box to real dimensions
        food_cm_w = food_px_w * best_scale
        food_cm_h = food_px_h * best_scale

        # Estimate depth using absolute clamp rather than pure ratio
        # Food on plates is typically 1.5-5cm deep regardless of plate size
        shorter_side = min(food_cm_w, food_cm_h)
        food_cm_depth = shorter_side * DEFAULT_DEPTH_RATIO
        # Clamp depth to realistic range for plated food
        food_cm_depth = max(1.0, min(food_cm_depth, 5.0))

        # For flat foods (pizza, pancake), use reduced depth
        flat_foods = {"pizza", "pancake", "waffle", "tortilla", "bread", "cookie", "cracker"}
        dish_lower = dish_name.lower().replace("-", " ")
        is_flat = any(f in dish_lower for f in flat_foods)

        if is_flat:
            food_cm_depth = max(0.5, min(shorter_side * 0.08, 2.0))

        # Calculate volume using rectangular approximation with fill factor
        # This is more accurate than ellipsoid for plated food
        volume_cm3 = food_cm_w * food_cm_h * food_cm_depth * BBOX_FILL_FACTOR

        if is_flat:
            # Flat foods have a higher fill factor (they spread out evenly)
            volume_cm3 = food_cm_w * food_cm_h * food_cm_depth * 0.70

        # Look up food density
        density = self._get_food_density(dish_name)

        # Weight = volume * density
        weight_g = volume_cm3 * density

        # Sanity check: clamp to reasonable range
        weight_g = max(5.0, min(weight_g, 5000.0))

        return PortionEstimate(
            weight_g=round(weight_g, 1),
            confidence="high",
            method="geometry",
            suggest_reference=False,
            details={
                "food_cm_w": round(food_cm_w, 2),
                "food_cm_h": round(food_cm_h, 2),
                "food_cm_depth": round(food_cm_depth, 2),
                "volume_cm3": round(volume_cm3, 2),
                "density_g_per_cm3": density,
                "reference_type": best_ref.get("type", "unknown"),
                "scale_cm_per_px": round(best_scale, 6),
                "is_flat_food": is_flat,
            },
        )

    def _estimate_from_history(
        self,
        bbox: tuple,
        image_size: tuple,
        dish_name: str,
        user_history: list,
    ) -> Optional[PortionEstimate]:
        """
        Level 2: Estimate from user's previous portion sizes, weighted by recency.
        """
        # Filter history for matching dish
        dish_lower = dish_name.lower().replace("-", " ").replace("_", " ")
        matching = [
            h for h in user_history
            if h.get("dish", "").lower().replace("-", " ").replace("_", " ") == dish_lower
        ]

        if not matching:
            # Try fuzzy match: any history entry where dish name contains the current dish
            matching = [
                h for h in user_history
                if dish_lower in h.get("dish", "").lower().replace("-", " ").replace("_", " ")
                or h.get("dish", "").lower().replace("-", " ").replace("_", " ") in dish_lower
            ]

        if not matching:
            return None

        # Weight by recency (more recent = higher weight)
        # If timestamps available, use exponential decay; otherwise use position
        weights = []
        for i, entry in enumerate(matching):
            # Simple recency: later entries in list are more recent
            recency_weight = 1.0 + 0.5 * (i / max(len(matching) - 1, 1))
            weights.append(recency_weight)

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(
            entry["weight_g"] * w for entry, w in zip(matching, weights)
        )
        avg_weight = weighted_sum / total_weight

        # Also consider current bbox size relative to typical portion
        # (if the current bbox is 2x larger than average, scale up)
        # For now, use the weighted average directly
        # Future: incorporate bbox area ratio for relative sizing

        # Calculate standard deviation for confidence assessment
        values = [e["weight_g"] for e in matching]
        if len(values) > 1:
            std_dev = (sum((v - avg_weight) ** 2 for v in values) / len(values)) ** 0.5
            cv = std_dev / avg_weight if avg_weight > 0 else 1.0  # coefficient of variation
        else:
            cv = 0.3  # single data point, moderate uncertainty

        return PortionEstimate(
            weight_g=round(avg_weight, 1),
            confidence="medium",
            method="user_history",
            suggest_reference=False,
            details={
                "matching_entries": len(matching),
                "values_g": [round(e["weight_g"], 1) for e in matching],
                "coefficient_of_variation": round(cv, 3),
                "weighted_avg_g": round(avg_weight, 1),
            },
        )

    def _estimate_from_usda_default(self, dish_name: str) -> PortionEstimate:
        """
        Level 3: Fall back to USDA standard serving size.

        Strategy: Try built-in standard serving first (curated, reliable),
        then knowledge graph for specific dishes not in the built-in table.
        This avoids fuzzy matching in the KG returning wrong dishes
        (e.g., "chicken" -> "chicken schnitzel").
        """
        typical_amount_g = None
        kg_source = None

        # Try built-in standard serving first (more reliable for common foods)
        builtin_g, builtin_source = self._builtin_serving_size(dish_name)

        # If built-in had a specific match (not generic fallback), use it
        if "generic" not in builtin_source:
            typical_amount_g = builtin_g
            kg_source = builtin_source
        else:
            # Try knowledge graph for specific dish matches
            if self._kg_available:
                kg_g, kg_src = self._query_kg_serving(dish_name)
                if kg_g is not None:
                    # Sanity check: cap KG results to reasonable serving range
                    # (KG sums all ingredients which can include large liquid amounts)
                    kg_g = min(kg_g, 600.0)  # max reasonable single serving
                    typical_amount_g = kg_g
                    kg_source = kg_src

            # If neither had it, use built-in (generic) fallback
            if typical_amount_g is None:
                typical_amount_g = builtin_g
                kg_source = builtin_source

        return PortionEstimate(
            weight_g=round(typical_amount_g, 1),
            confidence="low",
            method="usda_default",
            suggest_reference=True,
            details={
                "source": kg_source,
                "dish_queried": dish_name,
                "note": "Based on standard serving size. For more accuracy, include a reference object (coin, credit card) next to your food.",
            },
        )

    def _query_kg_serving(self, dish_name: str) -> tuple:
        """
        Query knowledge graph for typical serving weight.
        Only uses EXACT dish name match to avoid false positives from fuzzy matching
        (e.g., "chicken" matching "chicken schnitzel" which has very different weight).

        Returns (weight_g, source) or (None, None).
        """
        try:
            conn = sqlite3.connect(self.kg_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            normalized = dish_name.lower().replace("-", " ").replace("_", " ")

            # Exact match only
            cursor.execute(
                """SELECT SUM(di.typical_amount_g) as total
                   FROM dishes d
                   JOIN dish_ingredients di ON di.dish_id = d.id
                   WHERE d.name = ?
                   AND di.is_nutrition_significant = 1""",
                (normalized,),
            )
            row = cursor.fetchone()
            conn.close()

            if row and row["total"] and row["total"] > 0:
                return row["total"], f"knowledge_graph ({dish_name})"

        except Exception:
            pass

        return None, None

    def _builtin_serving_size(self, dish_name: str) -> tuple:
        """
        Built-in standard serving sizes when knowledge graph is unavailable.
        Returns (weight_g, source).
        """
        # USDA standard serving sizes for common food categories
        STANDARD_SERVINGS = {
            # Rice dishes
            "rice": 200.0,
            "fried rice": 250.0,
            "risotto": 250.0,
            "biryani": 300.0,

            # Noodle dishes
            "noodle": 250.0,
            "pasta": 220.0,
            "ramen": 400.0,
            "pho": 450.0,
            "udon": 350.0,
            "pad thai": 300.0,
            "chow mein": 280.0,

            # Protein dishes
            "steak": 200.0,
            "chicken": 150.0,
            "fish": 150.0,
            "pork": 150.0,
            "tofu": 150.0,
            "shrimp": 120.0,

            # Soups
            "soup": 350.0,
            "stew": 350.0,
            "curry": 300.0,
            "chili": 300.0,

            # Sandwiches & wraps
            "sandwich": 200.0,
            "burger": 250.0,
            "burrito": 300.0,
            "wrap": 250.0,
            "banh mi": 300.0,

            # Salads & vegetables
            "salad": 150.0,
            "coleslaw": 100.0,

            # Pizza & flatbreads
            "pizza": 200.0,  # ~2 slices

            # Asian dishes
            "sushi": 250.0,
            "dim sum": 200.0,
            "dumpling": 180.0,
            "spring roll": 120.0,
            "bibimbap": 400.0,
            "kimchi": 80.0,
            "tempura": 150.0,
            "tonkatsu": 200.0,
            "gyoza": 150.0,
            "bulgogi": 200.0,
            "tteokbokki": 250.0,

            # Breakfast
            "omelette": 180.0,
            "pancake": 150.0,
            "waffle": 120.0,
            "cereal": 60.0,
            "oatmeal": 250.0,

            # Desserts
            "cake": 120.0,
            "ice cream": 150.0,
            "cookie": 40.0,
            "brownie": 60.0,
            "mochi": 80.0,

            # Drinks (in case food detection captures them)
            "smoothie": 350.0,
            "juice": 250.0,
        }

        dish_lower = dish_name.lower().replace("-", " ").replace("_", " ")

        # Exact match
        if dish_lower in STANDARD_SERVINGS:
            return STANDARD_SERVINGS[dish_lower], f"builtin_standard ({dish_lower})"

        # Partial match
        for key, weight in STANDARD_SERVINGS.items():
            if key in dish_lower or dish_lower in key:
                return weight, f"builtin_standard (partial: {key})"

        # Category-based fallback
        category_defaults = {
            "soup": 350.0,
            "curry": 300.0,
            "rice": 250.0,
            "noodle": 250.0,
            "meat": 150.0,
            "fish": 150.0,
            "salad": 150.0,
            "dessert": 120.0,
        }

        for cat, weight in category_defaults.items():
            if cat in dish_lower:
                return weight, f"builtin_category ({cat})"

        # Ultimate fallback: generic portion
        return 250.0, "builtin_generic (250g default)"

    def _get_food_density(self, dish_name: str) -> float:
        """
        Look up food density from the density table.
        Returns density in g/cm^3.
        """
        dish_lower = dish_name.lower().replace("-", " ").replace("_", " ")

        # Exact match
        if dish_lower in FOOD_DENSITY_TABLE:
            return FOOD_DENSITY_TABLE[dish_lower]

        # Partial match
        for key, density in FOOD_DENSITY_TABLE.items():
            if key in dish_lower or dish_lower in key:
                return density

        # Default
        return FOOD_DENSITY_TABLE["default"]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def estimate_portion(
    bounding_box: tuple,
    image_size: tuple,
    dish_name: str,
    reference_objects: list = None,
    user_history: list = None,
    kg_db_path: str = None,
) -> PortionEstimate:
    """
    Convenience function for quick portion estimation.

    See PortionEstimator.estimate() for full documentation.
    """
    estimator = PortionEstimator(kg_db_path=kg_db_path)
    return estimator.estimate(
        bounding_box=bounding_box,
        image_size=image_size,
        dish_name=dish_name,
        reference_objects=reference_objects,
        user_history=user_history,
    )


if __name__ == "__main__":
    # Demo usage
    pe = PortionEstimator()

    print("=== Portion Estimation Demo ===\n")

    # Test Level 3: USDA default (no reference, no history)
    print("--- Level 3: USDA Default ---")
    est = pe.estimate(
        bounding_box=(100, 100, 400, 400),
        image_size=(640, 640),
        dish_name="fried rice",
        reference_objects=[],
    )
    print(f"  Fried rice: {est}")
    print(f"  Details: {est.details}")

    # Test Level 1: With reference object (plate)
    print("\n--- Level 1: Reference Object (plate) ---")
    est = pe.estimate(
        bounding_box=(150, 150, 450, 400),
        image_size=(640, 640),
        dish_name="fried rice",
        reference_objects=[{"type": "plate", "bbox": (50, 50, 590, 590)}],
    )
    print(f"  Fried rice with plate ref: {est}")
    print(f"  Details: {est.details}")

    # Test Level 2: With user history
    print("\n--- Level 2: User History ---")
    est = pe.estimate(
        bounding_box=(100, 100, 400, 400),
        image_size=(640, 640),
        dish_name="fried rice",
        reference_objects=[],
        user_history=[
            {"dish": "fried rice", "weight_g": 280},
            {"dish": "fried rice", "weight_g": 320},
            {"dish": "fried rice", "weight_g": 300},
        ],
    )
    print(f"  Fried rice with history: {est}")
    print(f"  Details: {est.details}")

    # Test various dishes with USDA default
    print("\n--- Various Dishes (USDA Default) ---")
    for dish in ["sushi", "pad-thai", "burger", "ramen", "salad", "bibimbap", "unknown-dish"]:
        est = pe.estimate(
            bounding_box=(100, 100, 400, 400),
            image_size=(640, 640),
            dish_name=dish,
            reference_objects=[],
        )
        print(f"  {dish:20s}: {est.weight_g:>6.1f}g  conf={est.confidence:<6s}  method={est.method}  suggest_ref={est.suggest_reference}")
