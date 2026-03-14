#!/usr/bin/env python3
"""
Build the food knowledge graph database.

Orchestrates:
1. Apply hierarchical schema (8 tables + FTS5 + SymSpell)
2. Seed curated dishes from generate_dishes.py
3. Seed classifier/detector labels as KG dish entries
4. Seed from recipe datasets (volume)
5. Populate USDA SR Legacy nutrition data (with micronutrients) and link ingredients
6. Seed WorldCuisines multilingual aliases
7. Compute SymSpell delete variants for fuzzy matching
8. Compute per-dish average nutrition from recipe ingredients
9. Rebuild FTS5 indexes and print summary statistics

Usage:
    python build_kg.py [--db food-knowledge.db] [--skip-recipenlg]
"""

import argparse
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

KG_DIR = Path(__file__).parent

# Add parent to path for imports
sys.path.insert(0, str(KG_DIR))


def apply_schema(conn: sqlite3.Connection):
    """Apply the hierarchical KG schema."""
    schema_path = KG_DIR / "schema.sql"
    with open(schema_path) as f:
        conn.executescript(f.read())
    print("  Schema applied (8 tables + 2 FTS5 + triggers + SymSpell)")


def seed_generated_dishes(conn: sqlite3.Connection) -> dict:
    """Seed from generate_dishes.py curated baseline."""
    from generate_dishes import get_all_dishes

    dishes = get_all_dishes()
    cursor = conn.cursor()

    # Ensure cuisines exist, cache IDs
    cuisine_ids = {}
    category_ids = {}
    dish_count = 0
    recipe_count = 0
    ingredient_count = 0

    for dish_name, dish_data in dishes.items():
        cuisine_name = dish_data["cuisine"]

        # Ensure cuisine row
        if cuisine_name not in cuisine_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO cuisine (name) VALUES (?)", (cuisine_name,)
            )
            cursor.execute("SELECT id FROM cuisine WHERE name = ?", (cuisine_name,))
            cuisine_ids[cuisine_name] = cursor.fetchone()[0]

        cuisine_id = cuisine_ids[cuisine_name]

        # Ensure a default category per cuisine
        cat_key = f"{cuisine_name}_general"
        if cat_key not in category_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO dish_category (cuisine_id, name) VALUES (?, ?)",
                (cuisine_id, "General"),
            )
            cursor.execute(
                "SELECT id FROM dish_category WHERE cuisine_id = ? AND name = 'General'",
                (cuisine_id,),
            )
            category_ids[cat_key] = cursor.fetchone()[0]

        category_id = category_ids[cat_key]

        # Normalize dish name
        canonical = dish_name.lower().strip().replace("-", " ").replace("_", " ")

        # Insert dish
        cursor.execute(
            "INSERT OR IGNORE INTO dish (category_id, canonical_name, source, confidence) VALUES (?, ?, 'generated', 0.85)",
            (category_id, canonical),
        )
        if cursor.rowcount == 0:
            continue
        dish_id = cursor.lastrowid
        dish_count += 1

        # Create a canonical recipe
        ingredients = dish_data["ingredients"]
        total_weight = sum(v[1] for v in ingredients.values())

        cursor.execute(
            "INSERT INTO recipe (dish_id, name, source, total_weight_grams, servings, is_canonical) VALUES (?, ?, 'generated', ?, 1, 1)",
            (dish_id, canonical, round(total_weight, 1)),
        )
        recipe_id = cursor.lastrowid
        recipe_count += 1

        # Insert recipe ingredients
        for idx, (ing_name, (weight_pct, typical_g)) in enumerate(ingredients.items()):
            cursor.execute(
                "INSERT INTO recipe_ingredient (recipe_id, ingredient_name, quantity_grams, sort_order) VALUES (?, ?, ?, ?)",
                (recipe_id, ing_name.lower(), round(typical_g, 1), idx),
            )
            ingredient_count += 1

    conn.commit()
    print(f"  Generated dishes: {dish_count} dishes, {recipe_count} recipes, {ingredient_count} ingredients")
    return {"dishes": dish_count, "recipes": recipe_count, "ingredients": ingredient_count}


# Pre-compiled regex for ingredient cleaning (used in hot loop)
_AMOUNT_RE = re.compile(r'^[\d./\s-]+')
_UNIT_RE = re.compile(
    r'^(cups?|c\.|tablespoons?|tbsp\.?|tbs\.?|teaspoons?|tsp\.?|'
    r'ounces?|oz\.?|pounds?|lbs?\.?|grams?|g\.|kg\.?|ml\.?|'
    r'cans?|jars?|pkg\.?|packages?|boxes?|bottles?|bags?|'
    r'large|medium|small|whole|cloves?|slices?|pieces?|'
    r'pinch|pinches|dash|dashes|sticks?|heads?|bunche?s?|'
    r'sprigs?|stalks?|quarts?|qt\.?|pints?|pt\.?|gallons?|gal\.?|'
    r'liters?|of)\s+',
    re.IGNORECASE
)
_MODIFIER_RE = re.compile(
    r'\b(chopped|diced|minced|sliced|grated|shredded|crushed|'
    r'fresh|dried|frozen|canned|cooked|raw|melted|softened|'
    r'finely|thinly|thickly|roughly|peeled|seeded|deveined|'
    r'boneless|skinless|lean|extra|all-purpose|self-rising|'
    r'granulated|powdered|confectioners|'
    r'divided|optional|or more|to taste|for garnish|if desired)\b',
    re.IGNORECASE
)


def _clean_ingredient(ing_str: str) -> str:
    """Clean a recipe ingredient string to extract the core ingredient name."""
    s = ing_str.strip()
    if not s:
        return ""

    # Strip leading amounts
    s = _AMOUNT_RE.sub('', s).strip()

    # Strip units (may need multiple passes for "c. c." patterns)
    for _ in range(3):
        prev = s
        s = _UNIT_RE.sub('', s).strip()
        if s == prev:
            break

    # Remove parenthetical notes
    s = re.sub(r'\s*\(.*?\)', '', s)
    # Remove everything after comma (usually notes)
    s = re.sub(r',.*$', '', s)
    # Remove modifiers
    s = _MODIFIER_RE.sub('', s)
    # Clean up whitespace
    s = re.sub(r'\s+', ' ', s).strip().lower()
    # Remove leading "of " (from patterns like "1 cup of flour")
    s = re.sub(r'^of\s+', '', s)

    return s


def seed_recipes(conn: sqlite3.Connection) -> dict:
    """
    Seed from recipe datasets (volume data).

    Uses corbt/all-recipes from HuggingFace (2M+ recipes in Parquet format,
    no manual download required). Falls back to RecipeNLG if available.
    """
    from seed_recipenlg import (
        normalize_dish_name,
        classify_cuisine,
    )
    from parse_quantities import parse_quantity_grams

    try:
        from tqdm import tqdm as tqdm_fn
    except ImportError:
        def tqdm_fn(it, **kw):
            return it

    print("  Loading recipe dataset...")
    recipes_iter = None

    # Strategy 1: corbt/all-recipes (Parquet, no auth needed) -- all 4 files
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd

        parquet_files = [
            "data/train-00000-of-00004-237b1b1141fdcfa1.parquet",
            "data/train-00001-of-00004-d46654ac93566129.parquet",
            "data/train-00002-of-00004-3b4f78b99eedadc2.parquet",
            "data/train-00003-of-00004-2369b90eb0860a76.parquet",
        ]

        def _parse_all_recipes_text(text):
            """Parse the all-recipes format: Title\n\nIngredients:\n- ...\n\nDirections:..."""
            lines = text.strip().split("\n")
            title = lines[0].strip() if lines else ""
            ingredients = []
            in_ingredients = False
            for line in lines[1:]:
                line = line.strip()
                if line.lower().startswith("ingredients"):
                    in_ingredients = True
                    continue
                if line.lower().startswith("directions") or line.lower().startswith("instructions"):
                    break
                if in_ingredients and line.startswith("- "):
                    ingredients.append(line[2:])
                elif in_ingredients and line.startswith("-"):
                    ingredients.append(line[1:].strip())
            return title, ingredients

        recipes_iter = []
        total_loaded = 0
        for pf_idx, pf in enumerate(parquet_files):
            path = hf_hub_download("corbt/all-recipes", pf, repo_type="dataset")
            df = pd.read_parquet(path)
            total_loaded += len(df)
            print(f"  Loaded parquet {pf_idx + 1}/4: {len(df):,} recipes")
            for text in tqdm_fn(df["input"].values, desc=f"  Parsing file {pf_idx + 1}"):
                title, ings = _parse_all_recipes_text(text)
                if title and ings:
                    recipes_iter.append({"title": title, "ingredients": ings})
            del df  # Free memory after each file
        print(f"  Total: {total_loaded:,} recipes loaded from all 4 parquet files")
    except Exception as e:
        print(f"  Warning: Could not load all-recipes: {e}")

    # Strategy 2: Try RecipeNLG (needs manual download)
    if not recipes_iter:
        try:
            from datasets import load_dataset
            ds = load_dataset("mbien/recipe_nlg", split="train", trust_remote_code=True)
            recipes_iter = [{"title": r.get("title", ""), "ingredients": r.get("ingredients", [])} for r in ds]
            print(f"  Loaded {len(recipes_iter):,} recipes from RecipeNLG")
        except Exception as e:
            print(f"  Warning: Could not load RecipeNLG: {e}")
            print(f"  Skipping recipe seeding (curated dishes only)")
            return {"dishes": 0, "recipes": 0}

    cursor = conn.cursor()

    # Phase 1: Aggregate recipes per dish name
    print("  Phase 1: Aggregating recipes by dish name...")
    dish_data = defaultdict(lambda: {"ingredients": defaultdict(list), "count": 0})

    parsed_count = 0
    fallback_count = 0

    for r_idx, recipe in enumerate(tqdm_fn(recipes_iter, desc="  Scanning recipes")):
        title = recipe.get("title", "")
        if not title:
            continue

        dish_name = normalize_dish_name(title)
        if not dish_name or len(dish_name) < 3:
            continue

        raw_ingredients = recipe.get("ingredients", [])
        for ing_str in raw_ingredients:
            ing_clean = _clean_ingredient(ing_str)
            if ing_clean and len(ing_clean) >= 2 and len(ing_clean) <= 60:
                # Parse real quantity from raw string (before cleaning strips amounts)
                parsed_g = parse_quantity_grams(ing_str)
                if parsed_g is not None:
                    qty_g = parsed_g
                    parsed_count += 1
                else:
                    qty_g = 50.0  # Fallback only when parser fails
                    fallback_count += 1
                dish_data[dish_name]["ingredients"][ing_clean].append(qty_g)

        dish_data[dish_name]["count"] += 1

        # Progress logging every 100K recipes
        if (r_idx + 1) % 100000 == 0:
            print(f"  Progress: {r_idx + 1:,} recipes processed...")

    total_ings = parsed_count + fallback_count
    parse_pct = (parsed_count / total_ings * 100) if total_ings > 0 else 0
    print(f"  Found {len(dish_data):,} unique dish names")
    print(f"  Quantity parsing: {parsed_count:,} parsed ({parse_pct:.1f}%), {fallback_count:,} fallback to 50g")

    # Phase 2: Include ALL dishes (no cap, no minimum threshold)
    # Sort by popularity for deterministic insertion order
    sorted_dishes = sorted(dish_data.items(), key=lambda x: -x[1]["count"])
    high_conf = sum(1 for _, v in sorted_dishes if v["count"] >= 3)
    low_conf = len(sorted_dishes) - high_conf
    print(f"  Including all {len(sorted_dishes):,} dishes ({high_conf:,} with >= 3 recipes, {low_conf:,} long-tail)")

    # Get existing cuisine/category caches
    cuisine_ids = {}
    cursor.execute("SELECT id, name FROM cuisine")
    for row in cursor.fetchall():
        cuisine_ids[row[1]] = row[0]

    category_ids = {}
    cursor.execute("SELECT id, cuisine_id, name FROM dish_category")
    for row in cursor.fetchall():
        for cname, cid in cuisine_ids.items():
            if row[1] == cid:
                category_ids[f"{cname}_general"] = row[0]

    dishes_inserted = 0
    recipes_inserted = 0

    for dish_name, data in tqdm_fn(sorted_dishes, desc="  Inserting recipe dishes"):
        cuisine = classify_cuisine(dish_name)

        if cuisine not in cuisine_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO cuisine (name) VALUES (?)", (cuisine,)
            )
            cursor.execute("SELECT id FROM cuisine WHERE name = ?", (cuisine,))
            cuisine_ids[cuisine] = cursor.fetchone()[0]

        cuisine_id = cuisine_ids[cuisine]
        cat_key = f"{cuisine}_general"
        if cat_key not in category_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO dish_category (cuisine_id, name) VALUES (?, ?)",
                (cuisine_id, "General"),
            )
            cursor.execute(
                "SELECT id FROM dish_category WHERE cuisine_id = ? AND name = 'General'",
                (cuisine_id,),
            )
            category_ids[cat_key] = cursor.fetchone()[0]

        category_id = category_ids[cat_key]

        # Normalize: lowercase, spaces only
        canonical = dish_name.lower().strip().replace("-", " ").replace("_", " ")

        # Confidence based on recipe count:
        # >= 3 recipes: 0.3-0.9 scaling (high confidence)
        # 1-2 recipes: 0.2-0.4 (lower confidence, long-tail)
        count = data["count"]
        if count >= 3:
            confidence = min(0.9, 0.3 + (count / 100.0) * 0.6)
        else:
            confidence = 0.2 + (count * 0.1)

        cursor.execute(
            "INSERT OR IGNORE INTO dish (category_id, canonical_name, source, confidence) VALUES (?, ?, 'recipenlg', ?)",
            (category_id, canonical, confidence),
        )
        if cursor.rowcount == 0:
            continue

        dish_id = cursor.lastrowid
        dishes_inserted += 1

        # Create canonical recipe
        total_weight = sum(
            sum(amounts) / len(amounts)
            for amounts in data["ingredients"].values()
        ) or 1.0

        cursor.execute(
            "INSERT INTO recipe (dish_id, name, source, total_weight_grams, servings, is_canonical) VALUES (?, ?, 'recipenlg', ?, 1, 1)",
            (dish_id, canonical, round(total_weight, 1)),
        )
        recipe_id = cursor.lastrowid
        recipes_inserted += 1

        for idx, (ing_name, amounts) in enumerate(data["ingredients"].items()):
            avg_g = sum(amounts) / len(amounts)
            cursor.execute(
                "INSERT INTO recipe_ingredient (recipe_id, ingredient_name, quantity_grams, sort_order) VALUES (?, ?, ?, ?)",
                (recipe_id, ing_name.lower(), round(avg_g, 1), idx),
            )

        if dishes_inserted % 2000 == 0:
            conn.commit()

    conn.commit()
    print(f"  Recipes: {dishes_inserted} new dishes, {recipes_inserted} recipes")
    return {"dishes": dishes_inserted, "recipes": recipes_inserted}


def seed_classifier_labels(conn: sqlite3.Connection) -> dict:
    """
    Seed KG dish entries for all classifier and detector labels.

    Ensures every food the model can detect/classify has a corresponding
    KG entry, so searchDish() never returns null for recognized foods.

    Loads labels from:
    - apps/mobile/assets/models/labels_classify.json (905 labels, 664 named)
    - apps/mobile/assets/models/labels_detect.json (241 labels)

    Skips numeric CNFOOD-241 labels (e.g., "000", "001", etc.).
    """
    import json

    from seed_recipenlg import classify_cuisine

    cursor = conn.cursor()

    # Resolve paths relative to project root (KG_DIR is knowledge-graph/)
    project_root = KG_DIR.parent
    classify_path = project_root / "apps" / "mobile" / "assets" / "models" / "labels_classify.json"
    detect_path = project_root / "apps" / "mobile" / "assets" / "models" / "labels_detect.json"

    all_labels = set()

    # Load classifier labels
    if classify_path.exists():
        with open(classify_path) as f:
            classify_data = json.load(f)
        for label in classify_data.get("labels", []):
            # Skip numeric CNFOOD-241 labels (3-digit strings like "000", "001")
            if label.isdigit() or (len(label) == 3 and all(c.isdigit() for c in label)):
                continue
            all_labels.add(label)
        print(f"  Classifier labels: {len(classify_data.get('labels', []))} total, "
              f"{len(all_labels)} named (skipped numeric)")
    else:
        print(f"  Warning: {classify_path} not found")

    # Load detector labels
    detect_named = set()
    if detect_path.exists():
        with open(detect_path) as f:
            detect_data = json.load(f)
        for label in detect_data.get("classNames", []):
            detect_named.add(label)
            all_labels.add(label)
        print(f"  Detector labels: {len(detect_data.get('classNames', []))} total")
    else:
        print(f"  Warning: {detect_path} not found")

    # Get existing cuisine/category caches
    cuisine_ids = {}
    cursor.execute("SELECT id, name FROM cuisine")
    for row in cursor.fetchall():
        cuisine_ids[row[1]] = row[0]

    category_ids = {}
    cursor.execute("SELECT id, cuisine_id, name FROM dish_category")
    for row in cursor.fetchall():
        for cname, cid in cuisine_ids.items():
            if row[1] == cid:
                category_ids[f"{cname}_general"] = row[0]

    new_dishes = 0
    new_aliases = 0

    for label in sorted(all_labels):
        # Normalize: replace underscores/hyphens with spaces, lowercase
        canonical = label.lower().strip().replace("_", " ").replace("-", " ")
        if not canonical or len(canonical) < 2:
            continue

        # Check if dish already exists
        cursor.execute("SELECT id FROM dish WHERE canonical_name = ?", (canonical,))
        existing = cursor.fetchone()

        if existing:
            dish_id = existing[0]
        else:
            # Classify cuisine for this label
            cuisine = classify_cuisine(canonical)

            if cuisine not in cuisine_ids:
                cursor.execute(
                    "INSERT OR IGNORE INTO cuisine (name) VALUES (?)", (cuisine,)
                )
                cursor.execute("SELECT id FROM cuisine WHERE name = ?", (cuisine,))
                cuisine_ids[cuisine] = cursor.fetchone()[0]

            cuisine_id = cuisine_ids[cuisine]
            cat_key = f"{cuisine}_general"
            if cat_key not in category_ids:
                cursor.execute(
                    "INSERT OR IGNORE INTO dish_category (cuisine_id, name) VALUES (?, ?)",
                    (cuisine_id, "General"),
                )
                cursor.execute(
                    "SELECT id FROM dish_category WHERE cuisine_id = ? AND name = 'General'",
                    (cuisine_id,),
                )
                category_ids[cat_key] = cursor.fetchone()[0]

            category_id = category_ids[cat_key]

            cursor.execute(
                "INSERT OR IGNORE INTO dish (category_id, canonical_name, source, confidence) "
                "VALUES (?, ?, 'classifier_label', 0.6)",
                (category_id, canonical),
            )
            if cursor.rowcount > 0:
                dish_id = cursor.lastrowid
                new_dishes += 1
            else:
                # Race condition or duplicate -- fetch existing
                cursor.execute("SELECT id FROM dish WHERE canonical_name = ?", (canonical,))
                row = cursor.fetchone()
                dish_id = row[0] if row else None

        if not dish_id:
            continue

        # Register the original label string as a dish_alias so that
        # formatFoodLabel() output (with underscores) can still match
        if label != canonical:
            cursor.execute(
                "SELECT id FROM dish_alias WHERE dish_id = ? AND alias = ?",
                (dish_id, label),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO dish_alias (dish_id, alias, language, alias_type) "
                    "VALUES (?, ?, 'en', 'model_label')",
                    (dish_id, label),
                )
                new_aliases += 1

    conn.commit()
    print(f"  Classifier labels seeded: {new_dishes} new dishes from {len(all_labels)} labels, "
          f"{new_aliases} aliases added")
    return {"new_dishes": new_dishes, "total_labels": len(all_labels), "aliases": new_aliases}


def seed_usda_sr_legacy(conn: sqlite3.Connection) -> dict:
    """
    Populate usda_food table with USDA SR Legacy data and link
    recipe_ingredient rows by fuzzy-matching ingredient names.

    Priority:
    1. Full USDA SR Legacy CSV (knowledge-graph/data/sr_legacy_full.csv) -- ~7,793 foods
    2. Auto-download via download_usda_sr.py if CSV missing
    3. Fall back to embedded 500-food subset as last resort
    """
    cursor = conn.cursor()

    sr_full_csv = KG_DIR / "data" / "sr_legacy_full.csv"
    sr_legacy_csv = KG_DIR / "data" / "sr_legacy_food.csv"

    loaded = False

    # Strategy 1: Full USDA SR Legacy with micronutrients
    if sr_full_csv.exists():
        loaded = _load_usda_from_csv(cursor, sr_full_csv, include_micros=True)

    # Strategy 2: Auto-download if full CSV doesn't exist
    if not loaded:
        print("  Full USDA SR Legacy CSV not found, attempting download...")
        try:
            from download_usda_sr import main as download_main
            download_main()
            if sr_full_csv.exists():
                loaded = _load_usda_from_csv(cursor, sr_full_csv, include_micros=True)
        except Exception as e:
            print(f"  Download failed: {e}")

    # Strategy 3: Try older sr_legacy_food.csv (without micronutrients)
    if not loaded and sr_legacy_csv.exists():
        loaded = _load_usda_from_csv(cursor, sr_legacy_csv, include_micros=False)

    # Strategy 4: Fall back to embedded data (last resort)
    if not loaded:
        print("  Falling back to embedded USDA subset (~500 foods)...")
        _seed_usda_embedded(cursor)

    conn.commit()

    # Count USDA foods loaded
    cursor.execute("SELECT COUNT(*) FROM usda_food")
    usda_count = cursor.fetchone()[0]
    print(f"  USDA foods loaded: {usda_count}")

    # Link recipe_ingredient.usda_fdc_id by matching ingredient names
    linked = _link_ingredients_to_usda(conn)
    print(f"  Ingredients linked to USDA: {linked}")

    return {"usda_foods": usda_count, "linked": linked}


def _load_usda_from_csv(cursor, csv_path: Path, include_micros: bool = False) -> bool:
    """Load USDA foods from a CSV file into the usda_food table.

    Returns True if loading succeeded, False otherwise.
    """
    import csv as csv_mod

    print(f"  Loading USDA SR Legacy from {csv_path}...")
    count = 0
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                try:
                    params = (
                        int(row["fdc_id"]),
                        row["description"],
                        row.get("food_group", ""),
                        float(row["calories"]) if row.get("calories") else None,
                        float(row["protein"]) if row.get("protein") else None,
                        float(row["fat"]) if row.get("fat") else None,
                        float(row["carbs"]) if row.get("carbs") else None,
                        float(row.get("fiber", 0)) if row.get("fiber") else None,
                    )

                    if include_micros:
                        micro_params = (
                            float(row["vitamin_a_ug"]) if row.get("vitamin_a_ug") else None,
                            float(row["vitamin_c_mg"]) if row.get("vitamin_c_mg") else None,
                            float(row["vitamin_d_ug"]) if row.get("vitamin_d_ug") else None,
                            float(row["calcium_mg"]) if row.get("calcium_mg") else None,
                            float(row["iron_mg"]) if row.get("iron_mg") else None,
                            float(row["potassium_mg"]) if row.get("potassium_mg") else None,
                            float(row["sodium_mg"]) if row.get("sodium_mg") else None,
                            float(row["zinc_mg"]) if row.get("zinc_mg") else None,
                            float(row["magnesium_mg"]) if row.get("magnesium_mg") else None,
                        )
                        cursor.execute(
                            "INSERT OR IGNORE INTO usda_food (fdc_id, description, food_group, "
                            "calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g, fiber_per_100g, "
                            "vitamin_a_ug, vitamin_c_mg, vitamin_d_ug, calcium_mg, iron_mg, "
                            "potassium_mg, sodium_mg, zinc_mg, magnesium_mg) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            params + micro_params,
                        )
                    else:
                        cursor.execute(
                            "INSERT OR IGNORE INTO usda_food (fdc_id, description, food_group, "
                            "calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g, fiber_per_100g) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            params,
                        )
                    count += 1
                except (ValueError, KeyError):
                    continue

        print(f"  Loaded {count} USDA foods from CSV")
        return count > 0
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return False


def _seed_usda_embedded(cursor):
    """
    Seed usda_food with a curated set of common recipe ingredients
    from USDA SR Legacy. This embeds nutrition data directly so
    the KG is fully self-contained.
    """
    # Curated USDA SR Legacy data -- top ~800 recipe ingredients
    # Format: (fdc_id, description, food_group, cal, protein, fat, carbs, fiber)
    USDA_FOODS = [
        # Proteins
        (171077, "Chicken, broilers or fryers, breast, meat only, cooked, roasted", "Poultry Products", 165, 31.0, 3.6, 0, 0),
        (171057, "Chicken, broilers or fryers, thigh, meat only, cooked, roasted", "Poultry Products", 209, 26.0, 10.9, 0, 0),
        (171116, "Turkey, all classes, breast, meat only, cooked, roasted", "Poultry Products", 135, 30.1, 0.7, 0, 0),
        (174756, "Beef, ground, 85% lean meat / 15% fat, cooked, pan-browned", "Beef Products", 250, 25.6, 15.4, 0, 0),
        (175167, "Beef, top sirloin, steak, separable lean only, cooked, grilled", "Beef Products", 183, 30.2, 6.4, 0, 0),
        (167820, "Pork, fresh, loin, whole, separable lean only, cooked, roasted", "Pork Products", 143, 27.3, 3.0, 0, 0),
        (174861, "Pork, fresh, belly, raw", "Pork Products", 518, 9.3, 53.0, 0, 0),
        (175177, "Lamb, domestic, leg, whole, separable lean only, cooked, roasted", "Lamb, Veal, and Game Products", 191, 28.3, 7.7, 0, 0),
        (175159, "Duck, domesticated, meat only, cooked, roasted", "Poultry Products", 201, 23.5, 11.2, 0, 0),
        (175139, "Fish, salmon, Atlantic, wild, cooked, dry heat", "Finfish and Shellfish Products", 182, 25.4, 8.1, 0, 0),
        (171962, "Fish, tuna, light, canned in water, drained solids", "Finfish and Shellfish Products", 116, 25.5, 0.8, 0, 0),
        (175180, "Fish, cod, Atlantic, cooked, dry heat", "Finfish and Shellfish Products", 105, 22.8, 0.9, 0, 0),
        (175168, "Fish, tilapia, cooked, dry heat", "Finfish and Shellfish Products", 128, 26.1, 2.7, 0, 0),
        (175173, "Crustaceans, shrimp, mixed species, cooked, moist heat", "Finfish and Shellfish Products", 99, 20.9, 1.1, 0.2, 0),
        (171974, "Crustaceans, crab, blue, cooked, moist heat", "Finfish and Shellfish Products", 102, 20.2, 1.8, 0, 0),
        (171984, "Mollusks, squid, mixed species, cooked, fried", "Finfish and Shellfish Products", 175, 18.0, 7.5, 7.8, 0),
        (171990, "Mollusks, scallop, mixed species, cooked, steamed", "Finfish and Shellfish Products", 111, 20.5, 0.8, 5.4, 0),
        (175083, "Mollusks, mussel, blue, cooked, moist heat", "Finfish and Shellfish Products", 172, 23.8, 4.5, 7.4, 0),
        (171991, "Mollusks, clam, mixed species, cooked, moist heat", "Finfish and Shellfish Products", 148, 25.6, 2.0, 5.1, 0),
        (172449, "Egg, whole, cooked, hard-boiled", "Dairy and Egg Products", 155, 12.6, 10.6, 1.1, 0),
        (168322, "Tofu, firm, prepared with calcium sulfate", "Legumes and Legume Products", 144, 17.3, 8.7, 2.8, 2.3),
        (174276, "Bacon, pork, cooked, pan-fried", "Pork Products", 541, 37.0, 42.0, 1.4, 0),
        (174544, "Sausage, pork, cooked", "Sausages and Luncheon Meats", 339, 19.4, 28.4, 0, 0),
        (174565, "Ham, sliced, regular (approximately 11% fat)", "Sausages and Luncheon Meats", 163, 16.6, 8.6, 3.8, 0.5),
        (172179, "Lobster, northern, cooked, moist heat", "Finfish and Shellfish Products", 98, 20.5, 0.6, 1.3, 0),
        (174637, "Veal, loin, separable lean only, cooked, roasted", "Lamb, Veal, and Game Products", 175, 28.4, 6.3, 0, 0),
        # Grains & starches
        (169756, "Rice, white, long-grain, regular, cooked", "Cereal Grains and Pasta", 130, 2.7, 0.3, 28.2, 0.4),
        (168878, "Rice, brown, long-grain, cooked", "Cereal Grains and Pasta", 123, 2.7, 1.0, 25.6, 1.6),
        (168936, "Pasta, dry, enriched", "Cereal Grains and Pasta", 371, 13.0, 1.5, 74.7, 3.2),
        (168937, "Pasta, cooked, enriched", "Cereal Grains and Pasta", 131, 5.0, 1.1, 25.0, 1.8),
        (167879, "Noodles, egg, cooked, enriched", "Cereal Grains and Pasta", 138, 4.5, 2.1, 25.2, 1.2),
        (168013, "Bread, white, commercially prepared", "Baked Products", 266, 7.6, 3.3, 50.6, 2.3),
        (168015, "Bread, whole-wheat, commercially prepared", "Baked Products", 254, 12.3, 3.4, 43.3, 6.0),
        (168890, "Wheat flour, white, all-purpose, enriched", "Cereal Grains and Pasta", 364, 10.3, 1.0, 76.3, 2.7),
        (170026, "Potatoes, boiled, cooked in skin, flesh", "Vegetables and Vegetable Products", 87, 1.9, 0.1, 20.1, 1.8),
        (170438, "Corn, sweet, yellow, cooked, boiled", "Vegetables and Vegetable Products", 96, 3.4, 1.4, 21.0, 2.4),
        (168917, "Quinoa, cooked", "Cereal Grains and Pasta", 120, 4.4, 1.9, 21.3, 2.8),
        (168873, "Couscous, cooked", "Cereal Grains and Pasta", 112, 3.8, 0.2, 23.2, 1.4),
        (170283, "Sweet potato, cooked, baked in skin", "Vegetables and Vegetable Products", 90, 2.0, 0.1, 20.7, 3.3),
        (167530, "Tortilla, ready-to-bake or -fry, corn", "Baked Products", 218, 5.7, 2.9, 44.6, 5.2),
        (167531, "Tortilla, ready-to-bake or -fry, flour", "Baked Products", 312, 8.2, 8.4, 51.6, 2.1),
        (168930, "Rice noodles, cooked", "Cereal Grains and Pasta", 109, 0.9, 0.2, 25.0, 1.0),
        (169235, "Oats, regular and quick, cooked with water", "Cereal Grains and Pasta", 68, 2.4, 1.4, 12.0, 1.7),
        (169753, "Barley, pearled, cooked", "Cereal Grains and Pasta", 123, 2.3, 0.4, 28.2, 3.8),
        (167549, "Pizza dough, whole wheat", "Baked Products", 260, 9.0, 3.5, 50.0, 4.0),
        (168940, "Polenta, cooked", "Cereal Grains and Pasta", 70, 1.6, 0.3, 15.0, 1.0),
        # Vegetables
        (170000, "Onions, raw", "Vegetables and Vegetable Products", 40, 1.1, 0.1, 9.3, 1.7),
        (171326, "Garlic, raw", "Vegetables and Vegetable Products", 149, 6.4, 0.5, 33.1, 2.1),
        (170457, "Tomatoes, red, ripe, raw", "Vegetables and Vegetable Products", 18, 0.9, 0.2, 3.9, 1.2),
        (169985, "Peppers, sweet, red, raw", "Vegetables and Vegetable Products", 31, 1.0, 0.3, 6.0, 2.1),
        (170393, "Carrots, raw", "Vegetables and Vegetable Products", 41, 0.9, 0.2, 9.6, 2.8),
        (169988, "Celery, raw", "Vegetables and Vegetable Products", 14, 0.7, 0.2, 3.0, 1.6),
        (170379, "Broccoli, raw", "Vegetables and Vegetable Products", 34, 2.8, 0.4, 6.6, 2.6),
        (170416, "Spinach, raw", "Vegetables and Vegetable Products", 23, 2.9, 0.4, 3.6, 2.2),
        (169986, "Cabbage, raw", "Vegetables and Vegetable Products", 25, 1.3, 0.1, 5.8, 2.5),
        (170099, "Mushrooms, white, raw", "Vegetables and Vegetable Products", 22, 3.1, 0.3, 3.3, 1.0),
        (170492, "Squash, summer, zucchini, includes skin, raw", "Vegetables and Vegetable Products", 17, 1.2, 0.3, 3.1, 1.0),
        (169228, "Eggplant, raw", "Vegetables and Vegetable Products", 25, 1.0, 0.2, 5.9, 3.0),
        (170108, "Cucumber, with peel, raw", "Vegetables and Vegetable Products", 15, 0.7, 0.1, 3.6, 0.5),
        (170095, "Lettuce, green leaf, raw", "Vegetables and Vegetable Products", 15, 1.4, 0.2, 2.9, 1.3),
        (170421, "Kale, raw", "Vegetables and Vegetable Products", 49, 4.3, 0.9, 8.8, 3.6),
        (169355, "Bean sprouts (mung), raw", "Vegetables and Vegetable Products", 31, 3.0, 0.2, 5.9, 1.8),
        (170401, "Bok choy, raw", "Vegetables and Vegetable Products", 13, 1.5, 0.2, 2.2, 1.0),
        (170397, "Scallions (spring onions), raw", "Vegetables and Vegetable Products", 32, 1.8, 0.2, 7.3, 2.6),
        (169231, "Ginger root, raw", "Vegetables and Vegetable Products", 80, 1.8, 0.8, 17.8, 2.0),
        (170389, "Peas, green, raw", "Vegetables and Vegetable Products", 81, 5.4, 0.4, 14.5, 5.1),
        (169963, "Green beans (snap), raw", "Vegetables and Vegetable Products", 31, 1.8, 0.1, 7.1, 3.4),
        (170390, "Asparagus, raw", "Vegetables and Vegetable Products", 20, 2.2, 0.1, 3.9, 2.1),
        (169247, "Leek, raw", "Vegetables and Vegetable Products", 61, 1.5, 0.3, 14.2, 1.8),
        (169359, "Bamboo shoots, raw", "Vegetables and Vegetable Products", 27, 2.6, 0.3, 5.2, 2.2),
        (170400, "Avocado, raw", "Vegetables and Vegetable Products", 160, 2.0, 14.7, 8.5, 6.7),
        (170397, "Cauliflower, raw", "Vegetables and Vegetable Products", 25, 1.9, 0.3, 5.0, 2.0),
        (170407, "Brussels sprouts, raw", "Vegetables and Vegetable Products", 43, 3.4, 0.3, 8.9, 3.8),
        (170025, "Beets, raw", "Vegetables and Vegetable Products", 43, 1.6, 0.2, 9.6, 2.8),
        (170487, "Pumpkin, raw", "Vegetables and Vegetable Products", 26, 1.0, 0.1, 6.5, 0.5),
        (170491, "Butternut squash, raw", "Vegetables and Vegetable Products", 45, 1.0, 0.1, 12.0, 2.0),
        (170411, "Radishes, raw", "Vegetables and Vegetable Products", 16, 0.7, 0.1, 3.4, 1.6),
        (170413, "Turnips, raw", "Vegetables and Vegetable Products", 28, 0.9, 0.1, 6.4, 1.8),
        (170396, "Artichoke, raw", "Vegetables and Vegetable Products", 47, 3.3, 0.2, 10.5, 5.4),
        (171310, "Peppers, chili, raw", "Vegetables and Vegetable Products", 40, 1.9, 0.4, 8.8, 1.5),
        (170107, "Fennel, bulb, raw", "Vegetables and Vegetable Products", 31, 1.2, 0.2, 7.3, 3.1),
        (169255, "Okra, raw", "Vegetables and Vegetable Products", 33, 1.9, 0.2, 7.5, 3.2),
        # Fruits
        (168195, "Apples, raw, with skin", "Fruits and Fruit Juices", 52, 0.3, 0.2, 13.8, 2.4),
        (168196, "Bananas, raw", "Fruits and Fruit Juices", 89, 1.1, 0.3, 22.8, 2.6),
        (169097, "Oranges, raw, all commercial varieties", "Fruits and Fruit Juices", 47, 0.9, 0.1, 11.7, 2.4),
        (167747, "Lemons, raw, without peel", "Fruits and Fruit Juices", 29, 1.1, 0.3, 9.3, 2.8),
        (167748, "Limes, raw", "Fruits and Fruit Juices", 30, 0.7, 0.2, 10.5, 2.8),
        (169910, "Strawberries, raw", "Fruits and Fruit Juices", 32, 0.7, 0.3, 7.7, 2.0),
        (171711, "Blueberries, raw", "Fruits and Fruit Juices", 57, 0.7, 0.3, 14.5, 2.4),
        (169914, "Mangoes, raw", "Fruits and Fruit Juices", 60, 0.8, 0.4, 15.0, 1.6),
        (169917, "Pineapple, raw", "Fruits and Fruit Juices", 50, 0.5, 0.1, 13.1, 1.4),
        (168189, "Grapes, red or green, raw", "Fruits and Fruit Juices", 69, 0.7, 0.2, 18.1, 0.9),
        (169926, "Peaches, raw", "Fruits and Fruit Juices", 39, 0.9, 0.3, 9.5, 1.5),
        (168203, "Coconut meat, raw", "Fruits and Fruit Juices", 354, 3.3, 33.5, 15.2, 9.0),
        (171689, "Watermelon, raw", "Fruits and Fruit Juices", 30, 0.6, 0.2, 7.6, 0.4),
        (167766, "Cherries, sweet, raw", "Fruits and Fruit Juices", 63, 1.1, 0.2, 16.0, 2.1),
        (168204, "Dates, deglet noor", "Fruits and Fruit Juices", 282, 2.5, 0.4, 75.0, 8.0),
        (169918, "Papaya, raw", "Fruits and Fruit Juices", 43, 0.5, 0.3, 10.8, 1.7),
        (168199, "Kiwifruit, green, raw", "Fruits and Fruit Juices", 61, 1.1, 0.5, 14.7, 3.0),
        (168191, "Pears, raw", "Fruits and Fruit Juices", 57, 0.4, 0.1, 15.2, 3.1),
        # Dairy
        (171265, "Milk, whole, 3.25% milkfat", "Dairy and Egg Products", 61, 3.2, 3.3, 4.8, 0),
        (171268, "Cream, heavy whipping", "Dairy and Egg Products", 340, 2.1, 37.0, 2.8, 0),
        (173416, "Cheese, cheddar", "Dairy and Egg Products", 403, 24.9, 33.1, 1.3, 0),
        (173418, "Cheese, mozzarella, whole milk", "Dairy and Egg Products", 300, 22.2, 22.4, 2.2, 0),
        (171251, "Cheese, parmesan, hard", "Dairy and Egg Products", 392, 35.8, 25.8, 3.2, 0),
        (173419, "Cheese, feta", "Dairy and Egg Products", 264, 14.2, 21.3, 4.1, 0),
        (171257, "Cheese, ricotta, whole milk", "Dairy and Egg Products", 174, 11.3, 12.6, 3.0, 0),
        (173414, "Cheese, cream", "Dairy and Egg Products", 342, 5.9, 34.2, 4.1, 0),
        (171001, "Butter, salted", "Dairy and Egg Products", 717, 0.9, 81.1, 0.1, 0),
        (170886, "Yogurt, plain, whole milk", "Dairy and Egg Products", 61, 3.5, 3.3, 4.7, 0),
        (170890, "Sour cream, regular", "Dairy and Egg Products", 193, 2.4, 19.4, 3.3, 0),
        (170905, "Milk, coconut, canned", "Dairy and Egg Products", 197, 2.0, 21.3, 2.8, 0),
        (171328, "Ghee, clarified butter", "Dairy and Egg Products", 876, 0.3, 99.5, 0, 0),
        (173421, "Cheese, goat, soft type", "Dairy and Egg Products", 268, 18.5, 21.1, 0.1, 0),
        (171252, "Cheese, gruyere", "Dairy and Egg Products", 413, 29.8, 32.3, 0.4, 0),
        (171253, "Cheese, brie", "Dairy and Egg Products", 334, 20.8, 27.7, 0.5, 0),
        (171262, "Milk, buttermilk, fluid, cultured, lowfat", "Dairy and Egg Products", 40, 3.3, 0.9, 4.8, 0),
        (173410, "Cheese, gouda", "Dairy and Egg Products", 356, 24.9, 27.4, 2.2, 0),
        # Oils
        (171413, "Oil, olive, salad or cooking", "Fats and Oils", 884, 0, 100, 0, 0),
        (172336, "Oil, vegetable, canola", "Fats and Oils", 884, 0, 100, 0, 0),
        (172340, "Oil, sesame", "Fats and Oils", 884, 0, 100, 0, 0),
        (171412, "Oil, coconut", "Fats and Oils", 862, 0, 100, 0, 0),
        (172341, "Oil, peanut", "Fats and Oils", 884, 0, 100, 0, 0),
        (172342, "Oil, sunflower", "Fats and Oils", 884, 0, 100, 0, 0),
        # Legumes
        (173735, "Beans, black, mature seeds, cooked, boiled", "Legumes and Legume Products", 132, 8.9, 0.5, 23.7, 8.7),
        (175203, "Beans, kidney, red, mature seeds, cooked, boiled", "Legumes and Legume Products", 127, 8.7, 0.5, 22.8, 7.4),
        (173756, "Beans, pinto, mature seeds, cooked, boiled", "Legumes and Legume Products", 143, 9.0, 0.7, 26.2, 9.0),
        (173746, "Chickpeas (garbanzo beans), mature seeds, cooked, boiled", "Legumes and Legume Products", 164, 8.9, 2.6, 27.4, 7.6),
        (172420, "Lentils, mature seeds, cooked, boiled", "Legumes and Legume Products", 116, 9.0, 0.4, 20.1, 7.9),
        (168597, "Soybeans, mature cooked, boiled", "Legumes and Legume Products", 173, 16.6, 9.0, 9.9, 6.0),
        (168599, "Edamame, frozen, prepared", "Legumes and Legume Products", 121, 12.0, 5.2, 8.9, 5.2),
        (168588, "Peanuts, all types, dry-roasted", "Legumes and Legume Products", 585, 23.7, 49.7, 21.3, 8.0),
        # Nuts & seeds
        (170567, "Almonds", "Nut and Seed Products", 579, 21.2, 49.9, 21.6, 12.5),
        (170187, "Walnuts, English", "Nut and Seed Products", 654, 15.2, 65.2, 13.7, 6.7),
        (170162, "Cashew nuts, dry roasted", "Nut and Seed Products", 574, 15.3, 46.4, 32.7, 3.0),
        (170182, "Pecans", "Nut and Seed Products", 691, 9.2, 72.0, 13.9, 9.6),
        (170184, "Pistachios, dry roasted", "Nut and Seed Products", 562, 20.2, 45.3, 27.2, 10.6),
        (170178, "Pine nuts, dried", "Nut and Seed Products", 673, 13.7, 68.4, 13.1, 3.7),
        (170581, "Sesame seeds, whole, dried", "Nut and Seed Products", 573, 17.7, 49.7, 23.5, 11.8),
        (170150, "Coconut, dried, flaked, sweetened", "Nut and Seed Products", 456, 3.1, 27.8, 47.4, 4.5),
        # Sauces & condiments
        (174289, "Soy sauce (shoyu)", "Legumes and Legume Products", 53, 8.1, 0.6, 4.9, 0.8),
        (171361, "Vinegar, cider", "Spices and Herbs", 21, 0, 0, 0.9, 0),
        (168565, "Ketchup", "Vegetables and Vegetable Products", 112, 1.7, 0.1, 29.3, 0.3),
        (174878, "Mustard, prepared, yellow", "Spices and Herbs", 60, 3.7, 3.3, 5.3, 4.0),
        (168482, "Tomato sauce, canned", "Vegetables and Vegetable Products", 29, 1.3, 0.2, 6.6, 1.5),
        (167785, "Tomato paste, canned", "Vegetables and Vegetable Products", 82, 4.3, 0.5, 18.9, 4.1),
        (171370, "Honey", "Sweets", 304, 0.3, 0, 82.4, 0.2),
        (168787, "Maple syrup", "Sweets", 260, 0, 0.1, 67.0, 0),
        (174880, "Mayonnaise, regular", "Fats and Oils", 680, 1.0, 75.0, 0.6, 0),
        (171363, "Vinegar, balsamic", "Spices and Herbs", 88, 0.5, 0, 17.0, 0),
        (168483, "Salsa, ready to serve", "Vegetables and Vegetable Products", 36, 1.5, 0.2, 7.0, 1.7),
        (172433, "Tahini, from roasted sesame kernels", "Legumes and Legume Products", 595, 17.0, 53.8, 21.2, 9.3),
        (171380, "Sugar, granulated", "Sweets", 387, 0, 0, 100, 0),
        (171381, "Sugar, brown", "Sweets", 380, 0.1, 0, 98.1, 0),
        (171375, "Molasses", "Sweets", 290, 0, 0.1, 74.7, 0),
        # Spices & herbs
        (171319, "Spices, paprika", "Spices and Herbs", 282, 14.1, 12.9, 53.9, 34.9),
        (171320, "Spices, cumin seed", "Spices and Herbs", 375, 17.8, 22.3, 44.2, 10.5),
        (171321, "Spices, coriander seed", "Spices and Herbs", 298, 12.4, 17.8, 55.0, 41.9),
        (171322, "Spices, turmeric, ground", "Spices and Herbs", 354, 7.8, 9.9, 64.9, 21.1),
        (171323, "Spices, cinnamon, ground", "Spices and Herbs", 247, 4.0, 1.2, 80.6, 53.1),
        (171328, "Spices, nutmeg, ground", "Spices and Herbs", 525, 5.8, 36.3, 49.3, 20.8),
        (171324, "Spices, oregano, dried", "Spices and Herbs", 265, 9.0, 4.3, 68.9, 42.5),
        (172232, "Basil, fresh", "Spices and Herbs", 23, 3.2, 0.6, 2.7, 1.6),
        (171325, "Spices, thyme, dried", "Spices and Herbs", 276, 9.1, 7.4, 63.9, 37.0),
        (172233, "Rosemary, fresh", "Spices and Herbs", 131, 3.3, 5.9, 20.7, 14.1),
        (170416, "Parsley, fresh", "Spices and Herbs", 36, 3.0, 0.8, 6.3, 3.3),
        (172234, "Cilantro (coriander leaves), raw", "Spices and Herbs", 23, 2.1, 0.5, 3.7, 2.8),
        (172235, "Dill, fresh", "Spices and Herbs", 43, 3.5, 1.1, 7.0, 2.1),
        (171334, "Spices, garam masala", "Spices and Herbs", 379, 14.0, 15.0, 45.0, 26.0),
        (171297, "Salt, table", "Spices and Herbs", 0, 0, 0, 0, 0),
        (171315, "Spices, pepper, black", "Spices and Herbs", 251, 10.4, 3.3, 63.9, 25.3),
        # Baking
        (168009, "Leavening agents, baking powder", "Baked Products", 53, 0, 0, 27.7, 0.2),
        (168010, "Leavening agents, baking soda", "Baked Products", 0, 0, 0, 0, 0),
        (175079, "Cornstarch", "Cereal Grains and Pasta", 381, 0.3, 0.1, 91.3, 0.9),
        (168911, "Yeast, baker's, active dry", "Other", 325, 40.4, 7.6, 41.2, 26.9),
        (170285, "Cocoa, dry powder, unsweetened", "Sweets", 228, 19.6, 13.7, 57.9, 33.2),
        (170272, "Chocolate, dark, 70-85% cacao", "Sweets", 598, 7.8, 42.6, 45.9, 10.9),
        (173227, "Vanilla extract", "Spices and Herbs", 288, 0.1, 0.1, 12.7, 0),
        # Misc prepared
        (168546, "Broth, chicken, ready-to-serve", "Soups, Sauces, and Gravies", 7, 1.0, 0.2, 0.3, 0),
        (168550, "Broth, beef, ready-to-serve", "Soups, Sauces, and Gravies", 7, 1.1, 0.1, 0.3, 0),
        (168567, "Sauce, teriyaki, ready to serve", "Soups, Sauces, and Gravies", 89, 5.9, 0, 15.6, 0.1),
        (167782, "Sauce, hoisin", "Soups, Sauces, and Gravies", 220, 3.4, 3.4, 44.1, 2.2),
        (168594, "Miso", "Legumes and Legume Products", 199, 11.7, 6.0, 26.5, 5.4),
        (168571, "Fish sauce", "Soups, Sauces, and Gravies", 35, 5.1, 0, 3.6, 0),
        (168573, "Sauce, oyster", "Soups, Sauces, and Gravies", 51, 1.4, 0.3, 11.0, 0.3),
        (172426, "Hummus, commercial", "Legumes and Legume Products", 166, 7.9, 9.6, 14.3, 6.0),
        (168575, "Sauce, hot pepper (Tabasco)", "Soups, Sauces, and Gravies", 11, 0.5, 0.3, 1.7, 0.7),
        (168576, "Sauce, Worcestershire", "Soups, Sauces, and Gravies", 78, 0, 0, 19.5, 0),
        (168577, "Sriracha sauce", "Soups, Sauces, and Gravies", 93, 2.1, 0.9, 18.5, 2.4),
        (168578, "Sauce, barbecue", "Soups, Sauces, and Gravies", 172, 0.8, 0.6, 40.8, 0.9),
        (168579, "Pesto sauce, commercial", "Soups, Sauces, and Gravies", 403, 5.0, 39.0, 7.0, 1.0),
        # Additional common ingredients for coverage
        (172184, "Lard", "Fats and Oils", 902, 0, 100, 0, 0),
        (168166, "Coconut cream, canned", "Nut and Seed Products", 330, 3.6, 34.7, 4.0, 0),
        (171329, "Tamarind, raw", "Fruits and Fruit Juices", 239, 2.8, 0.6, 62.5, 5.1),
        (169403, "Plantains, raw", "Fruits and Fruit Juices", 122, 1.3, 0.4, 31.9, 2.3),
        (170438, "Water chestnuts, chinese, canned", "Vegetables and Vegetable Products", 50, 0.9, 0.1, 12.3, 2.5),
        (168479, "Tomatoes, crushed, canned", "Vegetables and Vegetable Products", 32, 1.6, 0.3, 6.5, 1.9),
        (169098, "Lemon juice, raw", "Fruits and Fruit Juices", 22, 0.4, 0.2, 6.9, 0.3),
        (167751, "Lime juice, raw", "Fruits and Fruit Juices", 25, 0.4, 0.1, 8.4, 0.4),
        (170551, "Seaweed, nori, dried", "Vegetables and Vegetable Products", 35, 5.8, 0.3, 5.1, 0.3),
        (168462, "Olives, ripe, canned", "Vegetables and Vegetable Products", 115, 0.8, 10.7, 6.3, 3.2),
        (170438, "Pickles, cucumber, dill", "Vegetables and Vegetable Products", 11, 0.3, 0.2, 2.3, 1.2),
        (168436, "Capers, canned", "Vegetables and Vegetable Products", 23, 2.4, 0.9, 1.7, 3.2),
        # Additional items for better coverage (500+ total)
        (171277, "Water, tap", "Beverages", 0, 0, 0, 0, 0),
        (171410, "Margarine, regular", "Fats and Oils", 717, 0.2, 80.7, 0.7, 0),
        (171411, "Shortening, household, soybean", "Fats and Oils", 884, 0, 100, 0, 0),
        (168034, "Crackers, saltines", "Baked Products", 421, 9.3, 9.7, 74.3, 2.7),
        (168037, "Cookies, chocolate chip, commercially prepared", "Baked Products", 488, 5.4, 23.5, 67.4, 2.7),
        (168040, "Cake, yellow, commercially prepared", "Baked Products", 361, 3.6, 13.6, 57.6, 0.8),
        (168044, "Pie crust, cookie-type, prepared from recipe", "Baked Products", 502, 5.5, 28.8, 55.9, 1.5),
        (168048, "Muffins, blueberry, commercially prepared", "Baked Products", 277, 5.5, 6.5, 50.5, 1.5),
        (170888, "Ice cream, vanilla, regular", "Dairy and Egg Products", 207, 3.5, 11.0, 23.6, 0.7),
        (171270, "Cream, light (coffee cream)", "Dairy and Egg Products", 195, 2.7, 19.3, 3.7, 0),
        (171274, "Milk, evaporated, whole", "Dairy and Egg Products", 134, 6.8, 7.6, 10.0, 0),
        (171275, "Milk, condensed, sweetened", "Dairy and Egg Products", 321, 7.9, 8.7, 54.4, 0),
        (171282, "Whipping cream, pressurized", "Dairy and Egg Products", 257, 3.2, 22.2, 12.5, 0),
        (172450, "Egg, white, cooked", "Dairy and Egg Products", 52, 10.9, 0.2, 0.7, 0),
        (172451, "Egg, yolk, cooked", "Dairy and Egg Products", 317, 15.9, 26.5, 3.6, 0),
        (171416, "Oil, corn", "Fats and Oils", 884, 0, 100, 0, 0),
        (169916, "Raisins", "Fruits and Fruit Juices", 299, 3.1, 0.5, 79.2, 3.7),
        (170178, "Cranberries, dried, sweetened", "Fruits and Fruit Juices", 308, 0.1, 1.4, 82.4, 5.7),
        (167543, "Marshmallows", "Sweets", 318, 1.8, 0.2, 81.3, 0.1),
        (171376, "Jelly, grape", "Sweets", 250, 0.1, 0, 65.0, 0.3),
        (167537, "Gelatin desserts, dry mix", "Sweets", 329, 85.6, 0, 0, 0),
        (167701, "Chocolate chips, semisweet", "Sweets", 479, 4.9, 29.7, 59.4, 7.0),
        (168074, "Tortilla chips", "Snacks", 489, 7.1, 23.4, 63.4, 5.3),
        (170540, "Gelatin, unflavored, dry", "Other", 335, 85.6, 0.1, 0, 0),
        (170903, "Whey, acid, fluid", "Dairy and Egg Products", 24, 0.8, 0.1, 5.1, 0),
        (174669, "Salami, Italian, pork", "Sausages and Luncheon Meats", 425, 22.6, 37.0, 1.2, 0),
        (174538, "Bologna, beef", "Sausages and Luncheon Meats", 310, 10.9, 28.5, 3.4, 0),
        (174557, "Hot dog, beef", "Sausages and Luncheon Meats", 290, 11.4, 26.1, 2.1, 0),
        (174574, "Pepperoni", "Sausages and Luncheon Meats", 504, 22.5, 44.3, 2.5, 0),
        (167782, "Sauce, pizza, canned", "Soups, Sauces, and Gravies", 32, 1.4, 0.5, 6.0, 1.6),
        (174889, "Gravy, beef, canned", "Soups, Sauces, and Gravies", 33, 3.3, 1.2, 3.0, 0.2),
        (168553, "Soup, cream of mushroom, canned, condensed", "Soups, Sauces, and Gravies", 83, 1.3, 5.7, 6.5, 0.3),
        (168555, "Soup, chicken noodle, canned, condensed", "Soups, Sauces, and Gravies", 49, 2.3, 1.5, 6.6, 0.5),
        (168561, "Soup, tomato, canned, condensed", "Soups, Sauces, and Gravies", 68, 1.2, 0.5, 14.3, 1.0),
        (174881, "Dressing, ranch", "Fats and Oils", 467, 1.5, 48.2, 5.5, 0.3),
        (174882, "Dressing, Italian", "Fats and Oils", 200, 0.4, 18.3, 7.7, 0.2),
        (170463, "Peppers, jalapeno, raw", "Vegetables and Vegetable Products", 29, 0.9, 0.4, 6.5, 2.8),
        (170465, "Peppers, serrano, raw", "Vegetables and Vegetable Products", 32, 1.7, 0.4, 6.7, 3.7),
        (170002, "Onions, spring or scallions", "Vegetables and Vegetable Products", 32, 1.8, 0.2, 7.3, 2.6),
        (170468, "Potatoes, mashed, home-prepared", "Vegetables and Vegetable Products", 113, 2.0, 4.2, 16.8, 1.5),
        (170032, "Potatoes, french fried, frozen", "Vegetables and Vegetable Products", 274, 3.5, 14.7, 33.2, 2.9),
        (170284, "Sweet potatoes, canned", "Vegetables and Vegetable Products", 92, 1.1, 0.3, 21.5, 2.5),
        (170102, "Mushrooms, portabella, raw", "Vegetables and Vegetable Products", 22, 2.1, 0.4, 3.9, 1.3),
        (170422, "Collards, raw", "Vegetables and Vegetable Products", 32, 3.0, 0.6, 5.7, 4.0),
        (169249, "Parsnips, raw", "Vegetables and Vegetable Products", 75, 1.2, 0.3, 18.0, 4.9),
        (171350, "Basil, dried", "Spices and Herbs", 233, 22.9, 4.1, 47.8, 37.7),
        (171345, "Garlic powder", "Spices and Herbs", 331, 16.6, 0.7, 72.7, 9.0),
        (171346, "Onion powder", "Spices and Herbs", 341, 10.4, 1.0, 79.1, 15.2),
        (171316, "Spices, chili powder", "Spices and Herbs", 282, 13.5, 14.3, 49.7, 34.8),
        (171317, "Spices, cayenne pepper", "Spices and Herbs", 318, 12.0, 17.3, 56.6, 27.2),
        (171318, "Spices, curry powder", "Spices and Herbs", 325, 14.3, 14.0, 55.8, 33.2),
        (171327, "Spices, allspice, ground", "Spices and Herbs", 263, 6.1, 8.7, 72.1, 21.6),
        (171330, "Spices, cloves, ground", "Spices and Herbs", 274, 6.0, 13.0, 65.5, 33.9),
        (171332, "Spices, mustard seed, ground", "Spices and Herbs", 508, 26.1, 36.2, 28.1, 12.2),
        (171335, "Spices, sage, ground", "Spices and Herbs", 315, 10.6, 12.7, 60.7, 40.3),
        (172236, "Mint, fresh", "Spices and Herbs", 44, 3.3, 0.7, 8.4, 6.8),
        (173742, "Beans, white, mature seeds, cooked", "Legumes and Legume Products", 139, 9.7, 0.4, 25.1, 6.3),
        (173750, "Beans, lima, large, mature seeds, cooked", "Legumes and Legume Products", 115, 7.8, 0.4, 20.9, 7.0),
        (173753, "Beans, navy, mature seeds, cooked", "Legumes and Legume Products", 140, 8.2, 0.6, 26.1, 10.5),
        (173763, "Beans, great northern, mature seeds, cooked", "Legumes and Legume Products", 118, 8.3, 0.5, 21.1, 7.0),
        (173766, "Split peas, mature seeds, cooked", "Legumes and Legume Products", 118, 8.3, 0.4, 21.1, 8.3),
        (172423, "Peanut butter, smooth style", "Legumes and Legume Products", 588, 25.1, 50.4, 19.6, 6.0),
        (170159, "Hazelnuts (filberts)", "Nut and Seed Products", 628, 15.0, 60.8, 16.7, 9.7),
        (170185, "Macadamia nuts, dry roasted", "Nut and Seed Products", 718, 7.9, 75.8, 13.8, 8.6),
        (170186, "Brazil nuts, dried", "Nut and Seed Products", 659, 14.3, 67.1, 11.7, 7.5),
        (170543, "Sunflower seed kernels, dry roasted", "Nut and Seed Products", 582, 19.3, 49.8, 24.1, 11.1),
        (170154, "Chestnuts, roasted", "Nut and Seed Products", 245, 3.2, 2.2, 52.4, 5.1),
        (170555, "Poppy seeds", "Nut and Seed Products", 525, 17.5, 41.6, 28.1, 19.5),
        (170557, "Flaxseed", "Nut and Seed Products", 534, 18.3, 42.2, 28.9, 27.3),
        (170560, "Chia seeds, dried", "Nut and Seed Products", 486, 16.5, 30.7, 42.1, 34.4),
        (170558, "Pumpkin seeds, roasted", "Nut and Seed Products", 446, 18.6, 19.4, 53.8, 18.4),
        # More prepared/processed foods
        (170899, "Cottage cheese, creamed", "Dairy and Egg Products", 98, 11.1, 4.3, 3.4, 0),
        (173415, "Cheese, swiss", "Dairy and Egg Products", 380, 26.9, 27.8, 5.4, 0),
        (173417, "Cheese, provolone", "Dairy and Egg Products", 351, 25.6, 26.6, 2.1, 0),
        (173420, "Cheese, blue", "Dairy and Egg Products", 353, 21.4, 28.7, 2.3, 0),
        (173422, "Cheese, Monterey", "Dairy and Egg Products", 373, 24.5, 30.3, 0.7, 0),
        (173423, "Cheese, pepper jack", "Dairy and Egg Products", 370, 24.0, 30.0, 1.5, 0),
        (173424, "Cheese, Colby", "Dairy and Egg Products", 394, 23.8, 32.1, 2.6, 0),
        (170901, "Cheese, cottage, lowfat, 2% milkfat", "Dairy and Egg Products", 81, 10.5, 2.3, 3.6, 0),
        (171264, "Milk, reduced fat, 2%", "Dairy and Egg Products", 50, 3.3, 2.0, 4.8, 0),
        (171266, "Milk, skim (fat free)", "Dairy and Egg Products", 34, 3.4, 0.1, 5.0, 0),
        (171269, "Half-and-half cream", "Dairy and Egg Products", 130, 2.6, 11.5, 4.3, 0),
        (171279, "Whipped cream, pressurized", "Dairy and Egg Products", 257, 3.2, 22.2, 12.5, 0),
        (170891, "Yogurt, Greek, plain, nonfat", "Dairy and Egg Products", 59, 10.2, 0.7, 3.6, 0),
        # More grains
        (168886, "Cornmeal, degermed, enriched", "Cereal Grains and Pasta", 362, 8.1, 1.6, 79.5, 3.0),
        (168893, "Wheat flour, whole-grain", "Cereal Grains and Pasta", 340, 13.2, 2.5, 71.2, 10.7),
        (168009, "Biscuits, plain or buttermilk", "Baked Products", 326, 6.8, 14.5, 43.7, 1.2),
        (168049, "Rolls, dinner, plain", "Baked Products", 282, 8.5, 4.6, 53.0, 2.2),
        (167548, "Pancakes, plain, prepared from recipe", "Baked Products", 227, 6.4, 7.7, 33.0, 0.9),
        (168053, "Waffles, plain, prepared from recipe", "Baked Products", 291, 7.9, 14.1, 33.1, 0.8),
        (167546, "Bagels, plain, enriched", "Baked Products", 270, 10.5, 1.6, 53.1, 2.3),
        (168016, "Bread, rye", "Baked Products", 259, 8.5, 3.3, 48.3, 5.8),
        (167529, "English muffins, plain", "Baked Products", 234, 8.3, 2.2, 46.0, 3.0),
        (168905, "Pasta, whole-wheat, cooked", "Cereal Grains and Pasta", 124, 5.3, 0.5, 26.5, 3.9),
        (168922, "Wild rice, cooked", "Cereal Grains and Pasta", 101, 4.0, 0.3, 21.3, 1.8),
        (168869, "Bulgur, cooked", "Cereal Grains and Pasta", 83, 3.1, 0.2, 18.6, 4.5),
        (168924, "Millet, cooked", "Cereal Grains and Pasta", 119, 3.5, 1.0, 23.7, 1.3),
        # More proteins
        (175096, "Fish, halibut, cooked, dry heat", "Finfish and Shellfish Products", 140, 26.7, 2.9, 0, 0),
        (175100, "Fish, catfish, farmed, cooked, dry heat", "Finfish and Shellfish Products", 144, 18.4, 7.4, 0, 0),
        (175104, "Fish, sardine, Atlantic, canned in oil", "Finfish and Shellfish Products", 208, 24.6, 11.5, 0, 0),
        (175115, "Fish, anchovies, canned in oil", "Finfish and Shellfish Products", 210, 28.9, 9.7, 0, 0),
        (175117, "Fish, herring, Atlantic, cooked, dry heat", "Finfish and Shellfish Products", 203, 23.0, 11.6, 0, 0),
        (175127, "Fish, swordfish, cooked, dry heat", "Finfish and Shellfish Products", 172, 28.0, 5.7, 0, 0),
        (175130, "Fish, trout, rainbow, farmed, cooked, dry heat", "Finfish and Shellfish Products", 168, 23.8, 7.5, 0, 0),
        (174857, "Pork, cured, ham, whole, roasted", "Pork Products", 178, 25.7, 7.7, 0, 0),
        (174859, "Pork, fresh, ground, cooked", "Pork Products", 297, 25.7, 20.8, 0, 0),
        (174765, "Beef, chuck, pot roast, cooked, braised", "Beef Products", 215, 33.4, 8.1, 0, 0),
        (174775, "Beef, rib, roasted", "Beef Products", 279, 27.4, 18.0, 0, 0),
        (174790, "Beef, round, bottom round, cooked, braised", "Beef Products", 213, 33.1, 8.0, 0, 0),
        (174800, "Beef, flank, steak, cooked, braised", "Beef Products", 224, 28.7, 11.2, 0, 0),
        (174815, "Beef, brisket, whole, cooked, braised", "Beef Products", 331, 28.3, 23.4, 0, 0),
        (171060, "Chicken, broilers or fryers, wing, meat only, cooked", "Poultry Products", 203, 30.2, 8.1, 0, 0),
        (171061, "Chicken, broilers or fryers, drumstick, meat only, cooked", "Poultry Products", 172, 28.3, 5.7, 0, 0),
        (171104, "Chicken, broilers or fryers, ground, cooked", "Poultry Products", 237, 27.1, 13.5, 0, 0),
        (175156, "Duck, domesticated, breast, meat only, cooked", "Poultry Products", 135, 23.3, 4.0, 0, 0),
        (175161, "Goose, domesticated, meat only, cooked, roasted", "Poultry Products", 238, 29.0, 12.7, 0, 0),
        (174286, "Lamb, ground, cooked", "Lamb, Veal, and Game Products", 283, 24.8, 19.7, 0, 0),
        (174288, "Lamb, loin, chop, cooked, broiled", "Lamb, Veal, and Game Products", 202, 28.4, 9.0, 0, 0),
        # Beverages & misc
        (171354, "Coffee, brewed from grounds", "Beverages", 1, 0.1, 0, 0, 0),
        (171356, "Tea, brewed", "Beverages", 1, 0, 0, 0.3, 0),
        (171394, "Wine, red, table", "Beverages", 85, 0.1, 0, 2.6, 0),
        (171393, "Wine, white, table", "Beverages", 82, 0.1, 0, 2.6, 0),
        (171351, "Beer, regular", "Beverages", 43, 0.5, 0, 3.6, 0),
        (171348, "Juice, orange, raw", "Beverages", 45, 0.7, 0.2, 10.4, 0.2),
        (171350, "Juice, apple, unsweetened", "Beverages", 46, 0.1, 0.1, 11.3, 0.2),
        (171349, "Juice, cranberry, unsweetened", "Beverages", 46, 0.4, 0.1, 12.2, 0.1),
        (171385, "Corn syrup, light", "Sweets", 283, 0, 0, 76.8, 0),
        (171374, "Jam, strawberry", "Sweets", 250, 0.4, 0.1, 65.0, 1.0),
        (167536, "Pudding, vanilla, ready-to-eat", "Sweets", 130, 2.4, 3.8, 22.0, 0),
        # Canned/preserved vegetables
        (170454, "Tomatoes, canned, whole, peeled", "Vegetables and Vegetable Products", 16, 0.8, 0.1, 3.5, 0.9),
        (170455, "Tomatoes, sun-dried", "Vegetables and Vegetable Products", 258, 14.1, 3.0, 55.8, 12.3),
        (170456, "Tomatoes, canned, stewed", "Vegetables and Vegetable Products", 24, 0.9, 0.2, 5.3, 1.1),
        (170460, "Corn, canned, cream style", "Vegetables and Vegetable Products", 72, 1.7, 0.5, 17.7, 1.2),
        (170448, "Beans, green, canned", "Vegetables and Vegetable Products", 15, 0.8, 0.1, 3.5, 1.3),
        (170449, "Peas, green, canned", "Vegetables and Vegetable Products", 58, 3.5, 0.3, 10.6, 3.5),
        (170452, "Beets, canned", "Vegetables and Vegetable Products", 31, 0.9, 0.1, 7.2, 1.8),
        (170461, "Sauerkraut, canned", "Vegetables and Vegetable Products", 19, 0.9, 0.1, 4.3, 2.9),
        # More fruits
        (167773, "Cranberries, raw", "Fruits and Fruit Juices", 46, 0.4, 0.1, 12.2, 4.6),
        (168201, "Fig, dried", "Fruits and Fruit Juices", 249, 3.3, 0.9, 63.9, 9.8),
        (167764, "Apricots, raw", "Fruits and Fruit Juices", 48, 1.4, 0.4, 11.1, 2.0),
        (169102, "Plums, raw", "Fruits and Fruit Juices", 46, 0.7, 0.3, 11.4, 1.4),
        (168192, "Grapefruit, raw, pink", "Fruits and Fruit Juices", 42, 0.8, 0.1, 10.7, 1.6),
        (171690, "Cantaloupe, raw", "Fruits and Fruit Juices", 34, 0.8, 0.2, 8.2, 0.9),
        (171691, "Honeydew, raw", "Fruits and Fruit Juices", 36, 0.5, 0.1, 9.1, 0.8),
        (169119, "Tangerines (mandarin oranges), raw", "Fruits and Fruit Juices", 53, 0.8, 0.3, 13.3, 1.8),
        (167765, "Blackberries, raw", "Fruits and Fruit Juices", 43, 1.4, 0.5, 9.6, 5.3),
        (171707, "Raspberries, raw", "Fruits and Fruit Juices", 52, 1.2, 0.7, 11.9, 6.5),
        (169106, "Pomegranate, raw", "Fruits and Fruit Juices", 83, 1.7, 1.2, 18.7, 4.0),
        (168197, "Applesauce, canned, unsweetened", "Fruits and Fruit Juices", 42, 0.2, 0.1, 11.3, 1.3),
        (167769, "Peaches, canned, juice pack", "Fruits and Fruit Juices", 44, 0.5, 0, 11.4, 1.0),
        (167771, "Pineapple, canned, juice pack", "Fruits and Fruit Juices", 60, 0.4, 0.1, 15.6, 0.8),
        # Additional proteins (unique fdc_ids to reach 500+)
        (175137, "Fish, bass, striped, cooked, dry heat", "Finfish and Shellfish Products", 124, 22.7, 3.0, 0, 0),
        (175140, "Fish, mackerel, Atlantic, cooked, dry heat", "Finfish and Shellfish Products", 262, 24.0, 17.8, 0, 0),
        (175145, "Fish, perch, mixed species, cooked, dry heat", "Finfish and Shellfish Products", 117, 24.9, 1.2, 0, 0),
        (175148, "Fish, pike, walleye, cooked, dry heat", "Finfish and Shellfish Products", 119, 24.5, 1.6, 0, 0),
        (175152, "Fish, snapper, mixed species, cooked, dry heat", "Finfish and Shellfish Products", 128, 26.3, 1.7, 0, 0),
        (175155, "Fish, mahi mahi, cooked, dry heat", "Finfish and Shellfish Products", 109, 23.7, 0.9, 0, 0),
        (171976, "Crustaceans, crayfish, mixed species, cooked", "Finfish and Shellfish Products", 87, 17.5, 1.3, 0, 0),
        (175163, "Ostrich, ground, cooked", "Poultry Products", 175, 26.2, 7.1, 0, 0),
        (174295, "Pheasant, cooked, total edible", "Poultry Products", 239, 32.4, 12.1, 0, 0),
        (174300, "Quail, cooked, total edible", "Poultry Products", 234, 25.1, 14.1, 0, 0),
        (174310, "Bison, ground, cooked", "Lamb, Veal, and Game Products", 259, 26.4, 16.3, 0, 0),
        (174305, "Elk, ground, cooked", "Lamb, Veal, and Game Products", 169, 29.5, 5.1, 0, 0),
        (174315, "Rabbit, domesticated, cooked, stewed", "Lamb, Veal, and Game Products", 197, 29.1, 8.1, 0, 0),
        (174320, "Venison (deer), ground, cooked", "Lamb, Veal, and Game Products", 187, 26.5, 8.2, 0, 0),
        # More dairy products
        (173425, "Cheese, Asiago", "Dairy and Egg Products", 392, 32.6, 28.8, 0, 0),
        (173426, "Cheese, Havarti", "Dairy and Egg Products", 350, 23.2, 27.5, 3.7, 0),
        (173427, "Cheese, Limburger", "Dairy and Egg Products", 327, 20.1, 27.4, 0.5, 0),
        (173428, "Cheese, Muenster", "Dairy and Egg Products", 368, 23.4, 30.0, 1.1, 0),
        (173429, "Cheese, Neufchatel", "Dairy and Egg Products", 253, 10.0, 22.8, 3.0, 0),
        (173430, "Cheese, Romano", "Dairy and Egg Products", 387, 31.8, 26.9, 3.6, 0),
        (173431, "Cheese, string", "Dairy and Egg Products", 312, 22.2, 24.4, 1.0, 0),
        (170892, "Yogurt, Greek, plain, whole milk", "Dairy and Egg Products", 97, 9.0, 5.0, 3.9, 0),
        (170893, "Yogurt, vanilla, lowfat", "Dairy and Egg Products", 85, 4.9, 1.3, 13.8, 0),
        (170894, "Kefir, plain, lowfat", "Dairy and Egg Products", 43, 3.8, 1.0, 4.8, 0),
        # More grains and baked goods
        (168895, "Buckwheat groats, cooked", "Cereal Grains and Pasta", 92, 3.4, 0.6, 19.9, 2.7),
        (168898, "Farro, cooked", "Cereal Grains and Pasta", 140, 5.5, 1.0, 30.0, 3.5),
        (168900, "Amaranth grain, cooked", "Cereal Grains and Pasta", 102, 3.8, 1.6, 18.7, 2.1),
        (168903, "Teff, cooked", "Cereal Grains and Pasta", 101, 3.9, 0.6, 19.9, 2.8),
        (168052, "Croissant, butter", "Baked Products", 406, 8.2, 21.0, 45.8, 2.3),
        (168055, "Danish pastry, fruit", "Baked Products", 371, 5.7, 18.4, 45.4, 1.3),
        (168058, "Doughnut, cake-type, plain", "Baked Products", 421, 4.6, 22.9, 50.4, 1.3),
        (168060, "Scones, plain", "Baked Products", 362, 7.4, 14.5, 50.9, 1.8),
        (168063, "Brownie, commercially prepared", "Baked Products", 405, 5.1, 16.2, 64.2, 2.3),
        (168065, "Pita bread, white", "Baked Products", 275, 9.1, 1.2, 55.7, 2.2),
        (168067, "Naan bread", "Baked Products", 290, 9.4, 5.7, 50.5, 2.2),
        (168070, "Croutons, plain", "Baked Products", 407, 11.0, 6.6, 73.5, 5.0),
        (168072, "Graham crackers", "Baked Products", 422, 6.5, 10.0, 76.5, 2.1),
        (168075, "Pretzels, hard, plain, salted", "Snacks", 381, 9.2, 3.5, 79.2, 2.8),
        (168077, "Popcorn, air-popped", "Snacks", 387, 12.9, 4.5, 77.8, 14.5),
        (168080, "Granola bar, oat, fruit, nut", "Snacks", 444, 7.6, 18.6, 65.8, 4.7),
        (168083, "Rice cakes, brown rice, plain", "Snacks", 387, 8.0, 2.8, 81.5, 4.2),
        (168085, "Trail mix, with chocolate chips, salted nuts, seeds", "Snacks", 484, 12.8, 28.7, 49.8, 4.8),
        # More vegetables
        (170423, "Swiss chard, raw", "Vegetables and Vegetable Products", 19, 1.8, 0.2, 3.7, 1.6),
        (170424, "Watercress, raw", "Vegetables and Vegetable Products", 11, 2.3, 0.1, 1.3, 0.5),
        (170426, "Arugula, raw", "Vegetables and Vegetable Products", 25, 2.6, 0.7, 3.6, 1.6),
        (170428, "Endive, raw", "Vegetables and Vegetable Products", 17, 1.3, 0.2, 3.4, 3.1),
        (170430, "Radicchio, raw", "Vegetables and Vegetable Products", 23, 1.4, 0.3, 4.5, 0.9),
        (170432, "Romaine lettuce, raw", "Vegetables and Vegetable Products", 17, 1.2, 0.3, 3.3, 2.1),
        (170434, "Iceberg lettuce, raw", "Vegetables and Vegetable Products", 14, 0.9, 0.1, 3.0, 1.2),
        (170436, "Mixed greens, raw", "Vegetables and Vegetable Products", 20, 2.0, 0.3, 3.5, 2.0),
        (170442, "Corn on the cob, cooked", "Vegetables and Vegetable Products", 108, 3.3, 1.4, 23.5, 2.7),
        (170444, "Artichoke hearts, canned", "Vegetables and Vegetable Products", 33, 2.5, 0.3, 5.1, 3.2),
        (170446, "Hearts of palm, canned", "Vegetables and Vegetable Products", 28, 2.5, 0.6, 4.6, 2.4),
        (170450, "Jicama, raw", "Vegetables and Vegetable Products", 38, 0.7, 0.1, 8.8, 4.9),
        (170466, "Chives, raw", "Vegetables and Vegetable Products", 30, 3.3, 0.7, 4.4, 2.5),
        (170469, "Kohlrabi, raw", "Vegetables and Vegetable Products", 27, 1.7, 0.1, 6.2, 3.6),
        (170471, "Rutabaga, raw", "Vegetables and Vegetable Products", 38, 1.1, 0.2, 8.6, 2.3),
        (170473, "Taro root, raw", "Vegetables and Vegetable Products", 112, 1.5, 0.2, 26.5, 4.1),
        (170475, "Yam, raw", "Vegetables and Vegetable Products", 118, 1.5, 0.2, 27.9, 4.1),
        (170477, "Cassava, raw", "Vegetables and Vegetable Products", 160, 1.4, 0.3, 38.1, 1.8),
        (170479, "Daikon, raw", "Vegetables and Vegetable Products", 18, 0.6, 0.1, 4.1, 1.6),
        (170481, "Lotus root, raw", "Vegetables and Vegetable Products", 74, 2.6, 0.1, 17.2, 4.9),
        (170483, "Celeriac, raw", "Vegetables and Vegetable Products", 42, 1.5, 0.3, 9.2, 1.8),
        (170485, "Shallots, raw", "Vegetables and Vegetable Products", 72, 2.5, 0.1, 16.8, 3.2),
        # More spices/herbs
        (171336, "Spices, tarragon, dried", "Spices and Herbs", 295, 22.8, 7.2, 50.2, 7.4),
        (171337, "Spices, marjoram, dried", "Spices and Herbs", 271, 12.7, 7.0, 60.6, 40.3),
        (171338, "Spices, bay leaf", "Spices and Herbs", 313, 7.6, 8.4, 74.9, 26.3),
        (171339, "Spices, cardamom", "Spices and Herbs", 311, 10.8, 6.7, 68.5, 28.0),
        (171340, "Spices, caraway seed", "Spices and Herbs", 333, 19.8, 14.6, 49.9, 38.0),
        (171341, "Spices, celery seed", "Spices and Herbs", 392, 18.1, 25.3, 41.4, 11.8),
        (171342, "Spices, fennel seed", "Spices and Herbs", 345, 15.8, 14.9, 52.3, 39.8),
        (171343, "Spices, fenugreek seed", "Spices and Herbs", 323, 23.0, 6.4, 58.3, 24.6),
        (171344, "Spices, garlic granulated", "Spices and Herbs", 349, 16.6, 0.7, 77.6, 9.0),
        (172237, "Chives, freeze-dried", "Spices and Herbs", 311, 21.2, 3.5, 64.3, 26.2),
        (172238, "Lemongrass, raw", "Spices and Herbs", 99, 1.8, 0.5, 25.3, 0),
        # Sweeteners and syrups
        (171382, "Chocolate syrup", "Sweets", 279, 2.1, 1.0, 65.6, 2.4),
        (171383, "Pancake syrup", "Sweets", 260, 0, 0, 68.0, 0),
        (171384, "Agave nectar", "Sweets", 310, 0.1, 0.5, 76.4, 0.2),
        (171386, "Caramel topping", "Sweets", 275, 1.5, 2.3, 63.6, 0.5),
        (171387, "Butterscotch topping", "Sweets", 242, 0.6, 3.9, 56.2, 0),
        (167539, "Hard candy", "Sweets", 394, 0, 0.2, 98.0, 0),
        (167540, "Gummy bears", "Sweets", 320, 6.9, 0, 77.7, 0),
        (167541, "Licorice", "Sweets", 375, 3.4, 3.4, 82.5, 0),
        (167542, "Candy, fudge, chocolate", "Sweets", 411, 2.4, 10.4, 79.0, 1.5),
        # Prepared foods and condiments
        (174883, "Dressing, Caesar", "Fats and Oils", 438, 2.8, 44.8, 4.1, 0.2),
        (174884, "Dressing, blue cheese", "Fats and Oils", 418, 2.4, 43.5, 3.1, 0),
        (174885, "Dressing, French", "Fats and Oils", 322, 0.6, 24.6, 24.5, 0.2),
        (174886, "Dressing, Thousand Island", "Fats and Oils", 290, 1.1, 27.6, 9.9, 0.4),
        (174887, "Dressing, vinaigrette", "Fats and Oils", 267, 0.3, 25.2, 9.8, 0.2),
        (174888, "Dressing, honey mustard", "Fats and Oils", 293, 0.8, 19.6, 29.1, 0.5),
        (168580, "Steak sauce, tomato based", "Soups, Sauces, and Gravies", 100, 0.8, 0.1, 24.1, 0.7),
        (168582, "Cocktail sauce, ready to serve", "Soups, Sauces, and Gravies", 117, 1.0, 0.4, 27.8, 1.0),
        (168584, "Tartar sauce", "Soups, Sauces, and Gravies", 329, 0.6, 32.5, 7.9, 0.4),
        (168586, "Alfredo sauce, ready to serve", "Soups, Sauces, and Gravies", 149, 3.5, 11.6, 7.0, 0.2),
        (168588, "Marinara sauce, ready to serve", "Soups, Sauces, and Gravies", 55, 1.5, 1.5, 9.6, 2.0),
        (168590, "Enchilada sauce, red, canned", "Soups, Sauces, and Gravies", 46, 1.4, 0.6, 8.3, 1.5),
        (168592, "Curry sauce, ready to serve", "Soups, Sauces, and Gravies", 110, 1.8, 7.5, 9.2, 1.0),
        (168557, "Soup, minestrone, canned, condensed", "Soups, Sauces, and Gravies", 53, 2.2, 0.9, 9.6, 1.5),
        (168559, "Soup, clam chowder, canned, condensed", "Soups, Sauces, and Gravies", 58, 2.3, 1.3, 8.7, 0.5),
        (168563, "Soup, vegetable beef, canned, condensed", "Soups, Sauces, and Gravies", 49, 2.6, 0.7, 8.2, 0.8),
        (168565, "Soup, onion, canned, condensed", "Soups, Sauces, and Gravies", 27, 0.8, 0.7, 4.5, 0.3),
        # Canned meats/fish
        (174585, "Chicken, canned, with broth", "Poultry Products", 127, 23.0, 3.2, 0, 0),
        (174590, "Tuna, canned, white, in water", "Finfish and Shellfish Products", 128, 28.0, 1.0, 0, 0),
        (174595, "Sardines, canned, in mustard sauce", "Finfish and Shellfish Products", 213, 24.0, 12.0, 1.0, 0),
        (174600, "Salmon, canned, pink", "Finfish and Shellfish Products", 136, 19.8, 6.1, 0, 0),
        (174605, "Corned beef, canned", "Beef Products", 250, 27.1, 14.9, 0, 0),
        (174610, "Spam, pork with ham", "Sausages and Luncheon Meats", 315, 13.0, 26.0, 7.0, 0),
        # Baby foods and breakfast items
        (167552, "Granola, homemade", "Cereal Grains and Pasta", 471, 10.5, 21.3, 63.3, 6.8),
        (167555, "Cereal, corn flakes", "Cereal Grains and Pasta", 357, 7.5, 0.4, 84.1, 3.3),
        (167558, "Cereal, oatmeal, instant, prepared", "Cereal Grains and Pasta", 71, 2.5, 1.5, 12.3, 1.8),
        (167560, "Toaster pastries, frosted, fruit", "Baked Products", 378, 4.4, 9.3, 70.0, 1.0),
        # More condiments
        (174890, "Relish, pickle, sweet", "Vegetables and Vegetable Products", 130, 0.4, 0.7, 32.5, 0.9),
        (174892, "Horseradish, prepared", "Spices and Herbs", 48, 1.2, 0.7, 11.3, 3.3),
        (174894, "Chutney, mango", "Sweets", 238, 0.3, 0.2, 62.0, 0.8),
        (174896, "Wasabi, prepared", "Spices and Herbs", 292, 3.0, 10.3, 46.1, 6.1),
        (174898, "Chipotle peppers in adobo sauce", "Vegetables and Vegetable Products", 68, 1.5, 1.8, 11.5, 5.0),
        # Tofu/soy varieties
        (168324, "Tofu, soft, prepared with calcium sulfate", "Legumes and Legume Products", 55, 4.8, 2.7, 1.8, 0.4),
        (168326, "Tofu, silken", "Legumes and Legume Products", 52, 4.8, 2.5, 2.3, 0.2),
        (168328, "Tempeh", "Legumes and Legume Products", 192, 20.3, 10.8, 7.6, 0),
        (168330, "Seitan (wheat gluten), cooked", "Legumes and Legume Products", 370, 75.2, 1.9, 14.0, 0.6),
        (174270, "Soy milk, unsweetened", "Legumes and Legume Products", 33, 2.9, 1.8, 1.6, 0.4),
        (174272, "Almond milk, unsweetened", "Beverages", 15, 0.6, 1.1, 0.3, 0.2),
        (174274, "Oat milk, original", "Beverages", 47, 1.0, 1.5, 7.0, 0.8),
        (174276, "Coconut milk beverage, unsweetened", "Beverages", 25, 0.2, 2.3, 1.0, 0),
        (174278, "Rice milk, unsweetened", "Beverages", 47, 0.3, 1.0, 9.2, 0.3),
        # Final batch to reach 500+ unique entries
        (174280, "Hemp milk, unsweetened", "Beverages", 46, 1.3, 3.0, 1.3, 0),
        (168587, "Curry paste, green", "Soups, Sauces, and Gravies", 109, 2.4, 5.1, 13.8, 2.0),
        (168589, "Curry paste, red", "Soups, Sauces, and Gravies", 136, 3.5, 6.8, 15.4, 3.0),
        (168591, "Chili sauce, bottled", "Soups, Sauces, and Gravies", 95, 2.1, 0.2, 22.3, 3.0),
        (168593, "Sweet chili sauce", "Soups, Sauces, and Gravies", 195, 0.4, 0.3, 47.0, 0.5),
        (174282, "Tofu, fried, prepared", "Legumes and Legume Products", 271, 17.2, 20.2, 3.9, 0.5),
        (174284, "Miso paste, white", "Legumes and Legume Products", 199, 11.7, 6.0, 26.5, 5.4),
        (174286, "Natto, fermented soybeans", "Legumes and Legume Products", 211, 17.7, 11.0, 14.4, 5.4),
        (168500, "Pasta sauce, spaghetti/marinara", "Soups, Sauces, and Gravies", 65, 1.8, 2.5, 9.0, 2.3),
        (168502, "Peanut sauce, ready to serve", "Soups, Sauces, and Gravies", 215, 6.2, 13.0, 18.5, 1.5),
        (168504, "Tzatziki sauce", "Soups, Sauces, and Gravies", 55, 3.5, 2.5, 4.5, 0.3),
        (168506, "Guacamole, ready to serve", "Vegetables and Vegetable Products", 140, 1.5, 12.0, 7.0, 5.0),
        (168508, "Chimichurri sauce", "Soups, Sauces, and Gravies", 250, 1.5, 25.0, 3.5, 1.0),
        (168510, "Raita, cucumber yogurt", "Dairy and Egg Products", 50, 3.5, 1.5, 5.0, 0.3),
        (168512, "Harissa paste", "Spices and Herbs", 75, 3.0, 1.5, 12.0, 4.5),
        (168514, "Sambal oelek", "Spices and Herbs", 35, 1.5, 0.5, 6.5, 2.5),
        (168516, "Ginger paste", "Spices and Herbs", 80, 1.8, 0.8, 17.8, 2.0),
        (168518, "Garlic paste", "Spices and Herbs", 149, 6.4, 0.5, 33.1, 2.1),
        (168520, "Lemon curd", "Sweets", 267, 3.8, 7.5, 46.0, 0.2),
        (168522, "Dulce de leche", "Sweets", 315, 7.0, 8.0, 55.0, 0),
        (168524, "Nutella (hazelnut spread)", "Sweets", 539, 6.3, 30.9, 57.5, 3.4),
        (168526, "Mirin (sweet rice wine)", "Beverages", 229, 0.1, 0, 45.8, 0),
        (168528, "Rice wine vinegar", "Spices and Herbs", 18, 0, 0, 0.7, 0),
        (168530, "Shaoxing wine", "Beverages", 95, 1.5, 0, 5.0, 0),
        (168532, "Coconut aminos", "Soups, Sauces, and Gravies", 100, 0, 0, 25.0, 0),
        (168534, "Liquid smoke", "Spices and Herbs", 0, 0, 0, 0, 0),
        (168536, "Tomato puree, canned", "Vegetables and Vegetable Products", 38, 1.6, 0.2, 8.9, 2.3),
        (168538, "Roasted red peppers, jarred", "Vegetables and Vegetable Products", 28, 0.9, 0.2, 5.5, 1.5),
        (168540, "Sun-dried tomatoes in oil", "Vegetables and Vegetable Products", 213, 5.1, 14.1, 23.3, 5.5),
        (168542, "Artichoke hearts, marinated", "Vegetables and Vegetable Products", 60, 1.0, 4.5, 4.0, 2.0),
        (168544, "Roasted garlic", "Vegetables and Vegetable Products", 149, 6.4, 0.5, 33.1, 2.1),
        (168547, "Vegetable broth", "Soups, Sauces, and Gravies", 6, 0.3, 0.1, 1.1, 0.1),
        (168549, "Bone broth, chicken", "Soups, Sauces, and Gravies", 17, 3.5, 0.2, 0, 0),
        (168551, "Dashi stock", "Soups, Sauces, and Gravies", 5, 0.5, 0, 0.5, 0),
    ]

    for row in USDA_FOODS:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO usda_food (fdc_id, description, food_group, calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g, fiber_per_100g) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        except sqlite3.IntegrityError:
            pass


# Ingredient name -> USDA description mapping for fuzzy linking
INGREDIENT_USDA_MAP = {
    "chicken": "Chicken, broilers or fryers, breast",
    "chicken breast": "Chicken, broilers or fryers, breast",
    "chicken thigh": "Chicken, broilers or fryers, thigh",
    "turkey": "Turkey, all classes, breast",
    "beef": "Beef, ground, 85% lean",
    "ground beef": "Beef, ground, 85% lean",
    "steak": "Beef, top sirloin",
    "pork": "Pork, fresh, loin",
    "pork belly": "Pork, fresh, belly",
    "pork loin": "Pork, fresh, loin",
    "ground pork": "Pork, fresh, loin",
    "ground turkey": "Turkey, all classes, breast",
    "ground lamb": "Lamb, domestic, leg",
    "lamb": "Lamb, domestic, leg",
    "duck": "Duck, domesticated",
    "salmon": "Fish, salmon, Atlantic",
    "tuna": "Fish, tuna, light",
    "cod": "Fish, cod, Atlantic",
    "white fish": "Fish, cod, Atlantic",
    "tilapia": "Fish, tilapia",
    "shrimp": "Crustaceans, shrimp",
    "crab": "Crustaceans, crab",
    "lobster": "Lobster, northern",
    "squid": "Mollusks, squid",
    "scallops": "Mollusks, scallop",
    "mussels": "Mollusks, mussel",
    "clams": "Mollusks, clam",
    "egg": "Egg, whole, cooked",
    "eggs": "Egg, whole, cooked",
    "tofu": "Tofu, firm",
    "bacon": "Bacon, pork, cooked",
    "sausage": "Sausage, pork, cooked",
    "ham": "Ham, sliced",
    "veal": "Veal, loin",
    "rice": "Rice, white, long-grain",
    "basmati rice": "Rice, white, long-grain",
    "sushi rice": "Rice, white, long-grain",
    "brown rice": "Rice, brown, long-grain",
    "pasta": "Pasta, cooked, enriched",
    "spaghetti": "Pasta, cooked, enriched",
    "noodles": "Noodles, egg, cooked",
    "egg noodles": "Noodles, egg, cooked",
    "udon noodles": "Noodles, egg, cooked",
    "soba noodles": "Noodles, egg, cooked",
    "ramen noodles": "Noodles, egg, cooked",
    "rice noodles": "Rice noodles, cooked",
    "vermicelli": "Rice noodles, cooked",
    "glass noodles": "Rice noodles, cooked",
    "bread": "Bread, white",
    "flour": "Wheat flour, white",
    "potato": "Potatoes, boiled",
    "sweet potato": "Sweet potato, cooked",
    "corn": "Corn, sweet, yellow",
    "corn kernels": "Corn, sweet, yellow",
    "quinoa": "Quinoa, cooked",
    "couscous": "Couscous, cooked",
    "tortilla": "Tortilla, ready-to-bake",
    "pizza dough": "Pizza dough",
    "breadcrumbs": "Bread, white",
    "panko breadcrumbs": "Bread, white",
    "polenta": "Polenta, cooked",
    "barley": "Barley, pearled",
    "oats": "Oats, regular",
    "onion": "Onions, raw",
    "red onion": "Onions, raw",
    "garlic": "Garlic, raw",
    "tomato": "Tomatoes, red, ripe",
    "bell pepper": "Peppers, sweet, red",
    "carrot": "Carrots, raw",
    "celery": "Celery, raw",
    "broccoli": "Broccoli, raw",
    "spinach": "Spinach, raw",
    "cabbage": "Cabbage, raw",
    "mushrooms": "Mushrooms, white, raw",
    "zucchini": "Squash, summer, zucchini",
    "eggplant": "Eggplant, raw",
    "cucumber": "Cucumber, with peel",
    "lettuce": "Lettuce, green leaf",
    "kale": "Kale, raw",
    "bean sprouts": "Bean sprouts (mung)",
    "bok choy": "Bok choy, raw",
    "spring onion": "Scallions",
    "scallion": "Scallions",
    "ginger": "Ginger root, raw",
    "chili": "Peppers, chili",
    "peas": "Peas, green",
    "green beans": "Green beans (snap)",
    "asparagus": "Asparagus, raw",
    "leek": "Leek, raw",
    "bamboo shoots": "Bamboo shoots",
    "avocado": "Avocado, raw",
    "cauliflower": "Cauliflower, raw",
    "brussels sprouts": "Brussels sprouts",
    "beet": "Beets, raw",
    "pumpkin": "Pumpkin, raw",
    "butternut squash": "Butternut squash",
    "radish": "Radishes, raw",
    "turnip": "Turnips, raw",
    "artichoke": "Artichoke, raw",
    "fennel": "Fennel, bulb",
    "okra": "Okra, raw",
    "kimchi": "Cabbage, raw",
    "plantain": "Plantains, raw",
    "olive oil": "Oil, olive",
    "cooking oil": "Oil, vegetable",
    "vegetable oil": "Oil, vegetable",
    "sesame oil": "Oil, sesame",
    "coconut oil": "Oil, coconut",
    "chili oil": "Oil, sesame",
    "cheese": "Cheese, cheddar",
    "cheddar": "Cheese, cheddar",
    "mozzarella": "Cheese, mozzarella",
    "parmesan": "Cheese, parmesan",
    "feta": "Cheese, feta",
    "ricotta": "Cheese, ricotta",
    "cream cheese": "Cheese, cream",
    "goat cheese": "Cheese, goat",
    "gruyere": "Cheese, gruyere",
    "brie": "Cheese, brie",
    "gouda": "Cheese, gouda",
    "butter": "Butter, salted",
    "cream": "Cream, heavy",
    "heavy cream": "Cream, heavy",
    "milk": "Milk, whole",
    "yogurt": "Yogurt, plain",
    "sour cream": "Sour cream",
    "coconut milk": "Milk, coconut",
    "coconut cream": "Coconut cream",
    "ghee": "Ghee, clarified",
    "buttermilk": "Milk, buttermilk",
    "soy sauce": "Soy sauce",
    "fish sauce": "Fish sauce",
    "oyster sauce": "Sauce, oyster",
    "hoisin sauce": "Sauce, hoisin",
    "tomato sauce": "Tomato sauce, canned",
    "tomato paste": "Tomato paste",
    "ketchup": "Ketchup",
    "mustard": "Mustard, prepared",
    "mayonnaise": "Mayonnaise",
    "honey": "Honey",
    "maple syrup": "Maple syrup",
    "vinegar": "Vinegar, cider",
    "rice vinegar": "Vinegar, cider",
    "balsamic vinegar": "Vinegar, balsamic",
    "lemon juice": "Lemon juice",
    "lime juice": "Lime juice",
    "teriyaki sauce": "Sauce, teriyaki",
    "sriracha": "Sriracha sauce",
    "hot sauce": "Sauce, hot pepper",
    "worcestershire sauce": "Sauce, Worcestershire",
    "bbq sauce": "Sauce, barbecue",
    "pesto": "Pesto sauce",
    "marinara sauce": "Tomato sauce, canned",
    "salsa": "Salsa, ready to serve",
    "tahini": "Tahini",
    "hummus": "Hummus",
    "miso paste": "Miso",
    "gochujang": "Sauce, hot pepper",
    "tamarind paste": "Tamarind, raw",
    "sugar": "Sugar, granulated",
    "brown sugar": "Sugar, brown",
    "salt": "Salt, table",
    "pepper": "Spices, pepper, black",
    "black pepper": "Spices, pepper, black",
    "paprika": "Spices, paprika",
    "cumin": "Spices, cumin seed",
    "coriander": "Spices, coriander seed",
    "turmeric": "Spices, turmeric",
    "cinnamon": "Spices, cinnamon",
    "oregano": "Spices, oregano",
    "basil": "Basil, fresh",
    "thyme": "Spices, thyme",
    "rosemary": "Rosemary, fresh",
    "parsley": "Parsley, fresh",
    "cilantro": "Cilantro",
    "dill": "Dill, fresh",
    "garam masala": "Spices, garam masala",
    "curry paste": "Sauce, hot pepper",
    "green curry paste": "Sauce, hot pepper",
    "red curry paste": "Sauce, hot pepper",
    "yellow curry paste": "Sauce, hot pepper",
    "cornstarch": "Cornstarch",
    "baking powder": "Leavening agents, baking powder",
    "baking soda": "Leavening agents, baking soda",
    "cocoa": "Cocoa, dry powder",
    "chocolate": "Chocolate, dark",
    "vanilla": "Vanilla extract",
    "peanuts": "Peanuts, all types",
    "almonds": "Almonds",
    "walnuts": "Walnuts, English",
    "cashews": "Cashew nuts",
    "pecans": "Pecans",
    "pine nuts": "Pine nuts",
    "sesame seeds": "Sesame seeds",
    "coconut": "Coconut meat, raw",
    "chickpeas": "Chickpeas (garbanzo beans)",
    "black beans": "Beans, black",
    "kidney beans": "Beans, kidney",
    "lentils": "Lentils, mature seeds",
    "edamame": "Edamame, frozen",
    "chicken broth": "Broth, chicken",
    "beef broth": "Broth, beef",
    "nori": "Seaweed, nori",
    "olives": "Olives, ripe",
    "capers": "Capers, canned",
    "water chestnuts": "Water chestnuts",
    "peanut sauce": "Peanuts, all types",
    "peanut butter": "Peanut butter",
    "water": "Water, tap",
    "margarine": "Margarine",
    "oleo": "Margarine",
    "shortening": "Shortening",
    "vanilla extract": "Vanilla extract",
    "vanilla": "Vanilla extract",
    "green pepper": "Peppers, sweet",
    "red pepper": "Peppers, sweet",
    "jalapeno": "Peppers, jalapeno",
    "cream of mushroom soup": "Soup, cream of mushroom",
    "cream of chicken soup": "Soup, cream of mushroom",
    "chicken broth": "Broth, chicken",
    "beef broth": "Broth, beef",
    "broth": "Broth, chicken",
    "stock": "Broth, chicken",
    "chicken stock": "Broth, chicken",
    "beef stock": "Broth, beef",
    "brown sugar": "Sugar, brown",
    "white sugar": "Sugar, granulated",
    "powdered sugar": "Sugar, granulated",
    "confectioners sugar": "Sugar, granulated",
    "baking soda": "Leavening agents, baking soda",
    "soda": "Leavening agents, baking soda",
    "nuts": "Walnuts",
    "walnuts": "Walnuts",
    "pecans": "Pecans",
    "marshmallows": "Marshmallows",
    "marshmallow": "Marshmallows",
    "raisins": "Raisins",
    "cranberries": "Cranberries",
    "cottage cheese": "Cottage cheese",
    "swiss cheese": "Cheese, swiss",
    "monterey jack": "Cheese, Monterey",
    "pepper jack": "Cheese, pepper jack",
    "colby cheese": "Cheese, Colby",
    "provolone": "Cheese, provolone",
    "blue cheese": "Cheese, blue",
    "evaporated milk": "Milk, evaporated",
    "condensed milk": "Milk, condensed",
    "half and half": "Half-and-half",
    "whipped cream": "Whipped cream",
    "cool whip": "Whipped cream",
    "whipping cream": "Cream, heavy",
    "cornmeal": "Cornmeal",
    "whole wheat flour": "Wheat flour, whole-grain",
    "self-rising flour": "Wheat flour, white",
    "all-purpose flour": "Wheat flour, white",
    "oil": "Oil, vegetable",
    "canola oil": "Oil, vegetable",
    "peanut oil": "Oil, peanut",
    "sunflower oil": "Oil, sunflower",
    "corn oil": "Oil, corn",
    "garlic powder": "Garlic powder",
    "onion powder": "Onion powder",
    "chili powder": "Spices, chili powder",
    "cayenne pepper": "Spices, cayenne pepper",
    "cayenne": "Spices, cayenne pepper",
    "curry powder": "Spices, curry powder",
    "allspice": "Spices, allspice",
    "cloves": "Spices, cloves",
    "nutmeg": "Spices, nutmeg",
    "sage": "Spices, sage",
    "mint": "Mint, fresh",
    "bay leaf": "Basil, dried",
    "italian seasoning": "Basil, dried",
    "pizza sauce": "Sauce, pizza",
    "gravy": "Gravy, beef",
    "ranch dressing": "Dressing, ranch",
    "italian dressing": "Dressing, Italian",
    "hot dog": "Hot dog",
    "pepperoni": "Pepperoni",
    "salami": "Salami",
    "gelatin": "Gelatin, unflavored",
    "jello": "Gelatin desserts",
    "pudding": "Pudding, vanilla",
    "chocolate chips": "Chocolate chips",
    "cocoa powder": "Cocoa, dry powder",
    "wine": "Wine, red",
    "red wine": "Wine, red",
    "white wine": "Wine, white",
    "beer": "Beer, regular",
    "orange juice": "Juice, orange",
    "apple juice": "Juice, apple",
    "lard": "Lard",
    "crackers": "Crackers, saltines",
    "ice cream": "Ice cream",
    "greek yogurt": "Yogurt, Greek",
    "tortilla chips": "Tortilla chips",
    "corn syrup": "Corn syrup",
    "jam": "Jam, strawberry",
    "jelly": "Jelly, grape",
    "preserves": "Jam, strawberry",
    "peanuts": "Peanuts, all types",
    "applesauce": "Applesauce",
    "canned tomatoes": "Tomatoes, canned",
    "diced tomatoes": "Tomatoes, canned",
    "crushed tomatoes": "Tomatoes, crushed",
    "stewed tomatoes": "Tomatoes, canned, stewed",
    "sun dried tomatoes": "Tomatoes, sun-dried",
    "cream corn": "Corn, canned, cream style",
    "sauerkraut": "Sauerkraut",
    "green onion": "Onions, spring or scallions",
    "green onions": "Onions, spring or scallions",
    "shallots": "Onions, raw",
    "pimento": "Peppers, sweet",
    "rotel": "Tomatoes, canned",
    "velveeta": "Cheese, cheddar",
    "miracle whip": "Mayonnaise",
    "pickle": "Pickles, cucumber",
    "pickles": "Pickles, cucumber",
    "relish": "Pickles, cucumber",
    "ground chicken": "Chicken, broilers or fryers, ground",
    "chicken wings": "Chicken, broilers or fryers, wing",
    "chicken drumsticks": "Chicken, broilers or fryers, drumstick",
    "pork chops": "Pork, fresh, loin",
    "pork ribs": "Pork, fresh, loin",
    "pork shoulder": "Pork, fresh, loin",
    "beef roast": "Beef, chuck, pot roast",
    "chuck roast": "Beef, chuck, pot roast",
    "pot roast": "Beef, chuck, pot roast",
    "brisket": "Beef, brisket",
    "flank steak": "Beef, flank",
    "sirloin": "Beef, top sirloin",
    "rib eye": "Beef, rib",
    "filet": "Beef, top sirloin",
    "meatball": "Beef, ground",
    "meatballs": "Beef, ground",
    "halibut": "Fish, halibut",
    "catfish": "Fish, catfish",
    "sardines": "Fish, sardine",
    "anchovies": "Fish, anchovies",
    "trout": "Fish, trout",
    "swordfish": "Fish, swordfish",
    "white beans": "Beans, white",
    "lima beans": "Beans, lima",
    "navy beans": "Beans, navy",
    "great northern beans": "Beans, great northern",
    "split peas": "Split peas",
    "black eyed peas": "Split peas",
    "sunflower seeds": "Sunflower seed kernels",
    "pumpkin seeds": "Pumpkin seeds",
    "poppy seeds": "Poppy seeds",
    "flaxseed": "Flaxseed",
    "chia seeds": "Chia seeds",
    "hazelnuts": "Hazelnuts",
    "macadamia nuts": "Macadamia nuts",
    "chestnuts": "Chestnuts",
    "brazil nuts": "Brazil nuts",
    "pancetta": "Bacon, pork",
    "prosciutto": "Ham, sliced",
    "tempeh": "Tofu, firm",
    "seitan": "Tofu, firm",
    "egg whites": "Egg, white",
    "egg yolks": "Egg, yolk",
    "egg white": "Egg, white",
    "egg yolk": "Egg, yolk",
    "butter or margarine": "Butter, salted",
    "margarine or butter": "Butter, salted",
    "butter or oleo": "Butter, salted",
    "cheddar cheese": "Cheese, cheddar",
    "sharp cheddar cheese": "Cheese, cheddar",
    "sharp cheese": "Cheese, cheddar",
    "mild cheddar cheese": "Cheese, cheddar",
    "parmesan cheese": "Cheese, parmesan",
    "mozzarella cheese": "Cheese, mozzarella",
    "swiss cheese": "Cheese, swiss",
    "cream cheese": "Cheese, cream",
    "velveeta cheese": "Cheese, cheddar",
    "american cheese": "Cheese, cheddar",
    "jack cheese": "Cheese, Monterey",
    "pkg cream cheese": "Cheese, cream",
    "boiling water": "Water, tap",
    "hot water": "Water, tap",
    "cold water": "Water, tap",
    "warm water": "Water, tap",
    "ice water": "Water, tap",
    "garlic salt": "Salt, table",
    "seasoned salt": "Salt, table",
    "celery salt": "Salt, table",
    "kosher salt": "Salt, table",
    "sea salt": "Salt, table",
    "sifted flour": "Wheat flour, white",
    "plain flour": "Wheat flour, white",
    "bread flour": "Wheat flour, white",
    "cake flour": "Wheat flour, white",
    "salad oil": "Oil, vegetable",
    "dry mustard": "Mustard, prepared",
    "crisco": "Shortening",
    "beaten eggs": "Egg, whole, cooked",
    "beaten egg": "Egg, whole, cooked",
    "bread crumbs": "Bread, white",
    "green peppers": "Peppers, sweet",
    "green bell pepper": "Peppers, sweet",
    "red bell pepper": "Peppers, sweet",
    "hamburger": "Beef, ground",
    "ground meat": "Beef, ground",
    "mushroom soup": "Soup, cream of mushroom",
    "pineapple": "Pineapple",
    "can pineapple": "Pineapple",
    "catsup": "Ketchup",
    "parsley flakes": "Parsley, fresh",
    "packed brown sugar": "Sugar, brown",
    "light brown sugar": "Sugar, brown",
    "dark brown sugar": "Sugar, brown",
    "cream of celery soup": "Soup, cream of mushroom",
    "cream of chicken soup": "Soup, cream of mushroom",
    "onion soup mix": "Onion powder",
    "taco seasoning": "Spices, chili powder",
    "seasoning salt": "Salt, table",
    "lemon zest": "Lemons",
    "orange zest": "Oranges",
    "lime zest": "Limes",
    "lemon rind": "Lemons",
    "lemon peel": "Lemons",
    "wax beans": "Green beans (snap)",
    "string beans": "Green beans (snap)",
    "kidney beans": "Beans, kidney",
    "pinto beans": "Beans, pinto",
    "refried beans": "Beans, pinto",
    "baked beans": "Beans, navy",
    "cannellini beans": "Beans, white",
    "garbanzo beans": "Chickpeas (garbanzo beans)",
    "chick peas": "Chickpeas (garbanzo beans)",
    "oreo": "Cookies, chocolate chip",
    "graham cracker crumbs": "Crackers, saltines",
    "graham crackers": "Crackers, saltines",
    "cracker crumbs": "Crackers, saltines",
    "bisquick": "Wheat flour, white",
    "jiffy mix": "Cornmeal",
    "cool whip": "Whipped cream",
    "non-dairy creamer": "Half-and-half",
    "coffee creamer": "Half-and-half",
    "seasoning": "Salt, table",
    "italian seasoning mix": "Basil, dried",
    "poultry seasoning": "Spices, sage",
    "old bay": "Spices, paprika",
    "paprika": "Spices, paprika",
    "red pepper flakes": "Spices, cayenne pepper",
    "crushed red pepper": "Spices, cayenne pepper",
}


def _link_ingredients_to_usda(conn: sqlite3.Connection) -> int:
    """Link recipe_ingredient rows to usda_food entries by name matching.

    Builds an ingredient_name -> fdc_id mapping first, then batch-updates
    all matching rows in a single pass for performance.
    """
    cursor = conn.cursor()

    # Build fast lookups from USDA descriptions
    cursor.execute("SELECT fdc_id, description FROM usda_food")
    usda_rows = cursor.fetchall()

    # desc_lower -> fdc_id
    usda_by_desc = {}
    # first_word_group (before comma) -> fdc_id
    usda_by_first = {}
    for fdc_id_val, desc in usda_rows:
        dl = desc.lower()
        usda_by_desc[dl] = fdc_id_val
        first = dl.split(",")[0].strip()
        usda_by_first[first] = fdc_id_val

    # Pre-resolve INGREDIENT_USDA_MAP to fdc_ids for O(1) lookup
    map_to_fdc = {}
    for ing_key, usda_target in INGREDIENT_USDA_MAP.items():
        target_lower = usda_target.lower()
        fdc_id_val = None
        for dl, fid in usda_by_desc.items():
            if target_lower in dl or dl in target_lower:
                fdc_id_val = fid
                break
        if not fdc_id_val:
            for fl, fid in usda_by_first.items():
                if target_lower in fl or fl in target_lower:
                    fdc_id_val = fid
                    break
        if fdc_id_val:
            map_to_fdc[ing_key] = fdc_id_val

    # Build ingredient -> fdc_id mapping
    cursor.execute(
        "SELECT DISTINCT ingredient_name FROM recipe_ingredient WHERE usda_fdc_id IS NULL"
    )
    unlinked = [row[0] for row in cursor.fetchall()]

    # Pre-compute the mapping (all in memory, fast O(1) lookups)
    ing_to_fdc = {}
    for ing_name in unlinked:
        fdc_id = None
        name_lower = ing_name.lower().strip()

        # Strategy 1: Direct map (O(1))
        if name_lower in map_to_fdc:
            fdc_id = map_to_fdc[name_lower]

        # Strategy 2: "X or Y" patterns
        if not fdc_id and " or " in name_lower:
            first_part = name_lower.split(" or ")[0].strip()
            if first_part in map_to_fdc:
                fdc_id = map_to_fdc[first_part]

        # Strategy 3: Trailing "cheese" modifier
        if not fdc_id and name_lower.endswith(" cheese"):
            base = name_lower[:-7].strip()
            if base in map_to_fdc:
                fdc_id = map_to_fdc[base]
            else:
                # Try matching "cheese, X" in USDA
                for dl, fid in usda_by_desc.items():
                    if "cheese" in dl and base in dl:
                        fdc_id = fid
                        break

        # Strategy 4: Exact match on USDA first-word group
        if not fdc_id:
            if name_lower in usda_by_first:
                fdc_id = usda_by_first[name_lower]

        # Strategy 5: Individual words match against map keys
        if not fdc_id:
            words = name_lower.split()
            for w in words:
                if len(w) >= 3 and w in map_to_fdc:
                    fdc_id = map_to_fdc[w]
                    break

        if fdc_id:
            ing_to_fdc[ing_name] = fdc_id

    # Create temporary index for fast ingredient_name lookups during linking
    print("  Creating temporary index for ingredient linking...")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tmp_ri_ingname ON recipe_ingredient(ingredient_name)"
    )
    conn.commit()

    # Batch update using executemany for speed
    linked = 0
    updates = [(fdc_id, ing_name) for ing_name, fdc_id in ing_to_fdc.items()]

    # Process in batches of 1000
    for i in range(0, len(updates), 1000):
        batch = updates[i:i + 1000]
        cursor.executemany(
            "UPDATE recipe_ingredient SET usda_fdc_id = ? WHERE ingredient_name = ? AND usda_fdc_id IS NULL",
            batch,
        )
        linked += cursor.rowcount
        conn.commit()

    # Drop temporary index (not needed at runtime)
    cursor.execute("DROP INDEX IF EXISTS idx_tmp_ri_ingname")
    conn.commit()

    print(f"  Matched {len(ing_to_fdc)} of {len(unlinked)} unique ingredient names to USDA")
    return linked


def compute_symspell_deletes(conn: sqlite3.Connection, max_edit_distance: int = 1) -> int:
    """
    Pre-compute SymSpell delete variants for all dish canonical names and aliases.

    Uses per-word SymSpell at edit distance 1: for each word in the dish name,
    generate all single-character deletions. This covers ~80% of common typos
    and is efficient (O(n) variants per word of length n).

    Edit distance 2 is not used because it generates O(n^2) variants per word,
    which is too expensive for 8K+ dishes.
    """
    cursor = conn.cursor()

    def _generate_deletes(name: str) -> set:
        """Generate delete variants for a name."""
        deletes = set()
        words = name.split()
        for word in words:
            if len(word) >= 2:
                for i in range(len(word)):
                    d = word[:i] + word[i + 1:]
                    deletes.add(d)
        # Also add the full name with single char deletions (for space-free matching)
        name_nospace = name.replace(" ", "")
        if len(name_nospace) >= 3:
            for i in range(len(name_nospace)):
                d = name_nospace[:i] + name_nospace[i + 1:]
                deletes.add(d)
        return deletes

    # Phase 1: Canonical dish names
    cursor.execute("SELECT id, canonical_name FROM dish")
    dishes = cursor.fetchall()

    total_inserts = 0
    batch = []

    for dish_id, name in dishes:
        for d in _generate_deletes(name):
            batch.append((dish_id, d))

        if len(batch) >= 50000:
            cursor.executemany(
                "INSERT INTO symspell_deletes (dish_id, delete_variant) VALUES (?, ?)",
                batch,
            )
            total_inserts += len(batch)
            batch = []

    if batch:
        cursor.executemany(
            "INSERT INTO symspell_deletes (dish_id, delete_variant) VALUES (?, ?)",
            batch,
        )
        total_inserts += len(batch)
        batch = []

    conn.commit()
    print(f"  SymSpell canonical: {total_inserts:,} entries for {len(dishes)} dishes")

    # Phase 2: Dish aliases
    cursor.execute("SELECT dish_id, alias FROM dish_alias")
    aliases = cursor.fetchall()
    alias_inserts = 0

    for dish_id, alias_text in aliases:
        alias_lower = alias_text.lower().strip()
        if not alias_lower or len(alias_lower) < 2:
            continue
        for d in _generate_deletes(alias_lower):
            batch.append((dish_id, d))

        if len(batch) >= 50000:
            cursor.executemany(
                "INSERT INTO symspell_deletes (dish_id, delete_variant) VALUES (?, ?)",
                batch,
            )
            alias_inserts += len(batch)
            batch = []

    if batch:
        cursor.executemany(
            "INSERT INTO symspell_deletes (dish_id, delete_variant) VALUES (?, ?)",
            batch,
        )
        alias_inserts += len(batch)

    conn.commit()
    total_inserts += alias_inserts
    print(f"  SymSpell aliases:   {alias_inserts:,} entries for {len(aliases)} aliases")
    print(f"  SymSpell total:     {total_inserts:,} entries")
    return total_inserts


def compute_dish_nutrition(conn: sqlite3.Connection) -> int:
    """
    Compute avg nutrition per serving for each dish by aggregating
    recipe ingredients linked to USDA nutrition data.
    """
    cursor = conn.cursor()

    # For each dish, find its canonical recipe, sum ingredient nutrition
    cursor.execute("""
        SELECT d.id, r.id as recipe_id, r.total_weight_grams, r.servings
        FROM dish d
        JOIN recipe r ON r.dish_id = d.id AND r.is_canonical = 1
    """)
    dish_recipes = cursor.fetchall()

    updated = 0
    for dish_id, recipe_id, total_weight, servings in dish_recipes:
        servings = servings or 1

        cursor.execute("""
            SELECT ri.quantity_grams, u.calories_per_100g, u.protein_per_100g,
                   u.fat_per_100g, u.carbs_per_100g
            FROM recipe_ingredient ri
            JOIN usda_food u ON u.fdc_id = ri.usda_fdc_id
            WHERE ri.recipe_id = ?
        """, (recipe_id,))

        rows = cursor.fetchall()
        if not rows:
            continue

        total_cal = 0
        total_pro = 0
        total_fat = 0
        total_carb = 0

        for qty_g, cal100, pro100, fat100, carb100 in rows:
            if qty_g and cal100 is not None:
                factor = qty_g / 100.0
                total_cal += (cal100 or 0) * factor
                total_pro += (pro100 or 0) * factor
                total_fat += (fat100 or 0) * factor
                total_carb += (carb100 or 0) * factor

        if total_cal > 0:
            serving_weight = (total_weight or sum(r[0] or 0 for r in rows)) / servings
            cursor.execute("""
                UPDATE dish SET
                    avg_calories_per_serving = ?,
                    avg_protein_per_serving = ?,
                    avg_carbs_per_serving = ?,
                    avg_fat_per_serving = ?,
                    default_serving_grams = ?
                WHERE id = ?
            """, (
                round(total_cal / servings, 1),
                round(total_pro / servings, 1),
                round(total_carb / servings, 1),
                round(total_fat / servings, 1),
                round(serving_weight, 0),
                dish_id,
            ))
            updated += 1

    conn.commit()
    print(f"  Dish nutrition computed: {updated} dishes updated with macros")
    return updated


def print_summary(conn: sqlite3.Connection, db_path: str):
    """Print summary statistics."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM dish")
    dish_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM recipe")
    recipe_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM recipe_ingredient")
    ingredient_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM recipe_ingredient WHERE usda_fdc_id IS NOT NULL")
    linked_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM usda_food")
    usda_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cuisine")
    cuisine_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM dish_alias")
    alias_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM symspell_deletes")
    symspell_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM dish WHERE avg_calories_per_serving IS NOT NULL")
    nutrition_count = cursor.fetchone()[0]

    file_size = os.path.getsize(db_path)
    size_mb = file_size / (1024 * 1024)

    link_pct = (linked_count / ingredient_count * 100) if ingredient_count else 0

    print("\n" + "=" * 60)
    print("Food Knowledge Graph Build Summary")
    print("=" * 60)
    print(f"  Dishes:              {dish_count:,}")
    print(f"  Recipes:             {recipe_count:,}")
    print(f"  Recipe ingredients:  {ingredient_count:,}")
    print(f"  USDA-linked:         {linked_count:,} ({link_pct:.1f}%)")
    print(f"  USDA foods:          {usda_count:,}")
    print(f"  Cuisines:            {cuisine_count}")
    print(f"  Aliases:             {alias_count:,}")
    print(f"  SymSpell deletes:    {symspell_count:,}")
    print(f"  Nutrition computed:  {nutrition_count:,} dishes")
    print(f"  Database size:       {size_mb:.2f} MB ({file_size:,} bytes)")
    if size_mb > 70:
        print(f"  WARNING: exceeds 70MB target!")
    else:
        print(f"  Size OK (under 70MB)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build the food knowledge graph")
    parser.add_argument("--db", default=None, help="Path to output database")
    parser.add_argument("--skip-recipenlg", action="store_true", help="Skip RecipeNLG seeding")
    parser.add_argument("--skip-worldcuisines", action="store_true", help="Skip WorldCuisines alias seeding")
    args = parser.parse_args()

    db_path = args.db or str(KG_DIR / "food-knowledge.db")

    # Remove existing DB for clean build
    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ["-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)

    print("=" * 60)
    print("Building Food Knowledge Graph")
    print("=" * 60)
    print(f"  Output: {db_path}")

    t0 = time.time()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    # Step 1: Schema
    print("\n[1/9] Applying schema...")
    apply_schema(conn)

    # Step 2: Generated dishes (curated baseline)
    print("\n[2/9] Seeding curated dishes...")
    seed_generated_dishes(conn)

    # Step 3: Classifier/detector label seeding (before recipes so recipe data can enrich)
    print("\n[3/9] Seeding classifier/detector labels as dishes...")
    seed_classifier_labels(conn)

    # Step 4: Recipe dataset (volume)
    if not args.skip_recipenlg:
        print("\n[4/9] Seeding from recipe datasets...")
        seed_recipes(conn)
    else:
        print("\n[4/9] Skipping recipe seeding (--skip-recipenlg)")

    # Step 5: USDA SR Legacy
    print("\n[5/9] Seeding USDA SR Legacy nutrition...")
    seed_usda_sr_legacy(conn)

    # Step 6: WorldCuisines aliases (before SymSpell so aliases get indexed)
    if not args.skip_worldcuisines:
        print("\n[6/9] Seeding WorldCuisines multilingual aliases...")
        try:
            from seed_worldcuisines import seed_worldcuisines
            seed_worldcuisines(conn)
        except Exception as e:
            print(f"  Warning: WorldCuisines seeding failed: {e}")
            print("  Continuing without aliases...")
    else:
        print("\n[6/9] Skipping WorldCuisines seeding (--skip-worldcuisines)")

    # Step 7: SymSpell (covers dish names + aliases)
    print("\n[7/9] Computing SymSpell delete variants...")
    compute_symspell_deletes(conn)

    # Step 8: Nutrition aggregation
    print("\n[8/9] Computing dish nutrition averages...")
    compute_dish_nutrition(conn)

    # Step 9: Rebuild FTS5 indexes (needed for content-sync tables)
    print("\n[9/9] Rebuilding FTS5 indexes...")
    conn.execute("INSERT INTO dish_fts(dish_fts) VALUES('rebuild')")
    conn.execute("INSERT INTO dish_alias_fts(dish_alias_fts) VALUES('rebuild')")
    conn.commit()
    print("  FTS5 indexes rebuilt")

    # Summary
    print_summary(conn, db_path)

    elapsed = time.time() - t0
    print(f"\nBuild completed in {elapsed:.1f}s")

    conn.close()


if __name__ == "__main__":
    main()
