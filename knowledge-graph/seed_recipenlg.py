#!/usr/bin/env python3
"""
Seed the food knowledge graph from RecipeNLG dataset (2.2M recipes).

Downloads RecipeNLG from HuggingFace, extracts dish names and ingredient lists,
deduplicates, assigns cuisine labels, and populates the SQLite database.

Usage:
    python seed_recipenlg.py [--db food-knowledge.db] [--limit N] [--batch-size N]
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Unit conversion table: convert common recipe units to grams (approximate)
# ---------------------------------------------------------------------------
UNIT_TO_GRAMS = {
    # Volume-based (using water density as baseline, adjusted for common foods)
    "cup": 240.0,
    "cups": 240.0,
    "c": 240.0,
    "tablespoon": 15.0,
    "tablespoons": 15.0,
    "tbsp": 15.0,
    "tbs": 15.0,
    "teaspoon": 5.0,
    "teaspoons": 5.0,
    "tsp": 5.0,
    "fluid ounce": 30.0,
    "fluid ounces": 30.0,
    "fl oz": 30.0,
    "pint": 480.0,
    "pints": 480.0,
    "quart": 960.0,
    "quarts": 960.0,
    "gallon": 3840.0,
    "gallons": 3840.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    # Weight-based
    "ounce": 28.35,
    "ounces": 28.35,
    "oz": 28.35,
    "pound": 453.6,
    "pounds": 453.6,
    "lb": 453.6,
    "lbs": 453.6,
    "gram": 1.0,
    "grams": 1.0,
    "g": 1.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "kg": 1000.0,
    # Count-based (rough estimates)
    "clove": 5.0,
    "cloves": 5.0,
    "slice": 30.0,
    "slices": 30.0,
    "piece": 50.0,
    "pieces": 50.0,
    "whole": 100.0,
    "large": 150.0,
    "medium": 100.0,
    "small": 75.0,
    "bunch": 50.0,
    "pinch": 0.5,
    "dash": 0.5,
    "can": 400.0,
    "jar": 350.0,
    "package": 250.0,
    "pkg": 250.0,
    "stick": 113.0,  # butter stick
    "head": 200.0,
    "stalk": 60.0,
    "stalks": 60.0,
    "sprig": 2.0,
    "sprigs": 2.0,
}

# ---------------------------------------------------------------------------
# Cuisine keyword mapping
# ---------------------------------------------------------------------------
CUISINE_KEYWORDS = {
    "Japanese": [
        "sushi", "ramen", "tempura", "teriyaki", "miso", "udon", "soba",
        "tonkatsu", "onigiri", "yakitori", "edamame", "gyoza", "sashimi",
        "katsu", "dashi", "mochi", "matcha", "wasabi", "nori", "tofu",
        "takoyaki", "okonomiyaki", "donburi", "karaage", "unagi",
        "hibachi", "sukiyaki", "shabu", "japanese",
    ],
    "Chinese": [
        "kung pao", "dim sum", "wonton", "chow mein", "lo mein",
        "fried rice", "spring roll", "egg roll", "mapo tofu",
        "sweet and sour", "general tso", "szechuan", "sichuan",
        "hoisin", "char siu", "peking duck", "hot pot", "hotpot",
        "dumpling", "bao", "congee", "chop suey", "five spice",
        "oyster sauce", "black bean sauce", "wok", "stir fry",
        "stir-fry", "chinese", "cantonese", "hunan", "mandarin",
    ],
    "Korean": [
        "kimchi", "bibimbap", "bulgogi", "japchae", "tteokbokki",
        "samgyeopsal", "galbi", "kalbi", "gochujang", "doenjang",
        "korean", "banchan", "sundubu", "jjigae", "kimbap",
        "pajeon", "mandu", "ramyeon", "dakgalbi", "bossam",
    ],
    "Vietnamese": [
        "pho", "banh mi", "bun bo", "spring rolls vietnamese",
        "goi cuon", "bun cha", "com tam", "cao lau", "vietnamese",
        "nuoc mam", "fish sauce vietnamese", "banh xeo", "bun rieu",
    ],
    "Thai": [
        "pad thai", "green curry", "red curry", "yellow curry",
        "massaman", "tom yum", "tom kha", "som tum", "papaya salad",
        "pad see ew", "thai basil", "larb", "khao soi", "thai",
        "satay", "panang", "thai tea", "sticky rice mango",
        "lemongrass", "galangal",
    ],
    "Indian": [
        "curry", "tandoori", "tikka", "masala", "biryani", "naan",
        "samosa", "dal", "daal", "paneer", "vindaloo", "korma",
        "chapati", "roti", "paratha", "dosa", "idli", "chutney",
        "raita", "ghee", "turmeric", "cumin", "coriander", "garam masala",
        "butter chicken", "palak", "saag", "aloo", "chana", "indian",
    ],
    "Mexican": [
        "taco", "tacos", "burrito", "enchilada", "quesadilla",
        "guacamole", "salsa", "nacho", "nachos", "fajita", "fajitas",
        "churro", "tamale", "tamales", "mole", "pozole", "elote",
        "tostada", "carnitas", "chipotle", "jalape", "mexican",
        "tortilla", "cilantro lime",
    ],
    "Italian": [
        "pasta", "spaghetti", "lasagna", "risotto", "gnocchi",
        "carbonara", "bolognese", "pesto", "bruschetta", "caprese",
        "tiramisu", "gelato", "focaccia", "ciabatta", "prosciutto",
        "parmigiana", "parmesan", "mozzarella", "marinara",
        "alfredo", "ravioli", "tortellini", "cannoli", "italian",
        "pizza", "calzone", "antipasto", "minestrone", "ossobuco",
    ],
    "Mediterranean": [
        "falafel", "hummus", "shawarma", "kebab", "tzatziki",
        "tabbouleh", "baba ghanoush", "pita", "couscous",
        "moussaka", "dolma", "fattoush", "labneh", "za'atar",
        "mediterranean", "greek salad", "souvlaki", "gyro",
    ],
    "Western": [
        "burger", "hamburger", "cheeseburger", "hot dog", "sandwich",
        "steak", "roast", "fried chicken", "mac and cheese",
        "mashed potato", "baked potato", "french fries", "fries",
        "bbq", "barbecue", "grilled cheese", "club sandwich",
        "meatloaf", "pot pie", "shepherd's pie", "beef stew",
        "clam chowder", "coleslaw", "biscuits and gravy",
    ],
}

# Ingredient category mapping
INGREDIENT_CATEGORIES = {
    "protein": [
        "chicken", "beef", "pork", "lamb", "turkey", "duck", "fish",
        "salmon", "tuna", "shrimp", "prawn", "crab", "lobster",
        "tofu", "tempeh", "egg", "eggs", "bacon", "ham", "sausage",
        "pancetta", "prosciutto", "anchovy", "anchovies", "ground beef",
        "ground turkey", "ground pork", "steak", "veal", "venison",
        "scallop", "mussel", "clam", "squid", "octopus",
    ],
    "grain": [
        "rice", "pasta", "noodle", "noodles", "bread", "flour",
        "oat", "oats", "quinoa", "barley", "corn", "cornmeal",
        "tortilla", "pita", "couscous", "bulgur", "millet",
        "soba", "udon", "ramen", "spaghetti", "macaroni",
        "penne", "fettuccine", "linguine", "orzo", "risotto",
    ],
    "vegetable": [
        "onion", "garlic", "tomato", "potato", "carrot", "celery",
        "pepper", "bell pepper", "broccoli", "cauliflower", "spinach",
        "lettuce", "cucumber", "zucchini", "eggplant", "mushroom",
        "mushrooms", "pea", "peas", "green bean", "corn", "cabbage",
        "kale", "chard", "asparagus", "artichoke", "beet", "radish",
        "turnip", "squash", "pumpkin", "sweet potato", "yam",
        "bok choy", "bean sprout", "bamboo shoot", "water chestnut",
        "spring onion", "scallion", "shallot", "leek", "ginger",
    ],
    "fruit": [
        "apple", "banana", "orange", "lemon", "lime", "berry",
        "berries", "strawberry", "blueberry", "raspberry", "grape",
        "mango", "pineapple", "peach", "pear", "plum", "cherry",
        "watermelon", "melon", "coconut", "avocado", "fig", "date",
        "papaya", "passion fruit", "lychee", "kiwi",
    ],
    "dairy": [
        "milk", "cream", "cheese", "butter", "yogurt", "sour cream",
        "cream cheese", "mozzarella", "parmesan", "cheddar", "ricotta",
        "feta", "gouda", "brie", "goat cheese", "whipped cream",
        "heavy cream", "half and half", "condensed milk",
        "evaporated milk", "buttermilk", "ghee",
    ],
    "oil": [
        "oil", "olive oil", "vegetable oil", "canola oil", "sesame oil",
        "coconut oil", "peanut oil", "sunflower oil", "corn oil",
        "avocado oil", "lard", "shortening", "cooking spray",
    ],
    "seasoning": [
        "salt", "pepper", "sugar", "soy sauce", "vinegar",
        "worcestershire", "hot sauce", "mustard", "ketchup",
        "mayonnaise", "honey", "maple syrup", "fish sauce",
        "oyster sauce", "hoisin", "teriyaki", "sriracha",
        "chili flake", "red pepper flake", "paprika", "cumin",
        "coriander", "turmeric", "cinnamon", "nutmeg", "oregano",
        "basil", "thyme", "rosemary", "parsley", "cilantro",
        "dill", "bay leaf", "vanilla", "cocoa", "chocolate",
        "baking powder", "baking soda", "yeast", "cornstarch",
        "gochujang", "miso", "wasabi", "tahini",
    ],
    "legume": [
        "bean", "beans", "lentil", "lentils", "chickpea", "chickpeas",
        "black bean", "kidney bean", "pinto bean", "navy bean",
        "lima bean", "edamame", "soybean", "peanut", "peanuts",
    ],
    "nut": [
        "almond", "almonds", "walnut", "walnuts", "cashew", "cashews",
        "pecan", "pecans", "pistachio", "pine nut", "pine nuts",
        "hazelnut", "hazelnuts", "macadamia", "chestnut",
    ],
}


def normalize_dish_name(name: str) -> str:
    """Normalize a dish name for deduplication."""
    name = name.lower().strip()
    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name)
    # Remove common prefixes/suffixes that don't change the dish identity
    name = re.sub(r"^(easy|simple|best|quick|homemade|classic|traditional|my|mom'?s?|grandma'?s?)\s+", "", name)
    # Remove trailing recipe indicators
    name = re.sub(r"\s+(recipe|recipes|dish|meal)$", "", name)
    # Remove content in parentheses
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.strip()
    return name


def classify_cuisine(dish_name: str) -> str:
    """Assign a cuisine label based on keyword matching."""
    name_lower = dish_name.lower()
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return cuisine
    return "Other"


def categorize_ingredient(ingredient_name: str) -> str:
    """Categorize an ingredient based on keyword matching."""
    name_lower = ingredient_name.lower()
    for category, keywords in INGREDIENT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return "other"


def parse_amount_and_unit(amount_str: str) -> tuple:
    """
    Parse an amount string like '2 cups' or '1/2 teaspoon' into (quantity, unit).
    Returns (float, str) or (None, None) if unparseable.
    """
    amount_str = amount_str.strip().lower()
    if not amount_str:
        return None, None

    # Handle fractions like 1/2, 3/4
    fraction_match = re.match(r"(\d+)\s*/\s*(\d+)", amount_str)
    mixed_match = re.match(r"(\d+)\s+(\d+)\s*/\s*(\d+)", amount_str)

    quantity = None
    rest = amount_str

    if mixed_match:
        whole = float(mixed_match.group(1))
        numer = float(mixed_match.group(2))
        denom = float(mixed_match.group(3))
        if denom != 0:
            quantity = whole + numer / denom
        rest = amount_str[mixed_match.end():].strip()
    elif fraction_match:
        numer = float(fraction_match.group(1))
        denom = float(fraction_match.group(2))
        if denom != 0:
            quantity = numer / denom
        rest = amount_str[fraction_match.end():].strip()
    else:
        num_match = re.match(r"([\d.]+)", amount_str)
        if num_match:
            try:
                quantity = float(num_match.group(1))
            except ValueError:
                pass
            rest = amount_str[num_match.end():].strip()

    # Find unit in remaining text
    unit = None
    for u in sorted(UNIT_TO_GRAMS.keys(), key=len, reverse=True):
        if rest.startswith(u):
            unit = u
            break

    return quantity, unit


def estimate_grams(quantity, unit) -> float:
    """Estimate weight in grams from quantity and unit."""
    if quantity is None:
        return 50.0  # default guess for unparseable amounts
    if unit and unit in UNIT_TO_GRAMS:
        return quantity * UNIT_TO_GRAMS[unit]
    # If no unit recognized, treat as count-based
    return quantity * 100.0  # rough guess: 100g per unit


def parse_ingredient_text(ingredient_str: str) -> dict:
    """
    Parse a single ingredient string like '2 cups all-purpose flour' into
    structured data.
    """
    ingredient_str = ingredient_str.strip()
    if not ingredient_str:
        return None

    # Try to extract amount from the beginning
    # Pattern: optional number/fraction, optional unit, then ingredient name
    amount_pattern = re.match(
        r"^([\d./\s]+)?\s*"
        r"(cup|cups|tablespoon|tablespoons|tbsp|tbs|teaspoon|teaspoons|tsp|"
        r"ounce|ounces|oz|pound|pounds|lb|lbs|gram|grams|g|kg|"
        r"fluid ounce|fluid ounces|fl oz|pint|pints|quart|quarts|"
        r"gallon|gallons|liter|liters|ml|milliliter|milliliters|"
        r"clove|cloves|slice|slices|piece|pieces|whole|large|medium|small|"
        r"bunch|pinch|dash|can|jar|package|pkg|stick|head|stalk|stalks|"
        r"sprig|sprigs)?\s*"
        r"(?:of\s+)?"
        r"(.+)",
        ingredient_str,
        re.IGNORECASE,
    )

    if amount_pattern:
        amount_str = (amount_pattern.group(1) or "").strip()
        unit_str = (amount_pattern.group(2) or "").strip().lower()
        name = (amount_pattern.group(3) or "").strip()

        quantity, _ = parse_amount_and_unit(f"{amount_str} {unit_str}")
        estimated_g = estimate_grams(quantity, unit_str if unit_str else None)
    else:
        name = ingredient_str
        estimated_g = 50.0

    # Clean up ingredient name
    name = re.sub(r",.*$", "", name)  # Remove everything after comma
    name = re.sub(r"\(.*?\)", "", name)  # Remove parenthetical notes
    name = re.sub(r"\s+", " ", name).strip().lower()

    if not name or len(name) < 2:
        return None

    return {
        "name": name,
        "estimated_g": estimated_g,
    }


def parse_ner_ingredients(ner_list) -> list:
    """
    Parse NER-tagged ingredients from RecipeNLG.
    NER tags provide structured ingredient names directly.
    """
    if not ner_list:
        return []

    ingredients = []
    for ner_item in ner_list:
        if isinstance(ner_item, str):
            name = ner_item.strip().lower()
            name = re.sub(r"\s+", " ", name)
            if name and len(name) >= 2:
                ingredients.append(name)
    return ingredients


def link_variants(conn: sqlite3.Connection) -> int:
    """
    Link dish variants by finding related dish names.
    E.g., 'chicken fried rice' is a variant of 'fried rice'.
    """
    cursor = conn.cursor()

    # Get all dish names
    cursor.execute("SELECT id, name FROM dishes ORDER BY length(name) ASC")
    dishes = cursor.fetchall()

    dish_lookup = {name: did for did, name in dishes}
    linked = 0

    # For each dish, check if a shorter dish name is a suffix
    # (e.g., 'fried rice' is suffix of 'chicken fried rice')
    for did, name in dishes:
        words = name.split()
        if len(words) <= 1:
            continue

        # Try progressively longer suffixes
        for i in range(1, len(words)):
            suffix = " ".join(words[i:])
            if suffix in dish_lookup and dish_lookup[suffix] != did:
                canonical_id = dish_lookup[suffix]
                cursor.execute(
                    "UPDATE dishes SET canonical_id = ? WHERE id = ? AND canonical_id IS NULL",
                    (canonical_id, did),
                )
                if cursor.rowcount > 0:
                    linked += 1
                break  # Only link to the longest matching suffix

    conn.commit()
    return linked


def seed_from_recipenlg(db_path: str, limit: int = 0, batch_size: int = 5000):
    """
    Seed the knowledge graph from RecipeNLG dataset.

    Args:
        db_path: Path to SQLite database
        limit: Maximum number of recipes to process (0 = all)
        batch_size: Number of records to commit in each batch
    """
    print("=" * 60)
    print("Seeding knowledge graph from RecipeNLG")
    print("=" * 60)

    # Initialize database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

    # Apply schema
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        conn.executescript(f.read())

    # Load RecipeNLG dataset
    print("\nLoading RecipeNLG dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("mbien/recipe_nlg", split="train")
        total = len(ds)
        print(f"Loaded {total:,} recipes")
    except Exception as e:
        print(f"Error loading RecipeNLG from HuggingFace: {e}")
        print("Falling back to built-in common dishes...")
        _seed_builtin_dishes(conn)
        conn.close()
        return

    if limit > 0:
        total = min(total, limit)
        print(f"Processing first {total:,} recipes (limit={limit})")

    # Phase 1: Aggregate ingredient data per dish
    print("\nPhase 1: Aggregating recipes by dish name...")
    dish_data = defaultdict(lambda: {"ingredients": defaultdict(list), "count": 0})

    for idx in tqdm(range(total), desc="Processing recipes"):
        recipe = ds[idx]
        title = recipe.get("title", "")
        if not title:
            continue

        dish_name = normalize_dish_name(title)
        if not dish_name or len(dish_name) < 3:
            continue

        # Get ingredients from NER tags (preferred) or raw ingredient list
        ner = recipe.get("ner", [])
        raw_ingredients = recipe.get("ingredients", [])

        # Parse NER for ingredient names
        ner_names = parse_ner_ingredients(ner) if ner else []

        # Parse raw ingredient strings for amounts
        parsed_ingredients = []
        if raw_ingredients:
            for ing_str in raw_ingredients:
                parsed = parse_ingredient_text(ing_str)
                if parsed:
                    parsed_ingredients.append(parsed)

        # Merge: use NER names where possible, fall back to parsed names
        if ner_names:
            for ner_name in ner_names:
                # Find matching parsed ingredient for the amount
                matched_g = 50.0
                for parsed in parsed_ingredients:
                    if ner_name in parsed["name"] or parsed["name"] in ner_name:
                        matched_g = parsed["estimated_g"]
                        break
                dish_data[dish_name]["ingredients"][ner_name].append(matched_g)
        elif parsed_ingredients:
            for parsed in parsed_ingredients:
                dish_data[dish_name]["ingredients"][parsed["name"]].append(
                    parsed["estimated_g"]
                )

        dish_data[dish_name]["count"] += 1

    print(f"\nFound {len(dish_data):,} unique dish names from {total:,} recipes")

    # Phase 2: Insert into database
    print("\nPhase 2: Inserting into database...")
    ingredient_cache = {}
    dishes_inserted = 0
    relationships_inserted = 0
    cuisine_counts = defaultdict(int)

    cursor = conn.cursor()

    for dish_name, data in tqdm(dish_data.items(), desc="Inserting dishes"):
        if data["count"] < 1:
            continue

        cuisine = classify_cuisine(dish_name)
        cuisine_counts[cuisine] += 1

        # Confidence based on how many recipes we averaged
        confidence = min(0.9, 0.3 + (data["count"] / 100.0) * 0.6)

        try:
            cursor.execute(
                "INSERT OR IGNORE INTO dishes (name, cuisine, source, confidence) VALUES (?, ?, 'recipenlg', ?)",
                (dish_name, cuisine, confidence),
            )
            if cursor.rowcount == 0:
                continue  # Duplicate, skip
            dish_id = cursor.lastrowid
            dishes_inserted += 1
        except sqlite3.IntegrityError:
            continue

        # Calculate average amounts and weight percentages
        total_weight = sum(
            sum(amounts) / len(amounts)
            for amounts in data["ingredients"].values()
        )

        if total_weight == 0:
            total_weight = 1.0

        for ing_name, amounts in data["ingredients"].items():
            avg_amount = sum(amounts) / len(amounts)
            weight_pct = avg_amount / total_weight

            # Cap weight_pct at 1.0 and floor at 0.001
            weight_pct = max(0.001, min(1.0, weight_pct))

            # Determine if nutrition significant (>1% of weight)
            is_significant = weight_pct > 0.01

            # Get or create ingredient
            if ing_name not in ingredient_cache:
                category = categorize_ingredient(ing_name)
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO ingredients (name, category) VALUES (?, ?)",
                        (ing_name, category),
                    )
                    if cursor.rowcount > 0:
                        ingredient_cache[ing_name] = cursor.lastrowid
                    else:
                        cursor.execute(
                            "SELECT id FROM ingredients WHERE name = ?", (ing_name,)
                        )
                        row = cursor.fetchone()
                        ingredient_cache[ing_name] = row[0] if row else None
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM ingredients WHERE name = ?", (ing_name,)
                    )
                    row = cursor.fetchone()
                    ingredient_cache[ing_name] = row[0] if row else None

            ing_id = ingredient_cache.get(ing_name)
            if ing_id is None:
                continue

            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO dish_ingredients
                       (dish_id, ingredient_id, weight_pct, is_nutrition_significant,
                        typical_amount_g, source, confidence)
                       VALUES (?, ?, ?, ?, ?, 'recipenlg', ?)""",
                    (dish_id, ing_id, round(weight_pct, 4), is_significant,
                     round(avg_amount, 1), confidence),
                )
                if cursor.rowcount > 0:
                    relationships_inserted += 1
            except sqlite3.IntegrityError:
                pass

        # Batch commit
        if dishes_inserted % batch_size == 0:
            conn.commit()

    conn.commit()

    # Phase 3: Link variants
    print("\nPhase 3: Linking dish variants...")
    variants_linked = link_variants(conn)

    conn.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("RecipeNLG Seeding Complete")
    print("=" * 60)
    print(f"  Recipes processed:        {total:,}")
    print(f"  Unique dishes created:     {dishes_inserted:,}")
    print(f"  Unique ingredients:        {len(ingredient_cache):,}")
    print(f"  Dish-ingredient links:     {relationships_inserted:,}")
    print(f"  Variant links:             {variants_linked:,}")
    print(f"\n  Cuisine distribution:")
    for cuisine in sorted(cuisine_counts.keys()):
        count = cuisine_counts[cuisine]
        pct = (count / dishes_inserted * 100) if dishes_inserted > 0 else 0
        print(f"    {cuisine:20s}: {count:>6,} ({pct:5.1f}%)")

    conn.close()


def _seed_builtin_dishes(conn: sqlite3.Connection):
    """
    Seed database with a curated set of 1000+ common dishes as fallback
    when RecipeNLG is unavailable. Loads dish data from generate_dishes.py.
    """
    print("\nSeeding with built-in common dishes (1000+ entries)...")

    # Import the generated dish database
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dishes import get_all_dishes

    dishes = get_all_dishes()
    print(f"  Loaded {len(dishes)} dishes from generate_dishes.py")


    cursor = conn.cursor()
    ingredient_cache = {}
    dishes_inserted = 0
    relationships_inserted = 0
    cuisine_counts = defaultdict(int)

    for dish_name, dish_data in dishes.items():
        cuisine = dish_data["cuisine"]
        cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

        cursor.execute(
            "INSERT OR IGNORE INTO dishes (name, cuisine, source, confidence) VALUES (?, ?, 'recipenlg', 0.8)",
            (dish_name, cuisine),
        )
        if cursor.rowcount == 0:
            continue
        dish_id = cursor.lastrowid
        dishes_inserted += 1

        for ing_name, (weight_pct, typical_g) in dish_data["ingredients"].items():
            if ing_name not in ingredient_cache:
                category = categorize_ingredient(ing_name)
                cursor.execute(
                    "INSERT OR IGNORE INTO ingredients (name, category) VALUES (?, ?)",
                    (ing_name, category),
                )
                if cursor.rowcount > 0:
                    ingredient_cache[ing_name] = cursor.lastrowid
                else:
                    cursor.execute("SELECT id FROM ingredients WHERE name = ?", (ing_name,))
                    row = cursor.fetchone()
                    ingredient_cache[ing_name] = row[0] if row else None

            ing_id = ingredient_cache.get(ing_name)
            if ing_id is None:
                continue

            is_significant = weight_pct > 0.01
            cursor.execute(
                """INSERT OR IGNORE INTO dish_ingredients
                   (dish_id, ingredient_id, weight_pct, is_nutrition_significant,
                    typical_amount_g, source, confidence)
                   VALUES (?, ?, ?, ?, ?, 'recipenlg', 0.8)""",
                (dish_id, ing_id, weight_pct, is_significant, typical_g),
            )
            if cursor.rowcount > 0:
                relationships_inserted += 1

    conn.commit()

    # Link variants using the generic linker (suffix matching)
    print("\nLinking dish variants...")
    variants_linked = link_variants(conn)

    # Manual variant links for cross-language equivalents
    manual_variants = [
        ("nasi goreng", "fried rice"),
        ("mee goreng", "chow mein"),
        ("char kway teow", "pad see ew"),
        ("banh mi chay", "banh mi"),
        ("beef pho", "pho"),
        ("pho ga", "pho"),
        ("tofu pad thai", "pad thai"),
        ("spaghetti carbonara", "carbonara"),
        ("fettuccine bolognese", "bolognese"),
    ]
    for variant_name, canonical_name in manual_variants:
        cursor.execute("SELECT id FROM dishes WHERE name = ?", (canonical_name,))
        canon = cursor.fetchone()
        if canon:
            cursor.execute(
                "UPDATE dishes SET canonical_id = ? WHERE name = ? AND canonical_id IS NULL",
                (canon[0], variant_name),
            )
            if cursor.rowcount > 0:
                variants_linked += 1

    conn.commit()

    # Print summary
    print(f"\n  Inserted {dishes_inserted} dishes, {relationships_inserted} relationships")
    print(f"  Unique ingredients: {len(ingredient_cache)}")
    print(f"  Variant links: {variants_linked}")
    print(f"\n  Cuisine distribution:")
    for cuisine in sorted(cuisine_counts.keys()):
        count = cuisine_counts[cuisine]
        pct = (count / dishes_inserted * 100) if dishes_inserted > 0 else 0
        print(f"    {cuisine:20s}: {count:>6,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Seed knowledge graph from RecipeNLG")
    parser.add_argument("--db", default="knowledge-graph/food-knowledge.db",
                        help="Path to SQLite database")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of recipes to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch commit size")
    args = parser.parse_args()

    # Resolve path relative to project root
    db_path = args.db
    if not os.path.isabs(db_path):
        project_root = Path(__file__).parent.parent
        db_path = str(project_root / db_path)

    seed_from_recipenlg(db_path, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
