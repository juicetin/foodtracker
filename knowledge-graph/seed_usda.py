#!/usr/bin/env python3
"""
Seed the food knowledge graph from USDA FoodData Central API.

Queries USDA FDC for prepared/mixed food items (FNDDS data type),
matches to existing dishes in the knowledge graph, and adds USDA
FDC IDs and ingredient data.

Usage:
    python seed_usda.py [--db food-knowledge.db] [--api-key DEMO_KEY] [--max-pages N]
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# USDA FDC API configuration
FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_API_KEY = "DEMO_KEY"
# DEMO_KEY: 30 requests/hour, 1000/day
REQUEST_DELAY_SECONDS = 2.5  # ~24 requests/minute, well within 30/hour

# Cache directory for API responses
CACHE_DIR = Path(__file__).parent / "cache" / "usda_responses"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cached_request(url: str, params: dict, cache_key: str) -> dict:
    """
    Make an API request with filesystem caching.
    Returns cached response if available, otherwise makes the request and caches it.
    """
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Cache the response
        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data
    except requests.RequestException as e:
        print(f"  API request failed: {e}")
        return None


def search_fndds_foods(api_key: str, query: str = "chicken", page_number: int = 1, page_size: int = 50) -> dict:
    """
    Search USDA FDC for FNDDS (Survey) foods -- prepared/mixed food items.
    Requires a query term (the API search endpoint needs one).
    """
    url = f"{FDC_BASE_URL}/foods/search"
    params = {
        "api_key": api_key,
        "query": query,
        "dataType": "Survey (FNDDS)",
        "pageSize": page_size,
        "pageNumber": page_number,
        "sortBy": "dataType.keyword",
        "sortOrder": "asc",
    }
    cache_key = f"fndds_search_{query.replace(' ', '_')}_page{page_number}_size{page_size}"
    return cached_request(url, params, cache_key)


def get_food_details(fdc_id: int, api_key: str) -> dict:
    """
    Get detailed food item including inputFoods (component ingredients).
    """
    url = f"{FDC_BASE_URL}/food/{fdc_id}"
    params = {"api_key": api_key}
    cache_key = f"food_detail_{fdc_id}"
    return cached_request(url, params, cache_key)


def normalize_usda_name(description: str) -> str:
    """
    Normalize USDA food description to match our dish naming convention.
    USDA names are like "Pizza, cheese" -> "cheese pizza"
    """
    description = description.lower().strip()

    # Remove common USDA suffixes
    for suffix in [", ns as to type", ", nfs", ", from restaurant",
                   ", from fast food", ", homemade", ", commercially prepared",
                   ", canned", ", frozen", ", dried", ", raw", ", cooked"]:
        description = description.replace(suffix, "")

    # Reverse comma-separated parts: "Pizza, cheese" -> "cheese pizza"
    parts = [p.strip() for p in description.split(",")]
    if len(parts) == 2:
        description = f"{parts[1]} {parts[0]}"
    elif len(parts) > 2:
        # Keep first part, append rest
        description = f"{' '.join(parts[1:])} {parts[0]}"

    # Clean up
    description = " ".join(description.split())
    return description


def fuzzy_match_dish(conn: sqlite3.Connection, usda_name: str) -> tuple:
    """
    Try to match a USDA food name to an existing dish in the database.
    Returns (dish_id, dish_name) or (None, None) if no match.
    """
    cursor = conn.cursor()

    # Exact match
    cursor.execute("SELECT id, name FROM dishes WHERE name = ?", (usda_name,))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]

    # FTS5 search
    try:
        cursor.execute(
            "SELECT rowid, name FROM dishes_fts WHERE dishes_fts MATCH ? LIMIT 1",
            (usda_name,),
        )
        row = cursor.fetchone()
        if row:
            return row[0], row[1]
    except sqlite3.OperationalError:
        pass

    # Partial match: check if any word combination matches
    words = usda_name.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            phrase = " ".join(words[start : start + length])
            if len(phrase) < 4:
                continue
            cursor.execute(
                "SELECT id, name FROM dishes WHERE name LIKE ? LIMIT 1",
                (f"%{phrase}%",),
            )
            row = cursor.fetchone()
            if row:
                return row[0], row[1]

    return None, None


def categorize_usda_ingredient(food_description: str) -> str:
    """Categorize a USDA input food based on its description."""
    desc = food_description.lower()

    categories = {
        "protein": ["chicken", "beef", "pork", "fish", "shrimp", "egg", "turkey",
                     "lamb", "salmon", "tuna", "meat", "bacon", "sausage"],
        "grain": ["rice", "pasta", "noodle", "bread", "flour", "wheat", "oat",
                   "corn", "tortilla", "cereal"],
        "vegetable": ["onion", "garlic", "tomato", "potato", "carrot", "celery",
                       "pepper", "broccoli", "spinach", "lettuce", "mushroom",
                       "cabbage", "bean sprout"],
        "fruit": ["apple", "banana", "orange", "lemon", "lime", "berry",
                   "mango", "pineapple"],
        "dairy": ["milk", "cream", "cheese", "butter", "yogurt", "whey"],
        "oil": ["oil", "lard", "shortening", "margarine"],
        "seasoning": ["salt", "pepper", "sugar", "sauce", "vinegar", "spice",
                       "herb", "seasoning", "mustard", "ketchup"],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in desc:
                return category

    return "other"


def _process_usda_food(food, conn, cursor, api_key, counters):
    """Process a single USDA food item, updating the knowledge graph."""
    fdc_id = food.get("fdcId")
    description = food.get("description", "")
    if not fdc_id or not description:
        return

    usda_name = normalize_usda_name(description)
    if not usda_name or len(usda_name) < 3:
        return

    dish_id, matched_name = fuzzy_match_dish(conn, usda_name)

    if dish_id:
        cursor.execute(
            "UPDATE dishes SET confidence = MAX(confidence, 0.8), updated_at = datetime('now') WHERE id = ?",
            (dish_id,),
        )
        counters["dishes_updated"] += 1
    else:
        from seed_recipenlg import classify_cuisine
        cuisine = classify_cuisine(usda_name)
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO dishes (name, cuisine, source, confidence) VALUES (?, ?, 'usda', 0.8)",
                (usda_name, cuisine),
            )
            if cursor.rowcount > 0:
                dish_id = cursor.lastrowid
                counters["dishes_created"] += 1
            else:
                return
        except sqlite3.IntegrityError:
            return

    detail_cache = CACHE_DIR / f"food_detail_{fdc_id}.json"
    if not detail_cache.exists():
        time.sleep(REQUEST_DELAY_SECONDS)
        counters["api_calls"] += 1

    details = get_food_details(fdc_id, api_key)
    if not details:
        return

    input_foods = details.get("inputFoods", [])
    if not input_foods:
        return

    total_grams = sum(inp.get("gramWeight", 0) for inp in input_foods)
    if total_grams == 0:
        total_grams = 1.0

    for inp in input_foods:
        inp_desc = inp.get("foodDescription", inp.get("ingredientDescription", ""))
        inp_fdc_id = inp.get("inputFood", {}).get("fdcId") if isinstance(inp.get("inputFood"), dict) else inp.get("id")
        gram_weight = inp.get("gramWeight", 0)

        if not inp_desc:
            continue

        ing_name = inp_desc.lower().strip()
        for suffix in [", raw", ", cooked", ", fresh", ", canned", ", frozen", ", dried", ", ns as to form"]:
            ing_name = ing_name.replace(suffix, "")
        ing_name = " ".join(ing_name.split())

        if not ing_name or len(ing_name) < 2:
            continue

        cursor.execute("SELECT id FROM ingredients WHERE name = ?", (ing_name,))
        row = cursor.fetchone()

        if row:
            ing_id = row[0]
            if inp_fdc_id:
                cursor.execute("UPDATE ingredients SET usda_fdc_id = ? WHERE id = ? AND usda_fdc_id IS NULL", (inp_fdc_id, ing_id))
                if cursor.rowcount > 0:
                    counters["fdc_ids_set"] += 1
        else:
            category = categorize_usda_ingredient(ing_name)
            try:
                cursor.execute("INSERT OR IGNORE INTO ingredients (name, usda_fdc_id, category) VALUES (?, ?, ?)", (ing_name, inp_fdc_id, category))
                if cursor.rowcount > 0:
                    ing_id = cursor.lastrowid
                    if inp_fdc_id:
                        counters["fdc_ids_set"] += 1
                else:
                    cursor.execute("SELECT id FROM ingredients WHERE name = ?", (ing_name,))
                    row = cursor.fetchone()
                    ing_id = row[0] if row else None
            except sqlite3.IntegrityError:
                cursor.execute("SELECT id FROM ingredients WHERE name = ?", (ing_name,))
                row = cursor.fetchone()
                ing_id = row[0] if row else None

        if ing_id is None:
            continue

        weight_pct = max(0.001, min(1.0, gram_weight / total_grams))
        is_significant = weight_pct > 0.01

        try:
            cursor.execute(
                """INSERT OR REPLACE INTO dish_ingredients
                   (dish_id, ingredient_id, weight_pct, is_nutrition_significant,
                    typical_amount_g, source, confidence)
                   VALUES (?, ?, ?, ?, ?, 'usda', 0.8)""",
                (dish_id, ing_id, round(weight_pct, 4), is_significant, round(gram_weight, 1)),
            )
            if cursor.rowcount > 0:
                counters["ingredients_mapped"] += 1
        except sqlite3.IntegrityError:
            pass


def seed_from_usda(db_path: str, api_key: str = DEFAULT_API_KEY, max_pages: int = 5):
    """
    Seed/enrich the knowledge graph from USDA FoodData Central.
    """
    print("=" * 60)
    print("Enriching knowledge graph from USDA FoodData Central")
    print("=" * 60)

    ensure_cache_dir()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    counters = {"dishes_updated": 0, "dishes_created": 0, "ingredients_mapped": 0, "fdc_ids_set": 0, "api_calls": 0}

    search_queries = [
        "chicken", "beef", "pork", "fish", "rice", "pasta", "soup",
        "salad", "sandwich", "curry", "stew", "pizza", "taco", "noodle",
    ]

    for query in search_queries:
        for page in range(1, min(max_pages + 1, 3)):
            print(f"\nSearching FNDDS: '{query}' page {page}...")

            result = search_fndds_foods(api_key, query=query, page_number=page, page_size=50)
            counters["api_calls"] += 1

            if not result or "foods" not in result:
                break

            foods = result["foods"]
            print(f"  Found {len(foods)} foods (total: {result.get('totalHits', 0):,})")

            for food in tqdm(foods, desc=f"  {query} p{page}"):
                _process_usda_food(food, conn, cursor, api_key, counters)

            conn.commit()
            time.sleep(REQUEST_DELAY_SECONDS)

    conn.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("USDA FoodData Central Enrichment Complete")
    print("=" * 60)
    print(f"  API calls made:           {api_calls}")
    print(f"  Dishes updated:           {dishes_updated}")
    print(f"  New dishes created:       {dishes_created}")
    print(f"  Ingredient->FDC mappings: {fdc_ids_set}")
    print(f"  Dish-ingredient links:    {ingredients_mapped}")

    # Show ingredients with USDA FDC IDs
    cursor.execute("SELECT COUNT(*) FROM ingredients WHERE usda_fdc_id IS NOT NULL")
    total_fdc = cursor.fetchone()[0]
    print(f"  Total ingredients with FDC ID: {total_fdc}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Enrich knowledge graph from USDA FoodData Central"
    )
    parser.add_argument("--db", default="knowledge-graph/food-knowledge.db",
                        help="Path to SQLite database")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help="USDA FDC API key (default: DEMO_KEY)")
    parser.add_argument("--max-pages", type=int, default=5,
                        help="Maximum search result pages to process")
    args = parser.parse_args()

    # Resolve path relative to project root
    db_path = args.db
    if not os.path.isabs(db_path):
        project_root = Path(__file__).parent.parent
        db_path = str(project_root / db_path)

    seed_from_usda(db_path, api_key=args.api_key, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
