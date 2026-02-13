#!/usr/bin/env python3
"""
Query API for the food knowledge graph.

Provides functions for:
- get_ingredients: Recursive CTE-based ingredient lookup for dishes (including variants)
- search_dish: FTS5 full-text search on dish names
- get_variants: Find all variant dishes sharing a canonical ancestor
- get_best_guess: Always-returns-something fuzzy matcher
- get_cuisine_stats: Cuisine distribution statistics
"""

import os
import sqlite3
from pathlib import Path


# Default database path
DEFAULT_DB = str(Path(__file__).parent / "food-knowledge.db")


def _get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get a database connection with row factory."""
    db_path = db_path or DEFAULT_DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_ingredients(
    dish_name: str,
    include_all: bool = False,
    db_path: str = None,
) -> list:
    """
    Get all ingredients for a dish using recursive CTE to traverse variant chain.

    If the dish is a variant of another dish (has canonical_id), this also
    returns the canonical dish's ingredients. Ingredients from the specific
    variant take priority over canonical ingredients.

    Args:
        dish_name: Name of the dish to look up
        include_all: If True, include non-nutrition-significant ingredients
        db_path: Path to SQLite database (default: food-knowledge.db)

    Returns:
        List of dicts with keys: name, weight_pct, typical_amount_g,
        usda_fdc_id, is_nutrition_significant, source, confidence
    """
    conn = _get_conn(db_path)
    cursor = conn.cursor()

    # Use recursive CTE to find the dish and all its ancestors via canonical_id
    query = """
    WITH RECURSIVE dish_chain(id, name, canonical_id, depth) AS (
        -- Base case: find the dish by name
        SELECT id, name, canonical_id, 0 AS depth
        FROM dishes
        WHERE name = ?

        UNION ALL

        -- Recursive case: follow canonical_id chain upward
        SELECT d.id, d.name, d.canonical_id, dc.depth + 1
        FROM dishes d
        JOIN dish_chain dc ON d.id = dc.canonical_id
        WHERE dc.canonical_id IS NOT NULL
          AND dc.depth < 10  -- prevent infinite loops
    )
    SELECT DISTINCT
        i.name,
        di.weight_pct,
        di.typical_amount_g,
        i.usda_fdc_id,
        di.is_nutrition_significant,
        di.source,
        di.confidence,
        dc.depth
    FROM dish_chain dc
    JOIN dish_ingredients di ON di.dish_id = dc.id
    JOIN ingredients i ON i.id = di.ingredient_id
    """

    if not include_all:
        query += " WHERE di.is_nutrition_significant = 1"

    query += " ORDER BY dc.depth ASC, di.weight_pct DESC"

    cursor.execute(query, (dish_name,))
    rows = cursor.fetchall()
    conn.close()

    # Deduplicate: if same ingredient appears at multiple depths,
    # keep the one from the most specific dish (lowest depth)
    seen = set()
    results = []
    for row in rows:
        name = row["name"]
        if name not in seen:
            seen.add(name)
            results.append({
                "name": name,
                "weight_pct": row["weight_pct"],
                "typical_amount_g": row["typical_amount_g"],
                "usda_fdc_id": row["usda_fdc_id"],
                "is_nutrition_significant": bool(row["is_nutrition_significant"]),
                "source": row["source"],
                "confidence": row["confidence"],
            })

    return results


def search_dish(query: str, limit: int = 10, db_path: str = None) -> list:
    """
    Search for dishes using FTS5 full-text search.

    Supports prefix matching (e.g., "fried ri*" matches "fried rice").

    Args:
        query: Search query string
        limit: Maximum number of results (default 10)
        db_path: Path to SQLite database

    Returns:
        List of dicts with keys: id, name, cuisine, confidence
    """
    conn = _get_conn(db_path)
    cursor = conn.cursor()

    results = []

    # Try FTS5 search first
    try:
        # Add prefix matching if query doesn't already have special syntax
        fts_query = query
        if not any(c in query for c in ["*", '"', "AND", "OR", "NOT"]):
            # Add prefix match to last word
            words = query.strip().split()
            if words:
                words[-1] = words[-1] + "*"
                fts_query = " ".join(words)

        cursor.execute(
            """SELECT d.id, d.name, d.cuisine, d.confidence
               FROM dishes_fts
               JOIN dishes d ON d.id = dishes_fts.rowid
               WHERE dishes_fts MATCH ?
               LIMIT ?""",
            (fts_query, limit),
        )
        results = [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        pass

    # Fall back to LIKE search if FTS5 returns nothing
    if not results:
        cursor.execute(
            """SELECT id, name, cuisine, confidence
               FROM dishes
               WHERE name LIKE ?
               ORDER BY confidence DESC
               LIMIT ?""",
            (f"%{query}%", limit),
        )
        results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return results


def get_variants(dish_name: str, db_path: str = None) -> list:
    """
    Find all dishes that share the same canonical ancestor or are
    variants of the given dish.

    Args:
        dish_name: Name of the dish
        db_path: Path to SQLite database

    Returns:
        List of dicts with keys: id, name, cuisine
    """
    conn = _get_conn(db_path)
    cursor = conn.cursor()

    # First find the dish and its canonical root
    cursor.execute(
        """WITH RECURSIVE root_chain(id, name, canonical_id) AS (
               SELECT id, name, canonical_id FROM dishes WHERE name = ?
               UNION ALL
               SELECT d.id, d.name, d.canonical_id
               FROM dishes d JOIN root_chain rc ON d.id = rc.canonical_id
               WHERE rc.canonical_id IS NOT NULL
           )
           SELECT id FROM root_chain ORDER BY canonical_id NULLS LAST LIMIT 1""",
        (dish_name,),
    )
    root = cursor.fetchone()

    if not root:
        conn.close()
        return []

    root_id = root["id"]

    # Find all dishes that point to this root (or to the dish itself)
    # Also include the root dish and the original dish
    cursor.execute(
        """SELECT DISTINCT d.id, d.name, d.cuisine
           FROM dishes d
           WHERE d.canonical_id = ?
              OR d.id = ?
              OR d.canonical_id = (SELECT id FROM dishes WHERE name = ?)
              OR d.id = (SELECT canonical_id FROM dishes WHERE name = ?)
           ORDER BY d.name""",
        (root_id, root_id, dish_name, dish_name),
    )

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Exclude the queried dish itself from variants
    results = [r for r in results if r["name"] != dish_name]
    return results


def get_best_guess(partial_name: str, db_path: str = None) -> dict:
    """
    Always returns a dish suggestion, never None/empty.

    Matching strategy:
    1. Exact match
    2. FTS5 prefix search
    3. LIKE substring match
    4. Fallback: return the dish with highest confidence

    Args:
        partial_name: Partial or full dish name to match
        db_path: Path to SQLite database

    Returns:
        Dict with keys: name, cuisine, confidence, match_type
    """
    conn = _get_conn(db_path)
    cursor = conn.cursor()

    # 1. Try exact match
    cursor.execute(
        "SELECT name, cuisine, confidence FROM dishes WHERE name = ?",
        (partial_name.lower().strip(),),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "confidence": row["confidence"],
            "match_type": "exact",
        }

    # 2. Try FTS5 prefix search
    try:
        words = partial_name.strip().split()
        if words:
            words[-1] = words[-1] + "*"
            fts_query = " ".join(words)
            cursor.execute(
                """SELECT d.name, d.cuisine, d.confidence
                   FROM dishes_fts
                   JOIN dishes d ON d.id = dishes_fts.rowid
                   WHERE dishes_fts MATCH ?
                   ORDER BY d.confidence DESC
                   LIMIT 1""",
                (fts_query,),
            )
            row = cursor.fetchone()
            if row:
                conn.close()
                return {
                    "name": row["name"],
                    "cuisine": row["cuisine"],
                    "confidence": row["confidence"],
                    "match_type": "prefix",
                }
    except sqlite3.OperationalError:
        pass

    # 3. Try LIKE substring match
    cursor.execute(
        """SELECT name, cuisine, confidence FROM dishes
           WHERE name LIKE ?
           ORDER BY confidence DESC
           LIMIT 1""",
        (f"%{partial_name}%",),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "confidence": row["confidence"],
            "match_type": "fuzzy",
        }

    # 4. Try matching individual words
    for word in partial_name.strip().split():
        if len(word) < 3:
            continue
        cursor.execute(
            """SELECT name, cuisine, confidence FROM dishes
               WHERE name LIKE ?
               ORDER BY confidence DESC
               LIMIT 1""",
            (f"%{word}%",),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return {
                "name": row["name"],
                "cuisine": row["cuisine"],
                "confidence": row["confidence"],
                "match_type": "fuzzy",
            }

    # 5. Absolute fallback: return highest-confidence dish
    cursor.execute(
        "SELECT name, cuisine, confidence FROM dishes ORDER BY confidence DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "confidence": row["confidence"],
            "match_type": "fuzzy",
        }

    # Should never reach here if DB has any data
    return {
        "name": "unknown",
        "cuisine": "Other",
        "confidence": 0.0,
        "match_type": "fuzzy",
    }


def get_cuisine_stats(db_path: str = None) -> dict:
    """
    Get cuisine distribution statistics.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dict mapping cuisine name to {dish_count, ingredient_count}
    """
    conn = _get_conn(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               d.cuisine,
               COUNT(DISTINCT d.id) AS dish_count,
               COUNT(DISTINCT di.ingredient_id) AS ingredient_count
           FROM dishes d
           LEFT JOIN dish_ingredients di ON di.dish_id = d.id
           GROUP BY d.cuisine
           ORDER BY dish_count DESC"""
    )

    stats = {}
    for row in cursor.fetchall():
        stats[row["cuisine"]] = {
            "dish_count": row["dish_count"],
            "ingredient_count": row["ingredient_count"],
        }

    conn.close()
    return stats


if __name__ == "__main__":
    # Demo usage
    print("=== Food Knowledge Graph Query API ===\n")

    print("--- get_ingredients('carbonara') ---")
    for ing in get_ingredients("carbonara"):
        print(f"  {ing['name']:20s}  {ing['weight_pct']:.2f}  {ing['typical_amount_g']:.0f}g")

    print("\n--- search_dish('pad') ---")
    for d in search_dish("pad"):
        print(f"  {d['name']:30s}  {d['cuisine']:15s}  conf={d['confidence']:.2f}")

    print("\n--- get_variants('nasi goreng') ---")
    for v in get_variants("nasi goreng"):
        print(f"  {v['name']:30s}  {v['cuisine']}")

    print("\n--- get_best_guess('xyznonexistent') ---")
    guess = get_best_guess("xyznonexistent")
    print(f"  {guess}")

    print("\n--- get_cuisine_stats() ---")
    for cuisine, stats in get_cuisine_stats().items():
        print(f"  {cuisine:20s}: {stats['dish_count']:>4} dishes, {stats['ingredient_count']:>4} ingredients")
