#!/usr/bin/env python3
"""
Seed the food knowledge graph with multilingual dish aliases from
HuggingFace WorldCuisines food-kb dataset.

WorldCuisines provides dish names with aliases in 30+ languages/scripts,
cuisine classifications, and geographic metadata for 2400+ dishes.

Usage:
    Called from build_kg.py as part of the pipeline.
    python seed_worldcuisines.py [--db food-knowledge.db]
"""

import ast
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

KG_DIR = Path(__file__).parent


def _parse_alias_field(alias_str: str) -> list[tuple[str, str]]:
    """Parse the WorldCuisines alias field into (alias, language) pairs.

    The alias field is a JSON-like list of dicts:
      [{'alias_text': 'language'}, ...]
    Entries with 'no_alias' keys are skipped.

    Returns list of (alias_text, language) tuples.
    """
    if not alias_str or alias_str.strip() in ("", "[]"):
        return []

    try:
        entries = ast.literal_eval(alias_str)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(entries, list):
        return []

    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for alias_text, language in entry.items():
            # Skip no_alias markers
            if alias_text == "no_alias" or alias_text == "name":
                continue
            if not isinstance(alias_text, str) or not alias_text.strip():
                continue
            # Language might be a string, True, list, etc.
            if isinstance(language, str):
                lang = language.strip()
            else:
                lang = "unknown"
            if lang in ("not_stated", ""):
                lang = "unknown"
            results.append((alias_text.strip(), lang))

    return results


def _parse_json_list(s: str) -> list[str]:
    """Parse a JSON-encoded list of strings, handling malformed input."""
    if not s or s.strip() in ("", "[]"):
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v).strip() for v in val if v]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def seed_worldcuisines(conn: sqlite3.Connection) -> dict:
    """Seed WorldCuisines multilingual aliases and new dishes.

    For each WorldCuisines entry:
    1. Match against existing KG dishes by normalized name
    2. Add multilingual aliases to matched dishes
    3. Create new dish entries for unmatched WorldCuisines dishes
    4. Add aliases for new dishes too

    Returns dict with stats.
    """
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        "worldcuisines/food-kb", "worldcuisines.csv", repo_type="dataset"
    )

    cursor = conn.cursor()

    # Build lookup of existing dishes by normalized name
    cursor.execute("SELECT id, canonical_name FROM dish")
    existing_dishes = {}
    for row in cursor.fetchall():
        existing_dishes[row[1].lower().strip()] = row[0]

    # Also build a lookup for cuisine -> id
    cuisine_ids = {}
    cursor.execute("SELECT id, name FROM cuisine")
    for row in cursor.fetchall():
        cuisine_ids[row[1].lower()] = row[0]

    # Category lookup
    category_ids = {}
    cursor.execute("SELECT id, cuisine_id, name FROM dish_category")
    for row in cursor.fetchall():
        for cname, cid in cuisine_ids.items():
            if row[1] == cid:
                category_ids[f"{cname}_general"] = row[0]

    stats = {
        "matched": 0,
        "new_dishes": 0,
        "aliases_added": 0,
        "languages": set(),
    }

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue

            # Normalize name for matching
            canonical = name.lower().strip().replace("-", " ").replace("_", " ")

            # Parse aliases
            aliases = _parse_alias_field(row.get("Alias", ""))

            # Parse cuisine/area metadata
            cuisines = _parse_json_list(row.get("Cuisines", ""))
            area_list = _parse_json_list(row.get("Area", ""))
            description = row.get("Text Description", "").strip()

            # Try to match existing dish
            dish_id = existing_dishes.get(canonical)

            if dish_id is None:
                # Try fuzzy match: remove common suffixes/prefixes
                for variant in [
                    canonical,
                    canonical.replace(" ", ""),
                    re.sub(r"\s+(dish|soup|salad|curry|stew|rice|bread|cake)$", "", canonical),
                ]:
                    if variant in existing_dishes:
                        dish_id = existing_dishes[variant]
                        break

            if dish_id is None:
                # Try LIKE match for partial names
                cursor.execute(
                    "SELECT id FROM dish WHERE canonical_name LIKE ? LIMIT 1",
                    (f"%{canonical}%",),
                )
                result = cursor.fetchone()
                if result:
                    dish_id = result[0]

            if dish_id:
                stats["matched"] += 1
            else:
                # Create new dish from WorldCuisines
                cuisine_name = cuisines[0].lower() if cuisines else "international"

                if cuisine_name not in cuisine_ids:
                    cursor.execute(
                        "INSERT OR IGNORE INTO cuisine (name, region) VALUES (?, ?)",
                        (cuisine_name, area_list[0] if area_list else None),
                    )
                    cursor.execute(
                        "SELECT id FROM cuisine WHERE name = ?", (cuisine_name,)
                    )
                    cuisine_ids[cuisine_name] = cursor.fetchone()[0]

                cuisine_id = cuisine_ids[cuisine_name]
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

                cursor.execute(
                    "INSERT OR IGNORE INTO dish (category_id, canonical_name, description, source, confidence) "
                    "VALUES (?, ?, ?, 'worldcuisines', 0.7)",
                    (category_id, canonical, description[:500] if description else None),
                )
                if cursor.rowcount > 0:
                    dish_id = cursor.lastrowid
                    existing_dishes[canonical] = dish_id
                    stats["new_dishes"] += 1
                else:
                    # Already existed (race/duplicate)
                    cursor.execute(
                        "SELECT id FROM dish WHERE canonical_name = ?", (canonical,)
                    )
                    result = cursor.fetchone()
                    if result:
                        dish_id = result[0]

            if not dish_id:
                continue

            # Add aliases
            for alias_text, language in aliases:
                # Skip if alias is same as canonical name
                if alias_text.lower().strip() == canonical:
                    continue

                cursor.execute(
                    "INSERT OR IGNORE INTO dish_alias (dish_id, alias, language, alias_type) "
                    "VALUES (?, ?, ?, 'translation')",
                    (dish_id, alias_text, language),
                )
                if cursor.rowcount > 0:
                    stats["aliases_added"] += 1
                    stats["languages"].add(language)

            # Also add the original name as an English alias if different from canonical
            if name.lower().strip() != canonical:
                cursor.execute(
                    "INSERT OR IGNORE INTO dish_alias (dish_id, alias, language, alias_type) "
                    "VALUES (?, ?, 'en', 'original')",
                    (dish_id, name),
                )
                if cursor.rowcount > 0:
                    stats["aliases_added"] += 1

    conn.commit()

    lang_count = len(stats["languages"])
    print(
        f"  WorldCuisines: {stats['matched']} matched, "
        f"{stats['new_dishes']} new dishes, "
        f"{stats['aliases_added']} aliases across {lang_count} languages"
    )

    return {
        "matched": stats["matched"],
        "new_dishes": stats["new_dishes"],
        "aliases_added": stats["aliases_added"],
        "language_count": lang_count,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seed WorldCuisines aliases")
    parser.add_argument("--db", default=None, help="Path to knowledge graph database")
    args = parser.parse_args()

    db_path = args.db or str(KG_DIR / "food-knowledge.db")

    conn = sqlite3.connect(db_path)
    result = seed_worldcuisines(conn)
    conn.close()

    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
