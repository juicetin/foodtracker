#!/usr/bin/env python3
"""
Build a compact SQLite nutrition database from USDA FoodData Central CSV exports.

Converts USDA FDC CSV data into an optimized, FTS5-indexed SQLite database
suitable for on-device nutrition lookup in the mobile app.

Usage:
    python build_usda_db.py --input-dir /path/to/FoodData_Central_csv --output usda-core.db --pack-type core
    python build_usda_db.py --input-dir /path/to/FoodData_Central_csv --output usda-branded.db --pack-type branded

Pack types:
    core     -- Foundation + SR Legacy + FNDDS (general-purpose nutrition data)
    branded  -- Branded foods only (commercial products with UPC codes)
"""

import argparse
import csv
import logging
import os
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Data types included in each pack type
CORE_DATA_TYPES = {"foundation_food", "sr_legacy_food", "survey_fndds_food"}
BRANDED_DATA_TYPES = {"branded_food"}

SCHEMA_SQL = """
CREATE TABLE foods (
    fdc_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    food_category TEXT,
    data_type TEXT,
    publication_date TEXT
);

CREATE TABLE nutrients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    unit TEXT NOT NULL,
    nutrient_nbr TEXT
);

CREATE TABLE food_nutrients (
    food_id INTEGER REFERENCES foods(fdc_id),
    nutrient_id INTEGER REFERENCES nutrients(id),
    amount REAL NOT NULL,
    PRIMARY KEY (food_id, nutrient_id)
);

CREATE TABLE food_portions (
    id INTEGER PRIMARY KEY,
    food_id INTEGER REFERENCES foods(fdc_id),
    portion_description TEXT,
    gram_weight REAL,
    modifier TEXT
);

-- FTS5 full-text search on food descriptions and categories
CREATE VIRTUAL TABLE foods_fts USING fts5(
    description,
    food_category,
    content=foods,
    content_rowid=fdc_id
);

-- Indexes for common queries
CREATE INDEX idx_foods_category ON foods(food_category);
CREATE INDEX idx_foods_data_type ON foods(data_type);
CREATE INDEX idx_food_nutrients_food ON food_nutrients(food_id);
CREATE INDEX idx_food_nutrients_nutrient ON food_nutrients(nutrient_id);
"""


def get_data_types_for_pack(pack_type: str) -> set:
    """Return the set of USDA data_type values to include for a given pack type."""
    if pack_type == "core":
        return CORE_DATA_TYPES
    elif pack_type == "branded":
        return BRANDED_DATA_TYPES
    else:
        raise ValueError(f"Unknown pack type: {pack_type}. Must be 'core' or 'branded'.")


def load_food_categories(input_dir: str) -> dict:
    """Load food_category.csv and return {category_id: description} mapping."""
    categories = {}
    csv_path = os.path.join(input_dir, "food_category.csv")
    if not os.path.exists(csv_path):
        logger.warning("food_category.csv not found, food_category will be NULL")
        return categories

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories[row["id"]] = row["description"]

    logger.info(f"Loaded {len(categories)} food categories")
    return categories


def load_and_filter_foods(input_dir: str, allowed_types: set, categories: dict) -> dict:
    """
    Load food.csv, filter by allowed data types, return {fdc_id: food_row} dict.

    Each food_row is a dict with keys: fdc_id, description, food_category, data_type.
    """
    foods = {}
    csv_path = os.path.join(input_dir, "food.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_type = row["data_type"]
            if data_type in allowed_types:
                fdc_id = int(row["fdc_id"])
                category_id = row.get("food_category_id", "")
                foods[fdc_id] = {
                    "fdc_id": fdc_id,
                    "description": row["description"],
                    "food_category": categories.get(category_id),
                    "data_type": data_type,
                }

    logger.info(f"Loaded {len(foods)} foods (filtered to {allowed_types})")
    return foods


def load_nutrients(input_dir: str) -> dict:
    """Load nutrient.csv and return {nutrient_id: nutrient_row} dict."""
    nutrients = {}
    csv_path = os.path.join(input_dir, "nutrient.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nutrient_id = int(row["id"])
            nutrients[nutrient_id] = {
                "id": nutrient_id,
                "name": row["name"],
                "unit": row["unit_name"],
                "nutrient_nbr": row.get("nutrient_nbr"),
            }

    logger.info(f"Loaded {len(nutrients)} nutrient definitions")
    return nutrients


def insert_food_nutrients(conn: sqlite3.Connection, input_dir: str, valid_fdc_ids: set):
    """Load food_nutrient.csv and insert rows for valid foods only."""
    csv_path = os.path.join(input_dir, "food_nutrient.csv")
    inserted = 0
    batch = []
    batch_size = 5000

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fdc_id = int(row["fdc_id"])
            if fdc_id not in valid_fdc_ids:
                continue

            nutrient_id = int(row["nutrient_id"])
            amount_str = row.get("amount", "")
            if not amount_str:
                continue
            try:
                amount = float(amount_str)
            except ValueError:
                continue

            batch.append((fdc_id, nutrient_id, amount))
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT OR IGNORE INTO food_nutrients (food_id, nutrient_id, amount) VALUES (?, ?, ?)",
                    batch,
                )
                inserted += len(batch)
                batch = []

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO food_nutrients (food_id, nutrient_id, amount) VALUES (?, ?, ?)",
            batch,
        )
        inserted += len(batch)

    logger.info(f"Inserted {inserted} food-nutrient links")
    return inserted


def insert_food_portions(conn: sqlite3.Connection, input_dir: str, valid_fdc_ids: set):
    """Load food_portion.csv and insert rows for valid foods only."""
    csv_path = os.path.join(input_dir, "food_portion.csv")
    if not os.path.exists(csv_path):
        logger.warning("food_portion.csv not found, skipping portions")
        return 0

    inserted = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fdc_id = int(row["fdc_id"])
            if fdc_id not in valid_fdc_ids:
                continue

            portion_id = int(row["id"])
            description = row.get("portion_description", "")
            gram_weight_str = row.get("gram_weight", "")
            modifier = row.get("modifier", "")

            gram_weight = None
            if gram_weight_str:
                try:
                    gram_weight = float(gram_weight_str)
                except ValueError:
                    pass

            conn.execute(
                "INSERT OR IGNORE INTO food_portions (id, food_id, portion_description, gram_weight, modifier) "
                "VALUES (?, ?, ?, ?, ?)",
                (portion_id, fdc_id, description or None, gram_weight, modifier or None),
            )
            inserted += 1

    logger.info(f"Inserted {inserted} food portions")
    return inserted


def build_usda_db(input_dir: str, output_db: str, pack_type: str = "core"):
    """
    Build a USDA nutrition SQLite database from FDC CSV files.

    Args:
        input_dir: Path to directory containing USDA FDC CSV files
        output_db: Path for the output .db file
        pack_type: 'core' (Foundation+SR+FNDDS) or 'branded' (Branded only)
    """
    allowed_types = get_data_types_for_pack(pack_type)

    logger.info(f"Building USDA {pack_type} pack from {input_dir}")
    logger.info(f"Output: {output_db}")
    logger.info(f"Including data types: {allowed_types}")

    # Remove existing output to start clean
    if os.path.exists(output_db):
        os.remove(output_db)

    # Load categories first (needed for food enrichment)
    categories = load_food_categories(input_dir)

    # Load and filter foods
    foods = load_and_filter_foods(input_dir, allowed_types, categories)
    if not foods:
        logger.error("No foods matched the filter criteria")
        sys.exit(1)

    # Load nutrient definitions
    nutrients = load_nutrients(input_dir)

    # Create database and schema
    conn = sqlite3.connect(output_db)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.executescript(SCHEMA_SQL)

    # Insert foods
    for food in foods.values():
        conn.execute(
            "INSERT INTO foods (fdc_id, description, food_category, data_type) VALUES (?, ?, ?, ?)",
            (food["fdc_id"], food["description"], food["food_category"], food["data_type"]),
        )
    logger.info(f"Inserted {len(foods)} foods")

    # Insert nutrients
    for nutrient in nutrients.values():
        conn.execute(
            "INSERT INTO nutrients (id, name, unit, nutrient_nbr) VALUES (?, ?, ?, ?)",
            (nutrient["id"], nutrient["name"], nutrient["unit"], nutrient["nutrient_nbr"]),
        )
    logger.info(f"Inserted {len(nutrients)} nutrients")

    # Insert food-nutrient links
    valid_fdc_ids = set(foods.keys())
    nutrient_count = insert_food_nutrients(conn, input_dir, valid_fdc_ids)

    # Insert food portions
    portion_count = insert_food_portions(conn, input_dir, valid_fdc_ids)

    # Populate FTS5 index
    conn.execute("""
        INSERT INTO foods_fts (rowid, description, food_category)
        SELECT fdc_id, description, food_category FROM foods
    """)
    logger.info("Built FTS5 search index")

    # Set schema version
    conn.execute("PRAGMA user_version = 1")

    # Commit all data before running ANALYZE and VACUUM
    conn.commit()

    # Optimize -- VACUUM cannot run inside a transaction
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    logger.info("Ran ANALYZE and VACUUM")

    conn.close()

    # Log summary
    file_size = os.path.getsize(output_db)
    size_kb = file_size / 1024
    logger.info(f"Build complete: {len(foods)} foods, {nutrient_count} nutrient links, "
                f"{portion_count} portions, {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Build USDA FDC nutrition SQLite database from CSV exports"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to extracted USDA FDC CSV directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .db file path",
    )
    parser.add_argument(
        "--pack-type",
        choices=["core", "branded"],
        default="core",
        help="Pack type: 'core' (Foundation+SR+FNDDS) or 'branded' (Branded only)",
    )
    args = parser.parse_args()

    build_usda_db(args.input_dir, args.output, args.pack_type)


if __name__ == "__main__":
    main()
