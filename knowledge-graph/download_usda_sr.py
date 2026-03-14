#!/usr/bin/env python3
"""
Download and parse the full USDA FoodData Central SR Legacy dataset.

Downloads the official USDA SR Legacy CSV ZIP, extracts food.csv,
food_nutrient.csv, and nutrient.csv, then merges them into a single
processed CSV with macros and micronutrients.

Output: knowledge-graph/data/sr_legacy_full.csv
Expected: ~7,793 foods with full nutrient profiles.

Usage:
    python download_usda_sr.py
"""

import csv
import io
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    import urllib.request
except ImportError:
    pass

KG_DIR = Path(__file__).parent
DATA_DIR = KG_DIR / "data"
OUTPUT_CSV = DATA_DIR / "sr_legacy_full.csv"

USDA_ZIP_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"

# Key USDA nutrient IDs to extract
NUTRIENT_IDS = {
    1008: "calories",         # Energy (kcal)
    1003: "protein",          # Protein (g)
    1004: "fat",              # Total lipid/fat (g)
    1005: "carbs",            # Carbohydrate (g)
    1079: "fiber",            # Fiber, total dietary (g)
    1106: "vitamin_a_ug",     # Vitamin A, RAE (ug)
    1162: "vitamin_c_mg",     # Vitamin C (mg)
    1114: "vitamin_d_ug",     # Vitamin D (D2+D3) (ug)
    1087: "calcium_mg",       # Calcium (mg)
    1089: "iron_mg",          # Iron (mg)
    1092: "potassium_mg",     # Potassium (mg)
    1093: "sodium_mg",        # Sodium (mg)
    1095: "zinc_mg",          # Zinc (mg)
    1090: "magnesium_mg",     # Magnesium (mg)
}

OUTPUT_COLUMNS = [
    "fdc_id", "description", "food_group",
    "calories", "protein", "fat", "carbs", "fiber",
    "vitamin_a_ug", "vitamin_c_mg", "vitamin_d_ug",
    "calcium_mg", "iron_mg", "potassium_mg", "sodium_mg",
    "zinc_mg", "magnesium_mg",
]


def download_and_extract():
    """Download the USDA SR Legacy ZIP and extract needed CSVs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "sr_legacy.zip"

    if not zip_path.exists():
        print(f"Downloading USDA SR Legacy from {USDA_ZIP_URL}...")
        try:
            urllib.request.urlretrieve(USDA_ZIP_URL, str(zip_path))
        except Exception as e:
            print(f"Download failed: {e}")
            print("Trying alternative download method...")
            import subprocess
            result = subprocess.run(
                ["curl", "-L", "-o", str(zip_path), USDA_ZIP_URL],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download USDA SR Legacy: {result.stderr}")
        print(f"Downloaded to {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"Using cached ZIP: {zip_path}")

    return zip_path


def parse_zip(zip_path: Path) -> dict:
    """Parse the ZIP and extract food, nutrient, and food_nutrient data."""
    foods = {}        # fdc_id -> {description, food_group}
    nutrients = {}    # nutrient_id -> name (for debugging)
    food_nutrients = defaultdict(dict)  # fdc_id -> {column_name: amount}

    print("Extracting and parsing CSV files from ZIP...")

    with zipfile.ZipFile(zip_path) as zf:
        # List contents to find the right files
        names = zf.namelist()
        food_csv = None
        nutrient_csv = None
        food_nutrient_csv = None
        food_category_csv = None

        for name in names:
            lower = name.lower()
            basename = os.path.basename(lower)
            if basename == "food.csv":
                food_csv = name
            elif basename == "nutrient.csv":
                nutrient_csv = name
            elif basename == "food_nutrient.csv":
                food_nutrient_csv = name
            elif basename == "food_category.csv":
                food_category_csv = name

        if not food_csv or not food_nutrient_csv:
            print(f"ZIP contents: {names}")
            raise RuntimeError("Could not find required CSV files in ZIP")

        # Parse food_category.csv for food group names
        food_groups = {}
        if food_category_csv:
            with zf.open(food_category_csv) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    cat_id = row.get("id", "")
                    desc = row.get("description", "")
                    if cat_id and desc:
                        food_groups[cat_id] = desc

        # Parse food.csv
        print(f"  Parsing {food_csv}...")
        with zf.open(food_csv) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                fdc_id = row.get("fdc_id", "")
                desc = row.get("description", "")
                cat_id = row.get("food_category_id", "")
                if fdc_id and desc:
                    foods[fdc_id] = {
                        "description": desc,
                        "food_group": food_groups.get(cat_id, ""),
                    }
        print(f"    Found {len(foods)} foods")

        # Parse nutrient.csv (for debugging/verification)
        if nutrient_csv:
            with zf.open(nutrient_csv) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    nid = row.get("id", "")
                    nname = row.get("name", "")
                    if nid:
                        nutrients[nid] = nname

        # Parse food_nutrient.csv (the big one)
        print(f"  Parsing {food_nutrient_csv} (this may take a moment)...")
        nutrient_id_strs = {str(k): v for k, v in NUTRIENT_IDS.items()}

        with zf.open(food_nutrient_csv) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            row_count = 0
            matched = 0
            for row in reader:
                row_count += 1
                nid = row.get("nutrient_id", "")
                if nid in nutrient_id_strs:
                    fdc_id = row.get("fdc_id", "")
                    amount = row.get("amount", "")
                    if fdc_id and fdc_id in foods:
                        try:
                            food_nutrients[fdc_id][nutrient_id_strs[nid]] = float(amount) if amount else 0.0
                            matched += 1
                        except (ValueError, TypeError):
                            pass

                if row_count % 500000 == 0:
                    print(f"    Processed {row_count:,} nutrient rows ({matched:,} matched)...")

        print(f"    Total: {row_count:,} nutrient rows, {matched:,} matched to target nutrients")

    return foods, food_nutrients


def write_csv(foods: dict, food_nutrients: dict):
    """Write the merged CSV output."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for fdc_id, food_info in foods.items():
            nutrient_vals = food_nutrients.get(fdc_id, {})

            row = {
                "fdc_id": fdc_id,
                "description": food_info["description"],
                "food_group": food_info["food_group"],
            }

            # Fill nutrient columns
            for col_name in OUTPUT_COLUMNS[3:]:  # Skip fdc_id, description, food_group
                val = nutrient_vals.get(col_name)
                row[col_name] = round(val, 2) if val is not None else ""

            writer.writerow(row)
            count += 1

    print(f"\nWrote {count} foods to {OUTPUT_CSV}")
    print(f"File size: {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")
    return count


def main():
    """Download, parse, and write USDA SR Legacy full dataset."""
    print("=" * 60)
    print("USDA SR Legacy Full Dataset Downloader")
    print("=" * 60)

    zip_path = download_and_extract()
    foods, food_nutrients = parse_zip(zip_path)
    count = write_csv(foods, food_nutrients)

    print(f"\nDone! {count} USDA SR Legacy foods with micronutrients.")
    return count


if __name__ == "__main__":
    main()
