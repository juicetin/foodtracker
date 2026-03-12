#!/usr/bin/env python3
"""
Tests for USDA FDC SQLite build pipeline.

Uses small fixture CSV data (10-20 foods, 5 nutrients) to validate:
- Database creation and schema correctness
- FTS5 full-text search
- Data type filtering (core vs branded)
- PRAGMA user_version
- ANALYZE/VACUUM (file size sanity)
"""

import csv
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path so we can import the build module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fixture_dir(tmp_path):
    """Create small fixture CSV files mimicking USDA FDC download format."""

    # food.csv -- 15 foods across data types
    foods = [
        {"fdc_id": "100", "data_type": "foundation_food", "description": "Chicken breast raw", "food_category_id": "1"},
        {"fdc_id": "101", "data_type": "foundation_food", "description": "Broccoli raw", "food_category_id": "2"},
        {"fdc_id": "102", "data_type": "foundation_food", "description": "Salmon Atlantic raw", "food_category_id": "3"},
        {"fdc_id": "200", "data_type": "sr_legacy_food", "description": "White rice cooked", "food_category_id": "4"},
        {"fdc_id": "201", "data_type": "sr_legacy_food", "description": "Whole wheat bread", "food_category_id": "5"},
        {"fdc_id": "202", "data_type": "sr_legacy_food", "description": "Cheddar cheese", "food_category_id": "6"},
        {"fdc_id": "300", "data_type": "survey_fndds_food", "description": "Apple juice unsweetened", "food_category_id": "7"},
        {"fdc_id": "301", "data_type": "survey_fndds_food", "description": "Banana raw", "food_category_id": "8"},
        {"fdc_id": "302", "data_type": "survey_fndds_food", "description": "Orange juice fresh", "food_category_id": "7"},
        {"fdc_id": "400", "data_type": "branded_food", "description": "Coca-Cola Classic", "food_category_id": "9"},
        {"fdc_id": "401", "data_type": "branded_food", "description": "Doritos Nacho Cheese", "food_category_id": "10"},
        {"fdc_id": "402", "data_type": "branded_food", "description": "Kind Bar Dark Chocolate", "food_category_id": "10"},
        {"fdc_id": "500", "data_type": "foundation_food", "description": "Egg whole raw", "food_category_id": "11"},
        {"fdc_id": "501", "data_type": "sr_legacy_food", "description": "Olive oil", "food_category_id": "12"},
        {"fdc_id": "502", "data_type": "survey_fndds_food", "description": "Greek yogurt plain", "food_category_id": "6"},
    ]
    with open(tmp_path / "food.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fdc_id", "data_type", "description", "food_category_id"])
        writer.writeheader()
        writer.writerows(foods)

    # food_category.csv
    categories = [
        {"id": "1", "description": "Poultry Products"},
        {"id": "2", "description": "Vegetables and Vegetable Products"},
        {"id": "3", "description": "Finfish and Shellfish Products"},
        {"id": "4", "description": "Cereal Grains and Pasta"},
        {"id": "5", "description": "Baked Products"},
        {"id": "6", "description": "Dairy and Egg Products"},
        {"id": "7", "description": "Beverages"},
        {"id": "8", "description": "Fruits and Fruit Juices"},
        {"id": "9", "description": "Beverages"},
        {"id": "10", "description": "Snacks"},
        {"id": "11", "description": "Dairy and Egg Products"},
        {"id": "12", "description": "Fats and Oils"},
    ]
    with open(tmp_path / "food_category.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "description"])
        writer.writeheader()
        writer.writerows(categories)

    # nutrient.csv -- 5 key nutrients
    nutrients = [
        {"id": "1008", "name": "Energy", "unit_name": "kcal", "nutrient_nbr": "208"},
        {"id": "1003", "name": "Protein", "unit_name": "g", "nutrient_nbr": "203"},
        {"id": "1005", "name": "Carbohydrate, by difference", "unit_name": "g", "nutrient_nbr": "205"},
        {"id": "1004", "name": "Total lipid (fat)", "unit_name": "g", "nutrient_nbr": "204"},
        {"id": "1079", "name": "Fiber, total dietary", "unit_name": "g", "nutrient_nbr": "291"},
    ]
    with open(tmp_path / "nutrient.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "unit_name", "nutrient_nbr"])
        writer.writeheader()
        writer.writerows(nutrients)

    # food_nutrient.csv -- nutrient values for foods
    food_nutrients = [
        # Chicken breast
        {"id": "1", "fdc_id": "100", "nutrient_id": "1008", "amount": "165"},
        {"id": "2", "fdc_id": "100", "nutrient_id": "1003", "amount": "31.0"},
        {"id": "3", "fdc_id": "100", "nutrient_id": "1005", "amount": "0.0"},
        {"id": "4", "fdc_id": "100", "nutrient_id": "1004", "amount": "3.6"},
        # Broccoli
        {"id": "5", "fdc_id": "101", "nutrient_id": "1008", "amount": "34"},
        {"id": "6", "fdc_id": "101", "nutrient_id": "1003", "amount": "2.8"},
        {"id": "7", "fdc_id": "101", "nutrient_id": "1005", "amount": "6.6"},
        {"id": "8", "fdc_id": "101", "nutrient_id": "1004", "amount": "0.4"},
        {"id": "9", "fdc_id": "101", "nutrient_id": "1079", "amount": "2.6"},
        # Banana
        {"id": "10", "fdc_id": "301", "nutrient_id": "1008", "amount": "89"},
        {"id": "11", "fdc_id": "301", "nutrient_id": "1003", "amount": "1.1"},
        {"id": "12", "fdc_id": "301", "nutrient_id": "1005", "amount": "22.8"},
        {"id": "13", "fdc_id": "301", "nutrient_id": "1004", "amount": "0.3"},
        # Coca-Cola (branded)
        {"id": "14", "fdc_id": "400", "nutrient_id": "1008", "amount": "42"},
        {"id": "15", "fdc_id": "400", "nutrient_id": "1003", "amount": "0.0"},
        {"id": "16", "fdc_id": "400", "nutrient_id": "1005", "amount": "10.6"},
        {"id": "17", "fdc_id": "400", "nutrient_id": "1004", "amount": "0.0"},
        # White rice
        {"id": "18", "fdc_id": "200", "nutrient_id": "1008", "amount": "130"},
        {"id": "19", "fdc_id": "200", "nutrient_id": "1003", "amount": "2.7"},
        {"id": "20", "fdc_id": "200", "nutrient_id": "1005", "amount": "28.2"},
        {"id": "21", "fdc_id": "200", "nutrient_id": "1004", "amount": "0.3"},
    ]
    with open(tmp_path / "food_nutrient.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "fdc_id", "nutrient_id", "amount"])
        writer.writeheader()
        writer.writerows(food_nutrients)

    # food_portion.csv
    portions = [
        {"id": "1", "fdc_id": "100", "portion_description": "1 breast, bone and skin removed", "gram_weight": "172", "modifier": "breast"},
        {"id": "2", "fdc_id": "101", "portion_description": "1 cup, chopped", "gram_weight": "91", "modifier": "cup chopped"},
        {"id": "3", "fdc_id": "301", "portion_description": "1 medium (7\" to 7-7/8\" long)", "gram_weight": "118", "modifier": "medium"},
        {"id": "4", "fdc_id": "200", "portion_description": "1 cup", "gram_weight": "158", "modifier": "cup"},
        {"id": "5", "fdc_id": "400", "portion_description": "1 can (12 fl oz)", "gram_weight": "370", "modifier": "can"},
    ]
    with open(tmp_path / "food_portion.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "fdc_id", "portion_description", "gram_weight", "modifier"])
        writer.writeheader()
        writer.writerows(portions)

    return tmp_path


@pytest.fixture
def output_db(tmp_path):
    """Return a path for the output database."""
    return str(tmp_path / "usda_nutrition.db")


class TestUsdaBuildCoreType:
    """Test core pack build (Foundation + SR Legacy + FNDDS)."""

    def test_creates_valid_sqlite_database(self, fixture_dir, output_db):
        """Build script creates a valid SQLite database from test CSV data."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        assert os.path.exists(output_db)
        conn = sqlite3.connect(output_db)
        # Should not raise
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM foods")
        count = cursor.fetchone()[0]
        assert count > 0
        conn.close()

    def test_foods_table_has_expected_entries(self, fixture_dir, output_db):
        """Foods table contains expected entries with fdc_id, description, food_category, data_type."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("SELECT fdc_id, description, food_category, data_type FROM foods ORDER BY fdc_id")
        rows = cursor.fetchall()

        # Core pack: foundation (100,101,102,500) + sr_legacy (200,201,202,501) + fndds (300,301,302,502) = 12
        assert len(rows) == 12

        # Check specific entries
        fdc_ids = [r[0] for r in rows]
        assert 100 in fdc_ids  # foundation
        assert 200 in fdc_ids  # sr_legacy
        assert 301 in fdc_ids  # fndds

        # Check a specific food has all fields
        chicken = [r for r in rows if r[0] == 100][0]
        assert chicken[1] == "Chicken breast raw"
        assert chicken[2] == "Poultry Products"
        assert chicken[3] is not None  # data_type present

        conn.close()

    def test_food_nutrients_links_foods_to_nutrients(self, fixture_dir, output_db):
        """food_nutrients table links foods to nutrients with per-100g amounts."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()

        # Check food_nutrients has entries
        cursor.execute("SELECT COUNT(*) FROM food_nutrients")
        count = cursor.fetchone()[0]
        assert count > 0

        # Check chicken breast has calories
        cursor.execute("""
            SELECT fn.amount, n.name, n.unit
            FROM food_nutrients fn
            JOIN nutrients n ON fn.nutrient_id = n.id
            WHERE fn.food_id = 100 AND n.name = 'Energy'
        """)
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 165.0  # 165 kcal per 100g

        conn.close()

    def test_fts5_index_returns_results_for_partial_queries(self, fixture_dir, output_db):
        """FTS5 full-text search returns results for partial food name queries."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()

        # Search for "chicken"
        cursor.execute("""
            SELECT f.fdc_id, f.description
            FROM foods_fts fts
            JOIN foods f ON f.fdc_id = fts.rowid
            WHERE foods_fts MATCH 'chicken'
        """)
        results = cursor.fetchall()
        assert len(results) >= 1
        assert any("Chicken" in r[1] for r in results)

        # Search for "raw" should match multiple foods
        cursor.execute("""
            SELECT f.fdc_id, f.description
            FROM foods_fts fts
            JOIN foods f ON f.fdc_id = fts.rowid
            WHERE foods_fts MATCH 'raw'
        """)
        results = cursor.fetchall()
        assert len(results) >= 2

        conn.close()

    def test_pragma_user_version_is_set(self, fixture_dir, output_db):
        """Output database has PRAGMA user_version set."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA user_version")
        version = cursor.fetchone()[0]
        assert version == 1
        conn.close()

    def test_core_excludes_branded_foods(self, fixture_dir, output_db):
        """Foundation + SR Legacy + FNDDS data types are included in core pack; Branded excluded."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()

        # Should NOT have branded foods (fdc_ids 400, 401, 402)
        cursor.execute("SELECT fdc_id FROM foods WHERE fdc_id IN (400, 401, 402)")
        branded = cursor.fetchall()
        assert len(branded) == 0

        # Should have foundation foods
        cursor.execute("SELECT COUNT(*) FROM foods WHERE data_type = 'foundation_food'")
        foundation_count = cursor.fetchone()[0]
        assert foundation_count == 4  # 100, 101, 102, 500

        # Should have sr_legacy foods
        cursor.execute("SELECT COUNT(*) FROM foods WHERE data_type = 'sr_legacy_food'")
        sr_count = cursor.fetchone()[0]
        assert sr_count == 4  # 200, 201, 202, 501

        # Should have fndds foods
        cursor.execute("SELECT COUNT(*) FROM foods WHERE data_type = 'survey_fndds_food'")
        fndds_count = cursor.fetchone()[0]
        assert fndds_count == 4  # 300, 301, 302, 502

        conn.close()

    def test_vacuum_and_analyze_run(self, fixture_dir, output_db):
        """VACUUM and ANALYZE are run (file size check -- DB file is a single contiguous file)."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        # A VACUUMed DB should be reasonably sized -- non-zero and no journal files
        db_size = os.path.getsize(output_db)
        assert db_size > 0

        # No WAL or journal files after VACUUM
        assert not os.path.exists(output_db + "-wal")
        assert not os.path.exists(output_db + "-shm")
        assert not os.path.exists(output_db + "-journal")

        # ANALYZE creates sqlite_stat1 table
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'")
        stat_table = cursor.fetchone()
        assert stat_table is not None
        conn.close()

    def test_food_portions_table_populated(self, fixture_dir, output_db):
        """food_portions table has serving size data."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()

        # Only core foods should have portions
        cursor.execute("SELECT COUNT(*) FROM food_portions")
        count = cursor.fetchone()[0]
        assert count > 0

        # Chicken breast portion
        cursor.execute("SELECT portion_description, gram_weight FROM food_portions WHERE food_id = 100")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == 172.0

        # Branded portion (400) should NOT be present in core build
        cursor.execute("SELECT COUNT(*) FROM food_portions WHERE food_id = 400")
        branded_portions = cursor.fetchone()[0]
        assert branded_portions == 0

        conn.close()

    def test_indexes_created(self, fixture_dir, output_db):
        """Required indexes are created."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        expected = {
            "idx_foods_category",
            "idx_foods_data_type",
            "idx_food_nutrients_food",
            "idx_food_nutrients_nutrient",
        }
        for idx in expected:
            assert idx in indexes, f"Missing index: {idx}"

        conn.close()

    def test_nutrients_table_populated(self, fixture_dir, output_db):
        """Nutrients table has nutrient definitions."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="core")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nutrients")
        count = cursor.fetchone()[0]
        assert count == 5

        cursor.execute("SELECT name, unit FROM nutrients WHERE id = 1008")
        row = cursor.fetchone()
        assert row[0] == "Energy"
        assert row[1] == "kcal"

        conn.close()


class TestUsdaBuildBrandedType:
    """Test branded pack build."""

    def test_branded_only_includes_branded_foods(self, fixture_dir, output_db):
        """Branded pack includes only branded data type."""
        from build_usda_db import build_usda_db

        build_usda_db(str(fixture_dir), output_db, pack_type="branded")

        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM foods")
        total = cursor.fetchone()[0]
        assert total == 3  # Only 400, 401, 402

        cursor.execute("SELECT DISTINCT data_type FROM foods")
        types = [r[0] for r in cursor.fetchall()]
        assert types == ["branded_food"]

        conn.close()
