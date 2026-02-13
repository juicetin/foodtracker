#!/usr/bin/env python3
"""
Export the food knowledge graph as an optimized SQLite file for mobile bundling.

Creates a compact, read-optimized copy of the knowledge graph suitable for
inclusion in a React Native / Expo mobile app bundle.

Usage:
    python export_mobile.py [--db food-knowledge.db] [--output food-knowledge-mobile.db]
"""

import argparse
import os
import shutil
import sqlite3
from pathlib import Path


def export_mobile(
    source_db: str,
    output_db: str = None,
    mobile_dest: str = None,
):
    """
    Export an optimized mobile-ready copy of the knowledge graph.

    Args:
        source_db: Path to the source knowledge graph database
        output_db: Path for the mobile export (default: food-knowledge-mobile.db)
        mobile_dest: Path to copy the DB for mobile app bundling
    """
    source_path = Path(source_db)
    if not source_path.exists():
        print(f"Error: Source database not found: {source_db}")
        return False

    if output_db is None:
        output_db = str(source_path.parent / "food-knowledge-mobile.db")

    output_path = Path(output_db)

    print("=" * 60)
    print("Exporting Mobile-Ready Knowledge Graph")
    print("=" * 60)
    print(f"  Source: {source_db}")
    print(f"  Output: {output_db}")

    # Step 1: Copy the database
    print("\n1. Copying database...")
    shutil.copy2(source_db, output_db)

    # Also copy any WAL/SHM files and consolidate
    for ext in ["-wal", "-shm"]:
        wal_file = source_db + ext
        if os.path.exists(wal_file):
            shutil.copy2(wal_file, output_db + ext)

    # Step 2: Optimize
    print("2. Optimizing for mobile...")
    conn = sqlite3.connect(output_db)

    # Checkpoint WAL to merge any pending writes
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        pass

    # Switch from WAL to DELETE mode for mobile (simpler, single file)
    conn.execute("PRAGMA journal_mode=DELETE")

    # Run ANALYZE to update query planner statistics
    print("   Running ANALYZE...")
    conn.execute("ANALYZE")

    # Run VACUUM to defragment and minimize file size
    print("   Running VACUUM...")
    conn.execute("VACUUM")

    # Verify the export
    print("\n3. Verifying export...")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM dishes")
    dish_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ingredients")
    ingredient_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM dish_ingredients")
    relationship_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT cuisine) FROM dishes")
    cuisine_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ingredients WHERE usda_fdc_id IS NOT NULL")
    fdc_count = cursor.fetchone()[0]

    conn.close()

    # Remove any leftover WAL/SHM files from the output
    for ext in ["-wal", "-shm"]:
        wal_path = output_db + ext
        if os.path.exists(wal_path):
            os.remove(wal_path)

    # Get file size
    file_size = output_path.stat().st_size
    size_mb = file_size / (1024 * 1024)

    print(f"\n  Dishes:           {dish_count:,}")
    print(f"  Ingredients:      {ingredient_count:,}")
    print(f"  Relationships:    {relationship_count:,}")
    print(f"  Cuisines:         {cuisine_count}")
    print(f"  USDA FDC links:   {fdc_count}")
    print(f"  File size:        {size_mb:.2f} MB ({file_size:,} bytes)")

    if size_mb > 50:
        print(f"\n  WARNING: File size ({size_mb:.1f} MB) exceeds 50MB mobile threshold!")
    else:
        print(f"\n  File size OK for mobile bundling (<50MB)")

    # Step 4: Copy to mobile app directory if specified
    if mobile_dest:
        mobile_path = Path(mobile_dest)
        mobile_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_db, mobile_dest)
        print(f"\n4. Copied to mobile app: {mobile_dest}")
    else:
        print(f"\n4. Skipped mobile app copy (no destination specified)")

    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export mobile-ready knowledge graph SQLite database"
    )
    parser.add_argument(
        "--db",
        default="knowledge-graph/food-knowledge.db",
        help="Path to source SQLite database",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for mobile export (default: food-knowledge-mobile.db)",
    )
    parser.add_argument(
        "--mobile-dest",
        default="apps/mobile/src/data/food-knowledge.db",
        help="Path to copy DB for mobile app bundling",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = str(project_root / db_path)

    output_path = args.output
    if output_path and not os.path.isabs(output_path):
        output_path = str(project_root / output_path)

    mobile_dest = args.mobile_dest
    if mobile_dest and not os.path.isabs(mobile_dest):
        mobile_dest = str(project_root / mobile_dest)

    export_mobile(db_path, output_path, mobile_dest)


if __name__ == "__main__":
    main()
