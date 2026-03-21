#!/usr/bin/env python3
"""
Fix corrupted RecipeNLG recipe data in the knowledge graph database.

Root cause: seed_recipenlg.py calculated total_weight_grams from ALL unique
ingredient string variants aggregated across recipes (potentially 2,000+ per
dish), but only the top 30 are stored in recipe_ingredient. This caused a
75-80x inflation of total_weight_grams, cascading into absurd
default_serving_grams (400kg) and proportionally wrong avg_calories_per_serving.

Fix: For every RecipeNLG recipe, recalculate total_weight_grams as the sum of
the actual stored recipe_ingredient quantities, then recompute dish nutrition.

Usage:
    cd knowledge-graph
    python fix_recipenlg_data.py [--db food-knowledge.db] [--dry-run]
"""

import argparse
import sqlite3
from pathlib import Path

KG_DIR = Path(__file__).parent


def fix_recipenlg_total_weights(conn: sqlite3.Connection, dry_run: bool) -> dict:
    """
    Recalculate total_weight_grams for all RecipeNLG recipes from the
    actual stored recipe_ingredient rows.

    Returns stats dict with counts of recipes examined, fixed, and skipped.
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT r.id, r.total_weight_grams
        FROM recipe r
        WHERE r.source = 'recipenlg'
    """)
    recipes = cursor.fetchall()

    examined = 0
    fixed = 0
    already_ok = 0
    no_ingredients = 0

    for recipe_id, old_weight in recipes:
        examined += 1

        # Sum of actual stored ingredient weights
        cursor.execute("""
            SELECT COALESCE(SUM(quantity_grams), 0)
            FROM recipe_ingredient
            WHERE recipe_id = ?
        """, (recipe_id,))
        actual_weight = cursor.fetchone()[0]

        if actual_weight <= 0:
            no_ingredients += 1
            continue

        # Only update if the difference is > 1% (avoid floating point noise)
        if old_weight is not None and abs(old_weight - actual_weight) / max(old_weight, 1) < 0.01:
            already_ok += 1
            continue

        if not dry_run:
            cursor.execute(
                "UPDATE recipe SET total_weight_grams = ? WHERE id = ?",
                (round(actual_weight, 1), recipe_id),
            )
        fixed += 1

    if not dry_run:
        conn.commit()

    return {
        "examined": examined,
        "fixed": fixed,
        "already_ok": already_ok,
        "no_ingredients": no_ingredients,
    }


def recompute_dish_nutrition(conn: sqlite3.Connection, dry_run: bool) -> int:
    """
    Recompute avg_calories_per_serving, avg_protein_per_serving,
    avg_carbs_per_serving, avg_fat_per_serving, and default_serving_grams
    for all dishes that have a RecipeNLG canonical recipe.

    Uses the same logic as build_kg.py:compute_dish_nutrition() but scoped
    to RecipeNLG dishes to avoid clobbering curated data.
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT d.id, r.id as recipe_id, r.total_weight_grams, r.servings
        FROM dish d
        JOIN recipe r ON r.dish_id = d.id AND r.is_canonical = 1 AND r.source = 'recipenlg'
    """)
    dish_recipes = cursor.fetchall()

    updated = 0
    skipped_no_usda = 0

    for dish_id, recipe_id, total_weight, servings in dish_recipes:
        servings = servings or 1

        cursor.execute("""
            SELECT ri.quantity_grams,
                   u.calories_per_100g, u.protein_per_100g,
                   u.fat_per_100g, u.carbs_per_100g
            FROM recipe_ingredient ri
            JOIN usda_food u ON u.fdc_id = ri.usda_fdc_id
            WHERE ri.recipe_id = ?
        """, (recipe_id,))
        rows = cursor.fetchall()

        if not rows:
            skipped_no_usda += 1
            continue

        total_cal = 0.0
        total_pro = 0.0
        total_fat = 0.0
        total_carb = 0.0

        for qty_g, cal100, pro100, fat100, carb100 in rows:
            if qty_g and cal100 is not None:
                factor = qty_g / 100.0
                total_cal += (cal100 or 0) * factor
                total_pro += (pro100 or 0) * factor
                total_fat += (fat100 or 0) * factor
                total_carb += (carb100 or 0) * factor

        if total_cal <= 0:
            skipped_no_usda += 1
            continue

        # total_weight is now the sum of stored ingredient weights (fixed above)
        serving_weight = (total_weight or sum(r[0] or 0 for r in rows)) / servings

        if not dry_run:
            cursor.execute("""
                UPDATE dish SET
                    avg_calories_per_serving = ?,
                    avg_protein_per_serving  = ?,
                    avg_carbs_per_serving    = ?,
                    avg_fat_per_serving      = ?,
                    default_serving_grams    = ?
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

    if not dry_run:
        conn.commit()

    return updated


def sanity_check(conn: sqlite3.Connection):
    """Print before/after stats for a sample of known-bad dishes."""
    cursor = conn.cursor()

    sample_dishes = [
        "broccoli casserole",
        "quinoa salad",
        "quinoa pilaf",
        "broccoli cornbread",
        "mediterranean quinoa salad",
    ]

    print("\nSanity check — sample dishes:")
    print(f"  {'Dish':<35} {'kcal/serving':>12} {'serving_g':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10}")
    for name in sample_dishes:
        cursor.execute("""
            SELECT avg_calories_per_serving, default_serving_grams
            FROM dish WHERE canonical_name = ?
        """, (name,))
        row = cursor.fetchone()
        if row:
            cal, grams = row
            cal_str = f"{cal:.0f}" if cal else "NULL"
            g_str = f"{grams:.0f}" if grams else "NULL"
            print(f"  {name:<35} {cal_str:>12} {g_str:>10}")
        else:
            print(f"  {name:<35} {'(not found)':>12}")

    # Check for any remaining absurd values
    cursor.execute("""
        SELECT COUNT(*) FROM dish WHERE default_serving_grams > 5000
    """)
    absurd_count = cursor.fetchone()[0]
    print(f"\n  Dishes with serving > 5kg: {absurd_count}")

    cursor.execute("""
        SELECT COUNT(*) FROM dish WHERE avg_calories_per_serving > 5000
    """)
    high_cal = cursor.fetchone()[0]
    print(f"  Dishes with >5000 kcal/serving: {high_cal}")


def main():
    parser = argparse.ArgumentParser(description="Fix corrupted RecipeNLG data in KG database")
    parser.add_argument("--db", default="food-knowledge.db",
                        help="Path to the knowledge graph SQLite database")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without modifying the database")
    args = parser.parse_args()

    db_path = KG_DIR / args.db
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Fixing RecipeNLG data in: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Step 1: Show current state of known-bad dishes
    print("\nBefore fix:")
    sanity_check(conn)

    # Step 2: Fix total_weight_grams for all RecipeNLG recipes
    print("\nStep 1: Recalculating total_weight_grams from actual stored ingredients...")
    weight_stats = fix_recipenlg_total_weights(conn, args.dry_run)
    print(f"  Examined: {weight_stats['examined']:,} recipes")
    print(f"  Fixed:    {weight_stats['fixed']:,} (total_weight was inflated)")
    print(f"  OK:       {weight_stats['already_ok']:,} (already correct)")
    print(f"  Skipped:  {weight_stats['no_ingredients']:,} (no stored ingredients)")

    # Step 3: Recompute dish nutrition with corrected weights
    print("\nStep 2: Recomputing dish nutrition with corrected serving weights...")
    updated = recompute_dish_nutrition(conn, args.dry_run)
    print(f"  Updated: {updated:,} dishes")

    # Step 4: Show post-fix state
    if not args.dry_run:
        print("\nAfter fix:")
        sanity_check(conn)

    conn.close()
    print("\nDone." if not args.dry_run else "\n[DRY RUN] No changes written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
