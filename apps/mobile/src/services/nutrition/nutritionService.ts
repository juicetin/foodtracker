/**
 * Query layer over nutrition SQLite databases.
 *
 * Uses raw SQL queries via the op-sqlite connection returned by openNutritionDb
 * from db/client.ts. This does NOT use drizzle-orm because the nutrition DB
 * has its own schema that is separate from the user database migrations.
 *
 * The nutrition database follows the published schema in nutritionSchema.ts.
 */

import { openNutritionDb } from '../../../db/client';
import type {
  NutritionFood,
  NutrientValue,
  FoodPortion,
  MacroResult,
} from './nutritionSchema';
import { NUTRIENT_IDS } from './nutritionSchema';

/** Type for the op-sqlite connection returned by openNutritionDb. */
type OPSQLiteConnection = ReturnType<typeof openNutritionDb>;

export class NutritionService {
  private db: OPSQLiteConnection | null = null;

  /**
   * Open a nutrition database for querying.
   * Uses the canonical openNutritionDb from db/client.ts.
   *
   * @param dbPath - Path to the nutrition SQLite database file
   */
  open(dbPath: string): void {
    this.db = openNutritionDb(dbPath);
  }

  /**
   * Ensure the database connection is open.
   */
  private getDb(): OPSQLiteConnection {
    if (!this.db) {
      throw new Error(
        'NutritionService: database not opened. Call open(dbPath) first.'
      );
    }
    return this.db;
  }

  /**
   * Search foods by name using FTS5 full-text search.
   *
   * Appends '*' to the query for prefix matching (e.g., "chick" matches "chicken").
   * Results are ranked by FTS5 relevance.
   *
   * @param query - Search text
   * @param limit - Maximum number of results (default 20)
   * @returns Matching foods ordered by relevance
   */
  async searchFoods(query: string, limit: number = 20): Promise<NutritionFood[]> {
    const db = this.getDb();

    // Append * for prefix matching
    const ftsQuery = `${query.trim()}*`;

    const result = await db.execute(
      `SELECT f.fdc_id, f.description, f.food_category, f.data_type, f.publication_date,
              rank
       FROM foods_fts fts
       JOIN foods f ON f.fdc_id = fts.rowid
       WHERE foods_fts MATCH ?
       ORDER BY rank
       LIMIT ?`,
      [ftsQuery, limit]
    );

    return result.rows.map(mapRowToFood);
  }

  /**
   * Get a food by its FDC ID.
   *
   * @param fdcId - USDA FoodData Central ID
   * @returns The food, or null if not found
   */
  async getFoodById(fdcId: number): Promise<NutritionFood | null> {
    const db = this.getDb();

    const result = await db.execute(
      `SELECT fdc_id, description, food_category, data_type, publication_date
       FROM foods
       WHERE fdc_id = ?`,
      [fdcId]
    );

    if (result.rows.length === 0) return null;
    return mapRowToFood(result.rows[0]);
  }

  /**
   * Get all nutrient values for a food (per 100g).
   *
   * @param fdcId - USDA FoodData Central ID
   * @returns Array of nutrient values
   */
  async getFoodNutrients(fdcId: number): Promise<NutrientValue[]> {
    const db = this.getDb();

    const result = await db.execute(
      `SELECT fn.nutrient_id, n.name, fn.amount, n.unit
       FROM food_nutrients fn
       JOIN nutrients n ON fn.nutrient_id = n.id
       WHERE fn.food_id = ?
       ORDER BY n.name`,
      [fdcId]
    );

    return result.rows.map((row: Record<string, unknown>) => ({
      nutrientId: row.nutrient_id as number,
      name: row.name as string,
      amount: row.amount as number,
      unit: row.unit as string,
    }));
  }

  /**
   * Get serving size / portion options for a food.
   *
   * @param fdcId - USDA FoodData Central ID
   * @returns Array of available portions
   */
  async getFoodPortions(fdcId: number): Promise<FoodPortion[]> {
    const db = this.getDb();

    const result = await db.execute(
      `SELECT id, food_id, portion_description, gram_weight, modifier
       FROM food_portions
       WHERE food_id = ?`,
      [fdcId]
    );

    return result.rows.map((row: Record<string, unknown>) => ({
      id: row.id as number,
      foodId: row.food_id as number,
      portionDescription: (row.portion_description as string) ?? null,
      gramWeight: (row.gram_weight as number) ?? null,
      modifier: (row.modifier as string) ?? null,
    }));
  }

  /**
   * Calculate macronutrient totals for a given weight.
   *
   * Retrieves per-100g nutrient values and scales them to the requested weight.
   *
   * @param fdcId - USDA FoodData Central ID
   * @param weightGrams - Weight in grams to calculate for
   * @returns Calculated macro values
   */
  async getMacros(fdcId: number, weightGrams: number): Promise<MacroResult> {
    const nutrients = await this.getFoodNutrients(fdcId);
    const scale = weightGrams / 100;

    const findAmount = (nutrientId: number): number => {
      const n = nutrients.find((v) => v.nutrientId === nutrientId);
      return n ? n.amount * scale : 0;
    };

    return {
      calories: findAmount(NUTRIENT_IDS.ENERGY),
      protein: findAmount(NUTRIENT_IDS.PROTEIN),
      carbs: findAmount(NUTRIENT_IDS.CARBS),
      fat: findAmount(NUTRIENT_IDS.FAT),
      weightGrams,
    };
  }

  /**
   * Close the nutrition database connection.
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

/**
 * Map a raw SQL row to a NutritionFood object.
 */
function mapRowToFood(row: Record<string, unknown>): NutritionFood {
  return {
    fdcId: row.fdc_id as number,
    description: row.description as string,
    foodCategory: (row.food_category as string) ?? null,
    dataType: (row.data_type as string) ?? null,
    publicationDate: (row.publication_date as string) ?? null,
  };
}
