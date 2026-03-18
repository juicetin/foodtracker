/**
 * Core KG query service: searchDish, getCanonicalRecipe, calculateDishNutrition, getDishAverages.
 *
 * Uses raw SQL queries via the op-sqlite connection returned by openNutritionDb
 * from db/client.ts. Follows the same patterns as NutritionService.
 *
 * Query chain for calculateDishNutrition:
 *   1. searchDish(className) -- FTS5 + alias + SymSpell fuzzy
 *   2. getCanonicalRecipe(dish.id) -- recipe WHERE is_canonical=1
 *   3. Sum per-ingredient nutrition (quantity_grams/100 * USDA per-100g values)
 *   4. Scale to requested portion weight
 *
 * Fallback chain: recipe decomposition -> dish averages -> null (caller uses flat-rate proxy)
 */

import { openNutritionDb } from '../../../db/client';
import {
  SQL_SEARCH_DISH_FTS,
  SQL_SEARCH_ALIAS_FTS,
  SQL_GET_DISH_BY_ID,
  SQL_GET_CANONICAL_RECIPE,
  SQL_GET_RECIPE_INGREDIENTS,
} from './knowledgeGraphSchema';
import { SymSpellIndex } from './symspellIndex';

/** Type for the op-sqlite connection returned by openNutritionDb. */
type OPSQLiteConnection = ReturnType<typeof openNutritionDb>;

// ── Result types ──

/** Calculated macronutrient totals for a given portion. */
export interface MacroResult {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  /** The weight in grams these values were calculated for. */
  weightGrams: number;
  /** How the values were derived. */
  source: 'recipe' | 'dish_average' | 'proxy';
}

/** A dish found in the knowledge graph. */
export interface DishResult {
  id: number;
  canonicalName: string;
  avgCaloriesPerServing: number | null;
  avgProteinPerServing: number | null;
  avgCarbsPerServing: number | null;
  avgFatPerServing: number | null;
  defaultServingGrams: number | null;
}

/** A canonical recipe for a dish. */
export interface RecipeResult {
  id: number;
  dishId: number;
  name: string;
  source: string | null;
  totalWeightGrams: number;
  servings: number;
  isCanonical: boolean;
}

/** A recipe ingredient with USDA nutrition data. */
interface IngredientResult {
  id: number;
  recipeId: number;
  usdaFdcId: number | null;
  ingredientName: string;
  quantityGrams: number;
  caloriesPer100g: number;
  proteinPer100g: number;
  fatPer100g: number;
  carbsPer100g: number;
}

/**
 * Knowledge Graph query service.
 *
 * Provides dish search (FTS5 + fuzzy), recipe decomposition, and
 * nutrition calculation from USDA ingredient data.
 */
export class KnowledgeGraphService {
  private db: OPSQLiteConnection | null = null;
  private symspell: SymSpellIndex;

  constructor() {
    this.symspell = new SymSpellIndex();
  }

  /**
   * Open a KG database for querying.
   * Uses the canonical openNutritionDb from db/client.ts (read-only).
   * Initializes the SymSpell index from the database.
   */
  async open(dbPath: string): Promise<void> {
    this.db = openNutritionDb(dbPath);
    await this.symspell.loadFromDb(this.db);
  }

  /**
   * Close the KG database connection.
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  /**
   * Ensure the database connection is open.
   */
  private getDb(): OPSQLiteConnection {
    if (!this.db) {
      throw new Error(
        'KnowledgeGraphService: database not opened. Call open(dbPath) first.'
      );
    }
    return this.db;
  }

  /**
   * Search for a dish by name.
   *
   * Search chain:
   * 1. FTS5 on dish_fts (prefix match)
   * 2. FTS5 on dish_alias_fts (prefix match)
   * 3. SymSpell fuzzy match (edit distance <= 2)
   * 4. Returns null if all fail
   *
   * @param query - Dish name to search for (normalized automatically)
   * @returns The best matching dish, or null
   */
  async searchDish(query: string): Promise<DishResult | null> {
    const db = this.getDb();
    const normalized = this.normalize(query);
    if (normalized.length === 0) return null;

    const ftsQuery = `${normalized}*`;

    // 1. Try FTS5 on dish_fts
    const dishFtsResult = await db.execute(SQL_SEARCH_DISH_FTS, [ftsQuery]);
    if (dishFtsResult.rows.length > 0) {
      return this.mapRowToDish(dishFtsResult.rows[0] as Record<string, unknown>);
    }

    // 2. Try FTS5 on dish_alias_fts
    const aliasFtsResult = await db.execute(SQL_SEARCH_ALIAS_FTS, [ftsQuery]);
    if (aliasFtsResult.rows.length > 0) {
      return this.mapRowToDish(aliasFtsResult.rows[0] as Record<string, unknown>);
    }

    // 3. Try SymSpell fuzzy match
    const fuzzyMatches = this.symspell.lookup(normalized, 1);
    if (fuzzyMatches.length > 0) {
      const bestMatch = fuzzyMatches[0];
      const dishResult = await db.execute(SQL_GET_DISH_BY_ID, [
        bestMatch.dishId,
      ]);
      if (dishResult.rows.length > 0) {
        return this.mapRowToDish(dishResult.rows[0] as Record<string, unknown>);
      }
    }

    // 4. Not found
    return null;
  }

  /**
   * Get the canonical recipe for a dish.
   *
   * @param dishId - The dish ID to look up
   * @returns The canonical recipe, or null if none exists
   */
  async getCanonicalRecipe(dishId: number): Promise<RecipeResult | null> {
    const db = this.getDb();

    const result = await db.execute(SQL_GET_CANONICAL_RECIPE, [dishId]);
    if (result.rows.length === 0) return null;

    const row = result.rows[0] as Record<string, unknown>;
    return {
      id: row.id as number,
      dishId: row.dish_id as number,
      name: row.name as string,
      source: (row.source as string) ?? null,
      totalWeightGrams: row.total_weight_grams as number,
      servings: row.servings as number,
      isCanonical: (row.is_canonical as number) === 1,
    };
  }

  /**
   * Get recipe ingredients with joined USDA nutrition data.
   *
   * @param recipeId - The recipe ID to look up
   * @returns Array of ingredients with per-100g nutrition values
   */
  private async getRecipeIngredients(
    recipeId: number
  ): Promise<IngredientResult[]> {
    const db = this.getDb();

    const result = await db.execute(SQL_GET_RECIPE_INGREDIENTS, [recipeId]);

    return (result.rows as Array<Record<string, unknown>>).map((row) => ({
      id: row.id as number,
      recipeId: row.recipe_id as number,
      usdaFdcId: (row.usda_fdc_id as number) ?? null,
      ingredientName: row.ingredient_name as string,
      quantityGrams: row.quantity_grams as number,
      caloriesPer100g: (row.calories_per_100g as number) ?? 0,
      proteinPer100g: (row.protein_per_100g as number) ?? 0,
      fatPer100g: (row.fat_per_100g as number) ?? 0,
      carbsPer100g: (row.carbs_per_100g as number) ?? 0,
    }));
  }

  /**
   * Calculate nutrition for a dish by class name and portion weight.
   *
   * Fallback chain:
   * 1. Recipe decomposition (source='recipe') -- sum ingredient nutrition, scale to portion
   * 2. Dish averages (source='dish_average') -- use per-serving averages, scale to portion
   * 3. Return null -- caller applies flat-rate proxy
   *
   * @param className - The detected food class name (e.g., "pad_thai")
   * @param portionGrams - The estimated portion weight in grams
   * @returns Calculated macros, or null if dish not found
   */
  async calculateDishNutrition(
    className: string,
    portionGrams: number
  ): Promise<MacroResult | null> {
    // 1. Find the dish
    const dish = await this.searchDish(className);
    if (!dish) return null;

    // 2. Try recipe decomposition
    const recipe = await this.getCanonicalRecipe(dish.id);
    if (recipe) {
      const ingredients = await this.getRecipeIngredients(recipe.id);
      if (ingredients.length > 0) {
        return this.calculateFromRecipe(
          ingredients,
          recipe.totalWeightGrams,
          portionGrams
        );
      }
    }

    // 3. Fall back to dish averages
    return this.scaleDishAverages(dish, portionGrams);
  }

  /**
   * Get dish-level average macros (for tier 2 fallback).
   *
   * @param className - The detected food class name
   * @returns The dish with average macros, or null if not found
   */
  async getDishAverages(className: string): Promise<DishResult | null> {
    return this.searchDish(className);
  }

  /**
   * Search for ingredient names across dishes and recipe ingredients.
   * Returns deduplicated names, best matches first.
   * Used by the ingredient picker UI.
   */
  async searchIngredients(query: string, limit: number = 15): Promise<string[]> {
    const db = this.getDb();
    const normalized = this.normalize(query);
    if (normalized.length === 0) return [];

    const results: string[] = [];
    const seen = new Set<string>();

    // 1. FTS on dishes (dish names that match)
    try {
      const ftsQuery = `${normalized}*`;
      const dishRows = await db.execute(
        `SELECT d.canonical_name FROM dish_fts fts JOIN dish d ON d.id = fts.rowid WHERE dish_fts MATCH ? ORDER BY rank LIMIT ?`,
        [ftsQuery, limit],
      );
      for (const row of dishRows.rows as Array<Record<string, unknown>>) {
        const name = row.canonical_name as string;
        const lower = name.toLowerCase();
        if (!seen.has(lower)) {
          seen.add(lower);
          results.push(name);
        }
      }
    } catch {
      // FTS may not be available
    }

    // 2. LIKE search on recipe_ingredient names
    try {
      const likeQuery = `%${normalized}%`;
      const ingRows = await db.execute(
        `SELECT DISTINCT ri.ingredient_name FROM recipe_ingredient ri WHERE ri.ingredient_name LIKE ? ORDER BY ri.ingredient_name LIMIT ?`,
        [likeQuery, limit],
      );
      for (const row of ingRows.rows as Array<Record<string, unknown>>) {
        const name = row.ingredient_name as string;
        const lower = name.toLowerCase();
        if (!seen.has(lower)) {
          seen.add(lower);
          results.push(name);
        }
      }
    } catch {
      // Table may not exist
    }

    return results.slice(0, limit);
  }

  // ── Private helpers ──

  /**
   * Calculate macros from recipe ingredient decomposition.
   */
  private calculateFromRecipe(
    ingredients: IngredientResult[],
    recipeTotalGrams: number,
    portionGrams: number
  ): MacroResult {
    // Sum nutrition for the full recipe
    let totalCalories = 0;
    let totalProtein = 0;
    let totalCarbs = 0;
    let totalFat = 0;

    for (const ing of ingredients) {
      const scale = ing.quantityGrams / 100;
      totalCalories += ing.caloriesPer100g * scale;
      totalProtein += ing.proteinPer100g * scale;
      totalCarbs += ing.carbsPer100g * scale;
      totalFat += ing.fatPer100g * scale;
    }

    // Scale from full recipe to requested portion
    const portionScale = portionGrams / recipeTotalGrams;

    return {
      calories: totalCalories * portionScale,
      protein: totalProtein * portionScale,
      carbs: totalCarbs * portionScale,
      fat: totalFat * portionScale,
      weightGrams: portionGrams,
      source: 'recipe',
    };
  }

  /**
   * Scale dish average macros to a requested portion.
   */
  private scaleDishAverages(
    dish: DishResult,
    portionGrams: number
  ): MacroResult | null {
    const servingGrams = dish.defaultServingGrams;
    if (
      servingGrams == null ||
      dish.avgCaloriesPerServing == null
    ) {
      return null;
    }

    const scale = portionGrams / servingGrams;

    return {
      calories: (dish.avgCaloriesPerServing ?? 0) * scale,
      protein: (dish.avgProteinPerServing ?? 0) * scale,
      carbs: (dish.avgCarbsPerServing ?? 0) * scale,
      fat: (dish.avgFatPerServing ?? 0) * scale,
      weightGrams: portionGrams,
      source: 'dish_average',
    };
  }

  /**
   * Normalize input: lowercase, replace hyphens/underscores with spaces, trim.
   */
  private normalize(input: string): string {
    return input
      .toLowerCase()
      .replace(/[-_]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Map a raw SQL row to a DishResult object.
   */
  private mapRowToDish(row: Record<string, unknown>): DishResult {
    return {
      id: row.id as number,
      canonicalName: row.canonical_name as string,
      avgCaloriesPerServing: (row.avg_calories_per_serving as number) ?? null,
      avgProteinPerServing: (row.avg_protein_per_serving as number) ?? null,
      avgCarbsPerServing: (row.avg_carbs_per_serving as number) ?? null,
      avgFatPerServing: (row.avg_fat_per_serving as number) ?? null,
      defaultServingGrams: (row.default_serving_grams as number) ?? null,
    };
  }
}
