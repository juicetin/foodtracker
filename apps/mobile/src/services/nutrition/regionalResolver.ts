/**
 * Multi-database query resolver with regional priority ordering.
 *
 * Manages multiple NutritionService instances (USDA, AFCD, CoFID, CIQUAL, custom)
 * and routes queries with regional databases taking priority over USDA.
 *
 * Priority ordering:
 * 1. Regional packs (AFCD, CoFID, CIQUAL) in order of installation
 * 2. USDA core (Foundation + SR Legacy + FNDDS)
 * 3. USDA branded (if installed)
 * 4. Custom packs
 *
 * Also provides schema validation for custom pack imports.
 */

import { openNutritionDb } from '../../../db/client';
import { PackManager } from '../packs/packManager';
import { NutritionService } from './nutritionService';
import type {
  NutritionFood,
  NutrientValue,
  MacroResult,
} from './nutritionSchema';
import type { InstalledPack } from '../packs/types';

/** A food result enriched with the source database identifier. */
export interface ResolvedFood extends NutritionFood {
  /** Pack ID indicating which database this result came from. */
  source: string;
}

/** Summary of an installed database for Settings display. */
export interface InstalledDatabase {
  id: string;
  name: string;
  region: string | null;
}

/** Well-known regional pack IDs. */
const REGIONAL_PACK_IDS = new Set(['afcd', 'cofid', 'ciqual']);

/** Well-known USDA pack IDs. */
const USDA_PACK_IDS = new Set(['usda-core', 'usda-branded']);

/**
 * Determine priority for a pack. Lower number = higher priority.
 * Regional packs get priority 1, USDA core = 2, USDA branded = 3, custom = 4.
 */
function getPriority(packId: string): number {
  if (REGIONAL_PACK_IDS.has(packId)) return 1;
  if (packId === 'usda-core') return 2;
  if (packId === 'usda-branded') return 3;
  return 4; // Custom packs
}

export class RegionalResolver {
  private databases: Map<string, NutritionService> = new Map();
  private priority: string[] = [];
  private packMetadata: Map<string, InstalledPack> = new Map();

  /**
   * Initialize the resolver by querying installed nutrition packs
   * and opening a NutritionService for each.
   */
  async initialize(): Promise<void> {
    const installedPacks = await PackManager.getInstalledPacks();
    const nutritionPacks = installedPacks.filter((p) => p.type === 'nutrition');

    for (const pack of nutritionPacks) {
      const filePath = await PackManager.getPackFilePath(pack.id);
      if (filePath) {
        const service = new NutritionService();
        service.open(filePath);
        this.databases.set(pack.id, service);
        this.packMetadata.set(pack.id, pack);
      }
    }

    this._rebuildPriority();
  }

  /**
   * Search foods across all databases in priority order.
   *
   * Regional databases are queried first. Results include a `source` field
   * indicating which database the result came from.
   *
   * @param query - Search text
   * @param limit - Maximum total results (default 20)
   * @returns Resolved foods with source database indicator
   */
  async searchFoods(query: string, limit: number = 20): Promise<ResolvedFood[]> {
    const allResults: ResolvedFood[] = [];

    for (const packId of this.priority) {
      if (allResults.length >= limit) break;

      const service = this.databases.get(packId);
      if (!service) continue;

      const remaining = limit - allResults.length;
      const results = await service.searchFoods(query, remaining);

      for (const food of results) {
        allResults.push({
          ...food,
          source: packId,
        });
      }
    }

    return allResults.slice(0, limit);
  }

  /**
   * Get nutrient values for a food, routing to the correct database by source.
   *
   * @param fdcId - Food ID within the source database
   * @param source - Pack ID (e.g., 'afcd', 'usda-core')
   * @returns Nutrient values for the food
   */
  async getFoodNutrients(fdcId: number, source: string): Promise<NutrientValue[]> {
    const service = this.databases.get(source);
    if (!service) {
      throw new Error(`RegionalResolver: no database found for source "${source}"`);
    }
    return service.getFoodNutrients(fdcId);
  }

  /**
   * Calculate macronutrient totals, routing to the correct database by source.
   *
   * @param fdcId - Food ID within the source database
   * @param source - Pack ID (e.g., 'afcd', 'usda-core')
   * @param weightGrams - Weight in grams to calculate for
   * @returns Calculated macro values
   */
  async getMacros(fdcId: number, source: string, weightGrams: number): Promise<MacroResult> {
    const service = this.databases.get(source);
    if (!service) {
      throw new Error(`RegionalResolver: no database found for source "${source}"`);
    }
    return service.getMacros(fdcId, weightGrams);
  }

  /**
   * Get summary of all installed nutrition databases.
   *
   * Used by Settings > Data & Storage display (Phase 3 builds the UI).
   *
   * @returns Array of installed database summaries
   */
  getInstalledDatabases(): InstalledDatabase[] {
    return this.priority.map((packId) => {
      const meta = this.packMetadata.get(packId);
      return {
        id: packId,
        name: meta?.name ?? packId,
        region: meta?.region ?? null,
      };
    });
  }

  /**
   * Add a new database to the resolver.
   *
   * Opens a NutritionService for the given database file and adds it
   * to the priority list.
   *
   * @param packId - Unique pack identifier
   * @param dbPath - Path to the SQLite database file
   */
  async addDatabase(packId: string, dbPath: string): Promise<void> {
    const service = new NutritionService();
    service.open(dbPath);
    this.databases.set(packId, service);
    this.packMetadata.set(packId, {
      id: packId,
      name: packId,
      type: 'nutrition',
      version: '1.0.0',
      filePath: dbPath,
      sizeBytes: null,
      sha256: null,
      region: null,
      installedAt: new Date().toISOString(),
      lastChecked: null,
    });
    this._rebuildPriority();
  }

  /**
   * Remove a database from the resolver.
   *
   * Closes the NutritionService connection and removes from maps.
   *
   * @param packId - Pack identifier to remove
   */
  removeDatabase(packId: string): void {
    const service = this.databases.get(packId);
    if (service) {
      service.close();
    }
    this.databases.delete(packId);
    this.packMetadata.delete(packId);
    this._rebuildPriority();
  }

  /**
   * Rebuild priority list from current databases.
   * Sorts by: regional (1) > usda-core (2) > usda-branded (3) > custom (4).
   */
  private _rebuildPriority(): void {
    const ids = Array.from(this.databases.keys());
    ids.sort((a, b) => getPriority(a) - getPriority(b));
    this.priority = ids;
  }
}

/**
 * Validate that a SQLite database conforms to the published nutrition pack schema.
 *
 * Checks for required tables (foods, nutrients, food_nutrients, food_portions, foods_fts)
 * and required columns on each table.
 *
 * @param dbPath - Path to the SQLite database file to validate
 * @returns Validation result with specific error messages
 */
export async function validatePackSchema(
  dbPath: string
): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];

  const db = openNutritionDb(dbPath);

  try {
    // Check required tables
    const tableResult = await db.execute(
      "SELECT name FROM sqlite_master WHERE type='table' OR type='table'"
    );
    const tableNames = new Set(
      tableResult.rows.map((r: Record<string, unknown>) => r.name as string)
    );

    const requiredTables = ['foods', 'nutrients', 'food_nutrients', 'food_portions', 'foods_fts'];
    for (const table of requiredTables) {
      if (!tableNames.has(table)) {
        errors.push(`Missing required table: ${table}`);
      }
    }

    // If essential tables are missing, skip column checks
    if (errors.length > 0) {
      db.close();
      return { valid: false, errors };
    }

    // Check required columns on foods table
    const foodsColumns = await db.execute("PRAGMA table_info(foods)");
    const foodsColNames = new Set(
      foodsColumns.rows.map((r: Record<string, unknown>) => r.name as string)
    );
    for (const col of ['fdc_id', 'description']) {
      if (!foodsColNames.has(col)) {
        errors.push(`Missing column in foods table: ${col}`);
      }
    }

    // Check required columns on nutrients table
    const nutrientsColumns = await db.execute("PRAGMA table_info(nutrients)");
    const nutrientsColNames = new Set(
      nutrientsColumns.rows.map((r: Record<string, unknown>) => r.name as string)
    );
    for (const col of ['id', 'name', 'unit']) {
      if (!nutrientsColNames.has(col)) {
        errors.push(`Missing column in nutrients table: ${col}`);
      }
    }

    // Check required columns on food_nutrients table
    const fnColumns = await db.execute("PRAGMA table_info(food_nutrients)");
    const fnColNames = new Set(
      fnColumns.rows.map((r: Record<string, unknown>) => r.name as string)
    );
    for (const col of ['food_id', 'nutrient_id', 'amount']) {
      if (!fnColNames.has(col)) {
        errors.push(`Missing column in food_nutrients table: ${col}`);
      }
    }
  } finally {
    db.close();
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
