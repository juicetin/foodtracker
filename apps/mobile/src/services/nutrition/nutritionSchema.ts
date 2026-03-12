/**
 * Published schema spec for nutrition packs.
 *
 * Any nutrition pack (first-party or community-created) MUST implement
 * this SQLite schema. The pack manager validates schema conformance
 * after download.
 *
 * Schema version 1 -- matches USDA FDC build pipeline output.
 */

// ── SQL schema definition ──

export const NUTRITION_SCHEMA_VERSION = 1;

/**
 * The canonical SQL schema that every nutrition pack database must contain.
 * Community pack contributors: your build script must produce a database
 * matching this schema exactly (table names, column names, types).
 */
export const NUTRITION_SCHEMA_SQL = `
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

CREATE VIRTUAL TABLE foods_fts USING fts5(
    description,
    food_category,
    content=foods,
    content_rowid=fdc_id
);
` as const;

// ── TypeScript types matching the schema ──

/** A food item from the nutrition database. */
export interface NutritionFood {
  fdcId: number;
  description: string;
  foodCategory: string | null;
  dataType: string | null;
  publicationDate: string | null;
}

/** A nutrient definition (e.g., Energy, Protein, Fat). */
export interface Nutrient {
  id: number;
  name: string;
  unit: string;
  nutrientNbr: string | null;
}

/** A nutrient value for a specific food (per 100g). */
export interface NutrientValue {
  nutrientId: number;
  name: string;
  amount: number;
  unit: string;
}

/** A serving size / portion option for a food. */
export interface FoodPortion {
  id: number;
  foodId: number;
  portionDescription: string | null;
  gramWeight: number | null;
  modifier: string | null;
}

/** Calculated macronutrient totals for a given weight. */
export interface MacroResult {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  /** The weight in grams these values were calculated for. */
  weightGrams: number;
}

// ── Well-known nutrient IDs (USDA FDC numbering) ──

export const NUTRIENT_IDS = {
  ENERGY: 1008,
  PROTEIN: 1003,
  CARBS: 1005,
  FAT: 1004,
  FIBER: 1079,
  SUGAR: 2000,
  SODIUM: 1093,
  CHOLESTEROL: 1253,
  SATURATED_FAT: 1258,
} as const;
