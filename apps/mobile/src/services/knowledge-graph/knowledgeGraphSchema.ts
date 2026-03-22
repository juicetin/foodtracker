/**
 * Published KG table/column name constants for raw SQL.
 *
 * Prevents typos in SQL strings across the KnowledgeGraphService.
 * Follows the same pattern as nutritionSchema.ts.
 */

// ── Table name constants ──

export const KG_TABLES = {
  DISH: 'dish',
  DISH_ALIAS: 'dish_alias',
  RECIPE: 'recipe',
  RECIPE_INGREDIENT: 'recipe_ingredient',
  USDA_FOOD: 'usda_food',
  USDA_EMBEDDINGS: 'usda_embeddings',
  USDA_BM25_TERMS: 'usda_bm25_terms',
  DISH_FTS: 'dish_fts',
  DISH_ALIAS_FTS: 'dish_alias_fts',
  SYMSPELL_DELETES: 'symspell_deletes',
  CUISINE: 'cuisine',
  DISH_CATEGORY: 'dish_category',
} as const;

// ── SQL query templates ──

/** Search dishes by name via FTS5 on dish_fts. */
export const SQL_SEARCH_DISH_FTS = `
  SELECT d.id, d.canonical_name, d.avg_calories_per_serving,
         d.avg_protein_per_serving, d.avg_carbs_per_serving,
         d.avg_fat_per_serving, d.default_serving_grams
  FROM ${KG_TABLES.DISH_FTS} fts
  JOIN ${KG_TABLES.DISH} d ON d.id = fts.rowid
  WHERE ${KG_TABLES.DISH_FTS} MATCH ?
  ORDER BY rank
  LIMIT 1
`;

/** Search dishes by alias via FTS5 on dish_alias_fts. */
export const SQL_SEARCH_ALIAS_FTS = `
  SELECT d.id, d.canonical_name, d.avg_calories_per_serving,
         d.avg_protein_per_serving, d.avg_carbs_per_serving,
         d.avg_fat_per_serving, d.default_serving_grams
  FROM ${KG_TABLES.DISH_ALIAS_FTS} fts
  JOIN ${KG_TABLES.DISH_ALIAS} da ON da.id = fts.rowid
  JOIN ${KG_TABLES.DISH} d ON d.id = da.dish_id
  WHERE ${KG_TABLES.DISH_ALIAS_FTS} MATCH ?
  ORDER BY rank
  LIMIT 1
`;

/** Get a dish by its ID. */
export const SQL_GET_DISH_BY_ID = `
  SELECT id, canonical_name, avg_calories_per_serving,
         avg_protein_per_serving, avg_carbs_per_serving,
         avg_fat_per_serving, default_serving_grams
  FROM ${KG_TABLES.DISH}
  WHERE id = ?
`;

/** Get the canonical recipe for a dish. */
export const SQL_GET_CANONICAL_RECIPE = `
  SELECT id, dish_id, name, source, total_weight_grams, servings, is_canonical
  FROM ${KG_TABLES.RECIPE}
  WHERE dish_id = ? AND is_canonical = 1
  LIMIT 1
`;

/** Get recipe ingredients with joined USDA nutrition data. */
export const SQL_GET_RECIPE_INGREDIENTS = `
  SELECT ri.id, ri.recipe_id, ri.usda_fdc_id, ri.ingredient_name, ri.quantity_grams,
         uf.calories_per_100g, uf.protein_per_100g, uf.fat_per_100g, uf.carbs_per_100g
  FROM ${KG_TABLES.RECIPE_INGREDIENT} ri
  LEFT JOIN ${KG_TABLES.USDA_FOOD} uf ON ri.usda_fdc_id = uf.fdc_id
  WHERE ri.recipe_id = ?
  ORDER BY ri.sort_order
`;

/** Load all symspell delete variants. */
export const SQL_LOAD_SYMSPELL_DELETES = `
  SELECT dish_id, delete_variant
  FROM ${KG_TABLES.SYMSPELL_DELETES}
`;

/** Load dish names for SymSpell term lookup. */
export const SQL_LOAD_DISH_NAMES = `
  SELECT id, canonical_name
  FROM ${KG_TABLES.DISH}
`;

/** Multi-result FTS dish search (for ingredient picker). */
export const SQL_SEARCH_DISHES_FTS = `
  SELECT d.id, d.canonical_name, d.avg_calories_per_serving,
         d.avg_protein_per_serving, d.avg_carbs_per_serving,
         d.avg_fat_per_serving, d.default_serving_grams
  FROM ${KG_TABLES.DISH_FTS} fts
  JOIN ${KG_TABLES.DISH} d ON d.id = fts.rowid
  WHERE ${KG_TABLES.DISH_FTS} MATCH ?
  ORDER BY rank
  LIMIT ?
`;

/** Search recipe ingredient names (for ingredient picker). */
export const SQL_SEARCH_INGREDIENT_NAMES = `
  SELECT DISTINCT ri.ingredient_name
  FROM ${KG_TABLES.RECIPE_INGREDIENT} ri
  WHERE ri.ingredient_name LIKE ?
  ORDER BY ri.ingredient_name
  LIMIT ?
`;

/**
 * Semantic vector search: brute-force cosine similarity over pre-computed
 * MiniLM-L6-v2 float32 embeddings stored in usda_embeddings.
 *
 * Requires sqlite-vec extension (enabled via op-sqlite "sqliteVec": true).
 * The ? parameter must be a 1536-byte blob of 384 float32 values (little-endian).
 *
 * Returns top 5 USDA entries ordered by cosine distance ASC (0 = identical).
 * Caller should filter results by MAX_VEC_DISTANCE to reject weak matches.
 */
export const MAX_VEC_DISTANCE = 0.50; // reject matches with cosine distance > this

export const SQL_SEARCH_USDA_VEC = `
  SELECT u.fdc_id, u.description,
         u.calories_per_100g, u.protein_per_100g, u.fat_per_100g, u.carbs_per_100g,
         u.fiber_per_100g, u.sodium_mg,
         vec_distance_cosine(e.vector, vec_f32(?)) AS distance
  FROM ${KG_TABLES.USDA_EMBEDDINGS} e
  JOIN ${KG_TABLES.USDA_FOOD} u ON u.fdc_id = e.fdc_id
  WHERE u.calories_per_100g IS NOT NULL
  ORDER BY distance ASC
  LIMIT 5
`;

/**
 * BM25 keyword search over pre-computed term weights in usda_bm25_terms.
 *
 * Pass query tokens as a SQL IN list. The caller builds the parameterised
 * query dynamically since SQLite doesn't support array binding directly.
 * Use buildBm25Query() in knowledgeGraphService.ts to generate this SQL.
 *
 * Returns top 5 USDA entries ordered by BM25 score DESC.
 */
export const SQL_SEARCH_USDA_BM25_TEMPLATE = (placeholders: string) => `
  SELECT u.fdc_id, u.description,
         u.calories_per_100g, u.protein_per_100g, u.fat_per_100g, u.carbs_per_100g,
         u.fiber_per_100g, u.sodium_mg,
         SUM(t.weight) AS bm25_score
  FROM ${KG_TABLES.USDA_BM25_TERMS} t
  JOIN ${KG_TABLES.USDA_FOOD} u ON u.fdc_id = t.fdc_id
  WHERE t.term IN (${placeholders})
    AND u.calories_per_100g IS NOT NULL
  GROUP BY u.fdc_id
  ORDER BY bm25_score DESC
  LIMIT 5
`;

/**
 * Search USDA foods by description prefix.
 *
 * Matches "broccoli" → "Broccoli, cooked, boiled, drained, with salt" etc.
 * Ordered by description length ASC so the simplest/most basic entry wins
 * (e.g., "Quinoa, cooked" before "Quinoa, cooked, with added salt").
 */
export const SQL_SEARCH_USDA_FOOD = `
  SELECT fdc_id, description,
         calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g,
         fiber_per_100g, sodium_mg
  FROM ${KG_TABLES.USDA_FOOD}
  WHERE description LIKE ?
  ORDER BY LENGTH(description) ASC
  LIMIT 3
`;
