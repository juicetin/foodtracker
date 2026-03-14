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
