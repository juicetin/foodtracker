import {
  sqliteTable,
  text,
  integer,
  real,
} from 'drizzle-orm/sqlite-core';
import { sql } from 'drizzle-orm';

// ── User Settings (flattened from users + nutrition_goals, single-user) ──

export const userSettings = sqliteTable('user_settings', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  region: text('region').default('AU'),
  units: text('units').default('metric'),
  calorieGoal: real('calorie_goal').default(2000),
  proteinGoal: real('protein_goal').default(150),
  carbsGoal: real('carbs_goal').default(200),
  fatGoal: real('fat_goal').default(65),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
});

// ── Food Entries (no user_id FK -- local-first single-user) ──

export const foodEntries = sqliteTable('food_entries', {
  id: text('id').primaryKey(),
  mealType: text('meal_type').notNull(),
  entryDate: text('entry_date').notNull(),
  totalCalories: real('total_calories').default(0),
  totalProtein: real('total_protein').default(0),
  totalCarbs: real('total_carbs').default(0),
  totalFat: real('total_fat').default(0),
  notes: text('notes'),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
  isSynced: integer('is_synced', { mode: 'boolean' }).default(false),
  isDeleted: integer('is_deleted', { mode: 'boolean' }).default(false),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Ingredients ──

export const ingredients = sqliteTable('ingredients', {
  id: text('id').primaryKey(),
  entryId: text('entry_id')
    .notNull()
    .references(() => foodEntries.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  quantity: real('quantity').notNull(),
  unit: text('unit').notNull(),
  calories: real('calories').notNull(),
  protein: real('protein').default(0),
  carbs: real('carbs').default(0),
  fat: real('fat').default(0),
  fiber: real('fiber').default(0),
  sugar: real('sugar').default(0),
  dishId: text('dish_id'),
  amountG: real('amount_g'),
  originalAmountG: real('original_amount_g'),
  aiConfidence: real('ai_confidence'),
  boundingBoxX: real('bounding_box_x'),
  boundingBoxY: real('bounding_box_y'),
  boundingBoxW: real('bounding_box_w'),
  boundingBoxH: real('bounding_box_h'),
  databaseSource: text('database_source'),
  databaseId: text('database_id'),
  userModified: integer('user_modified', { mode: 'boolean' }).default(false),
  originalQuantity: real('original_quantity'),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Photos (localPath replaces gcs_url) ──

export const photos = sqliteTable('photos', {
  id: text('id').primaryKey(),
  entryId: text('entry_id')
    .notNull()
    .references(() => foodEntries.id, { onDelete: 'cascade' }),
  uri: text('uri').notNull(),
  localPath: text('local_path'),
  width: integer('width'),
  height: integer('height'),
  latitude: real('latitude'),
  longitude: real('longitude'),
  uploadedAt: text('uploaded_at').default(sql`(datetime('now'))`),
});

// ── Scanned Dishes (Gemini Nano pipeline output, grouped per food_entry) ──

export const scannedDishes = sqliteTable('scanned_dishes', {
  id: text('id').primaryKey(),
  entryId: text('entry_id')
    .notNull()
    .references(() => foodEntries.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  cuisine: text('cuisine'),
  portionScale: real('portion_scale').notNull().default(1.0),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Modification History (no modified_by FK) ──

export const modificationHistory = sqliteTable('modification_history', {
  id: text('id').primaryKey(),
  ingredientId: text('ingredient_id')
    .notNull()
    .references(() => ingredients.id, { onDelete: 'cascade' }),
  fieldName: text('field_name').notNull(),
  oldValue: text('old_value'),
  newValue: text('new_value'),
  modifiedAt: text('modified_at').default(sql`(datetime('now'))`),
});

// ── Custom Recipes (no user_id FK) ──

export const customRecipes = sqliteTable('custom_recipes', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  description: text('description'),
  sourceEntryId: text('source_entry_id').references(() => foodEntries.id, {
    onDelete: 'set null',
  }),
  totalCalories: real('total_calories').default(0),
  totalProtein: real('total_protein').default(0),
  totalCarbs: real('total_carbs').default(0),
  totalFat: real('total_fat').default(0),
  timesUsed: integer('times_used').default(0),
  lastUsedAt: text('last_used_at'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
});

// ── Recipe Ingredients ──

export const recipeIngredients = sqliteTable('recipe_ingredients', {
  id: text('id').primaryKey(),
  recipeId: text('recipe_id')
    .notNull()
    .references(() => customRecipes.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  quantity: real('quantity').notNull(),
  unit: text('unit').notNull(),
  calories: real('calories').notNull(),
  protein: real('protein').default(0),
  carbs: real('carbs').default(0),
  fat: real('fat').default(0),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Recipe Photos (localPath replaces gcs_url) ──

export const recipePhotos = sqliteTable('recipe_photos', {
  id: text('id').primaryKey(),
  recipeId: text('recipe_id')
    .notNull()
    .references(() => customRecipes.id, { onDelete: 'cascade' }),
  localPath: text('local_path').notNull(),
  isPrimary: integer('is_primary', { mode: 'boolean' }).default(false),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Sync Outbox (new) ──

export const syncOutbox = sqliteTable('sync_outbox', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  tableName: text('table_name').notNull(),
  recordId: text('record_id').notNull(),
  operation: text('operation').notNull(), // 'insert' | 'update' | 'delete'
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Installed Packs (new) ──

export const installedPacks = sqliteTable('installed_packs', {
  id: text('id').primaryKey(), // e.g. 'usda-core', 'smolvlm-256m'
  name: text('name').notNull(),
  type: text('type').notNull(), // 'nutrition' | 'model' | 'knowledge-graph' | 'vlm'
  version: text('version').notNull(),
  filePath: text('file_path').notNull(),
  mmprojFilePath: text('mmproj_file_path'), // Local path to mmproj file (VLM packs only)
  sizeBytes: integer('size_bytes'),
  sha256: text('sha256'),
  region: text('region'),
  installedAt: text('installed_at').default(sql`(datetime('now'))`),
  lastChecked: text('last_checked'),
});

// ── Scan Queue (new — extended in Phase 04 with EXIF + classification columns) ──

export const scanQueue = sqliteTable('scan_queue', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  assetId: text('asset_id').unique(),
  uri: text('uri').notNull(),
  status: text('status').notNull().default('pending'), // 'pending' | 'processing' | 'done' | 'error'
  creationTime: integer('creation_time'), // EXIF epoch ms
  latitude: real('latitude'),
  longitude: real('longitude'),
  isFood: integer('is_food', { mode: 'boolean' }),
  mealGroupId: text('meal_group_id'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
  processedAt: text('processed_at'),
});

// ── Photo Hashes (new) ──

export const photoHashes = sqliteTable('photo_hashes', {
  photoId: text('photo_id')
    .primaryKey()
    .references(() => photos.id, { onDelete: 'cascade' }),
  phash: text('phash').notNull(),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Weight Entries (Phase 05 -- Health Connect + manual weight tracking) ──

export const weightEntries = sqliteTable('weight_entries', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  date: text('date').notNull().unique(), // YYYY-MM-DD
  weightKg: real('weight_kg').notNull(),
  source: text('source').notNull(), // 'manual' | 'health_connect'
  healthConnectId: text('health_connect_id'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── Container Weights (new) ──

export const containerWeights = sqliteTable('container_weights', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  weightGrams: real('weight_grams').notNull(),
  timesUsed: integer('times_used').default(0),
  lastUsedAt: text('last_used_at'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ── OFF Product Cache (Phase 03.5 -- Open Food Facts API response cache) ──

export const offProductCache = sqliteTable('off_product_cache', {
  barcode: text('barcode').primaryKey(),
  name: text('name').notNull(),
  brand: text('brand'),
  responseJson: text('response_json').notNull(), // Full OFFProduct JSON
  cachedAt: text('cached_at').default(sql`(datetime('now'))`),
});

// ── OFF Search Cache (Phase 03.5 -- Open Food Facts search response cache) ──

export const offSearchCache = sqliteTable('off_search_cache', {
  cacheKey: text('cache_key').primaryKey(), // normalized_query::pageSize
  query: text('query').notNull(),
  pageSize: integer('page_size').notNull(),
  responseJson: text('response_json').notNull(), // JSON array of OFFProduct
  resultCount: integer('result_count').notNull(),
  cachedAt: text('cached_at').default(sql`(datetime('now'))`),
});

// ── Model Cache (new) ──

export const modelCache = sqliteTable('model_cache', {
  modelId: text('model_id').primaryKey(),
  version: text('version').notNull(),
  filePath: text('file_path').notNull(),
  sizeBytes: integer('size_bytes'),
  downloadedAt: text('downloaded_at').default(sql`(datetime('now'))`),
});

// ── Correction History (new -- Phase 2, DET-05) ──

export const correctionHistory = sqliteTable('correction_history', {
  id: text('id').primaryKey(),
  originalClassName: text('original_class_name').notNull(),
  correctedClassName: text('corrected_class_name').notNull(),
  confidence: real('confidence').notNull(),
  correctedAt: text('corrected_at').default(sql`(datetime('now'))`),
});
