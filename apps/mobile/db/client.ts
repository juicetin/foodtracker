import { open } from '@op-engineering/op-sqlite';
import { drizzle } from 'drizzle-orm/op-sqlite';
import * as schema from './schema';
import { TRACKED_TABLES } from '../src/services/backup/types';

// User data -- read-write, migrated via drizzle
export const opsqlite = open({ name: 'foodtracker.db' });
opsqlite.execute('PRAGMA journal_mode = WAL');
opsqlite.execute('PRAGMA foreign_keys = ON');

// Ensure core tables exist — full Drizzle migration isn't wired into App.tsx yet.
// CREATE TABLE IF NOT EXISTS is idempotent and safe to run on every app start.
opsqlite.execute(`CREATE TABLE IF NOT EXISTS food_entries (
  id TEXT PRIMARY KEY NOT NULL,
  meal_type TEXT NOT NULL,
  entry_date TEXT NOT NULL,
  total_calories REAL DEFAULT 0,
  total_protein REAL DEFAULT 0,
  total_carbs REAL DEFAULT 0,
  total_fat REAL DEFAULT 0,
  notes TEXT,
  updated_at TEXT DEFAULT (datetime('now')),
  is_synced INTEGER DEFAULT 0,
  is_deleted INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
)`);
opsqlite.execute(`CREATE TABLE IF NOT EXISTS installed_packs (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  version TEXT NOT NULL,
  file_path TEXT NOT NULL,
  size_bytes INTEGER,
  sha256 TEXT,
  region TEXT,
  installed_at TEXT DEFAULT (datetime('now')),
  last_checked TEXT
)`);
// Add mmproj_file_path column if missing (added in phase 02.6 for VLM paired files)
try { opsqlite.execute('ALTER TABLE installed_packs ADD COLUMN mmproj_file_path TEXT'); } catch {}

opsqlite.execute(`CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY NOT NULL,
  entry_id TEXT NOT NULL REFERENCES food_entries(id) ON DELETE CASCADE,
  uri TEXT NOT NULL,
  local_path TEXT,
  width INTEGER,
  height INTEGER,
  latitude REAL,
  longitude REAL,
  uploaded_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS ingredients (
  id TEXT PRIMARY KEY NOT NULL,
  entry_id TEXT NOT NULL REFERENCES food_entries(id) ON DELETE CASCADE,
  dish_id TEXT,
  name TEXT NOT NULL,
  quantity REAL NOT NULL DEFAULT 0,
  unit TEXT NOT NULL DEFAULT 'g',
  amount_g REAL,
  original_amount_g REAL,
  calories REAL NOT NULL DEFAULT 0,
  protein REAL DEFAULT 0,
  carbs REAL DEFAULT 0,
  fat REAL DEFAULT 0,
  fiber REAL DEFAULT 0,
  sodium REAL DEFAULT 0,
  ai_confidence REAL,
  database_source TEXT,
  user_modified INTEGER DEFAULT 0,
  updated_at TEXT DEFAULT (datetime('now')),
  created_at TEXT DEFAULT (datetime('now'))
)`);

// Add columns to ingredients if missing (idempotent)
try { opsqlite.execute('ALTER TABLE ingredients ADD COLUMN dish_id TEXT'); } catch {}
try { opsqlite.execute('ALTER TABLE ingredients ADD COLUMN amount_g REAL'); } catch {}
try { opsqlite.execute('ALTER TABLE ingredients ADD COLUMN original_amount_g REAL'); } catch {}
try { opsqlite.execute('ALTER TABLE ingredients ADD COLUMN sodium REAL DEFAULT 0'); } catch {}

opsqlite.execute(`CREATE TABLE IF NOT EXISTS favourite_meals (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  total_calories REAL DEFAULT 0,
  total_protein REAL DEFAULT 0,
  total_carbs REAL DEFAULT 0,
  total_fat REAL DEFAULT 0,
  times_used INTEGER DEFAULT 0,
  last_used_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS scanned_dishes (
  id TEXT PRIMARY KEY NOT NULL,
  entry_id TEXT NOT NULL REFERENCES food_entries(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  cuisine TEXT,
  portion_scale REAL NOT NULL DEFAULT 1.0,
  created_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS off_product_cache (
  barcode TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  brand TEXT,
  response_json TEXT NOT NULL,
  cached_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS off_search_cache (
  cache_key TEXT PRIMARY KEY NOT NULL,
  query TEXT NOT NULL,
  page_size INTEGER NOT NULL,
  response_json TEXT NOT NULL,
  result_count INTEGER NOT NULL,
  cached_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS correction_history (
  id TEXT PRIMARY KEY NOT NULL,
  original_class_name TEXT NOT NULL,
  corrected_class_name TEXT NOT NULL,
  confidence REAL NOT NULL,
  corrected_at TEXT DEFAULT (datetime('now'))
)`);

opsqlite.execute(`CREATE TABLE IF NOT EXISTS _change_journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_name TEXT NOT NULL,
  row_id INTEGER NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('INSERT','UPDATE','DELETE')),
  timestamp TEXT NOT NULL DEFAULT (datetime('now'))
)`);
opsqlite.execute('CREATE INDEX IF NOT EXISTS idx_change_journal_timestamp ON _change_journal(timestamp)');

opsqlite.execute(`CREATE TABLE IF NOT EXISTS _backup_metadata (
  id TEXT PRIMARY KEY NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('full','incremental')),
  filename TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  journal_from TEXT,
  journal_to TEXT,
  size_bytes INTEGER,
  parent_full_id TEXT
)`);

// Register updateHook to track changes on user-data tables
opsqlite.updateHook((params: { table: string; operation: 'INSERT' | 'DELETE' | 'UPDATE'; rowId: number }) => {
  if (!TRACKED_TABLES.has(params.table) || params.table === '_change_journal' || params.table === '_backup_metadata') return;
  opsqlite.executeSync(
    'INSERT INTO _change_journal (table_name, row_id, operation, timestamp) VALUES (?, ?, ?, ?)',
    [params.table, params.rowId, params.operation, new Date().toISOString()],
  );
});

export const userDb = drizzle(opsqlite, { schema });

// Nutrition data -- read-only, opened AFTER pack download completes
// Each nutrition DB (USDA, AFCD, etc.) gets its own connection
export function openNutritionDb(dbPath: string) {
  const nutritionOpsqlite = open({ name: dbPath });
  nutritionOpsqlite.execute('PRAGMA query_only = ON');
  return nutritionOpsqlite;
}
