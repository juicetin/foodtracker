import { open } from '@op-engineering/op-sqlite';
import { drizzle } from 'drizzle-orm/op-sqlite';
import * as schema from './schema';

// User data -- read-write, migrated via drizzle
const opsqlite = open({ name: 'foodtracker.db' });
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
try {
  opsqlite.execute('ALTER TABLE installed_packs ADD COLUMN mmproj_file_path TEXT');
} catch {
  // Column already exists — expected on fresh installs
}

opsqlite.execute(`CREATE TABLE IF NOT EXISTS correction_history (
  id TEXT PRIMARY KEY NOT NULL,
  original_class_name TEXT NOT NULL,
  corrected_class_name TEXT NOT NULL,
  confidence REAL NOT NULL,
  corrected_at TEXT DEFAULT (datetime('now'))
)`);

export const userDb = drizzle(opsqlite, { schema });

// Nutrition data -- read-only, opened AFTER pack download completes
// Each nutrition DB (USDA, AFCD, etc.) gets its own connection
export function openNutritionDb(dbPath: string) {
  const nutritionOpsqlite = open({ name: dbPath });
  nutritionOpsqlite.execute('PRAGMA query_only = ON');
  return nutritionOpsqlite;
}
