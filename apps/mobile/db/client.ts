import { open } from '@op-engineering/op-sqlite';
import { drizzle } from 'drizzle-orm/op-sqlite';
import * as schema from './schema';

// User data -- read-write, migrated via drizzle
const opsqlite = open({ name: 'foodtracker.db' });
opsqlite.execute('PRAGMA journal_mode = WAL');
opsqlite.execute('PRAGMA foreign_keys = ON');
export const userDb = drizzle(opsqlite, { schema });

// Nutrition data -- read-only, opened AFTER pack download completes
// Each nutrition DB (USDA, AFCD, etc.) gets its own connection
export function openNutritionDb(dbPath: string) {
  const nutritionOpsqlite = open({ name: dbPath });
  nutritionOpsqlite.execute('PRAGMA query_only = ON');
  return nutritionOpsqlite;
}
