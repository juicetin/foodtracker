/**
 * Database migrations module.
 * Re-exports migration utilities for App.tsx integration.
 * Wiring into the app root will happen in a later phase task.
 */
export { useMigrations } from 'drizzle-orm/op-sqlite/migrator';
