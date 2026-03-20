/**
 * Container Tare Weight Service -- CRUD + usage tracking for kitchen containers.
 *
 * Uses opsqlite raw SQL (consistent with historyService, backupService pattern).
 * container_weights table defined in db/schema.ts (Drizzle) and created in db/client.ts.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Container = {
  id: number;
  name: string;
  weightGrams: number;
  timesUsed: number;
  lastUsedAt: string | null;
  createdAt: string;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map a raw DB row to Container type. */
function rowToContainer(row: Record<string, unknown>): Container {
  return {
    id: row.id as number,
    name: row.name as string,
    weightGrams: row.weight_grams as number,
    timesUsed: (row.times_used as number) ?? 0,
    lastUsedAt: (row.last_used_at as string) ?? null,
    createdAt: row.created_at as string,
  };
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/**
 * Add a new container with name and tare weight.
 * Returns the inserted container.
 */
export async function addContainer(
  name: string,
  weightGrams: number,
): Promise<Container> {
  const insertResult = await opsqlite.execute(
    'INSERT INTO container_weights (name, weight_grams) VALUES (?, ?)',
    [name, weightGrams],
  );

  const selectResult = await opsqlite.execute(
    'SELECT * FROM container_weights WHERE id = ?',
    [insertResult.insertId],
  );

  return rowToContainer(selectResult.rows._array[0]);
}

/**
 * Get all containers sorted by usage frequency (most used first).
 * Ties broken by last_used_at DESC, then created_at DESC.
 */
export async function getContainers(): Promise<Container[]> {
  const result = await opsqlite.execute(
    'SELECT * FROM container_weights ORDER BY times_used DESC, last_used_at DESC, created_at DESC',
  );

  return result.rows._array.map(rowToContainer);
}

/**
 * Update an existing container's name and/or weight.
 */
export async function updateContainer(
  id: number,
  update: { name?: string; weightGrams?: number },
): Promise<void> {
  const setClauses: string[] = [];
  const params: unknown[] = [];

  if (update.name !== undefined) {
    setClauses.push('name = ?');
    params.push(update.name);
  }
  if (update.weightGrams !== undefined) {
    setClauses.push('weight_grams = ?');
    params.push(update.weightGrams);
  }

  if (setClauses.length === 0) return;

  params.push(id);
  await opsqlite.execute(
    `UPDATE container_weights SET ${setClauses.join(', ')} WHERE id = ?`,
    params,
  );
}

/**
 * Delete a container by ID.
 */
export async function deleteContainer(id: number): Promise<void> {
  await opsqlite.execute('DELETE FROM container_weights WHERE id = ?', [id]);
}

// ---------------------------------------------------------------------------
// Usage tracking
// ---------------------------------------------------------------------------

/**
 * Record that a container was used: increment timesUsed, update lastUsedAt.
 */
export async function recordContainerUsage(id: number): Promise<void> {
  await opsqlite.execute(
    "UPDATE container_weights SET times_used = times_used + 1, last_used_at = datetime('now') WHERE id = ?",
    [id],
  );
}

// ---------------------------------------------------------------------------
// Tare math
// ---------------------------------------------------------------------------

/**
 * Apply tare subtraction: net weight = gross - container, minimum 0.
 * Pure function -- no DB access.
 */
export function applyTare(grossWeightG: number, container: Container): number {
  return Math.max(0, grossWeightG - container.weightGrams);
}
