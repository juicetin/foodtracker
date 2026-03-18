/**
 * Favourite meals service — save, load, and re-log favourite meals.
 *
 * Favourites are stored in the favourite_meals SQLite table.
 * They represent a named meal with macro totals that can be re-logged with one tap.
 */

import { opsqlite } from '../../db/client';

export interface FavouriteMeal {
  id: string;
  name: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  timesUsed: number;
  lastUsedAt: string | null;
  createdAt: string;
}

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export function loadFavourites(limit: number = 20): FavouriteMeal[] {
  try {
    const rows = opsqlite.execute(
      'SELECT * FROM favourite_meals ORDER BY times_used DESC, last_used_at DESC LIMIT ?',
      [limit],
    ).rows as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: r.id as string,
      name: r.name as string,
      totalCalories: (r.total_calories as number) ?? 0,
      totalProtein: (r.total_protein as number) ?? 0,
      totalCarbs: (r.total_carbs as number) ?? 0,
      totalFat: (r.total_fat as number) ?? 0,
      timesUsed: (r.times_used as number) ?? 0,
      lastUsedAt: (r.last_used_at as string) ?? null,
      createdAt: r.created_at as string,
    }));
  } catch {
    return [];
  }
}

export function addFavourite(meal: {
  name: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
}): string {
  const id = generateId();
  opsqlite.execute(
    `INSERT INTO favourite_meals (id, name, total_calories, total_protein, total_carbs, total_fat, created_at)
     VALUES (?, ?, ?, ?, ?, ?, datetime('now'))`,
    [id, meal.name, meal.totalCalories, meal.totalProtein, meal.totalCarbs, meal.totalFat],
  );
  return id;
}

export function removeFavourite(id: string): void {
  opsqlite.execute('DELETE FROM favourite_meals WHERE id = ?', [id]);
}

export function incrementFavouriteUsage(id: string): void {
  opsqlite.execute(
    `UPDATE favourite_meals SET times_used = times_used + 1, last_used_at = datetime('now') WHERE id = ?`,
    [id],
  );
}

export function isFavourited(name: string): boolean {
  try {
    const rows = opsqlite.execute(
      'SELECT id FROM favourite_meals WHERE name = ? LIMIT 1',
      [name],
    ).rows;
    return rows.length > 0;
  } catch {
    return false;
  }
}
