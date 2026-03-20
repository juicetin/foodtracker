/**
 * History service: queries past food_entries/scanned_dishes for frequency+recency ranking.
 * Also provides macro validation for QuickAdd screen.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HistoryItem {
  name: string;
  totalCount: number;
  lastLogged: string;
  avgCalories: number;
  avgProtein: number;
  avgCarbs: number;
  avgFat: number;
}

// ---------------------------------------------------------------------------
// History queries
// ---------------------------------------------------------------------------

/**
 * Get recent food history ranked by frequency (totalCount DESC) then recency
 * (lastLogged DESC). Sources: scanned_dishes.name for camera entries UNION
 * food_entries.notes for manual entries without scanned dishes.
 */
export function getRecentHistory(limit = 20): HistoryItem[] {
  const sql = `
    SELECT
      name,
      SUM(total_count) as total_count,
      MAX(last_logged) as last_logged,
      AVG(avg_calories) as avg_calories,
      AVG(avg_protein) as avg_protein,
      AVG(avg_carbs) as avg_carbs,
      AVG(avg_fat) as avg_fat
    FROM (
      -- Camera entries: group by scanned dish name
      SELECT
        sd.name as name,
        COUNT(*) as total_count,
        MAX(fe.created_at) as last_logged,
        AVG(fe.total_calories / MAX(1, (SELECT COUNT(*) FROM scanned_dishes sd2 WHERE sd2.entry_id = fe.id))) as avg_calories,
        AVG(fe.total_protein / MAX(1, (SELECT COUNT(*) FROM scanned_dishes sd2 WHERE sd2.entry_id = fe.id))) as avg_protein,
        AVG(fe.total_carbs / MAX(1, (SELECT COUNT(*) FROM scanned_dishes sd2 WHERE sd2.entry_id = fe.id))) as avg_carbs,
        AVG(fe.total_fat / MAX(1, (SELECT COUNT(*) FROM scanned_dishes sd2 WHERE sd2.entry_id = fe.id))) as avg_fat
      FROM scanned_dishes sd
      JOIN food_entries fe ON fe.id = sd.entry_id
      WHERE fe.is_deleted = 0
      GROUP BY sd.name

      UNION ALL

      -- Manual entries: use notes field as food name
      SELECT
        fe.notes as name,
        COUNT(*) as total_count,
        MAX(fe.created_at) as last_logged,
        AVG(fe.total_calories) as avg_calories,
        AVG(fe.total_protein) as avg_protein,
        AVG(fe.total_carbs) as avg_carbs,
        AVG(fe.total_fat) as avg_fat
      FROM food_entries fe
      WHERE fe.is_deleted = 0
        AND fe.notes IS NOT NULL
        AND fe.notes != ''
        AND fe.id NOT IN (SELECT entry_id FROM scanned_dishes)
      GROUP BY fe.notes
    )
    GROUP BY name
    ORDER BY total_count DESC, last_logged DESC
    LIMIT ?
  `;

  const result = opsqlite.execute(sql, [limit]);
  const rows = (result.rows ?? []) as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    name: cleanFoodName(row.name as string),
    totalCount: (row.total_count as number) ?? 0,
    lastLogged: (row.last_logged as string) ?? '',
    avgCalories: Math.round((row.avg_calories as number) ?? 0),
    avgProtein: Math.round((row.avg_protein as number) ?? 0),
    avgCarbs: Math.round((row.avg_carbs as number) ?? 0),
    avgFat: Math.round((row.avg_fat as number) ?? 0),
  }));
}

/**
 * Search history items by name (case-insensitive substring match).
 * Fetches a larger set then filters in JS for simplicity.
 */
export function searchHistory(query: string, limit = 10): HistoryItem[] {
  const all = getRecentHistory(50);
  const q = query.toLowerCase();
  return all
    .filter((item) => item.name.toLowerCase().includes(q))
    .slice(0, limit);
}

// ---------------------------------------------------------------------------
// Macro validation
// ---------------------------------------------------------------------------

/**
 * Validate whether entered calories match the macro breakdown.
 * Formula: expected = protein*4 + carbs*4 + fat*9
 * Tolerance: |cal - expected| <= max(expected * 0.1, 20)
 * All zeros is considered valid (empty form state).
 */
export function validateMacros(
  cal: number,
  protein: number,
  carbs: number,
  fat: number,
): { isValid: boolean; expected: number } {
  const expected = protein * 4 + carbs * 4 + fat * 9;

  if (cal === 0 && protein === 0 && carbs === 0 && fat === 0) {
    return { isValid: true, expected: 0 };
  }

  const tolerance = Math.max(expected * 0.1, 20);
  const isValid = Math.abs(cal - expected) <= tolerance;

  return { isValid, expected };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Strip known prefixes and parenthetical portions from food names. */
function cleanFoodName(name: string): string {
  let cleaned = name;

  // Strip "Copied: " prefix
  if (cleaned.startsWith('Copied: ')) {
    cleaned = cleaned.slice('Copied: '.length);
  }

  // Strip "Quick Add: " prefix
  if (cleaned.startsWith('Quick Add: ')) {
    cleaned = cleaned.slice('Quick Add: '.length);
  }

  // Strip parenthetical portions like "(150g)"
  cleaned = cleaned.replace(/\s*\([^)]*\)\s*$/, '').trim();

  return cleaned;
}
