/**
 * Search dedup — fuzzy name matching to prefer regional DB (KG) over OFF.
 *
 * When both KG and OFF return the same generic food, the KG version wins
 * because it uses regional nutrition DBs (AFCD, CoFID, CIQUAL, USDA).
 * Branded OFF products (with a brand name) are always kept.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UnifiedSearchResult {
  id: string;
  name: string;
  brand?: string | null;
  source: 'kg' | 'off';
  calorieHint?: number;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/** Normalize a food name for comparison. */
function normalize(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Split into word set. */
function words(name: string): Set<string> {
  return new Set(normalize(name).split(' ').filter(Boolean));
}

/**
 * Check if two food names are similar enough to be considered the same food.
 * Uses word overlap — if 60%+ of the shorter name's words appear in the longer,
 * they're a match. Also handles substring containment.
 */
function isSimilar(a: string, b: string): boolean {
  const na = normalize(a);
  const nb = normalize(b);

  // Exact match
  if (na === nb) return true;

  // Substring containment
  if (na.includes(nb) || nb.includes(na)) return true;

  // Word overlap
  const wa = words(a);
  const wb = words(b);
  const smaller = wa.size <= wb.size ? wa : wb;
  const larger = wa.size <= wb.size ? wb : wa;

  if (smaller.size === 0) return false;

  let overlap = 0;
  for (const w of smaller) {
    // Check if the word appears in the larger set, or a close match (first 3+ chars)
    if (larger.has(w)) {
      overlap++;
    } else if (w.length >= 3) {
      // Prefix match for slight spelling variations (e.g. "makhani" vs "makhni")
      for (const lw of larger) {
        if (lw.length >= 3 && (lw.startsWith(w.substring(0, 3)) || w.startsWith(lw.substring(0, 3)))) {
          if (Math.abs(lw.length - w.length) <= 2) {
            overlap++;
            break;
          }
        }
      }
    }
  }

  return overlap / smaller.size >= 0.6;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Deduplicate search results — prefer KG (regional DB) over OFF for generic foods.
 * Branded OFF products (with a brand name) are always kept.
 * Preserves order: KG results first, then remaining OFF results.
 */
export function deduplicateResults(results: UnifiedSearchResult[]): UnifiedSearchResult[] {
  if (results.length === 0) return [];

  const kgResults = results.filter((r) => r.source === 'kg');
  const offResults = results.filter((r) => r.source === 'off');

  // Keep all KG results
  const output: UnifiedSearchResult[] = [...kgResults];

  // For each OFF result, check if it's a duplicate of any KG result
  for (const offItem of offResults) {
    // Branded products are always kept — they're specific packaged items
    if (offItem.brand && offItem.brand.trim().length > 0) {
      output.push(offItem);
      continue;
    }

    // Check if any KG result matches this OFF result
    const isDuplicate = kgResults.some((kg) => isSimilar(kg.name, offItem.name));
    if (!isDuplicate) {
      output.push(offItem);
    }
  }

  return output;
}
