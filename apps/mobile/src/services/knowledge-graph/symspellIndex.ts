/**
 * SymSpell fuzzy matcher loaded from pre-computed SQLite table.
 *
 * Implements the SymSpell algorithm for fast approximate string matching.
 * Pre-computed delete variants are stored in the KG database; at load time
 * they're read into a Map for O(1) lookup. Query-time generates deletes of
 * the input and intersects with the pre-computed set.
 *
 * No external npm dependency -- the algorithm is ~150 lines and stable.
 */

import type { openNutritionDb } from '../../../db/client';

type OPSQLiteConnection = ReturnType<typeof openNutritionDb>;

/** A single fuzzy match result. */
export interface SymSpellMatch {
  /** The matched dish canonical name. */
  term: string;
  /** The edit distance from the query to the matched term. */
  distance: number;
  /** The dish ID in the KG database. */
  dishId: number;
}

/**
 * SymSpell fuzzy search index.
 *
 * Loads pre-computed delete variants from the KG database and provides
 * fast typo-tolerant lookup of dish names.
 */
export class SymSpellIndex {
  /** Map from delete variant string -> array of dish IDs that produced it. */
  private deleteMap: Map<string, number[]> = new Map();

  /** Map from dish ID -> canonical name. */
  private dishNames: Map<number, string> = new Map();

  /** Set of all canonical names for exact-match short-circuit. */
  private nameSet: Set<string> = new Set();

  /** Maximum edit distance for lookup. */
  private readonly maxEditDistance = 2;

  /**
   * Load the SymSpell index from the KG database.
   * Reads the symspell_deletes table and dish names.
   */
  async loadFromDb(db: OPSQLiteConnection): Promise<void> {
    // Load dish names
    const dishResult = await db.execute(
      'SELECT id, canonical_name FROM dish'
    );
    for (const row of dishResult.rows as Array<
      Record<string, unknown>
    >) {
      const id = row.id as number;
      const name = row.canonical_name as string;
      this.dishNames.set(id, name);
      this.nameSet.add(name);
    }

    // Load pre-computed delete variants
    const deletesResult = await db.execute(
      'SELECT dish_id, delete_variant FROM symspell_deletes'
    );
    for (const row of deletesResult.rows as Array<
      Record<string, unknown>
    >) {
      const dishId = row.dish_id as number;
      const variant = row.delete_variant as string;
      const existing = this.deleteMap.get(variant);
      if (existing) {
        existing.push(dishId);
      } else {
        this.deleteMap.set(variant, [dishId]);
      }
    }
  }

  /**
   * Look up a query string and return fuzzy matches sorted by edit distance.
   *
   * @param query - The search term (will be normalized)
   * @param maxResults - Maximum number of results to return (default 5)
   * @returns Array of matches sorted by distance (ascending)
   */
  lookup(query: string, maxResults: number = 5): SymSpellMatch[] {
    const normalized = this.normalize(query);
    if (normalized.length === 0) return [];

    // Candidate dish IDs with their best known distance
    const candidates = new Map<number, number>();

    // 1. Check for exact match (distance 0)
    for (const [id, name] of this.dishNames) {
      if (name === normalized) {
        candidates.set(id, 0);
      }
    }

    // 2. Generate deletes of the query and look up in pre-computed map
    const queryDeletes = this.generateDeletes(
      normalized,
      this.maxEditDistance
    );
    // Also check the query itself as a delete variant of a longer term
    queryDeletes.add(normalized);

    for (const del of queryDeletes) {
      const dishIds = this.deleteMap.get(del);
      if (!dishIds) continue;

      for (const dishId of dishIds) {
        if (candidates.has(dishId) && candidates.get(dishId)! === 0)
          continue;

        const dishName = this.dishNames.get(dishId);
        if (!dishName) continue;

        const dist = this.editDistance(normalized, dishName);
        if (dist <= this.maxEditDistance) {
          const existing = candidates.get(dishId);
          if (existing === undefined || dist < existing) {
            candidates.set(dishId, dist);
          }
        }
      }
    }

    // 3. Build results sorted by distance
    const results: SymSpellMatch[] = [];
    for (const [dishId, distance] of candidates) {
      const term = this.dishNames.get(dishId);
      if (term) {
        results.push({ term, distance, dishId });
      }
    }

    results.sort((a, b) => a.distance - b.distance || a.term.localeCompare(b.term));
    return results.slice(0, maxResults);
  }

  /**
   * Normalize input: lowercase, replace hyphens/underscores with spaces, trim.
   */
  private normalize(input: string): string {
    return input
      .toLowerCase()
      .replace(/[-_]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Generate all delete variants of a word up to maxDist character deletions.
   * Uses breadth-first character deletion.
   */
  private generateDeletes(word: string, maxDist: number): Set<string> {
    const deletes = new Set<string>();
    const queue: Array<{ w: string; d: number }> = [{ w: word, d: 0 }];

    while (queue.length > 0) {
      const { w, d } = queue.shift()!;
      if (d < maxDist) {
        for (let i = 0; i < w.length; i++) {
          const del = w.slice(0, i) + w.slice(i + 1);
          if (!deletes.has(del)) {
            deletes.add(del);
            queue.push({ w: del, d: d + 1 });
          }
        }
      }
    }

    return deletes;
  }

  /**
   * Compute the Damerau-Levenshtein distance between two strings.
   * Supports insertion, deletion, substitution, and transposition.
   */
  private editDistance(a: string, b: string): number {
    const lenA = a.length;
    const lenB = b.length;

    // Quick bounds check
    if (Math.abs(lenA - lenB) > this.maxEditDistance) {
      return this.maxEditDistance + 1;
    }

    if (lenA === 0) return lenB;
    if (lenB === 0) return lenA;

    // Create matrix
    const d: number[][] = [];
    for (let i = 0; i <= lenA; i++) {
      d[i] = new Array(lenB + 1);
      d[i][0] = i;
    }
    for (let j = 0; j <= lenB; j++) {
      d[0][j] = j;
    }

    for (let i = 1; i <= lenA; i++) {
      for (let j = 1; j <= lenB; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;

        d[i][j] = Math.min(
          d[i - 1][j] + 1, // deletion
          d[i][j - 1] + 1, // insertion
          d[i - 1][j - 1] + cost // substitution
        );

        // Transposition
        if (
          i > 1 &&
          j > 1 &&
          a[i - 1] === b[j - 2] &&
          a[i - 2] === b[j - 1]
        ) {
          d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + cost);
        }
      }
    }

    return d[lenA][lenB];
  }
}
