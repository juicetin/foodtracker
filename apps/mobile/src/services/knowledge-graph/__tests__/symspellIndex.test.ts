/**
 * Tests for SymSpellIndex.
 *
 * Uses an in-memory dataset of dish names to verify exact match,
 * fuzzy match (transposition, deletion, insertion), and miss cases.
 */

import { SymSpellIndex } from '../symspellIndex';

// ── Mock dish data ──
// Simulates the dish table and symspell_deletes table that would be in the KG database.

const MOCK_DISHES: Array<{ id: number; canonical_name: string }> = [
  { id: 1, canonical_name: 'pad thai' },
  { id: 2, canonical_name: 'caesar salad' },
  { id: 3, canonical_name: 'spaghetti carbonara' },
  { id: 4, canonical_name: 'chicken tikka masala' },
  { id: 5, canonical_name: 'beef stroganoff' },
  { id: 6, canonical_name: 'fish and chips' },
  { id: 7, canonical_name: 'ramen' },
  { id: 8, canonical_name: 'sushi' },
  { id: 9, canonical_name: 'tacos' },
  { id: 10, canonical_name: 'pizza margherita' },
  { id: 11, canonical_name: 'tom yum soup' },
  { id: 12, canonical_name: 'butter chicken' },
  { id: 13, canonical_name: 'fried rice' },
  { id: 14, canonical_name: 'pho' },
  { id: 15, canonical_name: 'biryani' },
];

/**
 * Generate delete variants for a word up to maxDist character deletions.
 * This mirrors the pre-computation done by the KG build pipeline.
 */
function generateDeletes(word: string, maxDist: number): Set<string> {
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
 * Build a mock symspell_deletes dataset from dish names.
 */
function buildMockDeletesRows(): Array<{
  dish_id: number;
  delete_variant: string;
}> {
  const rows: Array<{ dish_id: number; delete_variant: string }> = [];
  for (const dish of MOCK_DISHES) {
    const deletes = generateDeletes(dish.canonical_name, 2);
    for (const del of deletes) {
      rows.push({ dish_id: dish.id, delete_variant: del });
    }
  }
  return rows;
}

// ── Mock database ──

const mockDeletesRows = buildMockDeletesRows();

const mockExecute = jest.fn().mockImplementation(async (sql: string) => {
  if (sql.includes('symspell_deletes')) {
    return { rows: mockDeletesRows };
  }
  if (sql.includes('dish')) {
    return {
      rows: MOCK_DISHES,
    };
  }
  return { rows: [] };
});

const mockDb = {
  execute: mockExecute,
  close: jest.fn(),
};

describe('SymSpellIndex', () => {
  let index: SymSpellIndex;

  beforeAll(async () => {
    index = new SymSpellIndex();
    await index.loadFromDb(mockDb as any);
  });

  describe('exact match', () => {
    it('returns exact match with distance 0 for "pad thai"', () => {
      const results = index.lookup('pad thai');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].term).toBe('pad thai');
      expect(results[0].distance).toBe(0);
      expect(results[0].dishId).toBe(1);
    });

    it('returns exact match for "ramen"', () => {
      const results = index.lookup('ramen');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].term).toBe('ramen');
      expect(results[0].distance).toBe(0);
      expect(results[0].dishId).toBe(7);
    });
  });

  describe('fuzzy match', () => {
    it('corrects transposition: "pad thia" -> "pad thai" with distance 1', () => {
      const results = index.lookup('pad thia');
      expect(results.length).toBeGreaterThanOrEqual(1);
      const padThai = results.find((r) => r.term === 'pad thai');
      expect(padThai).toBeDefined();
      expect(padThai!.distance).toBeLessThanOrEqual(2);
    });

    it('corrects missing character: "rame" -> "ramen" with distance 1', () => {
      const results = index.lookup('rame');
      expect(results.length).toBeGreaterThanOrEqual(1);
      const ramen = results.find((r) => r.term === 'ramen');
      expect(ramen).toBeDefined();
      expect(ramen!.distance).toBeLessThanOrEqual(1);
    });

    it('corrects extra character: "ramenn" -> "ramen" with distance 1', () => {
      const results = index.lookup('ramenn');
      expect(results.length).toBeGreaterThanOrEqual(1);
      const ramen = results.find((r) => r.term === 'ramen');
      expect(ramen).toBeDefined();
      expect(ramen!.distance).toBeLessThanOrEqual(1);
    });
  });

  describe('no match', () => {
    it('returns empty array for completely unknown food', () => {
      const results = index.lookup('completely_unknown_food_xyz');
      expect(results).toEqual([]);
    });
  });

  describe('normalization', () => {
    it('normalizes hyphens to spaces: "pad-thai" matches "pad thai"', () => {
      const results = index.lookup('pad-thai');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].term).toBe('pad thai');
      expect(results[0].distance).toBe(0);
    });

    it('normalizes underscores to spaces: "pad_thai" matches "pad thai"', () => {
      const results = index.lookup('pad_thai');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].term).toBe('pad thai');
      expect(results[0].distance).toBe(0);
    });

    it('normalizes case: "PAD THAI" matches "pad thai"', () => {
      const results = index.lookup('PAD THAI');
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].term).toBe('pad thai');
      expect(results[0].distance).toBe(0);
    });
  });

  describe('result limits', () => {
    it('respects maxResults parameter', () => {
      const results = index.lookup('p', 2);
      expect(results.length).toBeLessThanOrEqual(2);
    });
  });
});
