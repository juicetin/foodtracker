/**
 * Search dedup tests — fuzzy name matching to prefer regional DB over OFF.
 */

import { deduplicateResults, type UnifiedSearchResult } from '../searchDedup';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function kgResult(name: string, id = 'kg-1'): UnifiedSearchResult {
  return { id, name, source: 'kg', calorieHint: 200 };
}

function offResult(name: string, id = 'off-1', brand?: string): UnifiedSearchResult {
  return { id, name, brand: brand ?? null, source: 'off', calorieHint: 190 };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('deduplicateResults', () => {
  it('removes OFF duplicate when KG has exact same name', () => {
    const results = [
      kgResult('Chicken Breast', 'kg-1'),
      offResult('Chicken Breast', 'off-1'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].source).toBe('kg');
  });

  it('removes OFF duplicate with case-insensitive matching', () => {
    const results = [
      kgResult('chicken breast', 'kg-1'),
      offResult('Chicken Breast', 'off-1'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].source).toBe('kg');
  });

  it('keeps OFF result when KG has no match', () => {
    const results = [
      kgResult('Fried Rice', 'kg-1'),
      offResult('Nutella', 'off-1', 'Ferrero'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(2);
  });

  it('removes OFF duplicate with fuzzy match (substring)', () => {
    const results = [
      kgResult('Grilled Chicken Breast', 'kg-1'),
      offResult('Chicken Breast', 'off-1'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].source).toBe('kg');
  });

  it('keeps branded OFF products even if name partially matches KG', () => {
    const results = [
      kgResult('Chicken Breast', 'kg-1'),
      offResult('Chicken Breast Slices', 'off-1', 'Woolworths'),
    ];
    const deduped = deduplicateResults(results);
    // Branded product should be kept — it's a specific product, not a generic food
    expect(deduped).toHaveLength(2);
  });

  it('handles empty input', () => {
    expect(deduplicateResults([])).toEqual([]);
  });

  it('preserves order — KG results first, then OFF', () => {
    const results = [
      kgResult('Rice', 'kg-1'),
      kgResult('Pasta', 'kg-2'),
      offResult('Bread', 'off-1'),
      offResult('Cereal', 'off-2'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(4);
    expect(deduped[0].source).toBe('kg');
    expect(deduped[1].source).toBe('kg');
    expect(deduped[2].source).toBe('off');
  });

  it('deduplicates multiple matches', () => {
    const results = [
      kgResult('Chicken Breast', 'kg-1'),
      kgResult('Brown Rice', 'kg-2'),
      offResult('chicken breast', 'off-1'),
      offResult('brown rice', 'off-2'),
      offResult('Nutella', 'off-3', 'Ferrero'),
    ];
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(3); // 2 KG + 1 unique OFF
    expect(deduped.filter((r) => r.source === 'kg')).toHaveLength(2);
    expect(deduped.filter((r) => r.source === 'off')).toHaveLength(1);
  });

  it('uses word overlap for fuzzy matching', () => {
    const results = [
      kgResult('Dal Makhani', 'kg-1'),
      offResult('Dal Makhni', 'off-1'), // slight spelling difference
    ];
    // Should match because most words overlap
    const deduped = deduplicateResults(results);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].source).toBe('kg');
  });
});
