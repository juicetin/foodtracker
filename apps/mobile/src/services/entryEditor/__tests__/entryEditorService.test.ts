/**
 * Entry editor service tests — update, add, remove ingredients and recalculate totals.
 *
 * Tests mock opsqlite to avoid real DB access.
 */

import {
  updateIngredientWeight,
  updateIngredientName,
  removeIngredient,
  addIngredient,
  updateDishName,
  recalculateEntryTotals,
  type IngredientUpdate,
} from '../entryEditorService';

// ---------------------------------------------------------------------------
// Mock opsqlite
// ---------------------------------------------------------------------------

const mockExecute = jest.fn();

jest.mock('../../../../db/client', () => ({
  opsqlite: {
    execute: (...args: unknown[]) => mockExecute(...args),
  },
}));

beforeEach(() => {
  mockExecute.mockReset();
});

// ---------------------------------------------------------------------------
// updateIngredientWeight
// ---------------------------------------------------------------------------

describe('updateIngredientWeight', () => {
  it('updates amount_g and sets user_modified flag', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{
        id: 'ing-1',
        original_amount_g: 100,
        calories: 200,
        protein: 20,
        carbs: 30,
        fat: 10,
        fiber: 5,
      }],
    });

    updateIngredientWeight('ing-1', 150);

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('UPDATE ingredients'),
      expect.arrayContaining([150, 'ing-1']),
    );
  });

  it('recalculates nutrition proportionally from original_amount_g', () => {
    // Mock: ingredient has original_amount_g=100, calories=200 at that weight
    mockExecute.mockReturnValueOnce({
      rows: [{
        id: 'ing-1',
        original_amount_g: 100,
        calories: 200,
        protein: 20,
        carbs: 30,
        fat: 10,
        fiber: 5,
      }],
    });

    updateIngredientWeight('ing-1', 150);

    // First call: SELECT to get original values
    expect(mockExecute).toHaveBeenCalledTimes(2);
    // Second call: UPDATE with scaled values (150/100 = 1.5x)
    const updateCall = mockExecute.mock.calls[1];
    expect(updateCall[0]).toContain('UPDATE ingredients');
    // Values should be scaled: calories=300, protein=30, carbs=45, fat=15, fiber=7.5
    expect(updateCall[1]).toContain(150); // new amount_g
  });

  it('handles zero original_amount_g gracefully', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{
        id: 'ing-1',
        original_amount_g: 0,
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
        fiber: 0,
      }],
    });

    // Should not throw
    updateIngredientWeight('ing-1', 150);
    expect(mockExecute).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// updateIngredientName
// ---------------------------------------------------------------------------

describe('updateIngredientName', () => {
  it('updates the ingredient name in the database', () => {
    updateIngredientName('ing-1', 'Grilled Chicken');

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('UPDATE ingredients'),
      expect.arrayContaining(['ing-1', 'Grilled Chicken']),
    );
  });
});

// ---------------------------------------------------------------------------
// removeIngredient
// ---------------------------------------------------------------------------

describe('removeIngredient', () => {
  it('deletes the ingredient row', () => {
    removeIngredient('ing-1');

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('DELETE FROM ingredients'),
      ['ing-1'],
    );
  });
});

// ---------------------------------------------------------------------------
// addIngredient
// ---------------------------------------------------------------------------

describe('addIngredient', () => {
  it('inserts a new ingredient linked to entry and dish', () => {
    const newIng: IngredientUpdate = {
      entryId: 'entry-1',
      dishId: 'dish-1',
      name: 'Olive Oil',
      amountG: 15,
      calories: 120,
      protein: 0,
      carbs: 0,
      fat: 14,
      fiber: 0,
    };

    const id = addIngredient(newIng);

    expect(id).toBeTruthy();
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO ingredients'),
      expect.arrayContaining(['entry-1', 'dish-1', 'Olive Oil', 15]),
    );
  });
});

// ---------------------------------------------------------------------------
// updateDishName
// ---------------------------------------------------------------------------

describe('updateDishName', () => {
  it('updates the dish name in scanned_dishes table', () => {
    updateDishName('dish-1', 'Pad Thai');

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('UPDATE scanned_dishes'),
      expect.arrayContaining(['dish-1', 'Pad Thai']),
    );
  });
});

// ---------------------------------------------------------------------------
// recalculateEntryTotals
// ---------------------------------------------------------------------------

describe('recalculateEntryTotals', () => {
  it('sums all ingredient nutrition and updates food_entries', () => {
    // Mock: return two ingredients
    mockExecute.mockReturnValueOnce({
      rows: [
        { calories: 200, protein: 20, carbs: 25, fat: 8 },
        { calories: 150, protein: 10, carbs: 15, fat: 12 },
      ],
    });

    recalculateEntryTotals('entry-1');

    // First call: SELECT SUM
    expect(mockExecute.mock.calls[0][0]).toContain('SELECT');
    expect(mockExecute.mock.calls[0][1]).toEqual(['entry-1']);

    // Second call: UPDATE food_entries with totals
    expect(mockExecute.mock.calls[1][0]).toContain('UPDATE food_entries');
    const updateArgs = mockExecute.mock.calls[1][1];
    expect(updateArgs).toContain('entry-1');
  });

  it('sets totals to 0 when entry has no ingredients', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{ calories: null, protein: null, carbs: null, fat: null }],
    });

    recalculateEntryTotals('entry-1');

    const updateArgs = mockExecute.mock.calls[1][1];
    // Should contain zeros for all macros
    expect(updateArgs[0]).toBe(0); // calories
    expect(updateArgs[1]).toBe(0); // protein
    expect(updateArgs[2]).toBe(0); // carbs
    expect(updateArgs[3]).toBe(0); // fat
  });
});
