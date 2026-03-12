import { act } from '@testing-library/react-native';
import { useFoodLogStore } from '../useFoodLogStore';

// Reset store between tests
beforeEach(() => {
  useFoodLogStore.setState({
    entries: [],
    selectedPhotos: [],
    isProcessing: false,
  });
});

// Mock the db client
jest.mock('../../../db/client', () => {
  const mockInsert = jest.fn().mockReturnValue({
    values: jest.fn().mockResolvedValue(undefined),
  });
  const mockUpdate = jest.fn().mockReturnValue({
    set: jest.fn().mockReturnValue({
      where: jest.fn().mockResolvedValue(undefined),
    }),
  });
  const mockSelect = jest.fn().mockReturnValue({
    from: jest.fn().mockReturnValue({
      where: jest.fn().mockResolvedValue([]),
    }),
  });

  return {
    userDb: {
      insert: mockInsert,
      update: mockUpdate,
      select: mockSelect,
    },
  };
});

jest.mock('../../../db/schema', () => ({
  foodEntries: { id: 'id', entryDate: 'entry_date', isDeleted: 'is_deleted' },
}));

describe('useFoodLogStore', () => {
  describe('addEntry', () => {
    it('writes to SQLite via userDb.insert', async () => {
      const { userDb } = require('../../../db/client');

      await act(async () => {
        await useFoodLogStore.getState().addEntry({
          mealType: 'lunch',
          totalCalories: 500,
          totalProtein: 25,
          totalCarbs: 60,
          totalFat: 15,
        });
      });

      expect(userDb.insert).toHaveBeenCalled();
    });
  });

  describe('loadTodayEntries', () => {
    it('queries SQLite for today entries', async () => {
      const { userDb } = require('../../../db/client');

      await act(async () => {
        await useFoodLogStore.getState().loadTodayEntries();
      });

      expect(userDb.select).toHaveBeenCalled();
    });
  });

  describe('deleteEntry', () => {
    it('performs soft-delete (sets isDeleted=true) rather than hard delete', async () => {
      const { userDb } = require('../../../db/client');

      const mockSet = jest.fn().mockReturnValue({
        where: jest.fn().mockResolvedValue(undefined),
      });
      userDb.update.mockReturnValue({ set: mockSet });

      await act(async () => {
        await useFoodLogStore.getState().deleteEntry('test-id');
      });

      expect(userDb.update).toHaveBeenCalled();
      expect(mockSet).toHaveBeenCalledWith(
        expect.objectContaining({ isDeleted: true })
      );
    });
  });

  describe('updateEntry', () => {
    it('writes to SQLite then refreshes', async () => {
      const { userDb } = require('../../../db/client');

      const mockSet = jest.fn().mockReturnValue({
        where: jest.fn().mockResolvedValue(undefined),
      });
      userDb.update.mockReturnValue({ set: mockSet });

      await act(async () => {
        await useFoodLogStore.getState().updateEntry('test-id', {
          totalCalories: 600,
        });
      });

      expect(userDb.update).toHaveBeenCalled();
      expect(mockSet).toHaveBeenCalledWith(
        expect.objectContaining({ totalCalories: 600 })
      );
    });
  });

  describe('getTodayTotals', () => {
    it('aggregates from cached entries correctly', () => {
      useFoodLogStore.setState({
        entries: [
          {
            id: '1',
            mealType: 'breakfast',
            entryDate: new Date().toISOString().split('T')[0],
            totalCalories: 400,
            totalProtein: 20,
            totalCarbs: 50,
            totalFat: 10,
            createdAt: new Date().toISOString(),
            isSynced: false,
            isDeleted: false,
          },
          {
            id: '2',
            mealType: 'lunch',
            entryDate: new Date().toISOString().split('T')[0],
            totalCalories: 600,
            totalProtein: 30,
            totalCarbs: 70,
            totalFat: 20,
            createdAt: new Date().toISOString(),
            isSynced: false,
            isDeleted: false,
          },
        ] as any,
      });

      const totals = useFoodLogStore.getState().getTodayTotals();
      expect(totals.calories).toBe(1000);
      expect(totals.protein).toBe(50);
      expect(totals.carbs).toBe(120);
      expect(totals.fat).toBe(30);
    });
  });
});

describe('Type contracts', () => {
  it('FoodEntry type has no userId field', () => {
    // Import types and verify the type structure
    const types = require('../../types');
    // FoodEntry is an interface, so we verify by checking the module doesn't
    // reference userId in its exports (compilation-level check)
    const typeSource = require.resolve('../../types');
    const fs = require('fs');
    const content = fs.readFileSync(typeSource, 'utf8');
    expect(content).not.toMatch(/userId\s*[?:]?\s*:\s*string/);
  });

  it('Photo type has no gcsUrl field, has localPath', () => {
    const typeSource = require.resolve('../../types');
    const fs = require('fs');
    const content = fs.readFileSync(typeSource, 'utf8');
    expect(content).not.toMatch(/gcsUrl\s*[?:]?\s*:/);
    expect(content).toMatch(/localPath\s*\??\s*:/);
  });

  it('No APIResponse export from types', () => {
    const types = require('../../types');
    expect(types.APIResponse).toBeUndefined();
    // Also verify the source doesn't define it
    const typeSource = require.resolve('../../types');
    const fs = require('fs');
    const content = fs.readFileSync(typeSource, 'utf8');
    expect(content).not.toMatch(/export\s+interface\s+APIResponse/);
  });
});
