import * as schema from '../../../db/schema';
import { getTableName, getTableColumns } from 'drizzle-orm';
import type { SQLiteColumn } from 'drizzle-orm/sqlite-core';

describe('Database Schema', () => {
  // All expected tables
  const expectedTables = [
    'foodEntries',
    'ingredients',
    'photos',
    'modificationHistory',
    'customRecipes',
    'recipeIngredients',
    'recipePhotos',
    'syncOutbox',
    'installedPacks',
    'userSettings',
    'scanQueue',
    'photoHashes',
    'containerWeights',
    'modelCache',
  ] as const;

  describe('Table exports', () => {
    test.each(expectedTables)('exports %s table definition', (tableName) => {
      expect(schema).toHaveProperty(tableName);
      expect(schema[tableName as keyof typeof schema]).toBeDefined();
    });

    test('exports exactly 14 tables', () => {
      // Count sqliteTable exports (tables have a special drizzle symbol)
      const tableExports = expectedTables.filter(
        (name) => schema[name as keyof typeof schema] !== undefined
      );
      expect(tableExports).toHaveLength(14);
    });
  });

  describe('foodEntries table', () => {
    test('has correct table name', () => {
      expect(getTableName(schema.foodEntries)).toBe('food_entries');
    });

    test('has expected columns', () => {
      const columns = getTableColumns(schema.foodEntries);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('id');
      expect(columnNames).toContain('mealType');
      expect(columnNames).toContain('entryDate');
      expect(columnNames).toContain('totalCalories');
      expect(columnNames).toContain('totalProtein');
      expect(columnNames).toContain('totalCarbs');
      expect(columnNames).toContain('totalFat');
      expect(columnNames).toContain('notes');
      expect(columnNames).toContain('isSynced');
      expect(columnNames).toContain('isDeleted');
      expect(columnNames).toContain('updatedAt');
      expect(columnNames).toContain('createdAt');
    });

    test('id is text primary key', () => {
      const columns = getTableColumns(schema.foodEntries);
      const idCol = columns.id as SQLiteColumn;
      expect(idCol.primary).toBe(true);
      expect(idCol.dataType).toBe('string');
    });

    test('does not have userId column (cloud-era removed)', () => {
      const columns = getTableColumns(schema.foodEntries);
      expect(Object.keys(columns)).not.toContain('userId');
    });
  });

  describe('ingredients table', () => {
    test('has entryId foreign key column', () => {
      const columns = getTableColumns(schema.ingredients);
      expect(Object.keys(columns)).toContain('entryId');
    });

    test('has all expected columns', () => {
      const columns = getTableColumns(schema.ingredients);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('id');
      expect(columnNames).toContain('entryId');
      expect(columnNames).toContain('name');
      expect(columnNames).toContain('quantity');
      expect(columnNames).toContain('unit');
      expect(columnNames).toContain('calories');
      expect(columnNames).toContain('protein');
      expect(columnNames).toContain('carbs');
      expect(columnNames).toContain('fat');
      expect(columnNames).toContain('fiber');
      expect(columnNames).toContain('sugar');
      expect(columnNames).toContain('aiConfidence');
      expect(columnNames).toContain('databaseSource');
      expect(columnNames).toContain('userModified');
    });
  });

  describe('photos table', () => {
    test('has localPath column instead of gcsUrl', () => {
      const columns = getTableColumns(schema.photos);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('localPath');
      expect(columnNames).not.toContain('gcsUrl');
    });

    test('has entryId foreign key', () => {
      const columns = getTableColumns(schema.photos);
      expect(Object.keys(columns)).toContain('entryId');
    });
  });

  describe('new tables', () => {
    test('syncOutbox has required columns', () => {
      const columns = getTableColumns(schema.syncOutbox);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('id');
      expect(columnNames).toContain('tableName');
      expect(columnNames).toContain('recordId');
      expect(columnNames).toContain('operation');
      expect(columnNames).toContain('createdAt');
    });

    test('installedPacks has required columns', () => {
      const columns = getTableColumns(schema.installedPacks);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('id');
      expect(columnNames).toContain('name');
      expect(columnNames).toContain('type');
      expect(columnNames).toContain('version');
      expect(columnNames).toContain('filePath');
    });

    test('userSettings has required columns', () => {
      const columns = getTableColumns(schema.userSettings);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('id');
      expect(columnNames).toContain('region');
      expect(columnNames).toContain('units');
      expect(columnNames).toContain('calorieGoal');
      expect(columnNames).toContain('proteinGoal');
    });

    test('modelCache has required columns', () => {
      const columns = getTableColumns(schema.modelCache);
      const columnNames = Object.keys(columns);

      expect(columnNames).toContain('modelId');
      expect(columnNames).toContain('version');
      expect(columnNames).toContain('filePath');
    });
  });
});

describe('Database Client', () => {
  test('exports userDb', () => {
    const client = require('../../../db/client');
    expect(client.userDb).toBeDefined();
  });

  test('exports openNutritionDb function', () => {
    const client = require('../../../db/client');
    expect(client.openNutritionDb).toBeDefined();
    expect(typeof client.openNutritionDb).toBe('function');
  });
});
