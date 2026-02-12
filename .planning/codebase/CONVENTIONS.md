# Coding Conventions

**Analysis Date:** 2026-02-12

## Naming Patterns

**Files:**
- Services: `*Service.ts` (e.g., `foodEntryService.ts`, `recipeService.ts`)
- Routes: `*.ts` with descriptive names (e.g., `foodLogs.ts`, `recipes.ts`)
- Database: `client.ts` for connection management
- Tests: `*.test.ts` for unit tests, `*.e2e.test.ts` for integration tests
- Test directories: `__tests__` folder

**Functions:**
- camelCase for all function names
- Verb-first naming: `createFoodEntry()`, `getFoodEntry()`, `updateIngredient()`, `deleteRecipe()`
- Async functions consistently return Promise types
- Helper functions prefixed with common verbs: `map*` for transformations (e.g., `mapPhoto()`, `mapIngredient()`)

**Variables:**
- camelCase throughout
- Boolean variables prefixed with `is` or `has` (e.g., `isPrimary`, `userModified`)
- Database row mappings use snake_case in queries, converted to camelCase in returned objects
- Constants in database connection use screaming snake case: `DATABASE_URL`, `PORT`

**Types:**
- PascalCase for interfaces and types (e.g., `CreateFoodEntryParams`, `FoodEntry`, `CustomRecipe`)
- Interfaces export with `export interface` pattern
- Type unions for specific enums: `mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack'`

## Code Style

**Formatting:**
- TypeScript with strict mode enabled (`strict: true` in tsconfig.json)
- Target ES2022 with ESNext modules
- No explicit prettier/eslint config detected; follows implicit formatting conventions
- Consistent 2-space indentation
- Semicolons used throughout

**Linting:**
- No `.eslintrc` or `.prettierrc` files present
- TypeScript strict mode enforced via compiler options

## Import Organization

**Order:**
1. External framework imports (e.g., `import express from 'express'`)
2. Type imports (e.g., `import type { PoolClient } from 'pg'`)
3. Local service/utility imports (e.g., `import * as foodEntryService from '../services/foodEntryService.js'`)
4. Relative imports use `.js` extensions (ESM requirement)

**Path Aliases:**
- None configured; uses relative paths with `../` notation
- Absolute imports from `src/` root

**Examples:**
```typescript
// Route handler imports
import express from 'express';
import * as recipeService from '../services/recipeService.js';

// Service imports
import pool, { transaction } from '../db/client.js';
import type { PoolClient } from 'pg';
```

## Error Handling

**Patterns:**
- Try-catch blocks in route handlers
- Error type guards: `error instanceof Error ? error.message : 'default error'`
- Database errors propagate naturally from transaction helper
- Route handlers always respond with JSON error objects: `{ error: string }`
- HTTP status codes used properly: 400 for validation, 404 for not found, 500 for server errors

**Example from `foodLogs.ts`:**
```typescript
} catch (error) {
  console.error('Error processing photos:', error);
  res.status(500).json({
    error: error instanceof Error ? error.message : 'Failed to process photos',
  });
}
```

## Logging

**Framework:** `console` methods (console.log, console.error)

**Patterns:**
- Info logs use template strings with context: ``console.log(`Processing ${photos.length} photos for user ${userId}`);``
- Error logs include error object: `console.error('Error context:', error);`
- Database operations log execution details: query text, duration, row count
- Route handlers log errors before responding

**Examples:**
- `console.log('🚀 Food Tracker API running on http://localhost:${PORT}');` - startup
- `console.log('Executed query', { text, duration, rows: res.rowCount });` - performance
- `console.error('Error processing photos:', error);` - error handling

## Comments

**When to Comment:**
- Block comments above functions explaining purpose, inputs, outputs
- JSDoc-style documentation for exported functions
- No inline comments for obvious code

**JSDoc/TSDoc:**
- Functions have block comment headers
- Format: `/** * Purpose description */` above function declaration
- Include function purpose and responsibility

**Example:**
```typescript
/**
 * Map database row to camelCase photo object
 */
function mapPhoto(p: any) {
  return {
    id: p.id,
    // ... mapping
  };
}

/**
 * Create a food entry with photos and ingredients
 */
export async function createFoodEntry(params: CreateFoodEntryParams): Promise<FoodEntry> {
  // ...
}
```

## Function Design

**Size:**
- Functions range from 10-50 lines for service functions
- Route handlers 25-40 lines including error handling
- Helper functions (mappers) stay under 20 lines

**Parameters:**
- Single parameter objects preferred for complex inputs (e.g., `CreateFoodEntryParams`)
- Optional parameters use the `?` syntax
- Typed parameter objects exported as interfaces

**Return Values:**
- Explicit Promise types for async functions: `Promise<FoodEntry>`
- Nullable returns use `| null` pattern: `Promise<FoodEntry | null>`
- Consistent return of transformed objects (database rows converted to camelCase)

## Module Design

**Exports:**
- Named exports for functions and interfaces
- Default exports for Express routers only
- All service functions exported explicitly
- Type definitions exported as `export interface`

**Example from `foodEntryService.ts`:**
```typescript
export interface CreateFoodEntryParams { /* ... */ }
export interface FoodEntry { /* ... */ }
export async function createFoodEntry(params: CreateFoodEntryParams): Promise<FoodEntry> { /* ... */ }
export async function getFoodEntries(...): Promise<FoodEntry[]> { /* ... */ }
```

**Barrel Files:**
- Not used; imports are specific to needed functions
- Services import only required functions and types

## Database Conventions

**SQL Naming:**
- Table names: snake_case (food_entries, custom_recipes)
- Column names: snake_case (user_id, entry_date, total_calories)
- Automatic timestamp columns: created_at, updated_at

**Query Patterns:**
- Parameterized queries with $1, $2 notation
- String interpolation for dynamic query construction with parameter counts
- Transactions use helper: `transaction(async (client) => { /* ... */ })`
- Results mapped immediately: database rows converted to camelCase objects

---

*Convention analysis: 2026-02-12*
