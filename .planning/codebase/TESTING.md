# Testing Patterns

**Analysis Date:** 2026-02-12

## Test Framework

**Runner:**
- Jest 30.2.0
- Config: `backend/jest.config.js`
- Uses ts-jest preset with ESM support
- Node.js test environment

**Assertion Library:**
- Jest built-in (expect API)

**Run Commands:**
```bash
npm test              # Run all tests with NODE_OPTIONS=--experimental-vm-modules
npm run test:watch   # Watch mode
npm run build        # Build TypeScript before tests
```

## Test File Organization

**Location:**
- Unit tests co-located with services: `src/services/__tests__/`
- E2E tests in dedicated directory: `src/__tests__/e2e/`

**Naming:**
- Unit tests: `{service}.test.ts` (e.g., `foodEntryService.test.ts`)
- E2E tests: `{feature}.e2e.test.ts` (e.g., `foodLogs.e2e.test.ts`, `recipes.e2e.test.ts`)

**Jest Configuration:**
```javascript
testMatch: ['**/__tests__/**/*.test.ts']
collectCoverageFrom: [
  'src/**/*.ts',
  '!src/**/*.d.ts',
  '!src/**/__tests__/**',
]
```

## Test Structure

**Suite Organization:**
```typescript
describe('Food Entry Service', () => {
  describe('createFoodEntry', () => {
    it('should create a food entry with photos and ingredients', async () => {
      // Arrange
      const entry = await foodEntryService.createFoodEntry({...});

      // Act + Assert
      expect(entry).toBeDefined();
      expect(entry.id).toBeDefined();
      expect(entry.totalCalories).toBe(165);
    });
  });
});
```

**Lifecycle Hooks:**

```typescript
beforeAll(async () => {
  // Create test user once before all tests
  await pool.query(`
    INSERT INTO users (id, email, name, region)
    VALUES ($1, 'test@test.com', 'Test User', 'AU')
    ON CONFLICT (email) DO UPDATE SET id = $1
  `, [testUserId]);
});

afterAll(async () => {
  // Clean up after all tests
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
  await pool.end();
});

beforeEach(async () => {
  // Clean up before each test for isolation
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
});
```

**Patterns:**
- Setup phase: Database initialization and user creation in `beforeAll`
- Cleanup per-test: State reset in `beforeEach` to ensure isolation
- Teardown phase: Final cleanup and connection closing in `afterAll`
- Assertion phase: Explicit expect statements checking both existence and values

## Test Types

**Unit Tests:**
- Location: `src/services/__tests__/`
- Scope: Test service functions in isolation
- Approach: Direct function calls with test data, database queries through real connection
- Example: `foodEntryService.test.ts` tests ingredient totals calculation, modification tracking

**Integration Tests:**
- Location: `src/__tests__/e2e/`
- Scope: Test complete request/response flows through routes
- Approach: Create Express app with routes, use supertest for HTTP testing
- Example: `foodLogs.e2e.test.ts` tests POST /api/food-logs/process through full stack

**Database Tests:**
- All tests use real PostgreSQL database (test instance via DATABASE_URL)
- No mocking of database layer
- Tests verify full transaction behavior including cascading deletes

## Mocking

**Framework:** None detected

**Patterns:**
- No library mocks; tests use real database
- Mock data constructed in-memory for test payloads
- AI service mocked via delay: `await new Promise(resolve => setTimeout(resolve, 1500));`
- Returns hardcoded mock ingredients for testing

**What to Mock:**
- External APIs not yet integrated (e.g., actual AI agent service calls)
- Time-dependent operations if needed (though not currently mocked)

**What NOT to Mock:**
- Database operations (tests use real DB)
- Service layer functions (direct integration testing)
- Express/HTTP stack (use supertest instead)

## Fixtures and Factories

**Test Data:**
- Inline object construction for each test
- Common test user ID constant: `const testUserId = '00000000-0000-0000-0000-000000000001';`
- Test data varies per file to avoid cross-test contamination

**Example fixtures:**
```typescript
const testPayload = {
  userId: testUserId,
  mealType: 'lunch',
  entryDate: new Date('2026-02-02'),
  photos: [
    { uri: 'file:///photo1.jpg', width: 1920, height: 1080 },
  ],
  ingredients: [
    {
      name: 'Chicken Breast',
      quantity: 150,
      unit: 'g',
      calories: 165,
      protein: 31,
      carbs: 0,
      fat: 3.6,
      databaseSource: 'AFCD',
    },
  ],
  notes: 'Test meal',
};
```

**Location:**
- Fixtures defined at top of test file before test suites
- No shared fixture files; each test file self-contained

## Coverage

**Requirements:** Not enforced (no coverage thresholds in config)

**Measurement:**
```bash
npm test -- --coverage  # Generate coverage report (would need --coverage flag added)
```

**Collect Config:**
```javascript
collectCoverageFrom: [
  'src/**/*.ts',
  '!src/**/*.d.ts',
  '!src/**/__tests__/**',
]
```

## Async Testing

**Pattern:**
```typescript
it('should filter entries by date range', async () => {
  const today = new Date('2026-02-02');
  const yesterday = new Date('2026-02-01');

  // Async operations without await
  await foodEntryService.createFoodEntry({
    userId: testUserId,
    mealType: 'lunch',
    entryDate: today,
    photos: [],
    ingredients: [],
  });

  // Multiple async calls with await
  const entries = await foodEntryService.getFoodEntries(testUserId, {
    startDate: today,
    endDate: today,
  });

  // Assertions
  expect(entries.length).toBe(1);
});
```

**Key Pattern:**
- All async operations awaited explicitly
- Promise.all used for parallel operations when needed
- Test function marked async
- Jest handles async test completion automatically

## HTTP Testing (E2E)

**Framework:** supertest 7.2.2

**Pattern:**
```typescript
const response = await request(app)
  .post('/api/food-logs/process')
  .send({
    photos: [...],
    userId: testUserId,
    userRegion: 'AU',
    mealType: 'lunch',
  })
  .expect(200);

expect(response.body.entry).toBeDefined();
expect(response.body.entry.totalCalories).toBeGreaterThan(0);
```

**Test App Setup:**
```typescript
const app = express();
app.use(cors());
app.use(express.json());
app.use('/api/food-logs', foodLogsRouter);
app.use('/api/recipes', recipesRouter);
```

## Error Testing

**Pattern:**
```typescript
it('should return 400 if photos are missing', async () => {
  const response = await request(app)
    .post('/api/food-logs/process')
    .send({
      userId: testUserId,
      userRegion: 'AU',
      mealType: 'lunch',
    })
    .expect(400);

  expect(response.body.error).toBe('Photos are required');
});
```

**Approach:**
- Test invalid input conditions
- Verify correct HTTP status code
- Check error message content
- Use `.expect(statusCode)` for quick assertions

## Test Helpers

**Database Helpers:**
- `pool.query()` for direct queries in tests
- `pool.end()` for cleanup
- Direct DELETE statements for test isolation

**Service Invocation:**
```typescript
import * as foodEntryService from '../foodEntryService.js';
import * as recipeService from '../recipeService.js';

const entry = await foodEntryService.createFoodEntry({...});
```

## Test Flow Examples

**Complete E2E Workflow (from recipes.e2e.test.ts):**
```typescript
it('should handle full recipe lifecycle', async () => {
  // 1. Process photos to create entry
  const entryResponse = await request(app)
    .post('/api/food-logs/process')
    .send({...})
    .expect(200);

  // 2. Save as recipe
  const recipeResponse = await request(app)
    .post('/api/recipes')
    .send({...})
    .expect(200);

  // 3. Search for recipe
  const searchResponse = await request(app)
    .get('/api/recipes/search')
    .query({...})
    .expect(200);

  // 4. Use recipe to create entry
  const useResponse = await request(app)
    .post(`/api/recipes/${recipeId}/use`)
    .send({...})
    .expect(200);

  // 5. Verify usage tracking
  const recipesResponse = await request(app)
    .get('/api/recipes')
    .query({...})
    .expect(200);

  expect(recipe.timesUsed).toBe(2);
});
```

---

*Testing analysis: 2026-02-12*
