# Architecture

**Analysis Date:** 2026-02-12

## Pattern Overview

**Overall:** Layered REST API with Service-Domain pattern

**Key Characteristics:**
- Express.js HTTP server with route-based request handling
- Service layer abstraction for business logic
- PostgreSQL database with parameterized queries
- Transaction support for multi-table operations
- Type-safe interfaces for data contracts
- Test-driven development (E2E and unit tests included)

## Layers

**HTTP/Router Layer:**
- Purpose: Handle incoming HTTP requests, validation, error responses
- Location: `backend/src/routes/` (foodLogs.ts, recipes.ts)
- Contains: Express route handlers with request/response validation
- Depends on: Service layer for business logic
- Used by: Client applications (mobile, AI agents)

**Service/Business Logic Layer:**
- Purpose: Implement domain logic, orchestrate database operations, coordinate between services
- Location: `backend/src/services/` (foodEntryService.ts, recipeService.ts, aiService.ts)
- Contains: Pure business logic, transaction management, data transformation
- Depends on: Database client (pool.query, transaction)
- Used by: Routes and other services

**Database/Data Access Layer:**
- Purpose: Provide safe query execution, connection pooling, transaction handling
- Location: `backend/src/db/client.ts`
- Contains: PostgreSQL pool setup, query helper, transaction wrapper
- Depends on: pg (PostgreSQL client)
- Used by: All services

**Configuration Layer:**
- Purpose: Load environment variables and initialize middleware
- Location: `backend/src/index.ts`
- Contains: Express app setup, CORS, JSON parsing, route mounting
- Depends on: Dotenv, route handlers

## Data Flow

**Food Photo Processing Flow:**

1. Client sends POST to `/api/food-logs/process` with photos array
2. `foodLogs.ts` route handler validates request (photos, userId)
3. Calls `aiService.processFoodPhotos()`
4. AI service calls mock AI inference (TODO: integrate real AI agent)
5. `aiService` calls `foodEntryService.createFoodEntry()` with detected ingredients
6. Service uses `transaction()` to atomically:
   - Insert food_entries row
   - Insert photos rows
   - Insert ingredients rows with nutrition data
   - Map snake_case DB rows to camelCase response objects
7. Response sent with entry ID, photos, ingredients, nutrition totals

**Recipe Creation from Entry Flow:**

1. Client sends POST to `/api/recipes` with entryId, name, description
2. `recipes.ts` route validates request parameters
3. Calls `recipeService.createRecipeFromEntry()`
4. Service:
   - Retrieves source food entry
   - Copies nutrition totals to custom_recipes row
   - Copies ingredients to recipe_ingredients rows
   - Copies photos to recipe_photos rows (marks first as primary)
   - Returns complete recipe object with all relations

**Retrospective Editing Flow:**

1. Client sends PUT to `/api/food-logs/ingredients/:id` with updated values
2. Route calls `foodEntryService.updateIngredient()`
3. Service:
   - Updates ingredients table (marks userModified=true)
   - Can log change to modification_history table
   - Recalculates and updates parent food_entries totals
   - Returns updated ingredient

## State Management

**Server-Side State:**
- PostgreSQL database is source of truth
- Transaction support ensures ACID properties
- All writes immediately persisted to database

**Client-Side State:**
- Mobile app: Zustand store (`useFoodLogStore`) for local entry cache
- Store actions: addEntry, updateEntry, deleteEntry, setSelectedPhotos
- API client: ApiClient class with method interceptors (GET, POST, PUT, DELETE)
- Auth token stored in ApiClient instance

## Key Abstractions

**Service Pattern:**
- Purpose: Encapsulate domain business logic and hide database implementation
- Examples: `foodEntryService.createFoodEntry()`, `recipeService.getRecipes()`
- Pattern: Pure functions with side effects limited to database operations

**Transaction Wrapper:**
- Purpose: Ensure atomic multi-step database operations
- Examples: `recipeService.createRecipeFromEntry()`, `foodEntryService.createFoodEntry()`
- Pattern: High-order function that receives callback, handles BEGIN/COMMIT/ROLLBACK

**Data Mapping Functions:**
- Purpose: Convert PostgreSQL snake_case columns to TypeScript camelCase objects
- Examples: `mapPhoto()`, `mapIngredient()`
- Pattern: Small pure functions applied during result processing

**Type Interfaces:**
- Purpose: Define contracts between layers
- Examples: `CreateFoodEntryParams`, `FoodEntry`, `CustomRecipe`
- Pattern: Exported from service modules, used by routes and tests

## Entry Points

**HTTP Server:**
- Location: `backend/src/index.ts`
- Triggers: `npm start` or `npm run dev`
- Responsibilities:
  - Initialize Express app
  - Mount middleware (CORS, JSON parser)
  - Mount route handlers
  - Start listening on PORT (default 3100)
  - Health check endpoint at /health

**Route Handlers:**
- Food Logs Router: `backend/src/routes/foodLogs.ts`
  - POST /api/food-logs/process - Process photos
  - GET /api/food-logs - List user entries
  - GET /api/food-logs/:id - Get single entry
  - PUT /api/food-logs/ingredients/:id - Update ingredient
  - DELETE /api/food-logs/:id - Delete entry

- Recipes Router: `backend/src/routes/recipes.ts`
  - GET /api/recipes - List user recipes
  - GET /api/recipes/search - Search recipes
  - POST /api/recipes - Create recipe from entry
  - POST /api/recipes/:id/use - Use recipe as new entry
  - DELETE /api/recipes/:id - Delete recipe

## Error Handling

**Strategy:** Centralized error middleware with status code mapping

**Patterns:**
- Routes use try-catch with specific error messages
- Custom Error objects with status property
- Centralized error handler in index.ts converts errors to JSON responses
- Service layer throws Error instances; routes catch and format

**HTTP Status Codes:**
- 200: Success
- 400: Validation error (missing required fields)
- 404: Resource not found
- 500: Server error (database, unexpected exceptions)

## Cross-Cutting Concerns

**Logging:**
- Approach: console.log throughout code
- Log points: Request start, photo processing steps, database query execution with duration
- Production: Should be replaced with structured logger (winston, pino)

**Validation:**
- Approach: Manual validation in route handlers
- Checks: Required fields (userId, photos, entryId)
- Data types: String validation for IDs, array checks for photos
- Consider: Use validation library like Zod or Joi

**Authentication:**
- Approach: Authorization header Bearer token stored in ApiClient
- Implementation: Token passed in all requests via headers
- Note: No current validation on backend; all requests accepted
- TODO: Implement JWT validation middleware

**Database:**
- Connection pooling: pg Pool with max:20, idle timeout:30s, connection timeout:2s
- Query logging: All queries logged with duration
- Transactions: Available via `transaction()` helper for multi-step operations
- Indexes: Created for common filters (user_id, entry_date, recipe_id)

**Type Safety:**
- TypeScript strict mode enabled
- Parameterized queries prevent SQL injection
- Type interfaces for request/response contracts
- Database row mapping maintains type consistency
