# Codebase Structure

**Analysis Date:** 2026-02-12

## Directory Layout

```
foodtracker/
├── apps/                           # Packaged applications
│   └── mobile/                     # React Native + Expo mobile app
├── backend/                        # Node.js Express API server
│   ├── src/
│   │   ├── __tests__/             # E2E tests
│   │   ├── db/                    # Database setup and client
│   │   ├── routes/                # HTTP route handlers
│   │   ├── services/              # Business logic layer
│   │   └── index.ts               # Server entry point
│   ├── db/                        # Database schema and setup
│   ├── dist/                      # Compiled JavaScript (generated)
│   ├── jest.config.js             # Test configuration
│   ├── package.json               # Dependencies and scripts
│   └── tsconfig.json              # TypeScript configuration
├── services/
│   └── ai-agent/                  # Google ADK AI agents (TypeScript)
├── spike/
│   └── food-detection-poc/        # ML/food detection experiments (Python)
├── docs/
│   └── adr/                       # Architectural decision records
├── .planning/                     # GSD planning documents
├── CLAUDE.md                      # Project conventions and tools
├── README.md                      # Project overview
└── context.md                     # Full project context

```

## Directory Purposes

**apps/mobile/**
- Purpose: React Native + Expo mobile application for iOS/Android
- Contains: Screen components, navigation, API client, state management (Zustand)
- Key files: `App.tsx`, `src/screens/`, `src/components/`, `src/lib/api/`
- Build: Expo CLI for development and building

**backend/src/**
- Purpose: Express API backend implementation
- Contains: Route handlers, service layer, database access
- Key structure:
  - `index.ts`: Server startup and middleware
  - `routes/`: HTTP request handlers
  - `services/`: Business logic
  - `db/`: Database client and pool
  - `__tests__/`: Test files

**backend/src/routes/**
- Purpose: HTTP endpoint definitions and request validation
- Files:
  - `foodLogs.ts`: Food entry photo processing and retrieval endpoints
  - `recipes.ts`: Recipe CRUD and search endpoints
- Pattern: Express Router with async handlers, try-catch error handling

**backend/src/services/**
- Purpose: Domain business logic and database operations
- Files:
  - `foodEntryService.ts` (466 lines): Food entry creation, retrieval, ingredient updates
  - `recipeService.ts` (330 lines): Custom recipe management
  - `aiService.ts` (86 lines): AI integration (currently mocked)
- Pattern: Export async functions that call database via pool/transaction

**backend/src/db/**
- Purpose: PostgreSQL client setup and query helpers
- File: `client.ts` (47 lines)
  - Pool configuration with connection limits
  - `query()` helper for logging and execution
  - `transaction()` wrapper for atomic operations
- Exports: pool, query, transaction

**backend/db/**
- Purpose: Database schema definition
- File: `init.sql` (183 lines)
  - Table definitions: users, nutrition_goals, food_entries, photos, ingredients, custom_recipes
  - Indexes for performance
  - Triggers for updated_at timestamps
  - UUID primary keys with ON DELETE CASCADE

**services/ai-agent/**
- Purpose: Google ADK agents for food detection and nutritional analysis
- Structure:
  - `src/agents/`: Agent implementations
  - `src/tools/`: Custom tools/functions
  - `src/config/`: Agent configuration
- Note: Currently not integrated; backend has mock AI responses

**spike/food-detection-poc/**
- Purpose: Python notebooks for ML model experimentation
- Type: Not production code; used for testing food detection approaches
- Tools: Jupyter notebooks, .venv (Python virtual environment)

**backend/__tests__/**
- Purpose: E2E integration tests
- Files:
  - `e2e/foodLogs.e2e.test.ts` (279 lines): Tests food entry creation and retrieval
  - `e2e/recipes.e2e.test.ts` (345 lines): Tests recipe operations
- Pattern: Create test app, set up test user, run supertest requests

**backend/src/services/__tests__/**
- Purpose: Unit tests for service logic
- Files:
  - `foodEntryService.test.ts` (181 lines): Tests food entry operations
  - `recipeService.test.ts` (180 lines): Tests recipe operations
- Pattern: Mock database pool, test service functions directly

## Key File Locations

**Entry Points:**
- Backend server: `backend/src/index.ts` - Express app initialization
- Mobile app: `apps/mobile/App.tsx` - Root component with navigation
- Database: `backend/db/init.sql` - Schema initialization

**Configuration:**
- Backend TypeScript: `backend/tsconfig.json` - ES2022, ESNext modules
- Backend tests: `backend/jest.config.js` - ts-jest with ESM support
- Backend env: `.env` file (not committed) - DATABASE_URL, PORT
- Mobile env: `apps/mobile/.env` - API endpoints

**Core Logic:**
- Food entry operations: `backend/src/services/foodEntryService.ts`
- Recipe operations: `backend/src/services/recipeService.ts`
- Database client: `backend/src/db/client.ts`
- Route handlers: `backend/src/routes/foodLogs.ts`, `recipes.ts`

**Testing:**
- E2E tests: `backend/src/__tests__/e2e/` - Full request/response cycle
- Unit tests: `backend/src/services/__tests__/` - Service function isolation
- Config: `backend/jest.config.js` - Test discovery and transformation

**Type Definitions:**
- Backend services: Interfaces exported from each service file
  - `CreateFoodEntryParams`, `FoodEntry`, `Ingredient` in foodEntryService.ts
  - `CustomRecipe` in recipeService.ts
- Mobile app: `apps/mobile/src/types/index.ts` - Shared type definitions

## Naming Conventions

**Files:**
- Services: `*Service.ts` (e.g., foodEntryService.ts, recipeService.ts)
- Routes: `*` without suffix (e.g., foodLogs.ts, recipes.ts)
- Tests: `*.test.ts` for unit tests, `*.e2e.test.ts` for E2E tests
- Config: `*.config.js` or exact names (jest.config.js, tsconfig.json)

**Directories:**
- kebab-case for multi-word directory names (e.g., `food-detection-poc`, `ai-agent`)
- Plural for collections: `routes/`, `services/`, `__tests__/`
- Singular for abstractions: `db/`, `types/`, `config/`

**Functions:**
- camelCase for all function names (e.g., `createFoodEntry`, `mapIngredient`, `updateIngredient`)
- Export as named exports from service files
- Private functions start with underscore or kept in function scope

**Variables:**
- camelCase for all variable names
- Database: Use snake_case in SQL, map to camelCase in JavaScript
- Constants: UPPER_CASE (none currently in source, use for config)

**Types/Interfaces:**
- PascalCase for all interfaces and types (e.g., `FoodEntry`, `Ingredient`, `CreateFoodEntryParams`)
- Suffixes: `*Params` for input parameters, `*Response` for API responses
- Exported from service files where domain applies

**Database:**
- Tables: snake_case, plural (food_entries, custom_recipes, recipe_ingredients)
- Columns: snake_case (user_id, meal_type, total_calories)
- Indexes: `idx_` prefix with table and columns (idx_food_entries_user_id)
- Triggers/functions: descriptive snake_case (update_updated_at_column)

## Where to Add New Code

**New Feature (e.g., Meal Planning):**
- Primary code: `backend/src/services/mealPlanService.ts` - Business logic
- Routes: `backend/src/routes/mealPlans.ts` - HTTP handlers
- Database: Add tables to `backend/db/init.sql`, add migration if needed
- Tests:
  - Unit: `backend/src/services/__tests__/mealPlanService.test.ts`
  - E2E: `backend/src/__tests__/e2e/mealPlans.e2e.test.ts`
- Mount route: Add to `backend/src/index.ts` via `app.use('/api/meal-plans', mealPlansRouter)`

**New Component (React Native):**
- Implementation: `apps/mobile/src/screens/MealPlanScreen.tsx` or `apps/mobile/src/components/MealPlanCard.tsx`
- Types: Add to `apps/mobile/src/types/index.ts` or co-locate with component
- API integration: `apps/mobile/src/lib/api/mealPlanApi.ts` (similar to foodLogApi.ts)
- Store: Add to `apps/mobile/src/store/useMealPlanStore.ts` (similar to useFoodLogStore.ts)
- Navigation: Register in `apps/mobile/src/navigation/` (RootNavigator.tsx or MainTabNavigator.tsx)

**Utilities/Helpers:**
- Backend: `backend/src/utils/` or `backend/src/helpers/` (create if doesn't exist)
- Mobile: `apps/mobile/src/lib/utils/` or co-locate with usage
- Shared types: `backend/src/types/` or `apps/mobile/src/types/`

**Database Changes:**
- Schema: `backend/db/init.sql` for initial setup
- Migrations: Create `backend/db/migrations/` with timestamped files if needed
- Test fixtures: Add INSERT statements to jest setup or test beforeEach

## Special Directories

**backend/node_modules/**
- Purpose: NPM dependencies
- Generated: Yes
- Committed: No (.gitignored)
- Build: Populated by `npm install`

**backend/dist/**
- Purpose: Compiled JavaScript output
- Generated: Yes
- Committed: No (.gitignored)
- Build: Created by `npm run build` (tsc)

**apps/mobile/node_modules/**
- Purpose: NPM dependencies for mobile app
- Generated: Yes
- Committed: No (.gitignored)
- Build: Populated by `npm install` in apps/mobile

**spike/food-detection-poc/.venv/**
- Purpose: Python virtual environment for experiments
- Generated: Yes
- Committed: No (.gitignored)
- Build: Created by `python -m venv .venv`

**.planning/codebase/**
- Purpose: GSD codebase analysis documents
- Generated: By mapping agent
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md

**docs/adr/**
- Purpose: Architectural Decision Records
- Format: Markdown with Context, Decision, Consequences sections
- Committed: Yes
- Usage: Reference in CLAUDE.md, linked from project docs
