---
phase: 01-infrastructure-data-foundation
plan: 01
subsystem: database
tags: [op-sqlite, drizzle-orm, sqlite, zustand, expo, local-first]

# Dependency graph
requires: []
provides:
  - "op-sqlite + drizzle-orm SQLite storage layer with WAL mode and foreign keys"
  - "14-table schema migrated from PostgreSQL (cloud fields removed, sync metadata added)"
  - "userDb (read-write) and openNutritionDb factory (read-only) database clients"
  - "SQLite-backed Zustand store with write-first-then-refresh pattern"
  - "Refactored types without cloud dependencies (no userId, gcsUrl, APIResponse)"
  - "Jest test infrastructure with op-sqlite mock"
  - "Drizzle migration system with initial migration"
affects: [01-02, 01-03, 02-detection-pipeline, 03-nutrition-data]

# Tech tracking
tech-stack:
  added: ["@op-engineering/op-sqlite", "drizzle-orm", "drizzle-kit", "babel-plugin-inline-import", "babel-preset-expo"]
  patterns: ["write-first-then-refresh (SQLite source of truth)", "soft-delete via isDeleted flag", "op-sqlite autolink via codegen (no config plugin needed for v15.x)"]

key-files:
  created:
    - "apps/mobile/db/schema.ts"
    - "apps/mobile/db/client.ts"
    - "apps/mobile/db/migrations.ts"
    - "apps/mobile/babel.config.js"
    - "apps/mobile/metro.config.js"
    - "apps/mobile/drizzle.config.ts"
    - "apps/mobile/jest.config.js"
    - "apps/mobile/__mocks__/@op-engineering/op-sqlite.ts"
    - "apps/mobile/src/db/__tests__/schema.test.ts"
    - "apps/mobile/src/store/__tests__/useFoodLogStore.test.ts"
    - "apps/mobile/drizzle/0000_cuddly_rachel_grey.sql"
  modified:
    - "apps/mobile/package.json"
    - "apps/mobile/app.json"
    - "apps/mobile/src/types/index.ts"
    - "apps/mobile/src/store/useFoodLogStore.ts"

key-decisions:
  - "op-sqlite v15.x uses React Native codegen autolinking, not Expo config plugin -- plugins array left empty in app.json"
  - "jest.config.js used instead of jest.config.ts to avoid ts-node dependency"
  - "All timestamps stored as ISO strings (text columns) in SQLite, not Date objects"
  - "FoodEntry.createdAt changed from Date to string for SQLite compatibility"

patterns-established:
  - "Write-first-then-refresh: all store mutations write to SQLite first, then reload cache via loadTodayEntries()"
  - "Soft-delete pattern: isDeleted flag instead of row removal, filtered in queries"
  - "db/client.ts is canonical location for all database connections (userDb + openNutritionDb)"
  - "op-sqlite mock in __mocks__/@op-engineering/op-sqlite.ts for unit testing"

requirements-completed: [DAT-01]

# Metrics
duration: 8min
completed: 2026-03-12
---

# Phase 01 Plan 01: Local Data Foundation Summary

**op-sqlite + drizzle-orm SQLite storage with 14-table schema, write-first Zustand store, and legacy cloud code deletion**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-12T06:07:29Z
- **Completed:** 2026-03-12T06:16:07Z
- **Tasks:** 2
- **Files modified:** 26148 (bulk from legacy deletion; 16 meaningful changes)

## Accomplishments
- Installed op-sqlite + drizzle-orm with full build tooling (babel, metro, drizzle-kit)
- Defined 14 SQLite tables migrated from PostgreSQL with cloud fields removed and sync metadata added
- Created dual database client: userDb (WAL, FK-enabled) and openNutritionDb factory (read-only)
- Refactored Zustand store to SQLite-backed write-first pattern with soft-delete
- Cleaned types: removed userId, gcsUrl, APIResponse, HypothesisBranch; added sync metadata
- Deleted entire backend/, services/ai-agent/, API client, and stale planning files
- Established test infrastructure: 37 tests passing across schema validation and store integration

## Task Commits

Each task was committed atomically:

1. **Task 1: Install dependencies, configure build tooling, create schema and test infrastructure** - `76c590fc` (feat)
2. **Task 2: Refactor types, store, and delete legacy code** - `4567861e` (feat)

## Files Created/Modified
- `apps/mobile/db/schema.ts` - 14 Drizzle SQLite table definitions
- `apps/mobile/db/client.ts` - op-sqlite database clients (userDb + openNutritionDb)
- `apps/mobile/db/migrations.ts` - Re-export of useMigrations for App.tsx
- `apps/mobile/babel.config.js` - Babel config with inline-import for .sql files
- `apps/mobile/metro.config.js` - Metro config with sql source extension
- `apps/mobile/drizzle.config.ts` - Drizzle-kit config for SQLite/expo
- `apps/mobile/jest.config.js` - Jest config with op-sqlite mock mapping
- `apps/mobile/__mocks__/@op-engineering/op-sqlite.ts` - Mock for unit tests
- `apps/mobile/src/db/__tests__/schema.test.ts` - 29 schema validation tests
- `apps/mobile/src/store/__tests__/useFoodLogStore.test.ts` - 8 store integration tests
- `apps/mobile/drizzle/0000_cuddly_rachel_grey.sql` - Initial migration (14 tables)
- `apps/mobile/app.json` - Added plugins array (empty; op-sqlite autolinks via codegen)
- `apps/mobile/package.json` - Added op-sqlite, drizzle-orm, drizzle-kit, babel plugins
- `apps/mobile/src/types/index.ts` - Refactored: removed cloud types, added sync metadata
- `apps/mobile/src/store/useFoodLogStore.ts` - SQLite-backed with write-first pattern

## Decisions Made
- **op-sqlite v15.x autolinking:** The installed version (15.2.5) uses React Native codegen for native linking rather than an Expo config plugin. The plugins array in app.json is left empty. Prebuild completes successfully with op-sqlite autolinked.
- **jest.config.js over .ts:** Used JavaScript config to avoid requiring ts-node as a dependency. The plan specified .ts but this avoids an unnecessary dev dependency.
- **String timestamps in types:** Changed FoodEntry.createdAt from Date to string (ISO format) to match SQLite text column storage. This is more natural for drizzle-orm's SQLite dialect.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] op-sqlite v15.x has no Expo config plugin**
- **Found during:** Task 1 (expo prebuild)
- **Issue:** `npx expo prebuild --clean` failed with PluginError: "@op-engineering/op-sqlite" has no app.plugin.js. v15.x uses React Native codegen autolinking instead.
- **Fix:** Removed op-sqlite from app.json plugins array (left empty). Prebuild succeeds; native module is autolinked via codegenConfig in op-sqlite's package.json.
- **Files modified:** apps/mobile/app.json
- **Verification:** `npx expo prebuild --clean` completes without errors
- **Committed in:** 76c590fc (Task 1 commit)

**2. [Rule 3 - Blocking] babel-preset-expo not installed**
- **Found during:** Task 1 (jest test run)
- **Issue:** Jest failed with "Cannot find module 'babel-preset-expo'" -- the new babel.config.js references it but it wasn't a direct dependency.
- **Fix:** Installed babel-preset-expo as devDependency
- **Files modified:** apps/mobile/package.json
- **Verification:** All tests pass
- **Committed in:** 76c590fc (Task 1 commit)

**3. [Rule 3 - Blocking] jest.config.ts requires ts-node**
- **Found during:** Task 1 (jest test run)
- **Issue:** Jest cannot parse TypeScript config without ts-node installed
- **Fix:** Used jest.config.js (CommonJS) instead of jest.config.ts
- **Files modified:** apps/mobile/jest.config.js (created as .js instead of .ts)
- **Verification:** Jest runs successfully
- **Committed in:** 76c590fc (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (3 blocking)
**Impact on plan:** All fixes were necessary for correct operation. No scope creep. The op-sqlite autolinking approach is the correct pattern for v15.x.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Database layer complete: userDb and openNutritionDb ready for all subsequent plans
- Schema with 14 tables covers all data needs for Phase 1 and beyond
- Migration system ready for future schema changes
- Store pattern established for all future Zustand stores to follow
- Plan 01-02 (nutrition data) can import openNutritionDb from db/client.ts
- Plan 01-03 (regional resolver) can import from db/client.ts

## Self-Check: PASSED

All 11 created files verified present on disk. Both task commits (76c590fc, 4567861e) verified in git log. All 3 deleted directories (backend/, services/ai-agent/, apps/mobile/src/lib/api/) confirmed absent. 37 tests passing, TypeScript compiles clean.

---
*Phase: 01-infrastructure-data-foundation*
*Completed: 2026-03-12*
