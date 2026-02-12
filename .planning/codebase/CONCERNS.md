# Codebase Concerns

**Analysis Date:** 2026-02-12

## Tech Debt

**AI Service Integration Not Implemented:**
- Issue: The AI service endpoint is completely stubbed with mock data returning hardcoded ingredients
- Files: `backend/src/services/aiService.ts` (line 27), `apps/mobile/src/screens/HomeScreen.tsx` (line 40), `apps/mobile/src/lib/api/foodLogApi.ts` (line 25)
- Impact: AI photo processing returns fake results, won't work in production. Integration with Google ADK agents is pending
- Fix approach: Replace mock data generation with actual call to the AI agent service once Google ADK is integrated

**Unvalidated Input in Service Layer:**
- Issue: Service layer functions accept any data structure without validation. Routes do minimal validation (only check required fields exist)
- Files: `backend/src/services/foodEntryService.ts`, `backend/src/services/recipeService.ts`, `backend/src/routes/foodLogs.ts`
- Impact: Invalid data (negative quantities, malformed strings, SQL injection via unparameterized queries) could corrupt database
- Fix approach: Add schema validation using zod or similar before database operations. Validate nutrition values are positive numbers

**Overly Permissive CORS:**
- Issue: CORS is enabled without origin restrictions
- Files: `backend/src/index.ts` (line 11)
- Impact: Any domain can make requests to the API. In production, should whitelist specific origins
- Fix approach: Replace `cors()` with `cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') })`

**Loose Type Safety with `any`:**
- Issue: Multiple uses of `any` type bypassing TypeScript checks: function parameter mappings, database error handler, test fixtures
- Files: `backend/src/db/client.ts` (line 23), `backend/src/index.ts` (line 24), `backend/src/services/foodEntryService.ts` (lines 51, 77, 96, 240, 347), `backend/src/__tests__/e2e/recipes.e2e.test.ts`
- Impact: Type errors won't be caught at compile time, runtime crashes possible
- Fix approach: Replace `any` with proper types. Create strict database row types mapped from query results

**No Environment Validation:**
- Issue: DATABASE_URL is required but never validated at startup
- Files: `backend/src/db/client.ts` (line 5)
- Impact: App starts successfully with missing DATABASE_URL, crashes on first query
- Fix approach: Add startup check: `if (!process.env.DATABASE_URL) throw new Error('DATABASE_URL is required')`

**Hardcoded Production API URL:**
- Issue: Production API endpoint is placeholder comment
- Files: `apps/mobile/src/lib/api/client.ts` (line 5)
- Impact: Mobile app can't connect to actual production backend
- Fix approach: Move to environment config, ensure value is set for production builds

## Known Bugs

**Ingredient Update Logic Not Bulletproof:**
- Symptoms: Updating ingredients uses Object.entries() iteration over unvalidated updates object
- Files: `backend/src/services/foodEntryService.ts` (lines 325-370)
- Trigger: POST with arbitrary ingredient fields in update object
- Workaround: Currently parameterized queries prevent SQL injection, but type safety is weak

**Photo Upload Endpoint Not Functional:**
- Symptoms: Mobile app calls photo upload but it's not implemented—just returns local URI
- Files: `apps/mobile/src/lib/api/foodLogApi.ts` (line 25)
- Trigger: User captures and uploads photo from mobile
- Workaround: Photos are never persisted to GCS; AI service gets local URIs instead

**Modification History Not Linked to User:**
- Symptoms: Modification history records don't track which user made the change in some code paths
- Files: `backend/src/services/foodEntryService.ts` (line 357) - modified_by can be NULL
- Trigger: Any ingredient modification
- Workaround: Audit trail is incomplete for compliance/debugging

## Security Considerations

**No Authentication Middleware:**
- Risk: API endpoints are completely open to unauthenticated requests. Any user ID can be passed to retrieve/modify another user's data
- Files: `backend/src/index.ts`, `backend/src/routes/foodLogs.ts`, `backend/src/routes/recipes.ts`
- Current mitigation: None
- Recommendations: Implement JWT or API key validation middleware. Verify `userId` in request matches authenticated user

**SQL Parameter Binding Inconsistency:**
- Risk: Most queries use parameterized queries (safe), but some edge cases with dynamic field construction exist
- Files: `backend/src/services/foodEntryService.ts` (lines 236-260 dynamic query building)
- Current mitigation: Using `$${params.length}` pattern prevents injection, but is error-prone
- Recommendations: Use query builder library (knex, slonik) instead of string concatenation

**No Rate Limiting:**
- Risk: Malicious client can spam /api/food-logs/process (expensive AI operation) or retrieve entries at scale
- Files: `backend/src/routes/foodLogs.ts`, `backend/src/routes/recipes.ts`
- Current mitigation: None
- Recommendations: Add rate limiting middleware (express-rate-limit) with configurable limits per user

**No Input Sanitization:**
- Risk: Notes field and ingredient names accepted as raw strings, could contain XSS payloads if rendered client-side without escaping
- Files: `backend/src/services/foodEntryService.ts` (lines 146-155)
- Current mitigation: Stored as plain text (not executed), but mobile client could be vulnerable
- Recommendations: Sanitize on client before render; server-side validation of character sets

**No Secrets Rotation Plan:**
- Risk: DATABASE_URL and other secrets are environment variables with no rotation strategy
- Files: `backend/src/db/client.ts` (line 5)
- Current mitigation: None
- Recommendations: Plan for secret rotation (AWS Secrets Manager, HashiCorp Vault) before production

## Performance Bottlenecks

**Food Entry Service Large Result Sets:**
- Problem: `getFoodEntries()` retrieves all photos and ingredients for every entry without pagination, even for date ranges spanning months
- Files: `backend/src/services/foodEntryService.ts` (lines 228-290)
- Cause: No `LIMIT` clause in queries. With 1000s of entries, memory usage explodes
- Improvement path: Add pagination (offset/limit), implement cursor-based pagination for mobile app, lazy-load related objects

**No Database Indexes:**
- Problem: Queries filter by user_id and entry_date constantly but schema likely has no indexes
- Files: `backend/db/init.sql` (lines 31-43)
- Cause: init.sql creates tables without indexes
- Improvement path: Add indexes: `CREATE INDEX idx_food_entries_user_id ON food_entries(user_id)`, `CREATE INDEX idx_food_entries_entry_date ON food_entries(user_id, entry_date)`

**Loop-Based Database Inserts:**
- Problem: Creating a food entry with 5 photos and 10 ingredients = 15 separate DB queries (1 entry + 5 photo + 9 ingredient queries)
- Files: `backend/src/services/foodEntryService.ts` (lines 161-205)
- Cause: Inserting related records one-by-one instead of batch insert
- Improvement path: Use multi-value INSERT: `INSERT INTO photos (...) VALUES (...), (...), ...` or batch with Promise.all

**No Query Result Caching:**
- Problem: Same user's food logs queried repeatedly throughout day
- Files: `backend/src/services/foodEntryService.ts` (lines 228-290)
- Cause: No caching layer (Redis)
- Improvement path: Add Redis cache with 5-minute TTL for user's daily food logs

**Synchronous Serial Recipe Creation:**
- Problem: Creating recipe copies photos/ingredients sequentially in transaction, even though they're independent
- Files: `backend/src/services/recipeService.ts` (lines 61-89)
- Cause: for loop with await
- Improvement path: Use Promise.all for parallel inserts within transaction

## Fragile Areas

**foodEntryService.ts - 466 Lines:**
- Files: `backend/src/services/foodEntryService.ts`
- Why fragile: Monolithic service handles create, read, update, delete, and modification history. Multiple concerns mixed. Lots of object mapping logic
- Safe modification: Split into separate files (createFoodEntry.ts, getFoodEntries.ts, updateIngredient.ts). Move mapPhoto/mapIngredient to separate mapper module
- Test coverage: Only unit tests for foodEntryService exist (181 lines). Missing tests for edge cases: empty ingredients array, NULL optional fields, date boundaries

**aiService.ts - Mock Data Hardcoded:**
- Files: `backend/src/services/aiService.ts`
- Why fragile: Completely fake implementation. When real AI service is integrated, every call site will break if return schema changes
- Safe modification: Create integration test mocking actual AI agent response. Define strict TypeScript interfaces for AI response before integration
- Test coverage: No tests exist for aiService

**Recipe Photo Insertion Logic:**
- Files: `backend/src/services/recipeService.ts` (line 81)
- Why fragile: `photos.length === 0` check determines isPrimary, but relies on insertion order. If query returns rows out of order, wrong photo marked primary
- Safe modification: Explicitly mark first photo as primary before insertion, don't rely on loop index
- Test coverage: Tests exist but don't verify photo primary flag correctness

**HomeScreen Component - UI Blocked by TODO:**
- Files: `apps/mobile/src/screens/HomeScreen.tsx`
- Why fragile: Photo processing is simulated, not calling real service. Manual QA needed when real service is added
- Safe modification: Extract photo processing to separate service hook. Test against mock AI service before connecting real one
- Test coverage: No tests for HomeScreen component

## Scaling Limits

**Database Connection Pool Fixed at 20:**
- Current capacity: 20 concurrent connections
- Limit: With multiple application instances, pool exhaustion under heavy load (e.g., 10 app servers × 2 connections = max 20 concurrent users)
- Scaling path: Increase pool size in `backend/src/db/client.ts` (max: 20 → 50), or use managed connection pooling (PgBouncer)

**In-Memory User Session in aiService:**
- Current capacity: All processing happens in single app instance
- Limit: If 100 concurrent users process photos simultaneously, app crashes or drops requests
- Scaling path: Move AI processing to background job queue (Bull, RabbitMQ) and return job ID to client for polling

**Photo Storage Design:**
- Current capacity: Photos reference GCS URLs but upload is not implemented, so all photos are local URIs
- Limit: Mobile app storage fills up quickly (10 photos × 5MB = 50MB), database schema expects GCS URLs
- Scaling path: Implement actual GCS upload. Clean up old photos. Add soft delete (is_deleted flag) instead of hard delete

## Dependencies at Risk

**Express.js Framework:**
- Risk: Express 4.18.2 is not the latest, missing recent security patches
- Impact: Known vulnerabilities in middleware, potential XSS/RCE vectors
- Migration plan: Upgrade to Express 5.x (available in beta) or switch to Fastify/Hono for better TypeScript support

**PostgreSQL pg Library 8.18.0:**
- Risk: pg library receives infrequent updates, newer versions have performance improvements
- Impact: Slow pooling, connection leaks in edge cases
- Migration plan: Upgrade to pg 9.x when available, or use postgres (native) library

**No TypeScript strict Mode:**
- Risk: TypeScript config likely has `strict: false`, allowing unsafe patterns
- Impact: Silent bugs from loose typing
- Migration plan: Enable strict mode, fix type errors incrementally, use `@ts-expect-error` for temporary escapes

## Missing Critical Features

**No Pagination Support:**
- Problem: Mobile app has no way to load food logs without fetching entire history
- Blocks: Showing monthly history screen, efficient caching, offline support
- Impact: High: Data fetching performance will degrade as user logs grow

**No Batch Photo Upload:**
- Problem: Mobile app captures multiple photos but can't upload them together
- Blocks: Reliable multi-photo processing, atomic food entry creation
- Impact: Medium: User experience suffers, entries incomplete if app crashes mid-upload

**No Offline Support:**
- Problem: App requires network access to process photos
- Blocks: Offline-first mobile apps, poor UX in low-connectivity areas
- Impact: Medium: Power users will be frustrated by network dependency

**No Food Item Search/Autocomplete:**
- Problem: When manually editing ingredients, no way to search nutrition database
- Blocks: Fast ingredient correction, confidence in macros
- Impact: Medium: Users type wrong names, macros don't match reality

## Test Coverage Gaps

**aiService.ts - No Tests:**
- What's not tested: processFoodPhotos() mock implementation, getFoodLogs() integration with foodEntryService
- Files: `backend/src/services/aiService.ts`
- Risk: AI service integration is completely untested. When real service is added, no regression detection
- Priority: High - this is a critical path feature

**Routes Layer - Minimal Tests:**
- What's not tested: Input validation in foodLogs.ts and recipes.ts routes, error handling for malformed requests
- Files: `backend/src/routes/foodLogs.ts` (123 lines), `backend/src/routes/recipes.ts` (121 lines)
- Risk: Invalid requests could cause 500 errors instead of 400 bad request. No test coverage for edge cases
- Priority: High - routes are exposed to clients

**Database Migration/Schema - No Version Control:**
- What's not tested: Schema is in init.sql but no migration system (Flyway, Knex migrations) to track changes
- Files: `backend/db/init.sql`
- Risk: If schema needs updating, no clear path. Existing databases won't upgrade
- Priority: Medium - blocks production deployment

**Mobile Client - No Component Tests:**
- What's not tested: PhotoPicker, BatchPhotoGrid, HomeScreen component rendering and interactions
- Files: `apps/mobile/src/screens/HomeScreen.tsx`, `apps/mobile/src/components/*`
- Risk: UI bugs won't be caught. Regressions in user flow (photo selection, processing) not detected
- Priority: Medium - user-facing functionality should have tests

**Type Safety - Limited Type Tests:**
- What's not tested: API response types match frontend expectations, type guards for optional fields
- Files: `apps/mobile/src/types`, `backend/src/services/*`
- Risk: Frontend renders undefined values, crashes from missing fields
- Priority: Low - mostly mitigated by TypeScript, but integration tests would help

---

*Concerns audit: 2026-02-12*
