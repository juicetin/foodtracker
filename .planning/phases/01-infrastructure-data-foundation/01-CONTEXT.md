# Phase 1: Infrastructure + Data Foundation - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

All local data infrastructure in place so every subsequent module has a reliable storage, query, and asset delivery layer. Covers: op-sqlite setup with drizzle-orm, schema migration from PostgreSQL, bundled USDA FDC nutrition database (fast-follow), optional regional nutrition databases (on-demand), versioned asset pack system via Cloudflare R2, legacy code cleanup, and first-run onboarding flow.

</domain>

<decisions>
## Implementation Decisions

### Nutrition DB delivery
- Fast-follow asset pack via Play Asset Delivery (Android) / iOS ODR for initial download post-install
- USDA datasets included: Foundation + SR Legacy + Survey (FNDDS) in core pack (~70-80MB), Branded as separate on-demand pack (~200-300MB)
- Two-pack split: core pack (fast-follow, auto-downloads) + Branded pack (on-demand, user-triggered)
- DB is updatable without app update via versioned packs on Cloudflare R2
- Initial delivery uses platform-native asset delivery; subsequent updates check R2 for newer versions
- R2 bucket secured via app attestation (Play Integrity on Android, App Attest on iOS) + Cloudflare Worker that validates attestation before serving signed download URLs
- The versioned pack system is built generically — handles nutrition DBs now, ML model packs (YOLO, VLM) in later phases using the same R2 bucket, manifest format, and download/cache logic

### Regional DB handling
- All three regional databases available at launch: AFCD (Australia), CoFID (UK), CIQUAL (France)
- Each delivered as on-demand packs via the same R2 versioned pack system
- Auto-suggest regional DB based on device locale (en-AU → AFCD, en-GB → CoFID, fr-* → CIQUAL)
- User can also browse all available packs in Settings
- Regional DB takes priority over USDA for matching foods — if user has AFCD installed and queries "chicken breast", AFCD result shown first, USDA as fallback
- Users can import custom SQLite packs matching a published schema spec — enables community-created nutrition packs from any source
- In-app feedback form for users to suggest new nutrition databases (name, URL/source, region)
- Schema spec documented publicly so community can create packs

### Legacy code cleanup
- Delete `backend/` and `services/ai-agent/` directories entirely (in git history if needed)
- Delete `apps/mobile/src/lib/api/` (client.ts, foodLogApi.ts, index.ts)
- Refactor `apps/mobile/src/types/index.ts`: remove userId, gcsUrl, APIResponse; add sync metadata fields (updatedAt, isSynced, isDeleted); keep core types (FoodEntry, Ingredient, Photo, DetectedItem, ScaleReading)
- Delete stale root planning files: context.md, IMPLEMENTATION_PLAN.md, PROGRESS.md
- Minimal repo restructure: delete dead directories, keep monorepo layout (apps/mobile, knowledge-graph, training, spike, docs)

### First-run experience
- Block food logging with progress screen while nutrition DB downloads: "Setting up nutrition data..." with progress bar
- Minimal onboarding during download wait: one screen to set daily calorie/macro targets and preferred units (metric/imperial), auto-detect region for DB suggestion
- If offline on first launch: queue download, allow limited app exploration (browse empty diary, configure settings), block food logging until DB arrives, auto-resume download on connectivity
- Settings > Data & Storage screen: shows installed packs with names, sizes, last updated dates, download/delete buttons — transparent about what's on-device

### Claude's Discretion
- SQLite schema design details (column types, indexes, trigger implementation)
- drizzle-orm migration runner approach
- Expo prebuild config plugin setup and ordering
- R2 version manifest format and check frequency
- Cloudflare Worker implementation details for attestation validation
- Download/cache logic implementation
- Onboarding screen layout and UX details

</decisions>

<specifics>
## Specific Ideas

- User explicitly wants ability to serve new asset packs (nutrition DBs and ML models) without requiring an app update — R2 versioned pack system is designed for this
- User wants the pack system to be open and community-extensible — published schema spec + custom import + feedback form
- App attestation (Play Integrity / App Attest) required for R2 access to prevent unauthorized downloads
- The R2 hosting cost is essentially $0.01/month at any realistic scale — zero egress fees make this viable for a zero-subscription app
- Regional DB preferred over USDA for matching foods — users get nutrition values from their local food supply standards

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `apps/mobile/src/data/food-knowledge.db`: Bundled knowledge graph SQLite — already demonstrates the pattern for bundled read-only databases via op-sqlite
- `apps/mobile/src/store/useFoodLogStore.ts`: In-memory Zustand store — refactor target for SQLite-backed persistence (write to SQLite first, refresh Zustand cache)
- `apps/mobile/src/store/usePreferencesStore.ts`: AsyncStorage-based preferences — migrate to op-sqlite for consistency
- `apps/mobile/src/types/index.ts`: Core types (FoodEntry, Ingredient, Photo, DetectedItem, ScaleReading) — keep and extend with sync metadata
- `knowledge-graph/export_mobile.py`: Existing pipeline that produces mobile-friendly SQLite — pattern for USDA DB build pipeline
- `knowledge-graph/schema.sql`: SQLite schema with FTS5 full-text search — reference for nutrition DB schema design

### Established Patterns
- Zustand for state management — continue, but as thin cache over SQLite
- Expo SDK 54 with New Architecture enabled (`newArchEnabled: true`) — JSI-based libs (op-sqlite) work
- Monorepo layout: `apps/mobile/` for RN, `knowledge-graph/` and `training/` for Python tooling

### Integration Points
- `apps/mobile/app.json`: Needs config plugins for op-sqlite, expo-background-task
- `apps/mobile/package.json`: Add op-sqlite, drizzle-orm; remove async-storage (or keep for truly ephemeral settings)
- `backend/db/init.sql`: PostgreSQL schema to migrate — food_entries, ingredients, photos, modification_history, custom_recipes, recipe_ingredients, recipe_photos tables. Drop user_id FKs, gcs_url columns; add sync metadata (updated_at, is_synced, is_deleted). Add new tables: sync_outbox, scan_queue, photo_hashes, user_settings, container_weights, model_cache

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-infrastructure-data-foundation*
*Context gathered: 2026-03-12*
