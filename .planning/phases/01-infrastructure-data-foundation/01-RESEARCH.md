# Phase 1: Infrastructure + Data Foundation - Research

**Researched:** 2026-03-12
**Domain:** Local-first SQLite data layer (op-sqlite + drizzle-orm), nutrition database bundling/delivery, versioned asset pack system, legacy code cleanup
**Confidence:** HIGH

## Summary

Phase 1 replaces the superseded cloud backend (Express + PostgreSQL) with a fully local data layer built on op-sqlite and drizzle-orm, establishes the nutrition database delivery pipeline (USDA FDC + regional databases), and creates a versioned asset pack system via Cloudflare R2. This is the foundation phase -- every subsequent phase depends on the storage and query layer built here.

The core technical challenge is threefold: (1) correctly setting up op-sqlite with drizzle-orm migrations in an Expo CNG workflow that compiles on both platforms, (2) building a pipeline to convert USDA FDC CSV data into a compact, indexed SQLite database optimized for mobile queries, and (3) implementing a generic versioned pack system that handles both nutrition databases now and ML models in later phases. The `expo-play-asset-delivery` library has a critical limitation -- it only exposes a `loadPackedAssetAsBase64` method, which is unsuitable for a 70-80MB SQLite database. The recommended approach is to use platform-native delivery for the initial install (Play Asset Delivery on Android with a custom native module to get the file path, expo-file-system download from R2 on iOS) and R2 for all subsequent updates.

**Primary recommendation:** Start with dev build workflow (Expo CNG + prebuild), then op-sqlite + drizzle-orm schema, then USDA nutrition database build pipeline, then versioned pack download system, then legacy cleanup. Do NOT attempt to use `expo-play-asset-delivery`'s base64 API for large database files -- use native Android Play Core API via a thin Expo Module for file path access.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Fast-follow asset pack via Play Asset Delivery (Android) / iOS ODR for initial download post-install
- USDA datasets included: Foundation + SR Legacy + Survey (FNDDS) in core pack (~70-80MB), Branded as separate on-demand pack (~200-300MB)
- Two-pack split: core pack (fast-follow, auto-downloads) + Branded pack (on-demand, user-triggered)
- DB is updatable without app update via versioned packs on Cloudflare R2
- Initial delivery uses platform-native asset delivery; subsequent updates check R2 for newer versions
- R2 bucket secured via app attestation (Play Integrity on Android, App Attest on iOS) + Cloudflare Worker that validates attestation before serving signed download URLs
- The versioned pack system is built generically -- handles nutrition DBs now, ML model packs (YOLO, VLM) in later phases using the same R2 bucket, manifest format, and download/cache logic
- All three regional databases available at launch: AFCD (Australia), CoFID (UK), CIQUAL (France)
- Each delivered as on-demand packs via the same R2 versioned pack system
- Auto-suggest regional DB based on device locale (en-AU -> AFCD, en-GB -> CoFID, fr-* -> CIQUAL)
- User can browse all available packs in Settings
- Regional DB takes priority over USDA for matching foods
- Users can import custom SQLite packs matching a published schema spec
- In-app feedback form for users to suggest new nutrition databases
- Schema spec documented publicly so community can create packs
- Delete `backend/` and `services/ai-agent/` directories entirely
- Delete `apps/mobile/src/lib/api/` (client.ts, foodLogApi.ts, index.ts)
- Refactor `apps/mobile/src/types/index.ts`: remove userId, gcsUrl, APIResponse; add sync metadata fields (updatedAt, isSynced, isDeleted); keep core types (FoodEntry, Ingredient, Photo, DetectedItem, ScaleReading)
- Delete stale root planning files: context.md, IMPLEMENTATION_PLAN.md, PROGRESS.md
- Minimal repo restructure: delete dead directories, keep monorepo layout
- Block food logging with progress screen while nutrition DB downloads
- Minimal onboarding during download wait: one screen to set daily calorie/macro targets and preferred units (metric/imperial), auto-detect region for DB suggestion
- If offline on first launch: queue download, allow limited app exploration, block food logging until DB arrives, auto-resume download on connectivity
- Settings > Data & Storage screen: shows installed packs with names, sizes, last updated dates, download/delete buttons

### Claude's Discretion
- SQLite schema design details (column types, indexes, trigger implementation)
- drizzle-orm migration runner approach
- Expo prebuild config plugin setup and ordering
- R2 version manifest format and check frequency
- Cloudflare Worker implementation details for attestation validation
- Download/cache logic implementation
- Onboarding screen layout and UX details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DAT-01 | All user data (food entries, recipes, preferences, history) stored locally via op-sqlite with no backend dependency | op-sqlite + drizzle-orm setup, SQLite schema design from PostgreSQL migration, Zustand-as-cache pattern, migration system via `useMigrations` hook |
| DAT-02 | User has access to bundled USDA FDC nutrition database (~50-80MB) delivered as fast-follow asset pack, available before first food log | USDA FDC CSV-to-SQLite build pipeline, platform-native asset delivery with R2 fallback, progress screen UX, FTS5 search indexing |
| DAT-03 | User can download optional regional nutrition databases (AFCD, CoFID, CIQUAL) for non-US food coverage | Regional DB format research (AFCD: Excel, CoFID: Excel, CIQUAL: Excel/XML), on-demand R2 pack delivery, locale-based auto-suggestion, priority query logic |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| @op-engineering/op-sqlite | ^15.2.5 | Local SQLite database (user data + read-only nutrition DBs) | 8-9x faster than expo-sqlite in batch operations. JSI-based zero-copy. Required for 300K+ food entry queries. Verified via npm Feb 2026. |
| drizzle-orm | latest | Type-safe query builder and schema management | Official op-sqlite adapter (`drizzle-orm/op-sqlite`). Zero runtime overhead. `useMigrations` hook for startup migration. |
| drizzle-kit | latest (dev) | Migration generation from TypeScript schema | `npx drizzle-kit generate` produces SQL files. `dialect: 'sqlite'`, `driver: 'expo'` in config. |
| babel-plugin-inline-import | latest (dev) | Bundle .sql migration files as strings | Required by drizzle-orm to inline SQL migrations into the JS bundle. Without this, migrations fail at runtime. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| expo-file-system | ^19.0.21 (existing) | Download nutrition DB packs from R2, manage cached files | All pack download/cache operations, file existence checks |
| expo-crypto | SDK 54 | Generate UUIDs for database records | `crypto.randomUUID()` -- verify availability, fallback to expo-crypto if needed |
| zustand | ^5.0.11 (existing) | Reactive state cache over SQLite | Thin cache layer; SQLite is source of truth, Zustand refreshes from queries |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| op-sqlite | expo-sqlite | expo-sqlite is simpler setup but 8-9x slower for batch operations. Nutrition DB has 300K+ rows where this matters. |
| drizzle-orm | Raw SQL strings | Raw SQL works but becomes unmaintainable sprawl. Drizzle gives type-safe queries + migration framework at zero runtime cost. |
| expo-play-asset-delivery (base64) | Custom Expo Module for Play Core API | expo-play-asset-delivery only offers `loadPackedAssetAsBase64` -- unusable for 70-80MB DB files. Must use Android's native `AssetPackLocation.assetsPath()` via custom module. |
| Cloudflare R2 | AWS S3 / Firebase Storage | R2 has zero egress fees. At any realistic scale, hosting cost is ~$0.01/month. S3 egress would cost more. Firebase adds unwanted dependency. |

**Installation:**
```bash
# In apps/mobile/
npx expo install @op-engineering/op-sqlite
npm install drizzle-orm
npm install -D drizzle-kit babel-plugin-inline-import

# Regenerate native projects
npx expo prebuild --clean
```

## Architecture Patterns

### Recommended Project Structure
```
apps/mobile/
├── db/
│   ├── schema.ts              # Drizzle table definitions (user data)
│   ├── client.ts              # op-sqlite open() + drizzle() wrapper
│   ├── migrations.ts          # Re-export from drizzle/migrations
│   └── nutrition-schema.ts    # Published schema spec for nutrition packs
├── drizzle/
│   ├── migrations/            # Generated SQL migration files
│   └── meta/                  # Drizzle migration metadata
├── drizzle.config.ts          # Drizzle Kit configuration
├── src/
│   ├── data/
│   │   └── food-knowledge.db  # Existing bundled knowledge graph (keep)
│   ├── services/
│   │   ├── nutrition/
│   │   │   ├── nutritionService.ts   # Query nutrition DBs (USDA + regional)
│   │   │   └── nutritionSchema.ts    # Published pack schema spec
│   │   └── packs/
│   │       ├── packManager.ts        # Generic versioned pack download/cache
│   │       ├── packManifest.ts       # R2 manifest format + version checking
│   │       ├── platformDelivery.ts   # Platform-native asset delivery adapter
│   │       └── types.ts             # Pack types (NutritionPack, ModelPack)
│   ├── store/
│   │   ├── useFoodLogStore.ts        # Refactor: SQLite-backed
│   │   ├── usePreferencesStore.ts    # Keep AsyncStorage (small data)
│   │   └── usePackStore.ts           # New: pack download/install state
│   └── types/
│       └── index.ts                  # Refactored: remove userId/gcsUrl/APIResponse
├── babel.config.js            # Add inline-import plugin for .sql files
└── metro.config.js            # Add 'sql' to sourceExts
```

### Pattern 1: Separate User DB and Nutrition DB

**What:** Open two separate SQLite databases -- one for user data (read-write, migrated) and one or more for nutrition reference data (read-only, replaced on update).
**When to use:** Always. This is the foundational data separation pattern.
**Why critical:** Updating the nutrition DB (overwriting the file) must never touch user data. Conversely, user data migrations must never affect the nutrition DB.

```typescript
// db/client.ts
import { open } from '@op-engineering/op-sqlite';
import { drizzle } from 'drizzle-orm/op-sqlite';

// User data -- read-write, migrated via drizzle
const userOpsqlite = open({ name: 'foodtracker.db' });
export const userDb = drizzle(userOpsqlite);

// Nutrition data -- read-only, replaced on pack update
// Opened AFTER pack download completes
export function openNutritionDb(dbPath: string) {
  const nutritionOpsqlite = open({ name: dbPath });
  // Enable read-only pragmas for safety
  nutritionOpsqlite.execute('PRAGMA query_only = ON');
  return drizzle(nutritionOpsqlite);
}
```

### Pattern 2: Drizzle Migrations at App Startup

**What:** Run `useMigrations` hook before rendering the app to ensure schema is current.
**When to use:** App root component, before any data access.

```typescript
// App.tsx
import { useMigrations } from 'drizzle-orm/op-sqlite/migrator';
import migrations from './drizzle/migrations';
import { userDb } from './db/client';

export default function App() {
  const { success, error } = useMigrations(userDb, migrations);

  if (error) {
    return <MigrationErrorScreen error={error} />;
  }
  if (!success) {
    return <SplashScreen message="Updating database..." />;
  }
  return <RootNavigator />;
}
```

**Source:** [Drizzle ORM - OP SQLite](https://orm.drizzle.team/docs/connect-op-sqlite)

### Pattern 3: Zustand as Thin Cache Over SQLite

**What:** SQLite is source of truth. Zustand stores refresh from SQL queries. Never update Zustand optimistically before SQL write succeeds.
**When to use:** All data access from UI components.

```typescript
// store/useFoodLogStore.ts (refactored)
import { create } from 'zustand';
import { userDb } from '../db/client';
import { foodEntries } from '../db/schema';
import { eq } from 'drizzle-orm';

interface FoodLogState {
  entries: FoodEntry[];
  isLoading: boolean;
  loadTodayEntries: () => Promise<void>;
  addEntry: (entry: NewFoodEntry) => Promise<string>;
}

export const useFoodLogStore = create<FoodLogState>((set, get) => ({
  entries: [],
  isLoading: false,

  loadTodayEntries: async () => {
    set({ isLoading: true });
    const today = new Date().toISOString().split('T')[0];
    const results = await userDb
      .select()
      .from(foodEntries)
      .where(eq(foodEntries.entryDate, today));
    set({ entries: results, isLoading: false });
  },

  addEntry: async (entry) => {
    const id = crypto.randomUUID();
    // Write to SQLite FIRST
    await userDb.insert(foodEntries).values({ id, ...entry });
    // Then refresh cache from SQLite
    await get().loadTodayEntries();
    return id;
  },
}));
```

### Pattern 4: Versioned Pack Manifest (R2)

**What:** A JSON manifest on R2 lists available packs with versions, sizes, and download URLs. App checks manifest periodically.
**When to use:** All pack update checks (nutrition DBs, ML models in later phases).

```typescript
// Manifest format stored at: r2-bucket/manifest.json
interface PackManifest {
  version: number;                // Manifest schema version
  lastUpdated: string;            // ISO 8601
  packs: PackEntry[];
}

interface PackEntry {
  id: string;                     // e.g., "usda-core", "afcd", "yolo-v1"
  name: string;                   // Human-readable name
  type: 'nutrition' | 'model';    // Pack category
  version: string;                // Semver, e.g., "2025.12.1"
  sizeBytes: number;              // Download size (compressed)
  sha256: string;                 // Integrity hash
  url: string;                    // R2 presigned URL path (relative)
  region?: string;                // For nutrition packs: "AU", "UK", "FR"
  locale?: string;                // Auto-suggest locale: "en-AU", "en-GB", "fr"
  description: string;
  requiredAppVersion?: string;    // Minimum app version
}
```

### Pattern 5: Pre-populated Database Loading (op-sqlite)

**What:** Use `moveAssetsDatabase()` for bundled databases, or download from R2 and open from custom path.
**When to use:** Bundled knowledge graph (already works), downloaded nutrition packs.

```typescript
// For bundled assets (knowledge graph pattern already in use):
import { moveAssetsDatabase, open } from '@op-engineering/op-sqlite';

// Move from app bundle to writable location (idempotent)
await moveAssetsDatabase({ filename: 'food-knowledge.db' });
const knowledgeDb = open({ name: 'food-knowledge.db' });

// For downloaded packs (nutrition DB from R2):
import { open, ANDROID_DATABASE_PATH, IOS_DOCUMENT_PATH } from '@op-engineering/op-sqlite';
import * as FileSystem from 'expo-file-system';

const packDir = `${FileSystem.documentDirectory}packs/`;
const dbPath = `${packDir}usda-core.db`;

// After download completes:
const nutritionDb = open({
  name: 'usda-core.db',
  location: packDir,
});
```

**Source:** [OP-SQLite Configuration Docs](https://op-engineering.github.io/op-sqlite/docs/configuration/)

### Anti-Patterns to Avoid

- **Single database for user data AND nutrition data:** Updating nutrition DB would require complex data-preserving migration. Use separate DB files.
- **Optimistic Zustand updates before SQL write:** Crashes or failed writes leave store and DB out of sync. Always write to SQLite first, then refresh cache.
- **Using `loadPackedAssetAsBase64` for large files:** expo-play-asset-delivery's base64 method will OOM or be extremely slow for a 70-80MB SQLite database. Get the file path instead.
- **Storing nutrition DB in the APK/IPA bundle:** Bloats the base install. Use fast-follow (Android) or post-install download (iOS) instead.
- **Running migrations without transactions:** A failed migration leaves the DB in a partially migrated state. Drizzle handles this internally but verify.
- **Forgetting to add 'sql' to metro.config.js sourceExts:** Migrations silently fail at runtime because .sql files are not bundled.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SQL schema migrations | Custom version-tracking migration runner | drizzle-kit generate + `useMigrations` hook | Handles version tracking, sequential execution, error rollback. Custom runners miss edge cases (skipped versions, partial migrations). |
| Type-safe SQL queries | Raw SQL string template functions | drizzle-orm query builder | Compile-time type checking, autocomplete, refactor safety. Raw SQL strings drift from schema. |
| UUID generation | Custom UUID function | `crypto.randomUUID()` | Standard API, cryptographically random, available in React Native with Hermes. |
| File download with progress | Custom XMLHttpRequest wrapper | `FileSystem.createDownloadResumable()` | Handles resume on network interruption, progress callbacks, background download on iOS. |
| SQLite FTS5 search | Custom LIKE-based search with tokenization | SQLite FTS5 virtual table | Orders of magnitude faster for full-text search. Already proven in knowledge graph schema. |
| Presigned URL generation | Custom HMAC signing | Cloudflare Worker with R2 binding | Worker has direct R2 binding, no credential exposure, handles attestation validation. |

## Common Pitfalls

### Pitfall 1: Missing Metro/Babel Config for SQL Migration Files
**What goes wrong:** Drizzle migrations silently fail because `.sql` files are not bundled into the app. The app launches, `useMigrations` returns success (no migrations found), but no tables exist.
**Why it happens:** Metro bundler does not include `.sql` files by default. The babel plugin `inline-import` must be installed and configured to inline `.sql` file contents as strings.
**How to avoid:**
1. Install `babel-plugin-inline-import`
2. Add `['inline-import', { extensions: ['.sql'] }]` to `babel.config.js` plugins
3. Create `metro.config.js` with `config.resolver.sourceExts.push('sql')`
4. Verify by checking that `migrations` import is a non-empty object after `npx drizzle-kit generate`
**Warning signs:** `useMigrations` returns `{ success: true }` immediately but queries fail with "no such table"

### Pitfall 2: expo-play-asset-delivery Cannot Handle Large Database Files
**What goes wrong:** Attempting to load a 70-80MB SQLite database via `loadPackedAssetAsBase64()` either causes OOM, extreme latency (base64 encoding doubles the size), or silent failure.
**Why it happens:** The library only exposes a base64 loading method. There is no `getFilePath()` or `getAbsoluteAssetPath()` equivalent exposed to JavaScript. The underlying Android Play Core API does support file path access via `AssetPackLocation.assetsPath()`, but the Expo wrapper does not expose it.
**How to avoid:**
1. For Android: Create a thin Expo Native Module (~50-100 lines Kotlin) that wraps Android's `AssetPackManager.getPackLocation(packName).assetsPath()` to return the file path to JavaScript
2. For iOS: Use `expo-file-system` to download from R2 on first launch (iOS has no equivalent to Play Asset Delivery fast-follow)
3. Alternative: Skip platform-native delivery entirely and download from R2 on both platforms on first launch. Simpler, at the cost of the user needing network connectivity.
**Warning signs:** Using `expo-play-asset-delivery` for any file larger than ~5MB

### Pitfall 3: Bundled Database Not Copied on App Update
**What goes wrong:** The app ships version 1.0 with nutrition DB v1. App update ships nutrition DB v2. On update, the existing DB file in the app data directory is not overwritten (by design -- to preserve user data). Users get stale nutrition data.
**Why it happens:** `moveAssetsDatabase()` is idempotent -- it skips the copy if the target file already exists. This is correct for user databases but wrong for updatable reference databases.
**How to avoid:**
1. Store a version number in the nutrition DB itself (e.g., PRAGMA `user_version`)
2. On app launch, compare bundled DB version against installed DB version
3. If bundled is newer, overwrite the nutrition DB file (it is read-only reference data, not user data)
4. Better: Use R2 versioned packs for all updates, not bundled assets. Bundle only the initial seed.
**Warning signs:** No version-checking logic for the nutrition database

### Pitfall 4: Expo CNG Config Plugin Conflicts
**What goes wrong:** Adding op-sqlite, react-native-fast-tflite, llama.rn, and other native libraries all at once causes `npx expo prebuild --clean` to fail with native compilation errors (CocoaPods conflicts on iOS, Gradle merge failures on Android).
**Why it happens:** Config plugins modify the same native files (Podfile, build.gradle). Some plugins assume they are the only modifier. Order of plugins in app.json matters.
**How to avoid:**
1. Add native dependencies ONE AT A TIME. Prebuild, compile, and verify after each addition.
2. In Phase 1, add ONLY op-sqlite. Do not add react-native-fast-tflite or llama.rn until Phase 2.
3. Test both platforms after each prebuild: `npx expo run:ios` and `npx expo run:android`
4. Keep `ios/` and `android/` in `.gitignore` -- regenerate from config plugins only
**Warning signs:** Multiple new native deps added in a single commit without per-platform verification

### Pitfall 5: SQLite REAL vs DECIMAL Precision Loss
**What goes wrong:** PostgreSQL schema uses `DECIMAL(10,2)` for nutrition values. SQLite has no fixed-point type -- `REAL` is IEEE 754 double-precision float. Values like 0.1 calories are stored as 0.10000000000000001. UI shows ugly floating point artifacts.
**Why it happens:** SQLite type affinity system treats everything as TEXT, INTEGER, REAL, or BLOB. There is no DECIMAL.
**How to avoid:**
1. Store nutrition values as integers in the smallest unit (e.g., milligrams instead of grams, 10ths of calories instead of calories)
2. Or accept REAL and round at the display layer: `value.toFixed(1)` for macros
3. Document the decision in the schema so future contributors don't try to "fix" it
**Warning signs:** UI showing values like "150.00000000000001 kcal"

## Code Examples

### Metro Config for SQL Migration Bundling
```javascript
// metro.config.js
// Source: https://orm.drizzle.team/docs/connect-op-sqlite
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.sourceExts.push('sql');
module.exports = config;
```

### Babel Config for Inline SQL Import
```javascript
// babel.config.js
// Source: https://orm.drizzle.team/docs/connect-op-sqlite
module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      ['inline-import', { extensions: ['.sql'] }],
    ],
  };
};
```

### Drizzle Config
```typescript
// drizzle.config.ts
// Source: https://orm.drizzle.team/docs/get-started/op-sqlite-new
import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  dialect: 'sqlite',
  driver: 'expo',
  schema: './db/schema.ts',
  out: './drizzle',
});
```

### SQLite Schema (Migrated from PostgreSQL)

Key migration decisions from the existing `backend/db/init.sql`:

```typescript
// db/schema.ts
import { sqliteTable, text, integer, real } from 'drizzle-orm/sqlite-core';
import { sql } from 'drizzle-orm';

// user_settings: Flattened from users + nutrition_goals (single-user, no auth)
export const userSettings = sqliteTable('user_settings', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  region: text('region').default('AU'),
  units: text('units').default('metric'),
  calorieGoal: real('calorie_goal').default(2000),
  proteinGoal: real('protein_goal').default(150),
  carbsGoal: real('carbs_goal').default(200),
  fatGoal: real('fat_goal').default(65),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
});

// food_entries: Drop user_id FK, add sync metadata
export const foodEntries = sqliteTable('food_entries', {
  id: text('id').primaryKey(),          // UUID generated in app
  mealType: text('meal_type').notNull(),
  entryDate: text('entry_date').notNull(),
  totalCalories: real('total_calories').default(0),
  totalProtein: real('total_protein').default(0),
  totalCarbs: real('total_carbs').default(0),
  totalFat: real('total_fat').default(0),
  notes: text('notes'),
  // Sync metadata (new)
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
  isSynced: integer('is_synced', { mode: 'boolean' }).default(false),
  isDeleted: integer('is_deleted', { mode: 'boolean' }).default(false),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// ingredients: Keep all columns, add entry_id FK
export const ingredients = sqliteTable('ingredients', {
  id: text('id').primaryKey(),
  entryId: text('entry_id').references(() => foodEntries.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  quantity: real('quantity').notNull(),
  unit: text('unit').notNull(),
  calories: real('calories').notNull(),
  protein: real('protein').default(0),
  carbs: real('carbs').default(0),
  fat: real('fat').default(0),
  fiber: real('fiber').default(0),
  sugar: real('sugar').default(0),
  aiConfidence: real('ai_confidence'),
  boundingBoxX: real('bounding_box_x'),
  boundingBoxY: real('bounding_box_y'),
  boundingBoxWidth: real('bounding_box_width'),
  boundingBoxHeight: real('bounding_box_height'),
  databaseSource: text('database_source'),
  databaseId: text('database_id'),
  userModified: integer('user_modified', { mode: 'boolean' }).default(false),
  originalQuantity: real('original_quantity'),
  updatedAt: text('updated_at').default(sql`(datetime('now'))`),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

// photos: Drop gcs_url, add local_path
export const photos = sqliteTable('photos', {
  id: text('id').primaryKey(),
  entryId: text('entry_id').references(() => foodEntries.id, { onDelete: 'cascade' }),
  uri: text('uri').notNull(),
  localPath: text('local_path'),
  width: integer('width'),
  height: integer('height'),
  latitude: real('latitude'),
  longitude: real('longitude'),
  uploadedAt: text('uploaded_at').default(sql`(datetime('now'))`),
});

// New tables for local-first architecture
export const syncOutbox = sqliteTable('sync_outbox', {
  id: text('id').primaryKey(),
  tableName: text('table_name').notNull(),
  recordId: text('record_id').notNull(),
  operation: text('operation').notNull(), // 'insert' | 'update' | 'delete'
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});

export const installedPacks = sqliteTable('installed_packs', {
  id: text('id').primaryKey(),       // e.g., 'usda-core', 'afcd'
  name: text('name').notNull(),
  type: text('type').notNull(),      // 'nutrition' | 'model'
  version: text('version').notNull(),
  filePath: text('file_path').notNull(),
  sizeBytes: integer('size_bytes'),
  sha256: text('sha256'),
  region: text('region'),
  installedAt: text('installed_at').default(sql`(datetime('now'))`),
  lastChecked: text('last_checked'),
});
```

### PostgreSQL to SQLite Type Mapping Reference

| PostgreSQL | SQLite (drizzle) | Notes |
|------------|-----------------|-------|
| `UUID PRIMARY KEY DEFAULT uuid_generate_v4()` | `text('id').primaryKey()` | Generate UUID in app: `crypto.randomUUID()` |
| `SERIAL / BIGSERIAL` | `integer('id').primaryKey({ autoIncrement: true })` | Auto-increment integer |
| `DECIMAL(10,2)` | `real('col')` | IEEE 754 double; round at display layer |
| `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | `text('col').default(sql\`(datetime('now'))\`)` | SQLite stores dates as TEXT |
| `VARCHAR(n)` | `text('col')` | SQLite ignores length constraints |
| `BOOLEAN` | `integer('col', { mode: 'boolean' })` | 0/1 with drizzle boolean mode |
| `REFERENCES ... ON DELETE CASCADE` | `.references(() => table.col, { onDelete: 'cascade' })` | Requires `PRAGMA foreign_keys=ON` |
| PostgreSQL trigger functions | SQLite `CREATE TRIGGER` | Different syntax but same concept |

### USDA FDC Nutrition Database Schema (Published Pack Spec)

The nutrition database follows a normalized schema derived from USDA FDC CSV structure:

```sql
-- Nutrition pack schema spec (read-only, bundled or downloaded)
-- All nutrition packs (USDA, AFCD, CoFID, CIQUAL, community) must implement this schema

CREATE TABLE foods (
  fdc_id INTEGER PRIMARY KEY,
  description TEXT NOT NULL,
  food_category TEXT,
  data_type TEXT,                    -- 'foundation', 'sr_legacy', 'survey_fndds', 'branded'
  publication_date TEXT,
  UNIQUE(description, data_type)
);

CREATE TABLE nutrients (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  unit TEXT NOT NULL,                -- 'g', 'mg', 'ug', 'kcal', 'kJ'
  nutrient_nbr TEXT
);

CREATE TABLE food_nutrients (
  food_id INTEGER REFERENCES foods(fdc_id),
  nutrient_id INTEGER REFERENCES nutrients(id),
  amount REAL NOT NULL,              -- Per 100g of food
  PRIMARY KEY (food_id, nutrient_id)
);

-- Optional: serving sizes for FNDDS foods
CREATE TABLE food_portions (
  id INTEGER PRIMARY KEY,
  food_id INTEGER REFERENCES foods(fdc_id),
  portion_description TEXT,
  gram_weight REAL,
  modifier TEXT
);

-- FTS5 for fast food name search
CREATE VIRTUAL TABLE foods_fts USING fts5(description, food_category, content=foods, content_rowid=fdc_id);

-- Indexes
CREATE INDEX idx_foods_category ON foods(food_category);
CREATE INDEX idx_foods_data_type ON foods(data_type);
CREATE INDEX idx_food_nutrients_food ON food_nutrients(food_id);
CREATE INDEX idx_food_nutrients_nutrient ON food_nutrients(nutrient_id);

-- Metadata
PRAGMA user_version = 1;            -- Increment on schema changes
```

### Pack Download with Progress
```typescript
// services/packs/packManager.ts
import * as FileSystem from 'expo-file-system';

interface DownloadProgress {
  totalBytesWritten: number;
  totalBytesExpected: number;
  fraction: number;
}

async function downloadPack(
  url: string,
  destPath: string,
  onProgress: (progress: DownloadProgress) => void,
  expectedSha256: string,
): Promise<string> {
  const downloadResumable = FileSystem.createDownloadResumable(
    url,
    destPath,
    {},
    (data) => {
      onProgress({
        totalBytesWritten: data.totalBytesWritten,
        totalBytesExpected: data.totalBytesExpectedToWrite,
        fraction: data.totalBytesWritten / data.totalBytesExpectedToWrite,
      });
    }
  );

  const result = await downloadResumable.downloadAsync();
  if (!result?.uri) throw new Error('Download failed');

  // Verify integrity
  // TODO: compute SHA-256 of downloaded file and compare to expectedSha256

  return result.uri;
}
```

## USDA FDC Data Pipeline

### Source Data Format

USDA FDC provides data as CSV downloads (no native SQLite format). The build pipeline converts CSV to SQLite.

| Dataset | Format | Compressed Size | Uncompressed | Last Updated |
|---------|--------|-----------------|--------------|--------------|
| Foundation Foods | CSV/JSON | 3.4MB | 29MB | Dec 2025 |
| SR Legacy | CSV/JSON | (included in full download) | - | Apr 2018 (final) |
| FNDDS (Survey) | CSV/JSON | 200MB | 1.6GB | Oct 2024 |
| Branded Foods | CSV/JSON | 427MB | 2.9GB | Dec 2025 |
| Full Download | CSV | 458MB | 3.1GB | Dec 2025 |

### Key USDA FDC CSV Tables

| CSV File | Key Columns | Purpose |
|----------|-------------|---------|
| `food.csv` | fdc_id, description, food_category_id, data_type | Master food list |
| `food_nutrient.csv` | fdc_id, nutrient_id, amount | Nutrient values per food (per 100g) |
| `nutrient.csv` | id, name, unit_name, nutrient_nbr | Nutrient definitions |
| `food_category.csv` | id, description | Food categories |
| `food_portion.csv` | fdc_id, portion_description, gram_weight | Serving size info |
| `branded_food.csv` | fdc_id, brand_owner, ingredients, serving_size | Branded product details |

### Build Pipeline (Python, dev-only)

A Python script (analogous to existing `knowledge-graph/export_mobile.py`) processes USDA FDC CSVs into a compact SQLite database:

1. Download USDA FDC CSV archive from fdc.nal.usda.gov
2. Parse `food.csv` -- filter to Foundation + SR Legacy + FNDDS entries for core pack
3. Parse `food_nutrient.csv` -- join with nutrient.csv for nutrient names/units
4. Parse `food_portion.csv` -- extract serving sizes
5. Insert into normalized SQLite schema (foods, nutrients, food_nutrients, food_portions)
6. Build FTS5 index on food descriptions
7. Run ANALYZE + VACUUM for query optimization
8. Set PRAGMA user_version for versioning
9. Output as single `.db` file (~50-80MB estimated for core pack)

### Regional Database Sources

| Database | Source | Format | Foods | Last Updated |
|----------|--------|--------|-------|--------------|
| AFCD (Australia) | foodstandards.gov.au | Excel (5 files) | 1,588 | 2024 |
| CoFID (UK) | gov.uk | Excel | ~3,000+ | 2021 |
| CIQUAL (France) | anses.fr/ciqual | Excel/XML | 3,484 | 2025 |

All three are available as Excel/CSV downloads. The build pipeline converts each to the published nutrition pack SQLite schema spec (same schema as USDA, with `data_type` set to the regional source identifier).

## Asset Delivery Architecture

### Platform-Specific Strategy

| Platform | Initial Install | Subsequent Updates |
|----------|----------------|-------------------|
| **Android** | Play Asset Delivery fast-follow (auto-downloads after install) | Check R2 manifest, download from R2 |
| **iOS** | Download from R2 on first launch (no iOS equivalent to fast-follow) | Check R2 manifest, download from R2 |

### Critical: Android Play Asset Delivery File Access

`expo-play-asset-delivery` only provides `loadPackedAssetAsBase64()` -- unsuitable for large SQLite databases. Two options:

**Option A (Recommended for Phase 1):** Skip platform-native delivery. Download from R2 on first launch on BOTH platforms. Simpler, fewer native modules, same UX (progress screen during download). Add Play Asset Delivery optimization later if needed.

**Option B (Full implementation):** Create a thin Expo Native Module in Kotlin that wraps Android's Play Core AssetPackManager:

```kotlin
// modules/play-asset-path/android/src/main/.../PlayAssetPathModule.kt
// ~50 lines: getAssetPackPath(packName: String): String?
// Returns AssetPackLocation.assetsPath() for completed packs
```

This lets you get the absolute file path to the SQLite DB within the asset pack, which op-sqlite can open directly.

**Recommendation for Phase 1:** Use Option A (R2 download on both platforms). It is simpler, requires no custom native modules, and the R2 infrastructure is needed anyway for updates. Platform-native delivery can be added as a performance optimization in Phase 6 (Distribution) when the app is closer to release.

### R2 Bucket Structure

```
foodtracker-assets/
├── manifest.json                        # Pack manifest (checked on app launch)
├── nutrition/
│   ├── usda-core/
│   │   ├── v2025.12.1/usda-core.db.gz  # Gzipped SQLite
│   │   └── v2025.12.1/checksum.sha256
│   ├── usda-branded/
│   │   └── v2025.12.1/usda-branded.db.gz
│   ├── afcd/
│   │   └── v2024.1.0/afcd.db.gz
│   ├── cofid/
│   │   └── v2021.1.0/cofid.db.gz
│   └── ciqual/
│       └── v2025.1.0/ciqual.db.gz
└── models/                              # Future: ML model packs
    └── (Phase 2+)
```

### Cloudflare Worker (Attestation + Presigned URLs)

```
Client                    Worker                           R2
  |                         |                              |
  |-- attestation token --> |                              |
  |                         |-- validate token             |
  |                         |   (Play Integrity / App Attest)
  |                         |                              |
  |  <-- manifest.json ---  |-- fetch manifest.json -----> |
  |                         |                              |
  |-- request pack URL ---> |                              |
  |                         |-- generate presigned URL ---> |
  |  <-- presigned URL ---  |                              |
  |                         |                              |
  |-- download pack directly from R2 via presigned URL --> |
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PostgreSQL on cloud server | op-sqlite local SQLite | ADR-005 (Mar 2026) | Zero server dependency, offline-first |
| expo-sqlite | op-sqlite | Stack research (Mar 2026) | 8-9x faster batch operations for nutrition queries |
| Raw SQL strings | drizzle-orm typed queries | Stack research (Mar 2026) | Type safety, migration framework |
| USDA API runtime queries | Bundled USDA SQLite database | ADR-005 (Mar 2026) | Offline-first, zero API cost |
| Expo Go development | Expo Dev Client (CNG) | Phase 1 requirement | Required for native modules (op-sqlite, tflite) |
| expo-background-fetch | expo-background-task | Expo SDK 53+ | Deprecated; new API required |
| AsyncStorage for all data | op-sqlite for structured data, AsyncStorage for ephemeral prefs | Phase 1 design | Performance, query capability |

**Deprecated/outdated:**
- `expo-background-fetch`: Deprecated in SDK 53+. Use `expo-background-task` instead.
- `Expo Go`: Cannot run native modules. Use Expo Dev Client.
- `NNAPI` (Android): Deprecated in Android 15. Use LiteRT with vendor NPU delegates.
- `Realm/MongoDB Atlas Device Sync`: EOL September 2025.

## Open Questions

1. **USDA FDC Core Pack Size**
   - What we know: Foundation (29MB uncompressed CSV) + SR Legacy (subset of full download) + FNDDS (1.6GB uncompressed CSV). Converting to normalized SQLite with indexes will be smaller than raw CSV.
   - What's unclear: Actual SQLite file size after conversion. Could be 50MB or 120MB depending on how many FNDDS records are included and index overhead.
   - Recommendation: Build the pipeline early, measure the output. If >100MB, consider splitting FNDDS into a separate on-demand pack or including only macro nutrients (not all 74+ nutrients per food).

2. **expo-play-asset-delivery Fork or Custom Module**
   - What we know: The current library only supports base64 loading. Android Play Core API supports file path access natively.
   - What's unclear: Whether forking expo-play-asset-delivery to add `getAbsoluteAssetPath()` is worthwhile vs. writing a standalone Expo Module.
   - Recommendation: For Phase 1, skip platform-native delivery entirely. Download from R2 on first launch. Revisit in Phase 6.

3. **App Attestation Implementation Complexity**
   - What we know: Play Integrity API (Android) and App Attest (iOS) are documented. Cloudflare Worker can validate tokens.
   - What's unclear: Full implementation effort for attestation in React Native. May need custom native modules for both platforms.
   - Recommendation: For Phase 1, implement R2 downloads without attestation (use a simple API key or rate-limiting). Add attestation in Phase 6 before public release. The data is not secret -- attestation prevents abuse, not unauthorized access.

4. **Custom SQLite Pack Import Validation**
   - What we know: User decision requires custom pack import with schema validation.
   - What's unclear: How to validate arbitrary SQLite files safely (malicious SQL, oversized files, schema violations).
   - Recommendation: Validate schema by checking table/column existence against the published spec. Set a max file size (e.g., 500MB). Open read-only. Do NOT run any SQL from the imported file -- only SELECT queries against known tables.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | jest-expo ^54.0.16 (installed but no test files exist) |
| Config file | None -- needs jest.config.ts in Wave 0 |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern='test_name' --no-coverage` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DAT-01 | User data persists in op-sqlite across restarts | integration | `npx jest --testPathPattern='db/schema' --no-coverage` | No -- Wave 0 |
| DAT-01 | Drizzle migrations run successfully | unit | `npx jest --testPathPattern='db/migrations' --no-coverage` | No -- Wave 0 |
| DAT-01 | Zustand store reads/writes from SQLite | unit | `npx jest --testPathPattern='store/useFoodLogStore' --no-coverage` | No -- Wave 0 |
| DAT-01 | Types refactored: no userId/gcsUrl/APIResponse | unit (type check) | `cd apps/mobile && npx tsc --noEmit` | N/A (TypeScript) |
| DAT-02 | USDA nutrition DB build pipeline produces valid SQLite | unit (Python) | `cd knowledge-graph && python -m pytest tests/test_usda_build.py -x` | No -- Wave 0 |
| DAT-02 | Nutrition DB queried via FTS5 returns results | integration | `npx jest --testPathPattern='services/nutrition' --no-coverage` | No -- Wave 0 |
| DAT-02 | Pack download/cache/integrity verification | unit | `npx jest --testPathPattern='services/packs' --no-coverage` | No -- Wave 0 |
| DAT-03 | Regional DB opened alongside USDA, priority query works | integration | `npx jest --testPathPattern='services/nutrition/regional' --no-coverage` | No -- Wave 0 |
| DAT-03 | Locale-to-region auto-suggestion logic | unit | `npx jest --testPathPattern='services/packs/locale' --no-coverage` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --no-coverage`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage && npx tsc --noEmit`
- **Phase gate:** Full suite green + both platforms compile (`npx expo prebuild --clean && npx expo run:ios && npx expo run:android`)

### Wave 0 Gaps
- [ ] `apps/mobile/jest.config.ts` -- Jest configuration for op-sqlite mocking
- [ ] `apps/mobile/__mocks__/@op-engineering/op-sqlite.ts` -- Mock for unit tests (op-sqlite requires native runtime)
- [ ] `apps/mobile/src/db/__tests__/schema.test.ts` -- Schema definition tests
- [ ] `apps/mobile/src/services/packs/__tests__/packManager.test.ts` -- Pack download/cache tests
- [ ] `apps/mobile/src/services/nutrition/__tests__/nutritionService.test.ts` -- Nutrition query tests
- [ ] Framework install: jest-expo already in devDependencies; needs config file only

## Sources

### Primary (HIGH confidence)
- [Drizzle ORM - OP SQLite](https://orm.drizzle.team/docs/connect-op-sqlite) - Setup, configuration, useMigrations hook
- [Drizzle ORM - Getting Started with OP-SQLite](https://orm.drizzle.team/docs/get-started/op-sqlite-new) - Full setup guide
- [OP-SQLite Configuration](https://op-engineering.github.io/op-sqlite/docs/configuration/) - Database paths, moveAssetsDatabase, location constants
- [USDA FoodData Central Downloads](https://fdc.nal.usda.gov/download-datasets/) - CSV formats, dataset sizes, update dates
- [expo-play-asset-delivery GitHub](https://github.com/one-am-it/expo-play-asset-delivery) - API limitations (base64 only), config format
- [Android Play Asset Delivery (Kotlin/Java)](https://developer.android.com/guide/playcore/asset-delivery/integrate-java) - AssetPackLocation.assetsPath() for file access
- [Cloudflare R2 Presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/) - Signed URL generation for downloads

### Secondary (MEDIUM confidence)
- [AFCD Data Files](https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/data-files) - Excel format, 1,588 foods, 268 nutrients
- [CoFID - GOV.UK](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid) - Excel format, 2021 latest
- [CIQUAL 2025](https://ciqual.anses.fr/cms/en/2025-anses-ciqual-table) - Excel/XML, 3,484 foods, 74 components
- [USDA FDC Data Documentation](https://fdc.nal.usda.gov/data-documentation/) - Dataset type descriptions
- [expo-sqlite Pre-populated DB issue #10881](https://github.com/expo/expo/issues/10881) - Common bundled DB problems

### Tertiary (LOW confidence)
- USDA core pack estimated size (50-80MB) -- based on CSV sizes and compression ratios, not measured
- expo-play-asset-delivery compatibility with current Expo SDK 54 -- last published 2+ years ago
- Custom Expo Module effort estimate for Play Core file path access (~50-100 lines Kotlin) -- based on API surface analysis only

## Metadata

**Confidence breakdown:**
- Standard stack (op-sqlite + drizzle-orm): HIGH - Official docs verified, well-documented integration
- Architecture (separate DBs, Zustand cache, R2 packs): HIGH - Patterns validated by existing knowledge graph implementation and prior research
- USDA pipeline: MEDIUM - CSV format confirmed but output SQLite size is estimated, not measured
- Asset delivery: MEDIUM - expo-play-asset-delivery limitation confirmed but workaround (R2 download) is straightforward
- Pitfalls: HIGH - Multiple sources confirm each pitfall (GitHub issues, official docs, community reports)

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (30 days -- stack is stable, no major releases expected)
