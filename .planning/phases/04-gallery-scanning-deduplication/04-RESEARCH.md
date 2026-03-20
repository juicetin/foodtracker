# Phase 04: Gallery Scanning + Deduplication - Research

**Researched:** 2026-03-21
**Domain:** Android gallery access, background task scheduling, image deduplication, on-device ML classification
**Confidence:** HIGH

## Summary

This phase implements automatic discovery of food photos from the device gallery, classification via Gemini Nano, temporal+spatial meal grouping, and photo import into app storage. The existing codebase already has `expo-media-library` (v18.2.1), `expo-background-task`, `expo-task-manager`, `expo-image-manipulator`, and the `geminiNanoService` -- all key dependencies are installed.

**Critical architectural constraint:** Gemini Nano via AICore is **foreground-only**. The ML Kit GenAI API returns `ErrorCode.BACKGROUND_USE_BLOCKED` when the app is not the top foreground activity -- including from foreground services. This means the background WorkManager task can only discover and queue photos; the actual Gemini Nano food/not-food classification MUST happen in the foreground when the user opens the app. The architecture must be a two-phase pipeline: (1) background discovery + EXIF extraction + queueing, (2) foreground classification + meal grouping.

**Primary recommendation:** Implement a background-discover / foreground-classify split architecture. Background task enumerates new gallery photos and writes them to `scanQueue`. When the app is foregrounded, a foreground service drains the queue through Gemini Nano classification, groups food photos into meal events, and imports them into app storage.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use Gemini Nano via AICore for food photo classification (is this a food photo? yes/no)
- NO cloud ML, NO YOLO for detection -- Gemini Nano only per project strategy
- 1 hour default window for grouping photos into same meal (user-configurable)
- GPS proximity: ~100-200m as sensible default, not user-configurable for now
- Import food photos into app storage, downscaled to backup-friendly size
- Don't reference gallery paths (avoids broken refs if user deletes from gallery)

### Claude's Discretion
- Downscale target resolution (e.g. 1024px longest edge, or 512px)
- Background scan chunk size (how many photos per WorkManager invocation)
- EXIF extraction approach (expo-media-library metadata vs custom EXIF reader)
- Gallery scan UI (queue view layout, scan progress indicator)
- How to handle photos without GPS data (skip proximity check, use time only)
- Deduplication hash approach (perceptual hash vs timestamp+GPS only)

### Deferred Ideas (OUT OF SCOPE)
- iOS background scanning (deferred with iOS release)
- Cross-device photo deduplication (post-v1.0)
- Smart album detection (identifying restaurant meals vs home cooking)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GAL-01 | Manual gallery scan to discover and process food photos | expo-media-library getAssetsAsync with pagination + Gemini Nano foreground classification |
| GAL-02 | Background/periodic scanning to surface newly discovered food photos | expo-background-task (WorkManager, 15-min minimum) for discovery; foreground drain for classification |
| GAL-03 | Group multiple photos of same meal (temporal + GPS proximity) | 1-hour time window + Haversine distance ~150m; time-only fallback when GPS missing |
| GAL-04 | Each discovered photo retains EXIF metadata (timestamp, location) | expo-media-library getAssetInfoAsync with ACCESS_MEDIA_LOCATION permission |
| GAL-05 | Background scanning works within platform constraints (chunked processing) | WorkManager 15-min minimum interval; chunked getAssetsAsync with cursor pagination |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| expo-media-library | ^18.2.1 | Gallery access, asset enumeration, EXIF/location metadata | Already in project, covers all gallery access needs |
| expo-background-task | ~1.0.10 | Periodic background scanning via WorkManager | Already in project, used by backupScheduler |
| expo-task-manager | ~14.0.9 | defineTask at module scope for background tasks | Already in project, pattern proven in backupScheduler |
| expo-image-manipulator | ~14.0.8 | Downscale + compress photos before import | Already in project, legacy manipulateAsync API used |
| expo-file-system | ^19.0.21 | Copy downscaled photos to app storage | Already in project, v19 class API (Paths, File, Directory) |
| gemini-nano | local module | Food/not-food classification via AICore | Already in project, geminiNanoService.ts wraps it |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| op-sqlite | ^15.2.5 | Persist scan state, scan_queue, photo_hashes | Already in project for all DB operations |
| zustand | ^5.0.11 | Gallery scan UI state management | Already in project for all store patterns |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| expo-media-library EXIF | react-native-exif or piexifjs | Extra dependency; expo-media-library getAssetInfoAsync already provides location + exif |
| Perceptual hash (phash) | Timestamp + gallery asset ID only | pHash catches re-saved/edited duplicates; timestamp+assetId is simpler but misses cross-app duplicates |
| expo-background-task | react-native-background-worker | expo-background-task already integrated and proven in backup scheduler; no need for another native dependency |

**Installation:** No new packages needed. All dependencies are already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── services/
│   └── gallery/
│       ├── galleryScanService.ts      # Core scan logic (enumerate, filter, queue)
│       ├── galleryScanScheduler.ts     # Background task definition + registration
│       ├── foodClassifier.ts           # Gemini Nano food/not-food classification
│       ├── mealGrouper.ts              # Temporal + GPS proximity clustering
│       ├── photoImporter.ts            # Downscale + copy to app storage
│       └── __tests__/
│           ├── galleryScanService.test.ts
│           ├── mealGrouper.test.ts
│           └── photoImporter.test.ts
├── store/
│   └── useGalleryScanStore.ts          # Scan state, discovered photos queue
└── screens/
    └── GalleryScanScreen.tsx           # Queue view, scan progress, review UI
```

### Pattern 1: Two-Phase Pipeline (Background Discover + Foreground Classify)
**What:** Background task enumerates new photos and writes to scan_queue. Foreground drain classifies via Gemini Nano.
**When to use:** Always -- Gemini Nano is foreground-only.
**Example:**
```typescript
// Phase 1: Background discovery (runs in WorkManager)
// galleryScanScheduler.ts -- side-effect import in App.tsx
TaskManager.defineTask(GALLERY_SCAN_TASK, async () => {
  const lastScannedTimestamp = await getLastScanTimestamp();
  const assets = await MediaLibrary.getAssetsAsync({
    first: 50, // chunk size
    mediaType: 'photo',
    sortBy: ['creationTime', false], // newest first
    createdAfter: lastScannedTimestamp,
  });
  // Write to scan_queue table with status='pending'
  for (const asset of assets.assets) {
    await insertScanQueueItem(asset.id, asset.uri);
  }
  await setLastScanTimestamp(Date.now());
  return BackgroundTask.BackgroundTaskResult.Success;
});

// Phase 2: Foreground classification (runs when app is active)
// Called from useGalleryScanStore or on app foreground event
async function drainScanQueue() {
  const pending = await getPendingScanItems(batchSize);
  for (const item of pending) {
    const result = await geminiNanoService.identify(item.uri);
    if (result.dishes.length > 0) {
      // This is a food photo -- proceed to meal grouping + import
      await processFoodPhoto(item);
    } else {
      await markScanItemDone(item.id, 'not_food');
    }
  }
}
```

### Pattern 2: Cursor-Based Incremental Scanning
**What:** Track the last scanned creation timestamp so each scan only processes new photos.
**When to use:** Every background scan invocation.
**Example:**
```typescript
// Store last scan timestamp in AsyncStorage or SQLite
const LAST_SCAN_KEY = 'gallery_last_scan_timestamp';

async function getNewPhotos(chunkSize: number) {
  const lastTs = await AsyncStorage.getItem(LAST_SCAN_KEY);
  const after = lastTs ? parseInt(lastTs, 10) : Date.now() - 30 * 24 * 60 * 60 * 1000; // default: last 30 days

  const result = await MediaLibrary.getAssetsAsync({
    first: chunkSize,
    mediaType: 'photo',
    sortBy: ['creationTime', true], // oldest first for incremental
    createdAfter: after,
  });
  return result;
}
```

### Pattern 3: Meal Grouping with Temporal + GPS Clustering
**What:** Group food photos into meal events using time proximity + GPS distance.
**When to use:** After food/not-food classification confirms food photos.
**Example:**
```typescript
const MEAL_WINDOW_MS = 60 * 60 * 1000; // 1 hour default (user-configurable)
const GPS_PROXIMITY_M = 150; // ~150 meters

function haversineDistance(
  lat1: number, lon1: number, lat2: number, lon2: number
): number {
  const R = 6371000; // Earth radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2 +
    Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function groupIntoMeals(photos: ClassifiedPhoto[]): MealGroup[] {
  // Sort by creation time
  const sorted = [...photos].sort((a, b) => a.creationTime - b.creationTime);
  const groups: MealGroup[] = [];

  for (const photo of sorted) {
    const matchingGroup = groups.find(g => {
      const timeDiff = photo.creationTime - g.lastTimestamp;
      if (timeDiff > MEAL_WINDOW_MS) return false;
      // If either photo lacks GPS, group by time only
      if (!photo.location || !g.location) return true;
      return haversineDistance(
        photo.location.latitude, photo.location.longitude,
        g.location.latitude, g.location.longitude
      ) <= GPS_PROXIMITY_M;
    });

    if (matchingGroup) {
      matchingGroup.photos.push(photo);
      matchingGroup.lastTimestamp = photo.creationTime;
    } else {
      groups.push({
        photos: [photo],
        lastTimestamp: photo.creationTime,
        location: photo.location,
      });
    }
  }
  return groups;
}
```

### Pattern 4: Photo Import with Downscale
**What:** Copy gallery photo into app storage at reduced resolution.
**When to use:** After classification confirms food photo and meal grouping is complete.
**Example:**
```typescript
// Use legacy manipulateAsync API (project convention per 02-06 decision)
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { Paths, File } from 'expo-file-system/next';

const TARGET_LONGEST_EDGE = 1024; // Good balance: readable detail + small file size

async function importPhoto(galleryUri: string, assetId: string): Promise<string> {
  const info = await MediaLibrary.getAssetInfoAsync(assetId);
  const longestEdge = Math.max(info.width, info.height);

  const actions = longestEdge > TARGET_LONGEST_EDGE
    ? [{ resize: { width: TARGET_LONGEST_EDGE } }] // aspect ratio preserved
    : []; // no resize needed

  const result = await manipulateAsync(galleryUri, actions, {
    compress: 0.8,
    format: SaveFormat.JPEG,
  });

  // Move from cache to persistent app storage
  const destPath = `${Paths.document}/gallery-imports/${generateId()}.jpg`;
  await new File(result.uri).move(new File(destPath));
  return destPath;
}
```

### Anti-Patterns to Avoid
- **Running Gemini Nano in background task:** AICore blocks background inference with BACKGROUND_USE_BLOCKED error. Always classify in foreground.
- **Loading all gallery photos at once:** Use cursor pagination with `first` + `after` to avoid OOM on devices with thousands of photos.
- **Referencing gallery URIs directly:** Gallery URIs break when user deletes photo. Always import/copy to app storage.
- **Using getAssetsAsync with resolveWithFullInfo for bulk enumeration:** Expensive -- only call getAssetInfoAsync individually for confirmed food photos.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gallery photo enumeration | Custom MediaStore query | expo-media-library getAssetsAsync | Handles permissions, pagination, sorting, cross-platform |
| Photo resize/compress | Custom bitmap manipulation | expo-image-manipulator manipulateAsync | Native performance, handles EXIF orientation correctly |
| Background scheduling | Custom AlarmManager/JobScheduler | expo-background-task + expo-task-manager | Already integrated, proven pattern in backupScheduler.ts |
| GPS distance calculation | Google Maps API / external service | Haversine formula (~10 lines JS) | Simple math, no network dependency, accurate enough for 150m threshold |
| EXIF extraction | piexifjs or custom binary parser | expo-media-library getAssetInfoAsync | Returns location + exif natively, needs ACCESS_MEDIA_LOCATION permission |

**Key insight:** Every library needed is already installed. This phase is about wiring existing capabilities together, not adding new native dependencies.

## Common Pitfalls

### Pitfall 1: Gemini Nano Foreground-Only Restriction
**What goes wrong:** Attempting to run Gemini Nano classification inside a WorkManager background task results in BACKGROUND_USE_BLOCKED error.
**Why it happens:** AICore enforces that ML Kit GenAI API inference is only permitted when the app is the top foreground activity.
**How to avoid:** Split into background discovery (scan_queue writes) and foreground classification (drain queue when app active).
**Warning signs:** Empty classification results, GenAiException errors in background task logs.

### Pitfall 2: ACCESS_MEDIA_LOCATION Permission Missing
**What goes wrong:** `getAssetInfoAsync` returns null/undefined for `location` field even on photos with GPS EXIF data.
**Why it happens:** Android requires `ACCESS_MEDIA_LOCATION` permission (in addition to `READ_MEDIA_IMAGES`) to read GPS coordinates from photo EXIF. This permission is NOT in the current app.json.
**How to avoid:** Add `"android.permission.ACCESS_MEDIA_LOCATION"` to the android.permissions array in app.json. Also configure the expo-media-library plugin if needed.
**Warning signs:** Location always null on Android despite photos having GPS data in other apps.

### Pitfall 3: getAssetsAsync sortBy Crashes on Some Android Devices
**What goes wrong:** `sortBy: 'creationTime'` can crash on some Android devices where certain photos have corrupt or missing EXIF creation timestamps.
**Why it happens:** Android MediaStore falls back to file modification time when EXIF creation time is missing, but the cursor sorting can still fail on edge cases.
**How to avoid:** Wrap in try/catch; fallback to `sortBy: 'modificationTime'` if creationTime fails. Use `createdAfter` filter to limit result set size.
**Warning signs:** "Could not read file or parse EXIF tags" error messages.

### Pitfall 4: OOM from Large Gallery Enumeration
**What goes wrong:** Calling getAssetsAsync with large `first` value or without proper pagination exhausts memory.
**Why it happens:** Each asset object includes URI and metadata; thousands of assets at once causes heap pressure.
**How to avoid:** Use small chunk sizes (50-100 per page) with cursor-based pagination via `after` parameter. Process chunks sequentially.
**Warning signs:** App crashes during gallery scan on devices with 10,000+ photos.

### Pitfall 5: Duplicate Scan Queue Entries
**What goes wrong:** Same gallery photo gets added to scan_queue multiple times across different background task invocations.
**Why it happens:** Timestamp-based incremental scanning can have race conditions if the last-scanned timestamp is not atomically updated.
**How to avoid:** Use gallery asset ID as unique key in scan_queue table (UNIQUE constraint on assetId column). Use INSERT OR IGNORE.
**Warning signs:** Same photo appearing multiple times in classification queue.

### Pitfall 6: Stale Gallery URIs After Import
**What goes wrong:** Imported photo references break if expo-image-manipulator cache is cleared.
**Why it happens:** manipulateAsync saves to cache directory by default, which can be cleaned by OS.
**How to avoid:** Always move/copy the result from cache to persistent app document storage immediately after manipulation.
**Warning signs:** Photos showing as broken images after app restart or OS storage cleanup.

## Code Examples

### Gallery Permission Request
```typescript
import * as MediaLibrary from 'expo-media-library';

async function requestGalleryAccess(): Promise<boolean> {
  const { status } = await MediaLibrary.requestPermissionsAsync();
  return status === 'granted';
}
```

### Simplified Food/Not-Food Classification Prompt
```typescript
// Simpler prompt than full food identification -- just yes/no classification
const FOOD_DETECTION_PROMPT =
  'Is this a photo of food? Reply with exactly one word: "yes" or "no".';

// For batch classification, use the existing FOOD_PROMPT from geminiNanoService
// which returns { dishes: [] } -- empty dishes array means "not food"
async function isFood(photoUri: string): Promise<boolean> {
  const result = await geminiNanoService.identify(photoUri);
  return result.dishes.length > 0;
}
```

### Scan Queue Schema (already exists in schema.ts)
```typescript
// Existing scan_queue table in db/schema.ts
export const scanQueue = sqliteTable('scan_queue', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  assetId: text('asset_id'),  // Gallery asset ID for dedup
  uri: text('uri').notNull(),
  status: text('status').notNull().default('pending'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
  processedAt: text('processed_at'),
});
```

**Note:** The existing schema needs extension. Add columns for: `creationTime` (photo creation timestamp from EXIF), `latitude`/`longitude` (GPS from EXIF), `isFood` (classification result), `mealGroupId` (assigned meal group). Also add UNIQUE constraint on `assetId` to prevent duplicate entries.

### Photo Hashes Schema (already exists in schema.ts)
```typescript
// Existing photo_hashes table -- useful for perceptual hash dedup
export const photoHashes = sqliteTable('photo_hashes', {
  photoId: text('photo_id').primaryKey()
    .references(() => photos.id, { onDelete: 'cascade' }),
  phash: text('phash').notNull(),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| expo-background-fetch | expo-background-task (WorkManager) | Expo SDK 53 (2025) | More reliable scheduling, proper WorkManager integration |
| Manual EXIF parsing (piexifjs) | expo-media-library getAssetInfoAsync | Always available | Native EXIF extraction, no JS parsing overhead |
| Full VLM for classification | Gemini Nano zero-shot classification | Phase 02.7 | System-managed model, no download, fast inference |

**Deprecated/outdated:**
- `expo-background-fetch`: Replaced by `expo-background-task` in Expo SDK 53. The project already uses the new API.
- `MediaLibrary.getAssetsAsync` without `resolveWithFullInfo`: On Android, image orientation may be incorrect without this flag. Use it only when full metadata is needed (classification phase, not discovery phase).

## Open Questions

1. **Downscale target resolution**
   - What we know: Gemini Nano native module already scales to 512px max edge for inference. Gallery photos for display/review benefit from higher resolution.
   - What's unclear: Whether 1024px or 768px is the right balance for storage vs readability.
   - Recommendation: Use 1024px longest edge for imported photos. Gemini Nano's own scaler handles inference input; the stored photo is for user review.

2. **Perceptual hash vs timestamp+assetId deduplication**
   - What we know: Schema already has `photoHashes` table with `phash` column. pHash is ~10 lines of JS (resize to 8x8, DCT, median threshold).
   - What's unclear: Whether cross-app duplicates are common enough to justify pHash computation cost.
   - Recommendation: Use `assetId` as primary dedup key (INSERT OR IGNORE). Defer perceptual hashing to a future iteration unless testing reveals duplicate issues. The photoHashes table is already there if needed.

3. **Gemini Nano rate limiting / BUSY errors**
   - What we know: AICore enforces inference quota per app. Too many rapid requests return ErrorCode.BUSY. STATE.md notes thermal throttling at ~2.5 min sustained inference.
   - What's unclear: Exact quota limits, optimal batch pacing.
   - Recommendation: Process foreground queue with 500ms-1s delay between classifications. Implement exponential backoff on BUSY errors. Show progress indicator to user.

4. **First-time scan scope**
   - What we know: User's gallery may have thousands of photos spanning years.
   - What's unclear: How far back to scan on first use.
   - Recommendation: Default to last 30 days on first scan. Provide option to extend. Each subsequent scan is incremental from last timestamp.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | jest-expo (Jest) |
| Config file | apps/mobile/jest.config.js |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern gallery --no-coverage -x` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GAL-01 | Manual scan enumerates gallery + classifies food photos | unit | `npx jest src/services/gallery/__tests__/galleryScanService.test.ts -x` | Wave 0 |
| GAL-02 | Background task schedules and runs discovery | unit | `npx jest src/services/gallery/__tests__/galleryScanScheduler.test.ts -x` | Wave 0 |
| GAL-03 | Meal grouping clusters by time + GPS | unit | `npx jest src/services/gallery/__tests__/mealGrouper.test.ts -x` | Wave 0 |
| GAL-04 | EXIF metadata extracted and stored | unit | `npx jest src/services/gallery/__tests__/galleryScanService.test.ts -x` | Wave 0 |
| GAL-05 | Chunked processing with cursor pagination | unit | `npx jest src/services/gallery/__tests__/galleryScanService.test.ts -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern gallery --no-coverage -x`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/services/gallery/__tests__/galleryScanService.test.ts` -- covers GAL-01, GAL-04, GAL-05
- [ ] `src/services/gallery/__tests__/galleryScanScheduler.test.ts` -- covers GAL-02
- [ ] `src/services/gallery/__tests__/mealGrouper.test.ts` -- covers GAL-03
- [ ] `src/services/gallery/__tests__/photoImporter.test.ts` -- covers photo downscale + import
- [ ] `__mocks__/expo-media-library.ts` -- mock for getAssetsAsync, getAssetInfoAsync
- [ ] Jest moduleNameMapper for `expo-media-library` if not already mocked

## Sources

### Primary (HIGH confidence)
- [expo-media-library docs](https://docs.expo.dev/versions/latest/sdk/media-library/) - Asset type, getAssetsAsync pagination, getAssetInfoAsync EXIF/location, ACCESS_MEDIA_LOCATION permission
- [expo-background-task docs](https://docs.expo.dev/versions/latest/sdk/background-task/) - registerTaskAsync options, 15-min minimum interval on Android (WorkManager), BackgroundTaskResult
- Existing codebase: `backupScheduler.ts` pattern, `geminiNanoService.ts` API, `db/schema.ts` (scanQueue + photoHashes tables)

### Secondary (MEDIUM confidence)
- [ML Kit GenAI overview](https://developers.google.com/ml-kit/genai) - Foreground-only restriction confirmed: "GenAI API inference is permitted only when the app is the top foreground application"
- [Android Gemini Nano docs](https://developer.android.com/ai/gemini-nano) - ErrorCode.BACKGROUND_USE_BLOCKED, inference quota per app, BUSY error handling
- [expo-image-manipulator docs](https://docs.expo.dev/versions/latest/sdk/imagemanipulator/) - manipulateAsync resize + compress API

### Tertiary (LOW confidence)
- [GitHub expo issue #13123](https://github.com/expo/expo/issues/13123) - Android API 30+ EXIF location access requires explicit ACCESS_MEDIA_LOCATION permission (needs verification on current Expo SDK 54)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already installed and verified in codebase
- Architecture: HIGH - foreground-only constraint is well-documented by Google; two-phase pipeline is the clear solution
- Pitfalls: HIGH - multiple sources confirm ACCESS_MEDIA_LOCATION requirement and background inference blocking
- Meal grouping: MEDIUM - Haversine + time window is straightforward math, but edge cases (timezone changes, travel) need testing

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable -- all libraries are already integrated; Gemini Nano foreground restriction unlikely to change soon)
