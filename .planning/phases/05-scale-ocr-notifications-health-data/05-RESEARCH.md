# Phase 05: Scale OCR + Notifications + Health Data - Research

**Researched:** 2026-03-21
**Domain:** On-device OCR, local notifications, health platform integration
**Confidence:** MEDIUM

## Summary

Phase 05 introduces three distinct features: kitchen scale weight reading via Gemini Nano OCR, configurable daily macro push notifications, and weight trend tracking from Google Health Connect. The phase also addresses DET-04 (hidden ingredients via KG lookup), which is largely already supported by the existing KnowledgeGraphService's recipe decomposition -- the gap is surfacing ingredient lists to the user in detection results UI.

The notification and health data features use well-established Expo/React Native libraries with straightforward integration patterns. Scale OCR is the highest-risk feature -- Gemini Nano's ability to reliably read 7-segment displays on kitchen scales is unproven and needs a spike. The CONTEXT.md correctly identifies this: try Gemini Nano text extraction first, with manual weight input as fallback.

**Primary recommendation:** Spike Gemini Nano scale OCR early (it may fail on 7-segment displays). Notifications and Health Connect are standard integrations -- plan them as straightforward library wiring.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Manual weight input as fallback when OCR fails or no scale detected
- Container tare management: weigh empty container once, app remembers; also allow manual tare input
- Notifications set during first-time app onboarding
- Default notification time: 9pm
- User-configurable time or can disable entirely
- Health data off by default -- opt-in to auto-sync
- Google Health Connect on Android (Apple Health deferred with iOS release)
- Gemini Nano is primary and only inference engine throughout

### Claude's Discretion
- 7-segment OCR approach (Gemini Nano text extraction vs custom TFLite model vs ML Kit text recognition)
- Container tare UI (list view, recent containers, auto-detect from photo)
- Weight trend smoothing algorithm (EMA, Kalman filter, or simple moving average)
- Notification content format and styling
- Health Connect permission flow and data display

### Deferred Ideas (OUT OF SCOPE)
- Apple Health integration (deferred with iOS release)
- Multiple daily notification times
- Scale brand/model auto-detection
- Nutrition goal adjustment based on weight trend
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DET-04 | User sees inferred hidden ingredients from dish identification via knowledge graph lookup | KG's `getRecipeIngredients()` already returns ingredient lists for canonical recipes; need to surface these in detection results UI |
| SCL-01 | Kitchen scale weight reading via 7-segment OCR | Gemini Nano text extraction as primary approach; ML Kit Text Recognition v2 as fallback; manual input as ultimate fallback |
| SCL-02 | Container/vessel tare weight management | `containerWeights` table already exists in DB schema; need CRUD service + UI |
| SCL-03 | App learns frequently used container weights over time | `timesUsed` and `lastUsedAt` columns already in schema; sort by usage frequency |
| NTF-01 | Configurable end-of-day push notification with daily macro summary | expo-notifications ~0.32 with DailyTriggerInput; getTodayTotals() from useFoodLogStore |
| NTF-02 | Import weight data from Google Health Connect and view smoothed weight trend | react-native-health-connect 3.5.0 + expo-health-connect config plugin; EMA smoothing |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| expo-notifications | ~0.32.16 | Local scheduled notifications | Official Expo notification library; SDK 54 compatible |
| react-native-health-connect | 3.5.0 | Google Health Connect API | Only maintained RN Health Connect library; 3.5.0 published 2025-11-22 |
| expo-health-connect | 0.1.1 | Expo config plugin for health-connect | Auto-configures native setup via expo prebuild |
| gemini-nano (local module) | file:./modules/gemini-nano | Scale OCR text extraction | Already integrated; Gemini Nano-only strategy |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| expo-build-properties | (already installed or add) | Set compileSdkVersion for Health Connect | Required for Health Connect native setup |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Gemini Nano for scale OCR | ML Kit Text Recognition v2 | ML Kit is proven for general OCR but may not handle 7-segment well either; adds native dependency. Try Gemini Nano first per strategy. |
| Gemini Nano for scale OCR | Custom TFLite 7-segment model | High accuracy but requires training data collection and model training; defer unless Nano + ML Kit both fail. |
| EMA for weight smoothing | Kalman filter | Kalman is more sophisticated but overkill for simple weight trends; EMA with alpha=0.1-0.2 matches fitness app standards (Happy Scale, MacroFactor). |

**Installation:**
```bash
npx expo install expo-notifications react-native-health-connect expo-health-connect
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  services/
    scale/
      scaleOcrService.ts      # Gemini Nano text extraction for scale reading
      containerService.ts      # Container tare CRUD (uses containerWeights table)
    notifications/
      notificationService.ts   # Schedule/cancel daily macro notification
      notificationScheduler.ts # Module-scope setup (like galleryScanScheduler)
    health/
      healthConnectService.ts  # Google Health Connect weight data import
      weightTrendService.ts    # EMA smoothing and trend calculation
  screens/
    ScaleInputScreen.tsx       # Manual weight entry + OCR result display
    NotificationSettingsScreen.tsx  # Or section in ProfileScreen
    WeightTrendScreen.tsx      # Chart with raw + smoothed weight line
  store/
    useWeightStore.ts          # Weight entries (local SQLite + Health Connect sync)
```

### Pattern 1: Gemini Nano Scale OCR
**What:** Use existing geminiNanoService with a scale-specific prompt to extract weight text from photos
**When to use:** When a kitchen scale is visible in a food photo
**Example:**
```typescript
// Source: project pattern from geminiNanoService.ts
export const SCALE_OCR_PROMPT =
  'Look at this image. If there is a kitchen scale or digital display showing a weight, ' +
  'extract the number shown on the display. Return JSON: { "weight_g": number | null, "unit": "g" | "kg" | "oz" | "lb" | null }. ' +
  'Return null values if no scale/display is visible.';

export async function readScaleWeight(imageBase64: string): Promise<{ weightG: number; unit: string } | null> {
  const result = await geminiNanoModule.executePromptWithImage(SCALE_OCR_PROMPT, imageBase64);
  // Parse JSON, convert to grams, return
}
```

### Pattern 2: Daily Notification Scheduling
**What:** Schedule a repeating daily notification at a user-configured time
**When to use:** During onboarding or when user changes notification time
**Example:**
```typescript
// Source: Expo docs https://docs.expo.dev/versions/latest/sdk/notifications/
import * as Notifications from 'expo-notifications';

async function scheduleDailyMacroNotification(hour: number, minute: number): Promise<string> {
  // Cancel any existing daily notification first
  await Notifications.cancelAllScheduledNotificationsAsync();

  return Notifications.scheduleNotificationAsync({
    content: {
      title: 'Daily Nutrition Summary',
      body: '', // Populated at trigger time via notification handler
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.DAILY,
      hour,
      minute,
    },
  });
}
```

### Pattern 3: Health Connect Weight Read
**What:** Read weight records from Google Health Connect with permission flow
**When to use:** When user opts into health data sync
**Example:**
```typescript
// Source: react-native-health-connect docs
import { initialize, readRecords, requestPermission } from 'react-native-health-connect';

async function readWeightRecords(startDate: Date, endDate: Date) {
  await initialize();
  await requestPermission([{ accessType: 'read', recordType: 'Weight' }]);

  const result = await readRecords('Weight', {
    timeRangeFilter: {
      operator: 'between',
      startTime: startDate.toISOString(),
      endTime: endDate.toISOString(),
    },
  });
  return result.records; // Array of { time, weight: { inKilograms } }
}
```

### Pattern 4: EMA Weight Smoothing
**What:** Exponential Moving Average for weight trend line
**When to use:** Display smoothed weight trend alongside raw data points
**Example:**
```typescript
// Source: standard EMA algorithm, used by Happy Scale / MacroFactor
function emaSmooth(weights: { date: string; kg: number }[], alpha: number = 0.15): number[] {
  if (weights.length === 0) return [];
  const smoothed = [weights[0].kg];
  for (let i = 1; i < weights.length; i++) {
    smoothed.push(alpha * weights[i].kg + (1 - alpha) * smoothed[i - 1]);
  }
  return smoothed;
}
```

### Anti-Patterns to Avoid
- **Blocking notification permission at app launch:** Request during onboarding flow when context is clear, not on first app open
- **Polling Health Connect:** Read on-demand when user navigates to weight trend screen, not on a background timer
- **Trusting OCR blindly:** Always show the OCR result to the user for confirmation; never auto-apply a scale reading without user review
- **Storing notification content at schedule time:** The daily macro totals change throughout the day; use a notification handler to fetch current totals when the notification fires

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Notification scheduling | Custom AlarmManager/UNNotificationRequest | expo-notifications DailyTriggerInput | Handles platform differences, permission flow, Doze mode |
| Health data access | Custom Health Connect bindings | react-native-health-connect | Manages permission delegation, data types, lifecycle |
| Text recognition | Custom TFLite OCR model (initially) | Gemini Nano prompt-based extraction | Already integrated; zero additional native deps; spike first |
| Container weight persistence | In-memory state | containerWeights SQLite table (already in schema) | Already defined with timesUsed/lastUsedAt for learning |

**Key insight:** The containerWeights table already exists in the Drizzle schema with `timesUsed` and `lastUsedAt` columns -- SCL-02 and SCL-03 are mostly UI + service wiring, not schema work.

## Common Pitfalls

### Pitfall 1: Notification Permission Timing
**What goes wrong:** App requests notification permission on first launch with no context; user denies; cannot re-request
**Why it happens:** Android 13+ (API 33) requires POST_NOTIFICATIONS runtime permission; once denied, must go to Settings
**How to avoid:** Request during onboarding step that explains "Get a daily nutrition summary at 9pm"; show value before asking
**Warning signs:** Low notification opt-in rate

### Pitfall 2: Gemini Nano Foreground-Only for OCR
**What goes wrong:** Scale OCR called during background processing (gallery scan) and fails silently
**Why it happens:** Gemini Nano (AICore) requires foreground context per Phase 04 learnings
**How to avoid:** Scale OCR only triggered from active detection flow (user is looking at photo), never from background tasks
**Warning signs:** null returns from geminiNanoModule in background context

### Pitfall 3: Health Connect Not Installed
**What goes wrong:** App crashes or shows confusing error when Health Connect app is not installed on device
**Why it happens:** Health Connect is a separate app on Android < 14; bundled from Android 14+. Older devices need to install it from Play Store
**How to avoid:** Check `getSdkStatus()` or wrap in try/catch; show "Install Health Connect" prompt linking to Play Store for Android < 14
**Warning signs:** Crash on `initialize()` for Android 12-13 devices without Health Connect app

### Pitfall 4: 7-Segment Display OCR Unreliability
**What goes wrong:** Gemini Nano returns incorrect weight or fails to recognize 7-segment digits
**Why it happens:** 7-segment displays have character breaks that confuse standard OCR; LED reflections, viewing angles, partial occlusion
**How to avoid:** Always show manual input as primary; treat OCR as "helpful suggestion" not ground truth; validate parsed number is reasonable (0.1g - 10kg range)
**Warning signs:** Nonsense numbers, null results, confident but wrong readings

### Pitfall 5: Notification Content Stale at Trigger Time
**What goes wrong:** Notification shows "0 calories" because content was set at schedule time, not trigger time
**Why it happens:** expo-notifications trigger content is set when scheduled, not when fired
**How to avoid:** Use `setNotificationHandler` to intercept and dynamically build notification content with current day's totals when it fires
**Warning signs:** Always showing the same notification content regardless of actual intake

### Pitfall 6: expo-notifications on Android 12+ (SCHEDULE_EXACT_ALARM)
**What goes wrong:** Scheduled notifications don't fire reliably when device is in Doze mode
**Why it happens:** Android 12+ requires SCHEDULE_EXACT_ALARM permission; without it, inexact alarms may drift
**How to avoid:** For a daily 9pm notification, inexact timing is acceptable (within ~15min window); if exact timing is needed, add the permission. Daily macro summary does not need exact timing.
**Warning signs:** Notifications arriving late or not at all on some Android devices

## Code Examples

### Existing Assets to Reuse

**Container weights table (already in schema):**
```typescript
// Source: db/schema.ts line 216-223
export const containerWeights = sqliteTable('container_weights', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  weightGrams: real('weight_grams').notNull(),
  timesUsed: integer('times_used').default(0),
  lastUsedAt: text('last_used_at'),
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});
```

**getTodayTotals() for notification content:**
```typescript
// Source: store/useFoodLogStore.ts line 50-55
getTodayTotals: () => {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
};
```

**Gemini Nano image+prompt pattern:**
```typescript
// Source: services/vlm/geminiNanoService.ts
// The existing identifyFood method shows the pattern:
// geminiNanoModule.executePromptWithImage(prompt, base64Image)
// Parse JSON response, handle errors
```

### DET-04: Hidden Ingredients from KG
```typescript
// Source: services/knowledge-graph/knowledgeGraphService.ts
// KG already returns recipe ingredients via getCanonicalRecipe() + getRecipeIngredients()
// DET-04 requires: after dish identification, call KG to get ingredients list
// and display them in the detection results UI (e.g., "carbonara" -> egg, pancetta, parmesan)

// The ingredient data is already available; the gap is UI presentation:
// 1. After scanFood() identifies dishes, call KG.searchDish() + getCanonicalRecipe()
// 2. Extract ingredient names from the recipe
// 3. Display as "Contains: egg, pancetta, parmesan, black pepper" under the dish name
```

### Notification Handler for Dynamic Content
```typescript
// Source: Expo docs pattern
import * as Notifications from 'expo-notifications';

// Set handler at app startup (in App.tsx or notification module)
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});

// To dynamically update content, use a foreground listener approach:
// Schedule with placeholder content, then on notification received,
// update with actual totals if app is in foreground.
// For background: content must be set at schedule time.
// PRACTICAL APPROACH: Re-schedule the notification daily with fresh content
// each time the user opens the app or logs a meal after the notification time.
```

### Weight Trend New DB Table
```typescript
// New table needed for weight_entries (separate from Health Connect)
export const weightEntries = sqliteTable('weight_entries', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  date: text('date').notNull().unique(), // YYYY-MM-DD
  weightKg: real('weight_kg').notNull(),
  source: text('source').notNull(), // 'manual' | 'health_connect'
  healthConnectId: text('health_connect_id'), // ID from Health Connect for dedup
  createdAt: text('created_at').default(sql`(datetime('now'))`),
});
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Google Fit SDK | Google Health Connect | 2024+ (Fit sunset 2026) | Must use Health Connect, not Fit |
| expo-notifications push only | Local + push with DailyTriggerInput | SDK 52+ | Simplified daily scheduling |
| Custom native notification modules | expo-notifications managed | Stable since SDK 49+ | No custom native code needed |
| Google Fit deprecated | react-native-health-connect 3.x | 2025 | Ecosystem moved to Health Connect |

**Deprecated/outdated:**
- Google Fit SDK: Sunsetting in 2026; use Health Connect exclusively
- expo-notifications in Expo Go: Push notifications no longer work in Expo Go from SDK 54; requires development build (local notifications still work)

## Open Questions

1. **Gemini Nano 7-segment OCR capability**
   - What we know: Gemini (cloud) models are good at general OCR; Nano on-device capability for 7-segment displays is unproven
   - What's unclear: Whether Nano can reliably parse LED/LCD 7-segment digits given its limited model size
   - Recommendation: Plan a spike task as the first scale OCR work. If Nano fails, try ML Kit Text Recognition v2 as intermediate step before considering custom TFLite model

2. **Notification content freshness**
   - What we know: expo-notifications sets content at schedule time; DailyTriggerInput fires at the set hour daily
   - What's unclear: Whether `setNotificationHandler` can intercept and modify content before display on all Android versions
   - Recommendation: Re-schedule the notification with fresh macro totals each time the app is foregrounded (simple, reliable). This also naturally handles the case where the user changes notification time.

3. **Health Connect availability on older Android**
   - What we know: Health Connect is bundled from Android 14+; separate app install required for Android 12-13; project minSdk is 26
   - What's unclear: What percentage of target users have Android < 14 and would need to install Health Connect separately
   - Recommendation: Gate behind `getSdkStatus()` check; show install prompt for Android < 14; feature is opt-in so graceful degradation is fine

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Jest via jest-expo (already configured) |
| Config file | `apps/mobile/jest.config.js` |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern='scale\|notification\|health\|weight' --no-coverage -x` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DET-04 | Hidden ingredients surfaced from KG for identified dishes | unit | `cd apps/mobile && npx jest --testPathPattern='hiddenIngredients' -x` | No - Wave 0 |
| SCL-01 | Scale OCR extracts weight from photo via Gemini Nano | unit | `cd apps/mobile && npx jest --testPathPattern='scaleOcr' -x` | No - Wave 0 |
| SCL-02 | Container tare CRUD and auto-subtraction | unit | `cd apps/mobile && npx jest --testPathPattern='container' -x` | No - Wave 0 |
| SCL-03 | Container usage frequency tracking and sorting | unit | `cd apps/mobile && npx jest --testPathPattern='container' -x` | No - Wave 0 |
| NTF-01 | Daily macro notification scheduling at configured time | unit | `cd apps/mobile && npx jest --testPathPattern='notification' -x` | No - Wave 0 |
| NTF-02 | Weight data import from Health Connect + EMA smoothing | unit | `cd apps/mobile && npx jest --testPathPattern='weight\|health' -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern='scale\|notification\|health\|weight' --no-coverage -x`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/services/scale/__tests__/scaleOcrService.test.ts` -- covers SCL-01 (mock geminiNanoModule)
- [ ] `src/services/scale/__tests__/containerService.test.ts` -- covers SCL-02, SCL-03
- [ ] `src/services/notifications/__tests__/notificationService.test.ts` -- covers NTF-01
- [ ] `src/services/health/__tests__/healthConnectService.test.ts` -- covers NTF-02 (Health Connect read)
- [ ] `src/services/health/__tests__/weightTrendService.test.ts` -- covers NTF-02 (EMA smoothing)
- [ ] Mock for expo-notifications: `__mocks__/expo-notifications.ts`
- [ ] Mock for react-native-health-connect: `__mocks__/react-native-health-connect.ts`

## Sources

### Primary (HIGH confidence)
- [Expo notifications docs](https://docs.expo.dev/versions/latest/sdk/notifications/) -- DailyTriggerInput API, permissions, scheduling
- [react-native-health-connect docs](https://matinzd.github.io/react-native-health-connect/docs/get-started/) -- setup, permissions, readRecords API
- Project codebase -- geminiNanoService.ts, containerWeights schema, useFoodLogStore.getTodayTotals(), knowledgeGraphService.ts

### Secondary (MEDIUM confidence)
- [Expo SDK 54 changelog](https://expo.dev/changelog/sdk-54) -- SDK compatibility
- npm registry -- version verification (expo-notifications 0.32.16, react-native-health-connect 3.5.0, expo-health-connect 0.1.1)
- [Happy Scale support](https://happyscale.com/support) -- EMA smoothing approach in fitness apps

### Tertiary (LOW confidence)
- Gemini Nano 7-segment OCR capability -- no direct evidence of success; needs spike validation
- expo-health-connect 0.1.1 maturity -- published 2024-07-31, very early version (0.1.1); may need manual native config as backup

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - well-established Expo/RN libraries, verified versions
- Architecture: HIGH - follows existing project patterns (services, stores, raw SQL)
- Scale OCR: LOW - Gemini Nano 7-segment capability is unproven; approach is hypothesis not fact
- Notifications: HIGH - straightforward expo-notifications DailyTriggerInput
- Health Connect: MEDIUM - library is established but expo config plugin is very early (0.1.1)
- Weight smoothing: HIGH - standard EMA algorithm, well documented
- Pitfalls: MEDIUM - based on known platform constraints and project history

**Research date:** 2026-03-21
**Valid until:** 2026-04-20 (30 days -- stable domain, libraries unlikely to change)
