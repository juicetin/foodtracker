# Phase 9: UX Redesign - Research

**Researched:** 2026-03-23
**Domain:** React Native UX restructuring, navigation, bottom sheets, gesture handling, context menus
**Confidence:** HIGH

## Summary

Phase 9 is a full UX overhaul of the Tastimate app. The existing codebase has all the data layer, ML pipeline, and nutrition infrastructure built across phases 1-8. This phase is purely UI/interaction work: restructuring navigation (diary-first home, unified add-food flow), adding long-press context menus, converting entry detail to a bottom sheet, and implementing copy/move meal operations.

The existing stack already includes all critical dependencies: `@react-navigation/bottom-tabs` v7, `@gorhom/bottom-sheet` v5, `react-native-gesture-handler` v2.28, `react-native-reanimated` v4.1, and `expo-camera` v17 (which includes barcode scanning via `CameraView`). The current navigation has a 4-tab bottom bar (Home, Detect FAB, Diary, Profile) that needs to be restructured to (Today/Diary, + Add FAB, Insights, Profile). The current DiaryScreen already has date navigation, swipe gestures, time-period grouping, and sticky macro header. The current HomeScreen (calorie dashboard) gets merged into the new diary-first home.

Key new dependencies needed: `expo-haptics` (haptic feedback for long-press), `expo-clipboard` (copy to clipboard), and `expo-speech` (voice input for food search). No context menu library is needed -- the existing `@gorhom/bottom-sheet` can serve as the context menu surface (more consistent with app patterns than platform-native context menus).

**Primary recommendation:** Restructure MainTabNavigator to make DiaryScreen the home tab, merge HomeScreen macro header into DiaryScreen's StickyMacroHeader, build AddFoodScreen as a new unified entry point replacing the current separate navigation to Detection/FoodSearch/BarcodeScan, and convert EntryDetailScreen into a bottom sheet overlay rather than a stack-pushed screen.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Bottom navigation: Today (Diary), + Add (FAB), Insights, Profile
- "Today" tab is the home screen (diary-first design)
- Center FAB opens Add Food flow
- Diary Screen: Header with remaining calories + macro progress bars (P/C/F), date navigation with swipe/tap arrows, calendar picker on date tap, meal groups (Breakfast, Lunch, Dinner, Snacks) with expand/collapse and long-press menus
- Food items: tap = bottom sheet detail, long press = context menu
- Long press context menu: Copy to clipboard, Copy to another day, Move to other meal, Save as favorite, Delete
- Add Food Screen: search bar with camera/voice/barcode icons, quick access tabs (Recent, Frequent, Favorites, My Recipes), entry methods (Scan Photo, Scan Barcode, Quick Add Macros, From Gallery)
- AI Photo Scan Results: photo thumbnail, identified dishes with per-ingredient macro breakdown, meal selector, Log Meal button
- Barcode scan: camera viewfinder with overlay, match found -> product detail + portion -> log, no match -> text search fallback
- Item Detail Bottom Sheet: food name + star/delete/edit icons, total macros, ingredient list with per-ingredient macros, + Add ingredient, expandable micronutrients/nutrition source/view photo sections
- Meal Group Header Menu: copy from specific date, copy yesterday's meal, save as reusable meal template
- Adherence-neutral design: no shame colors, no red zones

### Claude's Discretion
- Animation/transition choices between screens
- Exact color palette and typography (follow existing theme or Material Design 3)
- Loading/shimmer states during AI processing
- Error states and empty states visual design
- Exact icon choices (use Material Icons or Ionicons -- Ionicons already in use)
- Swipe gestures on diary items (optional enhancement)
- Pull-to-refresh behavior
- Keyboard behavior in search
- Voice input integration details (speech-to-text API choice)

### Deferred Ideas (OUT OF SCOPE)
- Smart recents with ML-based suggestions
- Recipe creation flow (Phase 3.4)
- Gallery scanning improvements (Phase 4)
- Copy/move across multiple items at once (batch operations)
- Meal templates/presets management screen
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UX-01 | Diary-first home screen with remaining cal + macro bars | Merge existing HomeScreen macro card + DiaryScreen into single Today tab |
| UX-02 | Date navigation with swipe, arrows, calendar picker | Existing DiaryScreen swipe gesture + arrows; add calendar picker via date-tap modal |
| UX-03 | Meal group headers (Breakfast/Lunch/Dinner/Snacks) with expand/collapse + long-press menu | Replace time-period grouping (morning/afternoon/evening) with meal-type grouping |
| UX-04 | Food item tap = bottom sheet detail | Convert EntryDetailScreen to @gorhom/bottom-sheet overlay |
| UX-05 | Food item long press = context menu | Build context menu as bottom sheet or action sheet with 5 actions |
| UX-06 | Unified Add Food screen with search + camera + voice + barcode icons | New AddFoodScreen combining current FoodSearchScreen + entry method cards |
| UX-07 | Quick access tabs (Recent, Frequent, Favorites, My Recipes) | Reuse existing historyService + favourites + recipeService queries |
| UX-08 | Barcode scanning integration on Add Food screen | BarcodeScanScreen already exists with expo-camera CameraView |
| UX-09 | Voice input for food description | New: expo-speech for speech-to-text |
| UX-10 | AI Photo Scan results with per-ingredient macros | Existing DetectionScreen -- enhance display with per-ingredient breakdown |
| UX-11 | Copy/move meal operations | New service functions: copyEntryToDate, moveEntryToMeal |
| UX-12 | Item Detail Bottom Sheet with expandable sections | Convert EntryDetailScreen to bottom sheet with collapsible sections |
| QA-01 | Fix long press diary item crash (QA #1) | Replaced by new long-press context menu implementation |
| QA-02 | Fix re-log tap behavior (QA #2) | Tap now opens detail bottom sheet, not re-log |
| QA-03 | Remove third toggle view (QA #3) | Keep two states: summary + ingredients (already two in current ExpandableEntryCard) |
| QA-06 | Barcode option on add food screen (QA #6) | Barcode icon always visible in new AddFoodScreen search bar |
</phase_requirements>

## Standard Stack

### Core (Already Installed)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| @react-navigation/bottom-tabs | ^7.10.1 | Bottom tab navigator | Installed |
| @react-navigation/native-stack | ^7.11.0 | Stack navigator for modals | Installed |
| @gorhom/bottom-sheet | ^5.2.8 | Bottom sheet for item detail + context menus | Installed |
| react-native-gesture-handler | ~2.28.0 | Long press + swipe gestures | Installed |
| react-native-reanimated | ~4.1.1 | Animations for sheets + transitions | Installed |
| expo-camera | ~17.0.10 | Barcode scanning via CameraView | Installed |
| @expo/vector-icons (Ionicons) | ^15.0.3 | Icons throughout UI | Installed |
| zustand | ^5.0.11 | State management | Installed |
| react-native-safe-area-context | ^5.6.2 | Safe area insets | Installed |

### New Dependencies Needed
| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| expo-haptics | ~55.0.9 | Haptic feedback on long-press context menu | Platform-native feel for long-press actions |
| expo-clipboard | ~55.0.9 | Copy meal info to clipboard | Context menu "Copy to clipboard" action |
| expo-speech | ~55.0.9 | Speech-to-text for voice food input | Voice icon in search bar |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| expo-speech | @react-native-voice/voice (v3.2.4) | More powerful but requires native linking; expo-speech is simpler, Expo-managed |
| react-native-context-menu-view | @gorhom/bottom-sheet (already installed) | Native context menus look platform-native but are limited in customization; bottom sheet is consistent with existing app patterns |
| zeego (v3.0.6) | @gorhom/bottom-sheet | zeego provides native context menus on iOS + Android but adds a new dependency; not worth it when bottom-sheet is already in use |
| react-native-calendars | Simple modal date picker | Full calendar library is overkill; a simple month-view picker built with Views suffices for date-tap |

**Installation:**
```bash
npx expo install expo-haptics expo-clipboard expo-speech
```

## Architecture Patterns

### Navigation Restructuring

**Current structure:**
```
RootStack
  MainTabNavigator
    Home (calorie dashboard)
    Detect (FAB -> Detection)
    Diary (date-based food log)
    Profile (settings)
  Detection (modal)
  FoodSearch (modal)
  BarcodeScan (modal)
  EntryDetail (push)
  ...other screens
```

**Target structure:**
```
RootStack
  MainTabNavigator
    Today (merged diary-first home = DiaryScreen + macro header)
    AddFood (FAB -> AddFoodScreen)  -- NEW unified screen
    Insights (trends + charts -- extracted from current DiaryScreen TrendsCard)
    Profile (settings -- unchanged)
  Detection (modal -- keep for AI photo scan)
  BarcodeScan (modal -- keep)
  ...other screens (EntryDetail removed from stack, now bottom sheet)
```

### Key Architectural Decisions

**1. Meal-Type Grouping replaces Time-Period Grouping**

Current DiaryScreen groups entries by time period (morning/afternoon/evening). The new design groups by meal type (breakfast/lunch/dinner/snacks). The `food_entries.meal_type` column already stores this. The `timePeriods.ts` service and `TimePeriodSection` component get replaced by `MealGroupSection`.

**2. Bottom Sheet for Item Detail (not stack screen)**

EntryDetailScreen (currently a stack-pushed screen at ~700 lines) converts to a bottom sheet overlay on the diary. Use `@gorhom/bottom-sheet` with snap points at 50%/90% for compact/expanded views. The bottom sheet shows on tap of any food item, without leaving the diary screen.

**3. Context Menu as Action Sheet (not native context menu)**

Long-press on a food item triggers a compact action sheet via `@gorhom/bottom-sheet` with the 5 actions (copy clipboard, copy to day, move meal, save favorite, delete). This is simpler and more consistent than platform-native context menus.

**4. AddFoodScreen as Hub**

New screen that unifies entry methods: search bar with icons, quick access tabs, and entry method cards. When user taps camera icon -> navigates to Detection. Barcode icon -> navigates to BarcodeScan. Voice icon -> triggers expo-speech. Gallery -> opens image picker. Search results inline.

### Recommended Component Structure
```
src/
  screens/
    DiaryHomeScreen.tsx      # NEW: merged diary-first home (replaces HomeScreen + DiaryScreen)
    AddFoodScreen.tsx         # NEW: unified add food hub
    InsightsScreen.tsx        # NEW: extracted trends from DiaryScreen
    ProfileScreen.tsx         # Unchanged
    DetectionScreen.tsx       # Unchanged (AI photo scan results)
    BarcodeScanScreen.tsx     # Unchanged
    ...
  components/
    diary/
      MealGroupSection.tsx    # NEW: replaces TimePeriodSection (Breakfast/Lunch/Dinner/Snacks)
      MealGroupHeader.tsx     # NEW: tap expand/collapse, long-press three-dot menu
      FoodItemCard.tsx        # NEW: replaces ExpandableEntryCard (tap=detail, longpress=context)
      MacroSummaryHeader.tsx  # RENAME from StickyMacroHeader (add remaining cal display)
      DateNavigator.tsx       # EXTRACT from DiaryScreen (arrows + date label + calendar)
      CalendarPicker.tsx      # NEW: modal calendar for date-tap
      WeekOverviewBar.tsx     # Unchanged
    sheets/
      ItemDetailSheet.tsx     # NEW: bottom sheet version of EntryDetailScreen
      ContextMenuSheet.tsx    # NEW: long-press action sheet
      MealGroupMenuSheet.tsx  # NEW: meal group header long-press menu
    add-food/
      SearchBar.tsx           # NEW: search with camera/voice/barcode icons
      QuickAccessTabs.tsx     # NEW: Recent/Frequent/Favorites/My Recipes tabs
      EntryMethodCards.tsx    # NEW: Scan Photo/Barcode/Quick Add/Gallery cards
  services/
    diary/
      mealGroups.ts           # NEW: meal group constants, icons, labels
      copyMoveService.ts      # NEW: copy entry to date, move entry to meal
    search/
      voiceInputService.ts    # NEW: expo-speech wrapper
```

### Pattern 1: Bottom Sheet Context Menu
**What:** Long-press triggers a bottom sheet with action items instead of a platform-native context menu
**When to use:** Whenever a destructive or multi-step action menu is needed
**Example:**
```typescript
// Context menu triggered by long-press on food item
import BottomSheet, { BottomSheetView } from '@gorhom/bottom-sheet';

function ContextMenuSheet({ entry, visible, onDismiss, onAction }) {
  const snapPoints = useMemo(() => ['35%'], []);

  return (
    <BottomSheet
      ref={sheetRef}
      index={visible ? 0 : -1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onClose={onDismiss}
    >
      <BottomSheetView>
        <Text style={styles.menuTitle}>{entry.dishName}</Text>
        <MenuItem icon="clipboard-outline" label="Copy to clipboard" onPress={() => onAction('copy-clipboard')} />
        <MenuItem icon="calendar-outline" label="Copy to another day" onPress={() => onAction('copy-day')} />
        <MenuItem icon="swap-horizontal-outline" label="Move to other meal" onPress={() => onAction('move-meal')} />
        <MenuItem icon="heart-outline" label="Save as favorite" onPress={() => onAction('favorite')} />
        <MenuItem icon="trash-outline" label="Delete" onPress={() => onAction('delete')} destructive />
      </BottomSheetView>
    </BottomSheet>
  );
}
```

### Pattern 2: Meal Group Section with Expand/Collapse
**What:** Collapsible meal groups replacing time-period sections
**When to use:** Diary screen meal organization
**Example:**
```typescript
const MEAL_GROUPS = ['breakfast', 'lunch', 'dinner', 'snacks'] as const;
type MealGroup = typeof MEAL_GROUPS[number];

function MealGroupSection({ mealGroup, entries, onAddFood, onItemTap, onItemLongPress, onHeaderLongPress }) {
  const [expanded, setExpanded] = useState(true);
  const subtotals = computeSubtotals(entries);

  return (
    <View>
      <Pressable
        onPress={() => setExpanded(!expanded)}
        onLongPress={() => onHeaderLongPress(mealGroup)}
      >
        <MealGroupHeader mealGroup={mealGroup} subtotals={subtotals} expanded={expanded} />
      </Pressable>
      {expanded && entries.map(entry => (
        <FoodItemCard
          key={entry.id}
          entry={entry}
          onPress={() => onItemTap(entry.id)}
          onLongPress={() => onItemLongPress(entry)}
        />
      ))}
      {expanded && (
        <Pressable onPress={() => onAddFood(mealGroup)}>
          <Text>+ Add Food</Text>
        </Pressable>
      )}
    </View>
  );
}
```

### Pattern 3: Copy Entry to Another Day
**What:** Deep-copy a food entry (with dishes + ingredients) to a target date
**When to use:** Context menu "Copy to another day" and meal group "Copy from date"
**Example:**
```typescript
// services/diary/copyMoveService.ts
function copyEntryToDate(sourceEntryId: string, targetDate: string, targetMealType: string): string {
  const newId = generateId();
  const now = new Date().toISOString();

  // Copy food_entry row with new id + target date
  opsqlite.executeSync(
    `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes, created_at)
     SELECT ?, ?, ?, total_calories, total_protein, total_carbs, total_fat,
            'Copied from ' || entry_date || ': ' || COALESCE(notes, ''), ?
     FROM food_entries WHERE id = ?`,
    [newId, targetMealType, targetDate, now, sourceEntryId]
  );

  // Copy scanned_dishes
  // Copy ingredients (with new IDs referencing new entry + dish)
  // Do NOT copy photos (they reference original files)

  return newId;
}

function moveEntryToMeal(entryId: string, newMealType: string): void {
  opsqlite.executeSync(
    `UPDATE food_entries SET meal_type = ?, updated_at = ? WHERE id = ?`,
    [newMealType, new Date().toISOString(), entryId]
  );
}
```

### Anti-Patterns to Avoid
- **Deep nesting of bottom sheets:** Never open a bottom sheet from within another bottom sheet. If the item detail sheet needs to trigger an action, dismiss the detail sheet first, then open the context/action sheet.
- **Gesture conflicts:** The diary swipe gesture (date navigation) must not conflict with bottom sheet drag-to-dismiss. Use `failOffsetY` and `activeOffsetX` thresholds as the current DiaryScreen already does.
- **Direct DB access in components:** Continue the established pattern of service functions in `services/` calling `opsqlite.executeSync()`, not raw SQL in screen components.
- **State duplication:** Don't create a separate Zustand store for the diary home when `useFoodLogStore` and `usePreferencesStore` already cover the data. Use local component state for UI-only state (expanded/collapsed, selected date).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bottom sheets | Custom animated views | @gorhom/bottom-sheet v5 | Already installed, handles keyboard avoidance, gesture conflicts, snap points |
| Haptic feedback | Vibration API | expo-haptics | Cross-platform, intensity control, Expo-managed |
| Clipboard | Custom implementation | expo-clipboard | One-liner API, handles both platforms |
| Barcode scanning | Third-party barcode lib | expo-camera CameraView | Already installed and working in BarcodeScanScreen |
| Date formatting | Manual string manipulation | Existing diaryQueries.ts utilities | `formatDateLabel`, `dateToStr`, `getTodayDateStr` already exist |
| Meal type detection | Complex time-of-day logic | Existing `autoDetectMealType()` | Already in services/detection/types.ts |
| UUID generation | Custom implementation | Existing `generateId()` | Already in useFoodLogStore.ts and favourites.ts |

## Common Pitfalls

### Pitfall 1: Bottom Sheet Gesture Conflicts with Diary Swipe
**What goes wrong:** The horizontal swipe gesture for date navigation intercepts the bottom sheet's vertical drag-to-dismiss gesture.
**Why it happens:** react-native-gesture-handler processes gestures simultaneously by default.
**How to avoid:** The current DiaryScreen already configures `Gesture.Pan().activeOffsetX([-20, 20]).failOffsetY([-10, 10])` which prevents horizontal-only gestures from stealing vertical scroll. For bottom sheets, ensure the BottomSheet is rendered OUTSIDE the GestureDetector wrapper for the swipe gesture.
**Warning signs:** Bottom sheet won't dismiss when dragging down, or date changes unexpectedly when trying to dismiss sheet.

### Pitfall 2: Long Press vs Tap Race Condition
**What goes wrong:** Both onPress and onLongPress fire, or long press triggers tap action.
**Why it happens:** Pressable fires onPress on release after delayLongPress threshold isn't met.
**How to avoid:** Use `delayLongPress={500}` (default is 500ms which is fine). When long-press fires, set a flag to suppress the subsequent onPress. Or use react-native-gesture-handler's `Gesture.LongPress()` which has proper exclusive recognition.
**Warning signs:** Tapping a food item briefly opens both the detail sheet AND the context menu.

### Pitfall 3: Meal Type vs Time Period Grouping Migration
**What goes wrong:** Old entries use time-period grouping (morning/afternoon/evening) while new UI expects meal-type grouping (breakfast/lunch/dinner/snacks).
**Why it happens:** The `food_entries.meal_type` column already exists and stores 'breakfast'/'lunch'/'dinner'/'snack', but the DiaryScreen currently groups by computed time period.
**How to avoid:** Simply query `GROUP BY meal_type` instead of computing time periods. All existing entries already have `meal_type` set. Map 'snack' to 'Snacks' display label.
**Warning signs:** None -- this is a pure UI grouping change, data is already correct.

### Pitfall 4: EntryDetailScreen Complexity in Bottom Sheet
**What goes wrong:** The current EntryDetailScreen is ~700 lines with edit mode, undo/redo, ingredient search, photo viewer, recipe save modal. Putting all this in a bottom sheet makes it unwieldy.
**Why it happens:** Bottom sheets have constrained space and don't support nested modals well.
**How to avoid:** The bottom sheet shows READ-ONLY detail (macros, ingredients, expandable sections). Edit mode remains a full-screen push navigation. The bottom sheet has an "Edit" button that navigates to the full EntryDetailScreen. This keeps the bottom sheet lightweight.
**Warning signs:** Modals opening inside bottom sheets, keyboard avoidance issues.

### Pitfall 5: Calendar Picker Over-Engineering
**What goes wrong:** Adding a full calendar library (react-native-calendars) for a simple date picker.
**Why it happens:** Seems natural but the spec only needs tap-on-date -> pick a date.
**How to avoid:** Use `@react-native-community/datetimepicker` (v9.1.0) or a simple custom modal with month grid. The week overview bar already provides the primary date navigation.
**Warning signs:** Large bundle size increase for a rarely-used feature.

### Pitfall 6: expo-speech on Android Requires Specific Setup
**What goes wrong:** Speech recognition doesn't work or silently fails on Android.
**Why it happens:** expo-speech is for text-to-speech (output), NOT speech-to-text (input). These are different APIs.
**How to avoid:** For speech-to-text input, use `@react-native-voice/voice` (v3.2.4) which wraps Android SpeechRecognizer and iOS SFSpeechRecognizer. expo-speech is the WRONG library for voice input. Alternatively, use `expo-speech-recognition` if available in Expo SDK 54.
**Warning signs:** Can speak text aloud but can't convert user's speech to text.

## Code Examples

### Copy Entry to Another Day (Service Pattern)
```typescript
// services/diary/copyMoveService.ts
import { opsqlite } from '../../../db/client';

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export function copyEntryToDate(
  sourceEntryId: string,
  targetDate: string,
  targetMealType?: string,
): string {
  const newEntryId = generateId();
  const now = new Date().toISOString();

  // 1. Copy food_entry
  const source = opsqlite.executeSync(
    'SELECT * FROM food_entries WHERE id = ?', [sourceEntryId]
  ).rows[0] as Record<string, unknown>;

  opsqlite.executeSync(
    `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [newEntryId, targetMealType ?? source.meal_type, targetDate,
     source.total_calories, source.total_protein, source.total_carbs, source.total_fat,
     source.notes, now]
  );

  // 2. Copy scanned_dishes + ingredients
  const dishes = opsqlite.executeSync(
    'SELECT * FROM scanned_dishes WHERE entry_id = ?', [sourceEntryId]
  ).rows as Array<Record<string, unknown>>;

  for (const dish of dishes) {
    const newDishId = generateId();
    opsqlite.executeSync(
      `INSERT INTO scanned_dishes (id, entry_id, name, cuisine, portion_scale, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [newDishId, newEntryId, dish.name, dish.cuisine, dish.portion_scale, now]
    );

    const ings = opsqlite.executeSync(
      'SELECT * FROM ingredients WHERE dish_id = ?', [dish.id]
    ).rows as Array<Record<string, unknown>>;

    for (const ing of ings) {
      opsqlite.executeSync(
        `INSERT INTO ingredients (id, entry_id, dish_id, name, quantity, unit, calories, protein, carbs, fat, fiber, sugar, amount_g, original_amount_g, database_source, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [generateId(), newEntryId, newDishId, ing.name, ing.quantity, ing.unit,
         ing.calories, ing.protein, ing.carbs, ing.fat, ing.fiber, ing.sugar,
         ing.amount_g, ing.original_amount_g, ing.database_source, now]
      );
    }
  }

  return newEntryId;
}

export function moveEntryToMeal(entryId: string, newMealType: string): void {
  opsqlite.executeSync(
    'UPDATE food_entries SET meal_type = ?, updated_at = ? WHERE id = ?',
    [newMealType, new Date().toISOString(), entryId]
  );
}

export function copyAllEntriesFromDate(
  sourceDate: string,
  targetDate: string,
  filterMealType?: string,
): number {
  let query = 'SELECT id, meal_type FROM food_entries WHERE entry_date = ? AND is_deleted = 0';
  const params: unknown[] = [sourceDate];
  if (filterMealType) {
    query += ' AND meal_type = ?';
    params.push(filterMealType);
  }
  const entries = opsqlite.executeSync(query, params).rows as Array<Record<string, unknown>>;

  for (const entry of entries) {
    copyEntryToDate(entry.id as string, targetDate, entry.meal_type as string);
  }
  return entries.length;
}
```

### Bottom Sheet Item Detail (Pattern)
```typescript
// components/sheets/ItemDetailSheet.tsx
import BottomSheet, { BottomSheetScrollView } from '@gorhom/bottom-sheet';

interface ItemDetailSheetProps {
  entryId: string | null;
  onDismiss: () => void;
  onEdit: (entryId: string) => void;
}

export function ItemDetailSheet({ entryId, onDismiss, onEdit }: ItemDetailSheetProps) {
  const sheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ['50%', '90%'], []);

  // Load entry data when entryId changes
  const entry = useMemo(() => entryId ? loadEntry(entryId) : null, [entryId]);

  return (
    <BottomSheet
      ref={sheetRef}
      index={entryId ? 0 : -1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onClose={onDismiss}
      backdropComponent={BottomSheetBackdrop}
    >
      <BottomSheetScrollView>
        {/* Header: name + action icons */}
        {/* Total macros */}
        {/* Ingredient list with per-ingredient macros */}
        {/* Expandable: micronutrients */}
        {/* Expandable: nutrition source */}
        {/* Expandable: view photo */}
      </BottomSheetScrollView>
    </BottomSheet>
  );
}
```

### Meal Group Queries
```typescript
// services/diary/mealGroups.ts
export const MEAL_GROUPS = ['breakfast', 'lunch', 'dinner', 'snacks'] as const;
export type MealGroup = typeof MEAL_GROUPS[number];

export const MEAL_GROUP_CONFIG: Record<MealGroup, { label: string; icon: string; }> = {
  breakfast: { label: 'Breakfast', icon: 'sunny-outline' },
  lunch:     { label: 'Lunch',     icon: 'partly-sunny-outline' },
  dinner:    { label: 'Dinner',    icon: 'moon-outline' },
  snacks:    { label: 'Snacks',    icon: 'cafe-outline' },
};

// Query entries grouped by meal type (replacing time-period grouping)
export function loadEntriesGroupedByMeal(dateStr: string): Map<MealGroup, DiaryEntry[]> {
  const entries = loadEntriesForDate(dateStr, DEFAULT_BOUNDARIES);
  const grouped = new Map<MealGroup, DiaryEntry[]>();

  for (const group of MEAL_GROUPS) {
    grouped.set(group, []);
  }

  for (const entry of entries) {
    const mealType = entry.mealType as MealGroup;
    const group = MEAL_GROUPS.includes(mealType) ? mealType : 'snacks';
    grouped.get(group)!.push(entry);
  }

  return grouped;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Time-period grouping (morning/afternoon/evening) | Meal-type grouping (breakfast/lunch/dinner/snacks) | Phase 9 | Matches industry standard (MyFitnessPal, Cronometer, Lose It!) |
| Tap diary item = cycle card states | Tap = bottom sheet detail | Phase 9 | Follows Android Material Design tap convention |
| Long press = navigate to detail | Long press = context menu | Phase 9 | Android standard, fixes QA #1 crash |
| HomeScreen + DiaryScreen as separate tabs | Single diary-first home tab | Phase 9 | Industry consensus: diary IS the home screen |
| Separate FoodSearch/BarcodeScan/Detection entry points | Unified AddFoodScreen | Phase 9 | Reduces cognitive load, all methods visible |

**Deprecated/outdated (within this project):**
- `HomeScreen.tsx` -- replaced by diary-first home (content merged into DiaryHomeScreen)
- `TimePeriodSection.tsx` -- replaced by MealGroupSection
- `ExpandableEntryCard.tsx` -- replaced by FoodItemCard (no more cycle states)
- `timePeriods.ts` -- no longer used for diary grouping (may keep for other purposes)
- `SearchBar.tsx` in diary/components -- replaced by date navigator (search moves to AddFoodScreen)

## Open Questions

1. **Voice Input Library Choice**
   - What we know: `expo-speech` is text-to-speech (OUTPUT), not speech-to-text (INPUT). For speech-to-text, need `@react-native-voice/voice` (v3.2.4) or check if Expo SDK 54 has `expo-speech-recognition`.
   - What's unclear: Whether `expo-speech-recognition` exists in SDK 54, or if `@react-native-voice/voice` requires a config plugin for Expo.
   - Recommendation: Research `@react-native-voice/voice` Expo compatibility during implementation. If it requires native setup incompatible with Expo, voice input can be deferred to a simple text input with voice keyboard hint (`textContentType` hints).

2. **Calendar Picker Implementation**
   - What we know: User taps date header to open calendar picker. No calendar library currently installed.
   - What's unclear: Whether to use `@react-native-community/datetimepicker` (platform native) or build a simple custom month grid.
   - Recommendation: Use platform-native date picker via `@react-native-community/datetimepicker` for minimal effort. Only fall back to custom if the native picker UX is unsatisfactory.

3. **Insights Tab Content**
   - What we know: The current DiaryScreen has a TrendsCard (calorie bars, macro averages, streaks). The new "Insights" tab presumably holds this.
   - What's unclear: Whether Insights should be the full existing TrendsCard or something more.
   - Recommendation: For Phase 9, simply extract the TrendsCard as-is into an InsightsScreen. Future enhancement can add more analytics.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: MainTabNavigator.tsx, RootNavigator.tsx, DiaryScreen.tsx, HomeScreen.tsx, EntryDetailScreen.tsx, ExpandableEntryCard.tsx, FoodSearchScreen.tsx, BarcodeScanScreen.tsx, timePeriods.ts, diaryQueries.ts, db/schema.ts, favourites.ts, types/index.ts
- package.json dependency versions verified against installed

### Secondary (MEDIUM confidence)
- npm registry: verified current versions of expo-haptics (55.0.9), expo-clipboard (55.0.9), expo-speech (55.0.9), @react-native-voice/voice (3.2.4), @react-native-community/datetimepicker (9.1.0), zeego (3.0.6), react-native-context-menu-view (1.21.0)

### Tertiary (LOW confidence)
- expo-speech-recognition availability in Expo SDK 54 -- not verified, may not exist

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all core libraries already installed and in use, versions verified
- Architecture: HIGH - clear understanding of current codebase structure and required changes
- Pitfalls: HIGH - identified from direct code inspection (gesture conflicts, bottom sheet nesting, speech API confusion)
- Copy/move operations: HIGH - DB schema fully understood, service pattern established

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (30 days - stable RN + Expo stack)
