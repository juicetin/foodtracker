# Competitor Food & Recipe Management UX Research

> Date: 2026-03-19
> Focus: Core food/recipe management features (non-AI) across top calorie tracking apps
> Purpose: Inform feature planning for Tastimate

---

## 1. MyFitnessPal

### Food Logging
- **Database**: 18M+ foods (largest in market). User-contributed, so accuracy varies.
- **Search**: Text search with autocomplete. Tabs for Recent, Frequent, My Foods, Meals, Recipes. Summer 2025 redesign put all logging options (search, barcode, voice) in one unified interface.
- **Custom foods**: Full custom food creation with all macros. Can link custom food to a barcode. Community-contributed foods become available to all users.
- **Barcode scanning**: Premium-only. Scans product barcode, pulls exact nutrition label data. If barcode not found, prompts user to create the food.
- **Quick-add calories**: Yes. Enter raw calorie number without specifying a food. Useful for rough tracking.
- **Copy/paste meals**: "Quick Tools" under each meal slot. Can copy entire meals to/from other dates. "Add Yesterday's Meal" swipe shortcut on new day.
- **Favorites/frequent**: "Remembered Meals" feature saves groups of foods (e.g., "Two eggs and cereal"). Frequent and Recent tabs in search. Meals can be saved and re-logged as a group.

### Recipe Management
- **Create recipes**: Enter name, number of servings, then add ingredients by searching the database.
- **Import from URL**: Yes, Recipe Importer on web and mobile. Parses recipe websites, auto-matches ingredients to database. Requires manual review of matches.
- **Serving sizes**: Set total servings when creating recipe. Each serving gets proportional nutrition.
- **Saved meals**: Meals (groups of foods) are separate from recipes. Meals are a quick grouping; recipes have servings and can be portioned.

### Diary/History
- **Day view**: Four default meal slots: Breakfast, Lunch, Dinner, Snacks. Customizable via web (not mobile) -- can rename or add meal categories.
- **Nutrition breakdown**: Per-meal calorie subtotals. Daily total with remaining calories. Macros (carbs/fat/protein) shown. Full micro/vitamin breakdown requires Premium.
- **Weekly/monthly**: Premium provides weekly nutrition reports and trends.
- **Editing past entries**: Can navigate to any past date and edit freely.
- **Targets & progress**: Calorie goal with remaining counter. Macro targets (Premium). Progress bar for daily calories.

### Food Detail/Editing
- **Nutrition shown**: Calories, protein, carbs, fat. Premium adds micronutrients (vitamins, minerals, cholesterol, sodium, fiber, sugar, etc.).
- **Serving options**: Multiple serving sizes per food (cups, grams, oz, pieces, servings). User selects from dropdown.
- **Portion adjustment**: Numeric text field for quantity. Select serving unit from picker.

### Social/Gamification
- **Streaks**: Introduced Summer 2025. Counts consecutive days of meal logging.
- **Social**: Community forums. Can share diary with friends. Premium meal planning features.

---

## 2. CalAI

### Food Logging
- **Database**: Not disclosed size. Focused on AI photo recognition as primary input.
- **Search**: Text search with categorized tabs: My Meals, My Foods, Saved Scans. Universal search across all categories.
- **Custom foods**: Personal Food Library with complete nutritional profiles and flexible serving sizes. Can adjust all macros manually.
- **Barcode scanning**: Yes, included in free tier.
- **Quick-add calories**: Not prominently featured; photo-first approach.
- **Copy/paste meals**: Can relog past meals from history. Food Memory feature remembers frequent meals.
- **Favorites/frequent**: Saved Scans, My Meals, My Foods tabs organize previously logged items.

### Recipe Management
- **Create recipes**: Custom Meal Builder -- combine multiple foods with automatic nutrition calculation, one-tap logging.
- **Import from URL**: Not documented.
- **Serving sizes**: Flexible serving sizes in Personal Food Library.
- **Saved meals**: Yes, reusable meal combinations saved for one-tap logging.

### Diary/History
- **Day view**: Food diary with meal entries. Photos can be attached to any food entry. Meals organized by category.
- **Nutrition breakdown**: Calories, protein, carbs, fat per entry and daily total.
- **Weekly/monthly**: Not prominently featured.
- **Editing past entries**: Can adjust portion sizes and food items after logging.
- **Targets & progress**: Calorie and macro targets with progress tracking.

### Food Detail/Editing
- **Nutrition shown**: Calories, protein, carbs, fat (macros only -- no detailed micronutrients).
- **Serving options**: Flexible serving sizes, manual adjustment available.
- **Portion adjustment**: Manual adjustment of AI-estimated portions. Text input for quantities.

### Social/Gamification
- **Streaks**: Achievement badges for milestones (2025 update).
- **Social**: Groups feature (Fall 2025) for logging meals with friends. Progress Photos with visual timeline.

---

## 3. MacroFactor

### Food Logging
- **Database**: Curated verified database. Entries checked for accuracy before inclusion (unlike user-contributed databases). Smaller but more accurate.
- **Search**: Claims "fewest taps" of any food logger in 20-app benchmark. Optimized for speed and efficiency.
- **Custom foods**: Full custom food creation with complete macro and micronutrient profiles. Auto-converts default serving to all standard units (g, oz, ml, cup) when weight/volume provided.
- **Barcode scanning**: Yes. Can associate custom foods with barcodes for future scans.
- **Quick-add calories**: Yes, quick-add functionality available.
- **Copy/paste meals**: Flexible copy-and-paste. Smart history suggests likely items based on patterns.
- **Favorites/frequent**: Favorites system. Smart history learns patterns and surfaces likely foods.

### Recipe Management
- **Create recipes**: Full recipe builder with description and preparation steps. Ingredients added from database.
- **Import from URL**: Yes, recipe URL import supported.
- **Serving sizes**: Custom and standard serving sizes for recipes. Can sort and filter recipes.
- **Saved meals**: "Plate" concept -- a unified food logging interface. Meals built on the Plate can be saved and reused.

### Diary/History
- **Day view**: Scrollable timeline showing foods throughout the day. "Plate" metaphor with minified view (swipe up) or expanded view (swipe down) showing rich macro/micro breakdowns.
- **Nutrition breakdown**: Per-meal and per-day. Full macro and micronutrient breakdowns available in expanded view.
- **Weekly/monthly**: Customizable dashboard with trends, charts, and data visualization. Weight trend analysis with adaptive algorithm.
- **Editing past entries**: Full editing of past entries.
- **Targets & progress**: Adaptive macro targets that auto-adjust based on weight trends. This is MacroFactor's core differentiator.

### Food Detail/Editing
- **Nutrition shown**: Full macros and micronutrients.
- **Serving options**: Auto-generated standard conversions (g, oz, ml, cup, etc.) from single weight entry.
- **Portion adjustment**: Numeric input. Multiple unit options auto-generated.

### Social/Gamification
- **Streaks**: Not a focus. No gamification -- positioned as a serious tool for athletes/lifters.
- **Social**: No social features. Individual-focused.

---

## 4. Cronometer

### Food Logging
- **Database**: 1.1M+ foods, lab-analyzed and verified (NCCDB, USDA). Accuracy is the core selling point. Tracks up to 84 micronutrients.
- **Search**: Text search with database. Add from Recent, Frequent, or Custom categories.
- **Custom foods**: Full custom food creation with all nutrient fields. Can add multiple serving sizes if weight is defined.
- **Barcode scanning**: Yes, free tier. Quick-add nutrients also possible via camera scan of nutrition label.
- **Quick-add calories**: Yes, Quick Add feature. Can also scan a nutrition label directly to quick-add.
- **Copy/paste meals**: Standout feature. Copy entire days or individual meals to any other date with one click. Calendar-based date picker to copy from any past date.
- **Favorites/frequent**: Recent and Frequent tabs. Can schedule foods/meals to auto-appear in diary on specific days.

### Recipe Management
- **Create recipes**: Full recipe builder with ingredients from database.
- **Import from URL**: Yes (Gold/Premium). Recipe Importer parses recipe URLs, auto-creates custom recipe with ingredients, measurements, and verified nutritional data.
- **Serving sizes**: Custom serving sizes. Additional servings auto-calculated from weight.
- **Saved meals**: Diary Groups for meal categories. Can save and reuse meal combinations.

### Diary/History
- **Day view**: Diary Groups (customizable: Breakfast, Lunch, Dinner, Snacks, etc.). Color-coded nutrient bars show at-a-glance sufficiency.
- **Nutrition breakdown**: Most detailed of any app. Up to 92 nutrients and compounds. Vitamins, minerals, amino acids all broken down. Nutrition Scores grade overall diet quality. Per-meal and per-day views.
- **Weekly/monthly**: Charts and trends (Gold). Printable nutrition reports. Nutrient Oracle for insights.
- **Editing past entries**: Full editing of any past date.
- **Targets & progress**: Nutrient targets with color-coded progress (green = met, yellow = close, red = deficient). Macro targets plus micro targets for all 84 nutrients.

### Food Detail/Editing
- **Nutrition shown**: THE most comprehensive: macros, fiber, sugar, all vitamins (A, B1-B12, C, D, E, K), all minerals (calcium, iron, zinc, etc.), amino acids, fatty acid profiles. Up to 92 data points per food.
- **Serving options**: Multiple serving sizes. Weight-based and volume-based options.
- **Portion adjustment**: Numeric input for quantity with serving size picker.

### Social/Gamification
- **Streaks**: Not a focus.
- **Social**: Gold allows sharing custom foods and recipes. No social feed or community in-app.

---

## 5. Lose It!

### Food Logging
- **Database**: 56M+ foods (largest claim). Includes branded and restaurant items.
- **Search**: Text search with database. Recent and Frequent tabs.
- **Custom foods**: Full custom food and exercise creation.
- **Barcode scanning**: Yes, free tier.
- **Quick-add calories**: Yes.
- **Copy/paste meals**: Can log meals for upcoming week (Premium). Copy previous day functionality.
- **Favorites/frequent**: Recent and Frequent food lists.

### Recipe Management
- **Create recipes**: Recipe builder with ingredient search, serving count, auto-calculated per-serving nutrition.
- **Import from URL**: Not prominently documented.
- **Serving sizes**: Set total recipe servings. Adjusting serving count auto-recalculates per-serving nutrition.
- **Saved meals**: Meals as food groups for quick re-logging. Separate from recipes.

### Diary/History
- **Day view**: Daily calorie budget dashboard. Meals logged under meal categories. Remaining calories prominently shown.
- **Nutrition breakdown**: Calories and macros. Weekly summary of calorie intake and macro breakdown.
- **Weekly/monthly**: Weekly summary view with calorie and macro trends.
- **Editing past entries**: Can edit past entries.
- **Targets & progress**: Daily calorie budget with remaining counter. Macro targets.

### Food Detail/Editing
- **Nutrition shown**: Calories, macros. Some micronutrient data.
- **Serving options**: Standard serving size options per food.
- **Portion adjustment**: Numeric input for quantity.

### Social/Gamification
- **Streaks**: Yes, streak tracking for consecutive logging days.
- **Social**: Social features and challenges. Community elements.

---

## 6. FatSecret

### Food Logging
- **Database**: Large, regularly updated. Branded and restaurant items. Generic foods well-covered.
- **Search**: Text search with autocomplete. "Most Eaten" and "Recently Eaten" tabs populate automatically based on usage patterns.
- **Custom foods**: Full custom food creation.
- **Barcode scanning**: Yes, free tier.
- **Quick-add calories**: Yes.
- **Copy/paste meals**: Can copy meals between dates.
- **Favorites/frequent**: Most Eaten and Recently Eaten auto-populated tabs. Favorites API/list available.

### Recipe Management
- **Create recipes**: CookBook feature -- personal recipe list with ingredients, serving sizes, cooking instructions. Logging a recipe adds one serving to diary.
- **Import from URL**: Not documented.
- **Serving sizes**: Custom serving sizes in recipes. Standard sizes for database foods.
- **Saved meals**: Meals can be saved as recipes in CookBook for quick logging.

### Diary/History
- **Day view**: Diary page with configurable columns (checkmark which nutrition columns to show). "Detailed View" toggle shows expanded nutrition.
- **Nutrition breakdown**: In-depth macronutrient breakdowns. Configurable detail level.
- **Weekly/monthly**: Diet Calendar showing daily calorie consumption/burn. Monthly summary nutritional data.
- **Editing past entries**: Full editing of past dates.
- **Targets & progress**: Calorie and macro goals with progress tracking. Journal for progress notes.

### Food Detail/Editing
- **Nutrition shown**: Calories, macros, detailed breakdown. Good depth for a free app.
- **Serving options**: Multiple serving sizes per food.
- **Portion adjustment**: Select serving size, enter quantity.

### Social/Gamification
- **Streaks**: Not prominently featured.
- **Social**: Built-in community. Journal for sharing progress. Social engagement is a core differentiator for FatSecret.

---

## Cross-App Feature Matrix

| Feature | MFP | CalAI | MacroFactor | Cronometer | Lose It! | FatSecret |
|---|---|---|---|---|---|---|
| **Food database size** | 18M+ | ? | Curated/verified | 1.1M verified | 56M+ | Large |
| **Barcode scan (free)** | No (Premium) | Yes | Yes | Yes | Yes | Yes |
| **Custom foods** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Quick-add calories** | Yes | No | Yes | Yes | Yes | Yes |
| **Copy meals** | Yes | Partial | Yes | Best | Yes | Yes |
| **Favorites/frequent** | Yes | Yes | Smart history | Yes + scheduling | Yes | Auto-populated |
| **Recipe builder** | Yes | Meal builder | Yes | Yes | Yes | CookBook |
| **Recipe URL import** | Yes | No | Yes | Yes (Gold) | No | No |
| **Meal groups** | 4 default | Categories | Plate/timeline | Diary Groups | Meal categories | Diary groups |
| **Micronutrients** | Premium | No | Yes | 84-92 nutrients | Some | Some |
| **Weekly/monthly view** | Premium | No | Dashboard | Gold | Yes | Calendar |
| **Adaptive targets** | No | No | Yes (core feature) | No | No | No |
| **Streaks** | Yes (2025) | Badges | No | No | Yes | No |
| **Social/community** | Forums | Groups | No | No | Challenges | Community |
| **Free tier completeness** | Limited | Good | N/A (subscription) | Good | Good | Best |

---

## Table-Stakes vs Differentiator Analysis

### Table Stakes (must-have for credibility)

These features are expected by users in every calorie tracker in 2026:

1. **Food search with large database** -- text search with autocomplete, recent/frequent tabs
2. **Barcode scanning** -- free tier, instant lookup
3. **Custom food creation** -- full macro entry at minimum
4. **Daily food diary with meal groups** -- Breakfast/Lunch/Dinner/Snacks default slots
5. **Calorie and macro targets with progress visualization** -- remaining calories counter, macro progress bars
6. **Recipe builder** -- add ingredients, set servings, auto-calculate per-serving nutrition
7. **Copy/paste meals from previous days** -- minimum one-tap copy of yesterday
8. **Quick-add calories** -- enter raw number without food search
9. **Edit past diary entries** -- navigate to any date and modify
10. **Serving size picker with quantity input** -- grams, oz, cups, pieces, servings

### Strong Expectations (users will notice absence)

11. **Favorites/frequent foods** -- auto-populated from usage, quick re-logging
12. **Saved meals** -- group of foods logged as one action
13. **AI photo logging** -- now expected as of 2025-2026, no longer a differentiator
14. **Weekly summary/trends** -- at least basic weekly calorie trend

### Differentiators (not expected but valued)

15. **Recipe URL import** -- only MFP, MacroFactor, Cronometer have this
16. **Full micronutrient tracking (vitamins, minerals)** -- Cronometer dominates here
17. **Adaptive/smart targets** -- only MacroFactor does this
18. **Scheduled/recurring meals** -- only Cronometer
19. **Nutrition label camera scan** -- Cronometer's quick-add from label photo
20. **Smart history / pattern prediction** -- MacroFactor's learned suggestions

---

## Minimum Viable Feature Set for Tastimate

Based on this analysis, the minimum feature set to be a credible food tracker competing at the $9.99 one-off price point:

### Phase 1: MVP Food Tracker
- [ ] Food search with database (start with USDA/OpenFoodFacts)
- [ ] Barcode scanning (free, using Open Food Facts or similar)
- [ ] Custom food creation (macros at minimum)
- [ ] Daily diary with 4 meal slots (Breakfast, Lunch, Dinner, Snacks)
- [ ] Calorie + macro targets with remaining counter and progress bars
- [ ] Quick-add calories
- [ ] Serving size picker with quantity input (g, oz, cups, pieces)
- [ ] Edit past entries (date navigation)
- [ ] AI photo logging (already building this -- our core differentiator is on-device)

### Phase 2: Stickiness Features
- [ ] Recipe builder (add ingredients, set servings)
- [ ] Copy meals from previous days
- [ ] Favorites and frequent foods (auto-populated)
- [ ] Saved meals (food groups for one-tap logging)
- [ ] Weekly calorie/macro summary view
- [ ] Streaks counter

### Phase 3: Differentiation
- [ ] Recipe URL import
- [ ] Micronutrient tracking (vitamins, minerals -- leverage our nutrition knowledge graph)
- [ ] Recurring/scheduled meals
- [ ] Nutrition label scan (camera OCR)
- [ ] Smart suggestions based on logging patterns

### What We Skip
- Social/community features (not aligned with local-first, no-subscription model)
- Adaptive algorithm-based targets (MacroFactor's domain, complex to build)
- Dietitian-created meal plans (content-heavy, ongoing cost)
- Fitness/exercise tracking integration (out of scope for food tracker MVP)

---

## Key UX Patterns to Follow

### Logging Speed is King
MacroFactor won the "fastest food logger" benchmark across 20 apps. Every extra tap loses users. Prioritize:
- Search should be instant (local-first helps here)
- Recent/Frequent should be the default view, not empty search
- One-tap re-log from history
- Barcode scan should be < 2 seconds to result

### Progressive Disclosure for Nutrition
- Default view: Calories + 3 macros (protein, carbs, fat)
- Expandable: Fiber, sugar, sodium, cholesterol
- Deep view: Full micronutrients (when available)
- Cronometer's color-coded nutrient bars are an excellent UX pattern

### Meal Slots as Organization, Not Enforcement
- Default 4 slots, but don't force users into them
- MacroFactor's timeline approach is more flexible
- Let users rename/add/remove meal slots

### Copy/Paste is a Retention Feature
- Cronometer's "copy any past day" with calendar picker is the gold standard
- Most users eat similar meals repeatedly -- make this effortless
- "Add Yesterday's Meal" as a one-tap shortcut (MFP pattern)
