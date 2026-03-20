# MacroFactor Competitive Analysis

**Version analyzed**: 5.7.5 (Android, Pixel 7 Pro)
**Date**: 2026-03-19
**Subscription**: ~$12/month (or ~$72/year)

## Executive Summary

1. **Timeline-based food log is the core UX differentiator** — hourly slots replace traditional meal grouping (breakfast/lunch/dinner), reducing categorization friction while naturally organizing the day
2. **"Your Plate" staging area** is a key pattern — users build a plate of multiple foods before committing to log, reducing the number of save actions and enabling macro preview before logging
3. **Adaptive TDEE (Expenditure)** is what justifies the subscription — the algorithm continuously recalculates your total daily energy expenditure from weight + intake data, making static calorie targets obsolete. This is their moat.
4. **6 distinct food entry methods** (Search, Scan, AI Photo, AI Photo+Text, Quick Add, Library) — MacroFactor obsesses over reducing logging friction because that's the #1 reason users abandon food trackers
5. **AI food recognition is "Beta" and cloud-based** — they're still catching up here. Our on-device VLM approach has a real opportunity to be better/faster/private.
6. **Extreme customizability** — nearly every UI element (food tiles, timeline hours, dashboard widgets, shortcuts, themes) is configurable. This depth is impressive but only matters to power users.
7. **Open Food Facts integration exists but is off by default** — validates our OFF cache strategy, confirms branded food databases (like FatSecret/Nutritionix) are still the primary data source for search-based trackers
8. **Weight trend smoothing** is table stakes for any serious tracker — the signal-vs-noise visualization is simple but powerful
9. **The Strategy/Coaching system** (programs, goals, check-ins) is the second subscription justifier — we can replicate goal-setting locally but the adaptive coaching requires ongoing TDEE computation
10. **Notes per day** in the food log — lightweight but valuable for correlating how you feel with what you ate

## Screen-by-Screen Analysis

### 1. Dashboard

**Layout**: Vertical scroll with modular card system. Sections are: Weekly Nutrition, Insights & Analytics, Habits, Body Metrics, Nutrition, General, More (Customize Dashboard, Nutrition Data Manager).

**Weekly Nutrition Widget**:
- 7-day grid (M-S) with 4 rows: Calories, Protein, Fat, Carbs
- Current day highlighted with white border
- Each cell would show a filled bar when food is logged
- Right side shows "0 of 2132" format (consumed of target)
- Consumed/Remaining toggle changes the interpretation
- 3 horizontal page dots suggest swipeable alternative views

**Insights & Analytics Cards** (2x2 grid):
- Expenditure (Last 7 Days) — sparkline + value
- Weight Trend (Last 7 Days) — sparkline + value
- Energy Balance (Last 7 Days) — surplus/deficit
- Goal Progress (Last 240 Days) — percentage bar

**Habits Section**:
- Weigh-In: 30-day heatmap grid, "0/7 this week" streak counter
- Food Logging: Same heatmap, same streak counter
- Problem being solved: **consistency accountability** — users who log daily get better outcomes

**Body Metrics** (2-column cards):
- Scale Weight: Last 7 entries, sparkline, latest value (70.1 kg)
- Visual Body Fat: Last 7 entries, latest value (20.0%)

**Nutrition Cards** (2x2):
- Calories, Protein, Fat, Carbs — each with "Today" label, progress bar, and value

**General**:
- Steps: Last 7 Days, bar chart, value (1111 steps)

**Dashboard Customization**: "Customize Dashboard" and "Nutrition Data Manager" links at bottom.

**Key Pattern**: The dashboard is a _summary of summaries_ — every card is a preview that links to a detail view. Information density is high but scannable.

### 2. Food Log

**Layout**: Timeline-based, not meal-based. This is a radical departure from MyFitnessPal/Cronometer/Lose It.

**Header**:
- Hamburger menu (left), date label "Today" (center, tappable), forward/back arrows
- Week bar showing M-S with dates (16-22 for this week), current day highlighted
- Macro summary: Cal 0/2132, P 0/143, F 0/71, C 0/229 with color-coded progress bars
- Page dots (2) suggesting swipeable alternative view

**Timeline**:
- Hourly slots from 7 AM to 11 PM (customizable range)
- Each slot has a "+" button to add food at that time
- Timeline is connected with a vertical line on the left
- Food entries would appear as tiles at their logged time

**Bottom of Timeline**:
- **Notes** section with "+" to add daily notes
- **Nutrition Overview** link — detailed macro/micro breakdown for the day
- **Customize Food Log** link

**Persistent Search Bar**: Always visible at bottom, "Search for a food" with barcode scanner icon. This is genius — the #1 action (adding food) is always 1 tap away.

**Key Insight**: The timeline model solves two problems: (1) eliminates "what meal is this?" friction, and (2) naturally captures when you eat, which is valuable data for intermittent fasting / meal timing analysis.

### 3. Food Search / Add Food Flow

**Entry Point**: Tapping search bar or "+" on any hour slot opens the full-screen logger.

**Logger Tabs** (6 tabs, horizontally scrollable):
1. **Scan** — Barcode scanner
2. **Search** — Text search (default)
3. **AI** — Photo or Photo+Text recognition (Beta)
4. **Quick Add** — Manual cal/macro entry
5. **Library** — Saved/custom foods
6. (6th tab partially visible, may be "Describe" or similar)

**Search Results**:
- "From History" section with "See 36 More" link — prioritizes foods you've logged before
- Each result shows: Food name, calorie icon + value, P/F/C values, serving description, "+" to quick-add
- Branded results on by default, Open Food Facts off by default
- Keyboard suggestions include food emojis (nice touch)

**AI Tab**:
- Sub-tabs: Photo | Photo & Text
- "Beta" badge — this is new
- Camera viewfinder with shutter button
- Upload icon (from gallery) and trash icon
- Problem being solved: **reduce friction for unpackaged foods** where barcode scanning doesn't work

**Quick Add**:
- Energy field with kcal unit dropdown
- "Macro sum is 0 kcal" — auto-calculates from macros
- Protein, Fat, Carbs individual fields
- Can enter total calories OR individual macros (smart flexibility)
- Two buttons: "Quick Add" (add to plate) and "Log Foods" (commit)

### 4. Your Plate (Food Staging)

**Key UX Pattern**: Before foods are committed to the log, they go to "Your Plate" — a staging area.

- Shows all foods about to be logged
- **Nutrition section** with Plate/Day toggle — see macros for just this plate vs. entire day
- "Show all nutrients" toggle — expands beyond Cal/P/F/C to micronutrients
- "Log Foods" button commits everything to the timeline

**Why This Matters**: The plate metaphor maps to real eating behavior (you build a meal, then eat it). It reduces logging friction by batching multiple foods into one commit action.

### 5. Strategy (Goals & Coaching)

**Layout**: Goal-oriented with program management.

**Check-In Circle**: Large prominent "CHECK IN — it's time" prompt. Weekly check-ins drive the adaptive TDEE algorithm.

**Coached Program** (Feb 9 – Now):
- Weekly calorie targets displayed as chips (2132 cal/day)
- Color-coded macro distribution bars per day (green = on track)
- "New Program" and "Edit Program" buttons

**Weight Loss Goal** (Jul 22 – Now):
- Goal Weight: 65.0 kg
- Goal Rate: -0.35 kg / -0.5% per week
- "New Goal" and "Edit Goal" buttons

**Goal History**: Shows timeline of goal changes with current weight context (69.6 kg).

**What the Subscription Pays For**: The adaptive algorithm that adjusts your calorie targets based on actual weight change vs. predicted weight change. This is genuinely valuable and hard to replicate locally without ongoing computation.

### 6. Expenditure (Adaptive TDEE)

**The Core Algorithm**:
- Shows calculated daily energy expenditure over time
- Average: 2470 kcal, with difference from previous period
- "Flux Range" — confidence interval visualization (shaded area around the line)
- Status indicators: "Updating" (still gathering data) vs "Holding" (stable estimate)
- Time ranges: 1W, 1M, 3M, 6M, 1Y, All
- Granularity: Daily

**Problem Being Solved**: Traditional calorie calculators use static formulas (Harris-Benedict, Mifflin-St Jeor). MacroFactor uses your actual intake + weight change data to compute your real TDEE, which accounts for NEAT, metabolic adaptation, etc.

### 7. Weight Trend

- Two-line chart: **Scale Weight** (raw data points) and **Trend Weight** (smoothed exponential moving average)
- The trend line filters out daily water weight fluctuations
- Average and Difference metrics at top
- Same time range selectors

**Tutorial Quote**: "Scale Weight data tends to be quite noisy. Your Weight Trend is the signal in all of that noise."

### 8. Insights & Analytics

Four metric cards in a 2x2 grid:
- **Expenditure**: TDEE estimate
- **Weight Trend**: Smoothed weight
- **Energy Balance**: Surplus/deficit (intake - expenditure)
- **Goal Progress**: Percentage toward goal weight (69% over 240 days)

Each card is a mini-dashboard that links to a full detail view with time-series charts.

### 9. More / Settings

**General**: Account, Subscription, Integrations, Units
**Feature Settings**: Dashboard (customize), Food Log (customize)
**Theme**: 3 options (light colorful, light muted, dark)
**Data Management**: Data Export, Data Visibility, Account & Data Deletion
**Community**: Reddit, Facebook, Instagram, Knowledge Base, Roadmap, Support
**Other**: Legal, App Icon (customizable), Tutorials, About

### 10. Food Log Customization (Deep)

**Nutrient Reporting**: No Overages vs Show Overages (negative numbers when exceeding targets)
**Timeline Options**:
- Hour Range: 7 AM – 11 PM (adjustable)
- Alignment: Left aligned
- Add Foods to Hour: Plus icon visibility
- Food Timestamps: Show/hide
- Hourly Macro Totals: Show/hide

**Food Search**:
- Branded Results: On/Off
- **Open Food Facts Results: Off by default** (validates our approach of having OFF as supplementary)

**Food Tiles**: Customize how foods appear in timeline and in search
**Logger Options**: Logger Banner, Time Selection, Favorite Measurements, Optimization (speed vs detail)

### 11. + FAB / Shortcuts

The center "+" floating action button opens a **Shortcuts** bottom sheet:

**Quick Access Icons** (top row): Your Foods, Weight, Search, Barcode
**List Items**: Quick Add, Metrics, Recipes, New Recipe
**Customization**: Settings icon to reorder/configure shortcuts

**Key Pattern**: The FAB is a universal entry point that adapts to user preferences. Power users can pin their most-used actions.

## Feature Comparison Matrix

| Feature | MacroFactor | Tastimate (Current) | Tastimate (Planned) | Priority |
|---------|-------------|---------------------|---------------------|----------|
| **Food Logging** | | | | |
| Timeline-based log | Hourly slots, 7AM-11PM | Basic food_entries table | Phase 3 | High |
| Meal grouping | No (by time) | No | Phase 3 decision | — |
| Search database | Branded + OFF (optional) | OFF cache + KG | Phase 3 | High |
| Barcode scanning | Yes | No | Phase 5+ | Medium |
| AI photo recognition | Beta (cloud) | YOLO + VLM (on-device) | Core feature | High |
| Quick calorie add | Cal/macro entry | No | Phase 3 | Medium |
| "Plate" staging area | Yes, multi-food | No | Phase 3 | Medium |
| Notes per day | Yes | No | Phase 3 | Low |
| Persistent search bar | Always visible | No | Phase 3 | High |
| Food history/recents | "From History" first | No | Phase 3 | High |
| **Nutrition** | | | | |
| Macro tracking (CPFC) | Yes | Yes (KG data) | Phase 3 | High |
| Micro tracking | Toggle "Show all" | USDA data available | Phase 3+ | Low |
| Daily totals | Yes | No (DB only) | Phase 3 | High |
| Weekly overview | 7-day grid | No | Phase 3 | Medium |
| **Goals & Strategy** | | | | |
| Adaptive TDEE | Core feature ($$) | No | Maybe local algo | Low |
| Goal setting | Weight + rate | No | Phase 3+ | Medium |
| Programs/coaching | Yes (subscription) | No | Skip | — |
| Check-ins | Weekly prompt | No | Skip | — |
| **Analytics** | | | | |
| Weight trend (smoothed) | EMA line + raw | No | Phase 3+ | Medium |
| Expenditure chart | Yes | No | Skip | — |
| Energy balance | Surplus/deficit | No | Phase 3+ | Low |
| Goal progress | % toward target | No | Phase 3+ | Low |
| Habit heatmaps | 30-day grid | No | Phase 3+ | Low |
| Steps integration | Via Health Connect | No | Phase 5 | Low |
| **Recipes** | | | | |
| Recipe creation | Yes | custom_recipes table | Phase 3 | High |
| Recipe from log | Likely yes | No | Phase 3 | Medium |
| **Data & Export** | | | | |
| Data export | CSV/etc | CSV/JSON export | Done | — |
| Data visibility | Fine-grained | No | Low priority | Low |
| Backup | Cloud (subscription) | Google Drive backup | Phase 3.7 | High |
| **Customization** | | | | |
| Theme options | 3 themes | System theme | Low priority | Low |
| Food log layout | Highly customizable | N/A | Phase 3+ | Low |
| Dashboard widgets | Customizable | N/A | Phase 3+ | Low |
| App icon | Customizable | Default | Skip | — |
| Shortcuts/FAB | Configurable | No | Phase 3 | Medium |

## UX Patterns Worth Adopting

### 1. Persistent Search Bar
The always-visible search bar at the bottom of the Food Log is the single best UX decision in MacroFactor. It means adding food is always 1 tap away, not buried behind navigation.

**For Tastimate**: Our equivalent should be a persistent "Tap to scan or search" bar that triggers either the camera (AI-first) or text search.

### 2. "From History" First in Search
When searching, MacroFactor shows foods you've eaten before at the top. This dramatically speeds up logging for the 80% of meals that repeat weekly.

**For Tastimate**: We should rank search results by personal frequency, not just database relevance. Our `food_entries` table already tracks this data.

### 3. Quick Add as Escape Hatch
When the AI fails, when the database doesn't have your food, when you just know "I ate about 500 calories" — Quick Add lets you log rough numbers fast. This prevents the all-or-nothing problem where people stop logging entirely because one meal was too hard to look up.

**For Tastimate**: Essential. Our AI pipeline won't be perfect. A quick cal/macro entry bypasses the entire detection pipeline.

### 4. Plate Staging Area
Building a plate before committing solves multi-food meals elegantly. Instead of "add rice, save, add chicken, save, add veggies, save" you build the whole plate and commit once.

**For Tastimate**: Our AI photo detection already identifies multiple foods in one frame. The plate metaphor maps perfectly — show detected foods as a plate, let the user confirm/edit, then commit all at once.

### 5. Daily Macro Summary Always Visible
The Food Log header constantly shows Cal/P/F/C consumed vs target. Users never have to navigate somewhere else to check "how much protein do I have left?"

**For Tastimate**: Pin a macro summary to the diary view header. Even without explicit goals, showing daily totals is valuable.

### 6. Weight Trend Smoothing
Raw weight fluctuates ±1-2 kg daily from water/sodium/glycogen. Showing only raw weight causes anxiety. The trend line provides emotional reassurance that progress is real despite daily noise.

**For Tastimate**: When we add weight tracking, always show the smoothed trend prominently and the raw data as secondary. Simple exponential moving average (like Hacker's Diet) is sufficient.

## Features to Skip (and Why)

### 1. Adaptive TDEE / Expenditure Algorithm
**What it does**: Continuously recalculates your energy expenditure from intake + weight change data.
**Why skip**: This is MacroFactor's core subscription value prop. Implementing it well requires significant data science investment, and it only works after 2+ weeks of consistent logging. For our local-first v1, static TDEE formulas (Mifflin-St Jeor) are fine — users can adjust manually. Revisit post-launch if users demand it.

### 2. Coaching Programs & Check-ins
**What it does**: Weekly "check-in" drives adaptive algorithm, programs auto-adjust targets.
**Why skip**: Requires the adaptive TDEE backend. Our value prop is AI-first convenience, not coaching. Users who want coaching are MacroFactor's audience, not ours.

### 3. Extreme UI Customization
**What it does**: 20+ customizable settings for food tiles, timeline appearance, dashboard widgets.
**Why skip for v1**: Customization is a power-user feature that adds maintenance burden. Ship an opinionated, good-by-default UI first. Add customization later when users request specific changes.

### 4. Integrations (Health Connect, Apple Health)
**What it does**: Imports steps, syncs weight to/from health platforms.
**Why skip for v1**: Nice-to-have, not core. Phase 5 is already planned for health data import. Don't let integration work delay the core food logging experience.

### 5. Branded Food Database (FatSecret/Nutritionix)
**What it does**: Millions of branded products with exact nutrition data.
**Why skip**: These require paid API subscriptions, which contradicts our no-subscription model. We already have OFF (free, open source) + USDA + our knowledge graph. For branded products, barcode → OFF lookup covers most cases. Gap analysis can come later.

### 6. App Icon Customization
**Why skip**: Cute but zero impact on user outcomes. Ship one good icon.

## Recommendations for Phase 3 (Diary UI)

### The Core Problem to Solve
MacroFactor solves: "How do I log my food with minimal friction?"
Tastimate should solve: "How do I log my food with almost zero effort?"

We have a structural advantage: AI detection eliminates the search/type/select flow entirely for most meals. MacroFactor's AI is Beta and cloud-based. Ours is on-device and core to the product.

### Specific Recommendations

1. **Primary flow: Photo → Plate → Confirm → Logged**
   - User takes photo → YOLO detects → VLM identifies → KG looks up nutrition → presents as "plate"
   - User confirms/edits → committed to diary
   - This replaces MacroFactor's Search → Select → Adjust Serving → Add to Plate → Log flow

2. **Secondary flow: Search/Manual Add**
   - For when AI fails or user knows exactly what they ate
   - Search bar should search our KG + OFF cache simultaneously
   - Include Quick Add (cal/macro only) as escape hatch
   - Show "From History" results first

3. **Daily View: Hybrid Timeline**
   - Show entries chronologically (like MacroFactor's timeline) rather than by meal category
   - But use larger time blocks (morning/afternoon/evening) not hourly — hourly is overkill for most users
   - Always-visible macro summary at top

4. **Entry Cards Should Show**:
   - Photo thumbnail (our key differentiator!)
   - Food name(s) identified
   - Cal and P/F/C values
   - Time logged
   - Tap to expand/edit

5. **Implement "Your Plate" for Multi-Food Photos**:
   - When AI detects multiple foods, show them as a plate preview
   - User can remove/add/edit individual items before committing
   - Maps naturally to our `scanned_dishes` → `food_entries` flow

6. **Daily Macro Summary Header**:
   - Persistent Cal/P/F/C consumed display at top of diary
   - Goal targets can come later (static formula for v1)
   - Consumed/Remaining toggle is useful

7. **Recipe Management**:
   - "Save as recipe" from any logged plate/meal
   - Quick re-log of saved recipes (1-2 taps)
   - We already have `custom_recipes` table

8. **Weekly Overview** (lower priority for v1):
   - 7-day macro grid similar to MacroFactor's dashboard widget
   - Helps users see patterns ("I always undereat protein on weekends")

### What Makes Us Better Than MacroFactor

| Dimension | MacroFactor | Tastimate |
|-----------|-------------|-----------|
| Primary input | Manual search/type | Photo → AI detection |
| Subscription | ~$12/month forever | ~$9.99 one-time |
| Data privacy | Cloud-required | Local-first, on-device ML |
| AI recognition | Beta, cloud-dependent | Core feature, on-device |
| Photo association | No photos in log | Every entry can have a photo |
| Offline support | Limited | Full offline |
| Food database | Proprietary + branded | Open (USDA + OFF + KG) |

### Research Gaps (Things to Investigate Further)

1. **Cronometer's micro-nutrient UI** — MacroFactor focuses on macros. For users who care about vitamins/minerals, how does Cronometer display that data? We have USDA micro data available.
2. **Lose It's photo AI** — they have cloud-based photo recognition too. How does their UX compare?
3. **Samsung Food (formerly Whisk)** — on-device recipe management, local-first approach, possible UX patterns
4. **Yazio's free tier** — what food tracking features work without a subscription? Validates our one-time purchase model
5. **How CalAI handles multi-food detection** — they're our closest competitor in the AI-first space
