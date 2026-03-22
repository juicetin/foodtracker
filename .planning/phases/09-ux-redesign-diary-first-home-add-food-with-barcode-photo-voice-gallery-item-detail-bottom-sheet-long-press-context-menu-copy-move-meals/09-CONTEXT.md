# Phase 9: UX Redesign — Context

**Gathered:** 2026-03-23
**Status:** Ready for planning
**Source:** PRD Express Path (.planning/UX-FLOW-v1.md)

<domain>
## Phase Boundary

Complete UX overhaul of the Tastimate app. Replaces current navigation and screens with a diary-first home, unified add-food flow (barcode/photo/voice/gallery/search), item detail bottom sheet, long-press context menus, and copy/move meal operations. This phase delivers the visual and interaction layer — all underlying ML, nutrition DB, and data layer features already exist from prior phases.

Key drivers from QA-2026-03-23:
- #1 Long press diary item crashes (must be fixed as part of new long-press context menu)
- #2 Re-log meal tap triggers action directly (replaced by tap=detail, long-press=actions)
- #3 Third toggle view removed (keep two: macros + ingredient breakdown)
- #6 No barcode option on add food screen (new add food screen includes barcode)

</domain>

<decisions>
## Implementation Decisions

### Navigation Structure
- Bottom navigation: Today (Diary), + Add (FAB), Insights, Profile
- "Today" tab is the home screen (diary-first design)
- Center FAB opens Add Food flow
- Insights and Profile are separate tabs

### Diary Screen (Home)
- Header: remaining calories + macro progress bars (P/C/F)
- Date navigation: swipe or tap arrows, tap date for calendar picker
- Meal groups: Breakfast, Lunch, Dinner, Snacks — each with header + expandable items
- Each meal group has "+" button to add food pre-selecting that meal
- Meal group headers: tap = expand/collapse, long press = three-dot menu (Copy from date, Copy yesterday's, Save as meal)
- Food items: tap = bottom sheet detail, long press = context menu
- Adherence-neutral design: no shame colors, no red zones

### Long Press Context Menu (Food Item)
- Copy to clipboard
- Copy to another day
- Move to other meal
- Save as favorite
- Delete
- Fixes QA bug #1 (current long press crashes)

### Add Food Screen
- Search bar always visible with icons: search (left), camera (right-1), voice (right-2), barcode (right-3)
- Quick access tabs: Recent (last 20), Frequent (time-of-day aware), Favorites, My Recipes
- Entry methods section: Scan Photo, Scan Barcode, Quick Add Macros, From Gallery
- Fixes QA bug #6 (barcode now always visible)

### AI Photo Scan Results
- Photo thumbnail at top
- Identified dishes listed with total macros
- Per-dish ingredient breakdown with individual macros (fixes QA #4 display)
- Meal selector dropdown
- "Log Meal" button saves to diary and navigates back

### Barcode Scan Flow
- Camera viewfinder with barcode overlay
- Match found → product detail + portion setter → log
- No match → fallback to text search

### Item Detail Bottom Sheet
- Opened on tap of any diary food item
- Header: food name + star/delete/edit icons + logged time
- Total macros (cal, P, C, F)
- Ingredient list with per-ingredient macros and portions
- "+ Add ingredient" at bottom
- Expandable sections: micronutrients, nutrition source, view photo

### Meal Group Header Menu
- Copy from specific date
- Copy yesterday's meal
- Save current meal group as reusable meal template

### Claude's Discretion
- Animation/transition choices between screens
- Exact color palette and typography (follow existing theme or Material Design 3)
- Loading/shimmer states during AI processing
- Error states and empty states visual design
- Exact icon choices (emoji placeholders in spec, use Material Icons or similar)
- Swipe gestures on diary items (optional enhancement)
- Pull-to-refresh behavior
- Keyboard behavior in search
- Voice input integration details (speech-to-text API choice)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### UX Spec
- `.planning/UX-FLOW-v1.md` — Full UX flow spec with wireframes, interaction tables, and navigation structure

### QA Context
- `.planning/QA-2026-03-23.md` — Open bugs that overlap with this phase (#1 long-press crash, #2 re-log UX, #3 third toggle, #6 barcode missing)

### Architecture
- `docs/adr/005-local-first-no-subscription-architecture.md` — Local-first architecture decisions affecting data flow
- `docs/adr/004-usda-fdc-as-primary-nutrition-api.md` — Nutrition data source decisions

### Roadmap
- `.planning/ROADMAP.md` — Phase dependencies and overall project structure

</canonical_refs>

<specifics>
## Specific Ideas

- Date navigation should support both arrow taps and swipe gestures
- "Frequent" tab should be time-of-day aware (show breakfast foods in morning)
- Ingredient rows in scan results support: portion slider, swap ingredient, delete ingredient
- Below ingredients in scan results: "+" to add missing ingredient via search sheet
- Bottom sheet detail has expandable micronutrient section
- Nutrition source attribution shown in bottom sheet (e.g., "USDA", "OFF")

</specifics>

<deferred>
## Deferred Ideas

- Smart recents with ML-based suggestions (mentioned in handoff but not detailed in v1 spec)
- Recipe creation flow (covered by existing Phase 3.4)
- Gallery scanning improvements (Phase 4)
- Copy/move across multiple items at once (batch operations)
- Meal templates/presets management screen

</deferred>

---

*Phase: 09-ux-redesign*
*Context gathered: 2026-03-23 via PRD Express Path*
