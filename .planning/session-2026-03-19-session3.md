# Session Log — 2026-03-19 Session 3

## Autonomous work — user sleeping. Decisions documented for review.

### Decision 1: OFF Regional DB Priority Logic (USER CONFIRMED)
**Decision:**
- Regional DBs don't have barcodes → OFF is primary for barcode lookups
- Enrich OFF barcode results with regional nutrition data where available
- Auto-detect region from device locale, allow user prefs to override
- Text search: merge KG + OFF, KG first (local/faster)

### Decision 2: Weekly Trends (USER CONFIRMED)
**Decision:**
- 7/14/30 day toggle PLUS full historical view
- Macro trends (P/C/F over time), not just calories
- Daily average, goal adherence %, streak info
- Keep pure RN Views (no chart library)

### Decision 3: Data Export (USER CONFIRMED)
**Decision:**
- Both CSV and JSON formats
- Export ALL data (entries, recipes, favourites, ingredients, settings)
- Date range filtering supported
- Save to local device storage
- Google Drive sync support

## Work Plan
1. Weekly trends improvements (DiaryScreen) — 7/14/30/all + macro trends + stats
2. Data export service with TDD (CSV + JSON) + export UI on ProfileScreen
3. Google Drive sync for export (expo-file-system + google-drive API)
4. OFF barcode enrichment with regional data
5. Commit and push after each feature
