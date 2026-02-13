# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Accurate, effortless food tracking from photos you already take -- no manual entry, no barcode scanning, just eat, photograph, and review.
**Current focus:** Phase 1: Food Detection Foundation

## Current Position

**Phase:** 1 of 6 (Food Detection Foundation)
**Current Plan:** 3
**Total Plans in Phase:** 6
**Status:** Executing
**Last Activity:** 2026-02-13 -- Completed 01-04 (VLM Benchmark + Portion Estimation)

Progress: [████░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 21min
- Total execution time: 1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-food-detection-foundation | 3 | 64min | 21min |

**Recent Trend:**
- Last 5 plans: 01-01 (6min), 01-02 (45min), 01-04 (13min)
- Trend: improving

*Updated after each plan completion*

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 6min | 2 tasks | 11 files |
| Phase 01 P02 | 45min | 3 tasks | 8 files |
| Phase 01 P04 | 13min | 2 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Risk-retirement ordering -- validate ML accuracy before building UX on top
- [Roadmap]: Phase 3 (backend) can overlap Phase 2 (gallery scanning) since no dependency between them
- [Roadmap]: DET-04 (scale OCR) and DET-06 (LiDAR) deferred to Phase 6 as accuracy enhancers, not core detection
- [Revision]: Phase 1 includes go/no-go decision point on YOLO vs multimodal LLM -- accuracy is #1 priority above cost, speed, and on-device constraints
- [Revision]: Phase 3 expanded from USDA-only to multi-region food composition databases (NA, EU, APAC/Oceania, Asia) with Open Food Facts as global fallback
- [Revision]: DB-01 (region-specific routing) promoted from v2 to v1 and expanded into NUT-05 through NUT-10
- [01-01]: Food-101 via HuggingFace as primary classification source; ISIA-500 best-effort with graceful fallback
- [01-01]: Florence-2-base (0.23B) for auto-labeling with full-image fallback for zero-detection images
- [01-01]: Roboflow API key optional; pipeline degrades gracefully without it
- [01-02]: Programmatic dish generation (1003 dishes) instead of RecipeNLG HuggingFace download
- [01-02]: 5-tier best-guess fallback guarantees non-empty results for any input
- [01-02]: Manual cross-language variant links for dishes like nasi goreng -> fried rice
- [01-02]: DELETE journal mode for mobile SQLite export (single file, no WAL/SHM)
- [Phase 01-04]: Built-in standard serving sizes preferred over KG fuzzy matching to avoid false positives
- [Phase 01-04]: Geometry estimation needs Depth Anything V2 for +/-10% target; current baseline uses depth clamping (1-5cm)
- [Phase 01-04]: VLM benchmark requires HuggingFace auth for gated PaliGemma models; Florence-2 as non-gated fallback

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: iOS "limited photo access" may break passive scanning value prop -- Phase 2 must design for limited access as primary path
- [Research]: Training data bias causes 15-50% calorie errors for non-Western cuisines -- Phase 1 must audit training data by cuisine
- [Research]: CoreML/TFLite model conversion can fail silently -- Phase 1 must validate on-device outputs early
- [Research]: If YOLO accuracy benchmarks fail, Phase 1 pivots to LLM-primary detection -- this changes Phase 2 architecture (cloud pipeline vs on-device)
- [Research]: Multi-region database integration (Phase 3) requires evaluating API availability, data formats, and licensing for each regional database

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 01-04-PLAN.md (VLM Benchmark + Portion Estimation)
Resume file: None
