# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, no subscription, just eat, photograph, and review.
**Current focus:** Defining requirements (local-first architecture reset)

## Current Position

**Phase:** Not started (defining requirements)
**Plan:** —
**Status:** Defining requirements
**Last Activity:** 2026-03-12 — Milestone v1.0 reset for local-first architecture (ADR-005)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 3 (carried from pre-pivot Phase 1)
- Average duration: 21min
- Total execution time: 1.1 hours

**Previous Phase 1 (carried forward):**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 6min | 2 tasks | 11 files |
| Phase 01 P02 | 45min | 3 tasks | 8 files |
| Phase 01 P04 | 13min | 2 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [ADR-005]: Local-first, no-subscription architecture — all inference on-device, bundled nutrition data, optional cloud sync
- [ADR-005]: Tiered VLM delivery — SmolVLM-256M (budget), Moondream 0.5B (mid-range), Gemma 3n E2B (flagship)
- [ADR-005]: Custom 7-segment TFLite OCR for scale reading — ML Kit/Apple Vision unsupported for LCD displays
- [ADR-005]: op-sqlite replaces PostgreSQL; bundled USDA replaces runtime API
- [ADR-005]: LWW conflict resolution for sync; CRDTs overkill for single-user food logs
- [ADR-005]: Conditional cloud fallback — may add if device accuracy gaps are material and affected segment justifies complexity
- [Carried]: Risk-retirement ordering — validate ML accuracy before building UX on top
- [Carried]: Florence-2-base (0.23B) for auto-labeling with full-image fallback
- [Carried]: Programmatic dish generation (1003 dishes) instead of RecipeNLG download
- [Carried]: DELETE journal mode for mobile SQLite export (single file, no WAL/SHM)
- [Carried]: Built-in standard serving sizes preferred over KG fuzzy matching

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: ~30-35% of active Android devices have <=4GB RAM; 2026 DRAM shortage worsening this — tiered model delivery is critical
- [Research]: Thermal throttling at ~2.5min sustained inference, up to 4.3x degradation — batch processing needs bursty pattern
- [Research]: iOS has ~30-50% performance advantage over Android for VLM inference — test on both platforms early
- [Research]: Gemini Nano foreground-only restriction blocks background gallery scanning inference
- [Research]: iOS "limited photo access" may break passive scanning value prop — must design for limited access as primary path
- [Research]: Training data bias causes 15-50% calorie errors for non-Western cuisines — audit training data by cuisine
- [Research]: CoreML/LiteRT model conversion can fail silently — validate on-device outputs early
- [Research]: Base APK must stay under 100MB for emerging market viability (6MB = ~1% conversion drop)

## Session Continuity

Last session: 2026-03-12
Stopped at: Milestone v1.0 reset — defining requirements for local-first architecture
Resume file: None
