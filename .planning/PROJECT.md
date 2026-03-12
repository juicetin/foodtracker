# FoodTracker

## What This Is

A subscription-free, local-first AI food tracking app for iOS and Android. All inference runs on-device (YOLO for food detection, optional VLM for complex dishes, custom OCR for scale reading). Nutrition data is bundled, not queried from cloud APIs. No backend server required. Optional sync to Google Drive / iCloud for backup. Built for health-conscious fitness enthusiasts who want accurate macro tracking without friction or recurring costs.

## Core Value

Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, no subscription, just eat, photograph, and review.

## Current Milestone: v1.0 (Local-First Reset)

**Goal:** Ship a fully functional local-first food tracker with on-device ML inference, bundled nutrition data, local storage, and optional cloud sync — with zero subscription cost.

**Target features:**
- On-device food detection (YOLO pipeline + optional VLM)
- Bundled nutrition databases (USDA + regional)
- Local SQLite storage (op-sqlite)
- Gallery scanning & deduplication
- Food diary UI with edit/review
- Scale OCR (custom 7-segment model)
- Optional Google Drive / iCloud sync

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Food density lookup table for weight estimation — existing
- ✓ YOLO-based food detection POC notebook with end-to-end pipeline — existing
- ✓ Mobile app scaffold with navigation, state management (Zustand), photo picker — existing
- ✓ Multi-region food composition database configuration (AFCD, USDA, CoFID, CIQUAL, Open Food Facts) — existing
- ✓ Knowledge graph with SQLite schema, USDA/RecipeNLG seeding, mobile export — Phase 1 (01-02)
- ✓ Dataset acquisition with auto-labeling (Florence-2) and merged datasets — Phase 1 (01-01)
- ✓ VLM benchmark (PaliGemma 2 3B) + portion estimation module — Phase 1 (01-04)

### Active

<!-- Current scope. Building toward these. -->

- [ ] On-device food detection using YOLO (binary, detection, classification) via CoreML/LiteRT
- [ ] Tiered on-device VLM for complex dish identification (SmolVLM-256M / Moondream 0.5B / Gemma 3n E2B)
- [ ] Model export to CoreML (iOS) and LiteRT/TFLite (Android) with on-device validation
- [ ] Mobile ML integration (react-native-fast-tflite or LiteRT, inference router)
- [ ] Dish identification -> ingredient inference (hidden ingredients from known recipes)
- [ ] Bundled USDA FDC nutrition database as pre-populated SQLite (~50-80MB)
- [ ] Optional regional nutrition databases (AFCD, CoFID, CIQUAL) as downloadable packs
- [ ] Local data storage via op-sqlite for all user data (entries, recipes, history)
- [ ] On-device gallery scanning to discover food photos (background, periodic, manual)
- [ ] EXIF metadata extraction (timestamp, location, camera info) for meal context
- [ ] Intelligent photo deduplication (multiple angles, group into meal events)
- [ ] Scale/weight detection via custom 7-segment TFLite OCR model
- [ ] User-managed container weights for improved scale accuracy
- [ ] Per-ingredient macro and micro nutrient breakdown
- [ ] Daily food diary UI with per-meal and total macros
- [ ] Manual food search and add (fallback when AI detection fails)
- [ ] Editable meals — correct ingredients, portions, quantities after logging
- [ ] Saveable recipes with serving size adjustment and one-tap reuse
- [ ] Image retention linked to ingredients for later review and editing
- [ ] Configurable UX modes: zero-effort (default), confirm-only, guided-edit
- [ ] Configurable end-of-day summary notification
- [ ] Optional Google Drive sync (appDataFolder, cross-platform)
- [ ] Optional iCloud sync (iOS only)
- [ ] Play for On-Device AI integration for tiered model delivery by device capability
- [ ] Gemini Nano via AICore as opportunistic inference on supported devices

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- AI coaching/auto-adjust programs — user wants to track, not be told what to eat
- Social features — personal tracking tool, not a social network
- Meal planning/recommendations — track what you eat, not what you should eat
- Barcode scanning — may add later, but photo-first approach is the differentiator
- Web app — mobile-first, on-device ML makes web impractical
- Cloud backend server — local-first architecture per ADR-005; no Express, no Go, no PostgreSQL server
- Subscription/recurring costs — zero-cost guarantee is a core differentiator
- Health platform integration (Apple Health, Google Fit) — defer to post-v1.0
- Correlation graphs — defer to post-v1.0
- WebDAV sync — defer to post-v1.0 (P2 priority per ADR-005)

## Context

**Architecture pivot (ADR-005, 2026-03-12):** Pivoted from cloud backend (Express + PostgreSQL + Gemini API) to fully local-first. All inference on-device, bundled nutrition data, optional cloud sync. Eliminates server costs and subscription requirement. See `docs/adr/005-local-first-no-subscription-architecture.md`.

**Prior work carried forward:** YOLO training scripts, knowledge graph (SQLite + mobile export), VLM benchmark, portion estimator, dataset acquisition pipeline, React Native + Expo mobile scaffold.

**Prior work superseded:** Express.js backend, PostgreSQL schema, Google ADK agent structure (visionAgent, scaleAgent, coordinatorAgent). Code remains in repo but is not part of the local-first architecture.

**Competitive landscape:** MyFitnessPal (~$10/mo), MacroFactor (~$12/mo), Cronometer (~$10/mo). All require subscriptions. No subscription-free AI food tracker exists in the market.

**Research (2026-03-12):** 6 detailed research documents in `docs/research/001-006` covering Android fragmentation (~30-35% of devices have <=4GB RAM), on-device VLM feasibility (SmolVLM-256M at 100MB viable), Gemini Nano/AICore (foreground-only, battery quota), scale OCR (custom 7-segment model needed), app size impact (6MB = ~1% conversion drop), offline-first sync (PowerSync/WatermelonDB top picks, CRDTs overkill).

## Constraints

- **Accuracy first:** Detection and nutrition estimates must be trustworthy. If on-device accuracy is insufficient for a material user segment, a cloud fallback tier may be added (see ADR-005 conditional cloud fallback).
- **No backend server:** All core functionality runs on-device. No server to deploy/maintain/scale.
- **Zero subscription cost:** No recurring charges for users. No API billing. No server costs.
- **Base APK under 100MB:** Competitive with Cronometer (75-100MB). Nutrition DBs and VLMs delivered as post-install asset packs.
- **Mobile framework:** React Native + Expo. Open to native (Swift/Kotlin) if on-device ML performance demands it.
- **LiteRT over NNAPI:** NNAPI deprecated in Android 15. Use LiteRT with vendor NPU delegates.
- **Database coverage:** Nutrition database must cover global cuisines — not just Western food.
- **Device compatibility:** YOLO pipeline must work on devices from the last ~4 years. VLM is opt-in for devices with sufficient RAM.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Local-first, no-subscription architecture (ADR-005) | Zero recurring cost, privacy by default, offline-first, competitive differentiator | — Pending (pivot approved, implementation starting) |
| YOLO over multimodal LLMs for primary detection (ADR-003) | Cost ($0 on-device), speed (30ms), spatial accuracy (bounding boxes), privacy | ✓ Reinforced by ADR-005 |
| Tiered VLM delivery by device capability | SmolVLM-256M for budget, Moondream 0.5B mid-range, Gemma 3n flagship | — Pending |
| Custom 7-segment OCR for scale reading | ML Kit & Apple Vision explicitly don't support LCD displays; custom TFLite model (17KB-5MB) outperforms | — Pending |
| op-sqlite for local storage | 8-9x faster than alternatives; replaces PostgreSQL | — Pending |
| Bundled nutrition DBs (not runtime API) | USDA data is static, updates infrequently; eliminates API dependency (supersedes ADR-004) | — Pending |
| LWW conflict resolution for sync | Food logs are single-user, append-heavy; CRDTs overkill (Cinapse case study) | — Pending |
| Play for On-Device AI for model delivery | Device targeting by RAM/chipset; delta patching; 1.5GB per AI pack | — Pending |
| Python for ML training/POC only | ML ecosystem is Python-first; on-device serving via CoreML/LiteRT | ✓ Good |
| Gallery scanning over in-app camera | Removes friction — user photographs food naturally, app discovers later | — Pending |
| No Go backend | Superseded by local-first pivot (ADR-005). No server needed. | ✓ Decided |
| ADK removed as requirement | No cloud agent framework needed in local-first architecture | ✓ Decided |

---
*Last updated: 2026-03-12 after local-first architecture pivot (ADR-005)*
