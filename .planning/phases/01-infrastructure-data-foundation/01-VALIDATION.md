---
phase: 1
slug: infrastructure-data-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | jest-expo ^54.0.16 (installed, needs config) |
| **Config file** | None — Wave 0 creates `apps/mobile/jest.config.ts` |
| **Quick run command** | `cd apps/mobile && npx jest --testPathPattern='test_name' --no-coverage` |
| **Full suite command** | `cd apps/mobile && npx jest --no-coverage` |
| **Estimated runtime** | ~5 seconds (unit/integration, no device) |

---

## Sampling Rate

- **After every task commit:** Run `cd apps/mobile && npx jest --no-coverage`
- **After every plan wave:** Run `cd apps/mobile && npx jest --no-coverage && npx tsc --noEmit`
- **Before `/gsd:verify-work`:** Full suite must be green + both platforms compile (`npx expo prebuild --clean`)
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | DAT-01 | setup | `cd apps/mobile && npx jest --no-coverage` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | DAT-01 | unit | `npx jest --testPathPattern='db/schema' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | DAT-01 | unit | `npx jest --testPathPattern='db/migrations' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | DAT-01 | unit | `npx jest --testPathPattern='store/useFoodLogStore' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | DAT-01 | type check | `cd apps/mobile && npx tsc --noEmit` | N/A | ⬜ pending |
| 01-02-01 | 02 | 1 | DAT-02 | unit (Python) | `cd knowledge-graph && python -m pytest tests/test_usda_build.py -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | DAT-02 | integration | `npx jest --testPathPattern='services/nutrition' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | DAT-02 | unit | `npx jest --testPathPattern='services/packs' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 1 | DAT-03 | integration | `npx jest --testPathPattern='services/nutrition/regional' --no-coverage` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 1 | DAT-03 | unit | `npx jest --testPathPattern='services/packs/locale' --no-coverage` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `apps/mobile/jest.config.ts` — Jest configuration with op-sqlite mock setup
- [ ] `apps/mobile/__mocks__/@op-engineering/op-sqlite.ts` — Mock for unit tests (op-sqlite requires native runtime)
- [ ] `apps/mobile/src/db/__tests__/schema.test.ts` — Schema definition test stubs
- [ ] `apps/mobile/src/services/packs/__tests__/packManager.test.ts` — Pack download/cache test stubs
- [ ] `apps/mobile/src/services/nutrition/__tests__/nutritionService.test.ts` — Nutrition query test stubs
- [ ] Framework: jest-expo already in devDependencies; needs config file only

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| App builds on iOS device | DAT-01 | Requires Xcode + physical device/simulator | `npx expo prebuild --clean && npx expo run:ios` |
| App builds on Android device | DAT-01 | Requires Android Studio + emulator/device | `npx expo prebuild --clean && npx expo run:android` |
| Fast-follow pack downloads after install | DAT-02 | Requires network + first-launch flow | Install fresh build, observe download progress screen |
| Onboarding flow during download | DAT-02 | UI/UX verification | Install fresh, verify goals/units/region screens appear |
| Data & Storage settings screen | DAT-03 | UI verification | Navigate to Settings > Data & Storage, verify pack list |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
