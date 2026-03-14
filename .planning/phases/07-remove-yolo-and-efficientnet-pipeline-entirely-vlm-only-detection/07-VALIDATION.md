---
phase: 7
slug: remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | jest-expo (Jest with Expo preset) |
| **Config file** | `apps/mobile/jest.config.js` |
| **Quick run command** | `cd apps/mobile && npx jest --testPathPattern="detection\|vlm\|store" --no-coverage` |
| **Full suite command** | `cd apps/mobile && npx jest --no-coverage` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd apps/mobile && npx jest --testPathPattern="detection|vlm|store" --no-coverage`
- **After every plan wave:** Run `cd apps/mobile && npx jest --no-coverage`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | P7-01 | unit | `cd apps/mobile && npx jest inferenceRouter.test.ts -x` | Rewrite needed | ⬜ pending |
| 07-01-02 | 01 | 1 | P7-02 | unit | `cd apps/mobile && npx jest modelLoader.test.ts -x` | Rewrite needed | ⬜ pending |
| 07-01-03 | 01 | 1 | P7-03 | unit | `cd apps/mobile && npx jest modelLoader.test.ts -x` | Rewrite needed | ⬜ pending |
| 07-01-04 | 01 | 1 | P7-04 | unit | `cd apps/mobile && npx jest imagePreprocess.test.ts -x` | Rewrite needed | ⬜ pending |
| 07-02-01 | 02 | 1 | P7-05 | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | Rewrite needed | ⬜ pending |
| 07-02-02 | 02 | 1 | P7-06 | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | New test | ⬜ pending |
| 07-02-03 | 02 | 1 | P7-07 | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | New test | ⬜ pending |
| 07-03-01 | 03 | 2 | P7-08 | unit | `cd apps/mobile && npx jest vlmComponents.test.tsx -x` | New test | ⬜ pending |
| 07-03-02 | 03 | 2 | P7-09 | unit | `cd apps/mobile && npx jest useDetectionStore.test.ts -x` | Update needed | ⬜ pending |
| 07-04-01 | 04 | 2 | P7-10 | manual-only | Verify file absence after deletion | N/A | ⬜ pending |
| 07-04-02 | 04 | 2 | P7-11 | manual-only | Build APK and compare size | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Rewrite `inferenceRouter.test.ts` — remove classify stage expectations, test bbox-only pipeline
- [ ] Rewrite `modelLoader.test.ts` — remove classify model expectations, test detect-only loading
- [ ] Rewrite `imagePreprocess.test.ts` — remove imagenet normalization tests
- [ ] Rewrite `vlmPipeline.test.ts` — change from "refinement" to "primary identification" expectations
- [ ] Create `vlmPipeline.test.ts` stubs for VLM retry and text fallback behavior (P7-06, P7-07)
- [ ] Create `vlmComponents.test.tsx` stubs for ShimmerPlaceholder (P7-08)
- [ ] Update `useDetectionStore.test.ts` for shimmer display state (P7-09)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| classify.tflite and labels_classify.json deleted | P7-10 | File absence verification | Check `apps/mobile/assets/models/` — files must not exist |
| APK size reduced by ~4.9MB | P7-11 | Build comparison | Build APK, compare to pre-phase baseline |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
