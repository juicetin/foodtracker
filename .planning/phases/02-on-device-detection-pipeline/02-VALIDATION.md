---
phase: 2
slug: on-device-detection-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | jest-expo (jest + React Native) |
| **Config file** | apps/mobile/jest.config.js |
| **Quick run command** | `cd apps/mobile && npx jest --testPathPattern='detection' --no-coverage` |
| **Full suite command** | `cd apps/mobile && npx jest --no-coverage` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd apps/mobile && npx jest --testPathPattern='detection' --no-coverage`
- **After every plan wave:** Run `cd apps/mobile && npx jest --no-coverage`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DET-01 | unit | `cd apps/mobile && npx jest --testPathPattern='postProcess' -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DET-01 | unit | `cd apps/mobile && npx jest --testPathPattern='inferenceRouter' -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | DET-01 | unit | `cd apps/mobile && npx jest --testPathPattern='modelLoader' -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | DET-05 | unit | `cd apps/mobile && npx jest --testPathPattern='confidence' -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | DET-05 | unit | `cd apps/mobile && npx jest --testPathPattern='correctionStore' -x` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | DET-06 | unit | `cd apps/mobile && npx jest --testPathPattern='portionBridge' -x` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | DET-06 | unit | `cd apps/mobile && npx jest --testPathPattern='PortionSlider' -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | DET-01 | integration | `cd training && python train_detect.py --prepare-data --epochs 1 --device cpu` | ✅ | ⬜ pending |
| 02-01-05 | 01 | 1 | DET-01 | integration | `python -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='tflite', imgsz=640)"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `apps/mobile/src/services/detection/__tests__/postProcess.test.ts` — stubs for DET-01 YOLO output decoding + NMS
- [ ] `apps/mobile/src/services/detection/__tests__/inferenceRouter.test.ts` — stubs for DET-01 pipeline orchestration
- [ ] `apps/mobile/src/services/detection/__tests__/modelLoader.test.ts` — stubs for DET-01 model loading
- [ ] `apps/mobile/src/services/detection/__tests__/portionBridge.test.ts` — stubs for DET-06 TS port validation
- [ ] `apps/mobile/src/services/detection/__tests__/correctionStore.test.ts` — stubs for DET-05 correction history
- [ ] `training/export_mobile.py` — export script for TFLite models
- [ ] `apps/mobile/__mocks__/react-native-fast-tflite.ts` — mock for jest testing

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Bounding boxes drawn on photo with correct colors | DET-01, DET-05 | Visual rendering validation | Photograph food, verify colored boxes appear at correct positions with label chips |
| Pinch-to-zoom and pan on annotated photo | DET-01 | Gesture interaction | Open detection result, pinch-zoom and pan the annotated image |
| Inference completes within 2 seconds on mid-range device | DET-01 | Device-specific performance | Run detection on Pixel 6a / iPhone 12, verify spinner resolves within 2s |
| Bottom sheet detail card opens on tap | DET-05 | UI interaction | Tap bounding box or list item, verify detail card shows food name, confidence, portion, macros |
| Portion slider updates macros in real-time | DET-06 | UI interaction | Open detail card, drag portion slider, verify macros update smoothly |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
