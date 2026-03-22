---
phase: 08
slug: on-device-vector-search-embedding-via-tflite-minilm
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-22
---

# Phase 08 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | jest 29.x |
| **Config file** | apps/mobile/jest.config.js |
| **Quick run command** | `cd apps/mobile && npx jest --testPathPattern='embedding' --no-coverage` |
| **Full suite command** | `cd apps/mobile && npx jest --no-coverage` |
| **Estimated runtime** | ~15 seconds (embedding tests only) |

---

## Sampling Rate

- **After every task commit:** Run `cd apps/mobile && npx jest --testPathPattern='embedding' --no-coverage`
- **After every plan wave:** Run `cd apps/mobile && npx jest --no-coverage`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | Export TFLite | integration | `python3 -c "import tflite_runtime; ..."` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | WordPiece tokenizer | unit | `npx jest --testPathPattern='wordpiece'` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | EmbeddingService impl | unit | `npx jest --testPathPattern='embeddingService'` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 1 | Vec search E2E | integration | Manual — requires device with sqlite-vec | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `apps/mobile/src/services/embedding/__tests__/wordpieceTokenizer.test.ts` — unit tests for tokenizer
- [ ] `apps/mobile/src/services/embedding/__tests__/embeddingService.test.ts` — unit tests for embed/warmup/ready

*Existing jest infrastructure covers test framework — no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vec search returns better USDA match than BM25 for non-English food names | Semantic search quality | Requires device with sqlite-vec + food-knowledge.db | Scan "tonkatsu", verify vec returns "pork loin" vs BM25 returning nothing |
| TFLite model loads on Android | Device runtime | Emulator needed | Install APK, trigger detection, check logs for embedding warmup |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
