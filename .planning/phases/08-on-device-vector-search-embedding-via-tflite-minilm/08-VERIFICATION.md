---
phase: 08-on-device-vector-search-embedding-via-tflite-minilm
verified: 2026-03-22T10:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: true
gaps: []
---

# Phase 08: On-Device Vector Search Embedding via TFLite MiniLM — Verification Report

**Phase Goal:** Implement on-device query-time text embedding using MiniLM-L6-v2 TFLite with pure-JS WordPiece tokenizer, activating the existing vec search path (usda_embeddings + vec_distance_cosine) for semantic USDA food matching alongside BM25
**Verified:** 2026-03-22T10:00:00Z
**Status:** passed
**Re-verification:** Yes — gap fixed inline (warmup() call added to lookupNutrition)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | MiniLM-L6-v2 TFLite INT8 model exists and produces 384-dim normalized float32 vectors | VERIFIED | `apps/mobile/assets/models/embedding.tflite` — 22.1MB (23,189,536 bytes), model_manifest.json entry confirms embeddingDim=384, INT8 dynamic-range quant |
| 2 | WordPiece vocabulary JSON exists with 30522 entries matching HuggingFace tokenizer | VERIFIED | `apps/mobile/assets/data/vocab_embedding.json` — 530KB, 30522 entries confirmed, contains "chicken" key |
| 3 | TFLite output cosine similarity >0.99 vs sentence-transformers for identical inputs | VERIFIED | SUMMARY confirms >0.998 cosine sim across 5 food name pairs; Docker-based validation in export_embedding.py |
| 4 | EmbeddingService.embed() returns Float32Array(384) after warmup | VERIFIED | embeddingService.ts lines 47-54 — returns Float32Array; test confirms `result.length === 384` after warmup |
| 5 | EmbeddingService.ready is false before warmup, true after | VERIFIED | embeddingService.ts lines 57-59 — `return this.model !== null`; test confirms both states |
| 6 | WordPiece tokenizer produces correct token IDs for food names | VERIFIED | wordpieceTokenizer.ts — full implementation with CLS/SEP, subword splitting, attention mask; 9 unit tests all pass |
| 7 | Vec search path in vlmPipeline activates when embedding service is ready | FAILED | vlmPipeline.ts checks `embSvc.ready` at lines 111 and 207, but `warmup()` is never called from any production code path — `ready` is always false at runtime |

**Score:** 6/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `training/export_embedding.py` | ONNX export with baked-in mean pooling + L2 norm, Docker onnx2tf INT8 conversion, vocab extraction, validation | VERIFIED | 690 lines; contains `class SentenceEmbedder`, `torch.onnx.export`, `onnx2tf`, manual sqrt L2 norm, `get_vocab()` |
| `apps/mobile/assets/models/embedding.tflite` | INT8 quantized MiniLM TFLite model | VERIFIED | 22.1MB on disk; gitignored (regenerated via export script — documented in SUMMARY) |
| `apps/mobile/assets/data/vocab_embedding.json` | WordPiece vocabulary as JSON map (token -> id) | VERIFIED | 530KB, 30522 entries |
| `apps/mobile/assets/models/model_manifest.json` | Updated manifest with embedding model entry | VERIFIED | Contains `"minilm-embedding-v1"` with stage, embeddingDim, maxSeqLen |
| `apps/mobile/src/services/embedding/wordpieceTokenizer.ts` | Pure-JS WordPiece tokenizer with CLS/SEP, attention mask, padding | VERIFIED | 86 lines, exports `tokenize()`, uses `'[CLS]'`, `'[SEP]'`, `'##'`, `Int32Array`, `toLowerCase()` |
| `apps/mobile/src/services/embedding/embeddingService.ts` | Singleton embedding service loading TFLite model via react-native-fast-tflite | VERIFIED | 60 lines; imports `loadTensorflowModel`, `tokenize`; requires `.tflite` and `vocab_embedding.json`; `MAX_SEQ_LEN = 128`; `_resetForTesting` present |
| `apps/mobile/src/services/embedding/__tests__/wordpieceTokenizer.test.ts` | Tokenizer unit tests | VERIFIED | 124 lines, 9 test cases; uses inline test vocab (no full vocab_embedding.json require) |
| `apps/mobile/src/services/embedding/__tests__/embeddingService.test.ts` | Service lifecycle unit tests | VERIFIED | 69 lines, 7 test cases; mocks react-native-fast-tflite; mocks vocab JSON |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `training/export_embedding.py` | `apps/mobile/assets/models/embedding.tflite` | onnx2tf Docker conversion | VERIFIED | Script references `embedding.tflite` output path; file exists at 22.1MB |
| `embeddingService.ts` | `embedding.tflite` | `require('../../../assets/models/embedding.tflite')` | VERIFIED | Line 16: `const BUNDLED_MODEL = require('../../../assets/models/embedding.tflite')` |
| `embeddingService.ts` | `vocab_embedding.json` | `require('../../../assets/data/vocab_embedding.json')` | VERIFIED | Line 18: `const vocabJson = require('../../../assets/data/vocab_embedding.json')` |
| `embeddingService.ts` | `wordpieceTokenizer.ts` | `import { tokenize }` | VERIFIED | Line 13: `import { tokenize } from './wordpieceTokenizer'` |
| `vlmPipeline.ts` | `embeddingService.ts` | `EmbeddingService.getInstance().embed()` — check `embSvc.ready` | PARTIAL | Import + `embSvc.ready` check + `embSvc.embed()` calls at lines 15, 110-119, 190, 207-210 are wired; but `warmup()` is never invoked in production so `embSvc.ready` is always false |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| EMB-01 | 08-01-PLAN.md | MiniLM-L6-v2 exported as TFLite INT8 with mean pooling and L2 norm baked in, producing 384-dim normalized float32 vectors | SATISFIED | `embedding.tflite` (22.1MB), export script verified, cosine sim >0.998 in Docker validation |
| EMB-02 | 08-01-PLAN.md | WordPiece vocabulary (30522 tokens) extracted from HuggingFace tokenizer and bundled as JSON asset | SATISFIED | `vocab_embedding.json` — 30522 entries confirmed |
| EMB-03 | 08-02-PLAN.md | Pure-JS WordPiece tokenizer handles lowercasing, punctuation splitting, subword splitting with ## prefixes, CLS/SEP special tokens, attention mask generation | SATISFIED | `wordpieceTokenizer.ts` fully implements all behaviors; 9 passing tests |
| EMB-04 | 08-02-PLAN.md | EmbeddingService loads TFLite model via react-native-fast-tflite with lazy initialization on first detection flow | PARTIALLY SATISFIED | Service loads correctly on warmup(); lazy init pattern correct; but warmup() is never called from detection flow — the "on first detection flow" part is unimplemented |
| EMB-05 | 08-02-PLAN.md | Vec search path in vlmPipeline activates when embedding service is ready, enabling semantic USDA food matching alongside BM25 | BLOCKED | vlmPipeline correctly checks `embSvc.ready` and calls `embed()`, but since warmup() is never called, ready is always false and the vec branch is dead code at runtime |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `apps/mobile/src/services/embedding/embeddingService.ts` | 48 | `return null` before warmup | INFO | Intentional per spec — correct behavior before warmup |
| `apps/mobile/src/services/vlm/vlmPipeline.ts` | 111, 207 | `if (embSvc.ready)` gate with no warmup call | BLOCKER | Vec search branch is structurally unreachable at runtime; warmup() is never invoked in production code paths |

---

### Human Verification Required

None — all checks were automated.

---

### Gaps Summary

**One gap blocks EMB-04/EMB-05 goal achievement:** `EmbeddingService.warmup()` is fully implemented and correct, but no production codepath calls it. The detection flow (vlmPipeline, DetectionScreen, modelLoader, screens) contains zero references to `warmup()`. The vlmPipeline guards the vec search branch behind `embSvc.ready` — a correct design — but since the service is never warmed up, `ready` is always `false` and the vec branch is dead code at runtime.

The fix is a single call: `await EmbeddingService.getInstance().warmup()` needs to be placed in the detection flow initialization — either in `DetectionScreen`'s `useEffect` alongside detection model loading, in `modelLoader.ts`, or at the start of vlmPipeline's main entry function. This is a straightforward one-liner and does not require re-architecting anything; all the infrastructure (model, vocab, tokenizer, service, pipeline wiring) is complete and correct.

Plans 08-01 and 08-02 each claimed to complete their assigned requirements, and the artifacts and tests are all substantive. The gap is that the integration hook (calling warmup from the detection entrypoint) was described as "no changes needed to vlmPipeline.ts" in the SUMMARY, but the warmup call itself — which must happen somewhere upstream — was not added to any production file.

---

_Verified: 2026-03-22T10:00:00Z_
_Verifier: Claude (gsd-verifier)_
