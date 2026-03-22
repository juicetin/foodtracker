---
phase: 08-on-device-vector-search-embedding-via-tflite-minilm
plan: 02
subsystem: ml
tags: [tflite, wordpiece, embedding, minilm, react-native-fast-tflite]

requires:
  - phase: 08-01
    provides: embedding.tflite model and vocab_embedding.json vocabulary file
provides:
  - Pure-JS WordPiece tokenizer for BERT-family models
  - Working EmbeddingService with TFLite model loading via react-native-fast-tflite
  - Semantic USDA vector search path activation in vlmPipeline
affects: [vlmPipeline, knowledgeGraph, detection-flow]

tech-stack:
  added: []
  patterns: [WordPiece tokenizer in pure JS, singleton embedding service with lazy warmup]

key-files:
  created:
    - apps/mobile/src/services/embedding/wordpieceTokenizer.ts
    - apps/mobile/src/services/embedding/__tests__/wordpieceTokenizer.test.ts
    - apps/mobile/src/services/embedding/__tests__/embeddingService.test.ts
  modified:
    - apps/mobile/src/services/embedding/embeddingService.ts

key-decisions:
  - "Pure-JS WordPiece tokenizer (no native dependency) with [CLS]/[SEP] framing and Int32Array output"
  - "EmbeddingService loads vocab as Map<string, number> from JSON at warmup time"

patterns-established:
  - "WordPiece tokenizer pattern: lowercase -> punctuation split -> greedy longest-match subword with ## prefix"
  - "Embedding service pattern: singleton with lazy warmup, idempotent model loading, _resetForTesting for test isolation"

requirements-completed: [EMB-03, EMB-04, EMB-05]

duration: 3min
completed: 2026-03-22
---

# Phase 08 Plan 02: Tokenizer and Embedding Service Summary

**Pure-JS WordPiece tokenizer and TFLite-backed EmbeddingService producing Float32Array(384) embeddings for semantic USDA food search**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-22T09:23:10Z
- **Completed:** 2026-03-22T09:25:57Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- WordPiece tokenizer correctly tokenizes food names with CLS/SEP/padding/attention mask, punctuation splitting, and subword decomposition
- EmbeddingService loads bundled TFLite model and vocab on warmup(), returns Float32Array(384) embeddings
- vlmPipeline vec search path will automatically activate when EmbeddingService.warmup() is called during detection flow (no pipeline changes needed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement pure-JS WordPiece tokenizer with tests** - `1dc3d18c` (feat)
2. **Task 2: Implement EmbeddingService with TFLite model loading and tests** - `9f30ac66` (feat)

## Files Created/Modified
- `apps/mobile/src/services/embedding/wordpieceTokenizer.ts` - Pure-JS WordPiece tokenizer with CLS/SEP, attention mask, padding, subword splitting
- `apps/mobile/src/services/embedding/embeddingService.ts` - Singleton service loading TFLite model via react-native-fast-tflite, embed() returns Float32Array(384)
- `apps/mobile/src/services/embedding/__tests__/wordpieceTokenizer.test.ts` - 9 unit tests covering CLS/SEP, case, punctuation, subwords, truncation, types
- `apps/mobile/src/services/embedding/__tests__/embeddingService.test.ts` - 7 unit tests covering singleton, lifecycle, idempotent warmup, embedding output

## Decisions Made
- Pure-JS WordPiece tokenizer avoids native dependency -- runs in JS thread, no bridge overhead for text tokenization
- EmbeddingService builds vocab Map from JSON at warmup time (not module load) for lazy initialization
- Output handling covers both Float32Array and ArrayBuffer from model.run() for cross-platform compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Jest required `npm install` before running (dependencies not installed) -- resolved by running install
- npx jest `-x` flag not recognized by jest-expo preset -- used `--bail` instead

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Embedding pipeline complete: tokenizer + service + model + vocab all wired
- vlmPipeline already has vec search path that checks embSvc.ready -- will activate automatically
- No further phases depend on this plan

---
*Phase: 08-on-device-vector-search-embedding-via-tflite-minilm*
*Completed: 2026-03-22*
