# Phase 8: On-Device Vector Search Embedding via TFLite MiniLM - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-22
**Phase:** 08-on-device-vector-search-embedding-via-tflite-minilm
**Areas discussed:** Model export format, Tokenizer implementation, Model loading strategy, Embedding warmup timing
**Mode:** --auto (all decisions auto-selected based on codebase patterns and prior decisions)

---

## Model Export Format

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic range INT8 | Weights-only quantization, float32 I/O, ~11MB, matches existing models | [auto] |
| FP16 | Half-precision, ~22MB, slightly higher accuracy | |
| Full float32 | No quantization, ~44MB, maximum accuracy | |

**User's choice:** [auto] Dynamic range INT8
**Notes:** Matches existing classifier (EfficientNet-Lite0 INT8) and detector quantization patterns. Float32 I/O ensures compatibility with existing pipeline.

---

## Tokenizer Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| Inline JS (~200 lines) | Pure-JS WordPiece, no dependencies, vocab as JSON asset | [auto] |
| npm wordpiece package | External dependency, maintained by third party | |
| WASM-based tokenizer | Fast but adds native complexity | |

**User's choice:** [auto] Inline JS (~200 lines)
**Notes:** Matches SymSpell inline pattern (~230 lines). Food names are short strings — tokenizer performance is not a bottleneck.

---

## Model Loading Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Bundle in APK | ~11MB in assets/, zero first-run friction | [auto] |
| Download via pack system | Smaller APK, first-run download required | |
| Play AI Delivery | Native delivery, but adds AI pack config complexity | |

**User's choice:** [auto] Bundle in APK
**Notes:** 11MB is well under 100MB APK threshold. Matches food-knowledge.db bundling pattern. No download step needed.

---

## Embedding Warmup Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Lazy on first detection | Load model on first embed() call, no startup cost | [auto] |
| App boot | Pre-load at startup for instant first query | |
| Background after boot | Load after splash, before first use | |

**User's choice:** [auto] Lazy on first detection flow
**Notes:** Matches KG lazy init and VLM lazy init established patterns. Model load is fast (~100ms for 11MB TFLite).

---

## Claude's Discretion

- Token sequence length (128 vs 256)
- TFLite export script tooling
- Test strategy for embedding quality

## Deferred Ideas

- @huggingface/transformers when RN 0.82 ESM support ships
