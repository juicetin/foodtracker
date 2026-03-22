# Phase 8: On-Device Vector Search Embedding via TFLite MiniLM - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement on-device query-time text embedding using MiniLM-L6-v2 exported as TFLite, with a pure-JS WordPiece tokenizer. This activates the existing vec search path (usda_embeddings + vec_distance_cosine) alongside BM25 in the USDA nutrition lookup chain. The vec index, SQL queries, KG service method, and VLM pipeline integration point all exist — only the embedding generation at query time is missing.

</domain>

<decisions>
## Implementation Decisions

### Model Export Format
- **D-01:** Export MiniLM-L6-v2 (all-MiniLM-L6-v2 from sentence-transformers) as TFLite with dynamic range INT8 quantization — matches existing classifier/detector quantization pattern
- **D-02:** Float32 I/O (input token IDs as int32, output 384-dim float32 vector) — compatible with existing `vec_f32()` SQL function
- **D-03:** Normalize output embeddings to unit vectors (matches build_kg.py `normalize_embeddings=True`) so cosine distance = dot product

### Tokenizer Implementation
- **D-04:** Pure-JS WordPiece tokenizer (~200 lines), inline in the embedding service — no external npm dependency
- **D-05:** Bundle WordPiece vocabulary as JSON asset (extracted from HuggingFace all-MiniLM-L6-v2 tokenizer)
- **D-06:** Tokenizer must handle: lowercasing, basic punctuation splitting, WordPiece subword splitting with ## prefixes, [CLS]/[SEP] special tokens, attention mask generation

### Model Loading Strategy
- **D-07:** Bundle TFLite model in APK via assets/ (~11MB INT8, well under 100MB threshold) — same pattern as food-knowledge.db
- **D-08:** Use react-native-fast-tflite (already installed) for inference — same as detection pipeline

### Embedding Warmup Timing
- **D-09:** Lazy initialization on first detection flow, not at app boot — matches KG lazy init and VLM lazy init patterns
- **D-10:** `warmup()` loads TFLite model into memory; `embed()` returns null until warmup completes (existing stub pattern preserved)

### Claude's Discretion
- Token sequence length (128 vs 256 max tokens — food names are short, 128 likely sufficient)
- Exact TFLite export script tooling (Python with optimum/onnx2tf/ai-edge-litert)
- Test strategy for embedding quality validation (cosine distance spot-checks vs comprehensive benchmark)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Embedding Infrastructure (existing)
- `apps/mobile/src/services/embedding/embeddingService.ts` — Stub service with singleton pattern, embed() returns null, ready=false
- `apps/mobile/src/services/knowledge-graph/knowledgeGraphSchema.ts` §SQL_SEARCH_USDA_VEC — Vec search SQL using vec_distance_cosine(), MAX_VEC_DISTANCE=0.50
- `apps/mobile/src/services/knowledge-graph/knowledgeGraphService.ts` §searchUsdaByVector — Accepts Float32Array(384), returns MacroResult

### Pipeline Integration Point
- `apps/mobile/src/services/vlm/vlmPipeline.ts` §lookupUsdaNutrition (lines 108-114) — Already calls EmbeddingService.embed() and routes to searchUsdaByVector when ready

### Build-time Embedding Generation
- `knowledge-graph/build_kg.py` §seed_usda_embeddings — MiniLM-L6-v2 with normalize_embeddings=True, 384-dim float32, little-endian blobs
- `knowledge-graph/schema.sql` — usda_embeddings table schema

### TFLite Patterns
- `apps/mobile/src/services/detection/modelLoader.ts` — Existing TFLite model loading via react-native-fast-tflite
- `apps/mobile/assets/models/model_manifest.json` — Model manifest for bundled assets

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EmbeddingService` singleton stub: warmup/embed/ready interface already defined — just needs implementation
- `react-native-fast-tflite`: already installed and configured, used by detection pipeline
- `modelLoader.ts`: existing pattern for loading TFLite models from assets
- `expo-asset Asset.fromModule()`: used for food-knowledge.db, same pattern for .tflite model

### Established Patterns
- Dynamic range INT8 quantization with float32 I/O (classifier, detector)
- Lazy service initialization on first use (KG, VLM)
- Singleton service pattern (EmbeddingService, KnowledgeGraphService)
- Bundled assets in assets/ directory for sub-100MB files

### Integration Points
- `vlmPipeline.ts` line 110-114: checks `embSvc.ready`, calls `embSvc.embed()`, passes result to `kg.searchUsdaByVector()`
- No code changes needed in vlmPipeline or KG service — just implement the stub

</code_context>

<specifics>
## Specific Ideas

- User explicitly chose TFLite path over waiting for RN 0.82 ESM support to avoid being blocked
- Future note saved: when RN 0.82 ships, evaluate @huggingface/transformers as potential replacement for this custom path

</specifics>

<deferred>
## Deferred Ideas

- @huggingface/transformers OOTB path when RN 0.82 Metro ESM lands — check at each milestone
- None other — discussion stayed within phase scope

</deferred>

---

*Phase: 08-on-device-vector-search-embedding-via-tflite-minilm*
*Context gathered: 2026-03-22*
