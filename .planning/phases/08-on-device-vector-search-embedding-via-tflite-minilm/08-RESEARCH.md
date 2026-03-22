# Phase 08: On-Device Vector Search Embedding via TFLite MiniLM - Research

**Researched:** 2026-03-22
**Domain:** On-device text embedding (MiniLM-L6-v2 TFLite) + WordPiece tokenizer + react-native-fast-tflite inference
**Confidence:** HIGH

## Summary

This phase activates the existing vec search path (usda_embeddings + vec_distance_cosine) by implementing on-device query-time text embedding. All downstream infrastructure exists: the `EmbeddingService` singleton stub, `searchUsdaByVector()` in KG service, `vec_f32()` SQL queries, and the vlmPipeline integration points that already call `embSvc.embed()` and route to vec search when ready. The only missing piece is the embedding generation itself.

The approach is: (1) export all-MiniLM-L6-v2 to ONNX with mean pooling + L2 normalization baked into the graph, (2) convert to TFLite with dynamic range INT8 quantization via Docker-based onnx2tf (established project pattern), (3) implement a ~200-line pure-JS WordPiece tokenizer using the model's vocab.json, and (4) wire the TFLite model into the existing `EmbeddingService` stub using react-native-fast-tflite (already installed at v2.0.0).

**Primary recommendation:** Export a self-contained TFLite model that takes input_ids + attention_mask and outputs a single 384-dim normalized float32 vector. Do NOT rely on JS-side mean pooling -- bake pooling + L2 norm into the ONNX graph before TFLite conversion. This keeps the JS inference code to a simple `model.run([inputIds, attentionMask])` call.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Export MiniLM-L6-v2 (all-MiniLM-L6-v2 from sentence-transformers) as TFLite with dynamic range INT8 quantization -- matches existing classifier/detector quantization pattern
- **D-02:** Float32 I/O (input token IDs as int32, output 384-dim float32 vector) -- compatible with existing `vec_f32()` SQL function
- **D-03:** Normalize output embeddings to unit vectors (matches build_kg.py `normalize_embeddings=True`) so cosine distance = dot product
- **D-04:** Pure-JS WordPiece tokenizer (~200 lines), inline in the embedding service -- no external npm dependency
- **D-05:** Bundle WordPiece vocabulary as JSON asset (extracted from HuggingFace all-MiniLM-L6-v2 tokenizer)
- **D-06:** Tokenizer must handle: lowercasing, basic punctuation splitting, WordPiece subword splitting with ## prefixes, [CLS]/[SEP] special tokens, attention mask generation
- **D-07:** Bundle TFLite model in APK via assets/ (~11MB INT8, well under 100MB threshold) -- same pattern as food-knowledge.db
- **D-08:** Use react-native-fast-tflite (already installed) for inference -- same as detection pipeline
- **D-09:** Lazy initialization on first detection flow, not at app boot -- matches KG lazy init and VLM lazy init patterns
- **D-10:** `warmup()` loads TFLite model into memory; `embed()` returns null until warmup completes (existing stub pattern preserved)

### Claude's Discretion
- Token sequence length (128 vs 256 max tokens -- food names are short, 128 likely sufficient)
- Exact TFLite export script tooling (Python with optimum/onnx2tf/ai-edge-litert)
- Test strategy for embedding quality validation (cosine distance spot-checks vs comprehensive benchmark)

### Deferred Ideas (OUT OF SCOPE)
- @huggingface/transformers OOTB path when RN 0.82 Metro ESM lands -- check at each milestone
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| react-native-fast-tflite | 2.0.0 | TFLite inference runtime | Already installed, used by detection pipeline |
| sentence-transformers (Python) | latest | Source model for export | Official model host for all-MiniLM-L6-v2 |
| onnx2tf (Docker) | via tensorflow/tensorflow:2.18.0 | ONNX to TFLite conversion | Established project pattern (phases 02.3, 02.4) |
| optimum (Python) | latest | PyTorch to ONNX export | Official HuggingFace export tool |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnxruntime (Python) | latest | ONNX model validation | Verify ONNX output before TFLite conversion |
| numpy (Python) | latest | Embedding validation | Compare TFLite output vs sentence-transformers output |
| expo-asset | (installed) | Resolve bundled .tflite path | Same pattern as food-knowledge.db asset loading |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure-JS WordPiece | @xenova/transformers tokenizer | Adds ~500KB dep, ESM incompatible with Metro currently |
| Custom ONNX export | optimum-cli export | optimum-cli doesn't include pooling -- need wrapper |
| onnx2tf | ai-edge-litert/litert-torch | litert-torch is newer but less tested with BERT models |

**Installation:** No new npm packages needed. Python export dependencies installed in Docker/venv only.

## Architecture Patterns

### Recommended Project Structure
```
apps/mobile/
├── assets/
│   ├── models/
│   │   ├── detect.tflite          # existing
│   │   ├── embedding.tflite       # NEW: MiniLM INT8 (~11MB)
│   │   └── model_manifest.json    # updated with embedding entry
│   └── data/
│       ├── food-knowledge.db      # existing
│       └── vocab_embedding.json   # NEW: WordPiece vocab (232KB JSON)
├── src/services/embedding/
│   ├── embeddingService.ts        # REPLACE stub with real implementation
│   ├── wordpieceTokenizer.ts      # NEW: pure-JS tokenizer (~200 lines)
│   └── __tests__/
│       ├── wordpieceTokenizer.test.ts  # NEW
│       └── embeddingService.test.ts    # NEW
training/
└── export_embedding.py            # NEW: export script
```

### Pattern 1: PyTorch Wrapper for ONNX Export with Baked-in Pooling
**What:** Wrap the BERT model in a PyTorch nn.Module that performs mean pooling + L2 normalization, then export that wrapper to ONNX. This produces a single model with inputs (input_ids, attention_mask) and output (384-dim normalized embedding).
**When to use:** Always -- the alternative (JS-side pooling) adds complexity and error-prone tensor manipulation.
**Example:**
```python
# Source: community pattern from sentence-transformers ONNX export guides
import torch
import torch.nn as nn
from transformers import AutoModel

class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq, 384)
        # Mean pooling: average non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # (batch, 384)
        # L2 normalize to unit vector
        return torch.nn.functional.normalize(pooled, p=2, dim=1)
```

### Pattern 2: TFLite Model Loading via react-native-fast-tflite
**What:** Load bundled TFLite model using `require()` + `loadTensorflowModel()`, matching existing detection pipeline pattern.
**When to use:** For the embedding model, identical to modelLoader.ts pattern.
**Example:**
```typescript
// Same pattern as modelLoader.ts
import { loadTensorflowModel } from 'react-native-fast-tflite';

const BUNDLED_EMBEDDING = require('../../../assets/models/embedding.tflite');

// Load with default delegate
const model = await loadTensorflowModel(BUNDLED_EMBEDDING, 'default');

// Run inference: input_ids and attention_mask as Int32Array
const output = await model.run([inputIds, attentionMask]);
const embedding = output[0] instanceof Float32Array
  ? output[0]
  : new Float32Array(output[0] as ArrayBuffer);
```

### Pattern 3: WordPiece Tokenizer in Pure JS
**What:** Minimal tokenizer that lowercases, splits on punctuation, applies WordPiece subword splitting using the model's vocabulary.
**When to use:** For all text-to-token-ids conversion before embedding inference.
**Example:**
```typescript
// Simplified WordPiece tokenization flow
export function tokenize(text: string, vocab: Map<string, number>, maxLen: number): {
  inputIds: Int32Array;
  attentionMask: Int32Array;
} {
  const CLS = vocab.get('[CLS]')!;
  const SEP = vocab.get('[SEP]')!;
  const PAD = vocab.get('[PAD]')!;  // 0
  const UNK = vocab.get('[UNK]')!;

  // 1. Lowercase + basic tokenization (split on whitespace and punctuation)
  const words = text.toLowerCase().replace(/([.,!?;:'"()\[\]{}])/g, ' $1 ').trim().split(/\s+/);

  // 2. WordPiece subword splitting
  const tokens: number[] = [CLS];
  for (const word of words) {
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = false;
      while (start < end) {
        const substr = (start > 0 ? '##' : '') + word.slice(start, end);
        if (vocab.has(substr)) {
          tokens.push(vocab.get(substr)!);
          found = true;
          start = end;
          break;
        }
        end--;
      }
      if (!found) {
        tokens.push(UNK);
        start++;
      }
    }
    if (tokens.length >= maxLen - 1) break; // leave room for [SEP]
  }
  tokens.push(SEP);

  // 3. Pad to maxLen
  const inputIds = new Int32Array(maxLen);
  const attentionMask = new Int32Array(maxLen);
  for (let i = 0; i < tokens.length && i < maxLen; i++) {
    inputIds[i] = tokens[i];
    attentionMask[i] = 1;
  }
  // Remaining positions are already 0 (PAD) in Int32Array

  return { inputIds, attentionMask };
}
```

### Anti-Patterns to Avoid
- **JS-side mean pooling:** Do NOT compute mean pooling over token embeddings in JavaScript. The ONNX model should output a single 384-dim vector. JS-side pooling requires managing the attention mask over the full (seq_len, 384) output tensor, which is error-prone and wasteful.
- **Float32 model without quantization:** The float32 MiniLM is ~90MB. INT8 dynamic range quantization brings it to ~11MB with negligible quality loss for food name embedding.
- **Loading vocab.txt instead of JSON:** vocab.txt requires line-by-line parsing. Convert to JSON map `{"token": id}` at build time for instant `JSON.parse()` in the app.
- **Eager model loading at boot:** Embedding is needed only during food detection. Loading a TFLite model at app boot wastes startup time and memory for users who haven't started detecting yet.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX export with pooling | Manual graph surgery | PyTorch wrapper module + torch.onnx.export | Wrapper is ~20 lines; graph surgery is fragile |
| TFLite conversion | Manual TF SavedModel | Docker-based onnx2tf | Established project pattern, handles NCHW/NHWC |
| Vocabulary extraction | Manual download + parse | transformers AutoTokenizer + json.dump | Tokenizer config has special tokens, IDs, etc. |
| Embedding quality validation | Manual cosine distance calc | sentence-transformers + numpy comparison | Ground truth from same model that generated DB embeddings |

**Key insight:** The export pipeline is Python-only tooling that runs once. All complexity should be absorbed into the export script, keeping the mobile-side code (tokenizer + inference call) dead simple.

## Common Pitfalls

### Pitfall 1: ONNX Export Without Pooling Layer
**What goes wrong:** The default `optimum-cli export onnx --task feature-extraction` exports only the transformer, outputting (batch, seq_len, 384) token embeddings instead of (batch, 384) sentence embeddings.
**Why it happens:** Sentence-transformers applies pooling as a separate Python module, not part of the transformer graph.
**How to avoid:** Wrap the transformer in a PyTorch Module that includes mean pooling + L2 normalization before ONNX export.
**Warning signs:** TFLite output shape is [1, 128, 384] instead of [1, 384].

### Pitfall 2: Embedding Mismatch Between Build-time and Query-time
**What goes wrong:** Query embeddings don't match the embeddings stored in usda_embeddings, causing all cosine distances to be high (>0.5) and vec search to return no results.
**Why it happens:** Different tokenization, different pooling strategy, or missing L2 normalization. The build_kg.py uses `SentenceTransformer.encode(normalize_embeddings=True)` which applies WordPiece tokenization + mean pooling + L2 norm.
**How to avoid:** Validate the TFLite output against sentence-transformers output for identical inputs. Cosine similarity should be >0.99 for same text.
**Warning signs:** Vec search consistently returns null despite embedding service reporting ready=true.

### Pitfall 3: Token ID Data Type Mismatch
**What goes wrong:** react-native-fast-tflite may interpret Int32Array as float32 or vice versa, producing garbage embeddings.
**Why it happens:** TFLite models declare input tensor types (int32 for token IDs) but the JS-side TypedArray must match exactly.
**How to avoid:** Inspect the exported TFLite model in Netron to verify input tensor dtypes. Use Int32Array for input_ids and attention_mask (both int32 in the ONNX graph). Validate with the ai-edge-litert Python interpreter before mobile testing.
**Warning signs:** Embedding output is all zeros or NaN.

### Pitfall 4: Vocabulary Mismatch
**What goes wrong:** JS tokenizer produces different token IDs than the Python tokenizer, leading to different (and wrong) embeddings.
**Why it happens:** Vocab extraction missed special tokens, or the tokenizer doesn't handle accented characters / unicode normalization the same way.
**How to avoid:** Extract vocab from AutoTokenizer (which includes special tokens map). Test JS tokenizer output against Python `AutoTokenizer.encode()` for a set of food names.
**Warning signs:** Tokenization of simple words like "chicken" produces different IDs in JS vs Python.

### Pitfall 5: Sequence Length Exceeds Model Max
**What goes wrong:** Input longer than max_length causes TFLite inference to fail or produce wrong results.
**Why it happens:** Food descriptions can be multi-word ("grilled chicken breast with steamed broccoli").
**How to avoid:** Use max_length=128 (food names are short). Truncate at tokenization time, not at string level.
**Warning signs:** TFLite crash on longer inputs.

## Code Examples

### Export Script Pattern (Python)
```python
# training/export_embedding.py
# Source: Adapted from project pattern (export_ggcd_detect.py) + sentence-transformers docs

import torch
import torch.nn as nn
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

MAX_SEQ_LEN = 128

class SentenceEmbedder(nn.Module):
    """Wraps BERT with mean pooling + L2 norm for end-to-end sentence embedding."""
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tok_emb = out.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(tok_emb.size()).float()
        pooled = torch.sum(tok_emb * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

def export():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # 1. Export ONNX
    model = SentenceEmbedder(model_name)
    model.eval()
    dummy_ids = torch.randint(0, 30522, (1, MAX_SEQ_LEN), dtype=torch.int32)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.int32)
    torch.onnx.export(
        model, (dummy_ids, dummy_mask), "embedding.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes=None,  # fixed shape for TFLite
        opset_version=13,
    )

    # 2. Extract vocab as JSON
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()  # dict: token -> id
    with open("vocab_embedding.json", "w") as f:
        json.dump(vocab, f)

    # 3. Convert to TFLite via Docker (onnx2tf)
    # docker run -v $(pwd):/work tensorflow/tensorflow:2.18.0 bash -c \
    #   "pip install onnx2tf && onnx2tf -i /work/embedding.onnx -o /work/tflite_out \
    #    -oiqt"  # -oiqt = output INT8 quantized TFLite
```

### Embedding Service Implementation Pattern
```typescript
// apps/mobile/src/services/embedding/embeddingService.ts
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { tokenize } from './wordpieceTokenizer';

const BUNDLED_MODEL = require('../../../assets/models/embedding.tflite');
const vocabJson = require('../../../assets/data/vocab_embedding.json');

const MAX_SEQ_LEN = 128;

export class EmbeddingService {
  private static instance: EmbeddingService | null = null;
  private model: TensorflowModel | null = null;
  private vocab: Map<string, number> | null = null;

  private constructor() {}

  static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  async warmup(): Promise<void> {
    if (this.model) return;
    this.model = await loadTensorflowModel(BUNDLED_MODEL, 'default');
    this.vocab = new Map(Object.entries(vocabJson));
  }

  async embed(text: string): Promise<Float32Array | null> {
    if (!this.model || !this.vocab) return null;
    const { inputIds, attentionMask } = tokenize(text, this.vocab, MAX_SEQ_LEN);
    const output = await this.model.run([inputIds, attentionMask]);
    const raw = output[0];
    return raw instanceof Float32Array ? raw : new Float32Array(raw as ArrayBuffer);
  }

  get ready(): boolean {
    return this.model !== null;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No on-device embedding | TFLite MiniLM INT8 | This phase | Enables vec search for semantic food matching |
| @huggingface/transformers | TFLite + pure-JS tokenizer | 2026 (RN 0.81 lacks ESM) | Avoids Metro ESM blockers |
| onnx-tensorflow | onnx2tf | 2024+ | onnx2tf solves Transpose extrapolation issues |
| tflite-runtime (Python) | ai-edge-litert | 2025+ | tflite-runtime deprecated for Python 3.12+ |

**Deprecated/outdated:**
- `torch.onnx.export` with `torch.jit.script`: Legacy JIT tracing. Use standard eager mode export with opset 13+.
- `onnx-tf` (onnx-tensorflow): Replaced by onnx2tf for NCHW->NHWC conversion.
- `tflite-runtime`: Replaced by `ai-edge-litert` for Python 3.12+.

## Discretion Recommendations

### Token Sequence Length: 128 (recommended)
Food names are short. The longest realistic query is something like "grilled chicken breast with steamed broccoli and brown rice" which tokenizes to ~15-20 tokens. Max 128 tokens provides enormous headroom while keeping the input tensor small (128 * 4 bytes * 2 inputs = 1KB). Using 256 would double memory and inference time for zero benefit.

### Export Script Tooling: PyTorch wrapper + onnx2tf Docker
Follow the established project pattern from export_ggcd_detect.py. Use a PyTorch wrapper that includes mean pooling + L2 normalization, export to ONNX with opset 13, then convert via Docker-based onnx2tf with INT8 dynamic range quantization. Validate with ai-edge-litert Python interpreter before mobile deployment.

### Test Strategy: Cosine distance spot-checks (recommended)
Create a test set of ~20 food name pairs with known semantic relationships:
- Near-identical: ("chicken breast", "grilled chicken breast") -- expect cos_dist < 0.15
- Semantically similar: ("tonkatsu", "pork cutlet") -- expect cos_dist < 0.30
- Dissimilar: ("chicken breast", "chocolate cake") -- expect cos_dist > 0.50
Run these against both sentence-transformers Python output and TFLite output. Differences > 0.05 indicate a conversion problem.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Jest (existing) |
| Config file | apps/mobile/jest.config.js |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern=embedding -x` |
| Full suite command | `cd apps/mobile && npx jest` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| (no formal ID) | WordPiece tokenizer produces correct token IDs | unit | `npx jest wordpieceTokenizer.test -x` | Wave 0 |
| (no formal ID) | EmbeddingService.embed() returns Float32Array(384) | unit | `npx jest embeddingService.test -x` | Wave 0 |
| (no formal ID) | EmbeddingService.ready is false before warmup | unit | `npx jest embeddingService.test -x` | Wave 0 |
| (no formal ID) | TFLite output matches sentence-transformers output | manual | Python validation script | N/A |
| (no formal ID) | Vec search returns results when embedding is ready | integration | Emulator test | N/A |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern=embedding -x`
- **Per wave merge:** `cd apps/mobile && npx jest`
- **Phase gate:** Full suite green + emulator vec search validation

### Wave 0 Gaps
- [ ] `apps/mobile/src/services/embedding/__tests__/wordpieceTokenizer.test.ts` -- tokenizer correctness
- [ ] `apps/mobile/src/services/embedding/__tests__/embeddingService.test.ts` -- service lifecycle + mock model

## Open Questions

1. **TFLite INT8 model exact size**
   - What we know: Float32 MiniLM is ~90MB, INT8 should be ~22MB (4x reduction), community reports ~11MB for quantized version
   - What's unclear: Exact size after onnx2tf conversion with dynamic range quantization and baked-in pooling layer
   - Recommendation: Export and measure. If >20MB, consider weight-only quantization without the pooling layer weights (which are minimal)

2. **react-native-fast-tflite Int32Array input support**
   - What we know: The library accepts TypedArray[] as input to model.run(). Detection uses Float32Array. Text models need Int32Array for token IDs.
   - What's unclear: Whether react-native-fast-tflite handles Int32Array correctly or needs conversion to Float32Array
   - Recommendation: Test early in implementation. If Int32Array fails, cast token IDs to Float32Array (TFLite can accept float inputs for int32 tensors with implicit cast).

3. **vocab.json vs vocab.txt loading in RN**
   - What we know: HuggingFace provides vocab.txt (one token per line, 30522 lines, 232KB). Metro can bundle JSON via require().
   - What's unclear: Whether require() on a 232KB JSON will block the JS thread noticeably
   - Recommendation: Use JSON format with require(). 232KB parses in <5ms on modern phones. Load during warmup() alongside model.

## Sources

### Primary (HIGH confidence)
- HuggingFace model card: sentence-transformers/all-MiniLM-L6-v2 -- model architecture, vocab size (30522), embedding dim (384), max seq len (256, truncated from 512)
- Existing codebase: embeddingService.ts stub, vlmPipeline.ts integration points, knowledgeGraphService.ts searchUsdaByVector, build_kg.py seed_usda_embeddings
- Existing codebase: modelLoader.ts, inferenceRouter.ts -- established react-native-fast-tflite usage patterns
- Existing codebase: export_ggcd_detect.py -- Docker-based onnx2tf conversion pattern

### Secondary (MEDIUM confidence)
- [Nihal2000/all-MiniLM-L6-v2-quant.tflite](https://huggingface.co/Nihal2000/all-MiniLM-L6-v2-quant.tflite) -- community TFLite conversion confirms: inputs are input_ids + attention_mask, output is 384-dim, INT8 quantization works
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite) -- API: loadTensorflowModel, model.run([TypedArray[]])
- [ONNX sentence-transformers export guide](https://medium.com/@transformergpt/how-to-convert-sentence-transformer-pytorch-models-to-onnx-with-the-right-pooling-method-61b1c83515d2) -- ONNX export only includes transformer, pooling must be added separately
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf) -- ONNX to TFLite conversion tool, Docker usage

### Tertiary (LOW confidence)
- [react-native-fast-tflite guide 2025](https://javascript.plainenglish.io/react-native-fast-tflite-on-device-machine-learning-guide-2025-906b1a8181b1) -- general patterns for text model inference with RN fast tflite

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already installed/used in project, conversion tools well-established
- Architecture: HIGH - pattern directly follows existing detection pipeline, stub API already defined
- Pitfalls: HIGH - ONNX pooling gap is well-documented, embedding mismatch is verifiable, dtype issues are testable
- Export pipeline: MEDIUM - Docker onnx2tf works for YOLO but BERT graph is more complex (attention ops). May need opset tweaks.

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable domain, no fast-moving dependencies)
