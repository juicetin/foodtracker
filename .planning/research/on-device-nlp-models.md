# On-Device NLP & Multimodal Models for Mobile Food Tracking

**Research date:** 2026-03-14
**Constraint:** ALL compute must be on-device (no cloud APIs)

---

## 1. On-Device LLMs/SLMs for Mobile

### Model Landscape (as of early 2026)

| Model | Params | Quantized Size | RAM Needed | Decode Speed | Notes |
|-------|--------|---------------|------------|--------------|-------|
| **Gemma 3n E2B** | 5B (2B active) | ~1 GB | ~2 GB | Fast (2x prefill vs Gemma 3 4B) | Purpose-built for mobile. PLE (Per Layer Embeddings) keeps only 2B params on accelerator. Vision + audio built in. |
| **Gemma 3n E4B** | 8B (4B active) | ~1.5 GB | ~3 GB | Good | Best quality at mobile scale. First sub-10B model above 1300 LMArena Elo. MobileNet-V5 vision encoder. |
| **Gemma 3 1B** | 1B | 529 MB | ~1 GB | 2,585 tok/s prefill (mobile GPU via LiteRT) | Text-only. Extremely fast. Good for simple NLP tasks. |
| **Llama 3.2 1B** | 1.24B | 1.1 GiB (SpinQuant) | 1.9 GiB peak | 350+ tok/s prefill, 40+ tok/s decode (Samsung S24+) | Mature ecosystem. QAT + SpinQuant quantization. 128K context. |
| **Llama 3.2 3B** | 3.21B | ~2 GiB | ~3 GiB | ~15-20 tok/s decode (est.) | Better quality, but heavier. Outperforms Gemma 2 2.6B and Phi-3.5 mini. |
| **Phi-4 Mini** | 3.8B | ~2 GiB | ~3-4 GiB | Good | Strong reasoning and multilingual. Comparable to 7-9B models. |
| **SmolLM2 1.7B** | 1.7B | ~1 GiB | ~1.5 GiB | Good | Trained on 11T tokens. Outperforms Llama 3.2 3B at 3B scale (SmolLM3). |
| **SmolLM3 3B** | 3B | ~1.5 GiB | ~2.5 GiB | Good | Outperforms Llama-3.2-3B and Qwen2.5-3B across 12 benchmarks. |
| **MobileLLM 125M** | 125M | ~60 MB | ~200 MB | 50 tok/s on iPhone | Meta's ultra-small model. Very fast but limited capability. |

### Can These Do Food Entity Extraction?

**Yes, with caveats.** Any instruction-tuned SLM (1B+) can handle structured extraction from prompts like:

```
Extract food items from: "I had a bowl of ramen with extra chashu and a soft-boiled egg"
Output JSON: {"foods": ["ramen", "chashu", "soft-boiled egg"]}
```

**Practical quality tiers:**
- **1B models** (Gemma 3 1B, Llama 3.2 1B): Can do basic extraction but may miss nuances, misparse quantities, or hallucinate
- **2-3B models** (Gemma 3n E2B, Llama 3.2 3B, SmolLM3): Reliable for food entity extraction with structured prompting
- **4B models** (Gemma 3n E4B, Phi-4 Mini): Near-cloud quality for NLP tasks; can handle complex descriptions, quantities, cooking methods

**Key concern:** Latency. Even at 40 tok/s decode, generating a structured JSON response of ~50 tokens takes >1 second. For a food tracking app, this is acceptable but not instant.

### Memory/Battery Constraints

- Modern phones (2025-2026) have 6-12 GB RAM, but only ~3-4 GB is typically available to a single app
- 4-bit quantization reduces model size ~4x with 1-3% quality drop
- Battery: mobile inference should be "burst when needed, sleep otherwise" -- not sustained
- Decode is memory-bandwidth bound: mobile has 50-90 GB/s vs data center GPUs at 2-3 TB/s

### Recommendation for Food Tracking

**Primary: Gemma 3n E2B** -- purpose-built for mobile, 2 GB RAM, multimodal (vision + audio + text), excellent quality. This is the Google-blessed on-device model.

**Fallback: Llama 3.2 1B** -- smaller, faster, well-tested with ExecuTorch and llama.cpp, mature React Native bindings.

---

## 2. On-Device Multimodal Models (Text + Image)

### Models That Can Take Image + Text and Reason About Food

| Model | Params | RAM | Capabilities | Mobile-Ready? |
|-------|--------|-----|--------------|---------------|
| **Gemma 3n E4B** | 8B (4B active) | 3 GB | Text + image + audio + video. MobileNet-V5-300M vision encoder. Up to 60 FPS on Pixel. | **Yes** -- purpose-built |
| **Gemma 3n E2B** | 5B (2B active) | 2 GB | Same multimodal capabilities as E4B, lower quality | **Yes** -- purpose-built |
| **SmolVLM-256M** | 300M | <1 GB GPU | Image captioning, VQA, OCR, document understanding. 93M vision encoder. | **Yes** -- smallest VLM in the world |
| **SmolVLM-500M** | 500M | ~1 GB | Better accuracy than 256M across all benchmarks | **Yes** |
| **SmolVLM 2.2B** | 2.2B | ~2 GB | Strong multimodal performance. Video support. | **Yes** with quantization |
| **Moondream 0.5B** | 500M | 816 MiB (4-bit) / 996 MiB (8-bit) | VQA, captioning, object detection, OCR. SigLIP encoder. | **Yes** |
| **Moondream2** | 1.86B | ~2 GB | Better quality. SigLIP + Phi-1.5. 55+ vision-language tasks. | **Yes** with quantization |
| **MobileVLM V2 1.7B** | 1.7B | ~1.5 GB | Purpose-built for mobile. Outperforms 3B VLMs. | **Yes** |
| **MobileVLM V2 3B** | 3B | ~2.5 GB | Outperforms many 7B+ VLMs. | **Tight** on older phones |
| **PaliGemma 2 3B** | 3B | ~3 GB | SigLIP encoder + Gemma decoder. 100+ languages. Strong captioning/VQA. | **Marginal** -- 30-40s per response on CPU |

### Smallest Practical Multimodal Model for Food

**SmolVLM-256M** is the smallest at <1 GB RAM. It can:
- Describe food in images ("What food is in this image?")
- Answer questions about meals ("How many dishes are on the plate?")
- Read text from menus or nutrition labels (OCR)

Benchmark scores (256M vs 2.2B):
- Science_QA: 73.6 vs 84.5
- DocVQA: 58.3 vs 79.7
- TextVQA: 49.9 vs 72.1

Available in ONNX format (vision_encoder.onnx, embed_tokens.onnx, decoder_model_merged.onnx) and GGUF (22 quantized variants for llama.cpp).

**For food tracking specifically, the best balance is Gemma 3n E2B** -- it handles both the image understanding AND text extraction in a single model, with 2 GB RAM.

### Vision Encoder Details (Gemma 3n)

- MobileNet-V5-300M: 46% fewer params, 4x smaller memory vs baseline
- Supports 256x256, 512x512, 768x768 input resolutions
- 13x speedup with quantization on Pixel Edge TPU
- Up to 60 FPS on Google Pixel (real-time video analysis possible)

---

## 3. On-Device NER/Entity Extraction for Food (Lightweight Alternatives)

### Pre-trained Food NER Models

| Model | Base | Params | Size (est.) | F1 | Training Data | Notes |
|-------|------|--------|------------|-----|---------------|-------|
| **FoodBaseBERT-NER** | bert-base-cased | 110M | ~420 MB (FP32) | Not published | FoodBase corpus (274K food annotations) | MIT license. Single FOOD entity class. |
| **InstaFoodRoBERTa-NER** | roberta-base | 125M | ~480 MB (FP32) | **0.91** | InstaFoodSet (400 Instagram posts) | Handles informal text ("poké bowl", "chia seeds"). MIT license. |
| **chambliss/distilbert-for-food-extraction** | distilbert | 66M | ~250 MB (FP32) | Not published | Unknown | Smaller but undocumented. |
| **sgarbi/bert-fda-nutrition-ner** | bert-base | 110M | ~420 MB (FP32) | Not published | FDA nutrition data | Nutrition-specific entities. |

### Shrinking These for Mobile

| Approach | Model Size | Inference Time | Accuracy Impact |
|----------|-----------|---------------|-----------------|
| ONNX export (FP32) | ~250-480 MB | ~50-100ms | None |
| ONNX + INT8 quantization | ~60-120 MB | ~20-50ms | <1% drop |
| DistilBERT base | ~250 MB | ~30ms | 2-3% drop vs BERT-base |
| BERT-tiny (4.4M params) | ~17 MB | ~5-10ms | Significant drop, needs fine-tuning |
| BERT-mini (11M params) | ~45 MB | ~10-20ms | Moderate drop, fine-tunable |
| NeuroBERT-Mini (~7M params) | ~35 MB (quantized) | <10ms | Purpose-built for edge |
| MobileBERT | ~100 MB | 62ms (128 tokens, Pixel 4) | Near BERT-base quality |

### Recommended Pipeline for Food NER

1. **Start with InstaFoodRoBERTa-NER** (F1=0.91, handles informal text)
2. **Export to ONNX** via `optimum-cli export onnx`
3. **Quantize to INT8** -- model drops to ~120 MB, inference ~20-50ms
4. **Run via ONNX Runtime React Native** (`onnxruntime-react-native` npm package)

**Alternative: Fine-tune BERT-mini on food data** for an ultra-small model:
- Start from `boltuix/bert-mini` (8M params, ~15 MB quantized)
- Fine-tune on FoodBase corpus + TASTEset + InstaFoodSet
- Export to ONNX INT8 -- final size ~15-20 MB
- Inference: <10ms on any modern phone

### Food NER Datasets Available

| Dataset | Size | Entities | Notes |
|---------|------|----------|-------|
| **FoodBase** | 274K annotations, 13K unique | Food items | Academic benchmark corpus |
| **TASTEset** | 700 recipes, 13K+ entities | Food items, quantities, units | Recipe-focused |
| **InstaFoodSet** | 400 Instagram posts | Food items | Social media / informal text |
| **ARTI** | Large | Ingredient lines | Diverse ingredient data |
| **FINER** | Carefully annotated | Food ingredients | Fine-grained ingredients |

### SpaCy on Mobile?

Not practical. SpaCy requires Python runtime and has heavy dependencies. For mobile, the ONNX Runtime path (export transformer model -> ONNX -> quantize -> run with onnxruntime-react-native) is the correct approach.

---

## 4. Framework Support for React Native / Expo

### Framework Comparison

| Framework | RN Support | Models Supported | Maturity | GPU Accel | Notes |
|-----------|-----------|-----------------|----------|-----------|-------|
| **react-native-executorch** | Native (v0.7.x) | Qwen 3, Llama 3.2, SmolLM 2, Hammer 2.1, CLIP, Whisper | Production-ready | Yes (Metal, Qualcomm, Arm, MediaTek, Vulkan) | Best React Native integration. Expo-compatible. VLM support (Moondream) on roadmap (v0.8.0). |
| **llama.rn** | Native (v0.11.2) | Any GGUF model (Llama, Gemma, Qwen, etc.) | Mature | Metal (iOS), OpenCL (Android Adreno 700+), Hexagon NPU (experimental) | Most flexible -- runs any GGUF model. Multimodal via mmproj (images + audio). Requires New Architecture (v0.10+). |
| **expo-llm-mediapipe** | Expo native | Gemma series | Early | Via MediaPipe | Declarative useLLM hook. Note: Google recommends migrating to LiteRT-LM. |
| **onnxruntime-react-native** | Native | Any ONNX model (BERT, DistilBERT, custom) | Mature | NNAPI (Android), CoreML (iOS) | Best for small transformer models (NER, classification). Light-weight inference. |
| **LiteRT-LM** (Google AI Edge) | No direct RN bindings yet | Gemma 3n, Gemma 3, Phi-4, Qwen3, etc. | Production-ready | GPU (Android), CPU cross-platform | Best for Gemma 3n. No React Native bindings yet -- would need native module wrapper. |
| **react-native-transformers** | Expo/RN | HuggingFace models via ONNX Runtime | Early | Via ONNX Runtime | Convenience wrapper around onnxruntime-react-native. |

### Recommended Stack for Food Tracking

**For LLM inference (text parsing, food entity extraction):**
- **Option A:** `llama.rn` + Gemma 3n E2B in GGUF format -- most flexible, runs any model
- **Option B:** `react-native-executorch` + Llama 3.2 1B -- best integrated RN experience
- **Option C:** `expo-llm-mediapipe` + Gemma -- simplest Expo integration

**For lightweight NER (food entity extraction without full LLM):**
- `onnxruntime-react-native` + fine-tuned BERT/DistilBERT ONNX model

**For vision (food image understanding):**
- `llama.rn` with mmproj + multimodal GGUF model (Gemma 3n, Moondream)
- `react-native-executorch` CLIP hook for image embeddings

### ExecuTorch Roadmap (react-native-executorch)

| Version | Release | Key Features |
|---------|---------|-------------|
| v0.1.0 | Nov 2024 | Llama 3.2 1B/3B |
| v0.2.0 | Dec 2024 | Classification, object detection, style transfer |
| v0.3.0 | Mar 2025 | OCR, Whisper STT |
| v0.4.0 | May 2025 | More LLMs, embeddings, multilingual Whisper |
| v0.6.0 | Dec 2025 | Stable Diffusion, quantized Whisper |
| v0.7.0 | Current | Kokoro TTS |
| **v0.8.0** | **Planned** | **VLM support (Moondream), quantized CV models** |
| v0.9.0 | Planned | Segment Anything |

---

## 5. Practical Examples & Research

### Open-Source Food Tracking with On-Device NLP

**No production open-source food tracker uses on-device NLP as of early 2026.** All AI-powered food trackers (Nutritheous, NutriSmart, etc.) rely on cloud APIs (GPT-4 Vision, Nutritionix API). This is a greenfield opportunity.

Closest examples:
- **OpenNutriTracker** -- open source, privacy-focused, but uses Open Food Facts database lookup (no NLP)
- **NutriSmart** -- uses Nutritionix cloud NLP for food parsing
- **SlimLM** -- academic demo of on-device SLM on Samsung Galaxy S24 (document tasks, not food)

### Academic Research on Text + Image Food Recognition

Key papers:
1. **"Fine-grained food image classification and recipe extraction using a customized deep neural network and NLP"** (2024, ScienceDirect) -- MResNet-50 for image classification + Word2Vec/Transformers for ingredient extraction
2. **"NutriRAG"** (2025, medRxiv) -- RAG framework using LLMs for food identification from free text. Tested with Llama-2-70b (cloud-scale, but demonstrates the approach)
3. **"CBiAFormer"** (2025) -- Convolution-Enhanced Bi-Branch Adaptive Transformer for food classification with multi-task category-ingredient recognition
4. **"Explainable deep learning ensemble for food image analysis on edge devices"** (Computers in Biology and Medicine) -- ensemble methods for edge deployment
5. **"Deep Learning in Food Image Recognition: A Comprehensive Review"** (2025, MDPI) -- surveys edge computing and multi-modal integration trends

### Benchmark: Small Models on Food NLP Tasks

No published benchmarks exist specifically for small models on food NER/extraction. The closest proxies:
- InstaFoodRoBERTa-NER achieves F1=0.91 on food entity extraction from social media text
- General NER benchmarks show DistilBERT retains ~97% of BERT-base performance
- 3B instruction-tuned LLMs match GPT-3.5 level on structured extraction tasks

---

## 6. Text-Guided Image Classification (CLIP-like Models)

### On-Device CLIP Models

| Model | Image Encoder | Text Encoder | Image Latency (iPhone 12 Pro Max) | Text Latency | Zero-Shot IN-1K | Notes |
|-------|--------------|-------------|-----------------------------------|--------------|-----------------|-------|
| **MobileCLIP-S0** | 11.4M params | 42.4M params | 1.5ms | 3.3ms | 67.8% | Fastest. 4.8x faster than ViT-B/16 CLIP. |
| **MobileCLIP-S2** | 35.7M params | 63.4M params | 3.6ms | 3.3ms | 74.4% | Sweet spot. 2.3x faster than ViT-B/16 with better accuracy. |
| **MobileCLIP-B** | 86.3M params | 63.4M params | 10.4ms | 3.3ms | 77.2% | Full ViT-B quality at 10ms. |
| **MobileCLIP2-S0** | 11.4M params | 63.4M params | 1.5ms | 3.3ms | 71.5% | +3.7% over v1. |
| **MobileCLIP2-S2** | 35.7M params | 63.4M params | 3.6ms | 3.3ms | 77.2% | Matches MobileCLIP-B v1 accuracy at 3.6ms! |
| **MobileCLIP2-B** | 86.3M params | 63.4M params | 10.4ms | 3.3ms | 79.4% | Best sub-15ms accuracy. |
| **OpenCLIP ViT-B/16** | 86M params | 63M params | ~15-20ms (est.) | ~5ms | ~76% | Standard reference. Heavier but well-supported. |

### How CLIP Helps Food Tracking

**Zero-shot food classification:** Pre-compute text embeddings for food categories (e.g., "ramen", "salad", "pizza", "sushi"), then match image embeddings at inference time. No training needed.

**Text-guided disambiguation:** When image classification is uncertain between "pad thai" and "lo mein", user text input ("I had Thai food") can be used to re-rank candidates by computing text-image similarity.

**Practical architecture:**
```
[Camera Image] -> MobileCLIP Image Encoder (3.6ms) -> Image Embedding
[Food Labels]  -> MobileCLIP Text Encoder (3.3ms)  -> Text Embeddings (pre-computed)
                                                    -> Cosine Similarity -> Top-K foods
```

**Total latency: <7ms** for a complete zero-shot classification pass. This is faster than a single frame at 60 FPS.

### MobileCLIP Deployment

- CoreML export available: `apple/coreml-mobileclip` on HuggingFace (iOS native)
- ONNX export: via `reparameterize_model()` + standard PyTorch->ONNX conversion
- OpenCLIP API compatible: `open_clip.create_model_and_transforms("MobileCLIP2-S0")`
- MobileCLIP2 published August 2025 (TMLR), actively maintained by Apple

### Limitations for Food

- CLIP models are trained on general web data, not food-specific data
- Food categories can be ambiguous in images (similar-looking dishes from different cuisines)
- Fine-tuning MobileCLIP on food-specific data would improve accuracy significantly
- CLIP alone cannot estimate portions/quantities -- needs complementary model

---

## 7. Recommended Architecture for FoodTracker

### Tiered Approach (matching local-first architecture from ADR-005)

**Tier 1: Instant (<10ms) -- CLIP + NER**
- **MobileCLIP2-S2** for image classification (3.6ms image, 3.3ms text)
- **Fine-tuned BERT-mini ONNX** for food entity NER from text input (~10ms)
- Total model size: ~80 MB (CLIP) + ~15 MB (NER) = ~95 MB
- RAM: <500 MB combined

**Tier 2: Fast (<1s) -- Small VLM**
- **SmolVLM-256M** for image+text understanding when CLIP confidence is low
- Can ask "What food is in this image?" and get detailed description
- Model size: ~300 MB (quantized GGUF), RAM: <1 GB
- Latency: ~500ms-1s for short responses

**Tier 3: Quality (<3s) -- Full SLM**
- **Gemma 3n E2B** for complex food parsing, multi-item meals, quantity estimation
- Prompt: "Parse this meal into structured data: 'I had a large bowl of ramen with extra chashu, a soft-boiled egg, and a side of edamame'"
- Model size: ~1 GB (quantized), RAM: ~2 GB
- Latency: 1-3s for structured JSON output
- Also handles multimodal (image + text prompt) for detailed food analysis

### Integration Path

1. **Phase 1:** Add MobileCLIP for zero-shot food image classification (small, fast, no LLM needed)
2. **Phase 2:** Add ONNX NER model for text-based food logging ("I ate two tacos and a burrito")
3. **Phase 3:** Add SmolVLM-256M or Gemma 3n E2B for complex multimodal understanding
4. **Phase 4:** Fine-tune models on food-specific data for improved accuracy

### React Native Integration

```
Phase 1: onnxruntime-react-native + MobileCLIP ONNX models
Phase 2: onnxruntime-react-native + Food NER ONNX model
Phase 3: llama.rn + Gemma 3n GGUF (or react-native-executorch + Llama 3.2)
```

---

## Sources

### On-Device LLMs/SLMs
- [Best Open-Source SLMs in 2026 (BentoML)](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [Small Language Models Guide 2026 (LocalAIMaster)](https://localaimaster.com/blog/small-language-models-guide-2026)
- [Top 15 Small Language Models for 2026 (DataCamp)](https://www.datacamp.com/blog/top-small-language-models)
- [On-Device LLMs: State of the Union, 2026](https://v-chandra.github.io/on-device-llms/)
- [SlimLM: On-Device Document Assistance (ACL 2025)](https://aclanthology.org/2025.acl-demo.42/)
- [Feasibility and Trade-Offs of On-Device LLM Inference](https://dl.acm.org/doi/pdf/10.1145/3788870)

### Gemma 3n
- [Gemma 3n (Google DeepMind)](https://deepmind.google/models/gemma/gemma-3n/)
- [Gemma 3n Developer Guide (Google Developers Blog)](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [Gemma 3n August 2025 Update](https://www.gemma-3n.net/blog/gemma-3n-august-2025-update/)
- [Gemma 3n on Mobile with Google AI Edge](https://developers.googleblog.com/en/gemma-3-on-mobile-and-web-with-google-ai-edge/)
- [Image Analysis with Gemma 3n](https://www.gemma-3n.net/blog/image-analysis-with-gemma-3n/)

### Multimodal / Vision-Language Models
- [SmolVLM-256M-Instruct (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [SmolVLM Paper (arXiv)](https://arxiv.org/abs/2504.05299)
- [Moondream 0.5B (Moondream.ai)](https://moondream.ai/blog/introducing-moondream-0-5b)
- [Moondream2 (HuggingFace)](https://huggingface.co/vikhyatk/moondream2)
- [MobileVLM (arXiv)](https://arxiv.org/html/2312.16886v1)
- [PaliGemma 2 (Google Developers Blog)](https://developers.googleblog.com/en/introducing-paligemma-2-powerful-vision-language-models-simple-fine-tuning/)
- [Best Open-Source VLMs in 2026 (BentoML)](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)

### Food NER Models
- [FoodBaseBERT-NER (HuggingFace)](https://huggingface.co/Dizex/FoodBaseBERT-NER)
- [InstaFoodRoBERTa-NER (HuggingFace)](https://huggingface.co/Dizex/InstaFoodRoBERTa-NER)
- [chambliss/distilbert-for-food-extraction (HuggingFace)](https://huggingface.co/chambliss/distilbert-for-food-extraction)
- [TASTEset: Recipe Dataset and Food Entities Recognition Benchmark](https://arxiv.org/abs/2204.07775)
- [FoodBase Corpus (Oxford Academic)](https://academic.oup.com/database/article/doi/10.1093/database/baz121/5611291)
- [Food NER Survey (IEEE)](https://ieeexplore.ieee.org/document/8995569/)

### MobileCLIP
- [MobileCLIP (Apple ML Research)](https://machinelearning.apple.com/research/mobileclip)
- [MobileCLIP2 (Apple ML Research)](https://machinelearning.apple.com/research/mobileclip2)
- [MobileCLIP GitHub](https://github.com/apple/ml-mobileclip)
- [CoreML MobileCLIP (HuggingFace)](https://huggingface.co/apple/coreml-mobileclip)

### React Native Frameworks
- [React Native ExecuTorch](https://docs.swmansion.com/react-native-executorch/)
- [react-native-executorch GitHub](https://github.com/software-mansion/react-native-executorch)
- [llama.rn GitHub](https://github.com/mybigday/llama.rn)
- [expo-llm-mediapipe GitHub](https://github.com/tirthajyoti-ghosh/expo-llm-mediapipe)
- [ONNX Runtime React Native](https://onnxruntime.ai/docs/get-started/with-javascript/react-native.html)
- [How to Use ONNX in React Native (Simplico, Jan 2026)](https://simplico.net/2026/01/21/how-to-use-an-onnx-model-in-react-native-and-other-mobile-app-frameworks/)
- [LLM Inference on Edge (HuggingFace Blog)](https://huggingface.co/blog/llm-inference-on-edge)
- [LiteRT-LM (Google AI Edge)](https://github.com/google-ai-edge/LiteRT-LM)

### Mobile Inference Benchmarks
- [Llama 3.2 ExecuTorch + KleidiAI Benchmarks (Arm)](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
- [Llama 3.2 Mobile Announcement (Meta)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [ExecuTorch 1.0 GA (PyTorch)](https://pytorch.org/blog/unleashing-ai-mobile/)

### Food + AI Research
- [NutriRAG: LLMs for Food Identification (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11957177/)
- [Deep Learning in Food Image Recognition: Comprehensive Review (2025)](https://www.mdpi.com/2076-3417/15/14/7626)
- [Fine-grained food image classification and recipe extraction (2024)](https://www.sciencedirect.com/science/article/pii/S0010482524006127)
