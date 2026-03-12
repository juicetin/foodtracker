# On-Device Vision Language Models for Mobile Food Recognition

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## Available Small VLMs for Mobile (2025-2026)

| Model | Parameters | Quantized Size (INT4) | Runtime RAM | Mobile Runtimes |
|-------|-----------|----------------------|-------------|-----------------|
| SmolVLM-256M | 0.3B | ~100MB | ~300-500MB | llama.cpp (GGUF), WebGPU |
| SmolVLM2-500M | 0.5B | ~175MB (Q8) | ~400-600MB | llama.cpp (GGUF) |
| Moondream 0.5B | 0.5B | 375MB (INT4) | 816MB | Custom runtime, INT4/INT8 |
| Florence-2-Base | 0.23B | <200MB | Small | ONNX, OpenVINO |
| Moondream 2B | 1.86B | ~1GB | 2.45GB VRAM (4-bit) | llama.cpp (partial), custom |
| SmolVLM-2.2B | 2.2B | ~1.1GB | ~1.5-2GB | llama.cpp, bitsandbytes |
| MobileVLM v2 | 1.7B / 3B | ~0.9-1.5GB | ~1-2GB | MLC-LLM (modified) |
| Qwen2.5-VL-3B | 3B | GGUF (Q4_K) | ~2GB | llama.cpp (GGUF), MNN |
| PaliGemma 2 3B | 3B | ~1.5GB | ~2GB | gemma.cpp, ExecuTorch |
| Gemma 3n E2B | 5B total / 2B effective | ~2GB | 2GB | LiteRT-LM, MediaPipe |
| Gemma 3n E4B | 8B total / 4B effective | ~3GB | 3GB | LiteRT-LM, MediaPipe |
| FastVLM (Apple) | 0.5B / 1.5B / 7B | CoreML optimized | Varies | CoreML only |

**Sweet spot for mobile: 0.5B-3B parameters.** Models above ~4B active parameters push limits of current smartphone RAM.

## Mobile Inference Benchmarks

| Model / Framework | Device | Tokens/sec | Memory | Notes |
|---|---|---|---|---|
| Gemma 3n E2B | Modern smartphone | 60-70 tok/s | 2GB | TTFT ~0.3s. Best documented mobile VLM perf |
| Gemma 3n E4B | Modern smartphone | ~49.5 tok/s | 3GB | 40% less memory vs equivalent Llama models |
| MobileVLM 1.4B | Snapdragon 888 CPU | 21.5 tok/s | - | CPU-only |
| MobileVLM | Jetson Orin GPU | 65.3 tok/s | - | GPU-accelerated edge device |
| 1B LLM (generic) | iPhone 15 Pro (CPU) | 17 tok/s | - | CPU-only, 2 threads, F16 |
| Qwen3-0.6B | Pixel 8 / iPhone 15 Pro | ~40 tok/s | - | ExecuTorch INT8/INT4 |
| 8B LLM | Smartphone (ExecuTorch) | 30+ tok/s | - | Meta's production figure |
| Cactus v1 (INT8) | iPhone 17 Pro | 136 tok/s | - | NPU-accelerated |
| Cactus v1 (INT8) | Galaxy S25 Ultra | 91 tok/s | - | NPU-accelerated |

### Critical Finding: Vision Encoding Bottleneck

MobileAIBench found multimodal tasks on mobile have **TTFT exceeding 60 seconds** and input processing under 5 tokens/sec for typical VLMs. The image encoding step is the major latency source, not text generation.

LLaVA-v1.5-7B image encoding: 6 sec/image on Android (MLC-LLM modified) -- 87% improvement from 37s baseline.

### Memory Bandwidth Is the Fundamental Constraint

Mobile devices: **50-90 GB/s** memory bandwidth
Data center GPUs: **2-3 TB/s** memory bandwidth

Decode is memory-bandwidth bound -- each token requires streaming full model weights.

## iOS vs Android Performance Parity

iOS has a **~30-50% performance advantage** for on-device LLM/VLM inference:

- Apple Neural Engine: 2-3x faster than Qualcomm Hexagon NPU (Geekbench AI)
- LLM inference: Apple M4 at 48 tok/s vs Snapdragon X Elite at 24 tok/s
- Cactus v1: iPhone 17 Pro at 136 tok/s vs Galaxy S25 Ultra at 91 tok/s (1.5x)
- FastVLM: 85x faster TTFT than comparable models (CoreML/Neural Engine optimized)
- Apple's unified memory architecture provides higher effective bandwidth

Android advantages:
- More diverse hardware ecosystem
- Better open-source framework support (llama.cpp more straightforward on Android)
- MNN (Alibaba) optimized for Android SoCs
- Qualcomm raw TOPS higher (but Apple compensates with software efficiency)

For sub-1B models, the difference is less pronounced since both platforms handle them comfortably.

## Food-Specific Model Distillation

### JDNet (IEEE 2020) -- Direct Precedent
- Multi-stage knowledge distillation from large teacher to compact student
- **91.2% Top-1 on Food-101** and **84.0% on UECFood-256**
- Student model is **4x smaller** than teacher
- Specifically designed for mobile visual food recognition

### VL2Lite (CVPR 2025)
- Task-specific knowledge distillation from large VLMs to lightweight networks
- Three components: task-specific classification, visual KD, linguistic KD
- Outperforms traditional KD methods (ResNet152, CLIP teachers)
- Applicable to fine-grained food recognition

### FoodNExTDB VLM Benchmark (2025)
- 6 VLMs evaluated: ChatGPT, Gemini, Claude, Moondream, DeepSeek, LLaVA
- Closed-source models: >90% EWR on single food product recognition
- Open-source models: significantly lag behind
- All struggle with fine-grained cooking style differences
- Fast food is the most challenging category

### Achievable Targets
- Distilled food-specific model at 0.5-1B parameters: ~85-90% Food-101 accuracy
- At INT4: 250MB-500MB on disk, 500MB-1GB runtime RAM
- Approach: fine-tune SmolVLM-256M, Moondream 0.5B, or Florence-2-Base on food datasets, then quantize

## Mobile Inference Frameworks

| Framework | VLM Support | Status (2026) | Notes |
|---|---|---|---|
| **llama.cpp** | Yes (May 2025+) | Production-ready | GGUF + mmproj for vision. Most mature. SmolVLM2, Gemma 3, Qwen VL |
| **ExecuTorch** | Yes | GA 1.0 (Oct 2025) | 50KB base runtime, 12+ backends. Powers Meta apps |
| **LiteRT-LM** | Yes (Gemma 3n) | Production | Official for Gemma 3n. 1.4x faster GPU than TFLite |
| **MediaPipe LLM** | Yes | **Deprecated** | Migrate to LiteRT-LM |
| **MLC-LLM** | Partial | Experimental for VLMs | Not officially supported for VLM image+text chat |
| **NexaSDK** | Yes | Active | NPU/GPU/CPU. Qwen3-VL, Gemma3n vision |
| **CoreML** | Yes | Production | iOS/macOS only. FastVLM optimized for Neural Engine |

**Recommendation:** llama.cpp (most proven, cross-platform) or LiteRT-LM (if targeting Gemma 3n).

## Sources

- SmolVLM paper (arXiv 2504.05299)
- Moondream 0.5B blog (moondream.ai)
- Gemma 3n developer guide (developers.googleblog.com)
- JDNet: Joint-Learning Distilled Network (IEEE 2020)
- VL2Lite (CVPR 2025)
- FoodNExTDB: Are VLMs Ready for Dietary Assessment? (arXiv 2504.06925)
- MobileAIBench (arXiv 2406.10290)
- On-Device LLMs: State of the Union, 2026
- ExecuTorch 1.0 announcement
- llama.cpp vision support (Simon Willison, May 2025)
- FastVLM (Apple ML Research)
- Cactus v1 cross-platform benchmarks (InfoQ)
- MobileVLM GitHub (Meituan-AutoML)
