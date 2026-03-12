# Android Device Fragmentation & On-Device ML Inference

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## RAM Distribution (Active Android Devices, 2025-2026)

Precise per-tier percentages of the active installed base are not publicly reported by Google, IDC, or Counterpoint. Best available data from shipment share, ScientiaMobile, and Android Go penetration:

| RAM Tier | Estimated % of Active Devices | Notes |
|----------|-------------------------------|-------|
| <=2GB | ~5-8% | Legacy, Android Go |
| 3GB | ~10-12% | |
| 4GB | ~18-22% | Android Go targets <=4GB |
| 6GB | ~20-24% | |
| 8GB | ~22-26% | Largest single tier in new shipments (~38.5%) |
| 12GB+ | ~10-15% | Flagships only |

**~30-35% of the active base has 4GB or less.**

### 2026 DRAM Shortage Impact

TrendForce reports a DRAM shortage in 2026 that is **reversing the upward RAM trend**:
- Budget phones reverting to 4GB (from 6GB)
- Mid-range capping at 8GB (down from 12GB)
- Flagships stalling at 12GB
- Low-end smartphone BOM costs up 25%

Source: TrendForce (Dec 2025), Counterpoint (2026 forecasts), PhoneArena analysis.

### Android Go Edition

- Targets devices with up to 4GB RAM
- **250M+ monthly active devices** across 16,000+ device models in 180+ countries
- Android 16 Go edition released Nov 2025

## Chipset Market Share (Q1 2025)

Global smartphone SoC shipment share (Counterpoint Research):

| Vendor | Global Share | Android-Only Share | Notes |
|--------|-------------|-------------------|-------|
| MediaTek | 36% | ~43% | #1 globally; dominates mid-range and entry-tier |
| Qualcomm | 28% | ~34% | Dominant in flagships |
| Apple | 17% | N/A | iOS only |
| UNISOC | 10% | ~12% | Budget/entry-level; strong in India |
| Samsung (Exynos) | 5% | ~6% | Declining |
| Huawei (HiSilicon) | 4% | ~5% | Primarily China domestic |

### NPU Capability by Chipset

| Chipset | NPU TOPS | ML Capability |
|---------|----------|---------------|
| Snapdragon 8 Elite | 75 | Flagship; runs Gemini Nano, LLMs; 100x CPU speedup |
| Snapdragon 8 Gen 3 | 45 | Strong; gen AI capable |
| Snapdragon 8 Gen 2 | ~26 | Good; 47x speedup for YOLOv8n vs CPU |
| Dimensity 9400 | 50 | On-par with SD 8 Elite |
| Dimensity 8300/8400 | ~10-15 (est.) | Mid-range; capable for YOLO |
| Google Tensor G4 | Not disclosed | Optimized for Gemini Nano (45 tok/s) |
| Samsung Exynos 2400 | ~35 (est.) | Competitive but limited to Samsung |
| Snapdragon 6/7 series | ~4-12 | Basic NPU; classification OK, detection limited |
| UNISOC T760/T820 | 3.2 | Very limited; basic AI only |
| MediaTek Dimensity 7300 | ~4-6 (est.) | Basic on-device ML |

**~50-60% of active Android devices have meaningful NPU capability (>10 TOPS). ~30-40% have no NPU or very limited NPU (<5 TOPS).**

## YOLOv8 Nano Benchmarks on Android

From arXiv 2511.13453 (Gherasim, Nov 2025) on Samsung Galaxy Tab S9 (Snapdragon 8 Gen 2):

| Accelerator | Speedup vs CPU 1-core | Estimated Latency |
|-------------|----------------------|-------------------|
| NPU (INT8) | 47x | ~5-8ms |
| GPU (FP16) | Up to 39x (larger YOLO) | ~15-25ms |
| CPU (4-core) | 4-6x | ~50-80ms |
| CPU (1-core, FP32 baseline) | 1x | ~250-350ms |

From Ultralytics community and Google LiteRT blog:
- YOLOv8n with GPU delegate on modern phones: <30ms per frame
- YOLOv8s/YOLO26s: 30 FPS (~33ms/frame) via TFLite/CoreML
- Snapdragon 8 Elite NPU: 56/72 tested models run in <5ms; 64/72 fully delegate to NPU

**Accuracy tradeoff with INT8:** ~6.5 mAP points lost across all YOLO model sizes.

## NNAPI Deprecation & LiteRT Migration

### Timeline
- NNAPI introduced in Android 8.1 (2017)
- **Deprecated in Android 15** (Aug 2024)
- Not yet removed as of Android 16 (still functional, no new features)
- Neural Networks HAL continues for OEMs; only NDK API deprecated

### LiteRT (successor)
- Unified API: `CompiledModel::Create` supports CPU (XNNPack), GPU, NPU, EdgeTPU
- Vendor-specific NPU delegates replace NNAPI's generic approach:
  - **Qualcomm:** LiteRT QNN Accelerator (Qualcomm AI Engine Direct SDK)
  - **MediaTek:** LiteRT NeuroPilot Accelerator
  - Vendor SDKs auto-downloaded via PyPI
- Distribution via Google Play for On-device AI (PODAI)
- Two compilation paths: AOT (known target SoCs, reduces init cost) and On-Device
- Feature parity achieved with TensorFlow Lite

### Migration for developers
1. Custom ML models: use GPU delegate or vendor NPU delegate instead of NNAPI delegate
2. GenAI/LLM: use AICore (Android 14+)
3. Code: replace NNAPI delegate calls with LiteRT CompiledModel API

## Thermal Throttling During Sustained Inference

### Onset and Severity (arXiv 2503.21109, March 2025)
- Triggers after **~2.5 minutes (150 seconds)** of continuous inference
- CPU frequency drops from 3GHz to as low as 1GHz (52% reduction)
- GPU: 510/600 MHz usage drops to zero; 390 MHz usage jumps from 15% to 67%
- Results in up to **4.3x performance degradation**
- Device temperature rises ~18C on average
- CPU frequency fluctuations increase by 217%

### LLM-Specific (arXiv 2410.03613)
- Performance degradation: 10-20% in moderate use
- Kicks in after 5-15 minutes on mainstream flagships
- Performance drops 30-70% once engaged

### Power Consumption (OnePlus 13R case study)
- CPU-centric VLM inference: **10-12W, 90-95C**
- GPU-accelerated (MLC-Imp): **1.3W, 60C**
- GPU/NPU offloading caused device instability (screen freezes) in some cases

### Mitigation Strategies
1. Hysteresis-based active cooling: improves throughput up to ~90% vs no cooling
2. Dynamic shifting: adapt model complexity based on thermal state (ICML 2022)
3. Heterogeneous processor co-execution: distribute load across CPU/GPU/NPU
4. Bursty inference: short bursts with cooldown rather than sustained continuous

## Sources

- Counterpoint Research Q1 2025 global smartphone AP-SoC market share
- ScientiaMobile: How Much RAM is in Smartphones?
- Android (Go edition) Developer Guide
- TrendForce: Memory Price Surge to Persist (Dec 2025)
- Counterpoint: 2026 Smartphone Shipment Forecasts Revised Down
- arXiv 2511.13453: Hardware optimization on Android for inference of AI models (Nov 2025)
- Google Blog: Unlocking Peak Performance on Qualcomm NPU with LiteRT
- arXiv 2410.03613: Understanding LLMs in Your Pockets
- On-Device LLMs: State of the Union, 2026 (v-chandra.github.io)
- arXiv 2503.21109: Optimizing Multi-DNN Inference on Mobile Devices (March 2025)
- NNAPI Migration Guide (Android NDK)
- LiteRT GitHub (google-ai-edge)
- Google Blog: LiteRT Maximum Performance Simplified
- MLPerf Mobile v4.0 (May 2024)
