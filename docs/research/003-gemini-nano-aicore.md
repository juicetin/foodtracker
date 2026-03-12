# Gemini Nano & Android AICore for Third-Party Apps

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## Architecture

**Gemini Nano** is Google's on-device LLM, distilled from the larger Gemini family:
- **Nano-1:** 1.8B parameters (low-memory devices)
- **Nano-2:** 3.25B parameters (high-memory devices)
- Both 4-bit quantized, designed for sub-8GB inference
- **Nano-v3** (late 2025): ~2x performance improvement, same parameter count

**AICore** is an Android system service (Android 14+) that manages AI foundation models:
- Shared model instance across all apps on a device
- Routes inference to device NPU/TPU (Tensor, Hexagon, APU)
- Runs inside Android's Private Compute Core (isolated, no data stored after processing)
- Model updated via system-level updates, independent of app updates

## Device Availability

| OEM | Devices |
|---|---|
| Google | Pixel 10 series, Pixel 9 series, Pixel 8 Pro, Pixel 8, Pixel 8a |
| Samsung | Galaxy S26 series, Galaxy S25 series, Galaxy S24 series, Galaxy Z Fold 6, Galaxy Z Flip 6 |
| Motorola | Razr 60 Ultra |
| Xiaomi | Xiaomi 15, Xiaomi 14T |
| OnePlus | OnePlus 13 |
| HONOR | Magic 7 |

**Multimodal (image+text) support** initially Pixel 9 exclusive, expanded to Galaxy S25 and Pixel 10 (nano-v3). Not all listed devices necessarily support multimodal.

**Requires:** Android 14+ with AICore support. ML Kit GenAI library targets API 26+.

## Third-Party API Surface

Access via **ML Kit GenAI APIs** (the Google AI Edge SDK is deprecated):

### High-Level APIs (beta)
- **Summarization** -- condense text
- **Proofreading** -- grammar/spelling
- **Rewriting** -- rephrase text
- **Image Description** -- alt-text/short descriptions (English only)
- **Speech Recognition** -- transcribe audio

### Low-Level API (alpha)
- **Prompt API** -- arbitrary natural language prompts:
  - Text-only input -> text output
  - Multimodal (image + text) -> text output
  - Streaming responses
  - Session-based context retention

### NOT Available
- No fine-tuning API (Google uses internal LoRA for Pixel Recorder etc.)
- No embedding generation
- No function calling / tool use
- No audio/video input (only image+text or text-only)

## Limitations

| Constraint | Value |
|---|---|
| Per-prompt tokens | 1,024 |
| Session context | 4,096 tokens |
| Usage | **Foreground-only** (top app). Background returns `BACKGROUND_USE_BLOCKED` |
| Rate limit | Per-app inference quota; exceeding returns `BUSY` |
| Battery | Per-app daily battery quota; exceeding returns `PER_APP_BATTERY_USE_QUOTA_EXCEEDED` |
| Modalities | Text-in/text-out, image+text-in/text-out only |
| Language | English-focused (Image Description English only) |
| Offline | Yes, fully offline |

## Food Recognition Case Study: Atomic Robot "Can I Eat It?"

Atomic Robot built a food classification app using Gemini Nano Prompt API:
- Accepts food image + text prompt asking about food content
- **Structured JSON output: <1.36s** (vs freeform text: 5.9-8.35s)
- Initial version misclassified foods and was case-sensitive
- Required significant prompt engineering for accuracy
- Demonstrates feasibility but highlights quality limitations of a 3.25B model

## Alternative On-Device AI Services

| Service | API for 3rd Party? | Notes |
|---|---|---|
| Samsung Galaxy AI | No | System features only (Circle to Search, Live Translate) |
| Qualcomm AI Hub | Yes (bring your own model) | NexaSDK for Android; Snapdragon NPU optimized |
| MediaTek NeuroPilot | Yes (model deployment) | LiteRT NeuroPilot Accelerator; supports Qwen3, Gemma 3n |
| Google LiteRT | Yes | General-purpose; more flexibility but more work |

**Gemini Nano via AICore is the only system-level, zero-setup, shared-model approach.** All alternatives require bundling/downloading your own models.

## Implications for Food Tracker

**Opportunity:** Zero app size cost, hardware-optimized, offline. Can use Prompt API with multimodal input for food identification on supported devices.

**Constraints:**
- 1,024 token limit per prompt -- sufficient for food ID but limits complex reasoning
- Foreground-only -- cannot process photos in background gallery scan
- Battery quota -- limits how many food photos can be analyzed per day
- Device coverage growing but not universal -- need fallback regardless

## Sources

- Gemini Nano - Android Developers (developer.android.com/ai/gemini-nano)
- ML Kit GenAI APIs Overview (developers.google.com/ml-kit/genai)
- ML Kit Prompt API alpha release (Android Developers Blog, Oct 2025)
- Atomic Robot: On-Device AI with Gemini Nano (atomicrobot.com/blog)
- Atomic Robot: Gemini Nano Kept Getting It Wrong (atomicrobot.com/blog)
- Android Police: Every phone that supports Gemini Nano
- Android Authority: Gemini Nano features and supported phones
- SamMobile: Samsung devices supporting Gemini Nano
