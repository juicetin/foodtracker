# ADR-003: YOLO-Based Food Detection Over Multimodal LLMs

**Status:** Accepted
**Date:** 2026-02-08

## Context

Two approaches exist for food detection from images:

1. **Traditional deep learning (YOLO/CNN):** Train a model specifically for food detection. Runs locally, produces bounding boxes with pixel-precise coordinates, near-zero marginal cost.

2. **Multimodal LLMs (GPT-4o, Gemini):** Send image to a cloud API, get back natural language description of foods. Excellent at zero-shot identification of unusual foods but expensive per-image, slow (2-5s), and imprecise spatially.

## Decision

Use YOLO (specifically YOLOv8/v11 via Ultralytics) as the primary food detection model.

**Reasons:**

- **Cost:** Near-zero per inference when run on-device. An LLM API call costs ~$0.01-0.03 per image — at 5 photos/day/user, this adds up fast.
- **Speed:** ~30ms inference vs 2-5 seconds for an LLM API call. Enables live camera feed detection.
- **Spatial accuracy:** YOLO produces precise bounding box coordinates, which are essential for weight estimation (bbox area → real-world dimensions → volume → weight).
- **Privacy:** Images can stay on-device. No data leaves the phone.
- **Offline capability:** Works without internet connectivity.

**Trade-off acknowledged:** YOLO trained on COCO only knows ~10 food classes. Fine-tuning on food-specific datasets (Food-101, ISIA Food-500) is required for production accuracy. The POC spike will quantify how much improvement is needed.

**Not decided yet:** Whether to add an LLM fallback for low-confidence detections. The POC will inform this — if YOLO + fine-tuning achieves >80% accuracy on test photos, an LLM fallback may be unnecessary.

## Consequences

- Need to invest time in fine-tuning a food-specific YOLO model (training on Food-101 or similar).
- Model must be exported to mobile-friendly formats (CoreML, TFLite) — see ADR-002.
- Weight estimation pipeline depends on bounding box quality, so detection accuracy directly impacts calorie accuracy.
- Regional food varieties (e.g. Australian, Asian cuisines) will need targeted training data.
