# On-Device OCR for Kitchen Scale Displays

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## The Core Problem

Kitchen scales use 7-segment LCD/LED displays. General-purpose OCR (both cloud and on-device) was not designed for these displays and performs poorly on them.

## General-Purpose OCR Assessment

### Google ML Kit Text Recognition v2
- **Not designed for 7-segment displays.** Trained on printed and handwritten text.
- Frequently misinterprets digits: confuses 5/6, 8/3, 7/1
- No special handling for decimal points in numeric displays
- Requires each character to be at least 16x16 pixels
- Poor focus and low contrast significantly degrade accuracy
- Performance: ~10-50ms per frame, 3-10MB model, 11-19MB runtime memory
- Fully on-device, offline capable

### Apple Vision Framework (VNRecognizeTextRequest)
- **Explicitly unsupported for 7-segment displays.** Apple Developer Forums confirm Apple recommends training a custom detector for LCD/LED digits.
- Two modes: `.fast` and `.accurate` (powers Live Text)
- Real-time on Neural Engine (A12+)
- iOS 18: expanded to 18 languages
- Excellent for printed/handwritten text, not for displays

### Cloud OCR (Google Cloud Vision, Gemini)
- Google Cloud Vision API: 98.0-98.7% on clean printed text, up to 99.56% on standard documents
- **No specific support for digital displays**
- Gemini exhibits hallucination: may silently edit, merge lines, or alter punctuation
- Google recommends pairing Gemini with dedicated OCR for mission-critical accuracy
- Not suitable for exact character-for-character fidelity like scale readings

**Verdict: General-purpose OCR (cloud or on-device) is not suitable for scale display reading.**

## Specialized Solutions

### Purpose-Built Models and Tools

| Solution | Approach | Size | Suitability |
|---|---|---|---|
| Custom TFLite CNN | Trained on 7-segment digits | 17KB-5MB (INT8) | Best for mobile |
| ssocr | CLI tool for 7-segment | N/A (server-side) | Not mobile-friendly |
| YOLO + digit classifier | Detect display, classify digits | ~10-20MB | Good accuracy |
| Template matching (SegoDec) | OpenCV pixel comparison | Small | Simple, effective for consistent displays |

### Training Data Available

- **Roboflow Universe:** Multiple datasets:
  - 948 annotated images (Bhautik Pithadiya) with YOLO format
  - 698 images (labmonitor) compatible with YOLOv11
  - Additional datasets published in 2025
- **Edge Impulse:** 7-segment digit models at 17KB (INT8 quantized TFLite)

### Existing Projects

| Project | Approach | Notes |
|---|---|---|
| renjithsasidharan/seven-segment-ocr | Keras -> TFLite Float16 | Mobile-deployable |
| SachaIZADI/Seven-Segment-OCR | CNN (SVHN-style) | End-to-end digit sequence recognition |
| suyashkumar/seven-segment-ocr | Image/video digitizer | General 7-segment |
| scottmudge/SegoDec | OpenCV template matching | Simple, reliable |
| **jacnel/wetr** | **Kitchen scale specific** | Tesseract + digit/decimal whitelisting |

### The `wetr` Project (Most Relevant)

A weight tracking app specifically for reading kitchen scales:
- Uses Tesseract with digit+decimal whitelisting (`0-9` and `.` only)
- Key insight: character height of **~50 pixels minimum** for reliable decimal point recognition
- Ensures only one decimal exists in output string
- Validates weight range

## Recommended Approach

A two-stage pipeline:

1. **Display Region Detection**
   - Use camera to capture the display region
   - Preprocessing: threshold, perspective correction, crop to digits
   - ML Kit / Vision can help with general ROI detection, or use a simple UI guide overlay

2. **Digit Classification**
   - Small custom TFLite/CoreML model trained on 7-segment digits
   - Roboflow datasets provide training data
   - INT8 quantization: as small as 17KB

3. **Post-Processing**
   - Whitelist: digits (0-9) and single decimal point only
   - Range validation: kitchen scales typically 0-5000g
   - Reject implausible readings
   - Ensure only one decimal point

4. **Fallback**
   - Always allow manual weight entry
   - Show detected value for user confirmation before accepting

### Performance Characteristics

| Factor | Custom TFLite | ML Kit | Cloud API |
|---|---|---|---|
| Latency | ~5-20ms | ~10-50ms | 200-2000ms |
| Model size | 17KB-5MB | 3-10MB | N/A |
| Runtime memory | ~5-20MB | ~11-19MB | N/A |
| Offline | Yes | Yes | No |
| 7-segment accuracy | Excellent | Poor | Moderate |
| Decimal handling | Trainable | Unreliable | Better but imperfect |

## Sources

- Google ML Kit Text Recognition v2 documentation
- Apple VNRecognizeTextRequest documentation
- Apple Developer Forums: Recognizing LCD/LED number digits (thread/651059)
- GDELT OCR Benchmark: Cloud Vision vs Gemini vs GPT-4o
- Roboflow: 7-segment display datasets
- Edge Impulse: seven-segment digit models
- jacnel/wetr: kitchen scale OCR app
- scottmudge/SegoDec: OpenCV template matching
- Seven Segment Digit Recognition (arXiv 1807.04888)
