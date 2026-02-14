# ADR-002: Go Backend with Python ML Inference Layer

**Status:** Accepted
**Date:** 2026-02-08

## Context

The production backend needs to handle API routing, auth, food log CRUD, and orchestration. It also needs to run ML inference for food detection. The team strongly prefers a statically typed language for the backend.

ML model inference is fundamentally tied to the Python ecosystem for training, but has multiple options for production serving.

## Decision

Keep Go as the production backend language. ML inference will be handled by one of these approaches (to be decided after the POC):

1. **On-device inference (preferred):** Export trained YOLO model to CoreML (iOS) / TFLite (Android). Model runs on the phone's NPU. Go backend never touches image pixels for detection — it only receives structured ingredient data.

2. **Go + ONNX Runtime:** Use `onnxruntime-go` to load an ONNX-exported model directly in the Go service. Keeps everything in one process, avoids a separate Python service.

3. **Python microservice:** A thin FastAPI/gRPC service that wraps the YOLO model, called by the Go backend. Simplest to set up but adds operational complexity (two services to deploy).

**Recommendation after POC:** Start with option 3 (Python microservice) for server-side MVP, then migrate to option 1 (on-device) for production mobile app to eliminate per-inference costs.

## Consequences

- Go backend remains the source of truth for business logic, auth, and data persistence.
- Model training and export will always be done in Python regardless of serving strategy.
- The `spike/food-detection-poc/` notebook validates the model accuracy; the serving architecture is a separate concern.
- If on-device inference is chosen, the Go backend's role in food detection becomes purely receiving and storing results from the mobile client.
