# ADR-001: Python Notebook for Food Detection POC

**Status:** Accepted
**Date:** 2026-02-08

## Context

We need to validate whether traditional ML-based food detection (YOLO + CNN classifiers) can accurately identify ingredients and estimate weights from food photos, before committing to a full backend implementation.

The production backend is planned in Go. The question is what tool/language to use for rapid prototyping and human-in-the-loop validation.

## Decision

Use a Jupyter notebook (Python) in `spike/food-detection-poc/` for the POC/spike.

**Reasons:**

- The ML ecosystem is Python-first. Ultralytics (YOLO), PyTorch, TensorFlow, HuggingFace — all have Python as the primary SDK. No equivalent Go libraries exist for model training or interactive prototyping.
- Notebooks render images inline — critical for visually inspecting bounding boxes, detection accuracy, and weight estimates.
- Interactive cell-by-cell execution allows rapid iteration: tweak a parameter, re-run one cell, see the result immediately.
- ipywidgets and matplotlib provide a built-in review UI for annotating accuracy.
- The spike is throwaway code — it validates the approach, not the production architecture.

## Consequences

- The spike is not portable to Go directly. Once validated, the inference pipeline will need to be reimplemented (see ADR-002).
- Python dependency management (pip/venv) is needed in the repo alongside Node/Go tooling.
- Spike code lives in `spike/` and is not part of the production build.
