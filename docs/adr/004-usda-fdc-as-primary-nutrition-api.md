# ADR-004: USDA FoodData Central as Primary Nutrition API for POC

**Status:** Accepted
**Date:** 2026-02-08

## Context

The POC needs a nutrition database to convert detected ingredients + estimated weights into calorie/macro breakdowns. Multiple databases exist (see `services/ai-agent/src/config/databases.ts`):

- USDA FoodData Central (US)
- AFCD/NUTTAB (Australia)
- CoFID (UK)
- CIQUAL (France)
- Open Food Facts (global)

## Decision

Use USDA FoodData Central as the primary nutrition API for the POC spike.

**Reasons:**

- Free API with generous rate limits (1000 req/hour with free key, no key needed for DEMO_KEY).
- Well-documented REST API with search + nutrient detail endpoints.
- Largest English-language food database (~400k foods).
- Includes "Foundation" and "SR Legacy" datasets with reliable per-100g nutrient values.
- Easy to query programmatically — simple GET requests, JSON responses.

**For production:** Region-specific databases (AFCD for Australian users, etc.) will be added per the existing database routing design. The USDA will serve as the global fallback alongside Open Food Facts.

## Consequences

- POC nutrition values will be US-centric. Australian-specific foods (e.g. Vegemite, Tim Tams) may not have accurate results until AFCD integration.
- The nutrition lookup interface in the notebook is designed to be swappable — same `NutritionPer100g` dataclass regardless of source.
- API key management: free DEMO_KEY works for low-volume spike testing. Production will need a proper key.
