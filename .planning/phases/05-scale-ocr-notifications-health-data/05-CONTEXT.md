# Phase 5: Scale OCR + Notifications + Health Data - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Three distinct features: kitchen scale weight reading via OCR, configurable daily macro push notifications, and weight trend tracking from health platform data. Plus Gemini Nano as primary inference engine (consistent with project strategy).

</domain>

<decisions>
## Implementation Decisions

### Scale OCR
- Manual weight input as fallback when OCR fails or no scale detected
- Container tare management: weigh empty container once, app remembers; also allow manual tare input

### Notifications
- Set during first-time app onboarding
- Default time: 9pm
- User-configurable time or can disable entirely

### Health Data Import
- Off by default — opt-in to auto-sync from Apple Health / Google Fit
- Google Health Connect on Android (Apple Health deferred with iOS release)

### Gemini Nano Role
- Primary and only inference engine throughout (consistent with Nano-only strategy)
- NOT an "enhancement" or overlay — it IS the inference engine

### Claude's Discretion
- 7-segment OCR approach (Gemini Nano text extraction vs custom TFLite model vs ML Kit text recognition)
- Container tare UI (list view, recent containers, auto-detect from photo)
- Weight trend smoothing algorithm (EMA, Kalman filter, or simple moving average)
- Notification content format and styling
- Health Connect permission flow and data display

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **geminiNanoService** — can be used for scale reading (text extraction from photo)
- **expo-notifications** — likely available or easy to add for push notifications
- **usePreferencesStore** — notification time preference, health sync opt-in
- **Background task infrastructure** — Phase 3.6/3.7/4 established patterns

### Integration Points
- Scale OCR integrates with detection flow (photo → detect scale → read weight → adjust portion)
- Notifications → daily macro summary from useFoodLogStore.getTodayTotals()
- Health Connect → new weight tracking screen/component
- Onboarding flow → notification time selection

</code_context>

<specifics>
## Specific Ideas

- For 7-segment OCR, Gemini Nano can likely read scale displays directly — try that before building a custom model
- Notifications should be informative and non-annoying — single daily summary, not per-meal
- Weight trend should show a clear visual chart with smoothed line vs raw data points

</specifics>

<deferred>
## Deferred Ideas

- Apple Health integration (deferred with iOS release)
- Multiple daily notification times
- Scale brand/model auto-detection
- Nutrition goal adjustment based on weight trend

</deferred>
