# Phase 4: Gallery Scanning + Deduplication - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Automatic discovery of food photos from the device gallery, grouping into meal events, and queueing for Gemini Nano identification. Users no longer need to manually photograph food — the app finds it.

</domain>

<decisions>
## Implementation Decisions

### Food Photo Detection
- Use Gemini Nano via AICore for food photo classification (is this a food photo? yes/no)
- NO cloud ML, NO YOLO for detection — Gemini Nano only per project strategy

### Meal Grouping
- 1 hour default window for grouping photos into same meal (user-configurable)
- GPS proximity: ~100-200m as sensible default, not user-configurable for now

### Photo Storage
- Import food photos into app storage, downscaled to backup-friendly size
- Don't reference gallery paths (avoids broken refs if user deletes from gallery)

### Claude's Discretion
- Downscale target resolution (e.g. 1024px longest edge, or 512px)
- Background scan chunk size (how many photos per WorkManager invocation)
- EXIF extraction approach (expo-media-library metadata vs custom EXIF reader)
- Gallery scan UI (queue view layout, scan progress indicator)
- How to handle photos without GPS data (skip proximity check, use time only)
- Deduplication hash approach (perceptual hash vs timestamp+GPS only)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **geminiNanoService** — Phase 02.7 built identification service, can be used for food/not-food classification
- **expo-media-library** — likely already available for gallery access
- **Background task infrastructure** — Phase 3.6/3.7 set up expo-background-task + expo-task-manager
- **useFoodLogStore** — logScanResult for creating diary entries from discovered photos
- **Photo table** — existing photos table in schema

### Integration Points
- Background task → gallery scan → Gemini Nano classification → meal grouping → diary entry creation
- Discovered photos appear in diary via existing entry flow
- Settings for scan frequency, meal window in usePreferencesStore

</code_context>

<specifics>
## Specific Ideas

- The 5-minute window from the original success criteria is too short — user specified 1 hour default
- Photos without GPS should still be grouped by time proximity alone
- The scan should be non-intrusive — users shouldn't notice it running in the background

</specifics>

<deferred>
## Deferred Ideas

- iOS background scanning (deferred with iOS release)
- Cross-device photo deduplication (post-v1.0)
- Smart album detection (identifying restaurant meals vs home cooking)

</deferred>
