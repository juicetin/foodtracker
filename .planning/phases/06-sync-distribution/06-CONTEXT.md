# Phase 6: Sync + Distribution - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend sync to support FTP backup, and set up Android model distribution via Play for On-Device AI. iOS features (iCloud, On-Demand Resources) are deferred until Apple has comparable on-device AI.

Note: Google Drive sync was already implemented in Phase 3.7. This phase adds FTP as an alternative backup destination and Android model delivery infrastructure.

</domain>

<decisions>
## Implementation Decisions

### Sync Backends
- Allow multiple sync backends simultaneously on Android: Google Drive + FTP
- iOS release deferred until Apple offers comparable on-device AI inference — no iCloud, no iOS model delivery
- FTP server backup available on both platforms (when iOS eventually ships)

### Conflict Resolution
- Per-field LWW by default (already implemented in Phase 3.7)
- User prompted to resolve conflicts individually (already implemented)
- Auto-resolve toggle (already implemented)

### Model Delivery (Android)
- Play for On-Device AI for model distribution with device targeting
- Cloud fallback tier: deferred to post-v1.0

### Claude's Discretion
- FTP client library choice
- FTP settings UI (host, port, username, password, path)
- Play for On-Device AI configuration and targeting criteria
- How to handle model updates (delta patching vs full download)
- FTP backup file format (reuse Phase 3.6/3.7 backup format)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 3.7 sync infrastructure** — driveAuth, driveSync, conflictResolver, useSyncStore, SyncSettingsScreen all built
- **Phase 3.6 backup system** — incremental backups, compaction, full snapshots
- **syncScheduler** — background task + foreground drain pattern

### Integration Points
- FTP client → sync scheduler (new backup destination alongside Drive)
- SyncSettingsScreen → FTP settings section
- Play for On-Device AI → app.json / build configuration
- Model manifest → existing model loading infrastructure

</code_context>

<specifics>
## Specific Ideas

- FTP should reuse the exact same backup format as Google Drive — just a different transport
- FTP settings should be saved securely (password encryption or expo-secure-store)
- Play for On-Device AI may require specific Play Console configuration

</specifics>

<deferred>
## Deferred Ideas

- iCloud sync (deferred with iOS release)
- iOS On-Demand Resources / Background Assets API (deferred with iOS release)
- Cloud fallback inference tier (deferred to post-v1.0 per ADR-005)
- Model delta patching (implementation detail, full download for v1.0)

</deferred>
