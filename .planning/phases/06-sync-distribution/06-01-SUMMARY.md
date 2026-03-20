---
phase: 06-sync-distribution
plan: 01
subsystem: sync
tags: [ftp, apache-commons-net, expo-secure-store, backup, multi-backend]

# Dependency graph
requires:
  - phase: 03.7-google-drive-sync
    provides: syncScheduler, driveSync, useSyncStore, SyncSettingsScreen, backup infrastructure
provides:
  - FTP native module (Android via Apache Commons Net, iOS stub)
  - ftpClient service with SecureStore credential management
  - ftpSync service mirroring driveSync pattern
  - Multi-backend sync dispatch (Drive + FTP via Promise.allSettled)
  - SyncSettingsScreen FTP card with credential form and test connection
affects: [06-02-play-ai-packs]

# Tech tracking
tech-stack:
  added: [expo-secure-store, commons-net:3.11.1]
  patterns: [Promise.allSettled multi-backend dispatch, SecureStore for sensitive credentials]

key-files:
  created:
    - apps/mobile/modules/ftp-client/android/src/main/java/expo/modules/ftpclient/FtpClientModule.kt
    - apps/mobile/modules/ftp-client/src/ftpClientModule.ts
    - apps/mobile/src/services/sync/ftpClient.ts
    - apps/mobile/src/services/sync/ftpSync.ts
    - apps/mobile/src/services/sync/__tests__/ftpClient.test.ts
    - apps/mobile/src/services/sync/__tests__/ftpSync.test.ts
  modified:
    - apps/mobile/src/services/sync/syncScheduler.ts
    - apps/mobile/src/store/useSyncStore.ts
    - apps/mobile/src/screens/SyncSettingsScreen.tsx
    - apps/mobile/src/services/sync/types.ts
    - apps/mobile/src/services/sync/__tests__/syncScheduler.test.ts

key-decisions:
  - "FTP password stored only in expo-secure-store, never in Zustand/AsyncStorage"
  - "Promise.allSettled for independent Drive+FTP dispatch -- one failing never blocks the other"
  - "syncScheduler no longer early-returns on !isSignedIn -- FTP may be enabled without Google account"
  - "iOS FTP stub throws not-available -- iOS FTP deferred per user decision"

patterns-established:
  - "Multi-backend sync: build promises array conditionally, dispatch via Promise.allSettled, check results per-backend"
  - "Sensitive credentials in SecureStore, display-only values (host) in Zustand for UI"

requirements-completed: [DAT-04, DAT-05, DAT-06]

# Metrics
duration: 7min
completed: 2026-03-21
---

# Phase 06 Plan 01: FTP Backup Client Summary

**FTP native module wrapping Apache Commons Net with passive-mode upload/download, SecureStore credential management, and Promise.allSettled multi-backend dispatch alongside Google Drive**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-20T18:05:27Z
- **Completed:** 2026-03-20T18:12:27Z
- **Tasks:** 2
- **Files modified:** 14

## Accomplishments
- FTP native Expo module with Android implementation (Apache Commons Net passive mode) and iOS stub
- Secure credential storage via expo-secure-store with save/load/clear/test operations
- syncScheduler extended to dispatch backups to Drive and FTP independently via Promise.allSettled
- SyncSettingsScreen FTP card with host/port/username/password/path inputs, test connection, and save

## Task Commits

Each task was committed atomically:

1. **Task 1: FTP native module + ftpClient/ftpSync services + types extension** - `e865bb03` (feat)
2. **Task 2: Extend syncScheduler + useSyncStore + SyncSettingsScreen for FTP** - `7c20a45d` (feat, bundled with 06-02 commit)

## Files Created/Modified
- `apps/mobile/modules/ftp-client/` - Complete Expo native module (config, gradle, Kotlin, Swift stub, TS bindings)
- `apps/mobile/src/services/sync/ftpClient.ts` - SecureStore credential management + native module wrapper
- `apps/mobile/src/services/sync/ftpSync.ts` - FTP sync mirroring driveSync pattern
- `apps/mobile/src/services/sync/types.ts` - FtpSyncStatus and FtpCredentials type exports
- `apps/mobile/src/services/sync/syncScheduler.ts` - Multi-backend dispatch via Promise.allSettled
- `apps/mobile/src/store/useSyncStore.ts` - FTP state fields (ftpEnabled, ftpHost, ftpSyncStatus)
- `apps/mobile/src/screens/SyncSettingsScreen.tsx` - FTP Backup card with form and test connection
- `apps/mobile/src/services/sync/__tests__/ftpClient.test.ts` - 10 tests for credential and upload operations
- `apps/mobile/src/services/sync/__tests__/ftpSync.test.ts` - 2 tests for sync-to-FTP
- `apps/mobile/src/services/sync/__tests__/syncScheduler.test.ts` - 11 tests including multi-backend dispatch

## Decisions Made
- FTP password stored only in expo-secure-store, never in Zustand/AsyncStorage -- security requirement
- Promise.allSettled for independent Drive+FTP dispatch -- one backend failure never blocks the other
- Removed early-return on !isSignedIn in syncScheduler -- FTP may be enabled without Google account
- iOS FTP stub throws "not available" -- iOS FTP deferred per user decision
- Test connection saves credentials first, then tests -- ensures latest values are tested

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Task 2 changes were bundled into a prior 06-02 commit that was already at HEAD from a previous session. No data loss -- all changes verified present.

## User Setup Required

Users need an FTP server to test backup upload:
- Have an FTP server address, credentials, and remote path ready
- Configure in app Settings > Sync > FTP Backup section

## Next Phase Readiness
- FTP backup operational alongside Google Drive
- Ready for 06-02 (Play for On-Device AI model delivery)
- All 46 sync tests passing across 7 suites

---
*Phase: 06-sync-distribution*
*Completed: 2026-03-21*
