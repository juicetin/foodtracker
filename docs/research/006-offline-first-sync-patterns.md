# Offline-First Data Sync Patterns for React Native

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## Library Assessment

### Tier 1: Recommended

#### PowerSync
- **What it is:** Streams changes from backend Postgres/MongoDB/MySQL into client-side SQLite. Reads/writes against local SQLite; sync in background.
- **Pricing:** Free tier (deactivates after 1 week inactivity). Free self-hosted Open Edition (source-available). Paid plans bill on "data synced."
- **Performance:** Up to 1M rows per client. 2,000-4,000 small-row ops/second.
- **React Native SDK:** Official, with background thread operations and reactive query subscriptions. Uses `@journeyapps/react-native-quick-sqlite` or OP-SQLite (beta).
- **Conflict model:** Server-authoritative with client upload queue. Server decides outcomes.
- **Status:** MongoDB connector GA (March 2025). Active development.
- **Assessment:** Best purpose-built option for offline-first SQLite sync with React Native.

#### WatermelonDB (Nozbe)
- **What it is:** Lazy-loading database for React Native with documented sync protocol.
- **Maturity:** Production-tested across 15+ apps, 500K+ users, 99.8% sync success rate.
- **Performance:** Sub-100ms local query. Apps launch instantly regardless of dataset size.
- **Sync:** Pull/push protocol. You build your own backend sync endpoint. Server needs `last_modified` column. On push conflict: abort and re-pull.
- **Assessment:** Battle-tested, excellent performance. More DIY than PowerSync.

#### react-native-cloud-storage (v2.3.0)
- **What it is:** Unified API for iCloud + Google Drive file operations.
- **Features:** Read/write files, works with Expo, zero dependencies. Google Drive uses REST API (cross-platform).
- **Google Drive:** App data folder via `drive.appdata` scope (narrow permission). Hidden from user and other apps. Counts against 15GB quota.
- **iCloud:** Native iOS APIs on iOS.
- **Sync model:** File-level (coarse-grained). Upload/download entire SQLite DB or JSON exports.
- **Assessment:** Best for simple backup/restore without running a backend. Not row-level sync.

### Tier 2: Viable Alternatives

#### CouchDB/PouchDB
- **Adapter:** `@craftzdog/pouchdb-adapter-react-native-sqlite` v4.0.0. 8-9x faster than previous versions. Requires op-sqlite, react-native-quick-crypto, react-native-buffer.
- **PouchDB:** Moved to Apache Foundation. PouchDB 9.0.0 released (202 PRs merged).
- **Conflict model:** Revision tree with deterministic winner. Manual resolution for complex cases.
- **Issues:** Conflict handling is widely considered unpleasant. Requires React Native New Architecture.
- **Assessment:** Viable but showing its age. Newer solutions are simpler.

#### Expo SQLite + Custom Outbox Sync
- **Pattern:** Pending mutations go to an outbox table with idempotency keys. SyncManager checks connectivity and coalesces sync requests. Fields: `localUpdatedAt`, `serverUpdatedAt`, `isSynced`.
- **Expo guidance:** Recommends TinyBase or Yjs + y-expo-sqlite for state management.
- **Assessment:** Maximum control, no vendor dependency. More work but straightforward for single-user.

### Tier 3: Not Recommended

#### Realm (MongoDB) -- DEPRECATED
- **Atlas Device Sync reached EOL September 30, 2025.** SDKs no longer maintained.
- MongoDB recommends migrating to PowerSync, Ditto, Couchbase Mobile, or ObjectBox.
- **Do not use for new projects.**

#### ElectricSQL
- Major pivot (July 2024) to read-path sync only. Writes go directly to Postgres.
- Not a full bidirectional offline-first sync solution.
- v1.0 released Dec 2024, v1.1 Aug 2025.

#### CRDT-based (Automerge, Yjs)
- **Automerge:** WASM-based. **React Native does not support WASM** -- needs native C bindings.
- **Yjs:** Better RN story via `y-expo-sqlite`. But CRDTs are overkill for food logging.
- **Cinapse case study:** Moved from Automerge CRDTs to PowerSync. Per-character tracking unnecessary for structured data with distinct fields.
- CRDTs shine for collaborative text editing, not record-oriented data.

## Google Drive Sync Details

### appDataFolder
- Hidden folder only your app can access
- Contents invisible to user and other apps
- Counts against user's Drive quota (15GB free)
- Cannot share files, cannot trash (must permanently delete)
- API rate limits: 20,000 calls per 100 seconds

### Syncing a SQLite Database
- Upload entire DB file -> download on other device -> replace local DB
- **File-level sync** -- cannot merge changes from two devices, only pick one version
- Suitable for single-device primary use case with backup/restore

### User Experience
- Requires Google Sign-In (OAuth2)
- `drive.appdata` scope: narrow permission (app folder only, not full Drive)

## iCloud Sync from React Native

- **react-native-cloud-storage** (recommended): supports iCloud via native APIs on iOS
- **react-native-cloudkit:** CloudKit JS for structured data (less maintained)
- iCloud is iOS-only (CloudKit JS can technically work on Android but requires Apple ID)
- No unified iCloud + Google Drive library beyond react-native-cloud-storage

## Conflict Resolution for Food Logging

### Why LWW (Last-Write-Wins) Is Sufficient

Food logging data characteristics:
- **Single-user** (one person logs their own meals)
- **Append-heavy** (new meals) with occasional edits (correcting portions)
- **Record-oriented** (structured fields), not collaborative documents
- Conflicts are rare and low-stakes

Practitioner consensus: LWW is sufficient for ~95% of apps where users work with their own data.

### Implementation
- Use server timestamps (or hybrid logical clocks) for recency
- On conflict, most recent edit wins
- Soft-delete: `is_deleted` + `deleted_at` for deletion propagation
- Retain full edit history locally for manual resolution if needed

### What Other Systems Use
- **WatermelonDB:** Server-authoritative, abort-and-re-pull on conflicts
- **PowerSync:** Server-authoritative with client upload queue
- **CouchDB/PouchDB:** Revision tree with deterministic winner selection
- **Custom outbox:** Track `localUpdatedAt`/`serverUpdatedAt`/`isSynced`, apply LWW on server

## Recommendation for Food Tracker

**Phase 1 (MVP):** react-native-cloud-storage for Google Drive backup/restore. File-level sync of SQLite DB. Simple, no backend needed.

**Phase 2 (if multi-device needed):** PowerSync self-hosted Open Edition + op-sqlite. Row-level sync, server-authoritative, handles conflicts automatically.

**Skip:** CRDTs, Realm, ElectricSQL.

## Sources

- PouchDB adapter for React Native SQLite (github.com/craftzdog)
- PouchDB 9.0.0 release notes
- WatermelonDB Sync documentation (watermelondb.dev)
- PowerSync React Native SDK docs
- PowerSync pricing and self-hosted Open Edition
- Cinapse: Why We Moved Away from CRDTs (powersync.com/blog)
- MongoDB Atlas Device Sync deprecation notice
- ElectricSQL "Electric Next" announcement (July 2024)
- Google Drive appDataFolder documentation
- react-native-cloud-storage (github.com/kuatsu, v2.3.0)
- Expo local-first architecture guide
- AppsFlyer app uninstall benchmarks 2025
- Hasura: Design Guide for Offline-First Apps
