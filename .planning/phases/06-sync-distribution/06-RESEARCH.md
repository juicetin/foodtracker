# Phase 06: Sync + Distribution - Research

**Researched:** 2026-03-21
**Domain:** FTP backup transport, Play for On-Device AI model delivery, Expo native integration
**Confidence:** MEDIUM

## Summary

Phase 06 has two distinct workstreams: (1) adding FTP as an alternative backup destination alongside existing Google Drive sync, and (2) configuring Play for On-Device AI to deliver ML models through the Play Store instead of R2 downloads.

The FTP workstream is straightforward -- the existing backup infrastructure (Phase 3.6/3.7) produces files that just need a new transport. The React Native FTP library ecosystem is weak (all packages have <20 weekly downloads and stale maintenance), so a **custom Expo native module wrapping platform FTP APIs** is the recommended approach. FTP credentials must be stored via expo-secure-store, not AsyncStorage.

The Play for On-Device AI workstream requires Android-native Gradle configuration (AGP 8.8+, `com.android.ai-pack` plugin) with a custom Expo config plugin since `expo-play-asset-delivery` only handles standard asset packs (`com.android.asset-pack`), not AI packs. Device targeting by RAM and SoC is supported natively. The `play:ai-delivery:0.1.1-alpha01` library provides runtime status/download APIs that need a thin React Native bridge.

**Primary recommendation:** Write two small Expo native modules -- one for FTP upload (wrapping Android's `org.apache.commons.net.ftp` / iOS `CFStream`), one bridging Play AI Delivery APIs -- rather than depending on unmaintained community packages.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Allow multiple sync backends simultaneously on Android: Google Drive + FTP
- iOS release deferred until Apple offers comparable on-device AI inference -- no iCloud, no iOS model delivery
- FTP server backup available on both platforms (when iOS eventually ships)
- Per-field LWW by default (already implemented in Phase 3.7)
- User prompted to resolve conflicts individually (already implemented)
- Auto-resolve toggle (already implemented)
- Play for On-Device AI for model distribution with device targeting
- Cloud fallback tier: deferred to post-v1.0

### Claude's Discretion
- FTP client library choice
- FTP settings UI (host, port, username, password, path)
- Play for On-Device AI configuration and targeting criteria
- How to handle model updates (delta patching vs full download)
- FTP backup file format (reuse Phase 3.6/3.7 backup format)

### Deferred Ideas (OUT OF SCOPE)
- iCloud sync (deferred with iOS release)
- iOS On-Demand Resources / Background Assets API (deferred with iOS release)
- Cloud fallback inference tier (deferred to post-v1.0 per ADR-005)
- Model delta patching (implementation detail, full download for v1.0)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DAT-04 | User can opt into Google Drive backup/sync via app data folder | Already complete (Phase 3.7). No work needed. |
| DAT-05 | User on iOS can opt into iCloud backup/sync | **DEFERRED** -- iOS release deferred per CONTEXT.md. No work this phase. |
| DAT-06 | Sync conflicts resolved via LWW with timestamps, full edit history retained | Already complete (Phase 3.7). No work needed. |
| MDL-01 | Android app delivers ML models via Play for On-Device AI with device targeting by RAM and chipset | Play AI Delivery library + custom Expo config plugin + native bridge module |
| MDL-02 | iOS app delivers optional models via ODR or Background Assets API | **DEFERRED** -- iOS release deferred per CONTEXT.md. No work this phase. |

**Effective scope:** FTP backup client + FTP settings UI + Play for On-Device AI configuration (MDL-01 only). DAT-04, DAT-05, DAT-06 are already complete or deferred.

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| expo-secure-store | 55.0.9 | Secure FTP credential storage | Expo-native, encrypts at OS level, no AsyncStorage for passwords |
| expo-play-asset-delivery | 1.2.3 | **Reference only** -- asset pack patterns | Uses `com.android.asset-pack`; AI packs need `com.android.ai-pack` so custom config plugin required |
| play:ai-delivery | 0.1.1-alpha01 | Runtime AI pack status/download APIs | Official Google library for Play for On-Device AI |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @react-native-community/netinfo | 11.4.1 | WiFi gate for FTP sync | Already installed; reuse for FTP WiFi-only preference |
| expo-file-system | 19.0.21 | Read backup files for FTP upload | Already installed; File class for reading backup content |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom FTP native module | react-native-ftp-client (16 downloads/week) | Unmaintained, no New Architecture support, Android-only |
| Custom FTP native module | basic-ftp (Node.js) | Requires Node `net` module -- not available in React Native runtime |
| Custom AI pack config plugin | expo-play-asset-delivery | Only supports `com.android.asset-pack`, not `com.android.ai-pack` |

**Installation:**
```bash
npx expo install expo-secure-store
```

**Version verification:** expo-secure-store 55.0.9 confirmed via npm registry 2026-03-21.

## Architecture Patterns

### Recommended Project Structure
```
apps/mobile/
├── src/services/sync/
│   ├── ftpClient.ts           # FTP upload/download operations (wraps native module)
│   ├── ftpSync.ts             # FTP sync scheduler (mirrors driveSync.ts pattern)
│   ├── driveSync.ts           # (existing)
│   ├── driveAuth.ts           # (existing)
│   ├── syncScheduler.ts       # (extend: dispatch to Drive and/or FTP)
│   ├── conflictResolver.ts    # (existing, unchanged)
│   └── types.ts               # (extend: FTP config types)
├── src/screens/
│   └── SyncSettingsScreen.tsx  # (extend: FTP settings section)
├── src/store/
│   └── useSyncStore.ts        # (extend: FTP connection state)
├── modules/
│   ├── ftp-client/             # Custom Expo native module for FTP
│   │   ├── android/src/main/java/.../FtpClientModule.kt
│   │   ├── ios/FtpClientModule.swift   # Stub for now (iOS deferred)
│   │   ├── src/index.ts
│   │   └── expo-module.config.json
│   └── ai-pack-delivery/      # Custom Expo native module for AI pack bridge
│       ├── android/src/main/java/.../AiPackDeliveryModule.kt
│       ├── ios/AiPackDeliveryModule.swift  # No-op stub
│       ├── src/index.ts
│       └── expo-module.config.json
├── plugins/
│   └── withAiPack.js           # Expo config plugin for AI pack Gradle config
└── ai-packs/
    └── ml-models/
        └── src/main/assets/    # Model files for AI pack bundling
```

### Pattern 1: Transport-Agnostic Sync Scheduler

**What:** The sync scheduler dispatches to whichever backends are enabled (Drive, FTP, or both).
**When to use:** Every sync trigger (manual or scheduled).
**Example:**
```typescript
// syncScheduler.ts -- extended
export async function triggerManualSync(): Promise<void> {
  const store = useSyncStore.getState();
  const result = await performIncrementalBackup();
  if (!result) return;

  const promises: Promise<void>[] = [];

  if (isSignedIn()) {
    promises.push(syncToDrive(result));
  }
  if (store.ftpEnabled && store.ftpHost) {
    promises.push(syncToFtp(result));
  }

  await Promise.allSettled(promises); // Both can fail independently
}
```

### Pattern 2: Secure Credential Storage

**What:** FTP credentials stored in expo-secure-store, never AsyncStorage.
**When to use:** Any sensitive user input (passwords, tokens).
**Example:**
```typescript
import * as SecureStore from 'expo-secure-store';

const FTP_CREDS_KEY = 'ftp_credentials';

interface FtpCredentials {
  host: string;
  port: number;
  username: string;
  password: string;
  remotePath: string;
}

export async function saveFtpCredentials(creds: FtpCredentials): Promise<void> {
  await SecureStore.setItemAsync(FTP_CREDS_KEY, JSON.stringify(creds));
}

export async function loadFtpCredentials(): Promise<FtpCredentials | null> {
  const raw = await SecureStore.getItemAsync(FTP_CREDS_KEY);
  return raw ? JSON.parse(raw) : null;
}
```

### Pattern 3: AI Pack Delivery with Fallback

**What:** Check AI pack status at app launch; fall back to R2 download if pack not available.
**When to use:** Model loading in packManager.
**Example:**
```typescript
// In packManager.ts -- extend downloadPack
async function resolveModelPath(packId: string): Promise<string | null> {
  if (Platform.OS === 'android') {
    const aiPackPath = await AiPackDelivery.getPackLocation(packId);
    if (aiPackPath) return aiPackPath;
  }
  // Fall back to existing R2 download path
  return null;
}
```

### Anti-Patterns to Avoid
- **Storing FTP password in AsyncStorage or Zustand persist:** AsyncStorage is unencrypted. Always use expo-secure-store for passwords.
- **Blocking on both sync backends:** Use `Promise.allSettled`, not `Promise.all`. One backend failing should not block the other.
- **Hardcoding AI pack names in JS:** Define pack names in a config constant so they match the Gradle `aiPack.packName`.
- **Assuming AI pack is always available:** Fast-follow delivery may not complete before first app launch. Always check status and fall back.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Secure credential storage | Custom encryption/keychain wrapper | expo-secure-store | OS-level encryption, cross-platform, battle-tested |
| FTP protocol implementation | Pure JS FTP client | Native FTP libraries (Apache Commons Net on Android) | FTP requires raw sockets unavailable in RN JS runtime |
| AI pack Gradle configuration | Manual Gradle editing | Expo config plugin (withAiPack.js) | Prebuild-safe, reproducible, survives `npx expo prebuild --clean` |
| Model download progress tracking | Custom progress state | AiPackManager.registerListener / existing packManager progress | Both provide standardized progress callbacks |

**Key insight:** FTP and AI pack delivery both require native code. The correct Expo pattern is small native modules + config plugins, not community packages with 16 weekly downloads.

## Common Pitfalls

### Pitfall 1: expo-play-asset-delivery does NOT support AI packs
**What goes wrong:** Developers assume `expo-play-asset-delivery` handles AI packs, but it uses `com.android.asset-pack` Gradle plugin, not `com.android.ai-pack`. Models won't get AI-specific delivery features (device targeting by SoC/RAM).
**Why it happens:** The names are similar and the delivery concepts overlap.
**How to avoid:** Write a custom Expo config plugin (`withAiPack.js`) that generates the correct `com.android.ai-pack` build.gradle and wires up `play:ai-delivery` dependency.
**Warning signs:** Gradle build succeeds but device targeting doesn't work; models delivered to all devices regardless of RAM.

### Pitfall 2: AGP version requirement
**What goes wrong:** AI packs require AGP 8.8+, device targeting requires AGP 8.10.0+. Expo SDK 54 may ship an older AGP.
**Why it happens:** Expo pins AGP versions per SDK release.
**How to avoid:** Check `android/build.gradle` after prebuild. If AGP < 8.8, override in `build.gradle` or use a config plugin.
**Warning signs:** Gradle sync errors mentioning `ai-pack` plugin not found.

### Pitfall 3: FTP passive mode and NAT traversal
**What goes wrong:** FTP active mode fails behind NAT/firewalls. Server sends data connection to client's private IP.
**Why it happens:** Most consumer networks use NAT.
**How to avoid:** Always use passive mode (PASV) for FTP connections. Default the FTP client to passive mode.
**Warning signs:** Connection established but file transfer times out or fails.

### Pitfall 4: AI pack not ready at first launch (fast-follow)
**What goes wrong:** App launches before fast-follow AI pack download completes. Model loading fails.
**Why it happens:** Fast-follow downloads start after install but aren't guaranteed to finish before first launch.
**How to avoid:** Check `AiPackStatus.COMPLETED` before attempting to load. Show download progress UI. Fall back to R2 download if pack unavailable.
**Warning signs:** Model not found errors on fresh installs, especially on slow connections.

### Pitfall 5: FTP credentials in Zustand persist store
**What goes wrong:** FTP password stored in unencrypted AsyncStorage via Zustand persist.
**Why it happens:** Easy to add ftpPassword to useSyncStore which already uses AsyncStorage.
**How to avoid:** Store only non-sensitive FTP config (host, port, enabled flag) in Zustand. Password goes in expo-secure-store.
**Warning signs:** Password visible in AsyncStorage debug dump.

### Pitfall 6: AI pack size limits
**What goes wrong:** Individual AI pack exceeds 1.5GB compressed limit.
**Why it happens:** Multiple large models bundled in single pack.
**How to avoid:** Split into multiple AI packs if needed (e.g., one per model tier). Current VLM models: SmolVLM 365MB, mid 546MB, high 1.3GB -- all fit individually.
**Warning signs:** Play Console upload rejection.

## Code Examples

### FTP Native Module (Android - Kotlin)
```kotlin
// Conceptual -- uses Apache Commons Net FTPClient
// Source: Apache Commons Net documentation
class FtpClientModule : Module() {
  override fun definition() = ModuleDefinition {
    AsyncFunction("upload") { host: String, port: Int, user: String, pass: String, remotePath: String, localPath: String ->
      val client = FTPClient()
      client.enterLocalPassiveMode()
      client.connect(host, port)
      client.login(user, pass)
      client.setFileType(FTP.BINARY_FILE_TYPE)

      val inputStream = FileInputStream(localPath)
      client.storeFile(remotePath, inputStream)
      inputStream.close()
      client.logout()
      client.disconnect()
    }
  }
}
```

### AI Pack Config Plugin (withAiPack.js)
```javascript
// Source: Expo config plugin docs + Play for On-Device AI docs
const { withProjectBuildGradle, withSettingsGradle } = require('@expo/config-plugins');

function withAiPack(config, { packName, deliveryType = 'fast-follow' }) {
  // 1. Add AI pack to settings.gradle
  config = withSettingsGradle(config, (config) => {
    if (!config.modResults.contents.includes(`:${packName}`)) {
      config.modResults.contents += `\ninclude ':${packName}'`;
    }
    return config;
  });

  // 2. Add AI pack reference to app build.gradle
  config = withProjectBuildGradle(config, (config) => {
    // Add assetPacks reference
    return config;
  });

  return config;
}
```

### FTP Settings UI Section
```typescript
// Added to SyncSettingsScreen.tsx as a new card
<View style={styles.card}>
  <Text style={styles.cardTitle}>FTP Backup</Text>
  <View style={styles.toggleRow}>
    <Text style={styles.rowLabel}>Enable FTP backup</Text>
    <Switch value={ftpEnabled} onValueChange={handleFtpToggle} />
  </View>
  {ftpEnabled && (
    <>
      <TextInput placeholder="Host" value={ftpHost} onChangeText={setFtpHost} />
      <TextInput placeholder="Port" value={ftpPort} keyboardType="numeric" />
      <TextInput placeholder="Username" value={ftpUser} onChangeText={setFtpUser} />
      <TextInput placeholder="Password" secureTextEntry value={ftpPass} />
      <TextInput placeholder="Remote path" value={ftpPath} />
      <Pressable onPress={handleTestConnection}>
        <Text>Test Connection</Text>
      </Pressable>
    </>
  )}
</View>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Play Asset Delivery (asset packs) | Play for On-Device AI (AI packs) | 2024-2025 (beta) | AI-specific delivery with device targeting by RAM/SoC |
| R2 download for all model delivery | AI pack fast-follow + R2 fallback | This phase | Models auto-download post-install via Play Store infrastructure |
| Single sync backend (Drive) | Multiple simultaneous backends | This phase | Users with self-hosted FTP servers get backup without Google account |

**Deprecated/outdated:**
- `com.android.asset-pack` for ML models: Use `com.android.ai-pack` instead for device targeting support
- Play AI Delivery is in **alpha** (0.1.1-alpha01): API may change, but it's the only official option

## Open Questions

1. **AGP version in Expo SDK 54**
   - What we know: AI packs need AGP 8.8+, device targeting needs 8.10.0+
   - What's unclear: What AGP version Expo SDK 54 ships
   - Recommendation: Check after prebuild. Override if needed via config plugin.

2. **AI pack fast-follow vs on-demand for large VLM models**
   - What we know: Fast-follow auto-downloads post-install; on-demand requires explicit fetch
   - What's unclear: Whether fast-follow is reliable enough for 300MB+ VLM models on metered connections
   - Recommendation: Use fast-follow for core models (YOLO, classifier <10MB). Use on-demand for VLM tiers (300MB+) with explicit user consent.

3. **FTP module: custom Expo module vs bare native module**
   - What we know: Expo Modules API supports Kotlin/Swift with `npx create-expo-module`
   - What's unclear: Whether Apache Commons Net works smoothly in Expo module context
   - Recommendation: Start with Expo Modules API. Fall back to bare native module if issues arise.

4. **Play Console enrollment for Play for On-Device AI beta**
   - What we know: It's in beta, requires Play Console access
   - What's unclear: Whether beta enrollment has waitlist or is open
   - Recommendation: Verify access early; have R2 fallback ready as backup plan.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | jest-expo ~54.0.17 |
| Config file | apps/mobile/jest.config.js |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern sync --no-coverage` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MDL-01 | AI pack delivery config plugin generates correct Gradle | unit | `cd apps/mobile && npx jest --testPathPattern aiPack -x` | No -- Wave 0 |
| MDL-01 | AI pack bridge returns model path or null | unit | `cd apps/mobile && npx jest --testPathPattern aiPackDelivery -x` | No -- Wave 0 |
| FTP-NEW | FTP sync uploads backup file to remote server | unit | `cd apps/mobile && npx jest --testPathPattern ftpSync -x` | No -- Wave 0 |
| FTP-NEW | FTP credentials stored/loaded from secure store | unit | `cd apps/mobile && npx jest --testPathPattern ftpClient -x` | No -- Wave 0 |
| FTP-NEW | Sync scheduler dispatches to enabled backends | unit | `cd apps/mobile && npx jest --testPathPattern syncScheduler -x` | Yes (extend existing) |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern sync --no-coverage`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `apps/mobile/src/services/sync/__tests__/ftpSync.test.ts` -- covers FTP upload/download
- [ ] `apps/mobile/src/services/sync/__tests__/ftpClient.test.ts` -- covers credential storage
- [ ] `apps/mobile/modules/ai-pack-delivery/src/__tests__/index.test.ts` -- covers AI pack bridge
- [ ] `apps/mobile/plugins/__tests__/withAiPack.test.js` -- covers config plugin output

## Sources

### Primary (HIGH confidence)
- [Android Developers - Play for On-Device AI](https://developer.android.com/google/play/on-device-ai) -- AI pack configuration, delivery modes, device targeting, API reference
- [Android Developers - Device Targeting](https://developer.android.com/google/play/device-targeting) -- RAM/SoC targeting XML format
- [Expo Docs - SecureStore](https://docs.expo.dev/versions/latest/sdk/securestore/) -- Secure credential storage API

### Secondary (MEDIUM confidence)
- [expo-play-asset-delivery GitHub](https://github.com/one-am-it/expo-play-asset-delivery) -- Reference for asset pack Expo config plugin patterns (not directly usable for AI packs)
- [npm: expo-play-asset-delivery](https://www.npmjs.com/package/expo-play-asset-delivery) -- v1.2.3, last published 2024-04-12
- [Expo Docs - Config Plugins](https://docs.expo.dev/modules/config-plugin-and-native-module-tutorial/) -- Custom native module creation

### Tertiary (LOW confidence)
- [react-native-ftp-client](https://github.com/navico-mobile/react-native-ftp-client) -- 16 weekly downloads, unmaintained; not recommended
- [basic-ftp npm](https://www.npmjs.com/package/basic-ftp) -- Node.js only, not React Native compatible

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM -- expo-secure-store is solid; AI delivery library is alpha (0.1.1-alpha01)
- Architecture: HIGH -- patterns mirror existing driveSync.ts; transport-agnostic scheduler is proven
- Pitfalls: HIGH -- well-documented from official Android docs and RN ecosystem experience
- FTP library choice: MEDIUM -- custom native module is clearly correct but implementation specifics need validation

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (Play for On-Device AI is in beta; may change)
