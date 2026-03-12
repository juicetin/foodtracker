# App Size Impact on Install Conversion & Retention

**Date:** 2026-03-12
**Related:** ADR-005 (Local-First Architecture)

## Size vs Install Conversion

### Google's "Shrinking APKs, Growing Installs" Study

The most widely cited data in the Android ecosystem:
- **Every 6MB increase in APK size: ~1% drop in install conversion**
- **Emerging markets: removing 10MB = ~2.5% more installs** (~2.5x impact vs developed markets)
- **10MB app has ~30% higher download completion rate than 100MB app**

### Platform Size Thresholds

| Threshold | Impact |
|-----------|--------|
| 150MB | Google Play base AAB download size limit |
| **200MB (Android)** | **Wi-Fi suggestion dialog on mobile data** |
| **200MB (iOS)** | **Apple's default cellular download cap** |
| 4GB | Maximum total compressed app size (both stores) |

### Google Play Specific Limits

| Component | Limit |
|-----------|-------|
| APK (legacy) | 100MB |
| AAB base module | 200MB compressed download |
| Install-time asset packs | 1GB total |
| Fast-follow / on-demand packs | 512MB each |
| Total asset packs | 2GB |
| AI packs (Play for On-Device AI) | 1.5GB each |

### Apple iOS Limits

| Component | Limit |
|-----------|-------|
| Total uncompressed app | 4GB |
| On-Demand Resources per tag | 512MB (ideal: 64MB) |
| Total On-Demand Resources | 30GB after thinning |
| iOS 18+ individual asset pack | 8GB after thinning |

## Emerging Markets

### Data Costs per GB (2024-2025)

| Region / Country | Cost per 1GB |
|------------------|-------------|
| India | ~$0.09 |
| Nigeria | ~$0.38 |
| Kenya | ~$0.59 |
| Colombia | ~$0.20 |
| Chile | ~$0.64 |
| Panama | ~$2.98 |
| Sub-Saharan Africa (median) | >$3.70 |
| Zimbabwe | ~$43.75 |

A 200MB app download in Zimbabwe costs the equivalent of ~$8.75 in data.

### Storage Constraints

- Android 15 mandates minimum 32GB storage (16GB phased out)
- Budget phones ($100-200) in India: typically 64-128GB now
- Mass-budget segment ($100-200) declined 8.8% YoY in India as prices rise
- **70%+ of smartphone sales in some African markets are refurbished** (often 32-64GB)
- Android Go targets first-time smartphone users in emerging markets

### Uninstall Behavior

- **46.1% of all apps uninstalled within 30 days** (2024 global average)
- India: **71% uninstall rate** for social apps after non-organic downloads
- Brazil, Indonesia: **47% uninstall rate** for photo/video apps
- **Android users uninstall at 2x the rate of iOS users** (linked to storage constraints)
- Users are far more likely to uninstall an app than delete personal data when storage runs low

## Competitor Food Tracker Sizes

### iOS (App Store, March 2026)

| App | iOS Size |
|-----|---------|
| Yazio | 363.4MB |
| MyFitnessPal | 286.9MB |
| MacroFactor | 251.3MB |
| Lose It! | 202.9MB |
| Cronometer | 133.4MB |

### Android (varies by device due to App Bundles)

| App | Android Size |
|-----|-------------|
| MyFitnessPal | ~97-145MB |
| Cronometer | ~75-100MB |
| MacroFactor | ~87-102MB |
| Yazio | ~52-74MB |
| Lose It! | ~80-105MB |

All major food trackers are **under 200MB on Android**. Cronometer is the leanest.

## Large App Strategies

### Post-Install Download Pattern (Industry Standard)
- Ship small launcher/shell (~50-150MB) through store
- Download assets on first launch or as fast-follow
- Google reports: Play Asset Delivery fast-follow mode = **10% more new players** vs legacy CDN

### Play for On-Device AI (Beta)
Purpose-built for ML model delivery:
- Individual AI pack: **1.5GB** compressed
- Total: **4GB** cumulative
- Supports install-time, fast-follow, on-demand delivery
- **Automatic delta patching** for model updates
- **Device targeting** by RAM (min/max), brand/model (up to 10K per group), SoC, system features

```
project-root/
  app/
  food-recognition-model/         # AI pack
    build.gradle                  # deliveryType = "fast-follow"
    src/main/assets/
      classifier/model.tflite
  premium-model/                  # AI pack
    build.gradle                  # deliveryType = "on-demand"
    src/main/assets/
      advanced/model.tflite
```

Technical requirements: Android Gradle Plugin 8.8+, minimum SDK 21+.

### Apple Equivalent
- On-Demand Resources (legacy, functional)
- Background Assets API (newer replacement)
- Ship core content in base install, defer the rest

## Implications for Food Tracker

1. **Base APK target: <100MB.** Competitive with Cronometer (75-100MB). Include React Native + YOLO models only.
2. **Nutrition DB via fast-follow.** Auto-downloads seconds after install. Keeps base download small.
3. **VLMs via on-demand AI packs.** User-triggered download with device targeting for appropriate model tier.
4. **Stay under 200MB total installed** for emerging market viability.
5. **Every 10MB matters** in India/SE Asia/Africa (2.5x conversion impact).

## Sources

- Google: Shrinking APKs, Growing Installs (medium.com/googleplaydev)
- Google Play app size limits (support.google.com)
- Apple: Maximum Build File Sizes (developer.apple.com)
- Visual Capitalist: Cost of 1GB mobile data worldwide
- GSMA: Mobile Data Affordability 2024
- AppsFlyer: App Uninstall Benchmarks 2025
- Statista: Global app uninstall rate 2024
- Play for On-device AI (developer.android.com)
- Play Asset Delivery (developer.android.com)
