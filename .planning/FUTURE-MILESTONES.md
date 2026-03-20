# Future Milestones: Competitor-Driven Roadmap

**Created:** 2026-03-14
**Source:** Competitor research in `.planning/research/` (competitor-profiles-2026.md, ios-competitor-profiles.md, user-complaints-competitor-analysis.md)
**Strategy:** Close on competitor weak points, exploit structural advantages of local-first architecture, and build sustainable monetization.

---

## Milestone Overview

| Milestone | Theme | Closes On |
|-----------|-------|-----------|
| v1.0 | Local-First Reset | Foundation + on-device ML pipeline (in progress) |
| v1.1 | Launch Differentiators | MFP paywall, logging friction, 77% Day-3 abandonment, goals + weight tracking |
| v2.0 | Home-Cooking Revolution | #1 tracking abandonment cause (80% quit over manual entry tedium) |
| v3.0 | Inclusive & Mindful Tracking | Accessibility gaps (75% Android apps lack alt text), ED safety (73% say MFP contributed) |
| v4.0 | Adaptive Intelligence | MacroFactor's only moat (adaptive TDEE), health platform integration |
| v5.0 | Monetization & Growth | Sustainable revenue without subscriptions/ads, user acquisition |

---

## v1.1: Launch Differentiators

**Goal:** Ship the features that directly counter the #1 competitor complaints on Day 1, turning MFP refugees into FoodTracker users. Every feature here was chosen because a competitor paywalls it, does it badly, or doesn't do it at all.

**Timing:** Immediately after v1.0 ship. These are the "why switch?" features.

**Closes on:**
- MFP barcode paywall (rated #1 complaint across all stores)
- 77% of users abandon food apps within 3 days (onboarding friction)
- "Adding meals takes longer than preparing them" (MFP user quote)
- No competitor offers free voice logging + free barcode + 2-tap quick-add
- Macro summary without goals feels hollow — users need targets to make tracking actionable (MacroFactor analysis)

### Requirements

#### Barcode Scanning (Free, Forever)

- [ ] **BAR-01**: User can scan any packaged food barcode and get instant nutrition data from bundled Open Food Facts database (4.4M+ products)
- [ ] **BAR-02**: Barcode scan results show verified nutrition data with source attribution (OFF, USDA, or user-submitted with accuracy flag)
- [ ] **BAR-03**: User can submit corrections to scanned barcode data, stored locally (and optionally contributed back to OFF)
- [ ] **BAR-04**: Offline barcode scanning works with no network dependency (database is bundled/downloaded)

*Competitor gap: MFP paywalled barcode scanning in Oct 2022, causing rating drop from 4.7 to 4.2. Lose It! following same playbook. Our positioning: "Barcode scanning will never be paywalled."*

#### Voice Food Logging

- [ ] **VOI-01**: User can describe a meal by voice ("I had two scrambled eggs, toast with butter, and orange juice") and get it logged with nutrition data
- [ ] **VOI-02**: Voice recognition runs on-device using platform speech APIs (iOS Speech, Android SpeechRecognizer) with no cloud dependency
- [ ] **VOI-03**: Voice logging supports natural language portion descriptions ("a big bowl of", "half a plate of", "about a cup of")
- [ ] **VOI-04**: Voice logging works in at least 5 languages at launch (English, Spanish, Hindi, Mandarin, Arabic) using platform speech models

*Competitor gap: Lose It! "Say It" is Premium-only. MFP voice logging is Premium+. No competitor offers free on-device voice logging.*

#### Quick-Add & Friction Reduction

- [ ] **QCK-01**: User can log a frequently-eaten meal in 2 taps (home screen -> recent meal -> confirm)
- [ ] **QCK-02**: App presents top 5 most-frequently-logged meals on the home screen as quick-add cards
- [ ] **QCK-03**: User can copy a previous day's meals to today in one tap
- [ ] **QCK-04**: Onboarding completes in under 90 seconds with zero required fields (no account, no email, no weight goal required)
- [ ] **QCK-05**: First food log is possible within 30 seconds of install (skip all setup, go straight to camera/gallery)

*Competitor gap: 77% abandon within 3 days. 70% quit if app is "too complex or time-consuming." Noom onboarding is 10+ minutes. MFP requires account creation. Our north star: "2-tap logging."*

#### Widget & Quick Access

- [ ] **WDG-01**: Home screen widget shows today's calorie/macro progress (iOS 17+ WidgetKit, Android 12+ Glance)
- [ ] **WDG-02**: Long-press app icon shows quick actions: "Log meal", "Scan barcode", "Voice log" (iOS 3D Touch/Haptic, Android App Shortcuts)
- [ ] **WDG-03**: Widget tap opens directly to logging screen, not home screen

*Competitor gap: MFP's widget is broken after redesign. Most competitors don't have functional widgets.*

#### Goals + Weight Tracking (deferred from v1.0)

- [ ] **GOAL-01**: User can set calorie and macro targets via Mifflin-St Jeor TDEE formula (age, weight, height, activity level, goal)
- [ ] **GOAL-02**: Daily macro summary shows consumed vs target with progress bars (not just raw consumed values)
- [ ] **GOAL-03**: Consumed/Remaining toggle on macro summary header (inspired by MacroFactor)
- [ ] **GOAL-04**: User can log daily weight (manual entry) with exponentially-smoothed trend line (EMA)
- [ ] **GOAL-05**: Weight trend visualization shows both raw data points and smoothed trend, with clear visual distinction
- [ ] **GOAL-06**: Basic weekly overview showing 7-day macro adherence grid (consumed vs target per day)

*Competitor gap: MacroFactor's weight trend smoothing is table stakes for serious trackers. Without goals, the macro dashboard shows raw numbers with no context — "you ate 2100 cal" vs "you ate 2100 of 2400 cal". Static TDEE formula is sufficient for v1.1; adaptive TDEE deferred to v4.0.*
*Source: MacroFactor competitive analysis (.planning/research/macrofactor-analysis.md)*

### Phases (v1.1)

- Phase 7: Barcode Scanning & OFF Database Integration
- Phase 8: Voice Logging Pipeline
- Phase 8.1: Goals + Weight Tracking (static TDEE, weight EMA, target progress bars)
- Phase 9: Quick-Add, Widgets & Onboarding Polish

---

## v2.0: Home-Cooking Revolution

**Goal:** Solve the problem that causes 80% of food tracking abandonment: logging home-cooked meals. No competitor has cracked this. The app that makes home-cooking easy to log wins the retention game.

**Timing:** 4-8 weeks after v1.1.

**Closes on:**
- 80% of calorie trackers abandoned because manual entry is tedious
- "Would have been a perfect app but becomes utterly useless when trying to input my own foods" (user quote)
- Recipe builders across all competitors described as "clunky"
- Oil/fat underestimation in home cooking (severalfold)
- No competitor can log a home-cooked meal in under 60 seconds

### Requirements

#### Smart Recipe Builder

- [ ] **RCP-01**: User can voice-dictate a recipe while cooking ("add two cups of flour, three eggs, half cup of sugar") and app builds the recipe in real-time
- [ ] **RCP-02**: User can import a recipe from any URL (blog, website) and app extracts ingredients with nutrition data via on-device parsing
- [ ] **RCP-03**: Recipe auto-scales when user changes serving count, recalculating all ingredient quantities and nutrition
- [ ] **RCP-04**: User can substitute ingredients in a recipe (e.g., swap butter for olive oil) with automatic nutrition recalculation
- [ ] **RCP-05**: App suggests cooking oil/fat amounts based on cooking method (fry = 2 tbsp, saute = 1 tbsp, bake = spray) to reduce hidden calorie underestimation

#### Meal Prep & Batch Cooking

- [ ] **PREP-01**: User can log a batch cook (e.g., "made 8 servings of chili") and portion it across multiple future meals
- [ ] **PREP-02**: When user logs "leftover [recipe name]", app uses the saved recipe nutrition with optional portion adjustment
- [ ] **PREP-03**: User can plan and pre-log meals for the next 1-7 days
- [ ] **PREP-04**: Meal prep recipes show per-container nutrition when divided into specified number of containers

#### Ingredient Recognition

- [ ] **ING-01**: User can photograph individual ingredients (produce, meat, pantry items) and app identifies them for recipe building
- [ ] **ING-02**: User can photograph a grocery receipt and app extracts food items for pantry tracking
- [ ] **ING-03**: App learns user's frequently-used ingredients and suggests them first in recipe builder

### Phases (v2.0)

- Phase 10: Smart Recipe Builder (voice dictate, URL import, substitution)
- Phase 11: Meal Prep & Batch Cooking Pipeline
- Phase 12: Ingredient Recognition & Grocery Integration

---

## v3.0: Inclusive & Mindful Tracking

**Goal:** Be the only food tracker that is genuinely accessible, culturally inclusive, and eating-disorder-safe. This is both ethical and strategic: 73% of ED patients say MFP contributed to their disorder. An ED-safe tracker has zero real competition.

**Timing:** 8-12 weeks after v2.0.

**Closes on:**
- 75% of paid Android apps lack alt text features entirely
- 73% of people with eating disorders say MFP contributed to their disorder
- MFP assigns most women ~1,200 calories/day (clinically low)
- Red/green color coding reinforces harmful dichotomous thinking
- No major food tracker advertises WCAG compliance
- Indian, African, Middle Eastern foods severely underserved in all databases

### Requirements

#### Accessibility (WCAG 2.1 AA)

- [ ] **A11Y-01**: All interactive elements have accessible labels, roles, and hints for screen readers (VoiceOver, TalkBack)
- [ ] **A11Y-02**: Food logging flow is fully navigable via screen reader in under 60 seconds per meal
- [ ] **A11Y-03**: All text respects system Dynamic Type / font size preferences with no layout breakage up to 200%
- [ ] **A11Y-04**: Touch targets are minimum 44x44pt throughout the app
- [ ] **A11Y-05**: Color is never the sole indicator of information (confidence levels use icons + text alongside color)
- [ ] **A11Y-06**: App supports reduced motion preference (no mandatory animations)

#### Mindful Tracking & ED Safety

- [ ] **MIND-01**: User can enable "Mindful Mode" that hides calorie numbers and shows only food groups and portion balance
- [ ] **MIND-02**: App never shows red/negative indicators for calorie totals — uses neutral color palette with optional color-free mode
- [ ] **MIND-03**: Calorie minimums are evidence-based (never suggests below 1,500 for women or 1,800 for men without explicit user override with safety warning)
- [ ] **MIND-04**: No streak counters, no guilt-inducing missed-day messages, no "you went over your limit" warnings
- [ ] **MIND-05**: User can set a "check-in" prompt that asks "How are you feeling about tracking?" weekly, with easy path to pause or adjust goals
- [ ] **MIND-06**: App includes optional link to NEDA helpline and eating disorder resources in settings

#### Cultural Cuisine Expansion

- [ ] **CUI-01**: Dedicated Indian food database with 500+ dishes covering North, South, East, and West Indian regional cuisines with accurate nutrition
- [ ] **CUI-02**: Dedicated African food database covering Ethiopian (injera, wots), West African (jollof, fufu, egusi), and North African cuisines
- [ ] **CUI-03**: Dedicated Middle Eastern food database (hummus variants, shawarma, falafel, regional breads, mezze)
- [ ] **CUI-04**: Latin American food database covering Mexican, Brazilian, Peruvian, Colombian, and Caribbean cuisines
- [ ] **CUI-05**: Southeast Asian food database covering Vietnamese, Thai, Filipino, Indonesian, and Malaysian cuisines beyond the basics
- [ ] **CUI-06**: All food names available in native script alongside English transliteration (e.g., "दोसा (Dosa)", "パッタイ (Pad Thai)")
- [ ] **CUI-07**: Community submission pipeline for culturally-verified nutrition data with dietitian review queue

#### RTL & Internationalization

- [ ] **I18N-01**: Full RTL layout support for Arabic, Hebrew, Urdu, Persian
- [ ] **I18N-02**: App localized in 10+ languages at launch (English, Spanish, Hindi, Arabic, Mandarin, French, Portuguese, German, Japanese, Korean)
- [ ] **I18N-03**: Measurement units adapt to locale (metric/imperial) with easy toggle

### Phases (v3.0)

- Phase 13: WCAG 2.1 AA Accessibility Audit & Remediation
- Phase 14: Mindful Tracking Mode & ED Safety Features
- Phase 15: Cultural Cuisine Database Expansion (India, Africa, Middle East, LatAm, SEA)
- Phase 16: RTL, Internationalization & Localization

---

## v4.0: Adaptive Intelligence

**Goal:** Match MacroFactor's adaptive algorithm (their only real moat) and pair it with health platform integration — but free. This makes FoodTracker the only app that combines AI food detection + adaptive TDEE + health integration at zero cost.

**Timing:** 12-16 weeks after v3.0.

**Closes on:**
- MacroFactor's adaptive TDEE is their #1 differentiator ($72/yr)
- No free app offers adaptive calorie targets
- Users want "goals that adjust based on actual progress"
- Apple Health / Google Fit integration is expected by serious trackers
- Weight trend smoothing builds trust in the tracking process

### Requirements

#### Adaptive TDEE & Smart Goals

- [ ] **TDEE-01**: App calculates adaptive TDEE from user's logged food intake and weight trend data using exponentially-weighted moving average (similar to MacroFactor algorithm)
- [ ] **TDEE-02**: Calorie and macro targets adjust weekly based on actual weight change vs. expected change
- [ ] **TDEE-03**: User sees a clear explanation of why targets changed ("Your actual expenditure last week was ~2,400 cal/day based on your weight trend")
- [ ] **TDEE-04**: Algorithm handles common edge cases: water weight fluctuations, menstrual cycle, creatine loading, high-sodium meals
- [ ] **TDEE-05**: All TDEE calculations run on-device with no cloud dependency

#### Weight & Body Composition Tracking

- [ ] **WGT-01**: User can log daily weight (manual entry or smart scale import) with exponentially-smoothed trend line
- [ ] **WGT-02**: Weight trend visualization shows both raw data points and smoothed trend, with clear visual distinction
- [ ] **WGT-03**: User can optionally log body measurements (waist, hips, chest, arms) and body fat percentage
- [ ] **WGT-04**: Progress photos can be taken and compared side-by-side with date overlay

#### Health Platform Integration

- [ ] **HLTH-01**: Bidirectional Apple Health sync: export nutrition data, import weight, steps, active energy, workouts
- [ ] **HLTH-02**: Bidirectional Google Fit / Health Connect sync: same as above for Android
- [ ] **HLTH-03**: User can view a unified dashboard showing nutrition + activity + weight trends correlated over time
- [ ] **HLTH-04**: Imported exercise data automatically adjusts daily calorie budget (configurable: full adjustment, half, or none)

#### Insights & Patterns

- [ ] **INS-01**: Weekly summary shows macro adherence, most/least logged meals, calorie distribution by meal time
- [ ] **INS-02**: App identifies patterns: "You tend to eat 400 more calories on weekends" or "Your protein is consistently 20g below target"
- [ ] **INS-03**: Monthly trend report comparing current month to previous months
- [ ] **INS-04**: All insights computed on-device from local data — no cloud analytics

### Phases (v4.0)

- Phase 17: Adaptive TDEE Algorithm (on-device, privacy-preserving)
- Phase 18: Weight Tracking, Body Composition & Progress Photos
- Phase 19: Apple Health / Google Fit Bidirectional Sync
- Phase 20: Insights Engine & Pattern Detection

---

## v5.0: Monetization & Growth

**Goal:** Build sustainable revenue without subscriptions or ads. The competitor research shows users will pay for quality if they're not being exploited. FoodNoms proves privacy-first monetization works ($40/yr optional). MacroFactor proves users value accuracy. Our model: free core with optional one-time premium purchase for power features.

**Timing:** Concurrent with v2.0+ (monetization should be planned early, shipped when user base justifies it).

**Closes on:**
- "Why does a food log cost more than a streaming service?" (MFP user, on $80-160/yr pricing)
- FoodNoms proves users pay $40/yr optional for privacy-first tracker (iOS only)
- 60%+ of users express data privacy concerns — marketing opportunity
- MFP March 2026 redesign + Cal AI breach = active user churn window
- No competitor combines free + private + accurate + cross-platform

### Requirements

#### Monetization Model (Ethical, Transparent)

- [ ] **MON-01**: Core app is 100% free forever: food detection, barcode scanning, voice logging, diary, nutrition data, gallery scanning, all ML features
- [ ] **MON-02**: Optional one-time "FoodTracker Pro" purchase ($9.99-$14.99) unlocking: custom themes/icons, advanced analytics dashboard, CSV/JSON data export, multiple diet profile support
- [ ] **MON-03**: Optional one-time "FoodTracker Family" purchase ($19.99-$24.99) adding: family sharing (up to 6 members), shared recipe library, household grocery list generation
- [ ] **MON-04**: Pricing page is transparent and visible before any purchase prompt — no dark patterns, no fake urgency, no "85% off BUT ONLY FOR ONE MORE HOUR"
- [ ] **MON-05**: User can request full data export at any time (free), in standard formats (JSON, CSV) — even without Pro purchase
- [ ] **MON-06**: No feature will ever move from free to paid — features only move from paid to free over time

#### User Acquisition & Marketing

- [ ] **MKT-01**: App Store / Play Store listing optimized for competitor-switching keywords ("MyFitnessPal alternative", "free calorie tracker no subscription", "private food tracker")
- [ ] **MKT-02**: Landing page highlighting the 5 key differentiators: no subscription, no ads, no account required, on-device ML, offline-first
- [ ] **MKT-03**: Comparison page showing feature-by-feature matrix against MFP, Lose It!, Cronometer, Cal AI with verified pricing
- [ ] **MKT-04**: App Store screenshots and descriptions explicitly call out: "Free barcode scanning", "No account required", "Your data never leaves your phone"
- [ ] **MKT-05**: Press kit targeting tech/health publications with privacy angle (post-Cal AI breach timing)
- [ ] **MKT-06**: Open-source the core ML models and nutrition database to build community trust and contributions

#### Community & Verified Data

- [ ] **COM-01**: Public, curated food database that users can contribute to (with verification pipeline) — positioned as "the Wikipedia of food nutrition"
- [ ] **COM-02**: GitHub-hosted nutrition database accepting PRs with CI validation (automated nutrition range checks, duplicate detection)
- [ ] **COM-03**: Regional community maintainers who verify culturally-specific food data (Indian, African, Latin American, etc.)
- [ ] **COM-04**: Attribution system: "This entry verified by [community member/dietitian name]" — builds trust vs. anonymous crowdsourcing

#### Retention & Anti-Churn

- [ ] **RET-01**: "Welcome back" flow for returning users after absence (no guilt, shows what's new, offers fresh start or continue)
- [ ] **RET-02**: Configurable tracking intensity: daily detailed, daily summary, weekly check-in, or photo-only mode
- [ ] **RET-03**: "Pause tracking" option that preserves all data and settings — no account deletion required to take a break
- [ ] **RET-04**: Gentle re-engagement notification (configurable, max 1/week): "We noticed you haven't logged in 3 days — your data is safe and waiting when you're ready"

### Phases (v5.0)

- Phase 21: One-Time Purchase IAP Infrastructure (no subscription, no dark patterns)
- Phase 22: App Store Optimization & Marketing Site
- Phase 23: Community Nutrition Database & Verification Pipeline
- Phase 24: Retention Features & Anti-Churn Flows

---

## Competitor Weakness → Milestone Mapping

This table shows which competitor weaknesses each milestone directly addresses:

| Competitor Weakness | Severity | Milestone | Specific Requirements |
|---|---|---|---|
| Barcode scanning paywalled (MFP, Lose It!) | CRITICAL | v1.1 | BAR-01 through BAR-04 |
| 77% abandon within 3 days (all apps) | CRITICAL | v1.1 | QCK-01 through QCK-05 |
| Home-cooked meal logging causes 80% abandonment | CRITICAL | v2.0 | RCP-01 through RCP-05, PREP-01 through PREP-04 |
| Subscription fatigue ($40-210/yr) | CRITICAL | v5.0 | MON-01, MON-06 |
| Database accuracy (15-30% variance, crowdsourced) | HIGH | v5.0 | COM-01 through COM-04 |
| International food coverage (~10% non-US) | HIGH | v3.0 | CUI-01 through CUI-07 |
| Privacy/data breaches (150M MFP, 3.2M Cal AI) | HIGH | v5.0 | MON-05, MKT-02, MKT-05 |
| Intrusive ads destroying UX (MFP, Lose It!, YAZIO) | HIGH | v1.0 | Already addressed (no ads) |
| No accessibility compliance (75% lack alt text) | HIGH | v3.0 | A11Y-01 through A11Y-06 |
| ED safety (73% say MFP contributed) | HIGH | v3.0 | MIND-01 through MIND-06 |
| No adaptive TDEE (MacroFactor's moat, $72/yr) | MEDIUM | v4.0 | TDEE-01 through TDEE-05 |
| No health platform integration (defer in v1) | MEDIUM | v4.0 | HLTH-01 through HLTH-04 |
| Predatory billing/auto-renewal (Noom, YAZIO, Lifesum) | MEDIUM | v5.0 | MON-04, MON-06 |
| Forced gamification (YAZIO, Noom) | MEDIUM | v3.0 | MIND-04 |
| Voice logging paywalled (Lose It!, MFP) | MEDIUM | v1.1 | VOI-01 through VOI-04 |
| No offline mode (Lifesum, most apps) | MEDIUM | v1.0 | Already addressed (local-first) |
| App bloat & slow performance (MFP, Lifesum) | MEDIUM | v1.0 | Already addressed (on-device) |

---

## Revenue Projection Model

Based on FoodNoms ($40/yr optional, iOS only) and MacroFactor ($72/yr, no free tier) benchmarks:

| Metric | Conservative | Moderate | Optimistic |
|--------|-------------|----------|------------|
| Year 1 DAU | 5,000 | 15,000 | 50,000 |
| Pro conversion (one-time) | 3% | 5% | 8% |
| Pro price | $9.99 | $12.99 | $14.99 |
| Family conversion | 0.5% | 1% | 2% |
| Family price | $19.99 | $22.99 | $24.99 |
| Year 1 revenue | ~$2,000 | ~$12,000 | ~$75,000 |

**Key insight:** One-time purchase model means revenue comes from growth, not retention tax. Every new user is a potential one-time buyer. No churn-driven economics.

---

## Timeline Summary

```
v1.0 (in progress) ──► v1.1 (4 weeks) ──► v2.0 (8 weeks) ──► v3.0 (8 weeks) ──► v4.0 (12 weeks) ──► v5.0 (concurrent)
     Foundation          Quick wins         Home cooking        Inclusive           Intelligence         Monetization
     ML pipeline         Barcode/Voice      Recipe builder      Accessibility       Adaptive TDEE        One-time IAP
     Diary UI            2-tap logging      Meal prep           ED safety           Health sync          Marketing
     Gallery scan        Widgets            Ingredients         Cultural DBs        Insights             Community DB
```

---

*This document is research-driven. Milestones should be validated with user testing and market feedback before committing to full planning.*
*Source research: `.planning/research/competitor-profiles-2026.md`, `.planning/research/ios-competitor-profiles.md`, `.planning/research/user-complaints-competitor-analysis.md`*
